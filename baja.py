import io
import re
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd
import streamlit as st

DB_PATH = "gokart_telemetry.db"
REQUIRED_IMPORT_COLUMNS = {"timestamp_ms",
                           "primary_rpm", "secondary_rpm", "cvt_ratio"}

# ----------------------------
# Database setup
# ----------------------------


@st.cache_resource
def init_db():
    """Open a single shared connection and make sure the schema/indexes exist."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # lets the dashboard read while a logger writes
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cvt_runs (
            run_id TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            primary_rpm REAL,
            secondary_rpm REAL,
            cvt_ratio REAL,
            manual_mph REAL,
            PRIMARY KEY (run_id, timestamp_ms)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_time
        ON cvt_runs(run_id, timestamp_ms)
    """)

    # One row per run, instead of duplicating metadata across every sample.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            lap_time TEXT,
            gps_max_mph REAL,
            notes TEXT
        )
    """)

    conn.commit()
    return conn


conn = init_db()

# ----------------------------
# Data access
# ----------------------------


@st.cache_data(ttl=60)
def fetch_run_ids():
    df = pd.read_sql(
        "SELECT DISTINCT run_id FROM cvt_runs ORDER BY run_id", conn)
    return df["run_id"].tolist()


@st.cache_data(ttl=10)
def fetch_run(run_id: str) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT run_id, timestamp_ms, primary_rpm, secondary_rpm, cvt_ratio, manual_mph
        FROM cvt_runs
        WHERE run_id = ?
        ORDER BY timestamp_ms
        """,
        conn,
        params=(run_id,),
    )


@st.cache_data(ttl=10)
def fetch_run_metadata(run_id: str) -> dict:
    df = pd.read_sql("SELECT * FROM runs WHERE run_id = ?",
                     conn, params=(run_id,))
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def save_run_metadata(run_id: str, update_data: dict) -> bool:
    """Upsert one metadata row per run — never touches the sample rows."""
    if not update_data:
        return False
    try:
        cols = ", ".join(update_data.keys())
        placeholders = ", ".join("?" for _ in update_data)
        updates = ", ".join(f"{k} = excluded.{k}" for k in update_data)
        with conn:
            conn.execute(
                f"""
                INSERT INTO runs (run_id, {cols})
                VALUES (?, {placeholders})
                ON CONFLICT(run_id) DO UPDATE SET {updates}
                """,
                (run_id, *update_data.values()),
            )
        return True
    except sqlite3.Error as e:
        st.sidebar.error(f"Save failed: {e}")
        return False


def save_manual_mph(run_id: str, edited_df: pd.DataFrame) -> int:
    """Bulk-write manual MPH overrides for edited rows in a single transaction."""
    rows = [
        (row["manual_mph"], run_id, row["timestamp_ms"])
        for _, row in edited_df.iterrows()
        if pd.notna(row.get("manual_mph"))
    ]
    if not rows:
        return 0
    try:
        with conn:
            conn.executemany(
                "UPDATE cvt_runs SET manual_mph = ? WHERE run_id = ? AND timestamp_ms = ?",
                rows,
            )
        return len(rows)
    except sqlite3.Error as e:
        st.error(f"Save failed: {e}")
        return 0


def import_sd_csv(uploaded_file, run_id: str) -> tuple[int, int, str | None]:
    """Import an uploaded SD-card CSV as one new run.

    Returns (rows_imported, rows_skipped, error_message). error_message is
    None on success.
    """
    try:
        raw_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        return 0, 0, f"Couldn't read CSV: {e}"

    missing = REQUIRED_IMPORT_COLUMNS - set(raw_df.columns)
    if missing:
        return 0, 0, f"CSV is missing expected columns: {', '.join(sorted(missing))}"

    before = len(raw_df)
    raw_df = raw_df.dropna(subset=list(REQUIRED_IMPORT_COLUMNS))
    for col in ["timestamp_ms", "primary_rpm", "secondary_rpm", "cvt_ratio"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df = raw_df.dropna(subset=list(REQUIRED_IMPORT_COLUMNS))
    skipped = before - len(raw_df)

    if raw_df.empty:
        return 0, skipped, "No valid rows found in this file."

    rows = list(zip(
        [run_id] * len(raw_df),
        raw_df["timestamp_ms"].astype(int).tolist(),
        raw_df["primary_rpm"].tolist(),
        raw_df["secondary_rpm"].tolist(),
        raw_df["cvt_ratio"].tolist(),
    ))

    try:
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO cvt_runs
                    (run_id, timestamp_ms, primary_rpm, secondary_rpm, cvt_ratio)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
    except sqlite3.Error as e:
        return 0, skipped, f"Database error: {e}"

    return len(rows), skipped, None


def run_id_exists(run_id: str) -> bool:
    count = conn.execute(
        "SELECT COUNT(*) FROM cvt_runs WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    return count > 0


def clean_run_df(run_df: pd.DataFrame) -> pd.DataFrame:
    """Clean, filter, and normalize CVT sensor records."""
    required_cols = {"timestamp_ms", "primary_rpm", "secondary_rpm"}
    missing = required_cols - set(run_df.columns)
    if missing:
        st.warning(f"Run missing expected columns: {', '.join(missing)}")
        return run_df

    run_df = run_df.dropna(subset=list(required_cols))

    # Sanity bounds for go-kart engine RPMs — adjust to your hardware.
    RPM_MIN, RPM_MAX = 0, 20000
    run_df = run_df[
        run_df["primary_rpm"].between(RPM_MIN, RPM_MAX)
        & run_df["secondary_rpm"].between(RPM_MIN, RPM_MAX)
    ]

    run_df = run_df.sort_values("timestamp_ms").reset_index(drop=True)
    run_df["Timestamp"] = pd.to_datetime(run_df["timestamp_ms"], unit="ms")

    safe_secondary = run_df["secondary_rpm"].replace(0, pd.NA)
    run_df["computed_cvt_ratio"] = run_df["primary_rpm"] / safe_secondary

    return run_df


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Go-Kart Telemetry",
                   page_icon="🏎️", layout="wide")
st.title("🏎️ Go-Kart CVT Telemetry Dashboard")

# ----------------------------
# SD card import
# ----------------------------

with st.expander("📥 Import SD Card Log", expanded=not fetch_run_ids()):
    st.caption(
        "Upload the CSV log pulled from the SD card. Since it's one continuous "
        "log, the whole file is imported as a single new run."
    )
    uploaded_file = st.file_uploader(
        "CSV log file", type=["csv"], key="sd_upload")

    default_run_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    import_col1, import_col2 = st.columns([3, 1])
    with import_col1:
        run_name = st.text_input(
            "Run name:", value=default_run_name, key="sd_run_name")
    with import_col2:
        st.write("")
        st.write("")
        do_import = st.button(
            "Import", disabled=uploaded_file is None, use_container_width=True)

    if do_import and uploaded_file is not None:
        run_name = run_name.strip() or default_run_name
        already_confirmed = st.session_state.get(
            "sd_import_confirm_name") == run_name

        if run_id_exists(run_name) and not already_confirmed:
            st.session_state["sd_import_confirm_name"] = run_name
            st.warning(
                f"Run '{run_name}' already has data. Click Import again to add these rows anyway, "
                f"or change the run name above."
            )
        else:
            st.session_state.pop("sd_import_confirm_name", None)
            imported, skipped, error = import_sd_csv(uploaded_file, run_name)
            if error:
                st.error(error)
            else:
                msg = f"Imported {imported} samples as run '{run_name}'."
                if skipped:
                    msg += f" Skipped {skipped} malformed row(s)."
                st.success(msg)
                fetch_run_ids.clear()
                fetch_run.clear()
                st.rerun()

run_ids = fetch_run_ids()

if not run_ids:
    st.warning("No telemetry data found. Import a run above to get started.")
    st.stop()

col1, col2 = st.columns([4, 1])
with col1:
    selected_run = st.selectbox("Select Active Run:", run_ids)
with col2:
    st.write("")
    if st.button("🔄 Refresh Data", use_container_width=True):
        fetch_run.clear()
        fetch_run_ids.clear()
        fetch_run_metadata.clear()
        st.rerun()

# run_ids is non-empty (checked above), so selectbox always returns a str here —
# this just tells the type checker what we already know at runtime.
assert selected_run is not None

with st.sidebar:
    st.header("📝 Manual Run Metadata")
    st.caption(f"Editing metadata for: **{selected_run}**")

    existing_meta = fetch_run_metadata(selected_run)

    with st.form("manual_entry_form", clear_on_submit=False):
        lap_time = st.text_input(
            "Best Lap Time (m:ss.s):",
            value=existing_meta.get("lap_time") or "",
            help="Format: 1:14.2",
        )
        max_gps_mph = st.number_input(
            "Max GPS Speed (MPH):",
            min_value=0.0,
            max_value=120.0,
            step=0.1,
            value=float(existing_meta.get("gps_max_mph") or 0.0),
        )
        notes = st.text_area(
            "Track Conditions / Setup Notes:",
            value=existing_meta.get("notes") or "",
        )

        submitted = st.form_submit_button("💾 Save Metadata to Run")

        if submitted:
            update_data = {}
            if lap_time:
                if re.fullmatch(r"\d+:\d{2}(\.\d+)?", lap_time.strip()):
                    update_data["lap_time"] = lap_time.strip()
                else:
                    st.sidebar.warning(
                        "Lap time ignored — expected format m:ss.s (e.g. 1:14.2)")
            if max_gps_mph > 0:
                update_data["gps_max_mph"] = max_gps_mph
            if notes:
                update_data["notes"] = notes

            if save_run_metadata(selected_run, update_data):
                st.sidebar.success(f"Saved metadata for {selected_run}!")
                fetch_run_metadata.clear()
                st.rerun()

run_df = fetch_run(selected_run)

if run_df.empty:
    st.warning("Selected run contains no samples.")
    st.stop()

run_df = clean_run_df(run_df)

if run_df.empty:
    st.warning(
        "All records were removed during cleaning — check sensor data quality.")
    st.stop()

run_meta = fetch_run_metadata(selected_run)
best_lap = run_meta.get("lap_time") or "N/A"
gps_mph = run_meta.get("gps_max_mph")
run_notes = run_meta.get("notes")

m1, m2, m3, m4, m5 = st.columns(5)
duration = round((run_df["Timestamp"].max() -
                 run_df["Timestamp"].min()).total_seconds(), 2)
max_p_rpm = int(run_df["primary_rpm"].max())

ratio_series = run_df["computed_cvt_ratio"].dropna()
max_ratio = round(ratio_series.max(), 2) if not ratio_series.empty else "N/A"

m1.metric("Run Duration", f"{duration}s")
m2.metric("Best Lap Time", best_lap)
m3.metric("GPS Top Speed", f"{gps_mph} MPH" if gps_mph else "N/A")
m4.metric("Peak Primary RPM", f"{max_p_rpm:,}")
m5.metric("Max Shift Ratio", f"{max_ratio}")

if run_notes:
    st.info(f"**Run Notes:** {run_notes}")

st.divider()

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Engine vs. Secondary RPM")
    st.line_chart(run_df, x="Timestamp", y=["primary_rpm", "secondary_rpm"])

with chart_col2:
    st.subheader("CVT Shift Ratio Over Time")
    ratio_cols = [c for c in ["cvt_ratio",
                              "computed_cvt_ratio"] if c in run_df.columns]
    if ratio_cols:
        st.line_chart(run_df, x="Timestamp", y=ratio_cols)
    else:
        st.info("No ratio data available for this run.")

st.divider()
st.subheader("CVT Tuning Curve (Primary RPM vs. Shift Ratio)")
st.caption(
    "Plots engine RPM against drive ratio to help identify clutch shift-out points.")
st.scatter_chart(run_df, x="primary_rpm",
                 y="computed_cvt_ratio", color="secondary_rpm")

st.divider()
with st.expander("📋 Interactive Data Grid (Manual In-Line Edits)"):
    st.caption(
        "Edit the Manual MPH column and save to write those values back to the database.")

    display_cols = [
        "Timestamp", "timestamp_ms", "primary_rpm", "secondary_rpm",
        "cvt_ratio", "computed_cvt_ratio", "manual_mph",
    ]
    display_cols = [c for c in display_cols if c in run_df.columns]

    edited_df = st.data_editor(
        run_df[display_cols],
        num_rows="fixed",
        disabled=[c for c in display_cols if c != "manual_mph"],
        column_config={
            "manual_mph": st.column_config.NumberColumn(
                "Manual MPH", min_value=0, max_value=120
            ),
        },
        key="telemetry_editor",
    )

    if st.button("💾 Save Row Edits"):
        n = save_manual_mph(selected_run, edited_df)
        if n:
            st.success(f"Saved {n} row edits!")
            fetch_run.clear()
            st.rerun()
        else:
            st.info("No manual MPH values entered — nothing to save.")
