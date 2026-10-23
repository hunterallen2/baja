import io
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = "gokart_telemetry.db"

# CSV columns the Arduino must write. cvt_ratio is optional; we recompute it anyway.
REQUIRED_IMPORT_COLUMNS = ["timestamp_ms", "primary_rpm", "secondary_rpm"]
OPTIONAL_IMPORT_COLUMNS = ["cvt_ratio"]

# ---- Tuning knobs (adjust to your engine / clutches) ----
RPM_MIN, RPM_MAX = 0, 20000       # hard sensor limits
MIN_SECONDARY_RPM = 100           # below this the ratio is meaningless (kart stopped/crawling)
RATIO_MIN, RATIO_MAX = 0.5, 4.5   # plausible CVT ratio range
SPIKE_RPM_THRESHOLD = 5000        # drop samples this far from the local median
SMOOTH_WINDOW = 5                 # rolling-median window (samples)
MAX_CHART_POINTS = 3000           # downsample charts beyond this
EDITOR_MAX_ROWS = 1500            # rows shown in the editable grid at once


# ----------------------------
# Database
# ----------------------------

@contextmanager
def db():
    """Short-lived connection per call: safe across Streamlit sessions/threads."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@st.cache_resource
def init_db() -> bool:
    with db() as conn:
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
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                lap_time TEXT,
                gps_max_mph REAL,
                notes TEXT
            )
        """)
    return True


init_db()


@st.cache_data(ttl=60)
def fetch_run_ids() -> list[str]:
    with db() as conn:
        df = pd.read_sql("SELECT DISTINCT run_id FROM cvt_runs ORDER BY run_id", conn)
    return df["run_id"].tolist()


@st.cache_data(ttl=10)
def fetch_run(run_id: str) -> pd.DataFrame:
    with db() as conn:
        return pd.read_sql(
            """
            SELECT run_id, timestamp_ms, primary_rpm, secondary_rpm, cvt_ratio, manual_mph
            FROM cvt_runs WHERE run_id = ? ORDER BY timestamp_ms
            """,
            conn,
            params=(run_id,),
        )


@st.cache_data(ttl=10)
def fetch_run_metadata(run_id: str) -> dict:
    with db() as conn:
        df = pd.read_sql("SELECT * FROM runs WHERE run_id = ?", conn, params=(run_id,))
    return {} if df.empty else df.iloc[0].to_dict()


def clear_caches():
    fetch_run_ids.clear()
    fetch_run.clear()
    fetch_run_metadata.clear()


def save_run_metadata(run_id: str, lap_time, gps_max_mph, notes) -> bool:
    """Upsert metadata. None values are written as NULL, so fields can be cleared."""
    try:
        with db() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, lap_time, gps_max_mph, notes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    lap_time = excluded.lap_time,
                    gps_max_mph = excluded.gps_max_mph,
                    notes = excluded.notes
                """,
                (run_id, lap_time, gps_max_mph, notes),
            )
        return True
    except sqlite3.Error as e:
        st.sidebar.error(f"Save failed: {e}")
        return False


def save_manual_mph(run_id: str, changes: list[tuple]) -> int:
    """changes: list of (manual_mph or None, timestamp_ms). None clears the value."""
    if not changes:
        return 0
    rows = [
        (None if pd.isna(mph) else float(mph), run_id, int(ts)) for mph, ts in changes
    ]
    try:
        with db() as conn:
            conn.executemany(
                "UPDATE cvt_runs SET manual_mph = ? WHERE run_id = ? AND timestamp_ms = ?",
                rows,
            )
        return len(rows)
    except sqlite3.Error as e:
        st.error(f"Save failed: {e}")
        return 0


def run_id_exists(run_id: str) -> bool:
    with db() as conn:
        return conn.execute(
            "SELECT 1 FROM cvt_runs WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone() is not None


# ----------------------------
# Import
# ----------------------------

def import_sd_csv(uploaded_file, run_id: str) -> tuple[int, int, int, str | None]:
    """Returns (inserted, duplicates_ignored, malformed_skipped, error)."""
    try:
        raw = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), skipinitialspace=True)
    except Exception as e:
        return 0, 0, 0, f"Couldn't read CSV: {e}"

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED_IMPORT_COLUMNS if c not in raw.columns]
    if missing:
        return 0, 0, 0, (
            f"CSV is missing columns: {', '.join(missing)}. "
            f"Found: {', '.join(raw.columns) or 'none'}"
        )

    cols = REQUIRED_IMPORT_COLUMNS + [c for c in OPTIONAL_IMPORT_COLUMNS if c in raw.columns]
    df = raw[cols].copy()
    before = len(df)

    # Coerce everything to numbers. Repeated header lines (logger restarts),
    # half-written final lines, and garbage all become NaN and get dropped.
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_IMPORT_COLUMNS)
    df = df[(df["timestamp_ms"] >= 0) & np.isfinite(df["timestamp_ms"])]
    skipped = before - len(df)

    if df.empty:
        return 0, 0, skipped, "No valid rows found in this file."

    if "cvt_ratio" not in df.columns:
        df["cvt_ratio"] = np.nan

    # Same timestamp twice within the file: keep the first
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df = df.drop_duplicates(subset="timestamp_ms", keep="first")

    rows = [
        (
            run_id,
            int(r.timestamp_ms),
            float(r.primary_rpm),
            float(r.secondary_rpm),
            None if pd.isna(r.cvt_ratio) else float(r.cvt_ratio),
        )
        for r in df.itertuples(index=False)
    ]

    try:
        with db() as conn:
            before_changes = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO cvt_runs
                    (run_id, timestamp_ms, primary_rpm, secondary_rpm, cvt_ratio)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = conn.total_changes - before_changes
    except sqlite3.Error as e:
        return 0, 0, skipped, f"Database error: {e}"

    return inserted, len(rows) - inserted, skipped, None


# ----------------------------
# Cleaning / analysis
# ----------------------------

def clean_run_df(run_df: pd.DataFrame, smooth: bool) -> tuple[pd.DataFrame, dict]:
    """Clean sensor data. Returns (df, stats about what was removed)."""
    stats = {"input": len(run_df), "out_of_range": 0, "spikes": 0}

    df = run_df.copy()
    for c in ["timestamp_ms", "primary_rpm", "secondary_rpm", "cvt_ratio", "manual_mph"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp_ms", "primary_rpm", "secondary_rpm"])
    df = df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)

    n = len(df)
    in_range = (
        df["primary_rpm"].between(RPM_MIN, RPM_MAX)
        & df["secondary_rpm"].between(RPM_MIN, RPM_MAX)
    )
    df = df[in_range].reset_index(drop=True)
    stats["out_of_range"] = n - len(df)

    # Spike removal: drop points far from the local rolling median
    n = len(df)
    if n >= 5:
        keep = pd.Series(True, index=df.index)
        for col in ["primary_rpm", "secondary_rpm"]:
            med = df[col].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
            keep &= (df[col] - med).abs() <= SPIKE_RPM_THRESHOLD
        df = df[keep].reset_index(drop=True)
    stats["spikes"] = n - len(df)

    if df.empty:
        return df, stats

    # Arduino millis() counts from power-on, so use elapsed time, not a calendar date
    df["elapsed_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

    if smooth:
        for col in ["primary_rpm", "secondary_rpm"]:
            df[col] = df[col].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()

    # Ratio is only meaningful when the secondary is actually turning
    valid = df["secondary_rpm"] >= MIN_SECONDARY_RPM
    ratio = (df["primary_rpm"] / df["secondary_rpm"].where(valid)).astype(float)
    df["computed_cvt_ratio"] = ratio.where(ratio.between(RATIO_MIN, RATIO_MAX))

    logged = df["cvt_ratio"].astype(float)
    df["logged_cvt_ratio"] = logged.where(
        logged.between(RATIO_MIN, RATIO_MAX) & valid
    )

    return df, stats


def downsample(df: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = math.ceil(len(df) / max_points)
    return df.iloc[::step]


def estimate_shift_rpm(df: pd.DataFrame):
    """Rough shift-out RPM: median primary RPM while the ratio is mid-transition."""
    r = df["computed_cvt_ratio"].dropna()
    if len(r) < 20:
        return None
    lo, hi = r.quantile(0.05), r.quantile(0.95)
    if hi - lo < 0.3:
        return None
    band_lo, band_hi = lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)
    mid = df[df["computed_cvt_ratio"].between(band_lo, band_hi)]
    if mid.empty:
        return None
    return int(mid["primary_rpm"].median())


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="SUU Baja Telemetry", page_icon="🏎️", layout="wide")
st.title("🏎️ SUU Baja Go-Kart Telemetry Dashboard")

# ---- CSV import ----
with st.expander("📥 Import Telemetry CSV Log", expanded=not fetch_run_ids()):
    st.caption(
        "Upload the CSV from the SD card. Required columns: "
        + ", ".join(REQUIRED_IMPORT_COLUMNS)
        + ". Optional: "
        + ", ".join(OPTIONAL_IMPORT_COLUMNS)
        + "."
    )
    uploaded_file = st.file_uploader("CSV log file", type=["csv"], key="sd_upload")

    default_run_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    c1, c2 = st.columns([3, 1])
    with c1:
        run_name = st.text_input("Run name:", value=default_run_name, key="sd_run_name")
    with c2:
        st.write("")
        st.write("")
        do_import = st.button("Import", disabled=uploaded_file is None, use_container_width=True)

    if do_import and uploaded_file is not None:
        run_name = run_name.strip() or default_run_name
        confirmed = st.session_state.get("sd_import_confirm_name") == run_name

        if run_id_exists(run_name) and not confirmed:
            st.session_state["sd_import_confirm_name"] = run_name
            st.warning(
                f"Run '{run_name}' already has data. Click Import again to merge these rows "
                f"into it (existing timestamps are kept), or change the run name."
            )
        else:
            st.session_state.pop("sd_import_confirm_name", None)
            inserted, dupes, skipped, error = import_sd_csv(uploaded_file, run_name)
            if error:
                st.error(error)
            else:
                msg = f"Imported {inserted} new samples into run '{run_name}'."
                if dupes:
                    msg += f" {dupes} duplicate timestamp(s) ignored."
                if skipped:
                    msg += f" {skipped} malformed row(s) skipped."
                st.session_state["flash"] = msg
                clear_caches()
                st.rerun()

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

run_ids = fetch_run_ids()
if not run_ids:
    st.warning("No telemetry data found. Import a run above to get started.")
    st.stop()

# ---- Run selection / options ----
top1, top2, top3, top4 = st.columns([4, 2, 2, 1])
with top1:
    selected_run = st.selectbox("Select Active Run:", run_ids)
with top2:
    st.write("")
    smooth = st.toggle(f"Smooth RPM (rolling median, {SMOOTH_WINDOW})", value=True)
with top3:
    compare_runs = st.multiselect(
        "Overlay other runs:", [r for r in run_ids if r != selected_run]
    )
with top4:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        clear_caches()
        st.rerun()

assert selected_run is not None

# ---- Sidebar metadata ----
with st.sidebar:
    st.header("📝 Manual Run Metadata")
    st.caption(f"Editing metadata for: **{selected_run}**")

    existing = fetch_run_metadata(selected_run)

    def _clean(v, default):
        return default if v is None or (isinstance(v, float) and math.isnan(v)) else v

    with st.form("manual_entry_form", clear_on_submit=False):
        lap_time = st.text_input(
            "Best Lap Time (m:ss.s):",
            value=_clean(existing.get("lap_time"), ""),
            help="Format: 1:14.2. Leave blank to clear.",
        )
        max_gps_mph = st.number_input(
            "Max GPS Speed (MPH):",
            min_value=0.0,
            max_value=120.0,
            step=0.1,
            value=float(_clean(existing.get("gps_max_mph"), 0.0)),
            help="0 clears the value.",
        )
        notes = st.text_area(
            "Track Conditions / Setup Notes:",
            value=_clean(existing.get("notes"), ""),
        )

        if st.form_submit_button("💾 Save Metadata to Run"):
            lap = lap_time.strip()
            if lap and not re.fullmatch(r"\d+:[0-5]\d(\.\d+)?", lap):
                st.warning("Lap time not saved. Expected m:ss.s (e.g. 1:14.2).")
            else:
                ok = save_run_metadata(
                    selected_run,
                    lap or None,
                    max_gps_mph if max_gps_mph > 0 else None,
                    notes.strip() or None,
                )
                if ok:
                    fetch_run_metadata.clear()
                    st.session_state["flash"] = f"Saved metadata for {selected_run}."
                    st.rerun()

# ---- Load + clean ----
raw_run_df = fetch_run(selected_run)
if raw_run_df.empty:
    st.warning("Selected run contains no samples.")
    st.stop()

run_df, clean_stats = clean_run_df(raw_run_df, smooth)
if run_df.empty:
    st.warning("All records were removed during cleaning. Check sensor data quality.")
    st.stop()

removed = clean_stats["out_of_range"] + clean_stats["spikes"]
if removed:
    st.caption(
        f"Cleaning removed {removed} of {clean_stats['input']} samples "
        f"({clean_stats['out_of_range']} out of range, {clean_stats['spikes']} spikes)."
    )

run_meta = fetch_run_metadata(selected_run)
best_lap = run_meta.get("lap_time") or "N/A"
gps_mph = run_meta.get("gps_max_mph")
run_notes = run_meta.get("notes")
if gps_mph is not None and isinstance(gps_mph, float) and math.isnan(gps_mph):
    gps_mph = None

# ---- Metrics ----
duration = round(float(run_df["elapsed_s"].max()), 2)
max_p_rpm = int(run_df["primary_rpm"].max())
ratio_series = run_df["computed_cvt_ratio"].dropna()
max_ratio = f"{ratio_series.max():.2f}" if not ratio_series.empty else "N/A"
shift_rpm = estimate_shift_rpm(run_df)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Run Duration", f"{duration}s")
m2.metric("Best Lap Time", best_lap)
m3.metric("GPS Top Speed", f"{gps_mph:.1f} MPH" if gps_mph else "N/A")
m4.metric("Peak Primary RPM", f"{max_p_rpm:,}")
m5.metric("Max Shift Ratio", max_ratio)
m6.metric("Est. Shift-Out RPM", f"{shift_rpm:,}" if shift_rpm else "N/A")

if run_notes:
    st.info(f"**Run Notes:** {run_notes}")

st.download_button(
    "⬇️ Download cleaned run as CSV",
    data=run_df.drop(columns=["run_id"]).to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_run}_cleaned.csv",
    mime="text/csv",
)

st.divider()

# ---- Charts ----
plot_df = downsample(run_df)
if len(plot_df) < len(run_df):
    st.caption(f"Charts show every {math.ceil(len(run_df) / MAX_CHART_POINTS)}th sample for speed.")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Engine vs. Secondary RPM")
    st.line_chart(plot_df, x="elapsed_s", y=["primary_rpm", "secondary_rpm"],
                  x_label="Elapsed time (s)", y_label="RPM")

with chart_col2:
    st.subheader("CVT Shift Ratio Over Time")
    ratio_cols = [c for c in ["computed_cvt_ratio", "logged_cvt_ratio"]
                  if run_df[c].notna().any()]
    if ratio_cols:
        st.line_chart(plot_df, x="elapsed_s", y=ratio_cols,
                      x_label="Elapsed time (s)", y_label="Ratio")
        st.caption(f"Ratio shown only when secondary RPM ≥ {MIN_SECONDARY_RPM} "
                   f"and between {RATIO_MIN} and {RATIO_MAX}.")
    else:
        st.info("No valid ratio data for this run.")

st.divider()
st.subheader("CVT Tuning Curve (Primary RPM vs. Shift Ratio)")
st.caption("Plots engine RPM against drive ratio to help identify clutch shift-out points.")
tuning_df = downsample(run_df.dropna(subset=["computed_cvt_ratio"]))
if tuning_df.empty:
    st.info("No valid ratio samples to plot.")
else:
    st.scatter_chart(tuning_df, x="primary_rpm", y="computed_cvt_ratio",
                     color="secondary_rpm")

# ---- Run comparison ----
if compare_runs:
    st.divider()
    st.subheader("Run Comparison")
    frames = []
    for rid in [selected_run, *compare_runs]:
        other_df, _ = clean_run_df(fetch_run(rid), smooth)
        if not other_df.empty:
            frames.append(downsample(other_df)[["elapsed_s", "primary_rpm",
                                                "computed_cvt_ratio"]].assign(run=rid))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.caption("Primary RPM")
            st.line_chart(combined, x="elapsed_s", y="primary_rpm", color="run",
                          x_label="Elapsed time (s)", y_label="RPM")
        with cmp2:
            st.caption("CVT ratio")
            st.line_chart(combined.dropna(subset=["computed_cvt_ratio"]),
                          x="elapsed_s", y="computed_cvt_ratio", color="run",
                          x_label="Elapsed time (s)", y_label="Ratio")

# ---- Data grid ----
st.divider()
with st.expander("📋 Interactive Data Grid (Manual In-Line Edits)"):
    st.caption(
        "Edit Manual MPH and save. Clear a cell to remove its value. "
        f"Showing up to {EDITOR_MAX_ROWS} rows at a time."
    )

    max_start = max(len(run_df) - EDITOR_MAX_ROWS, 0)
    start = 0
    if max_start > 0:
        start = st.slider("Window start (row)", 0, max_start, 0, step=100)
    window = run_df.iloc[start:start + EDITOR_MAX_ROWS]

    display_cols = ["elapsed_s", "timestamp_ms", "primary_rpm", "secondary_rpm",
                    "computed_cvt_ratio", "manual_mph"]
    original = window[display_cols].reset_index(drop=True)

    edited = st.data_editor(
        original,
        num_rows="fixed",
        disabled=[c for c in display_cols if c != "manual_mph"],
        column_config={
            "manual_mph": st.column_config.NumberColumn("Manual MPH", min_value=0, max_value=120),
        },
        key=f"editor_{selected_run}_{start}",
    )

    if st.button("💾 Save Row Edits"):
        old = original["manual_mph"]
        new = edited["manual_mph"]
        changed = ~((old == new) | (old.isna() & new.isna()))
        changes = list(zip(new[changed], original.loc[changed, "timestamp_ms"]))
        n = save_manual_mph(selected_run, changes)
        if n:
            fetch_run.clear()
            st.session_state["flash"] = f"Saved {n} row edit(s)."
            st.rerun()
        else:
            st.info("No changes to save.")
