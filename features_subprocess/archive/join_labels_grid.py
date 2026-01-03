#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_labels_grid.py  —  schema-pinned label joiner (robust lon-mode; CSV or Parquet).

Keeps *all* feature columns from the input features file in the output,
then appends label columns (storm_point, storm_window, storm, near_storm,
pregen / pregen_h*, and t_to_storm_min_h). Writes every chunk with a fixed
schema so downstream steps see a stable set of columns.

Updates in this version:
  • --normalize-lon now accepts any string (trimmed) so YAML values like " -180..180" work.
  • CSV reads tolerate odd encodings / ragged lines.
  • Time normalization and long (up to 240h) pregen windows unchanged.
  • Adds continuous lead label t_to_storm_min_h (min future-hours to any storm
    within pregen_radius_deg, up to the max pregen horizon).
  • Supports true Parquet output via pyarrow when --out ends with .parquet/.parq/.pq.
  • --chunk-rows / --chunksize / --parquet-rows are aliases for --features-chunk-rows.
  • --overwrite controls whether an existing output file may be replaced.
  • Without --overwrite, if the output file already exists we SKIP (exit 0).
"""

import argparse
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd

# Optional parquet dependencies (for output)
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


# ---------------- I/O ----------------

def read_any(path: str, usecols=None) -> pd.DataFrame:
    low = str(path).lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(
        path,
        compression="infer",
        low_memory=False,
        usecols=usecols if usecols else None,
        encoding_errors="replace",
        on_bad_lines="skip",
    )


def iter_features(path: str | Path, chunksize: int | None):
    """
    Yield feature chunks.

    - For CSV: use pandas.read_csv(..., chunksize=...).
    - For Parquet: if pyarrow is available and chunksize > 0, stream via
      ParquetFile.iter_batches(batch_size=chunksize).
      Otherwise, fall back to a single read.
    """
    low = str(path).lower()
    is_parquet = low.endswith((".parquet", ".parq", ".pq"))

    # Parquet: stream via pyarrow if possible
    if is_parquet:
        # If we have pyarrow + a positive chunksize, stream batches
        if chunksize and chunksize > 0 and pq is not None:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=int(chunksize)):
                # Each batch is a pyarrow.Table or RecordBatch -> convert to pandas
                yield batch.to_pandas()
            return

        # Fallback: single shot read (may be heavy)
        yield read_any(path)
        return

    # Non-parquet: CSV (or similar) path
    if not chunksize or chunksize <= 0:
        # No chunking requested: read whole file once
        yield read_any(path)
        return

    # CSV streaming via pandas
    reader = pd.read_csv(
        path,
        compression="infer",
        low_memory=False,
        encoding_errors="replace",
        on_bad_lines="skip",
        chunksize=int(chunksize),
    )

    for chunk in reader:
        yield chunk


def write_any_csv_append(path: str | Path, df: pd.DataFrame, header: bool) -> None:
    """Append a chunk to a CSV/CSV.GZ file, creating it (with header) if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    name = p.name.lower()
    comp = "gzip" if (name.endswith(".csv.gz") or p.suffix.lower() == ".gz") else "infer"
    mode = "w" if header else "a"
    df.to_csv(
        p,
        index=False,
        compression=comp,
        date_format="%Y-%m-%d %H:%M:%S",
        mode=mode,
        header=header,
    )


# ---------------- utils ----------------

def to_utc_naive(s) -> pd.Series:
    """Parse to tz-naive UTC (datetime64[ns])."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    # convert from tz-aware UTC to naive
    return t.dt.tz_convert(None)


def _normalize_mode(mode: str | None) -> str:
    """Trim and normalize common tokens; default to '-180..180'."""
    if mode is None:
        return "-180..180"
    m = str(mode).strip()
    if m in ("-180..180", "0..360", "none"):
        return m
    # tolerate legacy spaced variants like " -180..180"
    if m.replace(" ", "") == "-180..180":
        return "-180..180"
    if m.replace(" ", "") == "0..360":
        return "0..360"
    if m.lower() in ("", "default"):
        return "-180..180"
    # fall back to safest
    return "-180..180"


def _norm_lon(vals, mode: str):
    x = pd.to_numeric(vals, errors="coerce")
    mode = _normalize_mode(mode)
    if mode == "none":
        return x.to_numpy()
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # default: -180..180
    return ((x + 180) % 360) - 180


def _deg2km_latlon(dlat_deg, dlon_deg, lat_ref_deg):
    lat_km = 111.2 * dlat_deg
    lon_km = 111.2 * np.cos(np.deg2rad(lat_ref_deg)) * dlon_deg
    return np.hypot(lat_km, lon_km)


def parse_hours_spec(spec: str | None) -> list[int]:
    if not spec:
        return []
    s = spec.strip()
    if ".." in s:
        a, b = [int(x.strip()) for x in s.split("..", 1)]
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = sorted({int(p) for p in parts})
    return [v for v in vals if v > 0]


def _apply_step(hours: list[int], step: int) -> list[int]:
    if step is None or step <= 1:
        return sorted(hours)
    return [h for h in sorted(hours) if (h % step) == 0]


def _by_hour_stats(name, y, t):
    if len(y) == 0:
        return
    idx = pd.to_datetime(t, utc=True, errors="coerce").dt.tz_convert(None).floor("h")
    s = (pd.Series(y, index=idx).groupby(level=0).sum()).astype(float)
    mean, std = float(s.mean()), float(s.std(ddof=0))
    cv = (std / mean) if mean else float("inf")
    print(
        f"[labels] {name}: hours={len(s):,} mean/h={mean:.1f} std={std:.1f} CV={cv:.3f} "
        f"min={int(s.min()) if len(s) else 0} max={int(s.max()) if len(s) else 0}",
        flush=True,
    )


# ---------------- column detection ----------------

TIME_CANDIDATES = ["time", "obs_time", "ISO_TIME", "datetime", "date_time", "valid_time"]
LAT_CANDIDATES  = ["lat", "LAT", "latitude", "y"]
LON_CANDIDATES  = ["lon", "LON", "longitude", "x"]


def _pick_col(df: pd.DataFrame, explicit: str | None, candidates: list[str], role: str) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {role} column. Tried {candidates}. "
        f"Available: {sorted(df.columns)[:20]} ..."
    )


# ---------------- matching core ----------------

def _slice_sorted_times(t_sorted, t_lo, t_hi):
    lo = np.datetime64(t_lo, "ns")
    hi = np.datetime64(t_hi, "ns")
    i0 = np.searchsorted(t_sorted, lo, side="left")
    i1 = np.searchsorted(t_sorted, hi, side="right")
    return i0, i1


def _has_within_radius(tr_lat, tr_lon, i0, i1, lat0, lon0, r_deg) -> bool:
    if i1 <= i0 or not np.isfinite(lat0) or not np.isfinite(lon0) or r_deg <= 0:
        return False
    la = tr_lat[i0:i1]
    lo = tr_lon[i0:i1]
    dlon = np.abs(lo - lon0)
    dlon = np.minimum(dlon, 360.0 - dlon)
    dlat = np.abs(la - lat0)
    mbox = (dlat <= r_deg) & (dlon <= r_deg)
    if not mbox.any():
        return False
    idx = np.nonzero(mbox)[0]
    dist_km = _deg2km_latlon(dlat[idx], dlon[idx], lat0)
    return bool((dist_km <= (r_deg * 111.2)).any())


def _min_future_dt_hours(tr_lat, tr_lon, t_sorted, i0, i1, lat0, lon0, ref_time, r_deg) -> float:
    if i1 <= i0 or not np.isfinite(lat0) or not np.isfinite(lon0) or r_deg <= 0:
        return np.inf
    la = tr_lat[i0:i1]
    lo = tr_lon[i0:i1]
    tt = t_sorted[i0:i1]
    m_future = tt >= ref_time
    if not m_future.any():
        return np.inf
    la = la[m_future]
    lo = lo[m_future]
    tt = tt[m_future]
    dlon = np.abs(lo - lon0)
    dlon = np.minimum(dlon, 360.0 - dlon)
    dlat = np.abs(la - lat0)
    mbox = (dlat <= r_deg) & (dlon <= r_deg)
    if not mbox.any():
        return np.inf
    la = la[mbox]
    lo = lo[mbox]
    tt = tt[mbox]
    dist_km = _deg2km_latlon(np.abs(la - lat0), np.abs(lo - lon0), lat0)
    mrad = dist_km <= (r_deg * 111.2)
    if not mrad.any():
        return np.inf
    dt_h = (tt[mrad].astype("datetime64[h]") - ref_time.astype("datetime64[h]")).astype(np.int64)
    if dt_h.size == 0:
        return np.inf
    return float(np.min(dt_h))


def _label_chunk(
    chunk: pd.DataFrame,
    tracks_df: pd.DataFrame,
    storm_radius_deg: float,
    storm_time_h: float,
    near_radius_deg: float,
    near_time_h: float,
    pregen_radius_deg: float,
    pregen_hours: list[int] | None,
    legacy_pregen_h: float | None,
) -> pd.DataFrame:
    """
    Compute storm_point (±0h), storm_window (±storm_time_h), near_storm (±near_time_h),
    pregen (single or multi), and t_to_storm_min_h (min future hours to any storm
    within pregen_radius_deg, up to max pregen horizon).
    Build all new columns once to avoid fragmentation.
    """
    out = chunk.copy()
    n = len(out)

    # Pre-allocate outputs
    storm_point  = np.zeros(n, dtype=np.int8)
    storm_window = np.zeros(n, dtype=np.int8)
    near         = np.zeros(n, dtype=np.int8)

    # continuous lead: initialise to +inf (no future storm seen yet)
    t_to_storm = np.full(n, np.inf, dtype=float)

    multi_h = sorted(pregen_hours or [])
    want_multi = len(multi_h) > 0
    max_future_h = (max(multi_h) if want_multi else (int(legacy_pregen_h) if legacy_pregen_h else 0))

    if want_multi:
        pre_mat = np.zeros((n, len(multi_h)), dtype=np.int8)
        pre_cols_map = {h: i for i, h in enumerate(multi_h)}
    else:
        pre_legacy = np.zeros(n, dtype=np.int8)

    # Early return if no tracks or empty chunk: still build labels once
    if tracks_df.empty or out.empty:
        label_dict = {
            "storm_point":        storm_point,
            "storm_window":       storm_window,
            "storm":              storm_window,  # legacy alias
            "near_storm":         near,
            "t_to_storm_min_h":   np.full(n, np.nan, dtype=float),
        }
        if want_multi:
            for h, j in pre_cols_map.items():
                label_dict[f"pregen_h{h}"] = pre_mat[:, j]
            label_dict["pregen"] = (
                pre_mat[:, pre_cols_map[max_future_h]] if max_future_h else np.zeros(n, dtype=np.int8)
            )
        else:
            label_dict["pregen"] = (
                pre_legacy if (legacy_pregen_h and legacy_pregen_h > 0) else np.zeros(n, dtype=np.int8)
            )

        labels_df = pd.DataFrame(label_dict, index=out.index)
        # only the int labels get cast to int8; t_to_storm_min_h stays float
        for c in ("storm_point", "storm_window", "storm", "near_storm"):
            labels_df[c] = labels_df[c].astype(np.int8, copy=False)
        if want_multi:
            for h in pre_cols_map:
                col = f"pregen_h{h}"
                labels_df[col] = labels_df[col].astype(np.int8, copy=False)
            labels_df["pregen"] = labels_df["pregen"].astype(np.int8, copy=False)
        else:
            labels_df["pregen"] = labels_df["pregen"].astype(np.int8, copy=False)

        return pd.concat([out, labels_df], axis=1, copy=False)

    # Prepare sorted track arrays (NumPy, once)
    t_tr = to_utc_naive(tracks_df["time"]).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(t_tr)
    t_sorted = t_tr[order]
    tr_lat = pd.to_numeric(tracks_df["lat"], errors="coerce").to_numpy()[order]
    tr_lon = pd.to_numeric(tracks_df["lon"], errors="coerce").to_numpy()[order]

    # Feature coords (NumPy, once)
    t_feat = to_utc_naive(out["time"]).to_numpy(dtype="datetime64[ns]")
    latv   = pd.to_numeric(out["lat"], errors="coerce").to_numpy()
    lonv   = pd.to_numeric(out["lon"], errors="coerce").to_numpy()

    # Main loop
    for i in range(n):
        tf  = t_feat[i]
        la0 = float(latv[i])
        lo0 = float(lonv[i])

        # storm_point: ±0h
        if storm_radius_deg > 0:
            i0, i1 = _slice_sorted_times(t_sorted, tf, tf)
            if _has_within_radius(tr_lat, tr_lon, i0, i1, la0, lo0, storm_radius_deg):
                storm_point[i] = 1

        # storm_window: ±storm_time_h
        if storm_radius_deg > 0 and storm_time_h > 0:
            j0, j1 = _slice_sorted_times(
                t_sorted,
                tf - np.timedelta64(int(storm_time_h), "h"),
                tf + np.timedelta64(int(storm_time_h), "h"),
            )
            if _has_within_radius(tr_lat, tr_lon, j0, j1, la0, lo0, storm_radius_deg):
                storm_window[i] = 1

        # near_storm: ±near_time_h
        if near_radius_deg > 0 and near_time_h > 0:
            k0, k1 = _slice_sorted_times(
                t_sorted,
                tf - np.timedelta64(int(near_time_h), "h"),
                tf + np.timedelta64(int(near_time_h), "h"),
            )
            if _has_within_radius(tr_lat, tr_lon, k0, k1, la0, lo0, near_radius_deg):
                near[i] = 1

        # pregen + continuous lead (future window)
        if pregen_radius_deg > 0 and max_future_h > 0:
            m0, m1 = _slice_sorted_times(
                t_sorted,
                tf,
                tf + np.timedelta64(int(max_future_h), "h"),
            )
            min_dt_h = _min_future_dt_hours(
                tr_lat, tr_lon, t_sorted, m0, m1, la0, lo0, tf, pregen_radius_deg
            )
            if np.isfinite(min_dt_h):
                # record continuous lead
                t_to_storm[i] = float(min_dt_h)

                if want_multi:
                    for h, j in pre_cols_map.items():
                        if min_dt_h <= h:
                            pre_mat[i, j] = 1
                else:
                    if legacy_pregen_h and (min_dt_h <= legacy_pregen_h):
                        pre_legacy[i] = 1

    # Assemble label columns ONCE
    label_dict = {
        "storm_point":      storm_point,
        "storm_window":     storm_window,
        "storm":            storm_window,
        "near_storm":       near,
        "t_to_storm_min_h": np.where(np.isfinite(t_to_storm), t_to_storm, np.nan),
    }
    if want_multi:
        for h, j in pre_cols_map.items():
            label_dict[f"pregen_h{h}"] = pre_mat[:, j]
        label_dict["pregen"] = (
            pre_mat[:, pre_cols_map[max_future_h]] if max_future_h else np.zeros(n, dtype=np.int8)
        )
    else:
        label_dict["pregen"] = (
            pre_legacy if (legacy_pregen_h and legacy_pregen_h > 0) else np.zeros(n, dtype=np.int8)
        )

    labels_df = pd.DataFrame(label_dict, index=out.index)
    # Cast only the 0/1-type labels to int8; leave t_to_storm_min_h as float
    for c in ("storm_point", "storm_window", "storm", "near_storm"):
        labels_df[c] = pd.to_numeric(labels_df[c], errors="coerce").fillna(0).astype(np.int8, copy=False)
    if want_multi:
        for h in pre_cols_map:
            col = f"pregen_h{h}"
            labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce").fillna(0).astype(np.int8, copy=False)
    labels_df["pregen"] = pd.to_numeric(labels_df["pregen"], errors="coerce").fillna(0).astype(np.int8, copy=False)
    # t_to_storm_min_h stays float with NaNs for "no future storm"

    return pd.concat([out, labels_df], axis=1, copy=False)


# ---------------- CLI / main ----------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Join labels (tracks) onto grid features (schema-pinned)."
    )
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels",   required=True)
    ap.add_argument("--out",      required=True)

    # Labeling knobs
    ap.add_argument("--storm_radius_deg", type=float, default=1.0)
    ap.add_argument("--storm_time_h",     type=float, default=3.0)
    ap.add_argument("--near_radius_deg",  type=float, default=5.0)
    ap.add_argument("--near_time_h",      type=float, default=12.0)

    # PREGEN: either legacy single horizon OR multi-horizon per hour
    ap.add_argument("--pregen_radius_deg", type=float, default=5.0)
    ap.add_argument(
        "--pregen_future_h",
        type=float,
        default=48.0,
        help="Legacy single-horizon pregen (ignored if --pregen-hours is set).",
    )
    ap.add_argument(
        "--pregen-hours",
        type=str,
        default=None,
        help='Multi-horizon per-hour labels, e.g., "1..240" or "6,12,18".',
    )
    ap.add_argument("--pregen-step", type=int, default=1)

    # Lon / chunking
    # Accept any string, trim inside, to tolerate YAML values like ' -180..180'
    ap.add_argument("--normalize-lon", default="-180..180")
    ap.add_argument("--chunk-hours", type=int, default=72)
    ap.add_argument(
        "--features-chunk-rows",
        type=int,
        default=2_000_000,
        help="Stream features CSV in row chunks to reduce memory (0 disables streaming).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Alias for --features-chunk-rows (compatibility with orchestrator hints).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Alias for --features-chunk-rows (compatibility with orchestrator hints).",
    )
    ap.add_argument(
        "--parquet-rows",
        type=int,
        default=None,
        help="Alias for --features-chunk-rows when using parquet (compatibility only).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file; otherwise skip if it exists.",
    )

    # Optional explicit label columns
    ap.add_argument("--labels-time-col", default=None)
    ap.add_argument("--labels-lat-col",  default=None)
    ap.add_argument("--labels-lon-col",  default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    # Harmonize chunk flags: aliases override the default
    if args.chunk_rows:
        args.features_chunk_rows = args.chunk_rows
    if args.chunksize:
        args.features_chunk_rows = args.chunksize
    if args.parquet_rows:
        args.features_chunk_rows = args.parquet_rows

    out_path = Path(args.out)
    out_low = out_path.suffix.lower()
    parquet_out = out_low in (".parquet", ".parq", ".pq")

    # Skip-if-exists behaviour (Option B)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] output already exists, not overwriting: {out_path}")
        return

    if parquet_out and pa is None:
        print(
            "\nERROR: pyarrow is required to write parquet output. "
            "Install 'pyarrow' or use a .csv/.csv.gz output path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If overwriting, remove existing file so writers start clean
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    # Read labels
    lab_raw = read_any(args.labels)

    # Pick columns from labels
    lt = _pick_col(lab_raw, args.labels_time_col, TIME_CANDIDATES, "labels time")
    la = _pick_col(lab_raw, args.labels_lat_col,  LAT_CANDIDATES,  "labels lat")
    lo = _pick_col(lab_raw, args.labels_lon_col,  LON_CANDIDATES,  "labels lon")

    lon_mode = _normalize_mode(args.normalize_lon)

    lab = pd.DataFrame(
        {
            "time": to_utc_naive(lab_raw[lt]),
            "lat":  pd.to_numeric(lab_raw[la], errors="coerce"),
            "lon":  _norm_lon(lab_raw[lo], lon_mode),
        }
    ).dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    # PREGEN horizons
    pregen_hours = parse_hours_spec(args.pregen_hours)
    if pregen_hours:
        pregen_hours = _apply_step(pregen_hours, int(args.pregen_step or 1))
    legacy_pregen_h = float(args.pregen_future_h) if (not pregen_hours) else None

    lab.sort_values("time", kind="mergesort", inplace=True, ignore_index=True)

    # Padding for time slices (windows + max pregen horizon)
    pad_back = pd.Timedelta(hours=max(args.storm_time_h, args.near_time_h, 0.0))
    max_future = (max(pregen_hours) if pregen_hours else (legacy_pregen_h or 0.0))
    pad_fwd  = pd.Timedelta(
        hours=max(args.storm_time_h, args.near_time_h, max_future, 0.0)
    )

    wrote_header = False  # CSV header tracking
    feature_cols = None

    # Parquet streaming writer state
    parquet_writer = None

    # Pre-compute label column order for output
    label_cols = ["storm_point", "storm_window", "storm", "near_storm"]
    if pregen_hours:
        label_cols += [f"pregen_h{h}" for h in sorted(pregen_hours)]
    label_cols += ["pregen", "t_to_storm_min_h"]

    step = pd.Timedelta(hours=max(1, int(args.chunk_hours)))

    for raw_chunk in iter_features(args.features, int(args.features_chunk_rows or 0)):
        need_f = {"time", "lat", "lon"}
        if raw_chunk is None or raw_chunk.empty:
            continue
        if not need_f.issubset(set(raw_chunk.columns)):
            raise ValueError(
                f"Features file must contain columns: {sorted(need_f)} "
                f"(got {sorted(raw_chunk.columns)[:12]}...)"
            )

        # Normalize coordinates (does not drop arbitrary columns)
        chunk = raw_chunk.copy()
        chunk["time"] = to_utc_naive(chunk["time"])
        chunk["lat"]  = pd.to_numeric(chunk["lat"], errors="coerce")
        chunk["lon"]  = _norm_lon(chunk["lon"], lon_mode)
        chunk = chunk.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

        if chunk.empty:
            continue

        if feature_cols is None:
            feature_cols = list(chunk.columns)  # includes physics vars + any extras
            for core in ["time", "lat", "lon"]:
                if core in feature_cols:
                    feature_cols.remove(core)
            feature_cols = ["time", "lat", "lon"] + feature_cols

            print(f"[schema] preserving {len(feature_cols)} feature columns:")
            print(
                "         " + ", ".join(feature_cols[:20]) +
                (" ..." if len(feature_cols) > 20 else "")
            )

            final_order = feature_cols + label_cols

        chunk = chunk.reindex(columns=feature_cols)
        chunk.sort_values("time", inplace=True, ignore_index=True)

        tmin = chunk["time"].min()
        tmax = chunk["time"].max()
        lo_t = tmin

        while lo_t <= tmax:
            hi_t = min(lo_t + step - pd.Timedelta(seconds=1), tmax)
            slice_chunk = chunk.loc[(chunk["time"] >= lo_t) & (chunk["time"] <= hi_t)].copy()
            lab_slice = lab.loc[
                (lab["time"] >= (lo_t - pad_back)) & (lab["time"] <= (hi_t + pad_fwd))
            ].copy()

            if lab_slice.empty:
                n_chunk = len(slice_chunk)
                label_dict = {
                    "storm_point":        np.zeros(n_chunk, dtype=np.int8),
                    "storm_window":       np.zeros(n_chunk, dtype=np.int8),
                    "storm":              np.zeros(n_chunk, dtype=np.int8),
                    "near_storm":         np.zeros(n_chunk, dtype=np.int8),
                    "t_to_storm_min_h":   np.full(n_chunk, np.nan, dtype=float),
                }
                if pregen_hours:
                    for h in pregen_hours:
                        label_dict[f"pregen_h{h}"] = np.zeros(n_chunk, dtype=np.int8)
                    label_dict["pregen"] = np.zeros(n_chunk, dtype=np.int8)
                else:
                    label_dict["pregen"] = np.zeros(n_chunk, dtype=np.int8)

                labels_df = pd.DataFrame(label_dict, index=slice_chunk.index)
                for c in ("storm_point", "storm_window", "storm", "near_storm", "pregen"):
                    labels_df[c] = labels_df[c].astype(np.int8, copy=False)
                if pregen_hours:
                    for h in pregen_hours:
                        col = f"pregen_h{h}"
                        labels_df[col] = labels_df[col].astype(np.int8, copy=False)

                labeled = pd.concat([slice_chunk, labels_df], axis=1, copy=False)
            else:
                labeled = _label_chunk(
                    slice_chunk,
                    lab_slice,
                    float(args.storm_radius_deg),
                    float(args.storm_time_h),
                    float(args.near_radius_deg),
                    float(args.near_time_h),
                    float(args.pregen_radius_deg),
                    pregen_hours,
                    legacy_pregen_h,
                )

            labeled = labeled.reindex(columns=final_order)

            if parquet_out:
                # True parquet streaming via ParquetWriter
                table = pa.Table.from_pandas(labeled, preserve_index=False)
                if parquet_writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    parquet_writer = pq.ParquetWriter(str(out_path), table.schema)
                parquet_writer.write_table(table)
            else:
                write_any_csv_append(args.out, labeled, header=(not wrote_header))
                wrote_header = True

            print(
                f"[label] wrote {len(labeled):,} rows  "
                f"{lo_t:%Y-%m-%d %H:%M} -> {hi_t:%Y-%m-%d %H:%M}  "
                f"feat={len(slice_chunk):,}  tracks={len(lab_slice):,}",
                flush=True,
            )

            del slice_chunk, lab_slice, labeled
            lo_t = hi_t + pd.Timedelta(seconds=1)

    # Close parquet writer if used
    if parquet_writer is not None:
        parquet_writer.close()

    # If we never saw any usable feature rows, bail with a clearer message
    if feature_cols is None:
        print(
            "\nERROR: no usable feature rows were found in the features file "
            "(after time/lat/lon parsing and NaN drops).",
            file=sys.stderr,
        )
        sys.exit(1)

    # quick label diagnostics (small readback)
    try:
        probe = None
        out_low_name = str(args.out).lower()
        if out_low_name.endswith(('.csv', '.csv.gz', '.tsv', '.tsv.gz', '.txt', '.txt.gz', '.gz')):
            probe = pd.read_csv(args.out, nrows=5000, low_memory=False, encoding_errors="replace")
        elif out_low_name.endswith((".parquet", ".parq", ".pq")):
            probe = pd.read_parquet(args.out)

        if probe is not None and {"time", "storm_point", "storm_window"}.issubset(probe.columns):
            _by_hour_stats(
                "storm_point",
                pd.to_numeric(probe["storm_point"], errors="coerce").fillna(0).to_numpy(),
                probe["time"],
            )
            _by_hour_stats(
                "storm_window",
                pd.to_numeric(probe["storm_window"], errors="coerce").fillna(0).to_numpy(),
                probe["time"],
            )
    except Exception:
        pass

    feat_count = len(feature_cols) if feature_cols is not None else 0
    label_count = len(label_cols)

    print(
        f"[ok] streamed output -> {args.out}  "
        f"(columns preserved: {feat_count} features + {label_count} labels)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Full traceback + clearer error line
        traceback.print_exc()
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
