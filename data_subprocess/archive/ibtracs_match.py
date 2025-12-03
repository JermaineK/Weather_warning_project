#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ibtracs_match.py — match scored alerts to IBTrACS tracks in space–time.

Inputs
------
1) Alerts table (typically produced by your scoring / threshold stage), e.g.

   • Heuristic scorer:
       results/alerts_scored.csv.gz
     with at least:
       - time      (UTC, parseable)
       - lat       (degrees)
       - lon       (degrees, any wrap; we'll normalise)
       - score     (or prob / prob_max — configurable)
       - row_id    (optional but recommended; kept if present)

   • ML pipeline:
       results/alerts/alerts_coral_sea_demo_lead24_thr*.csv.gz
     where you then point --alerts to that file and use
       --alerts-score-col prob_max
     or similar.

2) Tracks table (IBTrACS subset), e.g. data/tracks/tracks_subset.csv with columns:
     - time *or* obs_time  (storm observation time)
     - lat, lon            (storm centre)
     - vmax, pmin          (optional but useful)
     - name, basin, storm_id, source (optional metadata)

3) Optional storms list (to filter tracks), e.g. storms_list.csv with either:
     - sid                   (matches sid or storm_id in tracks)
     - storm_id              (same)
   Any extra columns are ignored.

What it does
------------
For each alert row:
  • Find the *nearest* storm track point in time (within ±max_time_h hours).
  • Compute great-circle distance [km] between alert (lat,lon) and storm centre.
  • Define:
        lead_h      = (storm_time - alert_time) in hours
        min_dist_km = that distance
        storm_hit   = 1 if:
                          |lead_h| <= max_lead_h
                      AND min_dist_km <= max_dist_km
                      AND a storm obs existed within ±max_time_h.
                    Otherwise 0.

Outputs
-------
A single CSV(.gz) with all alert rows plus:
  - storm_hit (0/1, int8)
  - lead_h (float)
  - min_dist_km (float)
  - storm_time (timestamp)
  - storm_name, storm_id, basin, source (when available)
  - storm_vmax, storm_pmin

Typical use
-----------
# Heuristic scorer (thresholds_and_alerts.py):
python data_subprocess/ibtracs_match.py ^
  --alerts results/alerts_scored.csv.gz ^
  --alerts-score-col score ^
  --tracks data/tracks/tracks_subset.csv ^
  --storms-list data/tracks/storms_list.csv ^
  --out results/alerts_with_targets.csv.gz ^
  --max-time-h 96 --max-lead-h 72 --max-dist-km 300

# ML pipeline alerts (e.g. per-lead thresholded file with prob_max):
python data_subprocess/ibtracs_match.py ^
  --alerts results/alerts/alerts_coral_sea_demo_lead24_thr_Fbeta.csv.gz ^
  --alerts-score-col prob_max ^
  --tracks data/tracks/tracks_subset.csv ^
  --storms-list data/tracks/storms_list.csv ^
  --out results/alerts_lead24_with_targets.csv.gz ^
  --max-time-h 96 --max-lead-h 72 --max-dist-km 300

This file is then suitable for threshold calibration / sanity checks.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd


# -------------------- small utilities --------------------

def _print(*a, **k):
    print(*a, **k, flush=True)


def _to_utc_naive(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    """Parse timestamps to tz-naive UTC (pandas datetime64[ns])."""
    if pd.api.types.is_datetime64_any_dtype(series):
        t = pd.to_datetime(series, utc=True, errors="coerce")
        return t.dt.tz_convert(None)
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    t = (
        pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
        if fmt
        else pd.to_datetime(raw, utc=True, errors="coerce")
    )
    return t.dt.tz_convert(None)


def _norm_lon(vals: pd.Series, mode: str = "-180..180") -> pd.Series:
    """Normalise longitudes if desired."""
    x = pd.to_numeric(vals, errors="coerce")
    mode = (mode or "-180..180").replace(" ", "")
    if mode.lower() == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # default: -180..180
    return ((x + 180.0) % 360.0) - 180.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Great-circle distance [km] using the haversine formula.
    Inputs in degrees; outputs float array.
    """
    R = 6371.0  # Earth radius [km]

    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def _is_probably_parquet(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        return head == b"PAR1" or tail == b"PAR1"
    except Exception:
        return False


# -------------------- core helpers --------------------

def load_tracks(
    path: Path,
    storms_list_path: Optional[Path] = None,
    time_col_prefer: str = "time",
    obs_time_col_alt: str = "obs_time",
) -> pd.DataFrame:
    """
    Load IBTrACS subset and harmonise column names.

    Returns a DataFrame with at least:
      storm_time, storm_lat, storm_lon, storm_id, storm_name,
      storm_vmax, storm_pmin, basin, source
    sorted by storm_time.
    """
    if path.suffix.lower() in {".parquet", ".parq", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            encoding_errors="replace",
            on_bad_lines="skip",
        )

    # Harmonise time column
    if time_col_prefer in df.columns:
        tcol = time_col_prefer
    elif obs_time_col_alt in df.columns:
        tcol = obs_time_col_alt
    elif "time_h" in df.columns:
        tcol = "time_h"
    else:
        raise ValueError(f"Tracks file {path} missing time/obs_time/time_h column.")

    df["storm_time"] = _to_utc_naive(df[tcol], None)
    df["storm_time"] = pd.to_datetime(df["storm_time"], errors="coerce").astype("datetime64[ns]")

    # Basic required coords
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(f"Tracks file {path} missing lat/lon columns.")

    df["storm_lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["storm_lon"] = _norm_lon(pd.to_numeric(df["lon"], errors="coerce"))

    # Metadata with safe defaults
    df["storm_name"] = df["name"] if "name" in df.columns else ""
    if "storm_id" in df.columns:
        df["storm_id"] = df["storm_id"]
    elif "sid" in df.columns:
        df["storm_id"] = df["sid"]
    else:
        df["storm_id"] = ""

    df["storm_vmax"] = (
        pd.to_numeric(df["vmax"], errors="coerce") if "vmax" in df.columns else np.nan
    )
    df["storm_pmin"] = (
        pd.to_numeric(df["pmin"], errors="coerce") if "pmin" in df.columns else np.nan
    )
    df["basin"] = df["basin"] if "basin" in df.columns else ""
    df["source"] = df["source"] if "source" in df.columns else ""

    # --- Robustness: drop original lat/lon/time columns to avoid *_x / *_y collisions ---
    drop_cols: List[str] = []
    # Original coordinate columns
    for c in ("lat", "lon"):
        if c in df.columns:
            drop_cols.append(c)
    # Original time-like columns (we now use storm_time)
    for c in (tcol, obs_time_col_alt, "time_h"):
        if c in df.columns and c != "storm_time":
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=list(dict.fromkeys(drop_cols)))  # de-dup while preserving order

    # Optional filter via storms_list
    if storms_list_path is not None:
        if storms_list_path.exists():
            sl = pd.read_csv(storms_list_path)
            id_col = None
            for cand in ["storm_id", "sid", "SID", "STORM_ID"]:
                if cand in sl.columns:
                    id_col = cand
                    break
            if id_col is None:
                _print(
                    f"[ibtracs] storms_list {storms_list_path} has no storm_id/sid; using all tracks."
                )
            else:
                keep_ids = set(sl[id_col].astype(str).str.strip())
                before = len(df)
                df = df[df["storm_id"].astype(str).str.strip().isin(keep_ids)].reset_index(
                    drop=True
                )
                _print(
                    f"[ibtracs] filtered tracks by storms_list: {before} -> {len(df)} rows"
                )
        else:
            _print(
                f"[ibtracs] storms_list file not found: {storms_list_path}; using all tracks"
            )

    # Drop rows without usable time/coords
    df = df.dropna(subset=["storm_time", "storm_lat", "storm_lon"]).reset_index(
        drop=True
    )
    df = df.sort_values("storm_time").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Tracks file {path} produced no usable rows after cleaning.")

    _print(f"[ibtracs] loaded {len(df)} track points from {path}")
    return df


def iter_alert_chunks(
    path: Path,
    chunksize: int,
    usecols: Optional[List[str]] = None,
) -> Any:
    if path.suffix.lower() in {".parquet", ".parq", ".pq"} or _is_probably_parquet(path):
        # Parquet: stream row groups via pyarrow if available; fallback to whole read
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            cols = usecols if usecols else None
            for batch in pf.iter_batches(batch_size=chunksize, columns=cols):
                yield batch.to_pandas()
            return
        except Exception:
            pass
        df = pd.read_parquet(path, columns=usecols if usecols else None)
        yield df
        return

    it = pd.read_csv(
        path,
        compression="infer",
        low_memory=False,
        chunksize=int(chunksize),
        usecols=usecols,
        encoding_errors="replace",
        on_bad_lines="skip",
    )
    if not hasattr(it, "__iter__"):
        it = [it]
    for ch in it:
        yield ch


def match_alerts_to_tracks_chunk(
    chunk: pd.DataFrame,
    storms: pd.DataFrame,
    max_time_h: float,
    max_lead_h: float,
    max_dist_km: float,
    time_col: str,
    lat_col: str,
    lon_col: str,
    score_col: str,
) -> pd.DataFrame:
    """
    For a single alerts chunk:
      • Find nearest storm in time (merge_asof)
      • Compute distance, lead_h, storm_hit
    """
    if chunk is None or chunk.empty:
        return chunk

    # Ensure required columns in the incoming alerts chunk
    for c in [time_col, lat_col, lon_col, score_col]:
        if c not in chunk.columns:
            raise ValueError(f"Alerts chunk missing column '{c}'.")

    # Parse/clean alert coordinates
    A = chunk.copy()
    A[time_col] = _to_utc_naive(A[time_col], None)
    A[time_col] = pd.to_datetime(A[time_col], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    A[lat_col] = pd.to_numeric(A[lat_col], errors="coerce")
    A[lon_col] = _norm_lon(pd.to_numeric(A[lon_col], errors="coerce"))
    A[score_col] = pd.to_numeric(A[score_col], errors="coerce")

    A = A.dropna(subset=[time_col, lat_col, lon_col, score_col]).reset_index(
        drop=True
    )
    if A.empty:
        return A

    # Ensure storms time dtype matches alerts; sort both by time
    storms_local = storms.copy()
    storms_local["storm_time"] = pd.to_datetime(storms_local["storm_time"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")

    A["_order_idx"] = np.arange(len(A), dtype="int64")
    A = A.sort_values(time_col).reset_index(drop=True)
    storms_local = storms_local.sort_values("storm_time").reset_index(drop=True)

    # Merge nearest in time (within ±max_time_h)
    tol = pd.Timedelta(hours=float(max_time_h))
    matched = pd.merge_asof(
        A,
        storms_local,
        left_on=time_col,
        right_on="storm_time",
        direction="nearest",
        tolerance=tol,
    )

    # Defensive: if merge created suffixed coord names (lat_x/lon_x), normalise them back
    # so downstream code can always rely on `lat_col` / `lon_col`.
    if lat_col not in matched.columns:
        alt = f"{lat_col}_x"
        if alt in matched.columns:
            matched = matched.rename(columns={alt: lat_col})
    if lon_col not in matched.columns:
        alt = f"{lon_col}_x"
        if alt in matched.columns:
            matched = matched.rename(columns={alt: lon_col})

    # Some rows may have no nearby storm (storm_time NaN); handle that later
    has_storm = matched["storm_time"].notna()

    # Compute spatial distance only for rows with a storm
    dist_km = np.full(len(matched), np.nan, dtype="float64")
    if has_storm.any():
        dist_km[has_storm.values] = _haversine_km(
            matched.loc[has_storm, lat_col].values,
            matched.loc[has_storm, lon_col].values,
            matched.loc[has_storm, "storm_lat"].values,
            matched.loc[has_storm, "storm_lon"].values,
        )

    matched["min_dist_km"] = dist_km

    # Lead time (storm_time - alert_time) in hours
    dt = (matched["storm_time"] - matched[time_col]).dt.total_seconds() / 3600.0
    matched["lead_h"] = dt

    # Hit condition
    hit = (
        has_storm
        & np.isfinite(dist_km)
        & (np.abs(dt) <= float(max_lead_h))
        & (dist_km <= float(max_dist_km))
    )
    matched["storm_hit"] = hit.astype("int8")

    # Restore original alert order
    matched = (
        matched.sort_values("_order_idx")
        .drop(columns=["_order_idx"])
        .reset_index(drop=True)
    )
    return matched


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(
        description="Match scored alerts to IBTrACS tracks and compute hit/lead/distance."
    )
    ap.add_argument(
        "--alerts",
        required=True,
        help="Alerts/scored CSV(.gz) with time, lat, lon, score/prob columns.",
    )
    ap.add_argument(
        "--tracks",
        required=True,
        help="IBTrACS subset CSV (e.g. data/tracks/tracks_subset.csv).",
    )
    ap.add_argument(
        "--storms-list",
        default=None,
        help="Optional CSV with storm_id/sid column to restrict tracks.",
    )
    ap.add_argument(
        "--out",
        default="results/alerts_with_targets.csv.gz",
        help="Output CSV(.gz) with storm_hit, lead_h, min_dist_km, etc.",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="Alerts chunk size for streaming (default: 200k).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Alias for --chunk-rows (compatibility with orchestrator hints).",
    )
    ap.add_argument(
        "--parquet-rows",
        type=int,
        default=None,
        help="Alias for --chunk-rows when using parquet (compatibility only).",
    )

    # Column name knobs for alerts
    ap.add_argument(
        "--alerts-time-col",
        default="time",
        help="Time column in alerts file (default: time).",
    )
    ap.add_argument(
        "--alerts-lat-col",
        default="lat",
        help="Latitude column in alerts file (default: lat).",
    )
    ap.add_argument(
        "--alerts-lon-col",
        default="lon",
        help="Longitude column in alerts file (default: lon).",
    )
    ap.add_argument(
        "--alerts-score-col",
        default="score",
        help="Score/probability column in alerts file (default: score).",
    )

    # Matching parameters
    ap.add_argument(
        "--max-time-h",
        type=float,
        default=96.0,
        help="Max time separation (hours) for nearest storm search.",
    )
    ap.add_argument(
        "--max-lead-h",
        type=float,
        default=72.0,
        help="Lead window |lead_h| ≤ this is required for storm_hit.",
    )
    ap.add_argument(
        "--max-dist-km",
        type=float,
        default=300.0,
        help="Max great-circle distance (km) for storm_hit.",
    )

    args = ap.parse_args()
    if args.chunksize and not args.chunk_rows:
        args.chunk_rows = args.chunksize
    if args.parquet_rows and not args.chunk_rows:
        args.chunk_rows = args.parquet_rows

    alerts_path = Path(args.alerts)
    tracks_path = Path(args.tracks)
    storms_list_path = Path(args.storms_list) if args.storms_list else None
    out_path = Path(args.out)

    if not alerts_path.exists():
        raise SystemExit(f"[fatal] alerts file not found: {alerts_path}")
    if not tracks_path.exists():
        raise SystemExit(f"[fatal] tracks file not found: {tracks_path}")

    # Load and clean tracks
    storms = load_tracks(tracks_path, storms_list_path=storms_list_path)

    # Stream alerts, match, and write out
    usecols = None  # read all columns; you can tighten this if needed
    chunk_rows = int(args.chunk_rows)
    first = True
    total_rows = 0
    hits = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    _print(
        f"[ibtracs] matching alerts from {alerts_path} -> {out_path}"
    )
    _print(
        f"[ibtracs] max_time_h={args.max_time_h} max_lead_h={args.max_lead_h} max_dist_km={args.max_dist_km}"
    )

    for i, chunk in enumerate(
        iter_alert_chunks(alerts_path, chunk_rows, usecols=usecols), start=1
    ):
        if chunk is None or chunk.empty:
            continue

        matched = match_alerts_to_tracks_chunk(
            chunk=chunk,
            storms=storms,
            max_time_h=args.max_time_h,
            max_lead_h=args.max_lead_h,
            max_dist_km=args.max_dist_km,
            time_col=args.alerts_time_col,
            lat_col=args.alerts_lat_col,
            lon_col=args.alerts_lon_col,
            score_col=args.alerts_score_col,
        )

        if matched.empty:
            continue

        total_rows += len(matched)
        if "storm_hit" in matched.columns:
            hits += int(matched["storm_hit"].sum())

        mode = "w" if first else "a"
        header = first
        compression = "gzip" if out_path.name.lower().endswith(".gz") else "infer"
        matched.to_csv(
            out_path,
            index=False,
            mode=mode,
            header=header,
            compression=compression,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        first = False

        _print(
            f"[ibtracs] chunk {i}: wrote {len(matched)} rows "
            f"(cum={total_rows}, hits={hits})"
        )

    _print(
        f"[ibtracs] done. total_rows={total_rows}, total_hits={hits}, out={out_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
