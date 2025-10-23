#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- I/O helpers (aligned with denoise) ----------------

CANDIDATE_TIME_COLS = ["time", "time_h", "datetime", "valid_time", "forecast_time"]

def read_any(path, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, **kw)

def write_any(path, df: pd.DataFrame) -> None:
    p = str(path).lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if p.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)

# ---------------- shared helpers ----------------

def _try_parse_time_raw(s: pd.Series, fmt: str | None) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (mid and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def _parse_area(aoi: str | None):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Throttle per-hour by keeping the top quantile (optionally only among alert rows)."
    )
    ap.add_argument("--alerts", required=True, help="Alerts file: CSV/CSV.GZ/Parquet; needs time, lat, lon, flag/score cols")
    ap.add_argument("--out", required=True, help="Output file: CSV/CSV.GZ/Parquet")

    # Selection logic
    ap.add_argument("--keep-quantile", type=float, default=0.90, help="Fraction to keep per hour (default: 0.90)")
    ap.add_argument("--min-keep-per-hour", type=int, default=1,
                    help="Minimum number to keep per hour if any rows exist (safety floor).")
    ap.add_argument("--keep-frac-of-alerts", type=float, default=0.0,
                    help="If --only-alerts, keep at least this fraction of EXISTING positives per hour (default 0).")
    ap.add_argument("--protect-score-threshold", type=float, default=None,
                    help="Always keep rows with score/prob ≥ this value (applied before quantile).")

    # Columns / parsing
    ap.add_argument("--time-col", default="time", help="Time column name (default: time)")
    ap.add_argument("--time-format", default=None, help="Optional strftime for custom time parsing")
    ap.add_argument("--prob-col", default=None, help="Legacy probability/score column")
    ap.add_argument("--score-col", default=None, help="Score column (takes precedence over --prob-col)")
    ap.add_argument("--flag-col", default="alert_final", help="Binary alert flag column (default: alert_final)")
    ap.add_argument("--only-alerts", action="store_true",
                    help="Throttle only among rows where <flag-col> == 1")

    # Geo & debug
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="Normalize longitudes before processing (default: none)")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization')
    ap.add_argument("--sparse-output", action="store_true", help="Write only rows kept after throttling.")
    ap.add_argument("--debug", action="store_true", help="Print ranges and hourly counts")
    args = ap.parse_args()

    # ---- Load
    df = read_any(args.alerts).replace([np.inf,-np.inf], np.nan)

    # Time column detection
    if args.time_col not in df.columns:
        for c in CANDIDATE_TIME_COLS:
            if c in df.columns:
                args.time_col = c
                break

    required = {args.time_col, "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    # Parse time → hour bins
    t = _try_parse_time_raw(df[args.time_col], args.time_format)
    bad = int(t.isna().sum())
    if bad:
        frac = bad / len(t)
        print(f"[THROTTLE] Warning: {bad:,} invalid times ({frac:.1%}); dropping.", flush=True)
    df = df.loc[t.notna()].copy()
    df[args.time_col] = t[t.notna()].dt.tz_localize(None)
    df["time_h"] = df[args.time_col].dt.floor("h")

    # Numeric + lon norm + AOI
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _norm_lon(df["lon"], args.normalize_lon)
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

    aoi = _parse_area(args.area)
    if aoi:
        latN, lonW, latS, lonE = aoi
        before = len(df)
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)
        print(f"[THROTTLE] AOI crop: kept {len(df):,}/{before:,} rows", flush=True)

    # Score column (optional but preferred)
    score_col = args.score_col or args.prob_col
    has_score = (score_col is not None) and (score_col in df.columns)
    if has_score:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Flag handling & selection universe
    base_flag_col = args.flag_col
    if base_flag_col not in df.columns:
        # If not present, synthesize zeros (we’ll still throttle over the universe)
        df[base_flag_col] = 0

    if args.only_alerts:
        elig = df.loc[df[base_flag_col].astype(int) == 1].copy()
        if elig.empty:
            # Preserve shape, but write all zeros into flag (no kept)
            out = df.copy()
            out[f"{base_flag_col}_base"] = out[base_flag_col].astype(int)
            out[base_flag_col] = 0
            out = out if not args.sparse_output else out.loc[out[base_flag_col] == 1]
            write_any(args.out, out)
            print(f"Wrote {args.out} | throttled: 0/{len(df):,} (no eligible alerts)")
            return
    else:
        elig = df.copy()

    # Deterministic tie-break (stable regardless of score presence)
    lat_i = (elig["lat"].to_numpy() * 10000).round().astype(np.int64)
    lon_i = (elig["lon"].to_numpy() * 10000).round().astype(np.int64)
    elig["__tiebreak__"] = lat_i * 1_000_000 + lon_i

    sort_cols = ["time_h", "__tiebreak__"]
    if has_score:
        s = elig[score_col].astype(float).fillna(-np.inf)
        elig["__negscore__"] = -s  # descending score via ascending sort
        sort_cols = ["time_h", "__negscore__", "__tiebreak__"]

    # Pre-select “protected” rows by score threshold (survival guard)
    protected_idx = pd.Index([])
    if has_score and args.protect_score_threshold is not None:
        protected_idx = elig.index[elig[score_col] >= float(args.protect_score_threshold)]

    # Sort once for stable per-hour selection
    elig = elig.sort_values(sort_cols, ascending=True, kind="mergesort")

    # Per-hour keep counts
    grp = elig.groupby("time_h", sort=False)
    pos = grp.cumcount()
    n = grp["time_h"].transform("size").astype(int)

    # Hourly floors
    keep_q = np.ceil(args.keep_quantile * n.to_numpy()).astype(int)
    keep_q = np.maximum(keep_q, int(max(0, args.min_keep_per_hour)))  # quantile ∨ floor

    if args.only_alerts and float(args.keep_frac_of_alerts) > 0:
        # Count positives per hour among elig (which are already positives)
        alert_counts = n.to_numpy()
        keep_min_alerts = np.ceil(float(args.keep_frac_of_alerts) * alert_counts).astype(int)
        keep_q = np.maximum(keep_q, keep_min_alerts)

    # Respect “protected” set first
    keep_mask = pos.to_numpy() < keep_q
    kept_idx = elig.index[keep_mask]
    if len(protected_idx):
        kept_idx = pd.Index(np.union1d(kept_idx.values, protected_idx.values))

    # Write back into the SAME flag col; preserve base
    out = df.copy()
    out[f"{base_flag_col}_base"] = out[base_flag_col].astype(int)
    out[base_flag_col] = 0
    out.loc[kept_idx, base_flag_col] = 1

    kept = int(out[base_flag_col].sum())
    total = len(out)
    hours = out["time_h"].nunique()

    if args.debug:
        tmin, tmax = out["time_h"].min(), out["time_h"].max()
        print(f"[THROTTLE][debug] time range: {tmin} → {tmax} (hours={hours})", flush=True)
        print(f"[THROTTLE][debug] lat range: {out['lat'].min():.3f} .. {out['lat'].max():.3f}", flush=True)
        print(f"[THROTTLE][debug] lon range: {out['lon'].min():.3f} .. {out['lon'].max():.3f}", flush=True)

    frac = kept / total if total else 0.0
    print(f"[THROTTLE] hours={hours} keep-quantile={args.keep_quantile:.2f} "
          f"| kept {kept:,}/{total:,} ({frac:.3f})", flush=True)

    # Sparse output (optional)
    if args.sparse_output:
        before = len(out)
        out = out.loc[out[base_flag_col] == 1].reset_index(drop=True)
        print(f"[THROTTLE] sparse-output: kept {len(out):,}/{before:,} rows", flush=True)

    # Clean temp cols and write
    out.drop(columns=[c for c in ["__tiebreak__","__negscore__","time_h"] if c in out], inplace=True, errors="ignore")
    write_any(args.out, out)
    print(f"Wrote {args.out} | throttled: {kept:,}/{total:,}", flush=True)

if __name__ == "__main__":
    main()