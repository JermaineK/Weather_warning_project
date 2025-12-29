#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

def _strip_choice(val: str) -> str:
    return str(val).strip()

def _preprocess_norm(argv: list[str]) -> list[str]:
    """
    Allow --normalize-lon values that look like options (e.g., -180..180) by
    rewriting them to --normalize-lon=<value> before argparse runs.
    """
    out = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--normalize-lon" and i + 1 < len(argv):
            val = argv[i + 1]
            out.append(f"--normalize-lon={val}")
            skip = True
        else:
            out.append(tok)
    return out

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

def _stable_hash_frac(df: pd.DataFrame) -> np.ndarray:
    """
    Deterministic hash -> [0,1) for selection without row-order bias.
    Uses pandas siphash with a fixed key for reproducibility across runs.
    """
    if df.empty:
        return np.array([], dtype=np.float64)
    hashed = hash_pandas_object(df, index=False, hash_key="keepq", encoding="utf8")
    vals = hashed.to_numpy(dtype=np.uint64)
    scale = float(1 << 53)  # keep within exact float mantissa range
    return (vals % np.uint64(1 << 53)).astype(np.float64) / scale

def _deterministic_keep_mask(elig: pd.DataFrame,
                             keep_counts: dict,
                             keep_quantile: float,
                             protected_idx: pd.Index) -> np.ndarray:
    """
    Order-invariant selection: within each hour, keep a fraction of rows
    based on a stable hash of (time_h, lat, lon[, row_id]).
    """
    n = len(elig)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if keep_quantile is None or keep_quantile >= 1.0:
        return np.ones(n, dtype=bool)

    key_cols = {
        "time_h": elig["time_h"].to_numpy(),
        "lat": elig["lat"].round(6).to_numpy(),
        "lon": elig["lon"].round(6).to_numpy(),
    }
    if "row_id" in elig.columns:
        key_cols["row_id"] = pd.to_numeric(elig["row_id"], errors="coerce")
    elig = elig.copy()
    elig["__keep_hash__"] = _stable_hash_frac(pd.DataFrame(key_cols))

    protected_mask = elig.index.isin(protected_idx)
    keep_mask = np.zeros(n, dtype=bool)

    for t_val, idx in elig.groupby("time_h", sort=False).indices.items():
        idx_arr = np.fromiter(idx, dtype=np.int64)
        sub = elig.loc[idx_arr]
        sub_protected = protected_mask[idx_arr]

        need = int(max(keep_counts.get(t_val, len(sub)), int(sub_protected.sum())))
        need = min(need, len(sub))
        if need >= len(sub):
            keep_mask[idx_arr] = True
            continue

        hashes = sub["__keep_hash__"].to_numpy()
        order = np.argsort(hashes, kind="mergesort")

        chosen = np.zeros(len(sub), dtype=bool)
        if sub_protected.any():
            chosen[sub_protected] = True
        remaining = [i for i in order if not sub_protected[i]]
        if need > chosen.sum():
            chosen[remaining[: max(0, need - int(chosen.sum()))]] = True

        keep_mask[idx_arr] = chosen

    return keep_mask

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Throttle per-hour by keeping the top quantile (optionally only among alert rows)."
    )
    ap.add_argument("--alerts", required=False, default=None,
                    help="Alerts file: CSV/CSV.GZ/Parquet; needs time, lat, lon, flag/score cols")
    ap.add_argument("--out", required=False, default=None, help="Output file: CSV/CSV.GZ/Parquet")

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
    ap.add_argument("--prob-col", default=None, help="Legacy probability/score column (deprecated; use --score-col)")
    ap.add_argument("--score-col", default="prob_viable",
                    help="Score column for ranking (default: 'prob_viable'; falls back to --prob-col if not present).")
    ap.add_argument("--flag-col", default="alert_base",
                    help="Binary alert flag column; auto-detects among ['alert','alert_throttled','alert_rule'] if omitted.")
    ap.add_argument("--only-alerts", action="store_true",
                    help="Throttle only among rows where <flag-col> == 1")

    # Geo & debug
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180",
                    type=_strip_choice,
                    help="Normalize longitudes before processing (default: -180..180)")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization')
    ap.add_argument("--sparse-output", action="store_true", help="Write only rows kept after throttling.")
    ap.add_argument("--debug", action="store_true", help="Print ranges and hourly counts")
    ap.add_argument("--run-name", default=None, help="Optional run name for default inputs/outputs.")
    argv = _preprocess_norm(sys.argv[1:])
    args = ap.parse_args(argv)

    if args.alerts is None:
        if args.run_name:
            args.alerts = f"results/alerts/alerts_{args.run_name}_base.parquet"
        else:
            raise SystemExit("--alerts is required (or provide --run-name for defaults).")
    if args.out is None:
        args.out = (
            f"results/alerts/alerts_{args.run_name}_thr.parquet"
            if args.run_name else "results/alerts/alerts_thr.parquet"
        )

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

    # Parse time -> hour bins
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

    # --- resolve flag column ---
    if args.flag_col is not None:
        base_flag_col = args.flag_col
    else:
        # auto-detect common names
        for cand in ["alert", "alert_throttled", "alert_rule"]:
            if cand in df.columns:
                base_flag_col = cand
                break
        else:
            base_flag_col = "alert"  # may be synthesized

    if base_flag_col not in df.columns:
        df[base_flag_col] = 0

    # --- score column (optional but preferred) ---
    score_col = args.score_col
    if score_col not in df.columns and args.prob_col is not None:
        score_col = args.prob_col

    has_score = (score_col is not None) and (score_col in df.columns)
    if has_score:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        print(f"[THROTTLE] warning: score column '{score_col}' not found; throttling will use random ordering per hour.")

    # Flag handling & selection universe
    if args.only_alerts:
        elig = df.loc[df[base_flag_col].astype(int) == 1].copy()
        if elig.empty:
            out = df.copy()
            out[f"{base_flag_col}_base"] = out[base_flag_col].astype(int)
            out[base_flag_col] = 0
            out = out if not args.sparse_output else out.loc[out[base_flag_col] == 1]
            write_any(args.out, out)
            print(f"Wrote {args.out} | throttled: 0/{len(df):,} (no eligible alerts)")
            return
    else:
        elig = df.copy()

    # Ordering: prefer score desc; otherwise, per-hour random to avoid lat bias
    if has_score:
        lat_i = (elig["lat"].to_numpy() * 10000).round().astype(np.int64)
        lon_i = (elig["lon"].to_numpy() * 10000).round().astype(np.int64)
        elig["__tiebreak__"] = lat_i * 1_000_000 + lon_i
        s = elig[score_col].astype(float).fillna(-np.inf)
        elig["__negscore__"] = -s  # descending score via ascending sort
        sort_cols = ["time_h", "__negscore__", "__tiebreak__"]
    else:
        rng = np.random.default_rng(42)
        elig["__rand__"] = rng.random(len(elig))
        sort_cols = ["time_h", "__rand__"]

    # Pre-select “protected” rows by score threshold (survival guard)
    protected_idx = pd.Index([])
    if has_score and args.protect_score_threshold is not None:
        protected_idx = elig.index[elig[score_col] >= float(args.protect_score_threshold)]

    # Sort once for stable per-hour grouping; selection itself is hash-based
    elig = elig.sort_values(sort_cols, ascending=True, kind="mergesort")

    # Per-hour keep targets
    keep_counts: dict = {}
    for t_val, count in elig.groupby("time_h", sort=False)["time_h"].size().items():
        base = math.ceil(float(args.keep_quantile) * int(count))
        base = max(base, int(max(0, args.min_keep_per_hour)))
        if args.only_alerts and float(args.keep_frac_of_alerts) > 0:
            base = max(base, math.ceil(float(args.keep_frac_of_alerts) * int(count)))
        keep_counts[t_val] = min(int(base), int(count))

    # Order-invariant selection via stable hash; always include protected rows
    keep_mask = _deterministic_keep_mask(
        elig=elig,
        keep_counts=keep_counts,
        keep_quantile=float(args.keep_quantile),
        protected_idx=protected_idx,
    )
    kept_idx = elig.index[keep_mask | elig.index.isin(protected_idx)]

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
        print(f"[THROTTLE][debug] time range: {tmin} -> {tmax} (hours={hours})", flush=True)
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
    out.drop(columns=[c for c in ["__tiebreak__","__negscore__","__rand__","__keep_hash__","time_h"] if c in out],
             inplace=True, errors="ignore")
    write_any(args.out, out)
    print(f"Wrote {args.out} | throttled: {kept:,}/{total:,}", flush=True)

if __name__ == "__main__":
    main()
