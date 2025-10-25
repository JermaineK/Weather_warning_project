#!/usr/bin/env python3
"""
per_hour_skill.py — robust per-hour coverage & F1 for a given lead.

Reads labelled+alerts (CSV/CSV.GZ/Parquet), tolerates messy CSVs (fallback to python engine),
and gives a tidy CSV: [issue_hour, coverage, f1, n].
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from typing import Optional

CANDIDATE_TIME_COLS = ["time", "time_h", "datetime", "valid_time", "forecast_time"]

# ---------------- I/O ----------------

def read_any(path: str, usecols: Optional[list]=None) -> pd.DataFrame:
    p = str(path)
    low = p.lower()

    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} (not found)")

    # Parquet fast path
    if low.endswith(".parquet") or low.endswith(".pq"):
        return pd.read_parquet(p, columns=usecols)

    # If the file is truly empty (0 bytes), bail with a friendly explanation.
    if os.path.getsize(p) == 0:
        raise RuntimeError(f"{p} is empty (0 bytes). Was the previous step interrupted?")

    # CSV/GZ — try a sequence of increasingly tolerant options
    attempts = [
        dict(engine="c", compression="infer"),
        dict(engine="python", compression="infer"),
        dict(engine="python", compression=None),
    ]
    last_err = None
    for opts in attempts:
        try:
            return pd.read_csv(
                p,
                usecols=usecols,
                **opts,
                on_bad_lines="warn" if opts["engine"] == "python" else "error"
            )
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read {p} via robust CSV path. Last error: {last_err}")

def pick_time_col(df: pd.DataFrame, preferred: str="time") -> str:
    if preferred in df.columns: return preferred
    for c in CANDIDATE_TIME_COLS:
        if c in df.columns: return c
    raise ValueError(f"No time-like column among {CANDIDATE_TIME_COLS} in {list(df.columns)}")

def to_utc_naive(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (mid and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

# ---------------- labels & metrics ----------------

def future_max_label_by_point(df: pd.DataFrame, target_col: str, hours: int) -> np.ndarray:
    def _per_point(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["time"]).sort_values("time", kind="mergesort")
        s = pd.to_numeric(g[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        sr = s[::-1]
        r  = pd.Series(sr).rolling(window=int(hours), min_periods=1).max()
        win = r.shift(1).fillna(0).astype(int).to_numpy()[::-1]
        return pd.Series(win, index=g.index, name="y_future")
    out = (
        df[["time","lat","lon",target_col]]
          .groupby(["lat","lon"], sort=False, group_keys=False)
          .apply(_per_point)
    )
    return out.reindex(df.index).fillna(0).to_numpy(dtype=int)

def f1_from_flags(y_true: np.ndarray, flags: np.ndarray) -> float:
    tp = int(((flags==1) & (y_true==1)).sum())
    fp = int(((flags==1) & (y_true==0)).sum())
    fn = int(((flags==0) & (y_true==1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return (2*prec*rec / (prec + rec)) if (prec + rec) else 0.0

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Per-hour coverage & F1 for a given lead.")
    ap.add_argument("--labelled", required=True, help="Labelled grid CSV/Parquet.")
    ap.add_argument("--alerts",   required=True, help="Alerts CSV(.gz)/Parquet.")
    ap.add_argument("--lead-hours", type=int, required=True, help="Lead window (hours).")
    ap.add_argument("--target", default="pregen", help="Target column in labelled (default: pregen).")
    ap.add_argument("--flag-col", default="alert_final",
                    help="Alerts flag column (default: alert_final). Fallbacks: alert_throttled, alert.")
    ap.add_argument("--time-col", default="time", help="Time column name if present.")
    ap.add_argument("--out", default=None, help="Output CSV (default results/per_hour/per_hour_skill_lead{lead}.csv)")
    args = ap.parse_args()

    # Read inputs
    base = read_any(args.labelled).replace([np.inf,-np.inf], np.nan)
    alrt = read_any(args.alerts).replace([np.inf,-np.inf], np.nan)

    # Time columns
    tcol_base = args.time_col if args.time_col in base.columns else pick_time_col(base, args.time_col)
    tcol_alrt = args.time_col if args.time_col in alrt.columns else pick_time_col(alrt, args.time_col)

    for c in [tcol_base, "lat", "lon", args.target]:
        if c not in base.columns:
            sys.exit(f"[ERROR] labelled missing column: {c}")

    fcol = args.flag_col
    if fcol not in alrt.columns:
        for alt in ["alert_throttled", "alert_final", "alert"]:
            if alt in alrt.columns:
                fcol = alt; break
    if fcol not in alrt.columns:
        sys.exit(f"[ERROR] alerts missing flag column: {args.flag_col} (also tried alert_throttled, alert_final, alert)")
    for c in [tcol_alrt, "lat", "lon", fcol]:
        if c not in alrt.columns:
            sys.exit(f"[ERROR] alerts missing column: {c}")

    # Parse/clean
    base[tcol_base] = to_utc_naive(base[tcol_base])
    alrt[tcol_alrt] = to_utc_naive(alrt[tcol_alrt])
    base = base.dropna(subset=[tcol_base, "lat", "lon", args.target]).reset_index(drop=True)
    alrt = alrt.dropna(subset=[tcol_alrt, "lat", "lon", fcol]).reset_index(drop=True)
    for c in ["lat","lon"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        alrt[c] = pd.to_numeric(alrt[c], errors="coerce")
    alrt[fcol] = pd.to_numeric(alrt[fcol], errors="coerce").fillna(0).astype(int)

    base["time_h"] = base[tcol_base].dt.floor("h")
    alrt["time_h"] = alrt[tcol_alrt].dt.floor("h")

    # Merge alerts into base grid on (time_h, lat, lon)
    df = base[["time_h", "time", "lat", "lon", args.target]].merge(
        alrt[["time_h", "lat", "lon", fcol]],
        on=["time_h", "lat", "lon"],
        how="left",
        validate="many_to_one"
    )
    df[fcol] = pd.to_numeric(df[fcol], errors="coerce").fillna(0).astype(int)

    # Build y at +lead hours
    print(f"[per-hour] preparing future labels for +{args.lead_hours}h …")
    y_future = future_max_label_by_point(
        df.rename(columns={"time":"time"})[["time","lat","lon",args.target]],
        args.target, int(args.lead_hours)
    )
    df["y_future"] = y_future.astype(int)

    # Per-hour metrics
    rows = []
    for t, g in df.groupby("time_h", sort=True):
        flags = g[fcol].to_numpy(dtype=int)
        y     = g["y_future"].to_numpy(dtype=int)
        cov   = float(flags.mean()) if len(flags) else 0.0
        f1    = f1_from_flags(y, flags)
        rows.append(dict(issue_hour=t, coverage=cov, f1=f1, n=len(g)))

    out = pd.DataFrame(rows).sort_values("issue_hour").reset_index(drop=True)
    out_path = args.out or os.path.join("results", "per_hour", f"per_hour_skill_lead{int(args.lead_hours)}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[per-hour] wrote {out_path}  rows={len(out)}  "
          f"time: {out['issue_hour'].min()} → {out['issue_hour'].max()}")

if __name__ == "__main__":
    main()