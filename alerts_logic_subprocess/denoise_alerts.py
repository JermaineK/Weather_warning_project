#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
denoise_alerts.py — Spatial + temporal denoiser for grid alerts.
Keeps time/lat/lon intact; no spatial averaging or temporal collapsing.
Prefers integer grid indices (ilat/ilon) if present for neighbor logic.

Patched version:
  • Optional --score-col (default 'risk') is passed through to output if present.
  • Optional --extra-cols to passthrough arbitrary extra fields.
  • Auto-preserves t_to_storm if present.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def read_any(path, usecols=None):
    low = str(path).lower()
    if low.endswith((".parquet",".parq",".pq")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols if usecols else None)

def write_any(path, df):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if (low.endswith(".csv.gz") or p.suffix.lower()==".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

def to_utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def modal_step(vals: np.ndarray) -> float:
    vals = np.asarray(vals)
    vals = np.unique(vals[np.isfinite(vals)])
    if vals.size < 2: return np.nan
    dif = np.diff(np.sort(vals))
    dif = dif[dif > 0]
    return float(np.median(dif)) if dif.size else np.nan

def spatial_denoise_hour(g: pd.DataFrame, flag_col: str, conn: int, min_neighbors: int) -> np.ndarray:
    """Count active neighbors with 4/8-connectivity. No averaging, no smoothing beyond the binary test."""
    if g.empty: return np.zeros(0, dtype=np.int8)

    act = pd.to_numeric(g[flag_col], errors="coerce").fillna(0).astype(int).to_numpy()
    if act.sum() == 0:
        return np.zeros_like(act, dtype=np.int8)

    # Prefer index-based neighbor logic if ilat/ilon exist
    if {"ilat","ilon"}.issubset(g.columns):
        il = pd.to_numeric(g["ilat"], errors="coerce").to_numpy()
        jl = pd.to_numeric(g["ilon"], errors="coerce").to_numpy()
        # Build hash of active cells
        buckets = {}
        for i,(ii,jj,a) in enumerate(zip(il,jl,act)):
            if a != 1: continue
            buckets.setdefault((int(ii),int(jj)), []).append(i)
        if conn == 4:
            offs = [(+1,0),(-1,0),(0,+1),(0,-1)]
        else:
            offs = [(+1,0),(-1,0),(0,+1),(0,-1),(+1,+1),(+1,-1),(-1,+1),(-1,-1)]
        keep = np.zeros_like(act, dtype=np.int8)
        for i,a in enumerate(act):
            if a != 1: continue
            ii,jj = int(il[i]), int(jl[i])
            hits = 0
            for di,dj in offs:
                hits += len(buckets.get((ii+di, jj+dj), ()))
                if hits >= min_neighbors: break
            keep[i] = 1 if hits >= min_neighbors else 0
        return keep

    # Fallback: coordinate-step neighbor logic
    lat = pd.to_numeric(g["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(g["lon"], errors="coerce").to_numpy()
    dlat = modal_step(lat); dlon = modal_step(lon)
    if not np.isfinite(dlat) or not np.isfinite(dlon) or dlat == 0 or dlon == 0:
        return act.astype(np.int8)

    if conn == 4:
        offsets = [(+dlat,0.0),(-dlat,0.0),(0.0,+dlon),(0.0,-dlon)]
    else:
        offsets = [(+dlat,0.0),(-dlat,0.0),(0.0,+dlon),(0.0,-dlon),
                   (+dlat,+dlon),(+dlat,-dlon),(-dlat,+dlon),(-dlat,-dlon)]
    def key(la,lo): return (round(float(la),10), round(float(lo),10))
    buckets = {}
    for i,(la,lo,a) in enumerate(zip(lat,lon,act)):
        if a != 1: continue
        buckets.setdefault(key(la,lo), []).append(i)
    keep = np.zeros_like(act, dtype=np.int8)
    for i,a in enumerate(act):
        if a != 1: continue
        la0,lo0 = lat[i], lon[i]
        hits = 0
        for dla,dlo in offsets:
            hits += len(buckets.get(key(la0+dla, lo0+dlo), ()))
            if hits >= min_neighbors: break
        keep[i] = 1 if hits >= min_neighbors else 0
    return keep

def temporal_persist(df: pd.DataFrame, flag_col: str, persist_hours: int) -> pd.Series:
    if persist_hours <= 1:
        return df[flag_col].astype(int)
    # group by integer grid if present to avoid FP jitter
    keys = ["ilat","ilon"] if {"ilat","ilon"}.issubset(df.columns) else ["lat","lon"]
    g = df.sort_values(keys+["time"]).groupby(keys, sort=False)
    def _ffill_limit(s):
        return s.replace({0: np.nan}).ffill(limit=persist_hours-1).fillna(0).astype(int)
    return g[flag_col].transform(_ffill_limit)

def parse_args():
    ap = argparse.ArgumentParser(description="Denoise alerts (spatial neighbors + temporal persistence).")
    ap.add_argument("--alerts", required=False, default=None)
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--flag-col", default="alert_base",
                    help="Binary alert flag column to denoise (default: alert_base; falls back to 'alert' if missing).")
    ap.add_argument("--flag-out", default="alert_final",
                    help="Output flag column name (default: alert_final; if empty, overwrite flag-col).")
    ap.add_argument("--persist-hours", type=int, default=3)
    ap.add_argument("--min-neighbors", type=int, default=3)
    ap.add_argument("--connectivity", type=int, choices=[4,8], default=4)
    ap.add_argument("--min-area", type=int, default=0)  # reserved
    ap.add_argument("--sparse-output", dest="sparse_output", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--score-col", default="prob_viable",
                    help="Optional score/probability column to passthrough (default: prob_viable; ignored if absent).")
    ap.add_argument("--extra-cols", default="",
                    help="Comma-separated list of extra columns to passthrough if present.")
    ap.add_argument("--run-name", default=None, help="Optional run name for default inputs/outputs.")
    argv = []
    skip = False
    # preprocess normalize-lon for consistency with other scripts (accept leading-space token forms)
    raw = sys.argv[1:]
    for i, tok in enumerate(raw):
        if skip:
            skip = False
            continue
        if tok == "--normalize-lon" and i + 1 < len(raw):
            argv.append(f"--normalize-lon={raw[i+1]}")
            skip = True
        else:
            argv.append(tok)
    return ap.parse_args(argv)

def main():
    args = parse_args()
    if args.alerts is None:
        if args.run_name:
            args.alerts = f"results/alerts/alerts_{args.run_name}_thr.parquet"
        else:
            raise SystemExit("--alerts is required (or provide --run-name for defaults).")
    if args.out is None:
        args.out = (
            f"results/alerts/alerts_{args.run_name}_final.parquet"
            if args.run_name else "results/alerts/alerts_final.parquet"
        )
    if Path(args.out).exists() and not args.overwrite:
        print(f"[skip] exists: {args.out}")
        return

    df = read_any(args.alerts)

    # Resolve flag column (explicit or fallback to 'alert')
    if args.flag_col in df.columns:
        flag_col = args.flag_col
    elif "alert" in df.columns:
        flag_col = "alert"
    else:
        raise ValueError(f"Need a flag column '{args.flag_col}' or 'alert'.")

    flag_out = args.flag_out if args.flag_out else flag_col

    # keep original geometry; normalize types and floor to hour
    df = df.copy()
    for c in ("time","lat","lon"):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {args.alerts}")
    df["time"] = to_utc_naive(df["time"]).dt.floor("h")
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"], errors="coerce")
    df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    keep_list = []
    for t, g in df.groupby("time", sort=True, as_index=False):
        kept = spatial_denoise_hour(g, flag_col, args.connectivity, args.min_neighbors)
        keep_list.append(pd.Series(kept, index=g.index))
    keep_mask = pd.concat(keep_list).sort_index()
    df["_kept"] = (df[flag_col].astype(int) & keep_mask.astype(int)).astype(int)

    # temporal persistence on the denoised flag
    df["_kept"] = temporal_persist(df.assign(**{flag_col: df["_kept"]}),
                                   flag_col="_kept", persist_hours=args.persist_hours)

    # ---- build output frame ----
    out = df[["time","lat","lon"]].copy()

    # Optional grid indices
    if "ilat" in df.columns and "ilon" in df.columns:
        out[["ilat","ilon"]] = df[["ilat","ilon"]]

    # Score/prob column passthrough
    score_col = args.score_col
    if score_col in df.columns:
        out[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    elif score_col != "prob" and "prob" in df.columns:
        # legacy fallback if user set score-col to default 'risk' but only 'prob' exists
        out["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    elif score_col == "prob" and "prob" in df.columns:
        out["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    # Always keep t_to_storm if present
    if "t_to_storm" in df.columns:
        out["t_to_storm"] = pd.to_numeric(df["t_to_storm"], errors="coerce")

    # Extra columns passthrough (if present)
    extra_cols = [c.strip() for c in args.extra_cols.split(",") if c.strip()]
    for c in extra_cols:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]

    # Final flag(s)
    out[flag_out] = df["_kept"].astype(int)
    if flag_out != flag_col and flag_col not in out.columns:
        out[flag_col] = df["_kept"].astype(int)

    if args.sparse_output:
        out = out.loc[out[flag_out] == 1].reset_index(drop=True)

    write_any(args.out, out)
    print(f"[denoise] wrote {len(out):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
