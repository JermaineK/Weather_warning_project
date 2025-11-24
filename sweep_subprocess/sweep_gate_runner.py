#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_gate_runner.py — gate (build×relax) threshold sweep with time-aware windows

Gate logic per row:
    gate = (past_max(p_build, build_window_h) >= tb)
    alerts = gate & (p_relax >= tr)
Optionally apply hourly throttling (keep top scores per hour).

Outputs a CSV grid with F1/Precision/Recall/Coverage plus AUC/PRAUC/Brier of the *score*
(where score = p_relax masked by the gate).

Notes:
- Past window uses a time-based rolling per (lat,lon) and excludes the current hour via shift(1).
- Future labels use a strict future window (t, t+lead].
- Features are median-imputed per column and Infs are coerced to NaN before scaling.
"""

import argparse, itertools, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_fscore_support
)

# ---------------- parsing ----------------

def parse_float_list(s: str):
    s = s.strip()
    if ":" in s:
        a, b, c = [float(x) for x in s.split(":")]
        out, x = [], a
        # inclusive range stepping
        while x <= b + 1e-12:
            out.append(round(x, 12))
            x += c
        return out
    return [float(x) for x in s.split(",") if x.strip()]

# ---------------- time-aware windows ----------------

def time_rolling_max_per_point(df: pd.DataFrame, col: str, hours: int) -> pd.Series:
    """
    Per (lat,lon), time-based rolling max of `col` over the past `hours`,
    EXCLUDING the current instant (shift(1)).
    Returns a Series aligned to df.index.
    """
    win = f"{int(hours)}H"
    # Work on a minimal frame
    tmp = df.loc[:, ["time", "lat", "lon", col]].copy()

    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        s = g.set_index("time")[col].rolling(win, min_periods=1).max().shift(1).fillna(0.0)
        s = s.reindex(g["time"])
        s.index = g.index
        return s

    out = tmp.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_one)
    return out.reindex(df.index)

def future_rolling_max_per_point(df: pd.DataFrame, target_col: str, hours: int) -> pd.Series:
    """
    Per (lat,lon), strict-future (t, t+hours] rolling max of binary `target_col`.
    Returns a Series aligned to df.index.
    """
    win = f"{int(hours)}H"
    tmp = df.loc[:, ["time", "lat", "lon", target_col]].copy()

    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        rev = g.set_index("time")[target_col].iloc[::-1]
        fut = rev.rolling(win, min_periods=1).max().shift(1)  # exclude current
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        fut.index = g.index
        return fut

    out = tmp.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_one)
    return out.reindex(df.index).astype(int)

# ---------------- model scoring ----------------

def load_model_probs(df: pd.DataFrame, model_path: str):
    m = joblib.load(model_path)
    feats = list(m["features"])
    Xdf = df[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    rows_with_nan = int(Xdf.isna().any(axis=1).sum())
    if rows_with_nan:
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med)

    Xv = Xdf.to_numpy()
    if m.get("scaler") is not None:
        Xv = m["scaler"].transform(Xv)
    p = m["model"].predict_proba(Xv)[:, 1]
    if rows_with_nan:
        print(f"  • {Path(model_path).name}: imputed NaNs on {rows_with_nan:,} rows", flush=True)
    return p, feats

# ---------------- post-filters ----------------

def throttle_hourly(score_vec: np.ndarray, times: pd.Series, q: float | None) -> np.ndarray:
    """
    Keep only the top (1-q) fraction by score within each hour.
    If q=None, keep all.
    """
    if q is None:
        return np.ones(len(score_vec), dtype=bool)

    s = pd.Series(score_vec, index=times.index)
    def _keep(group):
        if len(group) == 0:
            return pd.Series([], dtype=bool)
        k = int(np.ceil(len(group) * (1 - q)))
        if k <= 0:
            return pd.Series([False] * len(group), index=group.index)
        thr = group.nlargest(k).min()
        return group >= thr

    mask = s.groupby(times.dt.floor("H"), sort=False).apply(_keep)
    return mask.reset_index(level=0, drop=True).to_numpy()

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Sweep build×relax gate thresholds with progress.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--build-window", type=int, default=24, help="Past hours for build max (exclude current)")
    ap.add_argument("--tb", help="Build thresholds (e.g. 0.03:0.10:0.005 or 0.05,0.07,0.10)")
    ap.add_argument("--tr", help="Relax thresholds (e.g. 0.03:0.10:0.005 or 0.05,0.07,0.10)")
    ap.add_argument("--leads", required=True, help="Comma list of lead hours (e.g. 24,48,72)")
    ap.add_argument("--quantile", type=float, default=None, help="Hourly throttle quantile (e.g. 0.90 keeps top 10%)")
    ap.add_argument("--subsample-hours", type=float, default=0.0, help="Fraction of hours to keep for a quick sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    tb_list = parse_float_list(args.tb) if args.tb else [0.05, 0.07, 0.10]
    tr_list = parse_float_list(args.tr) if args.tr else [0.05, 0.07, 0.10]
    leads = [int(x) for x in args.leads.split(",")]

    print("== Gate Sweep (build×relax) ==")
    print(f"File         : {args.labelled}")
    print(f"Build model  : {args.build}")
    print(f"Relax model  : {args.relax}")
    print(f"Target       : {args.target}")
    print(f"Build window : {args.build_window} h")
    print(f"tb           : {tb_list}")
    print(f"tr           : {tr_list}")
    print(f"Leads        : {leads}")
    if args.quantile is not None: print(f"Hourly throttle quantile: {args.quantile}")
    if args.subsample_hours > 0:  print(f"Subsample hours fraction: {args.subsample_hours}")

    # read data
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    # normalize time
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)

    # Optional subsample by hour for quicker sweeps
    if args.subsample_hours and args.subsample_hours > 0:
        hrs = df["time"].dt.floor("H").drop_duplicates().sort_values()
        keep_hrs = hrs.sample(frac=args.subsample_hours, random_state=42)
        df = df[df["time"].dt.floor("H").isin(keep_hrs)].reset_index(drop=True)
        print(f"Subsampled hours → rows: {len(df):,}")

    # score models
    print("Scoring build model…", flush=True)
    p_build, _ = load_model_probs(df, args.build)
    print("Scoring relax model…", flush=True)
    p_relax, _ = load_model_probs(df, args.relax)
    df["p_build"] = p_build
    df["p_relax"] = p_relax

    # compute time-aware past max for build
    print("Computing build_recent window…", flush=True)
    t0 = time.time()
    df["build_recent"] = time_rolling_max_per_point(df, "p_build", hours=args.build_window)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    rows = []
    combos = list(itertools.product(leads, tb_list, tr_list))
    total = len(combos)
    start = time.time()
    last = start

    # pre-compute coincident target as sanity info (we still evaluate leads)
    # (kept here in case you later want a coincident line)
    # y0 = df[args.target].astype(int).to_numpy()

    for i, (lead_h, tb, tr) in enumerate(combos, start=1):
        # strict future window labels
        y = future_rolling_max_per_point(df, args.target, hours=lead_h).to_numpy()

        gate = (df["build_recent"].to_numpy() >= tb)
        score = np.where(gate, df["p_relax"].to_numpy(), 0.0)
        alerts = gate & (df["p_relax"].to_numpy() >= tr)

        if args.quantile is not None:
            keep_mask = throttle_hourly(score, df["time"], q=args.quantile)
            alerts = alerts & keep_mask

        cov = alerts.mean() if len(alerts) else 0.0
        if alerts.any():
            pr, rc, f1, _ = precision_recall_fscore_support(
                y, alerts.astype(int), average="binary", zero_division=0
            )
        else:
            pr = rc = f1 = 0.0

        try:   auc   = roc_auc_score(y, score)
        except ValueError: auc = np.nan
        try:   prauc = average_precision_score(y, score)
        except ValueError: prauc = np.nan
        try:   brier = brier_score_loss(y, score)
        except ValueError: brier = np.nan

        rows.append({
            "lead": lead_h, "tb": tb, "tr": tr,
            "F1": f1, "Precision": pr, "Recall": rc, "Coverage": cov,
            "AUC": auc, "PRAUC": prauc, "Brier": brier
        })

        if args.progress and (time.time() - last >= 0.5 or i == total):
            pct = 100.0 * i / total
            elapsed = time.time() - start
            eta = elapsed * (total / i - 1)
            print(f"Progress: {pct:5.1f}%  ({i}/{total})  ETA {eta:6.1f}s", end="\r", flush=True)
            last = time.time()

    if args.progress: print()

    out = pd.DataFrame(rows).sort_values(
        ["lead","F1","Recall","Precision"], ascending=[True, False, False, False]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(out)}")

    for L in leads:
        best = out[out["lead"] == L].head(1)
        if len(best):
            r = best.iloc[0]
            print(f"[lead {L:>3}h]  F1={r.F1:.3f}  P={r.Precision:.3f}  R={r.Recall:.3f}  "
                  f"Cov={r.Coverage:.3f}  tb={r.tb:.3f}  tr={r.tr:.3f}  "
                  f"AUC={r.AUC:.3f}  PRAUC={r.PRAUC:.3f}  Brier={r.Brier:.3f}")

if __name__ == "__main__":
    main()