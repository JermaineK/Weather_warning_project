#!/usr/bin/env python
import argparse, itertools, sys, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_fscore_support
)

def parse_float_list(s):
    s = s.strip()
    if ":" in s:
        a, b, c = [float(x) for x in s.split(":")]
        out, x = [], a
        while x <= b + 1e-12:
            out.append(round(x, 12))
            x += c
        return out
    return [float(x) for x in s.split(",") if x.strip()]

def time_rolling_max_per_point(df, col, hours):
    """
    Per (lat,lon), compute time-based rolling max of `col` with a window of `hours`.
    Returns a 1-D Series aligned to df.index.
    """
    win = f"{int(hours)}h"
    out = pd.Series(index=df.index, dtype=float, name=f"{col}_rollmax")

    for (la, lo), g in df.groupby(["lat", "lon"], sort=False, group_keys=False):
        g_sorted = g.sort_values("time")
        r = pd.Series(g_sorted[col].to_numpy(),
                      index=g_sorted["time"]).rolling(win, min_periods=1).max()
        # write back in the original row order for this group
        out.loc[g_sorted.index] = r.values

    # ensure exact df.index order
    return out.loc[df.index]


def future_rolling_max_per_point(df, target_col, hours):
    """
    Per (lat,lon), strict-future (t+Δ) rolling max of binary `target_col`.
    Returns a 1-D Series aligned to df.index.
    """
    win = f"{int(hours)}h"
    out = pd.Series(index=df.index, dtype=float, name=f"{target_col}_futuremax")

    for (la, lo), g in df.groupby(["lat", "lon"], sort=False, group_keys=False):
        g_sorted = g.sort_values("time")
        # reverse for future-looking, shift(1) so current time not included
        rev = pd.Series(g_sorted[target_col].to_numpy(),
                        index=g_sorted["time"]).iloc[::-1]
        fut = rev.rolling(win, min_periods=1).max().shift(1)
        fut = fut.iloc[::-1].reindex(g_sorted["time"]).fillna(0).astype(int)
        out.loc[g_sorted.index] = fut.values

    return out.loc[df.index].astype(int)# === PATCHED: robust scoring with median imputation & Inf cleanup ===
def load_model_probs(df, model_path):
    m = joblib.load(model_path)
    feats = m["features"]
    # Work on a copy of just the feature frame
    Xdf = df[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    # Count rows with any NaN
    rows_with_nan = int(Xdf.isna().any(axis=1).sum())
    if rows_with_nan:
        # Column-wise median imputation (robust & simple)
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med)
    # Scale & predict
    Xs = m["scaler"].transform(Xdf.to_numpy())
    p = m["model"].predict_proba(Xs)[:, 1]
    # Optional: tiny log to reassure during long runs
    if rows_with_nan:
        print(f"  • {model_path}: imputed NaNs on {rows_with_nan:,} rows", flush=True)
    return p, feats

def throttle_hourly(series, time_col, q):
    if q is None:
        return np.ones(len(series), dtype=bool)
    s = pd.Series(series, index=time_col.index)
    def _keep(group):
        k = int(np.ceil(len(group) * (1 - q)))
        if k <= 0:
            return pd.Series([False]*len(group), index=group.index)
        thr = group.nlargest(k).min()
        return group >= thr
    mask = s.groupby(time_col.dt.floor("h"), sort=False).apply(_keep)
    return mask.reset_index(level=0, drop=True).to_numpy()

def fmt_pct(x): return f"{100*x:.3f}%"

def main():
    ap = argparse.ArgumentParser(description="Sweep build/relax thresholds with progress.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--tb")
    ap.add_argument("--tr")
    ap.add_argument("--leads", required=True)
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--subsample-hours", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    tb_list = parse_float_list(args.tb) if args.tb else [0.05, 0.07, 0.10]
    tr_list = parse_float_list(args.tr) if args.tr else [0.05, 0.07, 0.10]
    leads = [int(x) for x in args.leads.split(",")]

    print("== Gate Sweep (build×relax) ==")
    print(f"File   : {args.labelled}")
    print(f"Build  : {args.build}")
    print(f"Relax  : {args.relax}")
    print(f"Target : {args.target}")
    print(f"Build window: {args.build_window} h")
    print(f"tb     : {tb_list}")
    print(f"tr     : {tr_list}")
    print(f"Leads  : {leads}")
    if args.quantile is not None: print(f"Hourly throttle quantile: {args.quantile}")
    if args.subsample_hours > 0:  print(f"Subsample hours fraction: {args.subsample_hours}")

    df = pd.read_csv(args.labelled, parse_dates=["time"], usecols=lambda c: True)

    if args.subsample_hours and args.subsample_hours > 0:
        hrs = df["time"].dt.floor("h").drop_duplicates()
        keep_hrs = hrs.sample(frac=args.subsample_hours, random_state=42).sort_values()
        df = df[df["time"].dt.floor("h").isin(keep_hrs)].reset_index(drop=True)
        print(f"Subsampled hours → rows: {len(df):,}")

    print("Scoring build model…", flush=True)
    p_build, _ = load_model_probs(df, args.build)
    print("Scoring relax model…", flush=True)
    p_relax, _ = load_model_probs(df, args.relax)
    df["p_build"] = p_build
    df["p_relax"] = p_relax

    print("Computing build_recent window…", flush=True)
    t0 = time.time()
    df["build_recent"] = time_rolling_max_per_point(df, "p_build", hours=args.build_window)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    rows = []
    combos = list(itertools.product(leads, tb_list, tr_list))
    total = len(combos)
    start = time.time()
    last = start

    for i, (lead_h, tb, tr) in enumerate(combos, start=1):
        y = future_rolling_max_per_point(df, args.target, hours=lead_h).fillna(0).astype(int).to_numpy()
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

    out = pd.DataFrame(rows)
    out.sort_values(["lead","F1","Precision","Recall"], ascending=[True,False,False,False], inplace=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(out)}")
    for L in leads:
        best = out[out["lead"]==L].head(1)
        if len(best):
            r = best.iloc[0]
            print(f"[lead {L:>2}h]  F1={r.F1:.3f}  P={r.Precision:.3f}  R={r.Recall:.3f}  "
                  f"Cov={r.Coverage:.3f}  tb={r.tb:.3f}  tr={r.tr:.3f}  "
                  f"AUC={r.AUC:.3f}  PRAUC={r.PRAUC:.3f}  Brier={r.Brier:.3f}")

if __name__ == "__main__":
    main()