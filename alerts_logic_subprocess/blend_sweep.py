#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve
)

# ---------- helpers (index-aligned, safe) ----------

def impute_then_scale(df: pd.DataFrame, feats, scaler):
    X = df[feats].copy()
    # coerce to float, kill infs → NaN, then median-impute
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return scaler.transform(X.to_numpy(float))

def time_rolling_max_per_point(df: pd.DataFrame, col: str, hours: int) -> pd.Series:
    """
    Past-window rolling max per (lat,lon), EXCLUDING current time.
    Returns Series aligned to df.index.
    """
    win = f"{int(hours)}h"
    def _roll(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        s = g.set_index("time")[col]
        # past window then shift(1) to exclude current
        r = s.rolling(win, min_periods=1).max().shift(1)
        return pd.Series(r.reindex(g["time"]).to_numpy(), index=g.index)
    return (df.groupby(["lat","lon"], sort=False, group_keys=False)
              .apply(_roll)
              .reindex(df.index))

def future_max_label(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    """Strict future window (t, t+H] max per (lat,lon). Series aligned to df.index."""
    win = f"{int(hours)}h"
    def _lead(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        y = g.set_index("time")[label_col].astype(int)
        rev = y.iloc[::-1]
        fut = rev.rolling(win, min_periods=1).max().shift(1)  # exclude current time
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        return pd.Series(fut.to_numpy(), index=g.index)
    return (df.groupby(["lat","lon"], sort=False, group_keys=False)
              .apply(_lead)
              .reindex(df.index))

def t_to_storm_label(df: pd.DataFrame, lead_h: int) -> np.ndarray:
    """
    Label = 1 iff 0 < t_to_storm_min_h <= lead_h.
    Assumes t_to_storm_min_h is strict-future min time to storm in hours.
    """
    tts = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce").fillna(np.inf).to_numpy()
    return ((tts > 0) & (tts <= float(lead_h))).astype(int)

def apply_persistence(df: pd.DataFrame, flag_col: str, persist_h: int) -> np.ndarray:
    """Require continuous presence for persist_h hours (0 => no persistence)."""
    if persist_h <= 0:
        return df[flag_col].astype(int).to_numpy()
    win = f"{int(persist_h)}h"
    def _persist(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        s = g.set_index("time")[flag_col].astype(int)
        r = s.rolling(win, min_periods=persist_h).mean()
        out = (r.reindex(g["time"]).fillna(0).to_numpy() >= 1.0).astype(int)
        return pd.Series(out, index=g.index)
    return (df.groupby(["lat","lon"], sort=False, group_keys=False)
              .apply(_persist)
              .reindex(df.index)
              .to_numpy())

def throttle_by_hour_quantile(df: pd.DataFrame, prob_col: str, base_flag: str, q: float) -> np.ndarray:
    """
    Keep only rows with prob >= hourly q-quantile among rows where base_flag==1.
    If an hour has no base_flag==1, keep none for that hour.
    Returns 0/1 array aligned to df.index.
    """
    def _keep(g: pd.DataFrame) -> pd.Series:
        sub = g[g[base_flag] == 1]
        if len(sub) == 0:
            return pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        thr = sub[prob_col].quantile(q)
        keep_sub = (sub[prob_col] >= thr).astype(int)
        out = pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        out.loc[keep_sub.index] = keep_sub.to_numpy()
        return out
    return (df.groupby(df["time"].dt.floor("H"), sort=False, group_keys=False)
              .apply(_keep)
              .reindex(df.index)
              .to_numpy())

def derive_metrics(y_true, p):
    # prob threshold by best-F1 (diagnostic only, not used to gate)
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = (2*pr*rc)/(pr+rc+1e-9)
    i = int(np.nanargmax(f1))
    thr = th[max(i-1, 0)] if len(th) else 0.5
    return {
        "AUC": float(roc_auc_score(y_true, p)) if y_true.sum() not in (0, len(y_true)) else float("nan"),
        "PRAUC": float(average_precision_score(y_true, p)),
        "Brier": float(brier_score_loss(y_true, p)),
        "F1": float(f1[i]),
        "Precision": float(pr[i]),
        "Recall": float(rc[i]),
        "thr_diag": float(thr)
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Soft blend sweep (build × relax)")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)

    ap.add_argument("--target", default="pregen",
                    help="Binary target column for label-mode=future_window.")
    ap.add_argument("--label-mode", choices=["future_window","t_to_storm"],
                    default="t_to_storm",
                    help=("Label construction:\n"
                          "  future_window: strict future (t,t+H] from --target column\n"
                          "  t_to_storm   : 0 < t_to_storm_min_h <= lead"))

    ap.add_argument("--alphas", default="0.5", help="comma list e.g. 0.3,0.5,0.7")
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--leads", default="24,48")
    ap.add_argument("--thr-grid", default="0.05:0.09:0.01", help="start:stop:step")
    ap.add_argument("--persist", default="0,1", help="comma list of hours")
    ap.add_argument("--quantiles", default="0.90", help="comma list (e.g. 0.85,0.90)")
    ap.add_argument("--subsample-hours", type=float, default=0.30, help="fraction of distinct hours")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load data (+ normalize time for rolling ops)
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    need = {"time","lat","lon"}
    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise SystemExit(f"Missing base columns: {missing}")
    df = df.dropna(subset=["time","lat","lon"]).sort_values(
        ["lat","lon","time"], kind="mergesort"
    ).reset_index(drop=True)

    if args.label_mode == "future_window":
        if args.target not in df.columns:
            raise SystemExit(f"Label-mode=future_window but target column '{args.target}' not found.")
    else:
        if "t_to_storm_min_h" not in df.columns:
            raise SystemExit("Label-mode=t_to_storm but column 't_to_storm_min_h' not found.")

    print("== Soft Blend Sweep ==")
    print(f"File       : {args.labelled}")
    print(f"Build model: {args.build}")
    print(f"Relax model: {args.relax}")
    print(f"Label-mode : {args.label_mode}")
    print(f"Target     : {args.target} (used only for label-mode=future_window)")

    alphas = [float(x) for x in args.alphas.split(",") if x]
    leads  = [int(x) for x in args.leads.split(",") if x]
    t0, t1, ts = [float(x) for x in args.thr_grid.split(":")]
    thr_values = np.round(np.arange(t0, t1 + 1e-12, ts), 3).tolist()
    persist_list = [int(x) for x in args.persist.split(",") if x]
    qlist = [float(x) for x in args.quantiles.split(",") if x]
    print(f"Alphas        : {alphas}")
    print(f"Build window  : {args.build_window}h")
    print(f"Leads         : {leads}")
    print(f"Thresh grid   : {thr_values[0:3]} … {thr_values[-3:]} (n={len(thr_values)})")
    print(f"Persist (h)   : {persist_list}")
    print(f"Quantiles     : {qlist}")
    print(f"Subsample hrs : {args.subsample_hours}")

    # Optional hour subsample
    if 0 < args.subsample_hours < 1.0:
        hours = df["time"].dt.floor("H").drop_duplicates().sort_values()
        k = max(1, int(len(hours) * args.subsample_hours))
        keep_hours = set(hours.sample(n=k, random_state=42))
        df = df[df["time"].dt.floor("H").isin(keep_hours)].copy()
    print(f"Rows → {len(df):,}")

    # Load models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)
    feats_b, sc_b, clf_b = mb["features"], mb["scaler"], mb["model"]
    feats_r, sc_r, clf_r = mr["features"], mr["scaler"], mr["model"]

    # Score both models (median-impute + Inf cleanup)
    print("  ⋅ Scoring build & relax models…", flush=True)
    Xb = impute_then_scale(df, feats_b, sc_b)
    Xr = impute_then_scale(df, feats_r, sc_r)
    df["p_build"] = clf_b.predict_proba(Xb)[:, 1]
    df["p_relax"] = clf_r.predict_proba(Xr)[:, 1]

    # Build recent window (past args.build_window hours, excluding current)
    print(f"  ⋅ Computing recent build window ({args.build_window}h, excl current)…", flush=True)
    df["build_recent"] = time_rolling_max_per_point(df, "p_build", hours=args.build_window)

    # Precompute labels for each lead
    labels_by_lead = {}
    for h in leads:
        print(f"  ⋅ Preparing labels for lead +{h}h …", flush=True)
        if args.label_mode == "future_window":
            # strict future (t, t+H] from binary target
            df_target = df[["time","lat","lon", args.target]].copy()
            df_target[args.target] = (
                pd.to_numeric(df_target[args.target], errors="coerce")
                  .fillna(0)
                  .astype(int)
            )
            labels_by_lead[h] = future_max_label(df_target, args.target, hours=h).to_numpy()
        else:
            labels_by_lead[h] = t_to_storm_label(df, h)

    total = len(alphas) * len(leads) * len(thr_values) * len(persist_list) * len(qlist)
    step = 0
    out_rows = []

    for alpha in alphas:
        # Soft blend
        df["p_blend"] = alpha * df["build_recent"].to_numpy() + (1.0 - alpha) * df["p_relax"].to_numpy()

        for lead in leads:
            y = labels_by_lead[lead]

            for thr in thr_values:
                df["alert"] = (df["p_blend"] >= thr).astype(int)

                for ph in persist_list:
                    df["alert_persist"] = apply_persistence(df, "alert", persist_h=ph)

                    for q in qlist:
                        df["alert_final"] = throttle_by_hour_quantile(df, "p_blend", "alert_persist", q)

                        # Evaluate (diagnostics on probabilities; coverage on final mask)
                        mask = df["alert_final"].to_numpy().astype(bool)
                        p = df["p_blend"].to_numpy()
                        metrics = derive_metrics(y, p)

                        out_rows.append({
                            "alpha": float(alpha),
                            "lead_h": int(lead),
                            "thr": float(thr),
                            "persist_h": int(ph),
                            "quantile": float(q),
                            "AUC": metrics["AUC"],
                            "PRAUC": metrics["PRAUC"],
                            "Brier": metrics["Brier"],
                            "F1": metrics["F1"],
                            "Precision": metrics["Precision"],
                            "Recall": metrics["Recall"],
                            "Coverage": float(mask.mean())
                        })

                        step += 1
                        if step % 25 == 0 or step == total:
                            print(f"  … {step}/{total} combinations", flush=True)

    out = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(out)}")

if __name__ == "__main__":
    main()