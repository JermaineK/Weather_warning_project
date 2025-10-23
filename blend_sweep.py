#!/usr/bin/env python
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve
)

# ---------- helpers (index-aligned, safe) ----------

def impute_then_scale(df: pd.DataFrame, feats, scaler):
    X = df[feats].copy()
    for c in feats:
        if X[c].isna().any():
            X[c] = X[c].fillna(np.nanmedian(X[c].to_numpy()))
    return scaler.transform(X.to_numpy(float))

def time_rolling_max_per_point(df: pd.DataFrame, col: str, hours: int) -> pd.Series:
    """Past-window rolling max per (lat,lon). Returns Series aligned to df.index."""
    def _roll(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("time")[col]
        r = s.rolling(f"{hours}h", min_periods=1).max()
        return pd.Series(r.reindex(g["time"]).to_numpy(), index=g.index)
    return df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_roll)

def future_max_label(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    """Future-window (look-ahead) max per (lat,lon). Series aligned to df.index."""
    def _lead(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("time")[label_col].astype(int)
        fwd = s.rolling(f"{hours}h", min_periods=1).max().shift(-1)
        return pd.Series(fwd.reindex(g["time"]).fillna(0).to_numpy(dtype=int), index=g.index)
    return df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)

def apply_persistence(df: pd.DataFrame, flag_col: str, persist_h: int) -> np.ndarray:
    """Require continuous presence for persist_h hours (0 => no persistence)."""
    if persist_h <= 0:
        return df[flag_col].astype(int).to_numpy()
    def _persist(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("time")[flag_col].astype(int)
        # mean==1.0 over a time window implies all ones in the window
        r = s.rolling(f"{persist_h}h", min_periods=persist_h).mean()
        out = (r.reindex(g["time"]).fillna(0).to_numpy() >= 1.0).astype(int)
        return pd.Series(out, index=g.index)
    return df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_persist).to_numpy()

def throttle_by_hour_quantile(df: pd.DataFrame, prob_col: str, base_flag: str, q: float) -> np.ndarray:
    """
    Keep only rows with prob >= hourly q-quantile among rows where base_flag==1.
    If an hour has no base_flag==1, keep none for that hour.
    """
    def _keep(g: pd.DataFrame) -> pd.Series:
        sub = g[g[base_flag] == 1]
        if len(sub) == 0:
            return pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        thr = sub[prob_col].quantile(q)
        return pd.Series((g[prob_col].to_numpy() >= thr).astype(int), index=g.index)
    return df.groupby("time", sort=False, group_keys=False).apply(_keep).to_numpy()

def derive_metrics(y_true, p):
    # prob threshold by best-F1 (diagnostic only)
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = (2*pr*rc)/(pr+rc+1e-9)
    i = int(np.nanargmax(f1))
    thr = th[max(i-1, 0)] if len(th) else 0.5
    yhat = (p >= thr).astype(int)
    return {
        "AUC": roc_auc_score(y_true, p),
        "PRAUC": average_precision_score(y_true, p),
        "Brier": brier_score_loss(y_true, p),
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
    ap.add_argument("--target", default="pregen", choices=["storm","near_storm","pregen"])
    ap.add_argument("--alphas", default="0.5", help="comma list e.g. 0.3,0.5,0.7")
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--leads", default="24,48")
    ap.add_argument("--thr-grid", default="0.05:0.09:0.01", help="start:stop:step")
    ap.add_argument("--persist", default="0,1", help="comma list of hours")
    ap.add_argument("--quantiles", default="0.90", help="comma list (e.g. 0.85,0.90)")
    ap.add_argument("--subsample-hours", type=float, default=0.30, help="fraction of distinct hours")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    print("== Soft Blend Sweep ==")
    print(f"File   : {args.labelled}")
    print(f"Build  : {args.build}")
    print(f"Relax  : {args.relax}")
    print(f"Target : {args.target}")
    alphas = [float(x) for x in args.alphas.split(",") if x]
    leads = [int(x) for x in args.leads.split(",") if x]
    t0, t1, ts = [float(x) for x in args.thr_grid.split(":")]
    thr_values = np.round(np.arange(t0, t1+1e-12, ts), 3).tolist()
    persist_list = [int(x) for x in args.persist.split(",") if x]
    qlist = [float(x) for x in args.quantiles.split(",") if x]
    print(f"Alphas : {alphas}")
    print(f"Build window (h): {args.build_window}")
    print(f"Leads  : {leads}")
    print(f"Thresh : {thr_values[0:3]} … {thr_values[-3:]} (n={len(thr_values)})")
    print(f"Persist: {persist_list} Quantiles: {qlist}")
    print(f"Subsample hours: {args.subsample_hours}")

    # Optional hour subsample
    if 0 < args.subsample_hours < 1.0:
        hours = df["time"].drop_duplicates().sort_values()
        k = max(1, int(len(hours) * args.subsample_hours))
        keep_hours = set(hours.sample(n=k, random_state=42))
        df = df[df["time"].isin(keep_hours)].copy()
    print(f"Rows → {len(df):,}  (from {pd.read_csv(args.labelled, usecols=['time']).shape[0]:,})")

    # Load models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)
    feats_b, sc_b, clf_b = mb["features"], mb["scaler"], mb["model"]
    feats_r, sc_r, clf_r = mr["features"], mr["scaler"], mr["model"]

    # Score both models (with simple median impute for any NaNs)
    print("  ⋅ Scoring build & relax models…", flush=True)
    Xb = impute_then_scale(df, feats_b, sc_b)
    Xr = impute_then_scale(df, feats_r, sc_r)
    p_build = clf_b.predict_proba(Xb)[:,1]
    p_relax = clf_r.predict_proba(Xr)[:,1]
    df["p_build"] = p_build
    df["p_relax"] = p_relax

    # Build recent window (past args.build_window hours)
    print(f"  ⋅ Computing recent build window ({args.build_window}h)…", flush=True)
    df["build_recent"] = time_rolling_max_per_point(df, "p_build", hours=args.build_window)

    # Labels for each lead
    labels_by_lead = {}
    for h in leads:
        print(f"  ⋅ Preparing labels for lead +{h}h …", flush=True)
        labels_by_lead[h] = future_max_label(df, args.target, hours=h).to_numpy()

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

                        # Evaluate
                        mask = df["alert_final"].to_numpy().astype(bool)
                        p = df["p_blend"].to_numpy()
                        metrics = derive_metrics(y, p)

                        out_rows.append({
                            "alpha": alpha,
                            "lead_h": lead,
                            "thr": thr,
                            "persist_h": ph,
                            "quantile": q,
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