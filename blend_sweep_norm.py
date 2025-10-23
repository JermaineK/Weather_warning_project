import argparse, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from scipy.special import expit as sigmoid
from scipy.stats import norm

# ---------- helpers ----------
def load_model(path):
    m = joblib.load(path)
    return m["model"], m["scaler"], m["features"]

def impute_then_scale(df, cols, scaler):
    X = df[cols].to_numpy(float)
    # median impute per column
    med = np.nanmedian(X, axis=0)
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.take(med, np.where(mask)[1])
    return scaler.transform(X)

def score_model(df, model, scaler, feats):
    Xs = impute_then_scale(df, feats, scaler)
    p = model.predict_proba(Xs)[:,1]
    return p

def recent_build_window(df, prob_col, hours):
    # rolling max over the *past* window per (lat,lon), excluding current hour (shift 1)
    def _roll(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)[prob_col].max().shift(1)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_roll)
    # reindex back to the original row order; fill missing with 0
    return out.reindex(df.index).fillna(0.0).to_numpy()

def future_max_label(df, label_col, hours):
    # rolling forward max over the *next* window per (lat,lon), excluding current hour (shift -1)
    def _lead(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)[label_col].max().shift(-1)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)
    return out.reindex(df.index).fillna(0).to_numpy().astype(int)

def apply_persistence(df, flag_col, persist_h):
    if persist_h <= 0:
        return df[flag_col].to_numpy().astype(int)
    def _persist(s):
        r = s.rolling(persist_h, min_periods=persist_h).sum()
        return (r >= persist_h).astype(int)
    out = df.groupby(["lat","lon"], sort=False)[flag_col].transform(_persist)
    return out.to_numpy().astype(int)

def throttle_by_hour_quantile(df, score_col, flag_col, q_keep):
    # Keep only top q_keep of score_col per hour among cells that are already flagged
    tmp = df.copy()
    tmp["_keep"] = 0
    def _keep_top(g):
        if (g[flag_col].sum() == 0):
            return pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        thr = g.loc[g[flag_col] == 1, score_col].quantile(q_keep)
        return (g[score_col] >= thr).astype(int)
    tmp["_keep"] = tmp.groupby("time", sort=False, group_keys=False).apply(_keep_top)
    return (tmp["_keep"] == 1).to_numpy().astype(int)

def metrics(y_true, scores, alerts):
    # scores for AUC/PRAUC/Brier, alerts (0/1) for P/R/F1
    auc  = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
    ap   = average_precision_score(y_true, scores)
    bri  = brier_score_loss(y_true, scores)
    tp   = (alerts & (y_true==1)).sum()
    pp   = alerts.sum()
    pos  = (y_true==1).sum()
    prec = (tp/pp) if pp>0 else 0.0
    rec  = (tp/pos) if pos>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec+1e-9)
    cov  = pp/len(y_true)
    return auc, ap, bri, f1, prec, rec, cov

def write_row(path, row, header=False):
    df = pd.DataFrame([row])
    mode = "w" if header or (not Path(path).exists()) else "a"
    df.to_csv(path, index=False, mode=mode, header=(header or (mode=="w")))

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Normalized probit-blend sweep with hard write-after-each-combo.")
    p.add_argument("--labelled", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--relax", required=True)
    p.add_argument("--target", required=True, choices=["storm","near_storm","pregen"])
    p.add_argument("--alphas", required=True, help="comma list e.g. 0.3,0.5,0.7")
    p.add_argument("--build-window", type=int, default=24)
    p.add_argument("--leads", required=True, help="comma list of hours e.g. 24,48")
    p.add_argument("--thr-grid", required=True, help="start:stop:step in prob space, e.g. 0.05:0.09:0.01")
    p.add_argument("--persist", required=True, help="comma list of hours e.g. 0,1,2")
    p.add_argument("--quantiles", required=True, help="comma list e.g. 0.85,0.90")
    p.add_argument("--subsample-hours", type=float, default=0.0, help="fraction of hours to sample (0..1)")
    p.add_argument("--calibrate", action="store_true", help="(placeholder) keep for API parity; no-op here")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print("== Soft Blend Sweep (norm/probit) ==", flush=True)
    print(f"File   : {args.labelled}", flush=True)
    print(f"Build  : {args.build}", flush=True)
    print(f"Relax  : {args.relax}", flush=True)
    print(f"Target : {args.target}", flush=True)

    alphas   = [float(x) for x in args.alphas.split(",")]
    leads    = [int(x)   for x in args.leads.split(",")]
    persists = [int(x)   for x in args.persist.split(",")]
    quants   = [float(x) for x in args.quantiles.split(",")]

    t0, t1, dt = [float(x) for x in args.thr_grid.split(":")]
    thr_grid = np.round(np.arange(t0, t1+1e-12, dt), 6).tolist()

    # load data
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    baseN = len(df)
    # optional subsample by hours
    if args.subsample_hours and args.subsample_hours>0 and args.subsample_hours<1:
        hours = df["time"].dt.floor("h").drop_duplicates()
        keep_hours = hours.sample(frac=args.subsample_hours, random_state=7).sort_values()
        df = df[df["time"].dt.floor("h").isin(keep_hours)].reset_index(drop=True)
    print(f"Rows  → {len(df):,}  (from {baseN:,})", flush=True)

    # load models & probs
    mdl_b, sc_b, feats_b = load_model(args.build)
    mdl_r, sc_r, feats_r = load_model(args.relax)

    print("Scoring build & relax models…", flush=True)
    p_build = score_model(df, mdl_b, sc_b, feats_b)
    p_relax = score_model(df, mdl_r, sc_r, feats_r)

    # recent build gate probability (max of last window, excluding current hour)
    print(f"Computing recent build window ({args.build_window}h)…", flush=True)
    df["p_build"] = p_build
    build_recent = recent_build_window(df, "p_build", args.build_window)
    df.drop(columns=["p_build"], inplace=True)

    # labels by lead
    labels_by_lead = {}
    for h in leads:
        print(f"Preparing labels for lead +{h}h …", flush=True)
        labels_by_lead[h] = future_max_label(df, args.target, hours=h)

    # precompute probit(z) of build/relax
    z_b = norm.ppf(np.clip(p_build, 1e-6, 1-1e-6))
    z_r = norm.ppf(np.clip(p_relax, 1e-6, 1-1e-6))

    # CSV header
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["alpha","lead_h","thr","persist_h","quantile","AUC","PRAUC","Brier","F1","Precision","Recall","Coverage"]
    write_row(out, {c:c for c in header_cols}, header=True)

    # sweep
    total = len(alphas)*len(leads)*len(thr_grid)*len(persists)*len(quants)
    k = 0
    for a in alphas:
        blend_score = sigmoid(a*z_b + (1.0-a)*z_r)        # back to [0,1]
        # mild gate: multiply by recent-build factor in [0,1]
        s_comb = blend_score * (0.5 + 0.5*build_recent)   # gives lift when recent build > 0

        for lead in leads:
            y = labels_by_lead[lead]
            # AUC/PRAUC/Brier always on the *continuous* score s_comb
            for thr in thr_grid:
                df["_alert"] = (s_comb >= thr).astype(int)
                for ph in persists:
                    keep_persist = apply_persistence(df, "_alert", ph)
                    for q in quants:
                        keep_top = throttle_by_hour_quantile(df.assign(score=s_comb, flag=keep_persist),
                                                             score_col="score", flag_col="flag", q_keep=q)
                        alerts = (keep_persist & keep_top).astype(int)

                        auc, ap, bri, f1, pr, rc, cov = metrics(y, s_comb, alerts)
                        row = dict(alpha=a, lead_h=lead, thr=thr, persist_h=ph, quantile=q,
                                   AUC=auc, PRAUC=ap, Brier=bri, F1=f1, Precision=pr, Recall=rc, Coverage=cov)
                        write_row(out, row, header=False)

                        k += 1
                        if k % 25 == 0 or k == total:
                            print(f"  … progress {k}/{total}", flush=True)

    print(f"Done. Wrote {out} .", flush=True)

if __name__ == "__main__":
    main()