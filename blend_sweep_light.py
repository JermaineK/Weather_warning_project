# blend_sweep_light.py
import argparse, sys, json, math, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import norm

pd.options.mode.copy_on_write = True

def log(msg): print(msg, flush=True)

def parse_float_grid(spec):
    # "0.05,0.06" or "0.05:0.09:0.01"
    spec = spec.strip()
    if ":" in spec:
        a,b,s = spec.split(":")
        a,b,s = float(a), float(b), float(s)
        n = int(round((b - a) / s)) + 1
        return [round(a + i*s, 10) for i in range(n)]
    return [float(x) for x in spec.split(",") if x]

def impute_then_scale(X: pd.DataFrame, scaler: StandardScaler):
    # Median impute (fast), then apply existing scaler
    X_imp = X.copy()
    if X_imp.isna().any().any():
        imp = SimpleImputer(strategy="median")
        X_imp[:] = imp.fit_transform(X_imp)
    return scaler.transform(X_imp.to_numpy(float))

def score_model_probs(df, model_path):
    m = joblib.load(model_path)
    feats, sc, clf = m["features"], m["scaler"], m["model"]
    X = df[feats]
    nans = int(X.isna().any(axis=1).sum())
    if nans: log(f"  • {Path(model_path).name}: imputed NaNs on {nans} rows")
    Xs = impute_then_scale(X, sc)
    return clf.predict_proba(Xs)[:,1]

def probit_blend(p_build, p_relax, alpha):
    # Blend in z-space; guard against 0/1
    eps = 1e-6
    z_b = norm.ppf(np.clip(p_build, eps, 1-eps))
    z_r = norm.ppf(np.clip(p_relax, eps, 1-eps))
    z   = alpha * z_b + (1 - alpha) * z_r
    return norm.cdf(z)

def recent_build_window(df, prob, hours):
    # For each (lat,lon), rolling max over last N hours; we use shift(1) so it’s “recent up to now”
    prob_series = pd.Series(prob, index=df.index, name="p_build")
    tmp = df[["lat","lon","time"]].copy()
    tmp["p_build"] = prob_series
    def _roll(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)["p_build"].max().shift(1)
    out = tmp.groupby(["lat","lon"], sort=False, group_keys=False).apply(_roll)
    return out.to_numpy()

def future_max_label(df, label_col, hours):
    # For each (lat,lon), “will there be a label in the next N hours?”
    s = df[label_col].astype(int)
    tmp = df[["lat","lon","time"]].copy()
    tmp[label_col] = s
    def _lead(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)[label_col].max().shift(-1)
    out = tmp.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)
    return out.reindex(df.index).fillna(0).astype(int).to_numpy()

def apply_persistence(df, flag, persist_h):
    if persist_h <= 0: return flag
    tmp = df[["lat","lon","time"]].copy()
    tmp["flag"] = flag.astype(int)
    def _persist(g):
        return g.rolling(f"{persist_h}h", on="time", min_periods=1)["flag"].max()
    out = tmp.groupby(["lat","lon"], sort=False, group_keys=False).apply(_persist)
    return out.reindex(df.index).fillna(0).astype(int).to_numpy()

def top_quantile_per_hour(df, score, q):
    # keep top q-quantile per hour
    s = pd.Series(score, index=df.index, name="score")
    tmp = df[["time"]].copy()
    tmp["score"] = s
    def keep_top(g):
        if len(g) == 0: return pd.Series([], dtype=int)
        thr = g["score"].quantile(q)
        return (g["score"] >= thr).astype(int)
    out = tmp.groupby("time", sort=False, group_keys=False).apply(keep_top)
    return out.reindex(df.index).fillna(0).astype(int).to_numpy()

def metrics(y, p, mask=None):
    if mask is not None:
        y = y[mask]; p = p[mask]
    if y.sum() == 0:
        return dict(AUC=np.nan, PRAUC=np.nan, Brier=np.nan)
    try: auc = roc_auc_score(y, p)
    except: auc = np.nan
    try: prauc = average_precision_score(y, p)
    except: prauc = np.nan
    try: brier = brier_score_loss(y, p)
    except: brier = np.nan
    return dict(AUC=auc, PRAUC=prauc, Brier=brier)

def prf(y_true, y_pred):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true.astype(bool) & (y_pred==1)).sum())
    fn = int((y_true==1 & (y_pred==0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec+rec+1e-9)
    return prec, rec, f1

def main():
    ap = argparse.ArgumentParser(description="Fast α/threshold sweep for probit blend (build+relax).")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", default="pregen", choices=["pregen","near_storm","storm"])
    ap.add_argument("--alphas", default="0.5")
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--leads", default="24,48")
    ap.add_argument("--thr-grid", default="0.05:0.09:0.01")
    ap.add_argument("--persist", default="0,1")
    ap.add_argument("--quantiles", default="0.90")
    ap.add_argument("--subsample-hours", type=float, default=0.25, help="fraction of hours to evaluate (speedup)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    alphas    = [float(x) for x in args.alphas.split(",") if x]
    leads     = [int(x)   for x in args.leads.split(",") if x]
    thrs      = parse_float_grid(args.thr_grid)
    persists  = [int(x)   for x in args.persist.split(",") if x]
    quants    = [float(x) for x in args.quantiles.split(",") if x]

    log("== Blend Sweep (light) ==")
    log(f"File   : {args.labelled}")
    log(f"Build  : {args.build}")
    log(f"Relax  : {args.relax}")
    log(f"Target : {args.target}")
    log(f"Alphas : {alphas}")
    log(f"Build window (h): {args.build_window}")
    log(f"Leads  : {leads}")
    log(f"Thresh : {thrs[0]} … {thrs[-1]} (n={len(thrs)})")
    log(f"Persist: {persists}  Quantiles: {quants}")
    log(f"Subsample hours: {args.subsample_hours}")

    # --- Load & (optionally) subsample by hour ---
    usecols = ["time","lat","lon", args.target, "S","relax","agree","zeta_mean","div_mean","msl_grad","dS_dt","drelax_dt","dagree_dt",
               "S_mean3h","S_std3h","zeta_mean3h","zeta_std3h","div_mean3h","div_std3h","msl"]
    df = pd.read_csv(args.labelled, parse_dates=["time"], usecols=lambda c: True)  # read all; features filtered later by models
    N0 = len(df)
    # subsample hours uniformly (speed)
    if 0 < args.subsample_hours < 1:
        hours = df["time"].dt.floor("h").drop_duplicates().sort_values()
        keep_hours = hours.sample(frac=args.subsample_hours, random_state=42)
        df = df[df["time"].dt.floor("h").isin(keep_hours)].reset_index(drop=True)
    log(f"Rows → {len(df):,}  (from {N0:,})")

    # --- Score build/relax once ---
    log("Scoring build model…")
    p_build = score_model_probs(df, args.build)
    log("Scoring relax model…")
    p_relax = score_model_probs(df, args.relax)

    # --- Build recent window (shift(1)) ---
    log(f"Computing recent build window ({args.build_window}h)…")
    build_recent = recent_build_window(df, p_build, args.build_window)

    # --- Labels for each lead ---
    labels_by_lead = {}
    for h in leads:
        log(f"Preparing labels for lead +{h}h …")
        labels_by_lead[h] = future_max_label(df, args.target, h)

    # --- Sweep ---
    rows = []
    total = len(alphas)*len(leads)*len(thrs)*len(persists)*len(quants)
    k = 0
    for a in alphas:
        # Probit-blend once (independent of thr/persist/q)
        p_blend = probit_blend(p_build, p_relax, a)

        # Slight “gate” via recent build: multiply by 0.5 + 0.5*norm(build_recent)
        # (keeps scale in [0,1], nudges scores up if recent build activity)
        br = np.nan_to_num(build_recent, nan=0.0)
        gate = 0.5 + 0.5*br  # simple, fast heuristic
        p_adj = np.clip(p_blend * gate, 0, 1)

        for lead_h in leads:
            y = labels_by_lead[lead_h].astype(int)
            for thr in thrs:
                base_flag = (p_adj >= thr).astype(int)
                for ph in persists:
                    flag_p = apply_persistence(df, base_flag, ph)
                    for q in quants:
                        keep = top_quantile_per_hour(df, p_adj, q)
                        alert = (flag_p & keep).astype(int)

                        # Metrics
                        m = metrics(y, p_adj)
                        P, R, F1 = prf(y.astype(bool), alert.astype(bool))
                        cov = alert.mean()

                        rows.append(dict(
                            alpha=a, lead_h=lead_h, thr=thr, persist_h=ph, quantile=q,
                            AUC=m["AUC"], PRAUC=m["PRAUC"], Brier=m["Brier"],
                            F1=F1, Precision=P, Recall=R, Coverage=cov
                        ))
                        k += 1
                        if k % 50 == 0: log(f"  … {k}/{total} combos")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    log(f"Wrote {args.out}  rows={len(out)}")

    # Print best per lead and an overall pick
    try:
        best_per = out.sort_values("F1", ascending=False).groupby("lead_h", as_index=False).head(1)
        log("\n== Best per lead ==")
        with pd.option_context("display.max_columns", 20):
            print(best_per[["lead_h","alpha","thr","persist_h","quantile","F1","Precision","Recall","Coverage","AUC","PRAUC","Brier"]].to_string(index=False))
        best = out.loc[out["F1"].idxmax()]
        log("\n== Best overall ==")
        for k in ["alpha","lead_h","thr","persist_h","quantile","F1","Precision","Recall","Coverage","AUC","PRAUC","Brier"]:
            print(f"{k}: {best[k]}")
    except Exception as e:
        log(f"(summary skipped: {e})")

if __name__ == "__main__":
    main()