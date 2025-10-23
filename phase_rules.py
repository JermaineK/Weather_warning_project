# phase_rules.py
# Phase comparison + simple rule mining (no deprecated pandas behaviors)

import argparse, warnings, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")  # sklearn side
pd.options.mode.copy_on_write = True  # safer pandas writes

# ---------- Utility (no deprecated groupby.apply patterns) ----------

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing or any(c not in df.columns for c in ["lat","lon","time"]):
        df = df.sort_values(["lat","lon","time"], kind="mergesort").reset_index(drop=True)
    return df

def rolling_future_max_by_point(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    """
    Compute per-(lat,lon) future max of label_col within 'hours' steps (assumes hourly cadence).
    Works directly from a DataFrame, not just a Series.
    """
    if not {"lat", "lon", label_col}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain lat, lon, and {label_col}")

    def _future_max(s: pd.Series):
        # reverse time → past rolling → re-reverse
        rev = s.iloc[::-1].shift(1)
        rev_max = rev.rolling(window=hours, min_periods=1).max()
        return rev_max.iloc[::-1].fillna(0)

    out = df.groupby(["lat", "lon"], sort=False, group_keys=False)[label_col].transform(_future_max)
    return out

def recent_build_window(prob_build: pd.Series, window_h: int) -> pd.Series:
    """
    Per (lat,lon) group, past-window max (includes only prior hours).
    No deprecated 'on=' or groupby.apply returning frames.
    """
    # naive past rolling: group, then simple rolling on integer window
    def _past_max(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=window_h, min_periods=1).max().fillna(0)

    out = prob_build.groupby(["lat","lon"], sort=False).transform(_past_max)
    return out

def impute_then_scale(df: pd.DataFrame, cols, scaler):
    X = df[cols].to_numpy(float)
    # column medians from the current batch (consistent and simple)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(med, inds[1])
    return scaler.transform(X)

def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    pr, rc, thr = precision_recall_curve(y_true, scores)
    f1 = 2*pr*rc/(pr+rc+1e-9)
    i = int(np.nanargmax(f1))
    # precision_recall_curve returns thresholds len = len(pr)-1
    use_thr = thr[max(i-1, 0)] if len(thr) else 0.5
    return float(f1[i]), float(use_thr), float(pr[i]), float(rc[i])

def rule_from_feature(df: pd.DataFrame, feature: str, y: np.ndarray, dir_hint: str = "pos", steps=25):
    """
    Try 1D threshold rule on a feature. If dir_hint='pos': rule is (feature >= thr); if 'neg': (feature <= thr)
    Returns best F1 and its precision/recall/threshold.
    """
    x = df[feature].to_numpy(float)
    q = np.nanpercentile(x, np.linspace(5, 95, steps))
    best = (-1, None, None, None)  # f1, thr, P, R
    for thr in q:
        if dir_hint == "pos":
            pred = (x >= thr).astype(int)
        else:
            pred = (x <= thr).astype(int)
        P = pred.sum()
        if P == 0:
            continue
        f1 = f1_score(y, pred)
        pr, rc, _ = precision_recall_curve(y, pred)
        # For binary preds, take final point
        P_hat = pred.mean()
        TP = (pred & (y==1)).sum()
        precision = TP / max(P, 1)
        recall = TP / max((y==1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(thr), float(precision), float(recall))
    return best

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Phase comparison + rule mining")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", default="pregen")
    ap.add_argument("--leads", type=lambda s: [int(x) for x in s.split(",")], default=[24,48])
    ap.add_argument("--phase-top-frac", type=float, default=0.10, help="Top fraction by phase prob (e.g. 0.10)")
    ap.add_argument("--subsample-hours", type=float, default=0.0, help="Fraction of distinct hours to sample (0..1)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    if not {"lat","lon","time", args.target}.issubset(df.columns):
        missing = {"lat","lon","time", args.target} - set(df.columns)
        raise ValueError(f"Labelled file missing columns: {missing}")
    df = ensure_sorted(df)

    # Optional hour subsample for speed
    if args.subsample_hours and args.subsample_hours > 0:
        unique_hours = df["time"].dt.floor("h").unique()
        rng = np.random.default_rng(42)
        keep_hours = set(rng.choice(unique_hours, size=int(len(unique_hours)*args.subsample_hours), replace=False))
        df = df[df["time"].dt.floor("h").isin(keep_hours)].reset_index(drop=True)

    # Load models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)

    feats_b = mb["features"]; sc_b = mb["scaler"]
    feats_r = mr["features"]; sc_r = mr["scaler"]

    # Score probs (impute+scale)
    Xb = impute_then_scale(df, feats_b, sc_b)
    Xr = impute_then_scale(df, feats_r, sc_r)
    p_build = mb["model"].predict_proba(Xb)[:, 1]
    p_relax = mr["model"].predict_proba(Xr)[:, 1]

    df["p_build"] = p_build
    df["p_relax"] = p_relax

    # Phase slices (top fraction by each prob)
    qb = df["p_build"].quantile(1.0 - args.phase_top_frac)
    qr = df["p_relax"].quantile(1.0 - args.phase_top_frac)
    top_build_idx = df["p_build"] >= qb
    top_relax_idx = df["p_relax"] >= qr

    # Prepare labels for each lead (strictly future)
    labels_by_lead = {}
    base = df[[args.target, "lat", "lon"]].copy()
    for h in args.leads:
        labels_by_lead[h] = (
            rolling_future_max_by_point(base, args.target, hours=h)
            .to_numpy()
            .astype(int)
        )

    # ---------- Feature influence per phase ----------
    # Union of used features
    feature_pool = list(dict.fromkeys(list(feats_b) + list(feats_r)))
    rows = []
    for h in args.leads:
        y = labels_by_lead[h]
        for tag, mask in [("build", top_build_idx), ("relax", top_relax_idx)]:
            sub = df.loc[mask, feature_pool]
            yy  = y[mask.to_numpy()]
            # Spearman is robust and monotonic
            from scipy.stats import spearmanr
            for col in sub.columns:
                xc = sub[col].to_numpy(float)
                ok = np.isfinite(xc) & np.isfinite(yy)
                if ok.sum() < 100:
                    continue
                r, pval = spearmanr(xc[ok], yy[ok])
                rows.append({
                    "phase": tag,
                    "lead_h": h,
                    "feature": col,
                    "spearman_r": float(r),
                    "abs_r": float(abs(r)),
                    "n": int(ok.sum())
                })
    feat_out = Path(args.outdir) / "phase_feature_influence.csv"
    pd.DataFrame(rows).sort_values(["lead_h","phase","abs_r"], ascending=[True, True, False]).to_csv(feat_out, index=False)

    # ---------- Simple 1-feature rule mining ----------
    # Direction hints: if a feature name suggests magnitude vs. divergence…
    dir_hints = {}
    for f in feature_pool:
        # heuristic hints
        if f in ("S","S_mean3h","S_std3h","zeta_mean","zeta_std3h","agree","relax"):
            dir_hints[f] = "pos"  # higher → more stormy (often true for relax/agree/S coherence)
        elif f in ("div_mean","div_std3h","msl_grad"):
            dir_hints[f] = "neg"  # lower divergence / lower gradient tends to precede organization
        else:
            dir_hints[f] = "pos"

    rule_rows = []
    for h in args.leads:
        y = labels_by_lead[h]
        for f in feature_pool:
            if f not in df.columns:
                continue
            best_f1, thr, P, R = rule_from_feature(df, f, y, dir_hint=dir_hints.get(f,"pos"), steps=31)
            rule_rows.append({
                "lead_h": h,
                "feature": f,
                "direction": dir_hints.get(f,"pos"),
                "thr": thr,
                "F1": best_f1,
                "Precision": P,
                "Recall": R
            })

    rule_df = pd.DataFrame(rule_rows).sort_values(["lead_h","F1"], ascending=[True, False])
    rules_out = Path(args.outdir) / "rules_candidates.csv"
    rule_df.to_csv(rules_out, index=False)

    # ---------- Quick console summary ----------
    def fmt_top(df, lead):
        d = df[df["lead_h"]==lead].head(8)
        lines = []
        for _,r in d.iterrows():
            lines.append(f"  • {lead:>2}h: {r['feature']:>12} ({r['direction']}) thr={r['thr']:.4f}  F1={r['F1']:.3f}  P={r['Precision']:.3f} R={r['Recall']:.3f}")
        return "\n".join(lines)

    print(f"\nSaved phase feature influence → {feat_out}")
    print(f"Saved rule candidates        → {rules_out}\n")
    for h in args.leads:
        print(f"Top 1-feature rules @ lead {h}h")
        print(fmt_top(rule_df, h))
        print("")

if __name__ == "__main__":
    main()