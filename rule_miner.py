#!/usr/bin/env python
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, brier_score_loss
from sklearn.model_selection import train_test_split

def future_max_label(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    def _lead(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("time")[label_col].astype(int)
        fwd = s.rolling(f"{hours}h", min_periods=1).max().shift(-1)
        return pd.Series(fwd.reindex(g["time"]).fillna(0).to_numpy(dtype=int), index=g.index)
    return df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)

def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = (2*pr*rc)/(pr+rc+1e-9)
    i = np.nanargmax(f1)
    thr = th[max(i-1, 0)] if len(th) else 0.5
    return float(thr), float(f1[i]), float(pr[i]), float(rc[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="data/grid_labelled_FMA_phasecols.csv.gz")
    ap.add_argument("--target", default="pregen", choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--lead-hours", type=int, default=24)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=8, help="Top-K abs-weight features to propose rule from")
    ap.add_argument("--C", type=float, default=0.5, help="Inverse L1 strength (smaller → sparser)")
    ap.add_argument("--out", default="models/rule_l1.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    # Candidate features (subset gracefully to what's present)
    preferred = ["S","relax","agree","msl_grad","zeta_mean","div_mean","dS_dt","drelax_dt","dagree_dt",
                 "S_mean3h","S_std3h","zeta_std3h","div_std3h"]
    use = [c for c in preferred if c in df.columns]
    if not use:
        raise SystemExit("No usable features found.")

    # Labels (lead)
    y = future_max_label(df, args.target, hours=args.lead_hours).to_numpy()
    base_pos = int(y.sum())
    print(f"Rows: {len(df)}  Positives(+{args.lead_hours}h): {base_pos}")

    # Drop rows with NaNs in features
    keep = ~df[use].isna().any(axis=1)
    df, y = df.loc[keep], y[keep.to_numpy()]
    X = df[use].to_numpy(float)

    # Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

    # Sparse logistic
    clf = LogisticRegression(penalty="l1", solver="liblinear",
                             class_weight="balanced", C=args.C,
                             max_iter=500, random_state=args.random_state)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    auc  = roc_auc_score(yte, p)
    ap   = average_precision_score(yte, p)
    bri  = brier_score_loss(yte, p)
    thr, f1, prec, rec = best_f1_threshold(yte, p)

    print("\n== L1-Logistic metrics ==")
    print(f"AUC={auc:.3f}  PRAUC={ap:.3f}  Brier={bri:.3f}")
    print(f"Best-F1 (test)={f1:.3f} at thr={thr:.3f}  (P={prec:.3f}, R={rec:.3f})")

    # Non-zero weights
    w = clf.coef_.ravel()
    nz = [(feat, float(coef)) for feat, coef in zip(use, w) if abs(coef) > 1e-8]
    nz_sorted = sorted(nz, key=lambda t: abs(t[1]), reverse=True)
    print("\nNon-zero weights (sorted by |coef|):")
    for f, c in nz_sorted:
        print(f"  {f:12s}  {c:+.4f}")

    # Propose a tiny rule from top-2 features
    top = [f for f,_ in nz_sorted][:min(args.max_features, 2)]
    if len(top) >= 1:
        # Grid search a single-threshold rule on the strongest feature in original scale
        f0 = top[0]
        v  = df[f0].to_numpy()
        # evaluate thresholds at percentiles
        grid = np.nanpercentile(v, np.linspace(10, 90, 17))
        best = (0.0, -1.0, 0.0, 0.0)  # thr, f1, prec, rec
        for t in grid:
            yhat = (v >= t).astype(int) if w[use.index(f0)] >= 0 else (v <= t).astype(int)
            # Use same test split indices for fairness
            mask = np.zeros(len(y), dtype=bool); mask[Xtr.shape[0]:] = True  # approximate split mapping
            # Simpler: evaluate on all rows (gives an upper bound, okay for guidance)
            pr, rc, _ = precision_recall_curve(y, yhat)
            f1s = (2*pr*rc)/(pr+rc+1e-9)
            j = np.nanargmax(f1s)
            if f1s[j] > best[1]:
                best = (float(t), float(f1s[j]), float(pr[j]), float(rc[j]))
        dirword = ">=" if w[use.index(f0)] >= 0 else "<="
        print(f"\n== Proposed 1-feature rule (global) ==")
        print(f"IF {f0} {dirword} {best[0]:.4f} THEN storm-within-{args.lead_hours}h")
        print(f"F1={best[1]:.3f}  (P={best[2]:.3f}, R={best[3]:.3f})  — heuristic estimate")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "model": clf, "features": use,
                 "lead_hours": args.lead_hours, "best_thr": thr}, args.out)
    print(f"\nSaved sparse rule model → {args.out}")

if __name__ == "__main__":
    main()