# train_phase.py
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PHASE_FEATURES = {
    "build": ["dS_dt", "drelax_dt", "dagree_dt"],
    "relax": ["S", "relax", "agree", "msl_grad", "zeta_mean", "div_mean"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--target", required=True, choices=["pregen","near_storm","storm"])
    ap.add_argument("--phase", required=True, choices=["build","relax"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-weight", default="balanced")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    use = PHASE_FEATURES[args.phase]
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    df = df.dropna(subset=use+[args.target]).copy()
    y  = df[args.target].astype(int).to_numpy()
    X  = df[use].to_numpy(float)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    base = LogisticRegression(max_iter=200, class_weight=args.class_weight, solver="liblinear")
    base.fit(Xtr_s, ytr)

    # Calibrate for better probabilities
    calib = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    calib.fit(Xtr_s, ytr)

    # Eval
    p = calib.predict_proba(Xte_s)[:,1]
    auc   = roc_auc_score(yte, p)
    prauc = average_precision_score(yte, p)
    brier = brier_score_loss(yte, p)
    pos   = int(y.sum())
    print(f"[{args.phase}] {args.target} | AUC={auc:.3f}  PRAUC={prauc:.3f}  Brier={brier:.3f}  Pos={pos}/{len(y)}  Feats={use}")

    joblib.dump({
        "phase": args.phase,
        "target": args.target,
        "features": use,
        "scaler": sc,
        "model": calib
    }, args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()