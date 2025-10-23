# train_with_labels.py  (flexible feature set)
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import joblib

# Preferred features in order; we’ll keep only those present in the CSV
PREFERRED = ["S","zeta_mean","div_mean","relax","agree","ws_mean","ws_var","dir_var","shear_var"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="data/features_era5_au_labelled.csv")
    ap.add_argument("--out", default="models/logit_labelled.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled)
    if "storm" not in df.columns:
        raise ValueError("No 'storm' column in labelled features.")

    # Select only features that actually exist
    feats = [c for c in PREFERRED if c in df.columns]
    if not feats:
        raise ValueError(f"No requested features found. Columns available: {list(df.columns)}")

    df = df.dropna(subset=feats+["storm"])
    X = df[feats].to_numpy(dtype=float)
    y = df["storm"].astype(int).to_numpy()

    # Guard for class imbalance edge cases
    if len(np.unique(y)) < 2 or y.sum() < 2:
        print(f"Not enough positives. Positives={int(y.sum())} / N={len(y)}")
        return

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    logit = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=None)
    logit.fit(Xtr_s, ytr)
    p = logit.predict_proba(Xte_s)[:, 1]

    auc   = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p)
    try:
        from sklearn.metrics import average_precision_score
        prauc = average_precision_score(yte, p)
    except Exception:
        prauc = np.nan

    print(f"[Logit] AUC={auc:.3f}  Brier={brier:.3f}  PRAUC={prauc:.3f}  Pos={yte.sum()}/{len(yte)}")
    print("Using features:", feats)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "model": logit, "features": feats}, args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()