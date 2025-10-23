# train_grid.py
# Train a logistic model on gridded ERA5 features with storm labels (AU grid).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib


def parse_args():
    ap = argparse.ArgumentParser(description="Train logistic model on labelled grid features.")
    ap.add_argument("--labelled", required=True, help="Labelled CSV with columns: time, lat, lon, targets + features.")
    ap.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--model-out", required=True, help="Where to save model .pkl")
    ap.add_argument("--class-weight", default="none",
                    help="Use 'balanced' or a numeric positive weight for the positive class (e.g. 5.0).")
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.labelled, low_memory=False)
    # Parse time if present
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.labelled}")

    # Columns we never train on
    never_feats = {"time", "lat", "lon", "storm", "near_storm", "pregen"}

    # Keep ONLY numeric columns (prevents 'could not convert string to float')
    numeric_cols = [c for c in df.columns if c not in never_feats and pd.api.types.is_numeric_dtype(df[c])]
    ignored_cols = [c for c in df.columns if c not in never_feats and c not in numeric_cols]

    # Target vector
    y = df[args.target].astype(int)

    # Feature matrix
    X = df[numeric_cols]

    # Drop rows with NaNs or non-finite in X or y
    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    mask_notna = ~X.isna().any(axis=1)
    mask = mask_finite & mask_notna & y.notna().to_numpy()

    X = X.loc[mask].to_numpy()
    y = y.loc[mask].to_numpy().astype(int)

    # If extremely imbalanced, stratify may fail; handle gracefully
    test_size = 0.2
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:
        # fallback without stratify
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    # Class weight
    cw = None
    if args.class_weight.lower() == "balanced":
        cw = "balanced"
    else:
        try:
            w = float(args.class_weight)
            if w > 0:
                cw = {0: 1.0, 1: w}
        except Exception:
            cw = None

    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=cw)
    model.fit(Xtr_s, ytr)

    pte = model.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, pte)
    ap = average_precision_score(yte, pte)
    brier = brier_score_loss(yte, pte)

    print(f"Rows used: {len(y)}  Positives (train+test): {int(y.sum())}/{len(y)}")
    print(f"Ignored non-numeric columns: {ignored_cols if ignored_cols else 'None'}")
    print(f"[{args.target}] AUC={auc:.3f}  PRAUC={ap:.3f}  Brier={brier:.3f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "model": model, "features": numeric_cols}, args.model_out)
    print(f"Saved {args.model_out}")


if __name__ == "__main__":
    main()