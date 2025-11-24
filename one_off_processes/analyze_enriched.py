#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def main(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, compression="infer", low_memory=False)

    print("Loaded:", df.shape)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "storm_hit" not in df.columns:
        raise ValueError("storm_hit column missing — cannot analyse hits vs non-hits")

    y = df["storm_hit"].astype(int)
    X = df[numeric_cols].copy()

    # Drop row_id / score / target from the scoring analysis
    drop_cols = set(["row_id", "storm_hit"])
    candidate_cols = [c for c in numeric_cols if c not in drop_cols]

    print(f"Numeric candidate columns: {len(candidate_cols)}")

    # Detect constant or near-constant features
    constant = []
    for c in candidate_cols:
        if X[c].nunique() <= 2:
            constant.append(c)
    print("\nConstant or useless columns:")
    print(constant)

    # Compute hit vs non-hit contrast
    results = []
    for c in candidate_cols:
        xv = pd.to_numeric(X[c], errors='coerce')
        all_mean = float(xv.mean())
        hit_mean = float(xv[y == 1].mean()) if (y == 1).any() else np.nan
        delta = hit_mean - all_mean
        results.append((c, all_mean, hit_mean, delta))

    contrast = pd.DataFrame(results, columns=["feature", "mean_all", "mean_hits", "delta"])
    contrast = contrast.sort_values("delta", ascending=False)

    print("\nTop 20 features by delta (hit_mean - all_mean):")
    print(contrast.head(20))

    # Univariate AUC per feature
    auc_rows = []
    for c in candidate_cols:
        if y.sum() > 0:
            try:
                xv = pd.to_numeric(X[c], errors='coerce')
                mask = xv.notna()
                auc = roc_auc_score(y[mask], xv[mask])
            except Exception:
                auc = np.nan
        else:
            auc = np.nan
        auc_rows.append((c, auc))

    auc_df = pd.DataFrame(auc_rows, columns=["feature", "auc"]).sort_values("auc", ascending=False)
    print("\nTop AUC features:")
    print(auc_df.head(20))

    # Correlation matrix (to detect redundancy)
    corr = X[candidate_cols].corr()
    corr.to_csv("feature_correlations.csv")
    print("Saved feature_correlations.csv")

    contrast.to_csv("feature_contrast.csv", index=False)
    auc_df.to_csv("feature_auc.csv", index=False)

    print("\nGenerated:")
    print(" feature_contrast.csv")
    print(" feature_auc.csv")
    print(" feature_correlations.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_enriched.py <csv.gz>")
        sys.exit(1)
    main(sys.argv[1])z