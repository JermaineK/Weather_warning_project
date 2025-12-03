#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_viability_model.py

Fit a simple viability model P(y_viable | G, S, E, ...).
Default: logistic regression with balanced class weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Agent: baseline viable/not-viable fit; keep maths simple and interpretable.

def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _subsample(df: pd.DataFrame, target: str, neg_pos_ratio: float, sample_frac: float, seed: int) -> pd.DataFrame:
    if target not in df:
        raise SystemExit(f"Target column '{target}' not found in training data.")

    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if neg_pos_ratio > 0 and len(pos) > 0 and len(neg) > 0:
        keep_neg = min(len(neg), int(neg_pos_ratio * len(pos)))
        neg = neg.sample(n=keep_neg, random_state=seed)
    df_fit = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=seed)
    return df_fit


def _prepare_xy(df: pd.DataFrame, features: Sequence[str], target: str):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy()
    return X, y


def _fit_model(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=500, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _metrics(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    return {"roc_auc": float(auc), "avg_precision": float(ap)}


def _coeff_table(model: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    lr = model.named_steps["lr"]
    coefs = lr.coef_.ravel()
    return pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values(
        "abs_coef", ascending=False
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a viability model on GSE panel features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", required=True, help="Training table with y_viable.")
    ap.add_argument("--target", default="y_viable", help="Target column.")
    ap.add_argument(
        "--features",
        nargs="+",
        default=["G_struct", "S_shear", "E_energy"],
        help="Feature columns to use.",
    )
    ap.add_argument("--neg-pos-ratio", type=float, default=3.0, help="Max negatives per positive (0 disables).")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="Optional overall subsample fraction (0-1].")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength for logistic regression.")
    ap.add_argument("--model-out", required=True, help="Output path for fitted model (.pkl).")
    ap.add_argument("--metrics-json", default=None, help="Where to write metrics JSON.")
    ap.add_argument("--coefs-csv", default=None, help="Where to write coefficient table CSV.")
    args = ap.parse_args()

    df = _load_table(args.train)
    df_fit = _subsample(df, args.target, args.neg_pos_ratio, args.sample_frac, args.seed)
    print(f"[train] using {len(df_fit):,} rows after subsample (features={args.features})")

    X, y = _prepare_xy(df_fit, args.features, args.target)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    model = _fit_model(X_train, y_train, args.C, args.seed)
    train_metrics = _metrics(model, X_train, y_train)
    val_metrics = _metrics(model, X_val, y_val)

    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    print(f"[save] model -> {out_model}")

    metrics = {"train": train_metrics, "val": val_metrics, "features": list(args.features), "target": args.target}
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))
        print(f"[save] metrics -> {args.metrics_json}")

    if args.coefs_csv:
        coef_df = _coeff_table(model, args.features)
        Path(args.coefs_csv).parent.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(args.coefs_csv, index=False)
        print(f"[save] coefficients -> {args.coefs_csv}")


if __name__ == "__main__":
    main()
