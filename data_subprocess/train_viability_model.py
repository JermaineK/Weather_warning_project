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
from sklearn.compose import ColumnTransformer
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


def _add_lead_features(df: pd.DataFrame, horizon: float = 240.0) -> pd.DataFrame:
    """Add lead-derived helper columns if lead is present."""
    if "t_to_storm_min_h" not in df.columns:
        return df
    lead = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
    lead_clip = lead.clip(lower=0.0, upper=horizon)
    df = df.copy()
    df["lead_clip"] = lead_clip
    df["lead_norm"] = 1.0 - (lead_clip / horizon)
    df["lead_inv"] = 1.0 / (1.0 + lead_clip)
    if "G_struct" in df.columns:
        df["G_lead_norm"] = df["G_struct"] * df["lead_norm"]
    # coarse lead band for stratified sampling (optional downstream)
    bins = [0, 24, 72, 120, horizon, np.inf]
    labels = ["0-24", "24-72", "72-120", "120-240", ">240"]
    df["lead_band"] = pd.cut(lead.fillna(horizon + 1), bins=bins, labels=labels, right=True)
    return df


def _filter_by_lead(df: pd.DataFrame, min_lead: float | None, max_lead: float | None) -> pd.DataFrame:
    if "t_to_storm_min_h" not in df.columns or (min_lead is None and max_lead is None):
        return df
    lead = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_lead is not None:
        mask &= lead >= float(min_lead)
    if max_lead is not None:
        mask &= lead <= float(max_lead)
    filtered = df.loc[mask]
    if filtered.empty:
        raise SystemExit(
            f"Lead filter produced empty frame (min_lead={min_lead}, max_lead={max_lead}). "
            "Relax the bounds or check lead column."
        )
    return filtered


def _subsample(
    df: pd.DataFrame,
    target: str,
    neg_pos_ratio: float,
    sample_frac: float,
    seed: int,
    max_train_rows: int | None = None,
) -> pd.DataFrame:
    if target not in df:
        raise SystemExit(f"Target column '{target}' not found in training data.")

    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    if "lead_band" in df.columns:
        groups = []
        total_rows = 0
        for _, gdf in df.groupby("lead_band"):
            pos = gdf[gdf[target] == 1]
            neg = gdf[gdf[target] == 0]
            if len(pos) == 0 and len(neg) == 0:
                continue
            if len(pos) == 0 or len(neg) == 0:
                groups.append(gdf)
                continue
            if neg_pos_ratio > 0:
                keep_neg = min(len(neg), int(neg_pos_ratio * len(pos)))
                neg = neg.sample(n=keep_neg, random_state=seed)
            g_take = pd.concat([pos, neg], axis=0)
            groups.append(g_take)
            total_rows += len(g_take)
            # keep accumulator bounded
            if max_train_rows and total_rows > max_train_rows * 2:
                merged = pd.concat(groups, axis=0)
                merged = merged.sample(n=max_train_rows, random_state=seed)
                groups = [merged]
                total_rows = len(merged)
        if groups:
            df_fit = pd.concat(groups, axis=0)
            if max_train_rows and len(df_fit) > max_train_rows:
                df_fit = df_fit.sample(n=max_train_rows, random_state=seed)
            return df_fit.sample(frac=1.0, random_state=seed)

    # fallback: original class-balanced sampling
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if neg_pos_ratio > 0 and len(pos) > 0 and len(neg) > 0:
        keep_neg = min(len(neg), int(neg_pos_ratio * len(pos)))
        neg = neg.sample(n=keep_neg, random_state=seed)
    df_fit = pd.concat([pos, neg], axis=0)
    if max_train_rows and len(df_fit) > max_train_rows:
        df_fit = df_fit.sample(n=max_train_rows, random_state=seed)
    return df_fit.sample(frac=1.0, random_state=seed)


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
        help="Feature columns to use (space-separated or comma-separated).",
    )
    ap.add_argument("--min-lead", type=float, default=None, help="Optional minimum lead (hours) to include (>=).")
    ap.add_argument("--max-lead", type=float, default=None, help="Optional maximum lead (hours) to include (<=).")
    ap.add_argument("--neg-pos-ratio", type=float, default=3.0, help="Max negatives per positive (0 disables).")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="Optional overall subsample fraction (0-1].")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength for logistic regression.")
    ap.add_argument("--model-out", required=True, help="Output path for fitted model (.pkl).")
    ap.add_argument("--metrics-json", default=None, help="Where to write metrics JSON.")
    ap.add_argument("--coefs-csv", default=None, help="Where to write coefficient table CSV.")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000, help="Optional cap on rows for fitting after sampling.")
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=None,
        help="Optional chunk size for streaming train input (0/None = load whole file).",
    )
    args = ap.parse_args()

    # normalize features: allow comma-separated single arg or space list
    if len(args.features) == 1 and "," in args.features[0]:
        args.features = [f.strip() for f in args.features[0].split(",") if f.strip()]

    if args.chunksize and args.chunksize > 0:
        chunk_rows = int(args.chunksize)
        if args.train.lower().endswith((".parquet", ".parq", ".pq")):
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(args.train)
            dfs = [batch.to_pandas() for batch in pf.iter_batches(batch_size=chunk_rows)]
        else:
            dfs = list(pd.read_csv(args.train, low_memory=False, chunksize=chunk_rows))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _load_table(args.train)

    # Optional lead filtering to create horizon-specific models
    df = _filter_by_lead(df, args.min_lead, args.max_lead)

    # Add lead-derived helper features for radial/time structure
    df = _add_lead_features(df, horizon=float(args.max_lead) if args.max_lead else 240.0)

    df_fit = _subsample(df, args.target, args.neg_pos_ratio, args.sample_frac, args.seed, args.max_train_rows)
    print(f"[train] using {len(df_fit):,} rows after subsample (features={args.features})")

    # Ensure we have both classes after subsampling; otherwise fall back to full data
    cls_counts = df_fit[args.target].value_counts(dropna=False)
    if cls_counts.nunique() == 1 or len(cls_counts) < 2:
        print(
            "[warn] Subsample produced a single-class dataset "
            f"({cls_counts.to_dict()}); retrying with full data and no class cap."
        )
        df_fit = _subsample(df, args.target, neg_pos_ratio=0, sample_frac=1.0, seed=args.seed)
        cls_counts = df_fit[args.target].value_counts(dropna=False)
        if cls_counts.nunique() == 1 or len(cls_counts) < 2:
            raise SystemExit(
                "Training data contains only one class even after retry.\n"
                f"Class counts: {cls_counts.to_dict()}\n"
                "This usually means build_viability_targets.py produced y_viable=0 for all rows. "
                "Check lead-window detection and g_min filtering there."
            )
        print(f"[train] retry succeeded; using {len(df_fit):,} rows with class counts {cls_counts.to_dict()}")

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
        coef_path = Path(args.coefs_csv)
        if coef_path.suffix.lower() in {".parquet", ".parq", ".pq", ".pqt"}:
            coef_df.to_parquet(coef_path, index=False)
        else:
            coef_df.to_csv(coef_path, index=False)
        print(f"[save] coefficients -> {coef_path}")


if __name__ == "__main__":
    main()
