#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_model_auc_pregen.py

Quick sanity check:
- load the grid_logit_perlead model bundle
- load the labelled grid (only needed columns)
- build future-24h labels from 'pregen'
- compute AUC(p) and AUC(1-p)

This tells us whether the model is actually inverted for pregen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# -------- config (edit if paths differ) --------

LABELLED = Path("data/grid_labelled_FMA_gka_realthermo_merged_with_id.parquet")
MODEL    = Path("models/grid_logit_perlead.pkl")
TARGET   = "pregen"
LEAD     = 24

# Optional: small sampling if you hit memory issues
SAMPLE_FRAC = 0.0   # e.g. 0.1 for 10% of rows, 0.0 = disable
MAX_ROWS    = 0     # e.g. 1_000_000 cap, 0 = no cap


# -------- I/O helper (minimal, same spirit as sweep_runner) --------

def read_any(path: Path | str,
             parse_dates: Optional[List[str]] = None,
             usecols: Optional[List[str]] = None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(
        path,
        low_memory=False,
        compression="infer",
        parse_dates=parse_dates if parse_dates else None,
        usecols=usecols if usecols else None,
        memory_map=True,
    )


# -------- model bundle helpers (trimmed version of sweep_runner) --------

def load_model_bundle(
    model_path: str | Path,
) -> Tuple[Any, Dict[int, Any], Optional[Any], List[str],
           Optional[Dict[str, float]], Optional[Dict[str, np.ndarray]]]:
    """
    Same logic as sweep_runner.load_model_bundle, but inlined here
    so we don't depend on importing sweep_runner.
    """
    m = joblib.load(model_path)

    global_model = None
    per_lead: Dict[int, Any] = {}
    scaler = None
    features: List[str] = []
    imputer_stats: Optional[Dict[str, float]] = None
    clip_stats: Optional[Dict[str, np.ndarray]] = None

    if isinstance(m, dict):
        features = list(m.get("features") or m.get("feats") or [])
        scaler = m.get("scaler", None)
        imputer_stats = m.get("imputer_stats", None)
        clip_stats = m.get("clip_stats", None)

        # Explicit per-lead dict
        if "per_lead_models" in m and isinstance(m["per_lead_models"], dict):
            for k, v in m["per_lead_models"].items():
                try:
                    lead = int(k)
                    per_lead[lead] = v
                except Exception:
                    pass

        # Generic "models" dict
        elif "models" in m and isinstance(m["models"], dict):
            for k, v in m["models"].items():
                try:
                    lead = int(k)
                    per_lead[lead] = v
                except Exception:
                    # e.g. "global"
                    global_model = v

        if global_model is None:
            global_model = (
                m.get("model")
                or m.get("model-out")
                or m.get("estimator")
                or m.get("pipe")
            )

        if global_model is None and not per_lead:
            raise ValueError(
                "Unsupported model bundle format: expected 'model', 'models', or 'per_lead_models'."
            )

        if not features:
            meta_feats = []
            meta = m.get("meta", {})
            if isinstance(meta, dict):
                meta_feats = list(meta.get("features", []))
            if meta_feats:
                features = meta_feats
            else:
                raise ValueError("Model bundle lacks 'features' list; cannot build X matrix.")

        return global_model, per_lead, scaler, features, imputer_stats, clip_stats

    # Plain estimator fallback
    global_model = m
    scaler = getattr(m, "scaler_", None)
    feat_arr = getattr(m, "feature_names_in_", None) or getattr(m, "features_", None)
    if feat_arr is None:
        raise ValueError(
            "Plain estimator bundle has no 'feature_names_in_' or 'features_'; "
            "please save as a dict bundle with 'features'."
        )
    features = list(feat_arr)
    return global_model, per_lead, scaler, features, None, None


def build_feature_matrix(
    df: pd.DataFrame,
    features: List[str],
    imputer_stats: Optional[Dict[str, float]],
    clip_stats: Optional[Dict[str, np.ndarray]],
    scaler: Optional[Any],
) -> np.ndarray:
    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan

    X = Xdf.to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan

    # Impute
    if imputer_stats:
        for j, c in enumerate(features):
            fill = float(imputer_stats.get(c, 0.0))
            m = ~np.isfinite(X[:, j])
            if m.any():
                X[m, j] = fill
    else:
        X = np.where(np.isfinite(X), X, 0.0)

    # Clip
    if clip_stats and "lo" in clip_stats and "hi" in clip_stats:
        lo = np.asarray(clip_stats["lo"], dtype=np.float32)
        hi = np.asarray(clip_stats["hi"], dtype=np.float32)
        if lo.shape == X.shape[1:] and hi.shape == X.shape[1:]:
            X = np.clip(X, lo, hi)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    return X.astype(np.float32, copy=False)


def score_with_estimator(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1].astype(np.float64)
    if hasattr(est, "decision_function"):
        from scipy.special import expit
        dec = est.decision_function(X).astype(np.float64)
        return expit(dec)
    pred = est.predict(X).astype(np.float64)
    return np.clip(pred, 0.0, 1.0)


# -------- label builder (same logic as sweep_runner.future_max_timeaware) --------

def future_max_timeaware(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)

    for (_, _), g in df.groupby(["lat", "lon"], sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y = pd.to_numeric(g[target], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        t_ns = pd.to_datetime(g["time"], errors="coerce", utc=True).view("int64").to_numpy()

        order = np.argsort(t_ns, kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        t_sorted = t_ns[order]
        y_sorted = y[order]

        ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
        ps[1:] = np.cumsum(y_sorted)

        t_end_sorted = t_sorted + hour_ns
        end_pos = np.searchsorted(t_sorted, t_end_sorted, side="right")

        any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
        any_future = any_future_sorted[inv].astype(np.int8)
        out[idx] = any_future

    return out


# -------- main debug flow --------

def main():
    print("== debug_model_auc_pregen ==")
    print("Labelled :", LABELLED)
    print("Model    :", MODEL)
    print("Target   :", TARGET)
    print("Lead     :", LEAD)

    global_model, per_lead, scaler, feats, imp_stats, clip_stats = load_model_bundle(MODEL)

    need_cols = set(feats) | {"time", "lat", "lon", TARGET}
    df = read_any(LABELLED, parse_dates=["time"], usecols=list(need_cols))

    # Optional sampling
    if SAMPLE_FRAC and 0 < SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
        print(f"[debug] SAMPLE_FRAC={SAMPLE_FRAC} -> rows={len(df):,}")

    if MAX_ROWS and MAX_ROWS > 0 and len(df) > MAX_ROWS:
        df = df.sample(n=int(MAX_ROWS), random_state=42).reset_index(drop=True)
        print(f"[debug] MAX_ROWS={MAX_ROWS} -> rows={len(df):,}")

    # Core typing
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time","lat","lon"]).sort_values(
        ["time","lat","lon"], kind="mergesort"
    ).reset_index(drop=True)

    # Force base target to 0/1
    y_base = pd.to_numeric(df[TARGET], errors="coerce").fillna(0)
    df[TARGET] = (y_base > 0).astype(np.int8)

    print(f"Rows used: {len(df):,}")

    # Build future label
    y = future_max_timeaware(df, TARGET, LEAD)
    pos = int(y.sum())
    print(f"Future positives (lead={LEAD}h): {pos:,} / {len(y):,} (frac={pos/len(y):.6f})")

    # Build features and scores
    X = build_feature_matrix(df, feats, imp_stats, clip_stats, scaler)
    est = per_lead.get(LEAD, global_model)
    p = score_with_estimator(est, X)

    # AUCs
    if pos == 0 or pos == len(y):
        print("Label is pure one-class; AUC is undefined (NaN).")
        return

    auc_p = roc_auc_score(y, p)
    auc_inv = roc_auc_score(y, 1.0 - p)

    print(f"AUC(p)   = {auc_p:.3f}")
    print(f"AUC(1-p) = {auc_inv:.3f}")


if __name__ == "__main__":
    main()
