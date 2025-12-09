#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_viability_leads.py

Lead-aware evaluation for the viability model using t_to_storm_min_h.

Semantics:
  - Coincident target: y_viable (or --target)
  - Lead-L target: 1 if 0 < t_to_storm_min_h <= L (or > lead-lower)

Defaults are aligned to the new pipeline layout:
  panel : data/grid_train_gse_panel_targets.parquet
  model : models/viability_model.pkl
  metrics-json : models/viability_model_metrics.json  (for feature list / target)
  out   : results/metrics/<run_name>_viability_leads.csv (when --run-name set)

Agent: add viability-focused evaluator without touching existing leadtime scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

pd.options.mode.copy_on_write = True


def _parse_feature_list(tokens: Iterable[str] | None) -> list[str] | None:
    if not tokens:
        return None
    if len(tokens) == 1 and "," in str(list(tokens)[0]):
        return [t.strip() for t in str(list(tokens)[0]).split(",") if t.strip()]
    return [str(t).strip() for t in tokens if str(t).strip()]


def _load_metrics_features(path: str | None) -> tuple[list[str] | None, str | None]:
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        meta = json.loads(p.read_text())
        feats = meta.get("features")
        tgt = meta.get("target")
        if feats is not None:
            feats = [str(f) for f in feats]
        tgt = str(tgt) if tgt else None
        return feats, tgt
    except Exception:
        return None, None


def _load_panel(path: str, need_cols: Sequence[str]) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=list(dict.fromkeys(need_cols)))
    return pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(need_cols))


def _parse_leads(raw: Iterable[str]) -> list[float]:
    out: list[float] = []
    for tok in raw:
        for part in str(tok).replace(",", " ").split():
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                raise SystemExit(f"Could not parse lead-hours token '{tok}'")
    return out


def _build_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Numeric matrix in the exact feature order; NaNs -> 0."""
    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan
    return Xdf.fillna(0.0).to_numpy(dtype=float, copy=False)


def _lead_mask(dt: np.ndarray, lead_h: float, lead_lower: float) -> np.ndarray:
    """Strict future window: (lead_lower, lead_h]."""
    return np.isfinite(dt) & (dt > lead_lower) & (dt <= lead_h)


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    mets: dict[str, float] = {}
    try:
        mets["auc"] = roc_auc_score(y, p)
    except Exception:
        mets["auc"] = float("nan")
    try:
        mets["prauc"] = average_precision_score(y, p)
    except Exception:
        mets["prauc"] = float("nan")
    try:
        mets["brier"] = brier_score_loss(y, p)
    except Exception:
        mets["brier"] = float("nan")
    return mets


def parse_args():
    ap = argparse.ArgumentParser(
        description="Lead-aware viability evaluation using t_to_storm_min_h.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--panel",
        default="data/grid_train_gse_panel_targets.parquet",
        help="Panel with y_viable, lead column, and features.",
    )
    ap.add_argument(
        "--model",
        default="models/viability_model.pkl",
        help="Joblib sklearn estimator or bundle dict with 'model'.",
    )
    ap.add_argument(
        "--model-metrics",
        default="models/viability_model_metrics.json",
        help="JSON with 'features' (and optionally 'target') for the viability model.",
    )
    # Chunking hints (accepted for pipeline compatibility; currently full in-memory)
    ap.add_argument("--chunk-rows", type=int, default=None, help="Optional chunk hint; accepted for compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Optional row-group hint; accepted for compatibility.")
    ap.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature list (space- or comma-separated). Defaults to metrics JSON if absent.",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Target column for coincident metrics. Defaults to metrics JSON target or y_viable.",
    )
    ap.add_argument(
        "--lead-col",
        default="t_to_storm_min_h",
        help="Column holding minutes-to-storm (hours).",
    )
    ap.add_argument(
        "--lead-hours",
        nargs="+",
        type=str,
        default=["24", "48", "72", "120"],
        help="Lead horizons (hours) to evaluate.",
    )
    ap.add_argument(
        "--lead-lower",
        type=float,
        default=0.0,
        help="Strict lower bound for lead window (exclude current/negative).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to stamp outputs (results/metrics/<run>_viability_leads.csv).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV for metrics (defaults to run-stamped path when run-name is set).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    feats_from_args = _parse_feature_list(args.features)
    feats_from_metrics, tgt_from_metrics = _load_metrics_features(args.model_metrics)

    # Load estimator (allow dict bundle with 'model')
    bundle = joblib.load(args.model)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    if model is None:
        raise SystemExit("Model bundle missing 'model' key.")

    features = feats_from_args or feats_from_metrics
    if not features and isinstance(bundle, dict):
        maybe_feats = bundle.get("features")
        if maybe_feats:
            features = list(maybe_feats)
    if not features:
        raise SystemExit("No feature list found. Provide --features, metrics JSON, or a bundle with 'features'.")

    target = args.target or tgt_from_metrics or "y_viable"
    lead_col = args.lead_col
    lead_hours = _parse_leads(args.lead_hours)

    # Resolve default output path
    out_path = args.out
    if not out_path and args.run_name:
        out_path = f"results/metrics/{args.run_name}_viability_leads.csv"
    if not out_path:
        out_path = "results/metrics/viability_leads.csv"

    need_cols = set(features) | {target, lead_col}
    df = _load_panel(args.panel, need_cols)

    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Panel missing columns: {missing}")

    # Convert lead/target to numeric
    lead_vals = pd.to_numeric(df[lead_col], errors="coerce").to_numpy(dtype=float)
    y_coincident = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy()

    X = _build_matrix(df, features)
    probs = model.predict_proba(X)[:, 1]

    rows = []

    # Coincident metrics (info only, lead_h=0 marker)
    coinc = _safe_metrics(y_coincident, probs)
    rows.append(
        {
            "kind": "coincident",
            "lead_h": 0.0,
            "pos": int(y_coincident.sum()),
            "samples": int(len(y_coincident)),
            "pos_rate": float(y_coincident.mean() if len(y_coincident) else np.nan),
            "auc": coinc["auc"],
            "prauc": coinc["prauc"],
            "brier": coinc["brier"],
            "lead_lower": args.lead_lower,
            "lead_col": lead_col,
            "target": target,
            "run_name": args.run_name or "",
        }
    )

    # Per-lead metrics
    for h in lead_hours:
        mask = _lead_mask(lead_vals, lead_h=float(h), lead_lower=float(args.lead_lower))
        y_lead = mask.astype(int)
        mets = _safe_metrics(y_lead, probs)
        rows.append(
            {
                "kind": "lead",
                "lead_h": float(h),
                "pos": int(y_lead.sum()),
                "samples": int(len(y_lead)),
                "pos_rate": float(y_lead.mean() if len(y_lead) else np.nan),
                "auc": mets["auc"],
                "prauc": mets["prauc"],
                "brier": mets["brier"],
                "lead_lower": args.lead_lower,
                "lead_col": lead_col,
                "target": target,
                "run_name": args.run_name or "",
            }
        )

    out_df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(
        f"[viability-eval] rows={len(df):,} coincident_pos={y_coincident.sum():,} "
        f"features={len(features)} out={out_path}"
    )
    for r in out_df.itertuples(index=False):
        print(
            f"  lead={r.lead_h:>6.1f}h | kind={r.kind:<10} | pos={r.pos:>8,} "
            f"| AUC={r.auc:.3f} PRAUC={r.prauc:.3f} Brier={r.brier:.3f}"
        )


if __name__ == "__main__":
    main()
