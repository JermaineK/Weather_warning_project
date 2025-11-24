#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_stage_score.py

Stage 1 (context): score all rows → risk_ctx
Keep the top quantile per hour (vectorized) → ctx_keep ∈ {0,1}

Stage 2 (pregen): score only kept rows → risk_pregen (others 0)

Extras:
- Works with CSV or Parquet labelled files
- Accepts either classic bundles {"model","scaler","features"} or per-lead bundles
  {"per_lead_models": {lead: sklearn-Pipeline, ...}, "features":[...]} with --lead
- Vectorized per-hour quantile keeping (no misaligned groupby-apply)
- Preserves time/lat/lon; writes risk_ctx, ctx_keep, risk_pregen, risk_final

Usage
-----
python two_stage_score.py \
  --labelled data/grid_labelled_FMA_gka.csv.gz \
  --model-context models/context.pkl \
  --model-pregen  models/pregen.pkl \
  --context-quantile 0.90 \
  --lead 24 \
  --out results/two_stage_scored.csv.gz
"""

import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path

# ---------- I/O helpers ----------

def read_any(path, usecols=None, parse_dates=None):
    p = str(path).lower()
    if p.endswith((".parquet",".pq",".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path,
                       usecols=usecols if usecols else None,
                       parse_dates=parse_dates if parse_dates else None,
                       low_memory=False)

def write_any(path, df):
    p = str(path).lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if p.endswith((".parquet",".pq",".pqt")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ---------- model helpers ----------

def load_bundle(bundle_path, wanted_lead=None):
    """
    Returns (predict_proba_fn, features)
    - predict_proba_fn(X: np.ndarray) -> np.ndarray of shape (n,)
    - features: list of feature column names to extract
    """
    b = joblib.load(bundle_path)

    # Per-lead bundle: pick a specific lead
    if isinstance(b, dict) and "per_lead_models" in b:
        if wanted_lead is None:
            raise ValueError(f"{bundle_path}: per-lead bundle requires --lead")
        models = b["per_lead_models"]
        if int(wanted_lead) not in models:
            raise ValueError(f"{bundle_path}: lead {wanted_lead} not found in bundle")
        pipe = models[int(wanted_lead)]
        feats = list(b.get("features", []))
        def proba(X): return pipe.predict_proba(X)[:, 1]
        return proba, feats

    # Classic bundle: {"model","scaler","features"}
    if not all(k in b for k in ("model","scaler","features")):
        raise ValueError(f"{bundle_path}: unsupported bundle keys {list(b.keys())}")
    model, scaler, feats = b["model"], b["scaler"], list(b["features"])
    def proba(X): return model.predict_proba(scaler.transform(X))[:, 1]
    return proba, feats

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Two-stage scoring: context filter → pregen scoring.")
    ap.add_argument("--labelled", required=True, help="CSV(.gz)/Parquet with feature superset")
    ap.add_argument("--model-context", required=True, help="joblib bundle (per-lead or classic)")
    ap.add_argument("--model-pregen",  required=True, help="joblib bundle (per-lead or classic)")
    ap.add_argument("--context-quantile", type=float, default=0.90, help="Keep top-q per hour from risk_ctx")
    ap.add_argument("--lead", type=int, default=None, help="Lead to select if bundles are per-lead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap (debug/quick)")
    args = ap.parse_args()

    # Load bundles
    ctx_proba, ctx_feats = load_bundle(args.model_context, wanted_lead=args.lead)
    pre_proba, pre_feats = load_bundle(args.model_pregen,  wanted_lead=args.lead)

    need = {"time","lat","lon", *ctx_feats, *pre_feats}
    df = read_any(args.labelled, usecols=lambda c: c in need, parse_dates=["time"])
    if args.limit:
        df = df.iloc[:args.limit].copy()

    # Basic hygiene
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]  = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # Score stage 1 (context) on all rows
    Xc = df[ctx_feats].to_numpy(float)
    df["risk_ctx"] = ctx_proba(Xc).astype("float32")

    # Keep top quantile per hour (vectorized; exact-hour groups)
    # Use floor("H") just in case time stamps aren't perfectly aligned
    hour = pd.to_datetime(df["time"]).dt.floor("H")
    qthr = df.groupby(hour, sort=False)["risk_ctx"].transform(lambda s: s.quantile(args.context_quantile))
    df["ctx_keep"] = (df["risk_ctx"] >= qthr).astype("int8")

    # Stage 2 (pregen) only on kept rows
    df["risk_pregen"] = np.zeros(len(df), dtype="float32")
    keep_mask = df["ctx_keep"].to_numpy(dtype=bool)
    if keep_mask.any():
        Xp = df.loc[keep_mask, pre_feats].to_numpy(float)
        df.loc[keep_mask, "risk_pregen"] = pre_proba(Xp).astype("float32")

    # Combined score for downstream; simple pass-through of stage2 (0 outside keep)
    # You can swap for product or min if you want a stricter gate:
    df["risk_final"] = df["risk_pregen"].astype("float32")

    write_any(args.out, df)
    kept = int(keep_mask.sum())
    print(f"[two-stage] wrote {args.out} | rows={len(df):,} | kept (context)={kept:,} ({kept/len(df):.1%})")
    print(f"  ctx_feats={len(ctx_feats)} pre_feats={len(pre_feats)}  lead={args.lead if args.lead is not None else '(n/a)'}")

if __name__ == "__main__":
    main()