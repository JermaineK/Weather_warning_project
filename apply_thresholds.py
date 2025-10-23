#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_thresholds.py  (v4)
Score a labelled grid with a saved model and emit alerts by threshold.

Inputs:
  --labelled  CSV(.gz) or Parquet with at least: time, lat, lon + numeric features
  --model     joblib/pickle file. Accepts:
                • {"model","features","scaler"} dict (calibrated OK)
                • sklearn Pipeline/CalibratedClassifierCV
                • (model, scaler, features) tuple
  --lead-hours  (metadata only)
  --thr       probability threshold
  --out       output CSV(.gz)/Parquet: time, lat, lon, prob, alert

Options:
  --normalize-lon {none,-180..180,0..360}  (default: none)  apply to input lon (and output)
  --area "latN,lonW,latS,lonE"             crop to AOI after lon normalization
  --debug                                  print ranges and sample stats
"""

import os, argparse, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- I/O helpers ----------

def read_any(path: str) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet without parsing/altering 'time'.
    (Pipeline will sanitize timestamps later.)
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="infer", low_memory=False)

def write_any(df: pd.DataFrame, path: str) -> None:
    low = path.lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
        return
    comp = "gzip" if low.endswith(".gz") else "infer"
    df.to_csv(path, index=False, compression=comp)

# ---------- model helpers ----------

def robust_load_model(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def unpack_model_bundle(bundle):
    """
    Returns (model, scaler_or_None, features_or_None)
    Prefers 'model' (post-calibration) then 'model-out' for back-compat.
    """
    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("model-out") or bundle.get("estimator")
        scaler = bundle.get("scaler")
        feats  = bundle.get("features") or bundle.get("feats")
        if model is None:
            raise ValueError("Loaded bundle lacks a model (expected 'model' or 'model-out').")
        return model, scaler, feats
    if isinstance(bundle, (list, tuple)) and len(bundle) == 3:
        return bundle
    return bundle, None, None

def infer_features(df, target_cols=("pregen","storm","near_storm")):
    drop = {"time","lat","lon", *[c for c in target_cols if c in df.columns]}
    feats = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
        else:
            # accept columns that are ~numeric after coercion
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.95:
                df[c] = s
                feats.append(c)
    return feats

def build_matrix(df, feats, scaler):
    from sklearn.impute import SimpleImputer
    X = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float, copy=False)
    X = SimpleImputer(strategy="median").fit_transform(X)
    return scaler.transform(X) if scaler is not None else X

# ---------- geo helpers ----------

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # "-180..180"
    return ((x + 180) % 360) - 180

def _parse_area(aoi: str | None):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Apply threshold to model probabilities.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lead-hours", type=int, default=0)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="normalize longitudes before scoring/writing")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization')
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.labelled): raise FileNotFoundError(args.labelled)
    if not os.path.exists(args.model):    raise FileNotFoundError(args.model)

    print(f"[APPLY v4] Labelled : {args.labelled}")
    print(f"[APPLY v4] Model    : {args.model}")
    print(f"[APPLY v4] Lead (h) : {args.lead_hours}")
    print(f"[APPLY v4] Thr      : {args.thr}")
    print(f"[APPLY v4] Out      : {args.out}")

    # 1) load (no time parsing here)
    df = read_any(args.labelled)

    # 2) keep ALL rows by default; only adjust lon and optional AOI
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("Input must contain 'lat' and 'lon' columns.")
    df["lon"] = _norm_lon(df["lon"], args.normalize_lon)
    aoi = _parse_area(args.area)
    if aoi:
        latN, lonW, latS, lonE = aoi
        before = len(df)
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)
        print(f"[APPLY v4] AOI crop: kept {len(df):,}/{before:,} rows")

    # 3) basic clean for features (don’t touch 'time')
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4) sort for reproducibility
    sort_cols = [c for c in ["time","lat","lon"] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True, ignore_index=True)

    # 5) model bundle
    bundle = robust_load_model(args.model)
    model, scaler, feats = unpack_model_bundle(bundle)
    if feats is None:
        feats = infer_features(df)
        print(f"[APPLY v4] Features inferred ({len(feats)}): {feats[:6]}{' ...' if len(feats)>6 else ''}")
    else:
        print(f"[APPLY v4] Features from model ({len(feats)})")

    if not feats:
        raise ValueError("No features available to score. Check your labelled file columns.")

    # 6) build matrix and score
    X = build_matrix(df, feats, scaler)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        from scipy.special import expit
        prob = expit(model.decision_function(X))

    out = pd.DataFrame({
        "time": df["time"].values if "time" in df.columns else np.nan,
        "lat":  pd.to_numeric(df["lat"], errors="coerce").values,
        "lon":  pd.to_numeric(df["lon"], errors="coerce").values,
        "prob": prob.astype("float64", copy=False),
    })
    out["alert"] = (out["prob"] >= args.thr).astype(int)

    if args.debug:
        try:
            tmin = pd.to_datetime(out["time"], errors="coerce").min()
            tmax = pd.to_datetime(out["time"], errors="coerce").max()
            print(f"[APPLY v4][debug] time range: {tmin} → {tmax}")
        except Exception:
            pass
        with np.errstate(all="ignore"):
            lon_min = np.nanmin(out["lon"]); lon_max = np.nanmax(out["lon"])
        print(f"[APPLY v4][debug] lon range: {lon_min:.3f} .. {lon_max:.3f}")
        print(f"[APPLY v4][debug] alerts: {int(out['alert'].sum()):,}/{len(out):,}  (cov={out['alert'].mean():.3f})")

    # 7) write CSV/GZ or Parquet
    write_any(out, args.out)
    print(f"[APPLY v4] Wrote {args.out} | alerts: {int(out['alert'].sum()):,}/{len(out):,}")

if __name__ == "__main__":
    main()