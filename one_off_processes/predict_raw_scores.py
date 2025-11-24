#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_raw_scores.py — v2
Robust scorer for (global or per-lead) bundles with strict feature alignment,
optional trainer-faithful preprocessing (imputer/clip/scaler), AOI cropping,
and CSV/Parquet I/O at scale.

Examples
--------
# Score a range of leads from a per-lead bundle, write gzipped CSVs
python predict_raw_scores.py \
  --features data/features_eoi.parquet \
  --bundle models/grid_logit_perlead.pkl \
  --lead 24..120 \
  --run-name coral_demo \
  --out-dir results/raw \
  --normalize-lon "-180..180" \
  --area "-10,135,-25,155"

# Score a single lead and also emit a binary flag at thr=0.8
python predict_raw_scores.py \
  --features data/features_eoi.parquet \
  --bundle models/grid_logit_cal.pkl \
  --lead 72 \
  --flag-thr 0.80 \
  --save-parquet
"""

from __future__ import annotations

import argparse, re, pickle, sys
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

import numpy as np
import pandas as pd

# -------------------- parsing helpers --------------------

def parse_leads(s):
    if isinstance(s, int):
        return [s]
    s = str(s).strip()
    m = re.match(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*$", s)
    if m:
        a, b = map(int, m.groups())
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    return [int(x) for x in re.split(r"[,\s]+", s) if x]

def _to_utc_naive(series: pd.Series, fmt: Optional[str]) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)

def norm_lon(s: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "none": return x
    if mode == "0..360": return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # default -180..180

def parse_area(aoi: Optional[str]):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(z.strip()) for z in aoi.split(",")]
    return latN, lonW, latS, lonE

# -------------------- file I/O --------------------

def read_any(path: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    low = str(path).lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols)

def read_csv_chunked(path: str, chunksize: int, usecols: Optional[list[str]] = None) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, compression="infer", low_memory=False, chunksize=int(chunksize), usecols=usecols):
        yield chunk

def write_any(path: str, df: pd.DataFrame):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or low.endswith(".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# -------------------- bundle/model plumbing --------------------

def safe_load(path):
    import joblib
    try:
        obj = joblib.load(path)
        print(f"[load] joblib OK: {path}")
        return obj
    except Exception as e:
        print(f"[warn] joblib failed: {e}  -> trying pickle")
        with open(path, "rb") as f:
            return pickle.load(f)

def unwrap_models(bundle):
    # Common shapes you’ve used across scripts
    if isinstance(bundle, dict):
        for k in ("per_lead_models", "models", "estimators_", "named_estimators_"):
            if k in bundle and bundle[k] is not None:
                print(f"[unwrap] dict['{k}']")
                return bundle[k]
        # Sometimes the dict *is* the mapping
        if all(isinstance(k, (int, str)) for k in bundle.keys()):
            print("[unwrap] bundle looks like a raw mapping {lead -> est}")
            return bundle
    for attr in ("models", "estimators_", "named_estimators_"):
        if hasattr(bundle, attr):
            sub = getattr(bundle, attr)
            print(f"[unwrap] using .{attr}")
            return unwrap_models(sub)
    if hasattr(bundle, "steps"):  # sklearn Pipeline
        try:
            last = bundle.steps[-1][1]
            print("[unwrap] pipeline → last step")
            return unwrap_models(last)
        except Exception:
            pass
    return bundle  # best effort

def get_model_accessor(container):
    if isinstance(container, dict):
        def acc(L):
            for k in (L, str(L), L-1, str(L-1), f"lead_{L}", f"L{L}"):
                if k in container: return container[k]
            return None
        return acc
    elif isinstance(container, (list, tuple)):
        def acc(L):
            i = L - 1
            if 0 <= i < len(container): return container[i]
            if 0 <= L < len(container): return container[L]
            return None
        return acc
    if hasattr(container, "predict_proba"):
        return lambda L: container  # global model fallback
    raise TypeError(f"Cannot interpret bundle type: {type(container)}")

# -------------------- feature building (trainer-faithful) --------------------

def build_matrix(df: pd.DataFrame,
                 bundle: Dict[str, Any],
                 prefer_bundle_features: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Reconstruct X to match trainer space using bundle['features'] if available,
    plus bundle imputer/clip/scaler if present (no fitting, strict transform).
    Returns (X_transformed, feature_names_used).
    """
    feats = list(bundle.get("features", []))
    if not feats or not prefer_bundle_features:
        # fallback: all numeric except meta
        meta = {"time", "lat", "lon", "_time", "_lat", "_lon"}
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta]

    # preserve order and create missing cols as NaN
    Xdf = pd.DataFrame(index=df.index)
    for c in feats:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan
    X = Xdf.to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan

    # impute (trainer stats)
    imp_stats = bundle.get("imputer_stats")
    if imp_stats:
        for j, c in enumerate(feats):
            fill = float(imp_stats.get(c, 0.0))
            m = ~np.isfinite(X[:, j])
            if m.any():
                X[m, j] = fill
    else:
        X = np.where(np.isfinite(X), X, 0.0)

    # clip (trainer quantiles)
    clip_stats = bundle.get("clip_stats")
    if isinstance(clip_stats, dict) and ("lo" in clip_stats) and ("hi" in clip_stats):
        lo = np.asarray(clip_stats["lo"]); hi = np.asarray(clip_stats["hi"])
        if lo.shape == (X.shape[1],) and hi.shape == (X.shape[1],):
            X = np.clip(X, lo, hi)

    # scale (trainer scaler)
    scaler = bundle.get("scaler")
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"[warn] scaler.transform failed ({e}); continuing unscaled")

    return X, feats

def predict_proba_any(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.clip(est.predict_proba(X)[:, 1], 1e-8, 1 - 1e-8)
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
        # map monotonically into (0,1) without fitting anything
        from scipy.special import expit
        return np.clip(expit(z), 1e-8, 1 - 1e-8)
    # last resort: numeric predictions -> [0,1] via min-max on-the-fly
    y = np.asarray(est.predict(X), dtype=float).ravel()
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(y, 0.5, dtype=float)
    return (y - lo) / (hi - lo + 1e-12)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Predict raw per-lead probabilities with robust feature alignment.")
    ap.add_argument("--features", required=True, help="CSV(.gz) or Parquet with meta + feature columns")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle", default=None, help="Trained bundle (joblib/pickle).")
    g.add_argument("--model", default=None, help="Single estimator (joblib/pickle).")
    ap.add_argument("--lead", required=True, help="Lead spec: e.g., 72 or 24..120 or 24,48,72")
    ap.add_argument("--out-dir", required=True, help="Directory to write raw_<run>_lead{L}.csv.gz")
    ap.add_argument("--run-name", required=True, help="Name token for output files")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180",
                    help="Normalize longitudes in OUTPUT")
    ap.add_argument("--area", default=None, help='Optional AOI "latN,lonW,latS,lonE" on OUTPUT coords')
    ap.add_argument("--time-format", default=None, help="Optional strptime for custom time parsing")
    ap.add_argument("--chunk-rows", type=int, default=0, help="If features is CSV, stream in chunks of this size")
    ap.add_argument("--save-parquet", action="store_true", help="Also write Parquet alongside CSV")
    ap.add_argument("--flag-thr", type=float, default=None, help="If set, emit a binary flag column at this threshold")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    leads = parse_leads(args.lead)
    aoi = parse_area(args.area)

    # Load bundle or model
    obj = safe_load(args.bundle or args.model)
    bundle = obj if isinstance(obj, dict) else {}
    container = unwrap_models(obj)
    get_model = get_model_accessor(container)

    # Determine if we can build trainer-faithful X (features/scaler/etc.)
    prefer_bundle_space = isinstance(bundle, dict) and ("features" in bundle)

    # Prepare a writer that standardizes meta columns and crops AOI
    def finalize_meta(df_meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame({
            "time": _to_utc_naive(df_meta["time"], args.time_format),
            "lat": pd.to_numeric(df_meta["lat"], errors="coerce"),
            "lon": norm_lon(pd.to_numeric(df_meta["lon"], errors="coerce"), args.normalize_lon),
            "prob": proba,
        }).dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
        if args.flag_thr is not None:
            out["flag"] = (out["prob"] >= float(args.flag_thr)).astype(int)
        if aoi is not None and not out.empty:
            latN, lonW, latS, lonE = aoi
            out = out.loc[(out["lat"] <= latN) & (out["lat"] >= latS) &
                          (out["lon"] >= lonW) & (out["lon"] <= lonE)].reset_index(drop=True)
        return out

    # Decide reading mode
    fpath = args.features
    is_parquet = str(fpath).lower().endswith((".parquet", ".parq", ".pq", ".pqt"))
    if is_parquet:
        base = read_any(fpath)
        # Basic meta presence
        for need in ("time", "lat", "lon"):
            if need not in base.columns:
                raise ValueError(f"features file missing required column '{need}'")
        # Build X once for all leads (trainer-faithful)
        X, feat_names = build_matrix(base, bundle, prefer_bundle_features=prefer_bundle_space)

        missing: list[int] = []
        for L in leads:
            est = get_model(L)
            if est is None:
                missing.append(L); continue
            try:
                prob = predict_proba_any(est, X)
            except Exception as e:
                print(f"[err] lead={L}: {e}")
                continue
            out = finalize_meta(base[["time", "lat", "lon"]], prob)
            csv_path = Path(out_dir) / f"raw_{args.run_name}_lead{L}.csv.gz"
            write_any(csv_path, out)
            if args.save_parquet:
                write_any(str(csv_path).replace(".csv.gz", ".parquet"), out)
            print(f"[raw] lead={L} -> {csv_path} rows={len(out)}")
        if missing:
            print(f"[warn] no estimator for leads: {missing[:15]}{'...' if len(missing)>15 else ''}")
        return

    # CSV streaming path
    # To avoid huge memory, we score lead-by-lead per chunk.
    if args.chunk_rows <= 0:
        args.chunk_rows = 1_500_000

    # Prepare append writers per lead
    writers: Dict[int, Dict[str, Any]] = {}
    def append_out(L: int, df_chunk: pd.DataFrame):
        csv_path = Path(out_dir) / f"raw_{args.run_name}_lead{L}.csv.gz"
        mode = "w" if not writers.get(L) else "a"
        header = not writers.get(L)
        df_chunk.to_csv(csv_path, index=False, compression="gzip",
                        date_format="%Y-%m-%d %H:%M:%S", mode=mode, header=header)
        writers[L] = {"path": csv_path}

    # Stream chunks
    for i, chunk in enumerate(read_csv_chunked(fpath, args.chunk_rows, usecols=None)):
        if chunk is None or chunk.empty:
            continue
        for need in ("time", "lat", "lon"):
            if need not in chunk.columns:
                if need == "time":
                    raise ValueError("CSV chunk missing 'time' column")
                chunk[need] = np.nan

        # Build trainer-faithful X for this chunk
        Xc, _ = build_matrix(chunk, bundle, prefer_bundle_features=prefer_bundle_space)

        for L in leads:
            est = get_model(L)
            if est is None:
                # don’t spam per chunk; warn once at the end
                continue
            try:
                prob = predict_proba_any(est, Xc)
            except Exception as e:
                print(f"[err] chunk={i+1} lead={L}: {e}")
                continue
            out = finalize_meta(chunk[["time", "lat", "lon"]], prob)
            if not out.empty:
                append_out(L, out)
        print(f"[raw] chunk {i+1} scored for leads {leads[:6]}{'...' if len(leads)>6 else ''}")

    # Note: for CSV stream we only produce CSV.gz (Parquet append is unsafe).
    # If Parquet is needed from CSV, run a after-pass converter.

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # allow piping to tools like head without ugly tracebacks
        try: sys.stdout.close()
        except Exception: pass
        try: sys.stderr.close()
        except Exception: pass