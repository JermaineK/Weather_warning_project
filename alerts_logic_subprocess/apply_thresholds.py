#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
apply_thresholds.py
Apply a trained (and optionally calibrated) grid model to a labelled CSV/Parquet,
producing viability-style BASE alerts with a probability column and a binary flag using a threshold.

Key features:
- Per-lead aware: uses bundle['per_lead_models'][lead] when present; otherwise falls back to bundle['model'].
- Trainer-faithful inference: exact feature alignment, trainer imputer stats, optional clip, and scaler.
- Big-file friendly: chunked CSV reads; Parquet is column-projected and done in one go.
- Niceties: customizable prob/flag col names, optional AOI crop, safe time parsing.

Extras (optional):
- --strict-features to error if any feature is missing
- --thr-map to pass a JSON string or file with per-lead thresholds (e.g. {"24":0.08,"72":0.04})
- Auto-detect internal scaler in the estimator (Pipeline) to avoid double scaling.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

# ------------- I/O helpers -------------

def read_any(path: str, usecols=None) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        try:
            return pd.read_parquet(path, columns=usecols if usecols else None)
        except Exception:
            # If some requested columns are missing, intersect with available schema
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(path)
                available = set(schema.names)
                cols = [c for c in usecols or [] if c in available] or None
                return pd.read_parquet(path, columns=cols)
            except Exception:
                return pd.read_parquet(path)
    try:
        return pd.read_csv(path, compression="infer", usecols=usecols if usecols else None, low_memory=False)
    except ValueError:
        # Fallback: read all and subset later if CSV columns mismatch
        return pd.read_csv(path, compression="infer", low_memory=False)

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower()==".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ------------- misc helpers -------------

def parse_time(series: pd.Series, fmt: str | None):
    if np.issubdtype(series.dtype, np.datetime64):
        # Already tz-naive; enforce naive UTC if tz-aware
        s = pd.to_datetime(series, utc=True, errors="coerce")
        return s.dt.tz_localize(None)
    raw = series.astype(str).str.strip().str.replace("Z","",regex=False)
    t = pd.to_datetime(raw, utc=True, errors="coerce") if not fmt else pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    return t.dt.tz_localize(None)

def normalize_lon_series(s: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # default -180..180

def parse_area(aoi: str | None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def _strip_choice(val: str) -> str:
    """Normalize choice strings to allow leading/trailing spaces (YAML quirks)."""
    return str(val).strip()

def _preprocess_norm(argv: list[str]) -> list[str]:
    """
    Allow --normalize-lon values that look like options (e.g., -180..180) by
    rewriting them to --normalize-lon=<value> before argparse runs.
    """
    out = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--normalize-lon" and i + 1 < len(argv):
            val = argv[i + 1]
            out.append(f"--normalize-lon={val}")
            skip = True
        else:
            out.append(tok)
    return out

def _estimator_has_internal_scaler(est) -> bool:
    # Detect a Pipeline with a StandardScaler-ish stage
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(est, Pipeline):
            names = [name for name, _ in est.steps]
            return any("scaler" in name.lower() for name in names)
    except Exception:
        pass
    return False

def _load_thr_map(spec: str | None) -> dict[int, float]:
    if not spec:
        return {}
    try:
        # JSON string first
        obj = json.loads(spec)
    except json.JSONDecodeError:
        # Then treat as a file path
        txt = Path(spec).read_text(encoding="utf-8")
        obj = json.loads(txt)
    out = {}
    for k, v in obj.items():
        out[int(k)] = float(v)
    return out


def _parse_feature_arg(tokens) -> list[str]:
    if not tokens:
        return []
    out: list[str] = []
    for tok in tokens:
        for part in str(tok).replace(",", " ").split():
            if part.strip():
                out.append(part.strip())
    return out


def _load_metrics_features(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        meta = json.loads(p.read_text())
        feats = meta.get("features", [])
        return [str(f) for f in feats]
    except Exception:
        return []

# ------------- feature pipeline -------------

def build_matrix(df: pd.DataFrame,
                 features: list[str],
                 imputer_stats: dict[str, float] | None,
                 clip_stats: dict[str, np.ndarray] | None,
                 scaler,
                 allow_scale: bool = True,
                 strict_features: bool = False) -> np.ndarray:
    """
    Create X with exactly the columns in `features` and in that order.
    Missing feature columns are created as NaN, then imputed using bundle stats (or 0.0 fallback).
    Then apply clip (if present) and (optionally) scale.
    """
    if strict_features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Strict feature check failed; missing: {missing[:12]}{' ...' if len(missing)>12 else ''}")

    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan

    X = Xdf.to_numpy(dtype=float, copy=True)
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
    if clip_stats and isinstance(clip_stats, dict) and ("lo" in clip_stats) and ("hi" in clip_stats):
        lo = np.asarray(clip_stats["lo"]); hi = np.asarray(clip_stats["hi"])
        if lo.shape == (X.shape[1],) and hi.shape == (X.shape[1],):
            X = np.clip(X, lo, hi)

    # Scale
    if allow_scale and scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # Continue unscaled if shapes drifted
            pass

    return X

def predict_prob(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.clip(est.predict_proba(X)[:, 1], 1e-8, 1 - 1e-8)
    # fallback for linear models without predict_proba
    try:
        from scipy.special import expit
        return np.clip(expit(est.decision_function(X)), 1e-8, 1 - 1e-8)
    except Exception:
        raise ValueError("Estimator lacks predict_proba/decision_function; cannot obtain probabilities.")

# ------------- args -------------

def parse_args():
    ap = argparse.ArgumentParser(description="Apply thresholds to produce BASE alerts from a labelled grid.")
    ap.add_argument("--labelled",
                    default="data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet",
                    help="CSV(.gz) or Parquet with features + meta (time,lat,lon,row_id).")
    ap.add_argument("--model",
                    default="models/viability_model.pkl",
                    help="Trained bundle .pkl (possibly calibrated or per-lead).")
    ap.add_argument("--metrics-json",
                    default="models/viability_model_metrics.json",
                    help="Optional metrics JSON with feature list (used when model is not a dict bundle).")
    ap.add_argument("--features", nargs="+", default=None,
                    help="Optional explicit feature list (space or comma separated).")
    ap.add_argument("--lead-hours", type=int, default=None,
                    help="Optional lead in hours (used for per-lead models/threshold maps).")
    ap.add_argument("--thr", type=float, required=False, default=0.15,
                    help="Global probability threshold (overridden by --thr-map).")
    ap.add_argument("--thr-map", default=None,
                    help='JSON string or path with per-lead thresholds, e.g. {"24":0.08,"72":0.04}')
    ap.add_argument("--out", required=False, default=None, help="Output CSV(.gz)/Parquet with prob + flag.")
    ap.add_argument("--prob-col", default="prob_viable", help="Probability column name (default: prob_viable).")
    ap.add_argument("--flag-col", default="alert_base", help="Binary flag column name (default: alert_base).")
    ap.add_argument(
        "--normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        type=_strip_choice,
        help="Normalize lon in the OUTPUT only (default: -180..180). AOI must match this frame.",
    )
    ap.add_argument("--time-format", default=None, help="Optional strftime to parse time if non-standard.")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" on OUTPUT coords.')
    ap.add_argument("--chunk-rows", type=int, default=1_500_000, help="Chunk size for large CSV inputs.")
    ap.add_argument("--strict-features", action="store_true", help="Error if any required feature is missing.")
    ap.add_argument("--force-csv-out-when-chunking", action="store_true",
                    help="When input is CSV and out is Parquet, force a CSV(.gz) sibling. Otherwise error.")
    ap.add_argument("--run-name", default=None, help="Optional run name for default outputs (alerts_<run>_base.parquet).")
    ap.add_argument("--passthrough-cols", default="row_id,ilat,ilon",
                    help="Comma list of extra columns to keep if present (e.g., row_id,ilat,ilon).")
    argv = _preprocess_norm(sys.argv[1:])
    return ap.parse_args(argv)

# ------------- main -------------

def main():
    args = parse_args()

    passthrough = [c.strip() for c in str(args.passthrough_cols).split(",") if c.strip()]
    feats_override = _parse_feature_arg(args.features)
    feats_from_metrics = _load_metrics_features(args.metrics_json)

    # Load bundle
    bundle = joblib.load(args.model)
    per_lead = {}
    imp = None
    clip = None
    scaler = None

    if isinstance(bundle, dict):
        feats = list(bundle.get("features", feats_override or feats_from_metrics))
        scaler = bundle.get("scaler")
        imp = bundle.get("imputer_stats", None)
        clip = bundle.get("clip_stats", None)
        per_lead = bundle.get("per_lead_models", {}) or {}
        est = bundle.get("model")
        if isinstance(per_lead, dict) and args.lead_hours is not None and (args.lead_hours in per_lead):
            est = per_lead[args.lead_hours]
            print(f"[APPLY] Using per-lead estimator for +{args.lead_hours}h.")
        else:
            print(f"[APPLY] Using global estimator.")
    else:
        est = bundle
        feats = feats_override or feats_from_metrics
        if not feats:
            raise SystemExit("Model bundle lacks features; provide --features or a --metrics-json with 'features'.")
        print("[APPLY] Using plain estimator (features from metrics/override).")

    if est is None:
        raise SystemExit("Model estimator missing from bundle.")
    if not feats:
        raise SystemExit("Feature list empty; provide --features or --metrics-json with 'features'.")

    # If estimator is a Pipeline that already scales, skip external scaling
    allow_external_scale = not _estimator_has_internal_scaler(est)

    # Threshold selection (map overrides)
    thr_map = _load_thr_map(args.thr_map)
    if args.lead_hours is not None and args.lead_hours in thr_map:
        thr = float(thr_map[args.lead_hours])
        print(f"[APPLY] Threshold from map for +{args.lead_hours}h: {thr}")
    else:
        if args.thr is None and not thr_map:
            print("[ERROR] Provide --thr or a --thr-map with this lead.", file=sys.stderr)
            sys.exit(2)
        thr = float(args.thr if args.thr is not None else list(thr_map.values())[0])

    # Columns to load
    need_cols = ["time", "lat", "lon", *feats, *passthrough]

    path = args.labelled
    low = path.lower()
    aoi = parse_area(args.area)

    # Resolve default out path
    out_path_cli = args.out
    if out_path_cli is None:
        if args.run_name:
            out_path_cli = f"results/alerts/alerts_{args.run_name}_base.parquet"
        else:
            out_path_cli = "results/alerts/alerts_base.parquet"

    def finalize_and_write(df_meta: pd.DataFrame, probs: np.ndarray):
        keep_cols = [c for c in ["time","lat","lon", *passthrough] if c in df_meta.columns]
        out = df_meta[keep_cols].copy()
        out[args.prob_col] = probs
        out[args.flag_col] = (out[args.prob_col] >= thr).astype(int)

        # drop invalid meta
        out["time"] = parse_time(out["time"], args.time_format)
        out["lat"]  = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"]  = normalize_lon_series(out["lon"], args.normalize_lon)
        out = out.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

        # AOI crop (optional)
        if aoi:
            latN, lonW, latS, lonE = aoi
            out = out.loc[(out["lat"] <= latN) & (out["lat"] >= latS) &
                          (out["lon"] >= lonW) & (out["lon"] <= lonE)].reset_index(drop=True)
        return out

    total_rows = 0

    if low.endswith((".parquet",".parq",".pq")):
        df = read_any(path, usecols=need_cols)
        missing_feats = [c for c in feats if c not in df.columns]
        for c in missing_feats:
            df[c] = np.nan
        if missing_feats:
            print(f"[warn] {len(missing_feats)} feature(s) missing from input parquet; filling with imputer stats/0.0: {missing_feats[:8]}{' ...' if len(missing_feats)>8 else ''}")
        X = build_matrix(df, feats, imp, clip, scaler,
                         allow_scale=allow_external_scale,
                         strict_features=args.strict_features)
        prob = predict_prob(est, X)
        out = finalize_and_write(df, prob)
        write_any(out_path_cli, out)
        total_rows = len(out)
        print(f"[APPLY] Wrote {len(out):,} rows -> {out_path_cli}")
        return

    # CSV: chunked
    out_path = Path(out_path_cli); out_path.parent.mkdir(parents=True, exist_ok=True)
    wants_parquet = out_path.suffix.lower() in (".parquet", ".parq", ".pq")
    if wants_parquet and not args.force_csv_out_when_chunking:
        print("[ERROR] Parquet output with chunked CSV input is unsafe. "
              "Re-run without Parquet out or pass --force-csv-out-when-chunking.", file=sys.stderr)
        sys.exit(2)

    # If forcing CSV out, adjust destination to .csv.gz sibling (document loudly)
    if wants_parquet and args.force_csv_out_when_chunking:
        dst = out_path.with_suffix(out_path.suffix + ".csv.gz")
        print(f"[APPLY] Input is CSV and output requested Parquet; writing streamed CSV instead: {dst}")
        out_path = dst

    is_gz = out_path.suffix.lower()==".gz" or out_path.name.lower().endswith(".csv.gz")
    first = True

    for i, df in enumerate(pd.read_csv(path, compression="infer", chunksize=int(args.chunk_rows), low_memory=False)):
        keep = [c for c in ["time","lat","lon"] if c in df.columns] + [c for c in feats if c in df.columns]
        keep += [c for c in passthrough if c in df.columns]
        for req in ["time","lat","lon"]:
            if req not in df.columns:
                df[req] = np.nan
                if req not in keep:
                    keep.insert(0, req)
        df_min = df[keep].copy()

        X = build_matrix(df_min, feats, imp, clip, scaler,
                         allow_scale=allow_external_scale,
                         strict_features=args.strict_features)
        prob = predict_prob(est, X)
        out = finalize_and_write(df_min, prob)
        total_rows += len(out)

        mode = "wt" if first else "at"
        header = first
        first = False

        out.to_csv(out_path, mode=mode, index=False, header=header,
                   compression=("gzip" if is_gz else "infer"),
                   date_format="%Y-%m-%d %H:%M:%S")
        print(f"[APPLY] chunk {i+1}: wrote {len(out):,} rows")

    print(f"[APPLY] Done -> {out_path} | total rows written: {total_rows:,}")

if __name__ == "__main__":
    main()
