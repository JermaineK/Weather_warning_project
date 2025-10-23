#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- lightweight I/O (CSV/GZ + Parquet) ----------
def read_any(path: str, columns=None) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet without parsing/altering 'time'.
    Pass a list of `columns` to load only what's needed.
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        # pandas supports column-level reads for parquet
        return pd.read_parquet(path, columns=columns if columns else None)
    # CSV/GZ: usecols supports callables or list; we handle list here for speed
    if columns:
        return pd.read_csv(path, usecols=lambda c: c in set(columns), low_memory=False)
    return pd.read_csv(path, low_memory=False)

# ---------- tiny progress helpers ----------
def pbar(iterable, total, title=""):
    done = 0
    for x in iterable:
        yield x
        done += 1
        width = 30
        frac  = min(1.0, done/total if total else 1.0)
        filled = int(width*frac)
        bar = "█"*filled + "·"*(width-filled)
        print(f"\r{title} [{bar}] {done}/{total}", end="", flush=True)
    print("", flush=True)

# ---------- label builders ----------
def future_max_label_by_point_samples(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """Sample-count window (assumes 1 row per hour per cell). Excludes current hour."""
    base = df[[target, "lat", "lon"]].reset_index(drop=True)
    base[target] = base[target].astype(int)
    def _rev_roll(g: pd.DataFrame) -> pd.Series:
        s = g[target]
        r = s.iloc[::-1].rolling(window=hours, min_periods=1).max().shift(1)
        return r.iloc[::-1].fillna(0).astype(int)
    out = base.groupby(["lat","lon"], sort=False, group_keys=False).apply(_rev_roll)
    return out.reset_index(drop=True).to_numpy()

def future_max_label_by_point_timeaware(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """
    Time-aware window: for each (lat,lon) and row t_i, 1 if any future row within (t_i, t_i+hours] has target=1.
    Robust to tz-aware times and missing hours. Uses per-cell sort + searchsorted on int64 nanoseconds.
    """
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)  # 1h in ns

    for (_, _), g in df.groupby(["lat","lon"], sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y   = g[target].astype(int).to_numpy()

        # to int64 ns, regardless of timezone; parse here (we intentionally didn't parse on read)
        t_series = pd.to_datetime(g["time"], errors="coerce", utc=True)
        t_ns = t_series.view("int64").to_numpy()

        order = np.argsort(t_ns, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        t_sorted = t_ns[order]
        y_sorted = y[order]

        ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
        ps[1:] = np.cumsum(y_sorted)

        t_end_sorted = t_sorted + hour_ns
        end_pos = np.searchsorted(t_sorted, t_end_sorted, side="right")

        any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
        any_future = any_future_sorted[inv_order].astype(np.int8)

        out[idx] = any_future

    return out

# ---------- checkpoint utilities ----------
def _ckpt_paths(ckpt_dir: Path):
    return {"state": ckpt_dir / "state.json", "probs": ckpt_dir / "probs.npy"}

def load_checkpoint(ckpt_dir: Path, total_rows: int):
    if not ckpt_dir or not ckpt_dir.exists():
        return 0, None
    paths = _ckpt_paths(ckpt_dir)
    if not paths["state"].exists() or not paths["probs"].exists():
        return 0, None
    try:
        state = json.loads(paths["state"].read_text())
        done_rows = int(state.get("done_rows", 0))
        p = np.load(paths["probs"])
        if done_rows == p.shape[0] and done_rows <= total_rows:
            print(f"Resuming from checkpoint: {done_rows}/{total_rows} rows already scored.", flush=True)
            return done_rows, p
    except Exception:
        pass
    return 0, None

def save_checkpoint(ckpt_dir: Path, p_so_far: np.ndarray, done_rows: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = _ckpt_paths(ckpt_dir)
    np.save(paths["probs"], p_so_far)
    paths["state"].write_text(json.dumps({"done_rows": int(done_rows)}))

# ---------- model bundle helpers ----------
def _load_bundle_any(path: str):
    m = joblib.load(path)
    if isinstance(m, dict):
        # Prefer calibrated model key if present; fall back to 'model' then 'model-out'
        model = m.get("model", None) or m.get("model-out", None)
        scaler = m.get("scaler", None)
        feats  = list(m.get("features", []))
        imp_stats = m.get("imputer_stats", None)
        if model is None:
            raise ValueError("Model bundle lacks a usable estimator ('model' or 'model-out').")
        if not feats:
            raise ValueError("Model bundle lacks 'features'.")
        return model, scaler, feats, imp_stats
    # Plain estimator
    return m, getattr(m, "scaler_", None), list(getattr(m, "features_", [])), None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Lead-time evaluation (GKA) with progress, checkpoints, and time-aware labels")
    ap.add_argument("--labelled", required=True, help="CSV(.gz) or Parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, required=True)
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--window-mode", choices=["time","samples"], default="time",
                    help="time: exact hours via timestamps (robust). samples: fixed sample count per cell.")
    args = ap.parse_args()

    print("== Lead-Time Evaluation (GKA model) ==", flush=True)
    print(f"Labelled : {args.labelled}")
    print(f"Model    : {args.model}")
    print(f"Target   : {args.target}")
    print(f"Leads    : {args.lead_hours}")

    model, scaler, USE, imp_stats = _load_bundle_any(args.model)
    if not USE:
        raise ValueError("Model does not expose 'features'; load a packaged dict model.")

    # Pull exactly what we need (fast for Parquet; safe for CSV)
    need_cols = {"time","lat","lon", args.target, *USE}
    df = read_any(args.labelled, columns=list(need_cols))
    nrows = len(df)
    pos0  = int(pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).sum())
    print(f"Rows evaluated: {nrows:,}  Positives (coincident {args.target}): {pos0:,}", flush=True)

    # Impute features BEFORE scaling (prefer trainer stats if available)
    Xnum = df[USE].copy()
    Xv = Xnum.to_numpy(dtype=float, copy=True)
    Xv[~np.isfinite(Xv)] = np.nan  # replace ±inf → NaN

    if imp_stats:
        for j, c in enumerate(USE):
            fill = float(imp_stats.get(c, np.nan))
            m = ~np.isfinite(Xv[:, j])
            if m.any():
                Xv[:, j] = np.where(m, fill, Xv[:, j])
        print("[IMPUTE] used trainer stats", flush=True)
    else:
        med = np.nanmedian(Xv, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        m = ~np.isfinite(Xv)
        if m.any():
            Xv[m] = np.take(med, np.where(m)[1])
        print("[IMPUTE] median (no trainer stats found)", flush=True)

    # checkpoint resume
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    start_row, p_all_prev = load_checkpoint(ckpt_dir, nrows)
    p_all = p_all_prev if p_all_prev is not None else np.empty((0,), dtype=float)

    # chunked scoring
    if p_all.shape[0] < nrows:
        print("Scoring probabilities…", flush=True)
        remaining = nrows - p_all.shape[0]
        steps = math.ceil(remaining / args.chunk_rows) or 1
        # Iterate over raw index positions to avoid re-slicing the dataframe
        for i0 in pbar(range(p_all.shape[0], nrows, args.chunk_rows), total=steps, title="  • chunks"):
            i1 = min(i0 + args.chunk_rows, nrows)
            X_chunk = Xv[i0:i1]
            if scaler is not None:
                try:
                    X_chunk = scaler.transform(X_chunk)
                except Exception:
                    # If scaler mismatched, fall back to unscaled chunk
                    pass
            p_chunk = model.predict_proba(X_chunk)[:, 1]
            p_all = np.concatenate([p_all, p_chunk])
            if ckpt_dir:
                save_checkpoint(ckpt_dir, p_all, i1)
    else:
        print("Probabilities already scored (checkpoint).", flush=True)

    # coincident metrics
    y0 = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).to_numpy()
    try:
        print(f"[COINCIDENT] AUC={roc_auc_score(y0, p_all):.3f}  "
              f"PRAUC={average_precision_score(y0, p_all):.3f}  "
              f"Brier={brier_score_loss(y0, p_all):.3f}", flush=True)
    except Exception as e:
        print(f"[COINCIDENT] metric error: {e}", flush=True)

    # choose labeler
    labeler = future_max_label_by_point_timeaware if args.window_mode == "time" \
              else future_max_label_by_point_samples

    # lead metrics
    labels_per_h = {}
    for h in args.lead_hours:
        print(f"Preparing lead +{h}h labels …", flush=True)
        yh = labeler(df, args.target, hours=h)
        posh = int(yh.sum())
        auc  = roc_auc_score(yh, p_all)
        ap   = average_precision_score(yh, p_all)
        br   = brier_score_loss(yh, p_all)
        labels_per_h[h] = yh
        print(f"Lead +{h}h  →  AUC={auc:.3f}  PRAUC={ap:.3f}  Brier={br:.3f}  Pos={posh:,}/{nrows:,}", flush=True)

    # sanity panel (overlap between 24h and 48h if both present)
    if len(labels_per_h) >= 2 and 24 in labels_per_h and 48 in labels_per_h:
        y24, y48 = labels_per_h[24], labels_per_h[48]
        inter = np.logical_and(y24==1, y48==1).sum()
        union = np.logical_or (y24==1, y48==1).sum()
        jacc  = inter / max(1, union)
        print(f"[Sanity] Overlap 24h vs 48h → Jaccard={jacc:.3f}  "
              f"Pos24={y24.sum():,}  Pos48={y48.sum():,}", flush=True)

    if ckpt_dir:
        (ckpt_dir / "_done.txt").write_text("ok")

if __name__ == "__main__":
    main()