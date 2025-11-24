#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_leadtime_grid_progress.py

Lead-time evaluation for large labelled grids with:

  • Training-faithful feature alignment:
      - uses bundle['features'], 'imputer_stats', 'clip_stats', 'scaler'
  • Global AND per-lead model support:
      - bundle may have a global 'model' and/or 'per_lead_models'{lead -> est}
  • Strict future labels:
      - window-mode="time": any event in (t, t+H] using timestamps
      - window-mode="samples": any event in the next H samples per (lat, lon)
  • Chunked scoring + checkpoints:
      - resumes probability scoring from disk per lead or global

CLI (key args):
  --labelled        CSV(.gz)/Parquet with at least: time, lat, lon, target, features
  --model           joblib bundle dict with model, features, scaler, imputer_stats[, clip_stats, per_lead_models]
  --target          one of: storm, near_storm, pregen
  --lead-hours      list of leads to evaluate: e.g. 6 12 24 48
  --chunk-rows      rows per chunk for predict_proba (default 2e6)
  --checkpoint-dir  folder for *.json + *.npy progress files (optional)
  --window-mode     'time' (timestamp-aware, default) or 'samples' (fixed-count per cell)
"""

import argparse
import json
import math
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# ---------- lightweight I/O (CSV/GZ + Parquet) ----------
def read_any(path: str, columns=None) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet without forcing time parsing.
    Pass a list of `columns` to load only what's needed.
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        # Parquet supports column-level reads; if some columns are missing,
        # we’ll add them later (don’t fail here).
        try:
            return pd.read_parquet(path, columns=columns if columns else None)
        except Exception:
            # Fallback: read all, add missing later
            return pd.read_parquet(path)
    if columns:
        cols_set = set(columns)
        return pd.read_csv(path, usecols=lambda c: c in cols_set, low_memory=False)
    return pd.read_csv(path, low_memory=False)


# ---------- tiny progress helpers ----------
def pbar(iterable, total, title=""):
    done = 0
    for x in iterable:
        yield x
        done += 1
        width = 30
        frac = min(1.0, done / total if total else 1.0)
        filled = int(width * frac)
        bar = "█" * filled + "·" * (width - filled)
        print(f"\r{title} [{bar}] {done}/{total}", end="", flush=True)
    print("", flush=True)


# ---------- label builders ----------
def future_max_label_by_point_samples(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """
    Sample-count window (assumes ≈1 row per hour per cell).
    For each (lat,lon), label = max(target) over the next `hours` samples,
    excluding the current sample (strict future).
    """
    base = df[[target, "lat", "lon"]].reset_index(drop=True)
    base[target] = base[target].astype(int)

    def _rev_roll(g: pd.DataFrame) -> pd.Series:
        s = g[target]
        r = s.iloc[::-1].rolling(window=hours, min_periods=1).max().shift(1)
        return r.iloc[::-1].fillna(0).astype(int)

    out = base.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_rev_roll)
    return out.reset_index(drop=True).to_numpy()


def future_max_label_by_point_timeaware(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """
    Time-aware window: for each (lat,lon) and row t_i, 1 if any future row within
    (t_i, t_i+hours] has target=1. Robust to tz-aware times and missing hours.

    Uses per-cell sort + searchsorted on int64 nanoseconds.
    """
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)  # 1h in ns

    for (_, _), g in df.groupby(["lat", "lon"], sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y = g[target].astype(int).to_numpy()

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
def _ckpt_paths(ckpt_dir: Path, lead: int | None = None):
    tag = "" if lead is None else f"_lead{lead}"
    return {
        "state": ckpt_dir / f"state{tag}.json",
        "probs": ckpt_dir / f"probs{tag}.npy",
    }


def load_checkpoint(ckpt_dir: Path, total_rows: int, lead: int | None = None):
    if not ckpt_dir or not ckpt_dir.exists():
        return 0, None
    paths = _ckpt_paths(ckpt_dir, lead)
    if not paths["state"].exists() or not paths["probs"].exists():
        return 0, None
    try:
        state = json.loads(paths["state"].read_text())
        done_rows = int(state.get("done_rows", 0))
        p = np.load(paths["probs"])
        if done_rows == p.shape[0] and done_rows <= total_rows:
            print(
                f"Resuming (lead={lead if lead is not None else 'global'}): "
                f"{done_rows}/{total_rows} rows already scored.",
                flush=True,
            )
            return done_rows, p
    except Exception:
        pass
    return 0, None


def save_checkpoint(ckpt_dir: Path, p_so_far: np.ndarray, done_rows: int, lead: int | None = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = _ckpt_paths(ckpt_dir, lead)
    np.save(paths["probs"], p_so_far)
    paths["state"].write_text(json.dumps({"done_rows": int(done_rows)}))


# ---------- model bundle helpers ----------
def _load_bundle(path: str):
    m = joblib.load(path)
    if not isinstance(m, dict):
        raise ValueError(
            "Expected a dict bundle (model, features, scaler, imputer_stats[, clip_stats, per_lead_models])."
        )
    model = m.get("model", None)
    scaler = m.get("scaler", None)
    feats = list(m.get("features", []))
    imp_stats = m.get("imputer_stats", None)
    clip_stats = m.get("clip_stats", None)
    per_lead = m.get("per_lead_models", None)
    if model is None and not (isinstance(per_lead, dict) and len(per_lead) > 0):
        raise ValueError("Bundle has neither a global 'model' nor any 'per_lead_models'.")
    if not feats:
        raise ValueError("Bundle lacks 'features'.")
    return (
        model,
        scaler,
        feats,
        imp_stats,
        clip_stats,
        (per_lead if isinstance(per_lead, dict) else {}),
    )


# ---------- feature builder (align to bundle) ----------
def build_matrix(
    df: pd.DataFrame,
    features: list[str],
    imputer_stats: dict[str, float] | None,
    clip_stats: dict[str, np.ndarray] | None,
    scaler,
):
    """
    Create X with the bundle's exact feature order; create missing columns; impute; clip; scale.
    """
    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan

    # impute
    if imputer_stats:
        arr = Xdf.to_numpy(dtype=float, copy=True)
        for j, c in enumerate(Xdf.columns):
            v = imputer_stats.get(c, 0.0)
            m = ~np.isfinite(arr[:, j])
            if m.any():
                arr[m, j] = v
    else:
        arr = Xdf.fillna(0.0).to_numpy(dtype=float, copy=True)

    # clip (if stats present and shapes match)
    if clip_stats and "lo" in clip_stats and "hi" in clip_stats:
        lo, hi = clip_stats["lo"], clip_stats["hi"]
        try:
            arr = np.clip(arr, lo, hi)
        except Exception:
            pass

    # scale
    if scaler is not None:
        arr = scaler.transform(arr)

    return arr


# ---------- metrics helper ----------
def safe_metrics(y: np.ndarray, p: np.ndarray):
    """AUC / PRAUC / Brier that won't explode on degenerate labels."""
    mets = {}
    try:
        mets["AUC"] = roc_auc_score(y, p)
    except Exception:
        mets["AUC"] = np.nan
    try:
        mets["PRAUC"] = average_precision_score(y, p)
    except Exception:
        mets["PRAUC"] = np.nan
    try:
        mets["Brier"] = brier_score_loss(y, p)
    except Exception:
        mets["Brier"] = np.nan
    return mets


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Lead-time evaluation with per-lead model support and robust feature alignment."
    )
    ap.add_argument("--labelled", required=True, help="CSV(.gz) or Parquet labelled grid.")
    ap.add_argument("--model", required=True, help="Joblib bundle with model + feature metadata.")
    ap.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, required=True)
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory for progress checkpoints.",
    )
    ap.add_argument(
        "--window-mode",
        choices=["time", "samples"],
        default="time",
        help="time: exact hours via timestamps (robust). "
             "samples: fixed sample-count per cell.",
    )
    args = ap.parse_args()

    print("== Lead-Time Evaluation (bundle) ==", flush=True)
    print(f"Labelled : {args.labelled}")
    print(f"Model    : {args.model}")
    print(f"Target   : {args.target}")
    print(f"Leads    : {args.lead_hours}")

    # Load bundle
    model_global, scaler, FEATS, imp_stats, clip_stats, per_lead = _load_bundle(args.model)

    # Pull exactly what we need (fast for Parquet; safe for CSV)
    need_cols = {"time", "lat", "lon", args.target, *FEATS}
    df = read_any(args.labelled, columns=list(need_cols))

    # Ensure required meta columns exist (create if missing so we can label later without hard crash)
    for req in ("time", "lat", "lon"):
        if req not in df.columns:
            df[req] = np.nan

    nrows = len(df)
    pos0 = int(pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).sum())
    print(
        f"Rows evaluated: {nrows:,}  Positives (coincident {args.target}): {pos0:,}",
        flush=True,
    )

    # Build once: aligned, imputed, clipped, scaled features
    Xs = build_matrix(df, FEATS, imp_stats, clip_stats, scaler)

    # Helper to score probabilities with chunking, per model
    def score_probs(estimator, ckpt_tag=None):
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        start_row, p_prev = load_checkpoint(ckpt_dir, nrows, lead=ckpt_tag)
        p_all = p_prev if p_prev is not None else np.empty((0,), dtype=float)

        if p_all.shape[0] < nrows:
            remaining = nrows - p_all.shape[0]
            steps = math.ceil(remaining / args.chunk_rows) or 1
            print("Scoring probabilities…", flush=True)
            for i0 in pbar(
                range(p_all.shape[0], nrows, args.chunk_rows),
                total=steps,
                title="  • chunks",
            ):
                i1 = min(i0 + args.chunk_rows, nrows)
                X_chunk = Xs[i0:i1]
                p_chunk = estimator.predict_proba(X_chunk)[:, 1]
                p_all = np.concatenate([p_all, p_chunk])
                if ckpt_dir:
                    save_checkpoint(ckpt_dir, p_all, i1, lead=ckpt_tag)
        else:
            print("Probabilities already scored (checkpoint).", flush=True)
        return p_all

    # If we have per-lead estimators, score separately per lead. Otherwise, score once with the global model.
    per_lead_probs = {}
    have_any_perlead = any((L in per_lead) for L in args.lead_hours)
    if have_any_perlead:
        print(
            "[EVAL] Using per-lead estimators where available; falling back to global otherwise.",
            flush=True,
        )
        for L in args.lead_hours:
            est = per_lead.get(L, model_global)
            tag = L if (L in per_lead) else None  # separate checkpoints per true per-lead
            per_lead_probs[L] = score_probs(est, ckpt_tag=tag)
    else:
        print(
            "[EVAL] No per-lead estimators found; using the global model for all leads.",
            flush=True,
        )
        p_global = score_probs(model_global, ckpt_tag=None)
        for L in args.lead_hours:
            per_lead_probs[L] = p_global

    # coincident metrics (global target now vs probs used for first lead shown)
    y0 = (
        pd.to_numeric(df[args.target], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    anyL = args.lead_hours[0]
    mets0 = safe_metrics(y0, per_lead_probs[anyL])
    print(
        f"[COINCIDENT] AUC={mets0['AUC']:.3f}  "
        f"PRAUC={mets0['PRAUC']:.3f}  "
        f"Brier={mets0['Brier']:.3f}",
        flush=True,
    )

    # choose labeler
    labeler = (
        future_max_label_by_point_timeaware
        if args.window_mode == "time"
        else future_max_label_by_point_samples
    )

    # lead metrics (each lead gets its own probabilities if per-lead model present)
    labels_per_h = {}
    for h in args.lead_hours:
        print(f"Preparing lead +{h}h labels …", flush=True)
        yh = labeler(df, args.target, hours=h)
        posh = int(yh.sum())
        p_use = per_lead_probs[h]

        mets = safe_metrics(yh, p_use)
        labels_per_h[h] = yh
        print(
            f"Lead +{h}h  →  "
            f"AUC={mets['AUC']:.3f}  "
            f"PRAUC={mets['PRAUC']:.3f}  "
            f"Brier={mets['Brier']:.3f}  "
            f"Pos={posh:,}/{nrows:,}",
            flush=True,
        )

    # sanity panel (overlap 24h vs 48h if both present)
    if len(labels_per_h) >= 2 and 24 in labels_per_h and 48 in labels_per_h:
        y24, y48 = labels_per_h[24], labels_per_h[48]
        inter = np.logical_and(y24 == 1, y48 == 1).sum()
        union = np.logical_or(y24 == 1, y48 == 1).sum()
        jacc = inter / max(1, union)
        print(
            f"[Sanity] Overlap 24h vs 48h → Jaccard={jacc:.3f}  "
            f"Pos24={y24.sum():,}  Pos48={y48.sum():,}",
            flush=True,
        )

    if args.checkpoint_dir:
        (Path(args.checkpoint_dir) / "_done.txt").write_text("ok")


if __name__ == "__main__":
    main()