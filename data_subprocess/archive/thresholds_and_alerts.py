#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thresholds_and_alerts.py — score slim feature tables and flag alerts.

Role in the pipeline
--------------------
This script operates on the *slim scoring table* emitted by stage_data.py,
which has:

    row_id, feat1, feat2, ..., featN

Optionally, the table may also include time/lat/lon, in which case those
columns are preserved into the scored alerts file so that downstream
matching (e.g. to IBTrACS) can work directly on the scored table.

What it does:
  1) Reads a CSV(.gz) with row_id + feature columns (and optionally time/lat/lon).
  2) Chooses feature columns (CLI or "all non-id numeric columns").
  3) Builds a score:

         score = sum_j w_j * x_j

     where w_j are column weights (default: same weight for all).
  4) Flags alerts where score >= score_thr.
  5) Writes a lean output:

         row_id, time, lat, lon, score, alert_final, ...

     Optionally, you can also keep the original feature columns.

Extras:
  • Optional per-column weights from a JSON file/string.
  • Optional global normalisation across the entire file:
      - scaled_score in [0,1] via global min/max.
      - score_z as a z-score via global mean/std.
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------- small utils -----------------


def _print(*a, **k):
    print(*a, **k, flush=True)


def load_weights(spec: Optional[str], feat_cols: List[str], default_w: float) -> Dict[str, float]:
    """
    Load per-column weights from a JSON string or file.
    Any feature not listed gets default_w.

    Example JSON (string or file contents):
      {
        "CAPE": 60,
        "shear_deep": 70,
        "S3": 55
      }
    """
    if not spec:
        return {c: float(default_w) for c in feat_cols}

    try:
        # try JSON string first
        obj = json.loads(spec)
    except json.JSONDecodeError:
        # fall back to treating spec as a file path
        txt = Path(spec).read_text(encoding="utf-8")
        obj = json.loads(txt)

    if not isinstance(obj, dict):
        raise ValueError("weights spec must decode to a JSON object {col: weight, ...}")

    out: Dict[str, float] = {}
    for c in feat_cols:
        if c in obj:
            try:
                out[c] = float(obj[c])
            except Exception:
                out[c] = float(default_w)
        else:
            out[c] = float(default_w)
    return out


def _normalize_explicit_cols(explicit_cols: Optional[List[str]]) -> Optional[List[str]]:
    """
    Handle the run_pipeline flattening quirk where a list of feature
    names like [SFI,SFI2,...] becomes ["SFI,SFI2,..."].

    If we see a single string with commas, split it.
    """
    if not explicit_cols:
        return None
    if len(explicit_cols) == 1 and isinstance(explicit_cols[0], str) and "," in explicit_cols[0]:
        parts = [c.strip() for c in explicit_cols[0].split(",") if c.strip()]
        return parts
    return explicit_cols


def detect_feature_columns(df: pd.DataFrame, id_col: str, explicit_cols: Optional[List[str]]) -> List[str]:
    cols = list(df.columns)
    if id_col not in cols:
        raise ValueError(f"id column '{id_col}' not found; got columns: {cols[:10]}...")

    explicit_cols = _normalize_explicit_cols(explicit_cols)

    if explicit_cols:
        # Only keep those that actually exist and are not the id_col
        feats = [c for c in explicit_cols if c in cols and c != id_col]
        if not feats:
            raise ValueError(
                f"None of the requested feature columns {explicit_cols} "
                f"were found in {cols[:10]}..."
            )
        return feats

    # Otherwise, auto-detect numeric-ish columns except id_col and common meta cols
    numeric = []
    meta_skip = {id_col, "time", "lat", "lon"}
    for c in cols:
        if c in meta_skip:
            continue
        try:
            sample = pd.to_numeric(df[c].head(100), errors="coerce")
            if sample.notna().any():
                numeric.append(c)
        except Exception:
            continue
    if not numeric:
        raise ValueError("Could not auto-detect any numeric feature columns.")
    return numeric


# ----------------- scoring core -----------------


def score_chunk(
    chunk: pd.DataFrame,
    id_col: str,
    feat_cols: List[str],
    weights: Dict[str, float],
    score_col: str,
    alert_col: str,
    score_thr: float,
) -> pd.DataFrame:
    """
    Given a chunk with id_col + feat_cols, compute:
      score = sum_j w_j * x_j
    and alert_flag = 1{score >= score_thr}.
    """
    out = pd.DataFrame()
    if id_col not in chunk.columns:
        raise ValueError(f"Chunk missing id_col '{id_col}'")

    out[id_col] = chunk[id_col].copy()

    # start with zeros
    score = np.zeros(len(chunk), dtype="float64")

    for c in feat_cols:
        if c not in chunk.columns:
            # should not happen if we detected properly, but be defensive
            _print(f"[warn] feature column '{c}' missing in chunk; treating as zeros.")
            continue
        v = pd.to_numeric(chunk[c], errors="coerce").fillna(0.0).values
        w = float(weights.get(c, 0.0))
        if w != 0.0:
            score += w * v

    out[score_col] = score
    out[alert_col] = (score >= score_thr).astype("int8")
    return out


# ----------------- main -----------------


def main():
    ap = argparse.ArgumentParser(
        description="Score row_id + feature tables and flag alerts based on total score."
    )
    ap.add_argument(
        "--scoring-src",
        required=True,
        help="Input CSV(.gz) with row_id + feature columns "
             "(e.g. data/grid_scoring_start.csv.gz)",
    )
    ap.add_argument(
        "--out",
        default="results/alerts_scored.csv.gz",
        help=(
            "Output CSV(.gz) with row_id,score,alert_final "
            "and optionally time/lat/lon and features "
            "(default: results/alerts_scored.csv.gz)"
        ),
    )
    ap.add_argument(
        "--id-col",
        default="row_id",
        help="Name of ID column to carry through (default: row_id)",
    )
    ap.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of feature columns to use. "
            "Can be provided as a space-separated list or as a single "
            "comma-separated string. If omitted, auto-detect numeric "
            "columns (excluding id/time/lat/lon)."
        ),
    )
    ap.add_argument(
        "--weights-json",
        default=None,
        help=(
            "JSON string or path giving per-column weights, e.g. "
            '\'{"CAPE":60,"shear_deep":70,"S3":55}\'. '
            "Any feature not listed gets --default-weight."
        ),
    )
    ap.add_argument(
        "--default-weight",
        type=float,
        default=50.0,
        help="Default weight for any feature not explicitly listed (default: 50.0).",
    )
    ap.add_argument(
        "--score-thr",
        type=float,
        default=300.0,
        help="Score threshold for alert_final=1 (default: 300.0).",
    )
    ap.add_argument(
        "--score-col",
        default="score",
        help="Name of score column to write (default: score).",
    )
    ap.add_argument(
        "--alert-col",
        default="alert_final",
        help="Name of alert flag column to write (default: alert_final).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Chunk size for streaming read (default: 500000).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Alias for --chunk-rows (compatibility with orchestrator hints).",
    )
    ap.add_argument(
        "--parquet-rows",
        type=int,
        default=None,
        help="Alias for --chunk-rows when using parquet (compatibility only).",
    )
    ap.add_argument(
        "--keep-features",
        action="store_true",
        help="If set, also keep original feature columns in the output (debugging).",
    )
    ap.add_argument(
        "--add-scaled-score",
        action="store_true",
        help="If set, also emit scaled_score in [0,1] using global score min/max.",
    )
    ap.add_argument(
        "--add-zscore",
        action="store_true",
        help="If set, also emit score_z using global mean/std of the score.",
    )

    args = ap.parse_args()
    if args.chunksize and not args.chunk_rows:
        args.chunk_rows = args.chunksize
    if args.parquet_rows and not args.chunk_rows:
        args.chunk_rows = args.parquet_rows

    src = Path(args.scoring_src)
    if not src.exists():
        raise SystemExit(f"[fatal] scoring-src not found: {src}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if out_path.name.lower().endswith(".gz") else "infer"

    # We need one small sample to decide feature set + weights
    _print(f"[init] sampling first chunk from {src} to detect feature columns...")
    sample = pd.read_csv(
        src,
        nrows=min(10_000, args.chunk_rows),
        compression="infer",
        low_memory=False,
    )

    feat_cols = detect_feature_columns(
        sample, id_col=args.id_col, explicit_cols=args.feature_cols
    )
    weights = load_weights(
        args.weights_json, feat_cols=feat_cols, default_w=args.default_weight
    )

    _print(f"[init] id_col={args.id_col}")
    _print(f"[init] feature columns ({len(feat_cols)}): {feat_cols}")
    _print(f"[init] default weight={args.default_weight}, score_thr={args.score_thr}")
    _print(f"[init] output -> {out_path}")

    need_norm = bool(args.add_scaled_score or args.add_zscore)

    # ---------- optional first pass: global score stats for normalisation ----------
    score_min = math.inf
    score_max = -math.inf
    sum_scores = 0.0
    sum_sq_scores = 0.0
    rows_seen_for_stats = 0

    if need_norm:
        _print("[init] normalisation requested; running first pass to collect score stats...")
        rdr_stats = pd.read_csv(
            src,
            compression="infer",
            low_memory=False,
            chunksize=int(args.chunk_rows),
        )
        if not hasattr(rdr_stats, "__iter__"):
            rdr_stats = [rdr_stats]

        for i, chunk in enumerate(rdr_stats, start=1):
            if chunk is None or chunk.empty:
                continue

            scored_stats = score_chunk(
                chunk=chunk,
                id_col=args.id_col,
                feat_cols=feat_cols,
                weights=weights,
                score_col=args.score_col,
                alert_col=args.alert_col,
                score_thr=float(args.score_thr),
            )
            s = scored_stats[args.score_col].to_numpy()
            if s.size == 0:
                continue

            rows_seen_for_stats += int(s.size)
            sum_scores += float(s.sum())
            sum_sq_scores += float((s * s).sum())
            smin = float(s.min())
            smax = float(s.max())
            if smin < score_min:
                score_min = smin
            if smax > score_max:
                score_max = smax

            if i % 10 == 0:
                _print(f"[stats pass] chunk {i} rows={len(chunk)} total_rows={rows_seen_for_stats}")

        if rows_seen_for_stats == 0:
            _print("[warn] no rows found in scoring-src during stats pass; nothing to write.")
            return

        mean_score = sum_scores / rows_seen_for_stats
        # Guard against tiny negative due to float drift
        var_score = max(sum_sq_scores / rows_seen_for_stats - mean_score * mean_score, 0.0)
        std_score = math.sqrt(var_score)

        _print(
            "[stats] score_min={:.3f}, score_max={:.3f}, mean={:.3f}, std={:.3f}, rows={:,}".format(
                score_min, score_max, mean_score, std_score, rows_seen_for_stats
            )
        )
    else:
        # Provide placeholders so we can still reference these names later
        mean_score = float("nan")
        std_score = float("nan")

    # ---------- second (or single) pass: scoring + write ----------
    rdr = pd.read_csv(
        src,
        compression="infer",
        low_memory=False,
        chunksize=int(args.chunk_rows),
    )
    if not hasattr(rdr, "__iter__"):
        rdr = [rdr]

    total_rows = 0
    total_alerts = 0
    first = True

    for i, chunk in enumerate(rdr, start=1):
        if chunk is None or chunk.empty:
            continue

        scored = score_chunk(
            chunk=chunk,
            id_col=args.id_col,
            feat_cols=feat_cols,
            weights=weights,
            score_col=args.score_col,
            alert_col=args.alert_col,
            score_thr=float(args.score_thr),
        )

        # Optional normalised scores
        if need_norm:
            s = scored[args.score_col].to_numpy(dtype="float64")
            if args.add_scaled_score:
                if math.isfinite(score_min) and math.isfinite(score_max) and score_max > score_min:
                    scaled = (s - score_min) / (score_max - score_min)
                else:
                    # Degenerate case: constant scores -> all 0.5
                    scaled = np.full_like(s, 0.5, dtype="float64")
                scored["scaled_score"] = scaled
            if args.add_zscore:
                if math.isfinite(mean_score) and math.isfinite(std_score) and std_score > 0.0:
                    z = (s - mean_score) / std_score
                else:
                    z = np.zeros_like(s, dtype="float64")
                scored["score_z"] = z

        # Always carry through geo/time columns if present
        for meta_col in ("time", "lat", "lon"):
            if meta_col in chunk.columns and meta_col not in scored.columns:
                scored[meta_col] = chunk[meta_col].values

        # Optionally keep feature columns too
        if args.keep_features:
            extras = [c for c in feat_cols if c in chunk.columns]
            for c in extras:
                if c not in scored.columns:
                    scored[c] = chunk[c].values

        # Nice column order: id, time/lat/lon, score, scaled_score, score_z, alert, then the rest
        front = []
        for c in (
            args.id_col,
            "time",
            "lat",
            "lon",
            args.score_col,
            "scaled_score" if args.add_scaled_score else None,
            "score_z" if args.add_zscore else None,
            args.alert_col,
        ):
            if c and c in scored.columns and c not in front:
                front.append(c)
        rest = [c for c in scored.columns if c not in front]
        scored = scored[front + rest]

        n = len(scored)
        total_rows += n
        total_alerts += int(scored[args.alert_col].sum())

        scored.to_csv(
            out_path,
            index=False,
            mode="w" if first else "a",
            header=first,
            compression=compression,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        first = False

        if i % 10 == 0:
            _print(
                f"[chunk {i}] wrote {n} rows (total={total_rows:,}, "
                f"alerts={total_alerts:,})"
            )

    _print(f"[done] scoring complete -> {out_path}")
    rate = (total_alerts / total_rows) if total_rows else float("nan")
    _print(
        f"        rows={total_rows:,}, alerts={total_alerts:,}, "
        f"alert_rate={rate:.4f}"
    )
    if need_norm:
        _print(
            "        (normalisation: scaled_score in [0,1], score_z mean≈0/std≈1 "
            "based on global score distribution)"
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
