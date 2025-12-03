#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thresholds_scan.py — mine high-score alerts for extra useful features.

Pipeline position
-----------------
Inputs:
  1) Alerts+targets file (typically from ibtracs_match), e.g.
       results/alerts_with_targets.csv.gz
     Must contain at least:
       - time   (UTC)
       - lat, lon
       - score (alert strength)
       - storm_hit (0/1), from ibtracs match
       - row_id (optional; computed here if missing)

  2) Rich grid file, e.g.
       data/grid_labelled_FMA_gka_realthermo_sph.csv.gz
     Same underlying grid as used by stage_data.py / id builder,
     with time/lat/lon and many candidate columns.

What it does
------------
1) Select "high-score" alerts:
     - either score >= --score-threshold
     - or score >= quantile(--score-quantile) if threshold is None.

   IMPORTANT: this version does NOT load the full alerts CSV into memory.
   It streams in chunks, uses a reservoir-sample to estimate the quantile,
   then makes a second streaming pass to keep only the high-score rows.

2) Ensure row_id for alerts:
     row_id = hash(time_floor_H, lat_round3, lon_round3) as uint64
     (identical to stage_data.py's build_ids_from_csv) if missing.

3) Stream the rich file, compute row_id if needed, and
   retain only rows whose row_id is in the high-score set.

4) Join high-score alerts to their rich-row on row_id and:
     - write an enriched high-score table
     - compute summary stats for candidate columns
       (mean / median / high quantiles for all highs vs hits_only).

Outputs
-------
--out-enriched (default: results/high_score_enriched.csv.gz):
    Each high-score alert with:
      row_id, time, lat, lon, score, storm_hit, lead_h, min_dist_km, ...
      + candidate columns pulled from the rich file.

--out-summary (default: results/feature_candidate_summary.csv):
    One row per candidate column with:
      n_all, mean_all, q50_all, q75_all, q90_all
      n_hits, mean_hits, q50_hits, q75_hits, q90_hits

Usage example
-------------
python data_subprocess/thresholds_scan.py ^
  --alerts-with-targets results/alerts_with_targets.csv.gz ^
  --rich data/grid_labelled_FMA_gka_realthermo_sph.csv.gz ^
  --out-enriched results/high_score_enriched.csv.gz ^
  --out-summary results/feature_candidate_summary.csv ^
  --score-quantile 0.9 ^
  --candidate-cols CAPE shear_deep S3 SFI sph_radial_abs spiral_score
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import random

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


# -------------------- small utilities --------------------


def _print(*a, **k):
    print(*a, **k, flush=True)


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Parse timestamps to tz-naive UTC (datetime64[ns])."""
    if pd.api.types.is_datetime64_any_dtype(series):
        t = pd.to_datetime(series, utc=True, errors="coerce")
        return t.dt.tz_convert(None)
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)


def _compute_row_id_from_tll(
    df: pd.DataFrame,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.Series:
    """
    Compute row_id exactly as in stage_data.build_ids_from_csv:

      • time parsed to UTC, then floored to hour
      • lat, lon parsed to numeric; lat/lon rounded to 3 decimals in hash key
      • hash_pandas_object(..., index=False).astype('uint64')
    """
    t = _to_utc_naive(df[time_col])
    t_floor = t.dt.floor("H")

    latv = pd.to_numeric(df[lat_col], errors="coerce")
    lonv = pd.to_numeric(df[lon_col], errors="coerce")

    key = pd.DataFrame(
        {
            "time": t_floor,
            "lat": latv.round(3),
            "lon": lonv.round(3),
        }
    )
    row_id = hash_pandas_object(key, index=False).astype("uint64")
    return row_id


def _iter_rich_chunks(path: Path, chunksize: int) -> Any:
    it = pd.read_csv(path, compression="infer", low_memory=False, chunksize=int(chunksize))
    if not hasattr(it, "__iter__"):
        it = [it]
    return it


def _as_list(x: Optional[List[str]]) -> List[str]:
    if x is None:
        return []
    return list(x)


# -------------------- original alerts loader (kept for small jobs) --------------------


def load_alerts_with_row_id(
    alerts_path: Path,
    time_col: str,
    lat_col: str,
    lon_col: str,
    score_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Legacy "load it all" version. Fine for smaller CSVs, but not for 70M rows.
    Left here for completeness; main() now uses a streaming version instead.
    """
    df = pd.read_csv(alerts_path, compression="infer", low_memory=False)

    need = [time_col, lat_col, lon_col, score_col, target_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"alerts file missing required columns: {missing}")

    if "row_id" not in df.columns:
        _print("[scan] alerts missing row_id; computing from time/lat/lon.")
        df["row_id"] = _compute_row_id_from_tll(df, time_col, lat_col, lon_col).astype("uint64")
    else:
        df["row_id"] = df["row_id"].astype("uint64")

    # Clean basic columns
    df[time_col] = _to_utc_naive(df[time_col])
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")

    df = df.dropna(subset=[time_col, lat_col, lon_col, score_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"alerts file {alerts_path} has no usable rows after cleaning.")
    return df


def select_high_score_alerts(
    alerts: pd.DataFrame,
    score_col: str,
    score_threshold: Optional[float],
    score_quantile: float,
) -> (pd.DataFrame, float):
    """Return subset of high-score alerts + the effective threshold (legacy, in-memory)."""
    scores = alerts[score_col].astype(float)

    if score_threshold is not None and np.isfinite(score_threshold):
        thr = float(score_threshold)
    else:
        q = float(score_quantile)
        q = min(max(q, 0.0), 1.0)
        thr = float(scores.quantile(q))

    high = alerts[scores >= thr].copy()
    _print(
        f"[scan] score threshold = {thr:.4f} "
        f"(kept {len(high)}/{len(alerts)} rows; frac={len(high)/len(alerts):.4f})"
    )
    return high, thr


# -------------------- NEW: streaming high-score selector --------------------


def _estimate_quantile_reservoir(
    alerts_path: Path,
    score_col: str,
    score_quantile: float,
    chunk_rows: int,
) -> Tuple[float, int]:
    """
    One streaming pass over alerts CSV using reservoir sampling to approximate
    the score quantile. Returns (threshold, total_rows_seen).
    """
    max_sample = 2_000_000  # ~16 MB float64, safe
    sample: List[float] = []
    seen = 0

    for chunk in pd.read_csv(
        alerts_path,
        compression="infer",
        low_memory=False,
        chunksize=int(chunk_rows),
    ):
        if chunk is None or chunk.empty:
            continue

        if score_col not in chunk.columns:
            raise ValueError(f"alerts file missing score column '{score_col}' in chunk")

        s = pd.to_numeric(chunk[score_col], errors="coerce").dropna().astype(float).values
        for v in s:
            seen += 1
            v = float(v)
            if len(sample) < max_sample:
                sample.append(v)
            else:
                # classic reservoir sampling
                j = random.randint(0, seen - 1)
                if j < max_sample:
                    sample[j] = v

    if seen == 0 or not sample:
        raise ValueError("No valid scores found in alerts file while estimating quantile.")

    q = float(score_quantile)
    q = min(max(q, 0.0), 1.0)
    thr = float(np.quantile(np.array(sample, dtype=float), q))
    return thr, seen


def select_high_score_alerts_streaming(
    alerts_path: Path,
    time_col: str,
    lat_col: str,
    lon_col: str,
    score_col: str,
    target_col: str,
    score_threshold: Optional[float],
    score_quantile: Optional[float],
    chunk_rows: int,
) -> Tuple[pd.DataFrame, float, np.ndarray, int]:
    """
    Streaming version: does NOT load full alerts CSV into memory.

    Steps:
      1) If score_threshold is provided -> use it directly.
         Else: streaming reservoir-sample quantile to estimate score threshold.
      2) Second streaming pass: clean basic columns, ensure row_id, and
         keep only rows with score >= threshold.
      3) Return:
           • high alerts DataFrame
           • threshold
           • high_ids (row_id array)
           • total_rows (for logging)
    """
    # --- Step 1: pick threshold ---
    if score_threshold is not None and np.isfinite(score_threshold):
        thr = float(score_threshold)
        # Still need total_rows for logging; count in a cheap pass
        total_rows = 0
        for chunk in pd.read_csv(
            alerts_path,
            compression="infer",
            low_memory=False,
            chunksize=int(chunk_rows),
        ):
            if chunk is not None:
                total_rows += len(chunk)
        _print(f"[scan] using fixed score threshold={thr:.4f} (total_rows={total_rows})")
    else:
        q = 0.9 if score_quantile is None else float(score_quantile)
        thr, total_rows = _estimate_quantile_reservoir(
            alerts_path=alerts_path,
            score_col=score_col,
            score_quantile=q,
            chunk_rows=chunk_rows,
        )
        _print(f"[scan] estimated score threshold (q={q:.3f}) = {thr:.4f} from {total_rows} rows")

    # --- Step 2: streaming pass to collect highs ---
    frames: List[pd.DataFrame] = []
    high_ids_list: List[np.ndarray] = []
    kept = 0
    seen2 = 0

    need = [time_col, lat_col, lon_col, score_col, target_col]

    for i, chunk in enumerate(
        pd.read_csv(
            alerts_path,
            compression="infer",
            low_memory=False,
            chunksize=int(chunk_rows),
        ),
        start=1,
    ):
        if chunk is None or chunk.empty:
            continue

        seen2 += len(chunk)

        missing = [c for c in need if c not in chunk.columns]
        if missing:
            raise ValueError(f"alerts file missing required columns in chunk {i}: {missing}")

        # row_id: use existing if present, else compute
        if "row_id" in chunk.columns:
            rid = pd.to_numeric(chunk["row_id"], errors="coerce").astype("uint64")
        else:
            rid = _compute_row_id_from_tll(chunk, time_col, lat_col, lon_col).astype("uint64")

        # Clean columns on this chunk only
        t = _to_utc_naive(chunk[time_col])
        latv = pd.to_numeric(chunk[lat_col], errors="coerce")
        lonv = pd.to_numeric(chunk[lon_col], errors="coerce")
        scores = pd.to_numeric(chunk[score_col], errors="coerce")
        tgt = pd.to_numeric(chunk[target_col], errors="coerce").fillna(0).astype("int8")

        # Drop unusable rows
        mask_good = t.notna() & latv.notna() & lonv.notna() & scores.notna()
        if not mask_good.any():
            continue

        t = t[mask_good]
        latv = latv[mask_good]
        lonv = lonv[mask_good]
        scores = scores[mask_good]
        tgt = tgt[mask_good]
        rid = rid[mask_good]

        # Filter by threshold
        mask_high = scores >= thr
        if not mask_high.any():
            continue

        sub = pd.DataFrame(
            {
                "row_id": rid[mask_high].astype("uint64"),
                time_col: t[mask_high],
                lat_col: latv[mask_high],
                lon_col: lonv[mask_high],
                score_col: scores[mask_high],
                target_col: tgt[mask_high],
            }
        )

        kept += len(sub)
        frames.append(sub)
        high_ids_list.append(sub["row_id"].values)

        _print(
            f"[scan] alerts chunk {i}: rows={len(chunk)} high={len(sub)} "
            f"(total_high={kept}, total_seen={seen2})"
        )

    if not frames:
        raise ValueError(
            f"No high-score alerts selected with threshold={thr:.4f}. "
            "Consider lowering the threshold or quantile."
        )

    high = pd.concat(frames, ignore_index=True)
    high_ids = np.concatenate(high_ids_list)
    _print(
        f"[scan] score threshold = {thr:.4f} "
        f"(kept {len(high)}/{total_rows} rows; frac={len(high)/max(total_rows,1):.4f})"
    )
    return high, thr, high_ids, total_rows


# -------------------- rich collector --------------------


def collect_rich_for_ids(
    rich_path: Path,
    high_ids: np.ndarray,
    time_col: str,
    lat_col: str,
    lon_col: str,
    candidate_cols: List[str],
    chunk_rows: int,
) -> pd.DataFrame:
    """
    Stream rich CSV, obtain row_id, and keep only rows with row_id in high_ids.

    If the rich file already has a 'row_id' column, we TRUST it and use that
    directly. Otherwise we recompute row_id from (time, lat, lon) using the
    same hashing as stage_data.build_ids_from_csv.
    """
    id_set = set(high_ids.tolist())
    use_cands = _as_list(candidate_cols)

    frames: List[pd.DataFrame] = []
    total_seen = 0
    total_kept = 0
    used_precomputed = False

    for i, chunk in enumerate(_iter_rich_chunks(rich_path, chunk_rows), start=1):
        if chunk is None or chunk.empty:
            continue

        total_seen += len(chunk)

        # Prefer precomputed row_id if present
        if "row_id" in chunk.columns:
            row_id = pd.to_numeric(chunk["row_id"], errors="coerce").astype("uint64")
            if not used_precomputed:
                _print(f"[scan] using precomputed row_id from rich file (chunk {i})")
                used_precomputed = True
        else:
            # Need time/lat/lon present to recompute
            for c in [time_col, lat_col, lon_col]:
                if c not in chunk.columns:
                    raise ValueError(
                        f"rich file missing '{c}' in chunk {i}. "
                        f"Columns seen: {list(chunk.columns)[:10]}"
                    )
            row_id = _compute_row_id_from_tll(chunk, time_col, lat_col, lon_col).astype("uint64")
            if i == 1:
                _print("[scan] rich file has no row_id; recomputing from time/lat/lon.")

        # Filter by high_ids
        mask = row_id.isin(id_set)
        if not mask.any():
            continue

        sub = chunk.loc[mask].copy()

        # Ensure row_id values are the clean uint64 version, and put it first
        rid_vals = row_id.loc[mask].values.astype("uint64")
        if "row_id" in sub.columns:
            sub["row_id"] = rid_vals
        else:
            sub.insert(0, "row_id", rid_vals)

        # Reorder so row_id is first column
        cols = list(sub.columns)
        cols = ["row_id"] + [c for c in cols if c != "row_id"]
        sub = sub[cols]

        if use_cands:
            keep_cols = ["row_id"] + [c for c in use_cands if c in sub.columns]
            sub = sub[keep_cols]
        else:
            # if candidate list is empty, we keep all numeric columns except coords/time
            reserved = {time_col, lat_col, lon_col, "storm", "label"}
            numeric = sub.select_dtypes(include=[np.number]).columns
            dyn_cands = [c for c in numeric if c not in reserved and c != "row_id"]
            keep_cols = ["row_id"] + dyn_cands
            sub = sub[keep_cols]

        kept_here = len(sub)
        total_kept += kept_here
        frames.append(sub)

        _print(
            f"[scan] rich chunk {i}: rows={len(chunk)} kept={kept_here} "
            f"(total_kept={total_kept}, total_seen={total_seen})"
        )

    if not frames:
        _print("[scan] no matching rows in rich file for high-score IDs.")
        return pd.DataFrame(columns=["row_id"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["row_id"]).reset_index(drop=True)
    _print(f"[scan] rich matches: {len(out)} unique row_id rows.")
    return out


# -------------------- summary --------------------


def summarize_candidates(
    enriched: pd.DataFrame,
    candidate_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Compute per-feature stats for high alerts vs high hits."""
    use_cands = [c for c in candidate_cols if c in enriched.columns]
    if not use_cands:
        _print("[scan] no candidate columns present in enriched table; summary will be empty.")
        return pd.DataFrame(
            columns=[
                "feature",
                "n_all",
                "mean_all",
                "q50_all",
                "q75_all",
                "q90_all",
                "n_hits",
                "mean_hits",
                "q50_hits",
                "q75_hits",
                "q90_hits",
            ]
        )

    hits = enriched[enriched[target_col] == 1]

    rows: List[Dict[str, Any]] = []
    for c in use_cands:
        x_all = pd.to_numeric(enriched[c], errors="coerce")
        x_hits = pd.to_numeric(hits[c], errors="coerce") if not hits.empty else pd.Series([], dtype=float)

        def stats(x: pd.Series) -> Dict[str, float]:
            x = x.dropna()
            if x.empty:
                return dict(n=0, mean=np.nan, q50=np.nan, q75=np.nan, q90=np.nan)
            return dict(
                n=int(x.size),
                mean=float(x.mean()),
                q50=float(x.quantile(0.5)),
                q75=float(x.quantile(0.75)),
                q90=float(x.quantile(0.9)),
            )

        s_all = stats(x_all)
        s_hits = stats(x_hits)

        rows.append(
            dict(
                feature=c,
                n_all=s_all["n"],
                mean_all=s_all["mean"],
                q50_all=s_all["q50"],
                q75_all=s_all["q75"],
                q90_all=s_all["q90"],
                n_hits=s_hits["n"],
                mean_hits=s_hits["mean"],
                q50_hits=s_hits["q50"],
                q75_hits=s_hits["q75"],
                q90_hits=s_hits["q90"],
            )
        )

    out = pd.DataFrame(rows)
    # Sort by how much the mean inflates for hits vs all (descending)
    out["delta_mean_hits_minus_all"] = out["mean_hits"] - out["mean_all"]
    out = out.sort_values("delta_mean_hits_minus_all", ascending=False).reset_index(drop=True)
    return out


# -------------------- main --------------------


def main():
    ap = argparse.ArgumentParser(
        description="Scan high-score alerts, rejoin to rich grid rows, and summarise extra features."
    )
    ap.add_argument(
        "--alerts-with-targets",
        required=True,
        help="Alerts+targets CSV(.gz), e.g. results/alerts_with_targets.csv.gz",
    )
    ap.add_argument(
        "--rich",
        required=True,
        help="Rich grid CSV(.gz), e.g. data/grid_labelled_FMA_gka_realthermo_sph.csv.gz",
    )

    ap.add_argument(
        "--out-enriched",
        default="results/high_score_enriched.csv.gz",
        help="Output CSV(.gz) with high-score alerts plus rich features.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/feature_candidate_summary.csv",
        help="Output CSV with per-feature summary stats.",
    )

    # Column names in alerts
    ap.add_argument("--alerts-time-col", default="time")
    ap.add_argument("--alerts-lat-col", default="lat")
    ap.add_argument("--alerts-lon-col", default="lon")
    ap.add_argument("--alerts-score-col", default="score")
    ap.add_argument("--alerts-target-col", default="storm_hit")

    # Column names in rich file
    ap.add_argument("--rich-time-col", default="time")
    ap.add_argument("--rich-lat-col", default="lat")
    ap.add_argument("--rich-lon-col", default="lon")

    # How to decide "high-score"
    ap.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Absolute threshold on score. If omitted, use score-quantile instead.",
    )
    ap.add_argument(
        "--score-quantile",
        type=float,
        default=0.9,
        help="Quantile for score-based high selection when threshold not set (default: 0.9).",
    )

    # Candidate feature list
    ap.add_argument(
        "--candidate-cols",
        nargs="*",
        default=None,
        help="Optional list of candidate feature columns from rich file. "
             "If omitted, all numeric non-coordinate columns are used.",
    )

    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Chunk size when streaming alerts and rich file (default: 500k).",
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

    args = ap.parse_args()
    if args.chunksize and not args.chunk_rows:
        args.chunk_rows = args.chunksize
    if args.parquet_rows and not args.chunk_rows:
        args.chunk_rows = args.parquet_rows

    alerts_path = Path(args.alerts_with_targets)
    rich_path = Path(args.rich)
    out_enriched = Path(args.out_enriched)
    out_summary = Path(args.out_summary)

    if not alerts_path.exists():
        raise SystemExit(f"[fatal] alerts file not found: {alerts_path}")
    if not rich_path.exists():
        raise SystemExit(f"[fatal] rich file not found: {rich_path}")

    # 1–2) Streaming high-score selection from alerts
    high, thr, high_ids, total_rows = select_high_score_alerts_streaming(
        alerts_path=alerts_path,
        time_col=args.alerts_time_col,
        lat_col=args.alerts_lat_col,
        lon_col=args.alerts_lon_col,
        score_col=args.alerts_score_col,
        target_col=args.alerts_target_col,
        score_threshold=args.score_threshold,
        score_quantile=args.score_quantile,
        chunk_rows=int(args.chunk_rows),
    )
    if high.empty:
        _print("[scan] no high-score alerts selected; nothing to do.")
        return

    # 3) Collect matching rich rows for those IDs
    rich_match = collect_rich_for_ids(
        rich_path=rich_path,
        high_ids=high_ids,
        time_col=args.rich_time_col,
        lat_col=args.rich_lat_col,
        lon_col=args.rich_lon_col,
        candidate_cols=_as_list(args.candidate_cols),
        chunk_rows=int(args.chunk_rows),
    )

    # 4) Join high alerts ↔ rich features on row_id
    if rich_match.empty:
        _print("[scan] WARNING: no matching rows between high-score alerts and rich file. "
               "Enriched output will be alerts-only.")
        enriched = high.copy()
    else:
        enriched = high.merge(rich_match, on="row_id", how="left")

    out_enriched.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if out_enriched.name.lower().endswith(".gz") else "infer"
    enriched.to_csv(
        out_enriched,
        index=False,
        compression=compression,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    _print(f"[scan] wrote enriched highs -> {out_enriched} rows={len(enriched)} "
           f"(threshold={thr:.4f}, total_alert_rows={total_rows})")

    # 5) Summarise candidate columns
    # Candidate list: explicit from CLI or inferred from rich_match
    if args.candidate_cols:
        cand_cols = [c for c in args.candidate_cols if c in enriched.columns]
    else:
        # All numeric columns from rich_match except row_id
        numeric = rich_match.select_dtypes(include=[np.number]).columns
        cand_cols = [c for c in numeric if c != "row_id"]

    summary = summarize_candidates(
        enriched=enriched,
        candidate_cols=cand_cols,
        target_col=args.alerts_target_col,
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    _print(f"[scan] wrote feature summary -> {out_summary} rows={len(summary)}")


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
