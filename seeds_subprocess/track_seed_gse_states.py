#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_seed_gse_states.py

Agent: add G/S/E state tracking + shear role diagnostics without changing physics.

Takes a GSE panel (or seeds merged with GSE features), filters to seed points,
assigns discrete G/S/E states, computes dG_dt + shear role (supportive vs
destructive), links seeds into proto-tracks, and summarises state transitions.

Outputs (in --out-dir, prefixed by --run-name):
  *_gse_track_points.csv       : per-step states, dG_dt, shear role, track_id
  *_gse_track_summaries.csv    : per-track stats + optional IBTrACS match flag
  *_gse_state_transitions.csv  : transition counts (all / builder / dead)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from utils import join_audit

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # optional; parquet chunking
    pq = None

# Reuse proto_outcomes utilities for IBTrACS matching + linking logic
try:
    from seeds_subprocess.proto_outcomes import (  # type: ignore
        link_tracks,
        load_ibtracs,
        label_tracks_kdtree,
    )
except Exception:
    # Fallback: relative import when executed from this folder
    from proto_outcomes import link_tracks, load_ibtracs, label_tracks_kdtree  # type: ignore


CANDIDATE_TIME_COLS = ["time", "time_h", "valid_time", "datetime"]
CANDIDATE_FLAG_COLS = ["alert_final", "alert_base", "alert", "flag", "any_alert", "__flag__"]
CANDIDATE_PROB_COLS = ["prob_viable", "prob", "prob_max", "p", "score"]


def _is_parquet(path: str) -> bool:
    return Path(path).suffix.lower() in {".parquet", ".parq", ".pq"}


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180


def _to_utc_naive(series: pd.Series, fmt: str | None = None) -> pd.Series:
    raw = series
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)


def _iter_batches(path: str, columns: Sequence[str] | None, chunk_rows: int | None) -> Iterable[pd.DataFrame]:
    low = path.lower()
    if _is_parquet(path):
        if chunk_rows and chunk_rows > 0 and pq is not None:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=int(chunk_rows), columns=list(columns) if columns else None):
                yield batch.to_pandas()
            return
        yield pd.read_parquet(path, columns=list(columns) if columns else None)
        return

    if chunk_rows and chunk_rows > 0:
        for ch in pd.read_csv(path, usecols=list(columns) if columns else None, chunksize=int(chunk_rows), low_memory=False):
            yield ch
        return
    yield pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)


def _digitize_levels(series: pd.Series, quantiles: Sequence[float]) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    bins = np.quantile(vals.dropna(), quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    levels = np.digitize(vals.to_numpy(), bins[1:-1], right=False)
    return pd.Series(levels, index=series.index, copy=False).astype("int64")


def _load_threshold(thr_file: str, lead_h: int, col: str) -> Optional[float]:
    if not thr_file:
        return None
    path = Path(thr_file)
    if not path.exists():
        raise FileNotFoundError(thr_file)
    if path.suffix.lower() in {".parquet", ".parq", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    lead_col = "lead_h" if "lead_h" in df.columns else "lead"
    if lead_col not in df.columns:
        raise SystemExit(f"[error] lead column not found in threshold table ({lead_col})")
    if col not in df.columns:
        raise SystemExit(f"[error] column '{col}' not found in threshold table; available: {list(df.columns)}")
    row = df.loc[df[lead_col] == lead_h]
    if row.empty:
        print(f"[warn] no row with {lead_col}=={lead_h} in {thr_file}; skipping threshold.")
        return None
    val = float(row.iloc[0][col])
    print(f"[thr] loaded {col}={val} for lead={lead_h} from {thr_file}")
    return val


def _pick_col(cols: Sequence[str], cands: Sequence[str]) -> Optional[str]:
    for c in cands:
        if c in cols:
            return c
    return None


def _prep_seed_mask(df: pd.DataFrame, prob_col: Optional[str], flag_col: Optional[str], thr: Optional[float]) -> pd.Series:
    base = pd.Series(True, index=df.index)
    masks: List[pd.Series] = []
    if flag_col and flag_col in df.columns:
        masks.append(pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int) > 0)
    if prob_col and prob_col in df.columns and thr is not None and math.isfinite(thr):
        masks.append(pd.to_numeric(df[prob_col], errors="coerce") >= float(thr))
    if not masks:
        raise SystemExit(
            "No usable seed mask columns found (prob/flag missing). "
            f"prob_col={prob_col} flag_col={flag_col} thr={thr}. "
            "Provide a probability or flag column so seeds can be filtered; "
            "otherwise the entire panel would be treated as seeds."
        )
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def _make_state_id(df: pd.DataFrame, g_col: str, s_col: str, e_col: str) -> pd.DataFrame:
    def _n_states(col: str) -> int:
        if col not in df.columns:
            return 0
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return 0
        vmax = vals.max()
        if not math.isfinite(vmax):
            return 0
        return int(vmax) + 1

    nG = _n_states(g_col)
    nS = _n_states(s_col)
    nE = _n_states(e_col)
    if nG == 0 or nS == 0 or nE == 0:
        df["state_id"] = np.nan
        df["state_label"] = "G?S?E?"
        return df
    df["state_id"] = (df[g_col] * nS * nE + df[s_col] * nE + df[e_col]).astype("int64")
    df["state_label"] = "G" + df[g_col].astype(str) + "S" + df[s_col].astype(str) + "E" + df[e_col].astype(str)
    return df


def _compute_transitions(pts: pd.DataFrame, label: str) -> pd.DataFrame:
    if pts.empty or "state_label" not in pts.columns:
        return pd.DataFrame(columns=["from_state", "to_state", "count", "track_class"])
    g = pts.sort_values(["track_id", "time"]).copy()
    g["next_state"] = g.groupby("track_id")["state_label"].shift(-1)
    g = g.dropna(subset=["next_state"])
    out = (
        g.groupby(["state_label", "next_state"])
        .size()
        .reset_index(name="count")
        .rename(columns={"state_label": "from_state", "next_state": "to_state"})
    )
    out["track_class"] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track seed G/S/E states and shear role (supportive vs destructive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input panel (CSV/Parquet) with time/lat/lon and G/S/E columns.")
    ap.add_argument("--out-dir", default="results/seedmaps", help="Output directory for tables.")
    ap.add_argument("--run-name", default="run", help="Prefix for output files.")
    ap.add_argument("--time-col", default=None, help="Time column (auto-detected if not provided).")
    ap.add_argument("--lat-col", default="lat", help="Latitude column.")
    ap.add_argument("--lon-col", default="lon", help="Longitude column.")
    ap.add_argument(
        "--normalize-lon",
        choices=["none", "-180..180", "0..360"],
        type=lambda s: s.strip(),
        default="-180..180",
    )
    ap.add_argument("--prob-col", default=None, help="Probability column (auto-detected if missing).")
    ap.add_argument("--flag-col", default=None, help="Seed flag column (auto-detected if missing).")
    ap.add_argument("--thr", type=float, default=None, help="Explicit probability threshold for seed selection.")
    ap.add_argument(
        "--thr-file",
        default=None,
        help="Optional threshold table (CSV/Parquet) to pull --thr from (uses --thr-lead and --thr-column).",
    )
    ap.add_argument("--thr-lead", type=int, default=240, help="Lead (hours) to select in threshold table.")
    ap.add_argument("--thr-column", default="thr_Fbeta", help="Column in threshold table to use for --thr.")
    ap.add_argument("--g-col", default="G_struct", help="Geometry/structure column.")
    ap.add_argument("--s-col", default="S_shear", help="Shear magnitude column.")
    ap.add_argument("--e-col", default="E_energy", help="Energy/thermo column.")
    ap.add_argument(
        "--quantiles",
        default="0,0.2,0.5,0.8,1.0",
        help="Comma-separated quantiles for state bin edges (used if *_level columns missing).",
    )
    ap.add_argument("--round-dp", type=int, default=3, help="Lat/lon rounding for same-cell dG_dt.")
    ap.add_argument("--dg-eps", type=float, default=0.05, help="Epsilon for classifying dG_dt as growth/decay.")
    ap.add_argument("--shear-quantile", type=float, default=0.8, help="Quantile threshold for high shear.")
    ap.add_argument("--shear-thr", type=float, default=None, help="Absolute threshold override for high shear.")
    ap.add_argument("--shear-use-abs", action="store_true", help="Use |S| for shear thresholding (default: raw S).")
    ap.add_argument("--direction-col", default=None, help="Optional column for shear sign (e.g., zeta or gka_chirality).")
    ap.add_argument("--prob-bin-step", type=float, default=0.1, help="Bin width for prob deciles (set 0 to disable).")
    ap.add_argument("--slow-phase-bins", type=int, default=0, help="Optional slow phase bins (requires slow_cos/sin).")
    ap.add_argument("--link-radius-km", type=float, default=90.0, help="Max distance between hourly steps when linking.")
    ap.add_argument("--max-gap-hours", type=int, default=1, help="Max gap hours when linking tracks.")
    ap.add_argument("--min-track-hours", type=int, default=2, help="Drop tracks shorter than this many hours.")
    ap.add_argument("--storm-radius-km", type=float, default=150.0, help="Storm match radius when IBTrACS provided.")
    ap.add_argument("--lookahead-hours", type=int, default=120, help="Storm lookahead horizon for builder label.")
    ap.add_argument("--ibtracs", default=None, help="Optional IBTrACS CSV/Parquet for builder/dead split.")
    ap.add_argument("--time-format", default=None, help="Optional strptime format for non-standard time strings.")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Chunk rows for CSV/parquet reading (0 = full load).")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on seed rows after filtering.")
    ap.add_argument("--write-parquet", action="store_true", help="Also write Parquet copies of outputs.")
    args = ap.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip() != ""]
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise SystemExit("Quantiles must start at 0 and end at 1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve threshold if provided via table
    thr = args.thr
    if thr is None and args.thr_file:
        thr = _load_threshold(args.thr_file, args.thr_lead, args.thr_column)

    # Columns to load (keep narrow)
    wanted_cols = {args.lat_col, args.lon_col, args.g_col, args.s_col, args.e_col}
    if args.time_col:
        wanted_cols.add(args.time_col)
    wanted_cols.update(CANDIDATE_TIME_COLS)
    wanted_cols.update([c for c in CANDIDATE_PROB_COLS + [args.prob_col] if c])
    wanted_cols.update([c for c in CANDIDATE_FLAG_COLS + [args.flag_col] if c])
    wanted_cols.update(["G_level", "S_level", "E_level"])
    if args.direction_col:
        wanted_cols.add(args.direction_col)
    wanted_cols.update(["slow_cos", "slow_sin"])

    seeds: List[pd.DataFrame] = []
    n_total = 0
    tcol_detected: Optional[str] = None
    pcol_detected: Optional[str] = None
    fcol_detected: Optional[str] = None
    chunk_rows = args.chunk_rows if args.chunk_rows and args.chunk_rows > 0 else None

    for chunk in _iter_batches(args.panel, list(wanted_cols), chunk_rows):
        n_total += len(chunk)
        cols = chunk.columns
        if not tcol_detected:
            tcol_detected = args.time_col or _pick_col(cols, CANDIDATE_TIME_COLS)
        if not pcol_detected:
            pcol_detected = args.prob_col or _pick_col(cols, CANDIDATE_PROB_COLS)
        if not fcol_detected:
            fcol_detected = args.flag_col or _pick_col(cols, CANDIDATE_FLAG_COLS)
        if not tcol_detected:
            raise SystemExit("No time column found; pass --time-col.")
        chunk = chunk.copy()
        chunk["time"] = _to_utc_naive(chunk[tcol_detected], args.time_format)
        chunk["lat"] = pd.to_numeric(chunk[args.lat_col], errors="coerce")
        chunk["lon"] = _norm_lon(pd.to_numeric(chunk[args.lon_col], errors="coerce"), args.normalize_lon)
        mask_valid = chunk["time"].notna() & chunk["lat"].notna() & chunk["lon"].notna()
        seed_mask = _prep_seed_mask(chunk, pcol_detected, fcol_detected, thr)
        chunk = chunk.loc[mask_valid & seed_mask].reset_index(drop=True)
        if chunk.empty:
            continue
        seeds.append(chunk)
        print(f"[read] kept {len(chunk):,} seed rows (running total={sum(len(s) for s in seeds):,})", flush=True)
        if args.max_rows and sum(len(s) for s in seeds) >= args.max_rows:
            break

    if not seeds:
        print("[exit] no seed rows after filtering; nothing to do.")
        return

    df = pd.concat(seeds, ignore_index=True)
    if (thr is None) and (fcol_detected is None):
        print("[warn] no threshold/flag applied; all rows treated as seeds.")
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=int(args.max_rows), random_state=42).reset_index(drop=True)
        print(f"[info] downsampled seeds to {len(df):,} rows (max_rows={args.max_rows})")

    print(f"[info] loaded {len(df):,} seed rows from {n_total:,} total rows.")

    # Assign levels if missing
    for col, lvl in ((args.g_col, "G_level"), (args.s_col, "S_level"), (args.e_col, "E_level")):
        if lvl not in df.columns or df[lvl].isna().all():
            df[lvl] = _digitize_levels(df[col], quantiles)

    # Optional slow phase bins
    if args.slow_phase_bins and "slow_cos" in df and "slow_sin" in df:
        phase = np.arctan2(pd.to_numeric(df["slow_sin"], errors="coerce"),
                           pd.to_numeric(df["slow_cos"], errors="coerce"))
        bins_phase = np.linspace(-np.pi, np.pi, args.slow_phase_bins + 1)
        df["slow_phase_bin"] = pd.cut(phase, bins=bins_phase, labels=False, include_lowest=True)

    # dG_dt via same-cell diff (guard against >1h gaps)
    df["lat_r"] = df["lat"].round(args.round_dp)
    df["lon_r"] = df["lon"].round(args.round_dp)
    df = df.sort_values(["lat_r", "lon_r", "time"])
    dt_h = df.groupby(["lat_r", "lon_r"])["time"].diff().dt.total_seconds() / 3600.0
    dg = df.groupby(["lat_r", "lon_r"])[args.g_col].diff()
    df["dG_dt"] = dg.mask(dt_h.abs() > 1.1)
    df = df.drop(columns=["lat_r", "lon_r"])

    # Growth/decay label
    eps = float(args.dg_eps)
    df["growth_label"] = np.where(
        df["dG_dt"] > eps,
        "growth",
        np.where(df["dG_dt"] < -eps, "decay", "flat"),
    )

    # Shear threshold
    shear_basis = df[args.s_col].abs() if args.shear_use_abs else df[args.s_col]
    s_thr = float(args.shear_thr) if args.shear_thr is not None else float(shear_basis.quantile(args.shear_quantile))
    print(f"[info] shear threshold={s_thr:.4f} (source={'|S|' if args.shear_use_abs else 'S'})")
    df["high_shear"] = shear_basis >= s_thr

    # Directional shear (optional)
    if args.direction_col and args.direction_col in df.columns:
        df["S_eff"] = df[args.s_col] * np.sign(pd.to_numeric(df[args.direction_col], errors="coerce").fillna(0))
    else:
        df["S_eff"] = df[args.s_col]

    # Shear role
    df["shear_role"] = np.where(
        ~df["high_shear"],
        "low",
        np.where(
            df["growth_label"] == "growth",
            "supportive_high",
            np.where(df["growth_label"] == "decay", "destructive_high", "neutral_high"),
        ),
    )

    # Prob bins (optional deciles)
    if args.prob_bin_step and pcol_detected and pcol_detected in df.columns:
        step = float(args.prob_bin_step)
        edges = np.arange(0, 1 + step + 1e-6, step)
        df["prob_bin"] = pd.cut(pd.to_numeric(df[pcol_detected], errors="coerce"), bins=edges, labels=False, include_lowest=True)

    # Build state ids
    df = _make_state_id(df, "G_level", "S_level", "E_level")

    # Link into proto-tracks (keeps columns)
    linked = link_tracks(df.copy(), link_radius_km=args.link_radius_km, max_gap_hours=args.max_gap_hours)
    linked["hour_idx"] = linked.groupby("track_id")["time"].rank(method="first").astype(int) - 1

    # Filter short tracks
    counts = linked.groupby("track_id").size().rename("n_hours").reset_index()
    good_ids = counts.loc[counts["n_hours"] >= int(args.min_track_hours), "track_id"].tolist()
    linked = linked[linked["track_id"].isin(good_ids)].reset_index(drop=True)
    print(f"[info] kept {len(good_ids):,} tracks (>= {args.min_track_hours}h) with {len(linked):,} rows.")

    # Per-track summary
    agg = (
        linked.groupby("track_id")
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            duration_h=("time", lambda x: (x.max() - x.min()).total_seconds() / 3600.0 + 1.0),
            start_lat=("lat", "first"),
            start_lon=("lon", "first"),
            end_lat=("lat", "last"),
            end_lon=("lon", "last"),
            mean_G=(args.g_col, "mean"),
            mean_S=(args.s_col, "mean"),
            mean_E=(args.e_col, "mean"),
            frac_supportive=("shear_role", lambda x: (x == "supportive_high").mean()),
            frac_destructive=("shear_role", lambda x: (x == "destructive_high").mean()),
            frac_high_shear=("high_shear", "mean"),
            start_state=("state_label", "first"),
            end_state=("state_label", "last"),
        )
        .reset_index()
    )

    # Optional IBTrACS matching -> builder/dead split
    if args.ibtracs:
        ib, sid = load_ibtracs(args.ibtracs, normalize_lon=args.normalize_lon, time_fmt=args.time_format)
        outcomes = label_tracks_kdtree(
            linked[["track_id", "time", "lat", "lon"]].copy(),
            ib,
            sid,
            storm_radius_km=args.storm_radius_km,
            lookahead_hours=args.lookahead_hours,
        )
        left_tracks = agg
        agg = agg.merge(outcomes, on="track_id", how="left")
        left_dupe = int(left_tracks.duplicated(subset=["track_id"]).sum())
        right_dupe = int(outcomes.duplicated(subset=["track_id"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_tracks, outcomes, ["track_id"])
        entry = join_audit.build_entry(
            step="seeds.gse-tracks.outcomes-merge",
            keys=["track_id"],
            join_type="left",
            left_rows=len(left_tracks),
            right_rows=len(outcomes),
            out_rows=len(agg),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={"ibtracs_path": str(args.ibtracs)},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
        agg["is_builder"] = agg["matched"].fillna(False)
        left_linked = linked
        linked = linked.merge(agg[["track_id", "is_builder"]], on="track_id", how="left")
        left_dupe = int(left_linked.duplicated(subset=["track_id"]).sum())
        right_dupe = int(agg[["track_id", "is_builder"]].duplicated(subset=["track_id"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_linked, agg[["track_id", "is_builder"]], ["track_id"])
        entry = join_audit.build_entry(
            step="seeds.gse-tracks.builder-merge",
            keys=["track_id"],
            join_type="left",
            left_rows=len(left_linked),
            right_rows=len(agg[["track_id", "is_builder"]]),
            out_rows=len(linked),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
    else:
        agg["is_builder"] = False
        linked["is_builder"] = False

    # State transitions
    trans_all = _compute_transitions(linked, "all")
    trans_builder = _compute_transitions(linked[linked["is_builder"] == True], "builder")
    trans_dead = _compute_transitions(linked[linked["is_builder"] == False], "dead")
    transitions = pd.concat([trans_all, trans_builder, trans_dead], ignore_index=True)

    # Write outputs
    base = args.run_name
    out_pts = out_dir / f"{base}_gse_track_points.csv"
    out_tracks = out_dir / f"{base}_gse_track_summaries.csv"
    out_trans = out_dir / f"{base}_gse_state_transitions.csv"

    linked.to_csv(out_pts, index=False, date_format="%Y-%m-%d %H:%M:%S")
    agg.to_csv(out_tracks, index=False, date_format="%Y-%m-%d %H:%M:%S")
    transitions.to_csv(out_trans, index=False)

    if args.write_parquet:
        linked.to_parquet(str(out_pts) + ".parquet", index=False)
        agg.to_parquet(str(out_tracks) + ".parquet", index=False)
        transitions.to_parquet(str(out_trans) + ".parquet", index=False)

    print(f"[write] {out_pts} rows={len(linked):,}")
    print(f"[write] {out_tracks} rows={len(agg):,}")
    print(f"[write] {out_trans} rows={len(transitions):,}")


if __name__ == "__main__":
    main()
