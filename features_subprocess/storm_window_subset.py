#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storm_window_subset.py — crop a feature grid to the UNION OF PER-STORM WINDOWS.

Motivation
    A full 4-month, full-domain feature chain costs ~112 GB per year. Every
    downstream genesis analysis (trigger, LOSO skill, GSE stack, backtrace)
    only ever looks at cells within a few degrees and a few days of a storm, so
    building the whole domain is wasted disk and compute.

    Inserted immediately after the `build` step, this keeps only rows inside
    ANY storm's space-time window:

        |lat - track_lat| <= pad_deg  AND  |lon - track_lon| <= pad_deg
        t0 - pre_h  <=  time  <=  t1 + post_h

    The union of those windows is typically ~5-10% of the full grid, so the rest
    of the chain (patch -> gka -> spherical -> gka-ms -> labels) runs on a small
    fraction of the rows. Neighbourhood/rolling operations downstream stay valid
    because pad_deg (default 6 deg) and pre_h (default 120 h) are far larger than
    any kernel radius (<= 2 cells) or rolling window (<= 24 h) used later.

USAGE (via features manager)
    python features_manager.py storm-window-subset \\
        --infile data/y2021/features_eoi.parquet \\
        --tracks data/tracks/tracks_2021.parquet \\
        --outfile data/y2021/features_eoi_win.parquet \\
        --pad-deg 6 --pre-h 120 --post-h 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_tracks(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).lower().endswith((".parquet", ".parq", ".pq")) \
        else pd.read_csv(path, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["time", "lat", "lon"])


def build_windows(tr: pd.DataFrame, pad: float, pre_h: float, post_h: float):
    """One space-time window per storm: (t_start, t_end, latS, latN, lonW, lonE)."""
    wins = []
    key = "storm_id" if "storm_id" in tr.columns else None
    groups = tr.groupby(tr[key].astype(str)) if key else [("all", tr)]
    for sid, g in groups:
        wins.append((
            g["time"].min() - pd.Timedelta(hours=pre_h),
            g["time"].max() + pd.Timedelta(hours=post_h),
            g["lat"].min() - pad, g["lat"].max() + pad,
            g["lon"].min() - pad, g["lon"].max() + pad,
        ))
    return wins


def mask_for(df: pd.DataFrame, wins) -> np.ndarray:
    t = pd.to_datetime(df["time"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    keep = np.zeros(len(df), dtype=bool)
    for (t0, t1, laS, laN, loW, loE) in wins:
        keep |= ((t >= t0) & (t <= t1)
                 & (lat >= laS) & (lat <= laN)
                 & (lon >= loW) & (lon <= loE)).to_numpy()
    return keep


def main() -> int:
    args = parse_args()
    outp = Path(args.outfile)
    if outp.exists() and not args.overwrite:
        raise SystemExit(f"[win] outfile exists; use --overwrite: {outp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    tr = read_tracks(args.tracks)
    wins = build_windows(tr, args.pad_deg, args.pre_h, args.post_h)
    print(f"[win] {len(wins)} storm window(s) from {args.tracks}")
    for i, (t0, t1, laS, laN, loW, loE) in enumerate(wins, 1):
        print(f"[win]   {i}: {t0:%Y-%m-%d %H:%M}..{t1:%Y-%m-%d %H:%M} "
              f"lat[{laS:.1f},{laN:.1f}] lon[{loW:.1f},{loE:.1f}]")

    pf = pq.ParquetFile(args.infile)
    writer = None
    kept = total = 0
    try:
        import pyarrow as pa
        for batch in pf.iter_batches(batch_size=args.batch_rows):
            df = batch.to_pandas()
            total += len(df)
            m = mask_for(df, wins)
            if not m.any():
                continue
            sub = df.loc[m]
            kept += len(sub)
            table = pa.Table.from_pandas(sub, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(outp), table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if kept == 0:
        raise SystemExit("[win] no rows fell inside any storm window; check tracks/inputs.")
    print(f"[win] kept {kept:,} / {total:,} rows ({kept/max(total,1):.1%}) -> {outp}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Crop a feature grid to the union of per-storm space-time windows.")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--post-h", type=float, default=24.0)
    ap.add_argument("--batch-rows", type=int, default=500_000)
    ap.add_argument("--overwrite", action="store_true")
    # orchestrator compatibility
    ap.add_argument("--chunk-rows", type=int, default=0)
    ap.add_argument("--parquet-rows", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
