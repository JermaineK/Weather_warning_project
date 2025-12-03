#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_gka_multiscale.py - multi-scale spiral probe, CSV/Parquet streaming

This is a second-layer GKA pass that assumes an augmented grid table
(e.g., build_features_grid.py + bulk shear + first-layer GKA),
and derives scalar indicators of how "spiral-like" the local state is.

It implements a refined GKA definition:
  1) coherence: spin-dominated vs divergent (spiral vs source/sink)
  2) build vs relax: how S (or S3) grows/decays in time
  3) parity / knee: local S / S3 structure, scale contrast
  4) shear quenching: spiral survivorship under shear

Inputs (defaults; aliases supported):
  - zeta_mean3h, div_mean3h
  - S3
  - dS_dt
  - shear10_def or shear_low / shear_deep / S3
  - msl (optional, reserved)

Outputs (added columns):
  - gka_spin_coh       in [0,1]  spin vs div coherence
  - gka_build          in [0,1]  growth of S (build mode)
  - gka_relax          in [0,1]  decay of S (relax mode)
  - gka_shear_quench   in [0,1]  how shear-suppressed the environment is
  - gka_knee_ms        in [0,1]  multi-scale knee-ish contrast from S/S3
  - gka_score          in [0,1]  composite spiral-alignment score

Design notes:
  - CSV(.gz) handled via pandas chunk streaming.
  - Parquet handled via pyarrow ParquetFile batch iteration; writing out row-groups.
  - Only appends new columns; does not drop existing ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_ARROW = True
except Exception:
    HAVE_ARROW = False


NEW_COLS = [
    "gka_spin_coh",
    "gka_build",
    "gka_relax",
    "gka_shear_quench",
    "gka_knee_ms",
    "gka_score",
]

ALIASES: Dict[str, List[str]] = {
    "zeta_mean3h": ["zeta_mean3h"],
    "div_mean3h": ["div_mean3h"],
    "S3": ["S3"],
    "dS_dt": ["dS_dt", "dSdt"],
    "shear": ["shear10_def", "shear_low", "shear_deep", "shear_proxy", "S3"],
    "msl": ["msl", "mean_sea_level_pressure", "MSL", "mslp"],
    # prefer first-layer knee; S_mean3h remains a fallback
    "knee": ["gka_knee_ratio", "S_mean3h"],
}


def _first_present(cols: Iterable[str], pool: List[str]) -> Optional[str]:
    s = set(cols)
    for c in pool:
        if c in s:
            return c
    return None


def _bind_columns(cols: Iterable[str]) -> Dict[str, Optional[str]]:
    cols = list(cols)
    return {canon: _first_present(cols, choices) for canon, choices in ALIASES.items()}


def _safe_num(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    med = np.nanmedian(x)
    mad = np.nanmean(np.abs(x - med)) + 1e-6
    return (x - med) / mad


def _robust01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    q1, q99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if not np.isfinite(q1) or not np.isfinite(q99) or q99 <= q1:
        return np.zeros_like(x, dtype=float)
    y = (x - q1) / (q99 - q1 + 1e-12)
    return np.clip(y, 0.0, 1.0)


def _compute_gka_multiscale(df: pd.DataFrame, bind: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()

    def col(name: str) -> Optional[pd.Series]:
        c = bind.get(name)
        return out[c] if c and c in out.columns else None

    zeta_m = col("zeta_mean3h")
    div_m = col("div_mean3h")
    S3 = col("S3")
    dS = col("dS_dt")
    shear = col("shear")
    knee = col("knee")

    # 1. Spin coherence: |zeta| / (|zeta| + |div|)
    if (zeta_m is not None) and (div_m is not None):
        z = _safe_num(zeta_m).to_numpy(float)
        d = _safe_num(div_m).to_numpy(float)
        denom = np.abs(z) + np.abs(d) + 1e-12
        spin_coh = np.abs(z) / denom
    else:
        spin_coh = np.zeros(len(out), dtype=float)
    out["gka_spin_coh"] = spin_coh

    # 2. Build / relax from dS_dt
    if dS is not None:
        dS_vals = _safe_num(dS).to_numpy(float)
        z = _robust_z(dS_vals)
        build = 1.0 / (1.0 + np.exp(-z))
        relax = 1.0 / (1.0 + np.exp(z))
    else:
        build = np.full(len(out), 0.5, dtype=float)
        relax = np.full(len(out), 0.5, dtype=float)
    out["gka_build"] = build
    out["gka_relax"] = relax

    # 3. Shear quench: high when shear is relatively low
    if shear is not None:
        s = _safe_num(shear).to_numpy(float)
        shear_norm = _robust01(s)
        shear_quench = 1.0 - shear_norm
    else:
        shear_quench = np.zeros(len(out), dtype=float)
    out["gka_shear_quench"] = shear_quench

    # 4. Multi-scale knee proxy from either gka_knee_ratio or S/S3 fallback
    if knee is not None:
        kv = _safe_num(knee).to_numpy(float)
        klog = np.log10(np.abs(kv) + 1e-9)
        knee_ms = _robust01(klog)
    elif S3 is not None and zeta_m is not None:
        z = np.abs(_safe_num(zeta_m).to_numpy(float))
        s3 = np.abs(_safe_num(S3).to_numpy(float))
        denom = np.where(s3 < 1e-9, 1e-9, s3)
        ratio = z / denom
        knee_ms = _robust01(np.log10(ratio + 1e-9))
    else:
        knee_ms = np.zeros(len(out), dtype=float)
    out["gka_knee_ms"] = knee_ms

    # 5. Composite spiral score
    raw = (
        0.35 * spin_coh +
        0.25 * build +
        0.20 * shear_quench +
        0.20 * knee_ms
    )
    score = _robust01(raw)
    out["gka_score"] = score

    for c in NEW_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


def _is_parquet(path: str | Path) -> bool:
    p = str(path).lower()
    return p.endswith((".parquet", ".parq", ".pq", ".pqt"))


def _comp_for_csv(path: str | Path) -> str:
    p = str(path).lower()
    return "gzip" if p.endswith(".gz") else "infer"


def _peek_columns(path: str | Path) -> List[str]:
    if _is_parquet(path):
        if not HAVE_ARROW:
            raise SystemExit("[GKA-MS] pyarrow is required for Parquet input.")
        pf = pq.ParquetFile(path)
        return pf.schema.names
    head = pd.read_csv(path, nrows=5, low_memory=False)
    return list(head.columns)


def _stream_csv(path: str | Path, chunksize: int):
    parse_dates = ["time"]
    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        low_memory=False,
        compression="infer",
        parse_dates=parse_dates,
    ):
        yield chunk


def _stream_parquet(path: str | Path, rows_per_batch: int):
    if not HAVE_ARROW:
        raise SystemExit("[GKA-MS] pyarrow is required for Parquet streaming.")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=rows_per_batch):
        df = batch.to_pandas()
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
        yield df


def _write_stream_csv(path: str | Path, df: pd.DataFrame, first: bool) -> None:
    comp = _comp_for_csv(path)
    mode = "w" if first else "a"
    df.to_csv(
        path,
        index=False,
        mode=mode,
        header=first,
        compression=comp,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-scale geometric kernel features (spiral probe) on augmented grid tables."
    )
    ap.add_argument("--infile", required=True, help="Input CSV(.gz)/Parquet with patched features")
    ap.add_argument("--outfile", required=True, help="Output CSV(.gz)/Parquet with extra gka_* columns")
    ap.add_argument("--chunksize", type=int, default=400_000, help="Rows per chunk when reading CSV(.gz).")
    ap.add_argument("--parquet-rows", type=int, default=250_000, help="Rows per Parquet batch when streaming.")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Alias for orchestrator compatibility; uses chunksize if set.")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing an existing outfile.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--require-core",
        action="store_true",
        help="Abort if no usable core columns (zeta/div/S3/dS_dt/shear/knee) are bound.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # normalize chunk sizing knobs
    chunk_rows = args.chunk_rows or args.chunksize
    parquet_rows = args.parquet_rows
    if chunk_rows <= 0:
        chunk_rows = args.chunksize

    in_is_parq = _is_parquet(args.infile)
    out_is_parq = _is_parquet(args.outfile)

    out_path = Path(args.outfile)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"[GKA-MS] Outfile exists; use --overwrite: {out_path}")

    cols = _peek_columns(args.infile)
    bind = _bind_columns(cols)
    if args.verbose:
        print("\n[GKA-MS] Column bindings:")
        for k in ("zeta_mean3h", "div_mean3h", "S3", "dS_dt", "shear", "msl", "knee"):
            v = bind.get(k)
            print(f"  {k:12s} -> {v if v else '(missing)'}", flush=True)

    if args.require_core:
        core_keys = ("zeta_mean3h", "div_mean3h", "S3", "dS_dt", "shear", "knee")
        if not any(bind.get(k) for k in core_keys):
            raise SystemExit("[GKA-MS] --require-core set but no usable core columns bound.")

    if not in_is_parq and not out_is_parq:
        first = True
        total = 0
        for i, chunk in enumerate(_stream_csv(args.infile, chunk_rows), start=1):
            out_df = _compute_gka_multiscale(chunk, bind)
            _write_stream_csv(args.outfile, out_df, first=first)
            first = False
            total += len(out_df)
            if args.verbose and (i % 5 == 0):
                print(f"[GKA-MS] processed {total:,} rows", flush=True)
        print(f"[GKA-MS] Wrote {args.outfile}  rows={total:,}")
        return

    if in_is_parq and out_is_parq:
        if not HAVE_ARROW:
            raise SystemExit("[GKA-MS] pyarrow is required for Parquet streaming.")
        if out_path.exists():
            out_path.unlink()
        writer = None
        total = 0
        for i, chunk in enumerate(_stream_parquet(args.infile, parquet_rows), start=1):
            out_df = _compute_gka_multiscale(chunk, bind)
            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            total += len(out_df)
            if args.verbose and (i % 5 == 0):
                print(f"[GKA-MS] row-groups written, rows={total:,}", flush=True)
        if writer is not None:
            writer.close()
        print(f"[GKA-MS] Wrote {args.outfile}  rows={total:,}")
        return

    if in_is_parq and not out_is_parq:
        first = True
        total = 0
        for i, chunk in enumerate(_stream_parquet(args.infile, parquet_rows), start=1):
            out_df = _compute_gka_multiscale(chunk, bind)
            _write_stream_csv(args.outfile, out_df, first=first)
            first = False
            total += len(out_df)
            if args.verbose and (i % 5 == 0):
                print(f"[GKA-MS] Parquet->CSV rows={total:,}", flush=True)
        print(f"[GKA-MS] Wrote {args.outfile}  rows={total:,}")
        return

    if not in_is_parq and out_is_parq:
        df = pd.read_csv(args.infile, low_memory=False, compression="infer")
        out_df = _compute_gka_multiscale(df, bind)
        out_df.to_parquet(out_path, index=False)
        print(f"[GKA-MS] Wrote {args.outfile}  rows={len(out_df):,}")
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR [{type(e).__name__}]: {e}", file=sys.stderr)
        sys.exit(1)
