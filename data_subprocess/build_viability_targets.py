#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_viability_targets.py

Derive viability (and optional intensification) targets from a GSE panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


# Agent: build labels without changing core feature maths.

def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _iter_file(path: str, chunksize: Optional[int]) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if pq is None:
            yield pd.read_parquet(path)
            return
        pf = pq.ParquetFile(path)
        if chunksize and chunksize > 0:
            for batch in pf.iter_batches(batch_size=int(chunksize)):
                yield batch.to_pandas()
        else:
            for rg in range(pf.num_row_groups):
                yield pf.read_row_group(rg).to_pandas()
        return

    if chunksize and chunksize > 0:
        for ch in pd.read_csv(path, low_memory=False, chunksize=int(chunksize)):
            yield ch
        return
    yield pd.read_csv(path, low_memory=False)


class _Writer:
    def __init__(self, dest: Path):
        self.dest = dest
        self._mode = "parquet" if _is_parquet(str(dest)) else "csv"
        self._header_written = False
        self._parquet_writer = None
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        if self.dest.exists():
            print(f"[warn] output {self.dest} exists; overwriting.")
            self.dest.unlink()
        if self._mode == "parquet" and pq is None:
            raise SystemExit("pyarrow is required for parquet output. Install pyarrow or choose CSV.")

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self._mode == "parquet":
            assert pa is not None and pq is not None
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.dest, table.schema)
            self._parquet_writer.write_table(table)
            return
        comp = "gzip" if self.dest.name.lower().endswith(".gz") else "infer"
        df.to_csv(
            self.dest,
            index=False,
            mode="w" if not self._header_written else "a",
            header=not self._header_written,
            compression=comp,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        self._header_written = True

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _estimate_g_min(path: str, g_col: str, quantile: float, chunksize: Optional[int], seed: int) -> float:
    rng = np.random.default_rng(seed)
    sample_limit = 1_000_000
    sample: np.ndarray = np.empty((0,), dtype=float)

    for chunk in _iter_file(path, chunksize):
        if g_col not in chunk.columns:
            raise SystemExit(f"G column '{g_col}' not found in panel.")
        vals = _num(chunk[g_col]).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if sample.size == 0:
            sample = vals[: min(len(vals), sample_limit)]
            continue
        concat = np.concatenate([sample, vals])
        if concat.size <= sample_limit:
            sample = concat
        else:
            idx = rng.choice(concat.size, size=sample_limit, replace=False)
            sample = concat[idx]

    if sample.size == 0:
        raise SystemExit("No finite values found for G_struct; cannot compute threshold.")
    g_min = float(np.nanquantile(sample, quantile))
    print(f"[g-threshold] estimated g_min (q={quantile}) = {g_min:.4f} from {sample.size:,} samples")
    return g_min


def _add_targets(
    chunk: pd.DataFrame, args: argparse.Namespace, g_min: float, lead_sign: str, lead_h_val: float
) -> pd.DataFrame:
    df = chunk.copy()
    lead = _num(df[args.lead_col]) if args.lead_col in df else np.nan
    g = _num(df[args.g_col]) if args.g_col in df else np.nan

    if lead_sign == "negative":
        viable_window = (lead < 0.0) & (lead >= -float(args.horizon_max))
    else:
        viable_window = (lead > 0.0) & (lead <= float(args.horizon_max))

    geom_ok = g >= g_min
    df["y_viable"] = (viable_window & geom_ok).astype("int8")
    if "lead_h" not in df.columns:
        df["lead_h"] = float(lead_h_val)

    # Add lead-derived helper features for downstream model scoring if not present.
    horizon = float(args.horizon_max)
    lead_clip = lead.clip(lower=0.0, upper=horizon)
    if "lead_norm" not in df.columns:
        df["lead_norm"] = 1.0 - (lead_clip / horizon)
    if "lead_inv" not in df.columns:
        df["lead_inv"] = 1.0 / (1.0 + lead_clip)
    if "G_lead_norm" not in df.columns and args.g_col in df:
        df["G_lead_norm"] = g * df["lead_norm"]

    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build viability targets from a GSE panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input GSE panel (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output panel with targets (CSV(.gz) or Parquet).")
    ap.add_argument("--g-col", default="G_struct", help="Geometry strength column.")
    ap.add_argument("--lead-col", default="t_to_storm_min_h", help="Lead/label column for horizon window.")
    ap.add_argument("--horizon-max", type=float, default=240.0, help="Max lead (hours) for viability window.")
    ap.add_argument("--g-min", type=float, default=None, help="Absolute G threshold. If set, bypass quantile.")
    ap.add_argument("--g-min-quantile", type=float, default=0.7, help="Quantile for G threshold when g-min not set.")
    ap.add_argument("--lead-h", type=float, default=None, help="Optional constant lead_h to add (default: horizon-max).")
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=400_000,
        help="Chunk size for streaming CSV or Parquet batches.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-threshold-json", default=None, help="Optional path to save g_min metadata.")
    args = ap.parse_args()

    # Pre-scan lead to detect sign convention for pre-storm window
    lead_all = []
    for ch in _iter_file(args.panel, args.chunksize):
        if ch is None or ch.empty:
            continue
        lead_all.append(_num(ch[args.lead_col]) if args.lead_col in ch else pd.Series([], dtype=float))
    lead_all = pd.concat(lead_all, ignore_index=True) if lead_all else pd.Series([], dtype=float)
    finite = lead_all[np.isfinite(lead_all)]
    horizon = float(args.horizon_max)
    mask_pos = (finite > 0.0) & (finite <= horizon)
    mask_neg = (finite < 0.0) & (finite >= -horizon)
    n_pos = int(mask_pos.sum())
    n_neg = int(mask_neg.sum())
    print(f"[diag] lead window counts: pos_window={n_pos} neg_window={n_neg} (H={horizon})")
    lead_sign = "negative" if n_pos == 0 and n_neg > 0 else "positive"
    print(f"[info] using {'lead<0' if lead_sign=='negative' else 'lead>0'} as viability window.")

    # Recompute g_min from the detected lead window only
    if args.g_min is not None:
        g_min = float(args.g_min)
        print(f"[g-threshold] using provided g_min = {g_min}")
    else:
        # Build a sample restricted to the lead window
        sample_vals = []
        for ch in _iter_file(args.panel, args.chunksize):
            if ch is None or ch.empty or args.g_col not in ch:
                continue
            lead = _num(ch[args.lead_col]) if args.lead_col in ch else pd.Series([], dtype=float)
            if lead_sign == "negative":
                mask_lead = (lead < 0.0) & (lead >= -horizon)
            else:
                mask_lead = (lead > 0.0) & (lead <= horizon)
            if mask_lead.any():
                sample_vals.append(_num(ch.loc[mask_lead, args.g_col]))
        if sample_vals:
            sample_arr = pd.concat(sample_vals, ignore_index=True)
            sample_arr = sample_arr[np.isfinite(sample_arr)]
            if len(sample_arr):
                g_min = float(sample_arr.quantile(args.g_min_quantile))
                print(
                    f"[g-threshold] estimated g_min (q={args.g_min_quantile}) = {g_min:.4f} "
                    f"from {len(sample_arr):,} lead-window samples"
                )
            else:
                g_min = 0.0
                print("[warn] no finite G samples in lead window; setting g_min=0.0")
        else:
            g_min = 0.0
            print("[warn] no lead-window samples found; setting g_min=0.0")
    print(f"[info] horizon_max={args.horizon_max} lead_col={args.lead_col}")

    writer = _Writer(Path(args.out))
    lead_h_val = float(args.lead_h) if args.lead_h is not None else float(args.horizon_max)
    total_rows = 0
    total_pos = 0
    # track whether we had any positives; if not, we will relax threshold at end
    fallback_used = False
    for i, chunk in enumerate(_iter_file(args.panel, args.chunksize), start=1):
        if chunk is None or chunk.empty:
            continue
        out_chunk = _add_targets(chunk, args, g_min, lead_sign, lead_h_val)
        pos_here = int(out_chunk["y_viable"].sum())
        total_pos += pos_here
        writer.write(out_chunk)
        total_rows += len(out_chunk)
        print(f"[targets] chunk {i}: wrote {len(out_chunk):,} rows (cum={total_rows:,})  pos={pos_here:,}")

    writer.close()
    print(f"[done] wrote {total_rows:,} rows -> {args.out} | positives={total_pos:,}")

    if total_pos == 0:
        # Fallback: relax quantile until we get positives; read once, rewrite.
        print("[warn] y_viable has zero positives; relaxing G threshold.")
        df_all = pd.concat(_iter_file(args.panel, args.chunksize), ignore_index=True)
        lead = _num(df_all[args.lead_col]) if args.lead_col in df_all else np.nan
        if lead_sign == "negative":
            funnel_mask = (lead < 0.0) & (lead >= -float(args.horizon_max))
        else:
            funnel_mask = (lead > 0.0) & (lead <= float(args.horizon_max))
        g_funnel = _num(df_all.loc[funnel_mask, args.g_col]) if args.g_col in df_all else pd.Series([], dtype=float)
        tried = []
        for q in [0.6, 0.5, 0.4, 0.3, 0.2]:
            if g_funnel.empty:
                break
            g_min_fallback = g_funnel.quantile(q)
            m_viable = funnel_mask & (_num(df_all[args.g_col]) >= g_min_fallback)
            n_pos = int(m_viable.sum())
            tried.append((q, g_min_fallback, n_pos))
            print(f"[fallback] q={q} g_min={g_min_fallback:.4f} positives={n_pos}")
            if n_pos > 0:
                df_all["y_viable"] = m_viable.astype("int8")
                horizon = float(args.horizon_max)
                lead_clip = lead.clip(lower=0.0, upper=horizon)
                df_all["lead_norm"] = 1.0 - (lead_clip / horizon)
                df_all["lead_inv"] = 1.0 / (1.0 + lead_clip)
                df_all["G_lead_norm"] = _num(df_all[args.g_col]) * df_all["lead_norm"]
                df_all["lead_h"] = lead_h_val
                df_all.to_parquet(args.out, index=False) if _is_parquet(args.out) else df_all.to_csv(
                    args.out, index=False
                )
                print(f"[targets] fallback succeeded with q={q}; rewrote {args.out} positives={n_pos:,}")
                fallback_used = True
                break
        if not fallback_used:
            if funnel_mask.any():
                df_all["y_viable"] = funnel_mask.astype("int8")
                horizon = float(args.horizon_max)
                lead_clip = lead.clip(lower=0.0, upper=horizon)
                df_all["lead_norm"] = 1.0 - (lead_clip / horizon)
                df_all["lead_inv"] = 1.0 / (1.0 + lead_clip)
                df_all["G_lead_norm"] = _num(df_all[args.g_col]) * df_all["lead_norm"]
                df_all["lead_h"] = lead_h_val
                df_all.to_parquet(args.out, index=False) if _is_parquet(args.out) else df_all.to_csv(
                    args.out, index=False
                )
                n_pos = int(funnel_mask.sum())
                print(f"[fallback] using funnel-only label; positives={n_pos:,}")
            else:
                print("[fatal] No rows within lead window; cannot build viability labels.")

    if args.save_threshold_json:
        meta = {"g_min": g_min, "g_min_quantile": args.g_min_quantile, "seed": args.seed, "lead_h": lead_h_val}
        Path(args.save_threshold_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_threshold_json).write_text(json.dumps(meta, indent=2))
        print(f"[meta] saved threshold info -> {args.save_threshold_json}")


if __name__ == "__main__":
    main()
