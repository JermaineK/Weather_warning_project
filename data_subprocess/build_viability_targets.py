#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_viability_targets.py

Derive viability (lead-window) or commitment (post-knee parity lock) targets
from a GSE panel.
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


def _parse_list_spec(spec: str | None) -> list[int]:
    if not spec:
        return []
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    vals = sorted({int(p) for p in parts if p.strip()})
    return vals


def _future_or_window(values: np.ndarray, horizon: int) -> np.ndarray:
    """
    Future-window OR over the next `horizon` steps (excluding current step).
    Assumes `values` are ordered in time. Returns int8 array.
    """
    if horizon <= 0:
        return np.zeros_like(values, dtype=np.int8)
    series = pd.Series(values, copy=False)
    # Reverse to reuse backward-looking rolling; shift(1) excludes current time.
    rev = series.iloc[::-1].shift(1)
    fut = rev.rolling(window=int(horizon), min_periods=1).max()
    out = fut.iloc[::-1].fillna(0).to_numpy(dtype=np.int8, copy=False)
    return out


def _future_or_by_group(
    df: pd.DataFrame,
    col: str,
    horizon: int,
    group_cols: tuple[str, str] = ("lat", "lon"),
) -> pd.Series:
    """Per-(lat,lon) future-window OR for a binary-ish column."""
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int8")
    out = np.zeros(len(df), dtype=np.int8)
    vals = _num(df[col]).fillna(0.0).to_numpy()
    idx_map = df.groupby(list(group_cols), sort=False).indices
    for _, idx in idx_map.items():
        ii = np.asarray(idx, dtype=np.int64)
        v = (vals[ii] > 0).astype(np.int8)
        out[ii] = _future_or_window(v, horizon)
    return pd.Series(out, index=df.index, dtype="int8")


def _lead_bucket(lead_vals: pd.Series, edges: list[int]) -> pd.Series:
    clean = pd.to_numeric(lead_vals, errors="coerce")
    if not edges or len(edges) < 2:
        return pd.Series(np.nan, index=lead_vals.index)
    bins = sorted({int(x) for x in edges})
    if bins[0] != 0:
        bins = [0] + bins
    labels = [int(x) for x in bins[1:]]
    out = pd.cut(clean, bins=bins, labels=labels, include_lowest=True, right=True)
    return pd.to_numeric(out.astype(str), errors="coerce")


def _peek_columns(path: str, chunksize: Optional[int]) -> list[str]:
    for ch in _iter_file(path, chunksize):
        if ch is None or ch.empty:
            continue
        return list(ch.columns)
    return []


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
    chunk: pd.DataFrame,
    args: argparse.Namespace,
    g_min: float,
    lead_sign: str,
    lead_h_val: float,
    target_mode: str,
    target_col: str,
) -> pd.DataFrame:
    df = chunk.copy()
    lead = _num(df[args.lead_col]) if args.lead_col in df else pd.Series(np.nan, index=df.index)

    if target_mode == "commitment":
        knee = _num(df[args.commitment_knee_col]) if args.commitment_knee_col in df else pd.Series(np.nan, index=df.index)
        lock = _num(df[args.commitment_lock_col]) if args.commitment_lock_col in df else pd.Series(np.nan, index=df.index)
        post_knee = knee >= float(args.commitment_post_state)
        lock_ok = lock > 0.0
        coh_ok = pd.Series(True, index=df.index)
        if args.commitment_sai_thr is not None and args.commitment_sai_col in df:
            coh_ok = _num(df[args.commitment_sai_col]) >= float(args.commitment_sai_thr)
        df[target_col] = (post_knee & lock_ok & coh_ok).astype("int8")
    else:
        g = _num(df[args.g_col]) if args.g_col in df else np.nan
        if lead_sign == "negative":
            viable_window = (lead < 0.0) & (lead >= -float(args.horizon_max))
        else:
            viable_window = (lead > 0.0) & (lead <= float(args.horizon_max))
        geom_ok = g >= g_min
        df[target_col] = (viable_window & geom_ok).astype("int8")

    if "lead_h" not in df.columns:
        df["lead_h"] = lead if args.lead_col in df else float(lead_h_val)
    if "lead_h_bucket" not in df.columns:
        df["lead_h_bucket"] = _lead_bucket(lead, args.lead_bins_list)

    # Add lead-derived helper features for downstream model scoring if requested.
    if target_mode != "commitment" or args.include_lead_features:
        horizon = float(args.horizon_max)
        lead_clip = lead.clip(lower=0.0, upper=horizon)
        if "lead_norm" not in df.columns:
            df["lead_norm"] = 1.0 - (lead_clip / horizon)
        if "lead_inv" not in df.columns:
            df["lead_inv"] = 1.0 / (1.0 + lead_clip)
        if "G_lead_norm" not in df.columns and args.g_col in df:
            g = _num(df[args.g_col])
            df["G_lead_norm"] = g * df["lead_norm"]

    return df


def _add_knee_forecast_targets(
    df: pd.DataFrame,
    args: argparse.Namespace,
    lead_hours: list[int],
) -> pd.DataFrame:
    """
    Add future-window knee targets per lead:
      - y_knee_cross_{H}h
      - y_lock_stable_{H}h
      - y_commit_{H}h (AND of above)
    Assumes df is a contiguous set of (lat,lon) groups.

    Definition: y_knee_cross_{H}h(t)=1 iff knee crossing occurs in (t, t+H].
    This is strictly future-only; features must be computed from times <= t.
    """
    # Agent: knee-forecast labels are future-window only (causal features remain past-only).
    out = df.copy()
    knee_col = args.knee_cross_col
    lock_col = args.lock_col
    for H in lead_hours:
        if H <= 0:
            continue
        y_knee = _future_or_by_group(out, knee_col, int(H))
        y_lock = _future_or_by_group(out, lock_col, int(H))
        out[f"y_knee_cross_{int(H)}h"] = y_knee
        out[f"y_lock_stable_{int(H)}h"] = y_lock
        out[f"y_commit_{int(H)}h"] = (y_knee & y_lock).astype("int8")
    return out


def _add_lead_helpers(
    df: pd.DataFrame,
    args: argparse.Namespace,
    lead_vals: pd.Series,
) -> pd.DataFrame:
    out = df
    if "lead_h" not in out.columns:
        out["lead_h"] = lead_vals
    if "lead_h_bucket" not in out.columns:
        out["lead_h_bucket"] = _lead_bucket(lead_vals, args.lead_bins_list)
    if args.include_lead_features:
        horizon = float(args.horizon_max)
        lead_clip = lead_vals.clip(lower=0.0, upper=horizon)
        if "lead_norm" not in out.columns:
            out["lead_norm"] = 1.0 - (lead_clip / horizon)
        if "lead_inv" not in out.columns:
            out["lead_inv"] = 1.0 / (1.0 + lead_clip)
        if "G_lead_norm" not in out.columns and args.g_col in out.columns:
            out["G_lead_norm"] = _num(out[args.g_col]) * out["lead_norm"]
    return out


def _stream_knee_forecast_targets(
    path: str,
    args: argparse.Namespace,
    lead_hours: list[int],
    has_lead_col: bool,
) -> None:
    """
    Streaming knee-forecast target builder. Requires input sorted by (lat,lon,time).
    """
    writer = _Writer(Path(args.out))
    total_rows = 0
    summary_counts = None
    if lead_hours:
        summary_counts = {p: {int(h): 0 for h in lead_hours} for p in ("y_knee_cross", "y_lock_stable", "y_commit")}
    carry = None
    last_key = None

    for chunk in _iter_file(path, args.chunksize):
        if chunk is None or chunk.empty:
            continue
        df = chunk
        if args.time_col in df.columns:
            df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_localize(None)
        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], ignore_index=False)

        # Detect ordering drift (best-effort warning).
        key_now = (df["lat"].iloc[0], df["lon"].iloc[0])
        if last_key is not None and key_now < last_key:
            print(
                "[warn] knee-forecast input appears unsorted by (lat,lon); "
                "labels may be incorrect. Rebuild gse-lagged to ensure sorting."
            )
        last_key = (df["lat"].iloc[-1], df["lon"].iloc[-1])

        # Split off the last group to carry across chunks.
        last_lat, last_lon = df["lat"].iloc[-1], df["lon"].iloc[-1]
        mask_last = (df["lat"] == last_lat) & (df["lon"] == last_lon)
        to_process = df.loc[~mask_last]
        carry = df.loc[mask_last]

        if not to_process.empty:
            processed = _add_knee_forecast_targets(to_process, args, lead_hours)
            if has_lead_col:
                processed = _add_lead_helpers(processed, args, _num(processed[args.lead_col]))
            if summary_counts is not None:
                for prefix, counts in summary_counts.items():
                    for h in lead_hours:
                        col = f"{prefix}_{int(h)}h"
                        if col not in processed.columns:
                            continue
                        vals = pd.to_numeric(processed[col], errors="coerce").fillna(0).astype(int)
                        counts[int(h)] += int(vals.sum())
            writer.write(processed)
            total_rows += len(processed)
            print(f"[targets] wrote {len(processed):,} rows (cum={total_rows:,})")

    if carry is not None and not carry.empty:
        processed = _add_knee_forecast_targets(carry, args, lead_hours)
        if has_lead_col:
            processed = _add_lead_helpers(processed, args, _num(processed[args.lead_col]))
        if summary_counts is not None:
            for prefix, counts in summary_counts.items():
                for h in lead_hours:
                    col = f"{prefix}_{int(h)}h"
                    if col not in processed.columns:
                        continue
                    vals = pd.to_numeric(processed[col], errors="coerce").fillna(0).astype(int)
                    counts[int(h)] += int(vals.sum())
        writer.write(processed)
        total_rows += len(processed)
        print(f"[targets] wrote {len(processed):,} rows (cum={total_rows:,})")

    writer.close()
    print(f"[done] wrote {total_rows:,} rows -> {args.out}")
    if summary_counts is not None:
        for prefix, counts in summary_counts.items():
            rows = []
            for h in sorted(lead_hours):
                pos = counts.get(int(h), 0)
                frac = float(pos / total_rows) if total_rows else float("nan")
                rows.append({"lead_h": int(h), "pos_count": pos, "pos_frac": frac})
            if rows:
                summary = pd.DataFrame(rows)
                print(f"[diag] per-lead label summary ({prefix}):")
                try:
                    print(summary.to_string(index=False))
                except Exception:
                    print(summary)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build viability or commitment targets from a GSE panel.",
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
        "--lead-hours",
        type=str,
        default="24,48,72,120,240",
        help="Comma-separated leads for label summary (hours).",
    )
    ap.add_argument(
        "--lead-bins",
        type=str,
        default="0,24,48,72,120,240",
        help="Comma-separated lead-hour bin edges for lead_h_bucket.",
    )
    ap.add_argument(
        "--target-mode",
        choices=["lead-window", "commitment", "knee-forecast"],
        default="lead-window",
        help="Target definition: lead-window (storm lead) or commitment (post-knee parity lock).",
    )
    ap.add_argument("--target-col", default=None, help="Override target column name.")
    ap.add_argument("--commitment-knee-col", default="gka_knee_state", help="Knee state column for commitment.")
    ap.add_argument("--commitment-lock-col", default="gka_parity_lock", help="Parity lock column for commitment.")
    ap.add_argument("--commitment-post-state", type=int, default=2, help="Knee state threshold for post-knee.")
    ap.add_argument("--commitment-sai-col", default="gka_SAI", help="Optional coherence column for commitment.")
    ap.add_argument("--commitment-sai-thr", type=float, default=None, help="Optional minimum gka_SAI for commitment.")
    ap.add_argument("--knee-cross-col", default="gka_knee_cross", help="Knee-cross indicator column.")
    ap.add_argument("--lock-col", default="gka_parity_lock", help="Parity lock indicator column.")
    ap.add_argument("--time-col", default="time", help="Time column for ordering knee targets.")
    ap.add_argument(
        "--include-lead-features",
        action="store_true",
        help="Include lead-derived features even in commitment mode.",
    )
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=None,
        help="Chunk size for streaming CSV or Parquet batches.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-threshold-json", default=None, help="Optional path to save g_min metadata.")
    ap.add_argument(
        "--save-targets-meta",
        default=None,
        help="Optional path to save target-definition metadata JSON.",
    )
    # Agent: accept overwrite flag for pipeline compatibility (writer overwrites by default).
    ap.add_argument("--overwrite", action="store_true", help="No-op; output is overwritten if present.")
    args = ap.parse_args()
    if args.chunksize in (None, 0):
        # Agent: prefer pipeline/global chunking; fall back to a safe default only if unset.
        args.chunksize = 400_000

    args.lead_bins_list = _parse_list_spec(args.lead_bins)
    lead_hours = [h for h in _parse_list_spec(args.lead_hours) if h > 0]

    target_mode = args.target_mode
    if target_mode == "knee-forecast":
        lead_for_target = max(lead_hours) if lead_hours else int(args.horizon_max)
        target_col = args.target_col or f"y_knee_cross_{int(lead_for_target)}h"
    else:
        target_col = args.target_col or ("y_viable" if target_mode == "lead-window" else "y_commit")
    args.target_col = target_col
    print(f"[info] target_mode={target_mode} target_col={target_col}")

    if target_mode == "knee-forecast":
        cols = _peek_columns(args.panel, args.chunksize)
        required = [args.knee_cross_col, args.lock_col, "lat", "lon"]
        missing = [c for c in required if c not in cols]
        if missing:
            raise SystemExit(
                "[targets] knee-forecast mode requires columns "
                f"{missing}; run state-transitions first."
            )
        if args.time_col not in cols:
            print(
                f"[warn] time column '{args.time_col}' not found; "
                "knee-forecast assumes panel is already sorted by time."
            )
        # Knee-forecast ignores g_min / lead window logic; labels are future-window of knee/lock.
        has_lead_col = args.lead_col in cols
        _stream_knee_forecast_targets(args.panel, args, lead_hours, has_lead_col)
        return
    elif target_mode == "commitment":
        cols = _peek_columns(args.panel, args.chunksize)
        missing = [c for c in (args.commitment_knee_col, args.commitment_lock_col) if c not in cols]
        if args.commitment_sai_thr is not None and args.commitment_sai_col not in cols:
            missing.append(args.commitment_sai_col)
        if missing:
            raise SystemExit(
                "[targets] commitment mode requires columns "
                f"{missing}; run state-transitions (and GKA features) first."
            )
        lead_sign = "positive"
        g_min = 0.0
    else:
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
    lead_counts = {h: 0 for h in lead_hours}
    lead_totals = {h: 0 for h in lead_hours}
    # track whether we had any positives; if not, we will relax threshold at end
    fallback_used = False
    for i, chunk in enumerate(_iter_file(args.panel, args.chunksize), start=1):
        if chunk is None or chunk.empty:
            continue
        out_chunk = _add_targets(chunk, args, g_min, lead_sign, lead_h_val, target_mode, target_col)
        pos_here = int(out_chunk[target_col].sum())
        total_pos += pos_here
        if lead_hours and args.lead_col in out_chunk:
            lead_vals = _num(out_chunk[args.lead_col])
            y_vals = pd.to_numeric(out_chunk[target_col], errors="coerce").fillna(0).astype(int)
            for h in lead_hours:
                if lead_sign == "negative":
                    mask = (lead_vals < 0.0) & (lead_vals >= -float(h))
                else:
                    mask = (lead_vals > 0.0) & (lead_vals <= float(h))
                lead_totals[h] += int(mask.sum())
                lead_counts[h] += int((mask & (y_vals == 1)).sum())
        writer.write(out_chunk)
        total_rows += len(out_chunk)
        print(f"[targets] chunk {i}: wrote {len(out_chunk):,} rows (cum={total_rows:,})  pos={pos_here:,}")

    writer.close()
    print(f"[done] wrote {total_rows:,} rows -> {args.out} | positives={total_pos:,}")

    if lead_hours and lead_totals:
        rows = []
        for h in sorted(lead_totals):
            tot = lead_totals[h]
            pos = lead_counts.get(h, 0)
            frac = (pos / tot) if tot else np.nan
            rows.append({"lead_h": h, "rows_in_window": tot, "pos_count": pos, "pos_frac": frac})
        summary = pd.DataFrame(rows)
        print(f"[diag] per-lead label summary (from {args.lead_col}, target={target_col}):")
        try:
            print(summary.to_string(index=False))
        except Exception:
            print(summary)

    if total_pos == 0 and target_mode == "lead-window":
        # Fallback: relax quantile until we get positives; read once, rewrite.
        print(f"[warn] {target_col} has zero positives; relaxing G threshold.")
        df_all = pd.concat(_iter_file(args.panel, args.chunksize), ignore_index=True)
        lead = _num(df_all[args.lead_col]) if args.lead_col in df_all else pd.Series(np.nan, index=df_all.index)
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
                df_all[target_col] = m_viable.astype("int8")
                horizon = float(args.horizon_max)
                lead_clip = lead.clip(lower=0.0, upper=horizon)
                df_all["lead_norm"] = 1.0 - (lead_clip / horizon)
                df_all["lead_inv"] = 1.0 / (1.0 + lead_clip)
                df_all["G_lead_norm"] = _num(df_all[args.g_col]) * df_all["lead_norm"]
                df_all["lead_h"] = lead if args.lead_col in df_all else float(lead_h_val)
                df_all["lead_h_bucket"] = _lead_bucket(lead, args.lead_bins_list)
                df_all.to_parquet(args.out, index=False) if _is_parquet(args.out) else df_all.to_csv(
                    args.out, index=False
                )
                print(f"[targets] fallback succeeded with q={q}; rewrote {args.out} positives={n_pos:,}")
                fallback_used = True
                break
        if not fallback_used:
            if funnel_mask.any():
                df_all[target_col] = funnel_mask.astype("int8")
                horizon = float(args.horizon_max)
                lead_clip = lead.clip(lower=0.0, upper=horizon)
                df_all["lead_norm"] = 1.0 - (lead_clip / horizon)
                df_all["lead_inv"] = 1.0 / (1.0 + lead_clip)
                df_all["G_lead_norm"] = _num(df_all[args.g_col]) * df_all["lead_norm"]
                df_all["lead_h"] = lead if args.lead_col in df_all else float(lead_h_val)
                df_all["lead_h_bucket"] = _lead_bucket(lead, args.lead_bins_list)
                df_all.to_parquet(args.out, index=False) if _is_parquet(args.out) else df_all.to_csv(
                    args.out, index=False
                )
                n_pos = int(funnel_mask.sum())
                print(f"[fallback] using funnel-only label; positives={n_pos:,}")
            else:
                print("[fatal] No rows within lead window; cannot build viability labels.")
    elif total_pos == 0:
        print(f"[warn] {target_col} has zero positives; consider lowering thresholds or inspecting inputs.")

    if args.save_threshold_json:
        meta = {
            "target_mode": target_mode,
            "target_col": target_col,
            "g_min": g_min,
            "g_min_quantile": args.g_min_quantile,
            "seed": args.seed,
            "lead_h": lead_h_val,
        }
        Path(args.save_threshold_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_threshold_json).write_text(json.dumps(meta, indent=2))
        print(f"[meta] saved threshold info -> {args.save_threshold_json}")

    if args.save_targets_meta:
        meta = {
            "target_mode": target_mode,
            "target_col": target_col,
            "time_col": args.time_col,
            "lead_hours": lead_hours,
            "definitions": {
                "y_knee_cross_H": "1 if knee crossing occurs in (t, t+H] (strictly future window).",
                "y_lock_stable_H": "1 if lock persists in (t, t+H] (strictly future window).",
                "y_commit_H": "y_knee_cross_H AND y_lock_stable_H (future window).",
            },
        }
        Path(args.save_targets_meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_targets_meta).write_text(json.dumps(meta, indent=2))
        print(f"[meta] saved targets metadata -> {args.save_targets_meta}")


if __name__ == "__main__":
    main()
