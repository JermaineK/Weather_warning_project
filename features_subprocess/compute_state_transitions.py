#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_state_transitions.py

Agent: derive per-cell state deltas/persistence from existing GKA/SFI style
features without changing their mathematics. Outputs lightweight geometry (G)
and excitability (E) proxies plus transition indicators and knee/parity tags.

Design goals
- Stream-friendly: iterates parquet via pyarrow batches or CSV chunks.
- Alias tolerant: prefers ilat/ilon for grouping; falls back to lat/lon.
- Fail-safe: if required inputs are missing, new columns are still written
  with zeros so downstream stages do not crash.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _read_columns(path: str | Path) -> List[str]:
    if _is_parquet(path):
        if pq is None:
            raise SystemExit("pyarrow is required to peek parquet columns.")
        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, nrows=2, low_memory=False).columns)


def _robust01(arr: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    """
    Quantile-based 0-1 scaling with clipping; resistant to outliers and safe on
    streaming chunks. Falls back to min/max if quantiles collapse.
    """
    out = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    vals = arr[mask]
    lo = np.nanquantile(vals, q_low) if vals.size else 0.0
    hi = np.nanquantile(vals, q_high) if vals.size else 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    out[mask] = np.clip((arr[mask] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _cell_keys(df: pd.DataFrame, lat_col: str, lon_col: str, ilat_col: str | None, ilon_col: str | None) -> np.ndarray:
    """
    Prefer integer grid indices if present; otherwise factorize lat/lon to
    stable integer codes per chunk.
    """
    if ilat_col in df.columns and ilon_col in df.columns:
        lat_idx = pd.to_numeric(df[ilat_col], errors="coerce").astype("Int64").to_numpy()
        lon_idx = pd.to_numeric(df[ilon_col], errors="coerce").astype("Int64").to_numpy()
    else:
        lat_idx, _ = pd.factorize(df[lat_col], sort=False)
        lon_idx, _ = pd.factorize(df[lon_col], sort=False)
    lat_idx = np.asarray(lat_idx, dtype=np.int64)
    lon_idx = np.asarray(lon_idx, dtype=np.int64)
    return (lat_idx << 32) ^ (lon_idx & 0xFFFFFFFF)


def _float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


@dataclass
class CellState:
    last_time: pd.Timestamp | None = None
    last_G: float = np.nan
    last_E: float = np.nan
    last_sign: float = np.nan
    last_knee_flag: bool = False
    knee_time: pd.Timestamp | None = None
    knee_sign: float = np.nan
    last_flip_time: pd.Timestamp | None = None
    g_window: Deque[Tuple[pd.Timestamp, float]] = field(default_factory=deque)
    e_window: Deque[Tuple[pd.Timestamp, float]] = field(default_factory=deque)
    g_pulses: Deque[pd.Timestamp] = field(default_factory=deque)
    e_pulses: Deque[pd.Timestamp] = field(default_factory=deque)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def _compute_G(df: pd.DataFrame, args) -> np.ndarray:
    dir_var = _float_series(df, "gka_dir_var")
    F = _float_series(df, "gka_F")
    knee = _float_series(df, "gka_knee_ratio")

    dir_term = 1.0 - dir_var.clip(lower=0.0)
    knee_term = _robust01(knee.to_numpy(), args.knee_q_low, args.knee_q_high)
    base = (
        np.power(dir_term.to_numpy(), args.G_w_dir)
        * np.power(F.clip(lower=0.0).to_numpy(), args.G_w_F)
        * np.power(knee_term, args.G_w_knee)
    )
    return _robust01(base, args.G_q_low, args.G_q_high)


def _compute_E(df: pd.DataFrame, args) -> np.ndarray:
    S3 = _float_series(df, "S3")
    S = _float_series(df, "S")
    msl_d1h = _float_series(df, "msl_d1h")
    shear_deep = _float_series(df, "shear_deep")

    s3_term = np.abs(S3.to_numpy()) if "S3" in df.columns else np.zeros(len(df), dtype=float)
    s_term = np.abs(S.to_numpy()) if "S" in df.columns else np.zeros(len(df), dtype=float)
    shear_term = np.abs(shear_deep.to_numpy()) if "shear_deep" in df.columns else np.zeros(len(df), dtype=float)
    msl_term = np.abs(msl_d1h.to_numpy())

    weighted = (
        args.E_w_S3 * s3_term
        + args.E_w_S * s_term
        + args.E_w_msl * msl_term
        + args.E_w_shear * shear_term
    )
    return _robust01(weighted, args.E_q_low, args.E_q_high)


def _sign_proxy(df: pd.DataFrame) -> np.ndarray:
    if "zeta_mean3h" in df.columns:
        return np.sign(pd.to_numeric(df["zeta_mean3h"], errors="coerce")).to_numpy()
    if "zeta" in df.columns:
        return np.sign(pd.to_numeric(df["zeta"], errors="coerce")).to_numpy()
    return np.zeros(len(df), dtype=float)


def _classify_transition(G: np.ndarray, dG: np.ndarray, sign_flip: np.ndarray, G_thr: float, decay_thr: float) -> np.ndarray:
    cls = np.zeros(len(G), dtype=np.int8)
    high = G >= G_thr
    cls[np.where(high & (dG < -abs(decay_thr)))] = 1  # high weakening
    cls[np.where(high & (dG >= -abs(decay_thr)))] = 2  # high strengthening/steady
    cls[np.where(high & (sign_flip > 0))] = 3          # sign flip while high
    return cls


def _iter_batches(path: str | Path, columns: Sequence[str], chunk_rows: int, parquet_rows: int) -> Iterable[pd.DataFrame]:
    """
    Yield DataFrames with only requested columns. Parquet uses pyarrow streaming
    when available; CSV uses pandas chunking.
    """
    if _is_parquet(path):
        if pq is None or pa is None:
            yield pd.read_parquet(path, columns=list(columns))
            return
        pf = pq.ParquetFile(path)
        batch_size = parquet_rows or chunk_rows or None
        for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
            yield batch.to_pandas()
    else:
        kwargs = {"usecols": list(columns), "low_memory": False}
        if "time" in columns:
            kwargs["parse_dates"] = ["time"]
        for chunk in pd.read_csv(path, chunksize=chunk_rows or None, **kwargs):
            yield chunk


def _write_any(path: str | Path, df: pd.DataFrame, first: bool, writer) -> tuple[bool, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(p):
        if pq is None or pa is None:
            if not first and p.exists():
                raise SystemExit("pyarrow required for streaming parquet writes; install pyarrow or use CSV output.")
            df.to_parquet(p, index=False)
            return False, writer
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(p, table.schema)
        writer.write_table(table)
        return False, writer
    comp = "gzip" if str(p).lower().endswith(".gz") else "infer"
    mode = "w" if first else "a"
    header = first
    df.to_csv(p, index=False, mode=mode, header=header, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
    return False, writer


# ---------------------------------------------------------------------------
# Streaming processor
# ---------------------------------------------------------------------------

def process_stream(path: str, args) -> None:
    all_cols = _read_columns(path)
    label_passthrough = [
        "storm",
        "storm_point",
        "storm_window",
        "near_storm",
        "pregen",
        "t_to_storm_min_h",
        "row_id",
    ]
    # Agent: preserve key GKA/SFI + thermo columns for downstream commitment/viability stages.
    feature_passthrough = [
        "gka_SAI",
        "gka_SII",
        "gka_parity_eta",
        "gka_chirality",
        "gka_vortdiv_ratio",
        "SFI",
        "SFI2",
        "thermo_shear",
        "pdrop_nd",
        "t2m_anom_local",
    ]
    cols_needed = {
        args.time_col,
        args.lat_col,
        args.lon_col,
        args.ilat_col,
        args.ilon_col,
        "gka_dir_var",
        "gka_F",
        "gka_knee_ratio",
        "S3",
        "S",
        "msl_d1h",
        "shear_deep",
        "zeta_mean3h",
        "zeta",
    }
    cols_needed.update(feature_passthrough)
    cols = [c for c in cols_needed if c and c in all_cols]
    cols += [c for c in label_passthrough if c in all_cols]
    missing_core = {"gka_dir_var", "gka_F", "gka_knee_ratio"} - set(cols)
    if missing_core:
        print(f"[state] Warning: missing core coherence columns {missing_core}; G will degrade gracefully.")
    state: Dict[int, CellState] = {}
    first_write = True
    writer = None
    total_rows = 0
    for chunk in _iter_batches(path, cols, args.chunk_rows, args.parquet_rows):
        if chunk.empty:
            continue
        chunk = chunk.copy()
        if args.time_col not in chunk.columns:
            raise SystemExit("Input requires a time column.")
        chunk[args.time_col] = pd.to_datetime(chunk[args.time_col])
        chunk["_cell_key"] = _cell_keys(chunk, args.lat_col, args.lon_col, args.ilat_col, args.ilon_col)
        chunk["_ord"] = np.arange(len(chunk))
        chunk.sort_values([args.time_col, "_cell_key"], inplace=True)

        G = _compute_G(chunk, args).astype("float32")
        E = _compute_E(chunk, args).astype("float32")
        S_sign = _sign_proxy(chunk).astype("float32")
        knee_raw = _float_series(chunk, "gka_knee_ratio").to_numpy(float)
        knee_score = _robust01(np.abs(knee_raw), args.knee_q_low, args.knee_q_high).astype("float32")

        chunk["G"] = G
        chunk["E"] = E
        chunk["S_sign"] = S_sign
        chunk["gka_knee_score"] = knee_score
        chunk["dG_1h"] = np.zeros(len(chunk), dtype=np.float32)
        chunk["dE_1h"] = np.zeros(len(chunk), dtype=np.float32)
        chunk["dSsign_flip"] = np.zeros(len(chunk), dtype=np.int8)
        chunk["gka_knee_cross"] = np.zeros(len(chunk), dtype=np.int8)
        chunk["gka_knee_state"] = np.zeros(len(chunk), dtype=np.int8)
        chunk["gka_knee_post_h"] = np.full(len(chunk), np.nan, dtype=np.float32)
        chunk["gka_parity_lock"] = np.zeros(len(chunk), dtype=np.int8)
        chunk["G_persist_24h"] = np.full(len(chunk), np.nan, dtype=np.float32)
        chunk["G_pulse_count_96h"] = np.zeros(len(chunk), dtype=np.int16)
        chunk["E_pulse_count_96h"] = np.zeros(len(chunk), dtype=np.int16)

        persist_td = pd.Timedelta(hours=args.persist_hours)
        pulse_td = pd.Timedelta(hours=args.pulse_hours)

        for key, sdf in chunk.groupby("_cell_key", sort=False):
            st = state.get(int(key), CellState())
            for idx, row in sdf.iterrows():
                t = row[args.time_col]
                gval = float(row["G"])
                eval_ = float(row["E"])
                sgn = float(row["S_sign"])
                knee_val = float(row["gka_knee_score"]) if "gka_knee_score" in row else 0.0
                knee_flag = bool(knee_val >= float(args.knee_thr))
                knee_cross = int(knee_flag and (not st.last_knee_flag))
                # knee_state: 0=pre, 1=crossing, 2=post
                knee_state = 1 if knee_cross else (2 if knee_flag else 0)

                if np.isfinite(st.last_G):
                    chunk.at[idx, "dG_1h"] = gval - st.last_G
                if np.isfinite(st.last_E):
                    chunk.at[idx, "dE_1h"] = eval_ - st.last_E
                if np.isfinite(st.last_sign):
                    chunk.at[idx, "dSsign_flip"] = int(sgn != st.last_sign)

                if knee_flag:
                    if knee_cross or st.knee_time is None:
                        st.knee_time = t
                        st.knee_sign = sgn
                else:
                    st.knee_time = None
                    st.knee_sign = np.nan

                if not args.lite:
                    cutoff_g = t - persist_td
                    while st.g_window and st.g_window[0][0] < cutoff_g:
                        st.g_window.popleft()
                    st.g_window.append((t, gval))
                    if st.g_window:
                        chunk.at[idx, "G_persist_24h"] = float(np.nanmean([v for _, v in st.g_window]))

                    cutoff_pulse = t - pulse_td
                    while st.g_pulses and st.g_pulses[0] < cutoff_pulse:
                        st.g_pulses.popleft()
                    if gval >= args.G_pulse_thr:
                        st.g_pulses.append(t)
                    chunk.at[idx, "G_pulse_count_96h"] = len(st.g_pulses)

                    while st.e_pulses and st.e_pulses[0] < cutoff_pulse:
                        st.e_pulses.popleft()
                    if eval_ >= args.E_pulse_thr:
                        st.e_pulses.append(t)
                    chunk.at[idx, "E_pulse_count_96h"] = len(st.e_pulses)

                if np.isfinite(st.last_sign) and np.isfinite(sgn) and (sgn != st.last_sign):
                    st.last_flip_time = t
                    if knee_flag:
                        st.knee_time = t
                        st.knee_sign = sgn

                knee_age_h = np.nan
                if st.knee_time is not None:
                    knee_age_h = float((t - st.knee_time) / np.timedelta64(1, "h"))

                lock_ok = False
                if knee_flag and st.knee_time is not None and np.isfinite(st.knee_sign):
                    if knee_age_h >= float(args.lock_hours):
                        if (st.last_flip_time is None) or (st.last_flip_time <= st.knee_time):
                            lock_ok = True

                chunk.at[idx, "gka_knee_cross"] = knee_cross
                chunk.at[idx, "gka_knee_state"] = knee_state
                chunk.at[idx, "gka_knee_post_h"] = knee_age_h
                chunk.at[idx, "gka_parity_lock"] = int(lock_ok)

                st.last_time = t
                st.last_G = gval
                st.last_E = eval_
                st.last_sign = sgn
                st.last_knee_flag = knee_flag
            state[int(key)] = st

        chunk["transition_class"] = _classify_transition(
            chunk["G"].to_numpy(),
            chunk["dG_1h"].to_numpy(),
            chunk["dSsign_flip"].to_numpy(),
            args.G_high_thr,
            args.dG_decay_thr,
        )

        # restore original order within chunk
        chunk.sort_values("_ord", inplace=True)
        out_cols = [
            args.time_col,
            args.lat_col,
            args.lon_col,
            "G",
            "E",
            "S_sign",
            "gka_knee_score",
            "gka_knee_cross",
            "gka_knee_state",
            "gka_knee_post_h",
            "gka_parity_lock",
            "dG_1h",
            "dE_1h",
            "dSsign_flip",
            "G_persist_24h",
            "G_pulse_count_96h",
            "E_pulse_count_96h",
            "transition_class",
        ]
        # keep any pre-existing columns alongside new ones
        pass_through = [c for c in chunk.columns if c not in out_cols + ["_cell_key", "_ord"]]
        chunk = chunk[[*pass_through, *out_cols]]
        first_write, writer = _write_any(args.outfile, chunk, first_write, writer)
        total_rows += len(chunk)
    if writer is not None:
        writer.close()
    print(f"[state] wrote {args.outfile} rows={total_rows:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute per-cell G/E state transitions and persistence features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--infile", required=True, help="Input grid table (CSV/Parquet).")
    ap.add_argument("--outfile", required=True, help="Output path with transitions added.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--ilat-col", default="ilat")
    ap.add_argument("--ilon-col", default="ilon")

    # G definition weights + scaling
    ap.add_argument("--G-w-dir", type=float, default=1.0, help="Weight on (1 - gka_dir_var).")
    ap.add_argument("--G-w-F", type=float, default=1.0, help="Weight on gka_F.")
    ap.add_argument("--G-w-knee", type=float, default=1.0, help="Weight on knee ratio term.")
    ap.add_argument("--G-q-low", type=float, default=0.05, help="Low quantile for robust01 on G.")
    ap.add_argument("--G-q-high", type=float, default=0.95, help="High quantile for robust01 on G.")
    ap.add_argument("--knee-q-low", type=float, default=0.05, help="Low quantile for robust01 on knee.")
    ap.add_argument("--knee-q-high", type=float, default=0.95, help="High quantile for robust01 on knee.")
    ap.add_argument("--knee-thr", type=float, default=0.7, help="Knee-score threshold for knee state tagging.")
    ap.add_argument("--lock-hours", type=float, default=12.0, help="Hours required for parity lock after knee.")

    # E definition weights + scaling
    ap.add_argument("--E-w-S3", type=float, default=1.0, help="Weight on |S3| term.")
    ap.add_argument("--E-w-S", type=float, default=0.5, help="Weight on |S| term.")
    ap.add_argument("--E-w-msl", type=float, default=0.5, help="Weight on |msl_d1h| term.")
    ap.add_argument("--E-w-shear", type=float, default=0.5, help="Weight on shear_deep term.")
    ap.add_argument("--E-q-low", type=float, default=0.05, help="Low quantile for robust01 on E.")
    ap.add_argument("--E-q-high", type=float, default=0.95, help="High quantile for robust01 on E.")

    # Persistence / pulses
    ap.add_argument("--persist-hours", type=float, default=24.0, help="Window for G persistence.")
    ap.add_argument("--pulse-hours", type=float, default=96.0, help="Window for pulse counts.")
    ap.add_argument("--G-pulse-thr", type=float, default=0.8, help="Threshold for G pulse counter.")
    ap.add_argument("--E-pulse-thr", type=float, default=0.8, help="Threshold for E pulse counter.")
    ap.add_argument("--G-high-thr", type=float, default=0.8, help="Threshold for transition_class high-G.")
    ap.add_argument("--dG-decay-thr", type=float, default=0.0, help="Allowance for deciding weakening vs steady.")

    ap.add_argument("--lite", action="store_true", help="Compute only dG/dE/sign-flip; skip persistence counts.")

    # Streaming hints
    ap.add_argument(
        "--chunk-rows",
        "--chunk_rows",
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunk size (rows).",
    )
    ap.add_argument(
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=200_000,
        help="Parquet batch size.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Ignored for compatibility; outfile is always replaced.")
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip work if outfile already exists.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if Path(args.infile).resolve() == Path(args.outfile).resolve():
        raise SystemExit("infile and outfile must differ.")
    out_path = Path(args.outfile)
    if out_path.exists():
        if args.skip_if_exists:
            print(f"[state] skip (exists): {out_path}")
            return
        if not args.overwrite:
            print(f"[state] skip (exists, use --overwrite or --skip-if-exists): {out_path}")
            return
    process_stream(args.infile, args)


if __name__ == "__main__":
    main()
