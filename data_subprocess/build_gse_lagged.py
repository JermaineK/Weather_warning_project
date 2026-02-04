#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_gse_lagged.py

Add temporal persistence features to a GSE panel (post-subset) to capture drift/persistence.

Outputs same-cell lag features and optional "nearby previous-hour" maxima so moving structure
still shows up even if it hopped a grid box.

Assumes the input already fits in memory (ID subset). If your subset is huge, downsample first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _read_any(path: str) -> pd.DataFrame:
    if _is_parquet(path):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(str(p)):
        df.to_parquet(p, index=False)
        return
    comp = "gzip" if p.name.lower().endswith(".gz") else "infer"
    df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


def _parse_lags(spec: str) -> List[int]:
    if not spec:
        return [1]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    lags = []
    for p in parts:
        try:
            lags.append(int(p))
        except Exception:
            continue
    lags = [l for l in lags if l > 0]
    return lags or [1]


def _parse_csv_list(spec: str | None) -> List[str]:
    if not spec:
        return []
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def _parse_windows(spec: str | None) -> List[int]:
    if not spec:
        return []
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    wins = []
    for p in parts:
        try:
            wins.append(int(p))
        except Exception:
            continue
    wins = [w for w in wins if w > 0]
    return wins


def _apply_past_roll(
    df: pd.DataFrame,
    cols: List[str],
    wins: List[int],
    kind: str,
) -> dict[str, np.ndarray]:
    """
    Apply past-only rolling features per (lat_r, lon_r).
    kind: mean | std | slope | fliprate
    """
    # Agent: enforce past-only windows to avoid target leakage.
    if not cols or not wins:
        return {}
    idx_map = df.groupby(["lat_r", "lon_r"], sort=False).indices
    new_cols: dict[str, np.ndarray] = {}

    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        for win in wins:
            out = np.zeros(len(df), dtype=np.float32)
            for _, idx in idx_map.items():
                ii = np.asarray(idx, dtype=np.int64)
                x = pd.Series(vals[ii])
                if kind == "mean":
                    res = x.shift(1).rolling(win, min_periods=1).mean()
                elif kind == "std":
                    res = x.shift(1).rolling(win, min_periods=1).std().fillna(0.0)
                elif kind == "slope":
                    if win <= 1:
                        res = pd.Series(0.0, index=x.index)
                    else:
                        xs = x.shift(1)
                        res = (xs - xs.shift(win - 1)) / float(win - 1)
                elif kind == "fliprate":
                    sgn = np.sign(x)
                    flips = (sgn * sgn.shift(1) < 0).astype(float)
                    res = flips.shift(1).rolling(win, min_periods=1).mean()
                else:
                    continue
                out[ii] = pd.to_numeric(res, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)

            suffix = {
                "mean": "mean_past",
                "std": "std_past",
                "slope": "slope_past",
                "fliprate": "fliprate_past",
            }[kind]
            new_cols[f"{col}_{suffix}{int(win)}h"] = out
    return new_cols


def _apply_past_corr(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    win: int,
    out_col: str,
) -> dict[str, np.ndarray]:
    if col_a not in df.columns or col_b not in df.columns or win <= 1:
        return {}
    idx_map = df.groupby(["lat_r", "lon_r"], sort=False).indices
    out = np.zeros(len(df), dtype=np.float32)
    a_vals = pd.to_numeric(df[col_a], errors="coerce").to_numpy(dtype=float)
    b_vals = pd.to_numeric(df[col_b], errors="coerce").to_numpy(dtype=float)
    for _, idx in idx_map.items():
        ii = np.asarray(idx, dtype=np.int64)
        a = pd.Series(a_vals[ii]).shift(1)
        b = pd.Series(b_vals[ii]).shift(1)
        corr = a.rolling(win, min_periods=2).corr(b)
        out[ii] = pd.to_numeric(corr, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    return {out_col: out}


def _max_within_radius(prev_block: pd.DataFrame, lat_now: np.ndarray, lon_now: np.ndarray, col: str, r: float) -> np.ndarray:
    """
    For each row in the current block, compute max of `col` in prev_block within |dlat|,|dlon| <= r.
    Uses a simple bounding-box filter (degrees). Assumes prev_block has columns ["lat_r","lon_r",col].
    """
    if prev_block.empty:
        return np.full_like(lat_now, np.nan, dtype=float)
    la_prev = prev_block["lat_r"].to_numpy()
    lo_prev = prev_block["lon_r"].to_numpy()
    vals = prev_block[col].to_numpy()
    out = np.full_like(lat_now, np.nan, dtype=float)

    for i, (la0, lo0) in enumerate(zip(lat_now, lon_now)):
        m = (np.abs(la_prev - la0) <= r) & (np.abs(lo_prev - lo0) <= r)
        if m.any():
            out[i] = np.nanmax(vals[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add temporal lag features (same-cell and optional nearby drift) to a GSE panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input GSE panel (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output with lagged features.")
    # Agent: accept overwrite flag for pipeline compatibility (output overwrites by default).
    ap.add_argument("--overwrite", action="store_true", help="No-op; output is overwritten if present.")
    ap.add_argument("--lags", default="1", help="Comma-separated hour lags (e.g., '1,2').")
    ap.add_argument(
        "--lag-cols",
        default="",
        help="Additional columns to lag (CSV, e.g., 'gka_parity_lock,gka_knee_state').",
    )
    ap.add_argument(
        "--past-windows",
        default="",
        help="Comma-separated past windows for rolling stats (hours).",
    )
    ap.add_argument(
        "--past-mean-cols",
        default="",
        help="CSV of columns for past rolling mean features.",
    )
    ap.add_argument(
        "--past-std-cols",
        default="",
        help="CSV of columns for past rolling std features.",
    )
    ap.add_argument(
        "--past-slope-cols",
        default="",
        help="CSV of columns for past slope features (endpoint trend).",
    )
    ap.add_argument(
        "--fliprate-cols",
        default="",
        help="CSV of sign-like columns for past flip-rate features.",
    )
    ap.add_argument("--radius-deg", type=float, default=0.0, help="Radius (deg) for drift-aware prev-hour max (0 disables).")
    ap.add_argument("--lat-col", default="lat", help="Latitude column.")
    ap.add_argument("--lon-col", default="lon", help="Longitude column.")
    ap.add_argument("--time-col", default="time", help="Timestamp column.")
    ap.add_argument("--round-dp", type=int, default=3, help="Decimal places to round lat/lon for same-cell grouping.")
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=None,
        help="Optional chunk size for streaming CSV or Parquet batches (0/None = load whole file).",
    )
    ap.add_argument(
        "--defragment",
        action="store_true",
        help="Optional final copy to defragment before write (uses extra memory).",
    )
    args = ap.parse_args()

    lags = _parse_lags(args.lags)
    lag_cols = _parse_csv_list(args.lag_cols)
    past_windows = _parse_windows(args.past_windows)
    past_mean_cols = _parse_csv_list(args.past_mean_cols)
    past_std_cols = _parse_csv_list(args.past_std_cols)
    past_slope_cols = _parse_csv_list(args.past_slope_cols)
    fliprate_cols = _parse_csv_list(args.fliprate_cols)
    chunk_rows = args.chunksize if args.chunksize and args.chunksize > 0 else None

    if _is_parquet(args.panel):
        df_iter = None
        if chunk_rows:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(args.panel)
            df_iter = (batch.to_pandas() for batch in pf.iter_batches(batch_size=chunk_rows))
            print("[warn] chunk_rows set, but lagged features still materialize full panel in memory.")
        if df_iter is None:
            df = _read_any(args.panel)
        else:
            # Note: still materializes the full frame for lagged features.
            df = pd.concat(list(df_iter), ignore_index=True, copy=False)
    else:
        if chunk_rows:
            df = pd.concat(
                list(pd.read_csv(args.panel, low_memory=False, chunksize=chunk_rows)),
                ignore_index=True,
                copy=False,
            )
            print("[warn] chunk_rows set, but lagged features still materialize full panel in memory.")
        else:
            df = _read_any(args.panel)

    if args.time_col not in df or args.lat_col not in df or args.lon_col not in df:
        missing = [c for c in (args.time_col, args.lat_col, args.lon_col) if c not in df]
        raise SystemExit(f"Missing required columns: {missing}")

    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_convert(None)
    df["lat_r"] = pd.to_numeric(df[args.lat_col], errors="coerce").round(args.round_dp)
    df["lon_r"] = pd.to_numeric(df[args.lon_col], errors="coerce").round(args.round_dp)
    df = df.sort_values(["lat_r", "lon_r", args.time_col])

    # same-cell lags
    grp = df.groupby(["lat_r", "lon_r"], sort=False)
    for lag in lags:
        for src, tgt in (("G_struct", f"G_prev{lag}h_same"),
                         ("S_shear", f"S_prev{lag}h_same"),
                         ("E_energy", f"E_prev{lag}h_same")):
            if src in df:
                df[tgt] = grp[src].shift(lag)

    # extra same-cell lags for requested columns
    if lag_cols:
        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag{lag}h"] = grp[col].shift(lag)

    # drift-aware lags (radius > 0)
    r = float(args.radius_deg)
    if r > 0:
        df["time_floor"] = df[args.time_col].dt.floor("h")
        time_groups = {t: block[["lat_r", "lon_r", "G_struct", "S_shear", "E_energy"]] for t, block in df.groupby("time_floor", sort=True)}

        for lag in lags:
            col_suffix = f"prev{lag}h_near"
            for src, tgt in (("G_struct", f"G_{col_suffix}"),
                             ("S_shear", f"S_{col_suffix}"),
                             ("E_energy", f"E_{col_suffix}")):
                df[tgt] = np.nan
            for t, block_now in df.groupby("time_floor", sort=True):
                prev_t = t - pd.Timedelta(hours=lag)
                prev_block = time_groups.get(prev_t)
                if prev_block is None or prev_block.empty:
                    continue
                la_now = block_now["lat_r"].to_numpy()
                lo_now = block_now["lon_r"].to_numpy()
                for src, tgt in (("G_struct", f"G_{col_suffix}"),
                                 ("S_shear", f"S_{col_suffix}"),
                                 ("E_energy", f"E_{col_suffix}")):
                    if src in prev_block:
                        vals = _max_within_radius(prev_block[["lat_r", "lon_r", src]], la_now, lo_now, src, r)
                        df.loc[block_now.index, tgt] = vals

        df.drop(columns=["time_floor"], inplace=True)

    # Past-only rolling features (computed on sorted, grouped data)
    extra_cols = []
    if past_windows:
        extra_cols.append(_apply_past_roll(df, past_mean_cols, past_windows, "mean"))
        extra_cols.append(_apply_past_roll(df, past_std_cols, past_windows, "std"))
        extra_cols.append(_apply_past_roll(df, past_slope_cols, past_windows, "slope"))
        extra_cols.append(_apply_past_roll(df, fliprate_cols, past_windows, "fliprate"))
    if extra_cols:
        merged = {}
        for block in extra_cols:
            merged.update(block)
        if merged:
            # Assign in-place to avoid large temporary concat allocations.
            for col, arr in merged.items():
                df[col] = arr

    # Mud churn + slopes (aliases for readability)
    if "S_shear_std_past6h" in df.columns and "S_churn_6h" not in df.columns:
        df["S_churn_6h"] = df["S_shear_std_past6h"].astype("float32")
    if "G_struct_slope_past6h" in df.columns and "G_slope_6h" not in df.columns:
        df["G_slope_6h"] = df["G_struct_slope_past6h"].astype("float32")
    if "S_shear_slope_past6h" in df.columns and "S_slope_6h" not in df.columns:
        df["S_slope_6h"] = df["S_shear_slope_past6h"].astype("float32")

    # Rolling corr(G, S) over past window (causal, shifted)
    corr_cols = _apply_past_corr(df, "G_struct", "S_shear", 24, "corr_G_S_past24h")
    if corr_cols:
        for col, arr in corr_cols.items():
            df[col] = arr

    # Regime tags + spikes
    if "S_shear" in df.columns:
        s_vals = pd.to_numeric(df["S_shear"], errors="coerce")
        s_q50 = float(np.nanquantile(s_vals, 0.50)) if s_vals.notna().any() else float("nan")
        s_q90 = float(np.nanquantile(s_vals, 0.90)) if s_vals.notna().any() else float("nan")
        if "G_struct" in df.columns:
            g_vals = pd.to_numeric(df["G_struct"], errors="coerce")
            g_q90 = float(np.nanquantile(g_vals, 0.90)) if g_vals.notna().any() else float("nan")
        else:
            g_q90 = float("nan")

        if "A_agree" in df.columns:
            a_vals = pd.to_numeric(df["A_agree"], errors="coerce")
        elif "gka_dir_var" in df.columns:
            a_vals = 1.0 - pd.to_numeric(df["gka_dir_var"], errors="coerce").clip(lower=0.0, upper=1.0)
        else:
            a_vals = pd.Series(np.nan, index=df.index)
        a_q90 = float(np.nanquantile(a_vals, 0.90)) if a_vals.notna().any() else float("nan")

        grp = df.groupby(["lat_r", "lon_r"], sort=False)
        ds = grp["S_shear"].diff(1)
        ds_q90 = float(np.nanquantile(ds, 0.90)) if ds.notna().any() else float("nan")

        df["mud_high"] = (s_vals > s_q90).astype("float32")
        df["mud_low"] = (s_vals < s_q50).astype("float32")
        if np.isfinite(g_q90):
            df["geom_high"] = (pd.to_numeric(df["G_struct"], errors="coerce") > g_q90).astype("float32")
        else:
            df["geom_high"] = 0.0
        if np.isfinite(a_q90):
            df["coh_high"] = (a_vals > a_q90).astype("float32")
        else:
            df["coh_high"] = 0.0
        df["mud_high_geom_high"] = (df["mud_high"] * df["geom_high"]).astype("float32")
        df["mud_high_geom_low"] = (df["mud_high"] * (1.0 - df["geom_high"]).astype("float32")).astype("float32")
        df["S_spike"] = ((s_vals > s_q90) | (ds > ds_q90)).astype("float32")

    # cleanup
    df.drop(columns=["lat_r", "lon_r"], inplace=True)

    # Defragment before write only if requested (saves memory).
    if args.defragment:
        df = df.copy()
    _write_any(args.out, df)
    print(f"[done] wrote {len(df):,} rows with lags -> {args.out}")


if __name__ == "__main__":
    main()
