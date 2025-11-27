#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_patch.py
Lightweight, pandas-only feature augmentation on the flattened grid table.

Adds/does:
  - msl_d1h, msl_d3h                        (pressure tendencies per (lat,lon))
  - zeta_mean3h, zeta_std3h                 (3h rolling within (lat,lon))
  - div_mean3h,  div_std3h
  - S_mean3h,    S_std3h
  - shear10_def                             (2-D deformation shear from u10/v10 on native grid; km-aware)
  - shear_proxy                             (|zeta|-|div|; only used if no real/bulk/deformation shear)
  - S3                                       3-hour rolling mean of chosen shear source (best available)
  - dS_dt, drelax_dt, dagree_dt              (temporal derivatives per (lat,lon), gap-robust; memory-lean)
  - msl_grad                                 (spatial gradient magnitude using ilat/ilon if present)
  - optional light column ops via --ops JSON  (rename/drop/select/add-const)

Shear selection order for S3 (first present wins unless --prefer-shear provided):
  1) 0–6 km shear:  shear_06km | bulk_shear_0_6km | shear06 | shear_06
  2) 0–1 km shear:  shear_01km | bulk_shear_0_1km | shear01 | shear_01
  3) 2-D deformation from u10/v10: shear10_def
  4) proxy:  shear_proxy = |zeta| - |div|

Memory notes
------------
This version avoids DataFrame-wide groupby/apply materialization. All per-(lat,lon)
ops write into preallocated arrays by iterating group index arrays (cheap), so peak
RAM stays close to your input frame + a handful of float32 buffers.
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

REQ_BASE = ["time", "lat", "lon"]

# Candidate shear columns (best → fallback)
CAND_SHEAR_06 = ["shear_06km", "bulk_shear_0_6km", "shear06", "shear_06"]
CAND_SHEAR_01 = ["shear_01km", "bulk_shear_0_1km", "shear01", "shear_01"]
CAND_SHEAR_DEF = ["shear10_def", "shear_2d10", "shear10"]  # deformation shear from u10/v10 (this script)

# u/v alias binding (column names in flattened features)
U_ALIASES = ["u10", "u", "U10M"]
V_ALIASES = ["v10", "v", "V10M"]


# ----------------------------- small utils -----------------------------

def _ensure_datetime(df: pd.DataFrame) -> None:
    if "time" not in df.columns:
        raise ValueError("augment: 'time' column is required")
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = (
            pd.to_datetime(df["time"], utc=True, errors="coerce")
              .dt.tz_localize(None)
        )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    # Sorted so each (lat,lon) group is a contiguous block in time order
    return df.sort_values(["lat", "lon", "time"], kind="mergesort").reset_index(drop=True)


def _bind_uv_columns(cols: Iterable[str]) -> Tuple[str | None, str | None]:
    cols = set(cols)
    ucol = next((c for c in U_ALIASES if c in cols), None)
    vcol = next((c for c in V_ALIASES if c in cols), None)
    return ucol, vcol


def _deg2km_lat(dlat_deg: float) -> float:
    return 111.2 * dlat_deg


def _deg2km_lon(dlon_deg: float, lat_deg_array: np.ndarray) -> np.ndarray:
    # per-row scaling by cos(lat)
    return 111.2 * np.cos(np.deg2rad(lat_deg_array)) * dlon_deg


# ------------------------ per-group memory-lean ops ------------------------

def _time_derivative_lean(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute d(col)/dt per (lat,lon) with O(N) extra memory (one float array)."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float32")

    out = np.zeros(len(df), dtype=np.float32)

    # Faster than groupby.apply: iterate indices
    idx_map = df.groupby(["lat", "lon"], sort=False).indices
    t_ns = pd.to_datetime(df["time"], utc=True, errors="coerce").view("int64").to_numpy()

    x_all = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    for _, idx in idx_map.items():
        if len(idx) <= 1:
            continue
        ii = np.asarray(idx, dtype=np.int64)
        # dt in hours
        dt_h = np.maximum((t_ns[ii][1:] - t_ns[ii][:-1]) / 3_600_000_000_000.0, 1e-9)
        dx = np.empty(ii.size, dtype=np.float32)
        dx[0] = 0.0
        np.divide((x_all[ii][1:] - x_all[ii][:-1]), dt_h, out=dx[1:], where=np.isfinite(dt_h))
        out[ii] = dx

    return pd.Series(out, index=df.index, dtype="float32")


def _rolling_mean_std_lean(df: pd.DataFrame, col: str, win: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Per-(lat,lon) rolling mean/std over a tiny window; writes into preallocated arrays.
    Uses pandas rolling on each small 1D slice to reuse optimized kernels without
    building a gigantic MultiIndex object.
    """
    if col not in df.columns:
        z = pd.Series(0.0, index=df.index, dtype="float32")
        return z, z

    mean_out = np.zeros(len(df), dtype=np.float32)
    std_out  = np.zeros(len(df), dtype=np.float32)

    idx_map = df.groupby(["lat", "lon"], sort=False).indices
    x_all = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

    for _, idx in idx_map.items():
        ii = np.asarray(idx, dtype=np.int64)
        x = pd.Series(x_all[ii])
        # small series → rolling is cheap; no giant object created
        m = x.rolling(win, min_periods=1).mean().astype("float32")
        s = x.rolling(win, min_periods=1).std().fillna(0.0).astype("float32")
        mean_out[ii] = m.to_numpy(dtype=np.float32, copy=False)
        std_out[ii]  = s.to_numpy(dtype=np.float32, copy=False)

    return (
        pd.Series(mean_out, index=df.index, dtype="float32"),
        pd.Series(std_out,  index=df.index, dtype="float32"),
    )


def _pressure_tendency_lean(df: pd.DataFrame):
    if "msl" not in df.columns:
        z = pd.Series(0.0, index=df.index, dtype="float32")
        return z, z

    out1 = np.zeros(len(df), dtype=np.float32)
    out3 = np.zeros(len(df), dtype=np.float32)

    idx_map = df.groupby(["lat", "lon"], sort=False).indices
    m_all = pd.to_numeric(df["msl"], errors="coerce").to_numpy(dtype=np.float32)

    for _, idx in idx_map.items():
        ii = np.asarray(idx, dtype=np.int64)
        x = m_all[ii]
        d1 = np.zeros_like(x, dtype=np.float32)
        d3 = np.zeros_like(x, dtype=np.float32)
        if x.size > 1:
            d1[1:] = x[1:] - x[:-1]
        if x.size > 3:
            d3[3:] = x[3:] - x[:-3]
        out1[ii] = d1
        out3[ii] = d3

    return (
        pd.Series(out1, index=df.index, dtype="float32"),
        pd.Series(out3, index=df.index, dtype="float32"),
    )


def _msl_grad_from_grid(df: pd.DataFrame) -> pd.Series:
    """
    Approximate |∇msl| using integer grid IDs (ilat/ilon) at each time slice.
    Requires columns: msl, ilat, ilon.
    """
    need = {"msl", "ilat", "ilon"}
    if not need.issubset(df.columns):
        return pd.Series(0.0, index=df.index, dtype="float32")

    ilat = pd.to_numeric(df["ilat"], errors="coerce").astype("Int32")
    ilon = pd.to_numeric(df["ilon"], errors="coerce").astype("Int32")
    out = pd.Series(0.0, index=df.index, dtype="float32")

    # Per-time blocks keep arrays small
    for _, idx in df.groupby("time", sort=False).indices.items():
        block = df.loc[idx, ["msl"]].copy()
        block["ilat"] = ilat.loc[idx].to_numpy()
        block["ilon"] = ilon.loc[idx].to_numpy()

        key = block["ilat"].astype(str) + "_" + block["ilon"].astype(str)
        msl_vals = pd.to_numeric(block["msl"], errors="coerce").to_numpy(dtype=float)
        msl_map = dict(zip(key, msl_vals))

        il = block["ilat"].to_numpy()
        jl = block["ilon"].to_numpy()
        m0 = msl_vals.astype(np.float32)

        def _neighbor(i, j):
            return msl_map.get(f"{i}_{j}", np.nan)

        m_up    = np.array([_neighbor(i-1, j) for i, j in zip(il, jl)], dtype=np.float32)
        m_down  = np.array([_neighbor(i+1, j) for i, j in zip(il, jl)], dtype=np.float32)
        m_left  = np.array([_neighbor(i, j-1) for i, j in zip(il, jl)], dtype=np.float32)
        m_right = np.array([_neighbor(i, j+1) for i, j in zip(il, jl)], dtype=np.float32)

        dmdy = np.where(
            np.isfinite(m_up) & np.isfinite(m_down), 0.5*(m_down - m_up),
            np.where(
                np.isfinite(m_down), m_down - m0,
                np.where(np.isfinite(m_up), m0 - m_up, 0.0),
            ),
        ).astype(np.float32)

        dmdx = np.where(
            np.isfinite(m_left) & np.isfinite(m_right), 0.5*(m_right - m_left),
            np.where(
                np.isfinite(m_right), m_right - m0,
                np.where(np.isfinite(m_left), m0 - m_left, 0.0),
            ),
        ).astype(np.float32)

        out.loc[idx] = np.hypot(dmdx, dmdy).astype(np.float32)

    return out


def _shear_proxy(df: pd.DataFrame) -> pd.Series:
    if ("zeta" not in df.columns) or ("div" not in df.columns):
        return pd.Series(0.0, index=df.index, dtype="float32")
    z = np.abs(pd.to_numeric(df["zeta"], errors="coerce").to_numpy(dtype=np.float32))
    d = np.abs(pd.to_numeric(df["div"],  errors="coerce").to_numpy(dtype=np.float32))
    return pd.Series((z - d).astype(np.float32), index=df.index, dtype="float32")


# -------------------- deformation shear (per-time, lean) --------------------

def _first_valid_delta_2d(arr: np.ndarray) -> float:
    """Estimate a representative grid spacing in degrees from a 2D field (lat or lon)."""
    if arr.shape[0] > 1:
        d = np.nanmean(arr[1:, :] - arr[:-1, :])
        if np.isfinite(d) and d != 0:
            return float(d)
    if arr.shape[1] > 1:
        d = np.nanmean(arr[:, 1:] - arr[:, :-1])
        if np.isfinite(d) and d != 0:
            return float(d)
    return np.nan


def _deformation_shear_from_uv(df: pd.DataFrame) -> pd.Series:
    """
    Compute 2-D deformation shear from u10/v10 using finite differences on the integer grid.
    Requires: time, lat, lon, ilat, ilon, and u/v (aliases supported).
    Returns a Series aligned to df.index (float32). Missing prerequisites → zeros.
    """
    need_cols = {"time", "lat", "lon", "ilat", "ilon"}
    if not need_cols.issubset(df.columns):
        return pd.Series(0.0, index=df.index, dtype="float32")

    ucol, vcol = _bind_uv_columns(df.columns)
    if not ucol or not vcol:
        return pd.Series(0.0, index=df.index, dtype="float32")

    ilat = pd.to_numeric(df["ilat"], errors="coerce").astype("Int32")
    ilon = pd.to_numeric(df["ilon"], errors="coerce").astype("Int32")

    out = pd.Series(0.0, index=df.index, dtype="float32")

    for _, idx in df.groupby("time", sort=False).indices.items():
        ii = ilat.loc[idx].to_numpy(dtype=np.int32, copy=False)
        jj = ilon.loc[idx].to_numpy(dtype=np.int32, copy=False)
        if len(ii) == 0:
            continue

        i_min, i_max = int(np.nanmin(ii)), int(np.nanmax(ii))
        j_min, j_max = int(np.nanmin(jj)), int(np.nanmax(jj))
        H = i_max - i_min + 1
        W = j_max - j_min + 1
        if H <= 1 or W <= 1:
            continue

        rel_i = ii - i_min
        rel_j = jj - j_min

        def _grid_from(colname):
            arr = np.full((H, W), np.nan, dtype=np.float32)
            arr[rel_i, rel_j] = pd.to_numeric(
                df.loc[idx, colname],
                errors="coerce",
            ).to_numpy(dtype=np.float32, copy=False)
            return arr

        U = _grid_from(ucol)
        V = _grid_from(vcol)
        LAT2 = _grid_from("lat")
        LON2 = _grid_from("lon")

        dlat_deg = _first_valid_delta_2d(LAT2)
        dlon_deg = _first_valid_delta_2d(LON2)
        if (not np.isfinite(dlat_deg)) or dlat_deg == 0 or (not np.isfinite(dlon_deg)) or dlon_deg == 0:
            continue

        dy_km = _deg2km_lat(abs(dlat_deg))
        dx_km_row = _deg2km_lon(abs(dlon_deg), LAT2.astype(float))

        with np.errstate(invalid="ignore"):
            row_med = np.nanmedian(dx_km_row, axis=1)

        if not np.all(np.isfinite(row_med)) or np.any(row_med <= 0):
            med = np.nanmedian(row_med)
            if not np.isfinite(med) or med <= 0:
                med = 111.2 * abs(dlon_deg)
            row_med = np.where(~np.isfinite(row_med) | (row_med <= 0), med, row_med)

        dx_km = np.repeat(row_med.reshape(H, 1), W, axis=1)

        def d_dx(A: np.ndarray) -> np.ndarray:
            left  = np.roll(A,  1, axis=1)
            right = np.roll(A, -1, axis=1)
            num = (right - left)
            den = (2.0 * dx_km)
            outA = num / den
            # one-sided fallbacks
            bad = ~np.isfinite(left) | ~np.isfinite(right) | ~np.isfinite(outA)
            one_left  = (A - left)  / dx_km
            one_right = (right - A) / dx_km
            outA = np.where(bad & np.isfinite(one_left),  one_left,  outA)
            outA = np.where(bad & ~np.isfinite(one_left) & np.isfinite(one_right), one_right, outA)
            return outA

        def d_dy(A: np.ndarray) -> np.ndarray:
            up   = np.vstack((np.full((1, A.shape[1]), np.nan, dtype=A.dtype), A[:-1, :]))
            down = np.vstack((A[1:, :], np.full((1, A.shape[1]), np.nan, dtype=A.dtype)))
            num = (down - up)
            den = (2.0 * dy_km)
            outA = num / den
            # one-sided fallbacks
            one_up   = (A - up)   / dy_km
            one_down = (down - A) / dy_km
            bad = ~np.isfinite(up) | ~np.isfinite(down) | ~np.isfinite(outA)
            outA = np.where(bad & np.isfinite(one_up),  one_up,  outA)
            outA = np.where(bad & ~np.isfinite(one_up) & np.isfinite(one_down), one_down, outA)
            return outA

        du_dx = d_dx(U)
        dv_dx = d_dx(V)
        du_dy = d_dy(U)
        dv_dy = d_dy(V)

        s1 = du_dx - dv_dy
        s2 = dv_dx + du_dy
        shear = 0.5 * np.sqrt(s1*s1 + s2*s2)

        flat = shear.reshape(-1)
        lin_idx = (rel_i * W + rel_j).astype(int)
        vals = np.full(len(idx), np.nan, dtype=np.float32)
        ok = (lin_idx >= 0) & (lin_idx < flat.size)
        vals[ok] = flat[lin_idx[ok]]

        finite_uv = np.isfinite(U[rel_i, rel_j]) & np.isfinite(V[rel_i, rel_j])
        vals[~finite_uv] = np.nan
        out.loc[idx] = np.nan_to_num(vals, nan=0.0).astype(np.float32)

        # keep memory tight
        del U, V, LAT2, LON2, shear, flat, vals
        gc.collect()

    return out


def _choose_shear_column(df: pd.DataFrame, prefer: str | None = None) -> tuple[str, str]:
    if prefer and prefer in df.columns:
        return prefer, f"prefer:{prefer}"
    for c in CAND_SHEAR_06:
        if c in df.columns:
            return c, "shear06"
    for c in CAND_SHEAR_01:
        if c in df.columns:
            return c, "shear01"
    for c in CAND_SHEAR_DEF:
        if c in df.columns:
            return c, "shear10_def"
    return "__proxy__", "proxy"


# ----------------------------- light ops --------------------------------

def _apply_ops(df: pd.DataFrame, ops_json: str | None) -> pd.DataFrame:
    if not ops_json:
        return df
    try:
        ops = json.loads(ops_json)
    except Exception:
        return df

    if isinstance(ops, dict):
        if "rename" in ops and isinstance(ops["rename"], dict):
            df = df.rename(columns=ops["rename"])
        if "drop" in ops and isinstance(ops["drop"], (list, tuple)):
            cols = [c for c in ops["drop"] if c in df.columns]
            if cols:
                df = df.drop(columns=cols)
        if "add_const" in ops and isinstance(ops["add_const"], dict):
            for k, v in ops["add_const"].items():
                df[k] = v
        if "select" in ops and isinstance(ops["select"], (list, tuple)) and len(ops["select"]) > 0:
            keep = [c for c in ops["select"] if c in df.columns]
            base = [c for c in ["time", "lat", "lon"] if c not in keep]
            df = df[base + keep]
    return df


# ------------------------- main augmentation -----------------------------

def add_all_feature_enhancements(
    df_in: pd.DataFrame,
    prefer_shear: str | None = None,
    s3_window: int = 3,
    lite_derivatives: bool = False,
) -> pd.DataFrame:
    missing = [c for c in REQ_BASE if c not in df_in.columns]
    if missing:
        raise ValueError(f"augment: required columns missing: {missing}")

    df = df_in.copy()
    _ensure_datetime(df)
    df = _sorted(df)

    # --- deformation shear from u10/v10 if possible (per-time, lean) ---
    if "shear10_def" not in df.columns:
        try:
            df["shear10_def"] = _deformation_shear_from_uv(df).astype("float32")
        except Exception:
            # fail-closed: just skip if anything goes sideways
            df["shear10_def"] = np.float32(0.0)

    # --- Pressure tendencies (per-(lat,lon), lean) ---
    d1h, d3h = _pressure_tendency_lean(df)
    df["msl_d1h"] = d1h
    df["msl_d3h"] = d3h
    del d1h, d3h
    gc.collect()

    # --- Rolling stats for zeta/div/S (per-(lat,lon), lean) ---
    for col, pref in [("zeta", "zeta"), ("div", "div"), ("S", "S")]:
        m, s = _rolling_mean_std_lean(df, col, win=s3_window)
        df[f"{pref}_mean3h"] = m
        df[f"{pref}_std3h"]  = s
        del m, s
        gc.collect()

    # --- Shear proxy fallback ---
    if ("zeta" in df.columns) or ("div" in df.columns):
        df["shear_proxy"] = _shear_proxy(df)

    # --- Choose shear and compute S3 (rolling mean over chosen shear) ---
    shear_col, src = _choose_shear_column(df, prefer=prefer_shear)
    if shear_col == "__proxy__":
        sbase = df["shear_proxy"] if "shear_proxy" in df.columns else pd.Series(
            0.0,
            index=df.index,
            dtype="float32",
        )
        sname = "shear_proxy"
    else:
        sbase = pd.to_numeric(df[shear_col], errors="coerce").astype("float32")
        sname = shear_col

    # Lean rolling mean over groups
    S3_out = np.zeros(len(df), dtype=np.float32)
    idx_map = df.groupby(["lat", "lon"], sort=False).indices
    for _, idx in idx_map.items():
        ii = np.asarray(idx, dtype=np.int64)
        x = pd.Series(sbase.iloc[ii].to_numpy(dtype=np.float32, copy=False))
        S3_out[ii] = (
            x.rolling(s3_window, min_periods=1)
             .mean()
             .fillna(0.0)
             .to_numpy(dtype=np.float32, copy=False)
        )

    df["S3"] = pd.Series(S3_out, index=df.index, dtype="float32")
    df["S3_src"] = src
    del S3_out, sbase
    gc.collect()

    # --- Temporal derivatives (lean path) ---
    if not lite_derivatives:
        if "S" in df.columns:
            df["dS_dt"] = _time_derivative_lean(df, "S")
        else:
            df["dS_dt"] = np.float32(0.0)
        if "relax" in df.columns:
            df["drelax_dt"] = _time_derivative_lean(df, "relax")
        else:
            df["drelax_dt"] = np.float32(0.0)
        if "agree" in df.columns:
            df["dagree_dt"] = _time_derivative_lean(df, "agree")
        else:
            df["dagree_dt"] = np.float32(0.0)
    else:
        df["dS_dt"] = np.float32(0.0)
        df["drelax_dt"] = np.float32(0.0)
        df["dagree_dt"] = np.float32(0.0)

    # --- Spatial pressure gradient magnitude ---
    df["msl_grad"] = _msl_grad_from_grid(df).astype("float32")

    # Compact a few known heavy columns (idempotent). Coords (time/lat/lon/ilat/ilon) are left alone.
    for c in [
        "msl_d1h", "msl_d3h",
        "zeta_mean3h", "zeta_std3h",
        "div_mean3h",  "div_std3h",
        "S_mean3h",    "S_std3h",
        "shear10_def", "shear_proxy", "S3",
        "dS_dt", "drelax_dt", "dagree_dt",
        "msl_grad",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df


# --------------------------------- CLI ----------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Feature patcher: tendencies/rolling/deformation shear + S3 and simple ops.",
    )
    ap.add_argument("--in",  dest="infile",  required=True, help="Input CSV(.gz)/Parquet")
    ap.add_argument("--out", dest="outfile", required=True, help="Output CSV(.gz)/Parquet")
    ap.add_argument(
        "--prefer-shear",
        default=None,
        help="Column to prefer for S3 if present (e.g., shear_06km or shear10_def)",
    )
    ap.add_argument("--s3-window", type=int, default=3, help="Rolling hours for S3")
    ap.add_argument(
        "--ops",
        default=None,
        help='Optional JSON with light ops: '
             '{"rename":{...},"drop":[...],"select":[...],"add_const":{...}}',
    )
    ap.add_argument(
        "--lite-derivatives",
        action="store_true",
        help="Skip temporal derivatives to conserve memory.",
    )
    # Compatibility no-ops to avoid YAML/manager breaks (accepted, ignored)
    ap.add_argument("--neighbor-step", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--radius-cells", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    low = p.suffix.lower()
    if low in [".parquet", ".parq", ".pq"]:
        return pd.read_parquet(p)
    # time parsed as datetime; _ensure_datetime will enforce tz-naive UTC
    return pd.read_csv(p, low_memory=False, parse_dates=["time"])


def _write_table(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    low = p.suffix.lower()
    if low in [".parquet", ".parq", ".pq"]:
        df.to_parquet(p, index=False)
        return
    comp = "gzip" if str(p).lower().endswith(".gz") else "infer"
    df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    df = _read_table(args.infile)

    df = add_all_feature_enhancements(
        df,
        prefer_shear=args.prefer_shear,
        s3_window=int(args.s3_window),
        lite_derivatives=bool(args.lite_derivatives),
    )

    df = _apply_ops(df, args.ops)

    # Column ordering: keep base coords up front, then key diagnostics, then everything else
    base = ["time", "lat", "lon"]
    front = [
        c for c in [
            "msl", "msl_d1h", "msl_d3h",
            "zeta", "div", "S",
            "zeta_mean3h", "zeta_std3h",
            "div_mean3h",  "div_std3h",
            "S_mean3h",    "S_std3h",
            "shear10_def", "shear_proxy", "S3", "S3_src",
            "dS_dt", "drelax_dt", "dagree_dt",
            "msl_grad", "ilat", "ilon",
        ]
        if c in df.columns
    ]
    others = [c for c in df.columns if c not in base + front]
    df = df[base + front + others]

    _write_table(df, args.outfile)
    src = (df["S3_src"].iloc[0] if "S3_src" in df.columns and len(df) > 0 else "n/a")
    print(f"[patch] wrote {args.outfile} rows={len(df):,}  S3_src={src}")


if __name__ == "__main__":
    main()