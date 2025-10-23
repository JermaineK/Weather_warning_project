#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
denoise_alerts.py  (safe-v2)
Denoise alert maps with optional morphology, neighbor filtering, connected-component
pruning, and temporal persistence — without dropping important persistent signals.

Required input columns: time (or detected), lat, lon, <flag-col>

Key safety features:
  • Persistence can be applied BEFORE or AFTER spatial ops, or used alone.
  • Persistence uses a hits-within-window rule (>= M hits in any H-hour window).
  • Spatial steps can be disabled: --no-morphology, --skip-neighbor-filter.
  • File-agnostic I/O: CSV/CSV.GZ/Parquet.

Outputs:
  time, lat, lon, alert_final
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_opening, binary_closing, generate_binary_structure, convolve,
    label as cc_label
)

CANDIDATE_TIME_COLS = ["time", "time_h", "datetime", "valid_time", "forecast_time"]

# --------------------- file I/O helpers ---------------------

def read_any(path, **kw):
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        # ignore CSV-specific kwargs like 'compression'
        return pd.read_parquet(path)
    return pd.read_csv(path, **kw)

def write_any(path, df: pd.DataFrame):
    p = str(path).lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if p.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)

# --------------------- helpers ---------------------

def _pick_time_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for c in CANDIDATE_TIME_COLS:
        if c in df.columns:
            print(f"[DENOISE] Using detected time column: {c}", flush=True)
            return c
    raise ValueError(
        f"No time-like column found. Tried { [preferred]+CANDIDATE_TIME_COLS } "
        f"but columns are {list(df.columns)}"
    )

def _try_parse_time_raw(s: pd.Series, fmt: str | None) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)

    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)

    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass

    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (mid and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)

    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

def _neighbor_filter(sl: np.ndarray, min_neighbors: int, connectivity: int) -> np.ndarray:
    """Keep a cell only if it has >= min_neighbors active neighbors (exclude self)."""
    if min_neighbors <= 0:
        return sl.astype(bool)
    if connectivity == 8:
        kernel = np.ones((3, 3), dtype=int)
        kernel[1,1] = 0
    else:  # 4-connectivity
        kernel = np.array([[0,1,0],
                           [1,0,1],
                           [0,1,0]], dtype=int)
    neigh = convolve(sl.astype(int), kernel, mode="constant", cval=0)
    keep = neigh >= min_neighbors
    return (sl.astype(bool) & keep)

def _build_cube_sparse(df: pd.DataFrame, time_col: str, flag_col: str):
    """
    Build a [T,H,W] cube from sparse rows. Missing cells -> 0.
    Returns (cube bool array, times array, uniq_lats array, uniq_lons array)
    with times sorted (hourly), lats ascending, lons ascending.
    """
    df = df.copy()
    df["time_h"] = df[time_col].dt.floor("h")

    times = np.array(sorted(df["time_h"].unique()))
    lats = np.array(sorted(df["lat"].unique()))
    lons = np.array(sorted(df["lon"].unique()))

    t_index = {t: i for i, t in enumerate(times)}
    lat_index = {v: i for i, v in enumerate(lats)}
    lon_index = {v: i for i, v in enumerate(lons)}

    T, H, W = len(times), len(lats), len(lons)
    cube = np.zeros((T, H, W), dtype=bool)

    # populate 1's for any nonzero flag
    for tval, g in df.groupby("time_h", sort=False):
        ti = t_index[tval]
        g_on = g.loc[g[flag_col].astype(bool)]
        if g_on.empty:
            continue
        ii = g_on["lat"].map(lat_index).to_numpy()
        jj = g_on["lon"].map(lon_index).to_numpy()
        cube[ti, ii, jj] = True

    return cube, times, lats, lons

def _apply_morphology(sl: np.ndarray, connectivity: int, do_morph: bool) -> np.ndarray:
    if not do_morph:
        return sl
    struct = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    return binary_closing(binary_opening(sl, structure=struct), structure=struct)

def _remove_small_components(sl: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 1:
        return sl
    struct = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    lab, n = cc_label(sl.astype(bool), structure=struct)
    if n == 0:
        return sl
    counts = np.bincount(lab.ravel())
    # drop labels with count < min_area (keep 0 background)
    drop = set(np.where(counts < min_area)[0])
    drop.discard(0)
    if not drop:
        return sl
    out = sl.copy()
    out[np.isin(lab, list(drop))] = False
    return out

def _spatial_pass(cube: np.ndarray,
                  min_neighbors: int,
                  connectivity: int,
                  min_area: int,
                  do_morph: bool,
                  do_neighbor: bool) -> np.ndarray:
    """Per-hour spatial filtering block with toggles."""
    out = np.zeros_like(cube, dtype=bool)
    T = cube.shape[0]
    for t in range(T):
        sl = cube[t]
        sl = _apply_morphology(sl, connectivity, do_morph)
        if min_area and min_area > 1:
            sl = _remove_small_components(sl, min_area=min_area, connectivity=connectivity)
        if do_neighbor:
            sl = _neighbor_filter(sl, min_neighbors=min_neighbors, connectivity=connectivity)
        out[t] = sl
        if t % 100 == 0:
            print(f"  processed {t}/{T} hours", flush=True)
    return out

def _persistence_mask(cube_bool: np.ndarray, hours: int, min_hits: int) -> np.ndarray:
    """
    Return mask where a cell is kept if within ANY sliding window of `hours`
    it fires at least `min_hits` times. Uses strided window over time axis.
    """
    T, H, W = cube_bool.shape
    if hours <= 1:
        # legacy behavior: any hit -> keep
        return cube_bool.copy()

    k = int(hours)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    # pad at front with zeros so first windows are well-defined
    pad = np.zeros((k - 1, H, W), dtype=bool)
    sw = swv(np.concatenate([pad, cube_bool], axis=0), window_shape=(k, 1, 1))
    # sw shape: (T + k -1 - k +1, k, 1, 1, H, W) == (T, k, 1, 1, H, W)
    # sum over window
    hits = sw.sum(axis=1)  # (T, 1, 1, H, W)
    keep = (hits >= int(min_hits)).reshape(T, H, W)
    return keep

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Denoise alerts with spatial neighbors + temporal persistence.")
    ap.add_argument("--alerts", required=True, help="Input alerts (CSV/CSV.GZ/Parquet)")
    ap.add_argument("--persist-hours", type=int, default=1, help="Temporal window length (hours) for persistence.")
    ap.add_argument("--persist-min-hits", type=int, default=1,
                    help="Keep a cell if it has at least this many hits within any persistence window.")
    ap.add_argument("--persist-mode", choices=["before","after","or-only"], default="after",
                    help="Apply persistence before spatial ops, after (kept if passes either), or use only persistence.")
    ap.add_argument("--min-neighbors", type=int, default=3, help="Min active neighbors to keep a cell in an hour.")
    ap.add_argument("--connectivity", type=int, choices=[4,8], default=4, help="Connectivity for morphology/CC (default 4).")
    ap.add_argument("--min-area", type=int, default=0, help="Remove connected components smaller than this size (0=off).")
    ap.add_argument("--no-morphology", action="store_true", help="Disable morphological open/close.")
    ap.add_argument("--skip-neighbor-filter", action="store_true", help="Disable neighbor-count requirement.")
    ap.add_argument("--sparse-output", action="store_true", help="Write only rows where alert_final==1.")
    ap.add_argument("--out", required=True, help="Output path (CSV/CSV.GZ/Parquet)")
    ap.add_argument("--time-col", default="time", help="Time column name if known (default: time)")
    ap.add_argument("--flag-col", default="alert", help="Binary flag column name (default: alert)")
    ap.add_argument("--time-format", default=None, help="Optional strftime format, e.g. '%Y/%m/%d %H:%M'")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = ap.parse_args()

    if not os.path.exists(args.alerts):
        raise FileNotFoundError(f"Input not found: {args.alerts}")

    if os.path.exists(args.out) and not args.overwrite:
        print(f"[DENOISE] Output exists, skipping (use --overwrite): {args.out}")
        sys.exit(0)

    print(f"[DENOISE] reading {args.alerts} …", flush=True)
    df = read_any(args.alerts)

    # Pick/validate columns
    time_col = _pick_time_col(df, args.time_col)
    required = {time_col, "lat", "lon", args.flag_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain columns {sorted(required)}; found {sorted(df.columns)}")

    # Parse time robustly
    t = _try_parse_time_raw(df[time_col], args.time_format)
    bad = int(t.isna().sum())
    if bad:
        frac = bad / len(t)
        print(f"[DENOISE] {bad:,} rows have invalid {time_col} ({frac:.1%}).", flush=True)
        if frac > 0.5:
            print("[DENOISE] Example raw values from time column:", flush=True)
            print(df.loc[t.isna(), time_col].head(12), flush=True)
            raise ValueError(
                f"Too many invalid {time_col} after robust parsing; pass --time-format if custom."
            )
    df[time_col] = t
    df = df.dropna(subset=[time_col]).reset_index(drop=True)

    # Numeric cleanups
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df[args.flag_col] = pd.to_numeric(df[args.flag_col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Sort for reproducibility
    df.sort_values([time_col, "lat", "lon"], inplace=True, ignore_index=True)

    # Build cube (sparse-safe)
    cube, times, lats, lons = _build_cube_sparse(df, time_col, args.flag_col)
    T, H, W = cube.shape
    print(f"Grid inferred (from sparse): H={H} W={W} T={T}", flush=True)

    # ---------- persistence & spatial logic ----------
    # Base signal (raw)
    base = cube.copy()

    # Persistence mask on the chosen input stream
    if args.persist_mode == "before":
        # Promote persistent raw hits BEFORE spatial pruning
        keep_persist = _persistence_mask(base, hours=int(args.persist_hours), min_hits=int(args.persist_min_hits))
        # OR the base with persistence (so persistent isolated pixels survive spatial ops)
        pre = np.logical_or(base, keep_persist)
        spatial_in = pre
        persist_to_or_after = None
    elif args.persist_mode == "or-only":
        # Only persistence; spatial ops are bypassed
        keep_persist = _persistence_mask(base, hours=int(args.persist_hours), min_hits=int(args.persist_min_hits))
        den = keep_persist
        persist_to_or_after = None
    else:  # "after" (default): apply spatial ops; then OR with persistence computed from raw base
        spatial_in = base
        persist_to_or_after = _persistence_mask(base, hours=int(args.persist_hours), min_hits=int(args.persist_min_hits))

    # Spatial cleanup (if not 'or-only')
    if args.persist_mode != "or-only":
        den_spatial = _spatial_pass(
            spatial_in,
            min_neighbors=int(args.min_neighbors),
            connectivity=int(args.connectivity),
            min_area=int(args.min_area),
            do_morph=(not args.no_morphology),
            do_neighbor=(not args.skip_neighbor_filter),
        )
        if persist_to_or_after is None:
            den = den_spatial
        else:
            # SAFE: keep anything that survives spatial OR is temporally persistent
            den = np.logical_or(den_spatial, persist_to_or_after)

    # Flatten back to a DataFrame
    tt = np.repeat(times, H * W)
    yy = np.tile(np.repeat(lats, W), T)
    xx = np.tile(lons, T * H)
    out = pd.DataFrame({
        time_col: tt,
        "lat": yy,
        "lon": xx,
        "alert_final": den.reshape(-1).astype(int)
    })

    if args.sparse_output:
        before = len(out)
        out = out.loc[out["alert_final"] == 1].reset_index(drop=True)
        print(f"[DENOISE] sparse-output: kept {len(out):,}/{before:,} rows", flush=True)

    # Write
    write_any(args.out, out)
    active = int(out["alert_final"].sum()) if "alert_final" in out.columns else 0
    print(f"Wrote {args.out} | active cells: {active:,}/{len(out):,}", flush=True)

    # Quick summary
    cov = out["alert_final"].mean() if "alert_final" in out.columns and len(out) else 0.0
    print(
        f"[DENOISE] Summary: T={T} H={H} W={W}  persist={args.persist_hours}h (min_hits={args.persist_min_hints if hasattr(args,'persist_min_hints') else args.persist_min_hits})  "
        f"mode={args.persist_mode}  min_neighbors={args.min_neighbors}  "
        f"conn={args.connectivity}  min_area={args.min_area}  coverage={cov:.3f}",
        flush=True
    )

if __name__ == "__main__":
    main()