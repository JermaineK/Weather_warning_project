#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_gka_features.py — alias-aware, CSV/Parquet compatible, memory-safe η

Adds classic light-form GKA features and first-order geometric-kernel responses.

Highlights:
  • Streams CSV (.csv/.csv.gz) in chunks; handles Parquet in-memory.
  • Verbose error reporting with tracebacks.
  • Robust alias binding for columns (u/v/zeta/div/msl/S/S3/etc).
  • Memory-safe parity-eta using integer keys (prefers ilat/ilon when present).
  • Guards: --disable-eta, --eta-max-rows, --eta-window.

Usage examples
--------------
# Parquet → Parquet
python compute_gka_features.py \
  --infile data/grid_labelled_base.parquet \
  --outfile data/grid_labelled_FMA_gka.parquet \
  --allow-S-from-S3 --verbose --overwrite

# CSV → CSV.GZ (streamed)
python compute_gka_features.py \
  --infile data/grid_labelled_base.csv.gz \
  --outfile data/grid_labelled_FMA_gka.csv.gz \
  --chunksize 500000 --allow-S-from-S3 --overwrite
"""

from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# ---------------- configuration ----------------

NEW_COLS = [
    "gka_kappa","gka_tau","gka_parity_eta","gka_A_overlap",
    "gka_F","gka_msl_nd","gka_knee_ratio",
    "gka_chirality","gka_Q","gka_dir_var","gka_vortdiv_ratio"
]

ALIASES: Dict[str, List[str]] = {
    "u":    ["u", "u10", "U10M"],
    "v":    ["v", "v10", "V10M"],
    "zeta": ["zeta", "zeta_mean"],
    "div":  ["div", "div_mean"],
    "msl":  ["msl", "mean_sea_level_pressure", "MSL", "mslp"],
    "agree":["agree", "agreement", "overlap"],
    "shear":["shear", "shear_proxy", "sh", "shear10_def"],
    "S":    ["S"],
    "S3":   ["S3", "S_mean3h"],
    # fallbacks for direction variance if only u10/v10 exist:
    "u10":  ["u10"],
    "v10":  ["v10"],
}

CORE_FOR_REQUIRE = ("zeta","div","u","v")  # enforced when --require-core


# ---------------- utilities ----------------

def _first_present(cols: Iterable[str], pool: List[str]) -> Optional[str]:
    s = set(cols)
    for c in pool:
        if c in s:
            return c
    return None

def _bind_columns(cols: Iterable[str], allow_S_from_S3: bool=False) -> Dict[str, Optional[str]]:
    cols = list(cols)
    bound = {canon: _first_present(cols, choices) for canon, choices in ALIASES.items()}
    if allow_S_from_S3 and not bound.get("S") and bound.get("S3"):
        bound["S"] = bound["S3"]
    return bound

def _safe_num(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def print_bindings(bind: Dict[str, Optional[str]]):
    print("\n[GKA] Column bindings:")
    for k in ("S","S3","agree","div","msl","shear","u","v","zeta"):
        v = bind.get(k)
        print(f"  {k:5s} ->  {v if v else '(missing)'}")

def _pick_series(out_df: pd.DataFrame, bind: Dict[str, Optional[str]], *canon_names: str) -> Optional[pd.Series]:
    """Return the first bound Series among the provided canonical names."""
    for name in canon_names:
        colname = bind.get(name)
        if colname and colname in out_df.columns:
            return out_df[colname]
    return None


# ---------- memory-safe parity-eta ----------

def _combine_codes(lat_codes: np.ndarray, lon_codes: np.ndarray) -> np.ndarray:
    """
    Combine two non-negative int32 codes into a single int64 key without allocating tuples.
    """
    lat_i = lat_codes.astype(np.int64, copy=False)
    lon_i = lon_codes.astype(np.int64, copy=False)
    return (lat_i << 32) ^ (lon_i & 0xFFFFFFFF)


def _group_codes(lat: pd.Series,
                 lon: pd.Series,
                 ilat: Optional[pd.Series] = None,
                 ilon: Optional[pd.Series] = None) -> np.ndarray:
    """
    Prefer integer grid indices ilat/ilon if provided; otherwise factorize lat/lon separately.
    Returns an int64 array of keys (one per row).
    """
    if ilat is not None and ilon is not None:
        lat_codes = pd.to_numeric(ilat, errors="coerce").astype("Int64").to_numpy()
        lon_codes = pd.to_numeric(ilon, errors="coerce").astype("Int64").to_numpy()
    else:
        lat_codes, _ = pd.factorize(lat, sort=False)
        lon_codes, _ = pd.factorize(lon, sort=False)

    return _combine_codes(
        np.asarray(lat_codes, dtype=np.int32),
        np.asarray(lon_codes, dtype=np.int32),
    )


def _roll_group_eta(sign_series: pd.Series,
                    lat: pd.Series,
                    lon: pd.Series,
                    ilat: Optional[pd.Series] = None,
                    ilon: Optional[pd.Series] = None,
                    window: int = 7) -> np.ndarray:
    """
    Rolling mean of sign(zeta) per grid cell using integer keys.

    If ilat/ilon are provided, they define the groups; otherwise we factorize lat/lon.
    """
    keys = _group_codes(lat, lon, ilat=ilat, ilon=ilon)
    s = pd.Series(sign_series.to_numpy(), index=np.arange(len(sign_series)))
    m = (
        s.groupby(keys, sort=False)
         .rolling(window, min_periods=1, center=True)
         .mean()
         .reset_index(level=0, drop=True)
    )
    return m.to_numpy()


# ---------------- feature computation ----------------

def _compute_chunk_features(df: pd.DataFrame,
                            bind: Dict[str, Optional[str]],
                            verbose: bool=False,
                            disable_eta: bool=False,
                            eta_window: int=7,
                            eta_max_rows: int=5_000_000) -> pd.DataFrame:
    # sanity: coords
    need = {"lat","lon"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Missing required coordinates in chunk: need {sorted(need)}, have {list(df.columns)[:15]}...")

    out = df.copy()

    # fetch bound columns safely
    zeta  = _pick_series(out, bind, "zeta")
    divv  = _pick_series(out, bind, "div")
    agree = _pick_series(out, bind, "agree")
    msl   = _pick_series(out, bind, "msl")
    shear = _pick_series(out, bind, "shear")
    S     = _pick_series(out, bind, "S")
    S3    = _pick_series(out, bind, "S3")
    u     = _pick_series(out, bind, "u", "u10")
    v     = _pick_series(out, bind, "v", "v10")

    if verbose:
        print(
            f"[GKA] chunk rows={len(out):,}  present:"
            f" zeta={zeta is not None} div={divv is not None} msl={msl is not None}"
            f" S={S is not None} S3={S3 is not None} shear={shear is not None}"
            f" u={u is not None} v={v is not None}"
        )

    # 1) curvature / torsion proxies
    out["gka_kappa"] = _safe_num(zeta) if zeta is not None else 0.0
    out["gka_tau"]   = -_safe_num(divv) if divv is not None else 0.0

    # 2) parity-odd local eta (guarded)
    if not disable_eta and (zeta is not None) and {"lat","lon"}.issubset(out.columns):
        if len(out) > eta_max_rows:
            if verbose:
                print(f"[GKA] eta skipped: rows {len(out):,} > eta_max_rows {eta_max_rows:,}")
            out["gka_parity_eta"] = 0.0
        else:
            try:
                z_vals = _safe_num(zeta).to_numpy(float)
                signv = np.sign(z_vals)
                eta = _roll_group_eta(
                    pd.Series(signv, index=out.index),
                    out["lat"],
                    out["lon"],
                    out["ilat"] if "ilat" in out.columns else None,
                    out["ilon"] if "ilon" in out.columns else None,
                    window=max(1, int(eta_window)),
                )
                out["gka_parity_eta"] = eta
            except MemoryError:
                if verbose:
                    print("[GKA] eta computation hit MemoryError; filling with zeros.")
                out["gka_parity_eta"] = 0.0
            except Exception as ex:
                if verbose:
                    print(f"[GKA] eta computation failed ({type(ex).__name__}: {ex}); filling zeros.")
                out["gka_parity_eta"] = 0.0
    else:
        out["gka_parity_eta"] = 0.0

    # 3) overlap proxy
    out["gka_A_overlap"] = _safe_num(agree) if agree is not None else 0.0

    # 4) freedom-of-movement from shear proxy (logistic squash of z-score)
    if shear is not None:
        shp = _safe_num(shear).to_numpy(float)
        med = np.nanmedian(shp)
        mad = np.nanmean(np.abs(shp - med)) + 1e-6
        z = (shp - med)/mad
        out["gka_F"] = 1/(1+np.exp(z))
    else:
        out["gka_F"] = 0.0

    # 5) msl dimensionless (median/MAD)
    if msl is not None:
        m = _safe_num(msl).to_numpy(float)
        m_med = np.nanmedian(m)
        m_mad = np.nanmean(np.abs(m - m_med)) + 1e-6
        out["gka_msl_nd"] = (m - m_med)/m_mad
    else:
        out["gka_msl_nd"] = 0.0

    # 6) knee ratio (S / S3)
    if (S is not None) and (S3 is not None):
        S3v = _safe_num(S3).to_numpy(float)
        denom = np.where(np.abs(S3v) < 1e-9, 1e-9, S3v)
        out["gka_knee_ratio"] = _safe_num(S).to_numpy(float) / denom
    else:
        out["gka_knee_ratio"] = 0.0

    # === geometric-kernel responses ===
    if zeta is not None:
        z = _safe_num(zeta).to_numpy(float)
        out["gka_chirality"] = np.sign(z) * np.abs(z)
    else:
        out["gka_chirality"] = 0.0

    if (zeta is not None) and (divv is not None):
        z = _safe_num(zeta).to_numpy(float)
        d = _safe_num(divv).to_numpy(float)
        out["gka_Q"] = d**2 - z**2
        denom = (np.abs(d) + np.abs(z))
        out["gka_vortdiv_ratio"] = np.where(denom > 1e-9, z/denom, 0.0)
    else:
        out["gka_Q"] = 0.0
        out["gka_vortdiv_ratio"] = 0.0

    if (u is not None) and (v is not None):
        uu = _safe_num(u).to_numpy(float)
        vv = _safe_num(v).to_numpy(float)
        theta = np.arctan2(vv, uu)
        sinm, cosm = np.nanmean(np.sin(theta)), np.nanmean(np.cos(theta))
        R = np.hypot(sinm, cosm)
        out["gka_dir_var"] = 1.0 - R
    else:
        out["gka_dir_var"] = 0.0

    # sanitize dtypes for new cols (downcast to float32 to keep size reasonable)
    for c in NEW_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    return out


# ---------------- I/O helpers ----------------

def _is_parquet(path: str | Path) -> bool:
    p = str(path).lower()
    return p.endswith(".parquet") or p.endswith(".parq") or p.endswith(".pq")

def _comp_for_csv(path: str | Path) -> str:
    p = str(path).lower()
    return "gzip" if p.endswith(".gz") else "infer"

def _peek_columns(path: str | Path, verbose: bool=False) -> List[str]:
    if _is_parquet(path):
        df = pd.read_parquet(path, engine="pyarrow")
        cols = list(df.columns)
    else:
        head = pd.read_csv(path, nrows=5, low_memory=False)
        cols = list(head.columns)
    if verbose:
        print(f"[GKA] peek columns ({len(cols)}): {cols[:40]}{' ...' if len(cols)>40 else ''}")
    return cols

def _stream_input(path: str | Path, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrames; parquet yields a single full frame."""
    if _is_parquet(path):
        yield pd.read_parquet(path, engine="pyarrow")
    else:
        parse_dates = ["time"]  # best effort; if absent, pandas will ignore
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, parse_dates=parse_dates):
            yield chunk

def _write_stream_csv(path: str | Path, df: pd.DataFrame, first: bool) -> None:
    comp = _comp_for_csv(path)
    mode = "w" if first else "a"
    df.to_csv(path, index=False, mode=mode, header=first, compression=comp)

def _write_parquet(path: str | Path, df: pd.DataFrame, overwrite: bool) -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise SystemExit(f"[GKA] Outfile exists; use --overwrite: {p}")
    df.to_parquet(p, index=False, engine="pyarrow")


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="Rows per chunk when reading CSV(.gz). Ignored for Parquet.")
    ap.add_argument("--require-core", action="store_true",
                    help="Require {zeta, div, u, v} (after alias binding).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow replacing an existing outfile.")
    ap.add_argument("--verbose", action="store_true",
                    help="Extra logging.")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Abort on first CSV chunk error.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Print head of input (N rows) after reading.")
    ap.add_argument("--allow-S-from-S3", action="store_true",
                    help="If S is missing, bind S := S3.")
    # η controls
    ap.add_argument("--disable-eta", action="store_true",
                    help="Skip gka_parity_eta computation entirely.")
    ap.add_argument("--eta-window", type=int, default=7,
                    help="Centered rolling window for eta (per (lat,lon)).")
    ap.add_argument("--eta-max-rows", type=int, default=5_000_000,
                    help="Skip eta if chunk has more than this many rows.")
    return ap.parse_args()


# ---------------- main ----------------

def main():
    args = parse_args()

    if Path(args.infile).resolve() == Path(args.outfile).resolve():
        raise SystemExit("[GKA] infile and outfile are the same path; please choose a different outfile.")

    in_is_parq  = _is_parquet(args.infile)
    out_is_parq = _is_parquet(args.outfile)

    # Bind aliases from the real columns
    cols = _peek_columns(args.infile, verbose=args.verbose)
    bind = _bind_columns(cols, allow_S_from_S3=args.allow_S_from_S3)
    print_bindings(bind)

    if args.require_core:
        missing = [c for c in CORE_FOR_REQUIRE if not bind.get(c)]
        if missing:
            raise SystemExit(f"[GKA] --require-core set but missing columns after alias binding: {missing}")

    # CSV streaming path
    if not in_is_parq and not out_is_parq:
        if Path(args.outfile).exists() and not args.overwrite:
            raise SystemExit(f"[GKA] Outfile exists; use --overwrite: {args.outfile}")

        first = True
        total_rows = 0
        bad_chunks = 0
        for i, chunk in enumerate(_stream_input(args.infile, args.chunksize), 1):
            try:
                if args.sample and i == 1:
                    print(f"[GKA] sample head ({min(args.sample, len(chunk))} rows):")
                    print(chunk.head(args.sample))
                # rebind in case CSV chunks differ in header (rare but safe)
                b = _bind_columns(list(chunk.columns), allow_S_from_S3=args.allow_S_from_S3)
                if args.require_core:
                    miss = [c for c in CORE_FOR_REQUIRE if not b.get(c)]
                    if miss:
                        raise SystemExit(f"[GKA] --require-core set but missing columns in this chunk: {miss}")

                if not {"lat","lon"}.issubset(set(chunk.columns)):
                    raise ValueError(f"Chunk {i} missing lat/lon; columns={list(chunk.columns)[:20]}")

                out = _compute_chunk_features(
                    chunk, b, verbose=args.verbose,
                    disable_eta=args.disable_eta,
                    eta_window=args.eta_window,
                    eta_max_rows=args.eta_max_rows,
                )
                _write_stream_csv(args.outfile, out, first=first)
                first = False
                total_rows += len(out)
                if args.verbose:
                    print(f"[GKA] chunk {i} ok → rows {len(out):,}  total {total_rows:,}")
            except Exception as ex:
                bad_chunks += 1
                print(
                    "\n[GKA] ERROR in CSV chunk {}: {}: {}\n{}".format(
                        i, type(ex).__name__, repr(ex), traceback.format_exc()
                    ),
                    file=sys.stderr,
                )
                if args.fail_fast:
                    raise

        if first:
            raise SystemExit("[GKA] No chunks were successfully processed; outfile not created.")
        print(f"\n[GKA] Wrote {args.outfile}  rows: {total_rows:,}  bad_chunks: {bad_chunks}")
        return

    # Parquet input path (reads once); output can be Parquet or CSV
    df = next(iter(_stream_input(args.infile, args.chunksize)))
    if args.sample:
        print(f"[GKA] sample head ({min(args.sample, len(df))} rows):")
        print(df.head(args.sample))

    if not {"lat","lon"}.issubset(set(df.columns)):
        raise SystemExit(f"[GKA] Input missing required coord columns lat/lon; have {list(df.columns)[:20]}")

    out = _compute_chunk_features(
        df, bind, verbose=args.verbose,
        disable_eta=args.disable_eta,
        eta_window=args.eta_window,
        eta_max_rows=args.eta_max_rows,
    )

    if out_is_parq:
        _write_parquet(args.outfile, out, overwrite=args.overwrite)
        print(f"[GKA] Wrote {args.outfile}  rows={len(out):,} cols={out.shape[1]}")
    else:
        _write_stream_csv(args.outfile, out, first=True)
        print(f"[GKA] Wrote {args.outfile}  rows={len(out):,} cols={out.shape[1]}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        msg = f"\nERROR [{type(e).__name__}]: {repr(e)}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        sys.exit(1)