#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_year_slim.py — build one season's genesis-analysis grid, storm-windowed and slim.

Runs the canonical feature chain (the same scripts the orchestrator calls) for a
single year, but:
  * crops to the union of per-storm space-time windows right after `build`
    (~6-15% of the full grid), and
  * DELETES each intermediate as soon as the next stage has consumed it, and
  * writes a slim final file with only the columns the genesis analyses need.

This keeps peak disk in the tens of GB instead of ~112 GB/year, which is what
makes multi-year processing possible on a nearly-full disk.

USAGE
    python build_year_slim.py --year 2021
    python build_year_slim.py --year 2021 --keep-intermediates --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FEAT = HERE / "features_subprocess"
DATA = Path("data")

# columns the genesis analyses actually consume
SLIM_COLS = [
    "time", "lat", "lon", "ilat", "ilon",
    "pregen", "near_storm", "storm", "t_to_storm_min_h",
    "zeta", "zeta_mean3h", "zeta_std3h", "div", "msl", "msl_d1h", "msl_d3h",
    "S", "S3", "shear_low", "shear_deep", "shear_proxy", "dS_dt",
    "agree", "drelax_dt", "dagree_dt", "u10", "v10",
    "SFI", "SFI2", "sph_vdr_std", "pdrop_nd", "thermo_shear", "t2m_anom_local",
    "gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap", "gka_F",
    "gka_msl_nd", "gka_knee_ratio", "gka_chirality", "gka_Q",
    "gka_vortdiv_ratio", "gka_dir_var", "gka_SAI", "gka_SII",
    "gka_spin_coh", "gka_build", "gka_relax", "gka_shear_quench",
    "gka_knee_ms", "gka_score",
]


def sh(cmd, dry=False) -> None:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(f"[year] step failed (exit {p.returncode}): {' '.join(cmd[:3])}")
    print(f"[year] ok in {time.time()-t0:,.0f}s", flush=True)


def gb(p: Path) -> float:
    try:
        return p.stat().st_size / 1073741824
    except Exception:
        return 0.0


def drop(p: Path, keep: bool) -> None:
    if keep or not p.exists():
        return
    size = gb(p)
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"[year] freed {size:.2f} GB  ({p.name})", flush=True)
    except Exception as e:
        print(f"[year] warn: could not delete {p}: {e}", file=sys.stderr)


def free_gb() -> float:
    return shutil.disk_usage(".").free / 1073741824


def main() -> int:
    a = parse_args()
    Y = a.year
    out_dir = DATA / f"y{Y}"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = a.keep_intermediates
    dry = a.dry_run

    tracks = DATA / "tracks" / f"tracks_{Y}.parquet"
    nc_glob = f"data_era5/extracted/{Y}/**/era5_{Y}*_oper.nc"
    pl_glob = f"data_era5/extracted/{Y}/**/era5_pl_{Y}*_uv.nc"

    f_eoi   = out_dir / "features_eoi.parquet"
    f_win   = out_dir / "features_eoi_win.parquet"
    f_shear = out_dir / "features_bulk_shear.parquet"
    f_merge = out_dir / "features_merged.parquet"
    f_patch = out_dir / "features_patched.parquet"
    f_gka   = out_dir / "grid_gka.parquet"
    f_therm = out_dir / "grid_realthermo.parquet"
    f_sph   = out_dir / "grid_sph.parquet"
    f_ms    = out_dir / "grid_sph_ms.parquet"
    f_lab   = out_dir / "grid_labelled.parquet"
    f_id    = out_dir / "grid_labelled_id.parquet"
    f_slim  = DATA / f"genesis_{Y}_slim.parquet"

    print(f"=== build_year_slim {Y} ===  free={free_gb():.1f} GB")

    # 0) tracks for the season
    if not tracks.exists():
        sh([sys.executable, HERE / "fetch_subprocess" / "ibtracs_fetch.py",
            "--out", tracks, "--start", f"{Y}-02-01", "--end", f"{Y}-05-31",
            f"--area={a.area}", "--normalize-lon=-180..180"], dry)
    if not dry:
        n = pd.read_parquet(tracks)["storm_id"].nunique()
        print(f"[year] {Y}: {n} storms")

    # 1) base grid from ERA5 single-level
    sh([sys.executable, FEAT / "build_features_grid.py",
        "--nc-glob", nc_glob, "--out", f_eoi,
        "--normalize-lon=-180..180", f"--area={a.area}",
        "--export-uv", "--with-vortdiv",
        "--require-vars", "u10,v10,msl,t2m",
        "--dedup", "time_lat_lon", "--emit-grid-index", "--overwrite"], dry)

    # 2) crop to the union of per-storm windows  <-- the disk win
    sh([sys.executable, FEAT / "storm_window_subset.py",
        "--infile", f_eoi, "--outfile", f_win, "--tracks", tracks,
        "--pad-deg", a.pad_deg, "--pre-h", a.pre_h, "--post-h", a.post_h,
        "--overwrite"], dry)
    drop(f_eoi, keep)

    # 3) pressure-level bulk shear
    sh([sys.executable, FEAT / "features_bulk_shear.py",
        "--pl-glob", pl_glob, "--out", f_shear,
        "--low-pair", "1000,925", "--deep-pair", "1000,500",
        "--normalize-lon=-180..180", f"--area={a.area}", "--overwrite"], dry)

    # 4) join shear onto the windowed grid
    sh([sys.executable, FEAT / "features_join_features.py",
        "--left", f_win, "--right", f_shear, "--on", "time,lat,lon",
        "--out", f_merge, "--overwrite"], dry)
    drop(f_win, keep); drop(f_shear, keep)

    # 5) patch / rolling stats / shear proxies / S3
    sh([sys.executable, FEAT / "features_patch.py",
        "--in", f_merge, "--out", f_patch,
        "--neighbor-step", "0.0", "--radius-cells", "1",
        "--prefer-shear", "shear_deep", "--s3-window", "3", "--overwrite"], dry)
    drop(f_merge, keep)

    # 6) GKA features
    sh([sys.executable, FEAT / "compute_gka_features.py",
        "--infile", f_patch, "--outfile", f_gka,
        "--allow-S-from-S3", "--overwrite"], dry)
    drop(f_patch, keep)

    # 7) integrate ERA5 thermo
    sh([sys.executable, FEAT / "integrate_era5_thermo.py",
        "--features", f_gka, "--thermo-glob", nc_glob, "--out", f_therm,
        "--nearest", "--nearest-maxdeg", "0.4",
        "--normalize-lon=-180..180", "--overwrite"], dry)
    drop(f_gka, keep)

    # 8) spherical feedback (SFI / SFI2)
    sh([sys.executable, FEAT / "compute_spherical_feedback.py",
        "--infile", f_therm, "--out", f_sph,
        "--neighbor-step", "0.0", "--radius-cells", "2",
        "--normalize-lon=-180..180", f"--area={a.area}",
        "--lead-hours", "24", "--overwrite"], dry)
    drop(f_therm, keep)

    # 9) multi-scale GKA
    sh([sys.executable, FEAT / "compute_gka_multiscale.py",
        "--infile", f_sph, "--outfile", f_ms,
        "--parquet-rows", "250000", "--overwrite"], dry)
    drop(f_sph, keep)

    # 10) labels (pregen / near_storm / t_to_storm_min_h)
    sh([sys.executable, FEAT / "join_labels_grid.py",
        "--features", f_ms, "--labels", tracks, "--out", f_lab,
        "--normalize-lon=-180..180",
        "--pregen_future_h", "240.0", "--pregen-step", "1",
        "--storm_radius_deg", "1.0", "--storm_time_h", "3.0",
        "--near_radius_deg", "5.0", "--near_time_h", "12.0",
        "--features-chunk-rows", "400000", "--chunk-hours", "72",
        "--overwrite"], dry)
    drop(f_ms, keep)

    # 11) row ids
    sh([sys.executable, FEAT / "add_row_id.py",
        "--infile", f_lab, "--outfile", f_id,
        "--chunk-rows", "200000", "--overwrite"], dry)
    drop(f_lab, keep)

    # 12) slim to analysis columns
    if not dry:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(str(f_id)).schema.names)
        cols = [c for c in SLIM_COLS if c in have]
        missing = [c for c in SLIM_COLS if c not in have]
        if missing:
            print(f"[year] note: {len(missing)} slim cols absent: {missing}")
        df = pd.read_parquet(f_id, columns=cols)
        df["season"] = Y
        df.to_parquet(f_slim, index=False)
        print(f"[year] slim -> {f_slim}  rows={len(df):,}  {gb(f_slim):.2f} GB  cols={len(cols)+1}")
    drop(f_id, keep)
    if not keep and out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()

    print(f"\n[year] {Y} DONE.  free={free_gb():.1f} GB")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Storm-windowed, slim, single-season genesis grid build.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--area", default="-5,125,-35,175", help="latN,lonW,latS,lonE")
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--post-h", type=float, default=24.0)
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
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
