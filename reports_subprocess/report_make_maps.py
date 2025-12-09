#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_make_maps.py

Quick, dependency-light QA maps:
  - scatter of seed union points (by hour or cell)
  - scatter of patch centroids

Pipeline-aware defaults:
  * If --run-name is provided and no explicit paths are given:
      union file   -> results/seedmaps/<run_name>_union_byhour.csv
      patches file -> results/seedmaps/<run_name>_seed_patches.csv
      out dir      -> results/reports/<run_name>/figs
  * If explicit paths are given, they win.
  * If neither run-name nor explicit paths are given, fall back to the
    legacy defaults:
      union file   -> results/seedmaps/union_byhour.csv
      patches file -> results/seedmaps/seed_patches.csv
      out dir      -> results/figs

This script does *not* try to manage run IDs itself; it just writes PNGs
into the given --out-dir. Overwrite protection is the caller's job.
"""

import argparse
from pathlib import Path

import pandas as pd


def try_imports():
    """Return pyplot if available; otherwise None (graceful no-op)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # noqa: F401
        return plt
    except Exception:
        return None


def plot_points(df: pd.DataFrame, out_png: Path, title: str, value_col: str | None = None):
    """
    Simple lat/lon scatter plot.

    - Caps sample at 10k points for sanity.
    - If value_col is present & numeric-ish, use as colour; otherwise fixed colour.
    """
    plt = try_imports()
    if plt is None:
        print(f"[maps] matplotlib not available; skipping {out_png.name}")
        return
    if df.empty:
        print(f"[maps] no rows to plot for {out_png.name}; skipping.")
        return

    # Pick columns
    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"[maps] missing lat/lon in dataframe; skipping {out_png.name}")
        return

    d = df.copy()
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
    d = d.dropna(subset=["lat", "lon"])
    if d.empty:
        print(f"[maps] no finite lat/lon for {out_png.name}; skipping.")
        return

    # Sample to avoid huge PNGs
    if len(d) > 10_000:
        d = d.sample(10_000, random_state=42)

    cvals = None
    clabel = None
    if value_col and value_col in d.columns:
        cvals = pd.to_numeric(d[value_col], errors="coerce")
        if cvals.notna().any():
            clabel = value_col
        else:
            cvals = None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    if cvals is not None:
        sc = ax.scatter(d["lon"], d["lat"], s=6, c=cvals, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(clabel or value_col)
    else:
        ax.scatter(d["lon"], d["lat"], s=6, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[maps] wrote {out_png}")


def _infer_union_path(run_name: str | None, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if run_name:
        return Path("results/seedmaps") / f"{run_name}_union_byhour.csv"
    return Path("results/seedmaps/union_byhour.csv")


def _infer_patches_path(run_name: str | None, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if run_name:
        return Path("results/seedmaps") / f"{run_name}_seed_patches.csv"
    return Path("results/seedmaps/seed_patches.csv")


def _infer_out_dir(run_name: str | None, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if run_name:
        return Path("results/reports") / run_name / "figs"
    return Path("results/figs")


def _read_any(path: Path) -> pd.DataFrame:
    suf = "".join(path.suffixes).lower()
    if suf.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description="Simple PNG maps for quick QA (union + patch centroids).")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Logical run label; used in titles and to infer default input/output paths.",
    )
    ap.add_argument(
        "--union-path",
        "--union-csv",
        default=None,
        help=(
            "Explicit path to seed union file (CSV/Parquet with lat,lon[,prob_max]). "
            "If omitted, inferred from --run-name."
        ),
    )
    ap.add_argument(
        "--patches-path",
        "--patches-csv",
        default=None,
        help=(
            "Explicit path to seed patches file (CSV/Parquet with lat_cen,lon_cen). "
            "If omitted, inferred from --run-name."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output folder for PNGs. If omitted, uses results/reports/<run_name>/figs when --run-name is set, "
            "else results/figs."
        ),
    )
    ap.add_argument(
        "--union-value-col",
        default="prob_max",
        help="Optional numeric column in union file to colour by (default: prob_max).",
    )
    args = ap.parse_args()

    out_dir = _infer_out_dir(args.run_name, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_label = f" — {args.run_name}" if args.run_name else ""

    # Union map
    try:
        upath = _infer_union_path(args.run_name, args.union_path)
        u = _read_any(upath)
        if {"lat", "lon"}.issubset(u.columns):
            plot_points(
                u,
                out_dir / "union_points.png",
                f"Seed Union (by hour){run_label}",
                value_col=args.union_value_col,
            )
        else:
            print(f"[maps] union file missing lat/lon: {upath}")
    except FileNotFoundError:
        print(f"[maps] union file not found: {upath}")
    except Exception as e:
        print(f"[maps] error reading union file {upath}: {e}")

    # Patch centroids
    try:
        ppath = _infer_patches_path(args.run_name, args.patches_path)
        p = _read_any(ppath)
        if {"lat_cen", "lon_cen"}.issubset(p.columns):
            rename = p.rename(columns={"lat_cen": "lat", "lon_cen": "lon"})
            plot_points(
                rename,
                out_dir / "patch_centroids.png",
                f"Patch centroids{run_label}",
                value_col=None,
            )
        else:
            print(f"[maps] patches file missing lat_cen/lon_cen: {ppath}")
    except FileNotFoundError:
        print(f"[maps] patches file not found: {ppath}")
    except Exception as e:
        print(f"[maps] error reading patches file {ppath}: {e}")

    print(f"[maps] QA PNGs (if any) are under {out_dir}")


if __name__ == "__main__":
    main()
