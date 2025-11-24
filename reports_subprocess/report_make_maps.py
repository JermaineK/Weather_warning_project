#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_make_maps.py

Quick, dependency-light QA maps:
  - scatter of seed union points (by hour or cell)
  - scatter of patch centroids

Intended to be called by a higher-level run manager which:
  - Creates a unique run folder, e.g. results/reports/20251122_run001
  - Passes that as --out-dir (optionally plus a figs/ subfolder)

This script does *not* try to manage run IDs itself; it just writes PNGs
into the given --out-dir without overwriting protection (the manager
is responsible for making that unique per run).
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


def main():
    ap = argparse.ArgumentParser(description="Simple PNG maps for quick QA (union + patch centroids).")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional logical run label (used only in titles; manager should provide per-run out-dir).",
    )
    ap.add_argument(
        "--union-csv",
        default="results/seedmaps/union_byhour.csv",
        help="Seed union CSV/Parquet with at least lat,lon[,prob_max].",
    )
    ap.add_argument(
        "--patches-csv",
        default="results/seedmaps/seed_patches.csv",
        help="Seed patches CSV/Parquet with lat_cen,lon_cen.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/figs",
        help="Output folder for PNGs (manager should make this per-run).",
    )
    ap.add_argument(
        "--union-value-col",
        default="prob_max",
        help="Optional numeric column in union file to colour by (default: prob_max).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_label = f" — {args.run_name}" if args.run_name else ""

    # Union map
    try:
        upath = Path(args.union_csv)
        if upath.suffix.lower() in (".parquet", ".pq", ".pqt"):
            u = pd.read_parquet(upath)
        else:
            u = pd.read_csv(upath)
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
        print(f"[maps] union file not found: {args.union_csv}")
    except Exception as e:
        print(f"[maps] error reading union file {args.union_csv}: {e}")

    # Patch centroids
    try:
        ppath = Path(args.patches_csv)
        if ppath.suffix.lower() in (".parquet", ".pq", ".pqt"):
            p = pd.read_parquet(ppath)
        else:
            p = pd.read_csv(ppath)
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
        print(f"[maps] patches file not found: {args.patches_csv}")
    except Exception as e:
        print(f"[maps] error reading patches file {args.patches_csv}: {e}")

    print(f"[maps] QA PNGs (if any) are under {out_dir}")


if __name__ == "__main__":
    main()