#!/usr/bin/env python3
"""
Scalable analysis and plotting for 40GB multi-time, multi-column weather CSV.
- Converts CSV -> Parquet (partitioned by year/month)
- Computes key aggregations (daily means, spatial bins, correlations)
- Produces manageable plots from aggregated data
"""

import os
import argparse
import warnings

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import pandas as pd
import numpy as np

import hvplot.dask  # enables hvplot on dask objects
import hvplot.pandas
import datashader as ds
import holoviews as hv

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Configurable analysis choices
# -----------------------------
NUMERIC_COLS_DEFAULT = [
    "msl", "msl_d1h", "msl_d3h",
    "zeta", "div", "S",
    "zeta_mean3h", "zeta_std3h", "div_mean3h", "div_std3h",
    "S_mean3h", "S_std3h",
    "shear10_def", "shear_proxy", "S3",
    "dS_dt", "drelax_dt", "dagree_dt",
    "msl_grad",
    "u10", "v10", "wspd",
    "agree", "t2m",
    "shear_low", "shear_deep",
    "gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap",
    "gka_F", "gka_msl_nd", "gka_knee_ratio", "gka_chirality",
    "gka_Q", "gka_vortdiv_ratio", "gka_dir_var",
]

ID_COLUMNS = ["id", "ilat", "ilon", "number", "expver", "number_r", "expver_r"]  # kept but excluded from plots

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def infer_datetime(df, time_col="time"):
    # Robust UTC parsing for large data
    df[time_col] = dd.to_datetime(df[time_col], utc=True, errors="coerce")
    # Drop rows with invalid time (optional)
    df = df.dropna(subset=[time_col])
    return df

def add_time_fields(df, time_col="time"):
    # Derive year/month/day/hour for partitioning and grouping
    df = df.assign(
        year=df[time_col].dt.year.astype("int64"),
        month=df[time_col].dt.month.astype("int64"),
        day=df[time_col].dt.day.astype("int64"),
        hour=df[time_col].dt.hour.astype("int64"),
        date=df[time_col].dt.floor("D"),
    )
    return df

def normalize_lon(df, lon_col="lon"):
    # Normalize longitude to [-180, 180] for mapping consistency
    df[lon_col] = ((df[lon_col] + 180) % 360) - 180
    return df

def to_parquet_partitioned(df, out_dir, partition_cols=["year", "month"], write_index=False):
    ensure_dir(out_dir)
    df.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="snappy",
        write_index=write_index,
        partition_on=partition_cols,
        schema="infer",
    )

def select_numeric(df, numeric_cols):
    present = [c for c in numeric_cols if c in df.columns]
    return df[present]

# -----------------------------
# Aggregations
# -----------------------------
def daily_means(df, time_col="time", group_cols=None, numeric_cols=None):
    if group_cols is None:
        group_cols = []
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS_DEFAULT

    df = infer_datetime(df, time_col)
    df = add_time_fields(df, time_col)
    num = select_numeric(df, numeric_cols)

    # Group by date (+ optional group columns like lat/lon bins)
    grouped = num.groupby(df["date"])
    daily = grouped.mean().reset_index()
    return daily

def latlon_hexbin(df, lat_col="lat", lon_col="lon", binsize_deg=1.0, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = ["msl", "wspd", "zeta", "div", "S", "agree", "t2m"]

    # Bin to a coarse grid for spatial summaries (reduces plot size)
    df = normalize_lon(df, lon_col)
    lat_bin = (df[lat_col] / binsize_deg).floor() * binsize_deg
    lon_bin = (df[lon_col] / binsize_deg).floor() * binsize_deg

    df = df.assign(lat_bin=lat_bin, lon_bin=lon_bin)
    num = select_numeric(df, numeric_cols)

    spatial = num.groupby(["lat_bin", "lon_bin"]).mean().reset_index()
    return spatial

def correlations(df, numeric_cols=None, sample_rows=5_000_000):
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS_DEFAULT

    # Sample before correlation to keep memory controlled
    cols = [c for c in numeric_cols if c in df.columns]
    sampled = df[cols].sample(frac=None, n=sample_rows, random_state=42) if sample_rows else df[cols]
    corr = sampled.corr()  # dask computes lazily; triggers on compute downstream
    return corr

# -----------------------------
# Plotting (from aggregated data)
# -----------------------------
def plot_daily_series(daily_ddf, y_cols, out_dir, title_prefix="Daily mean"):
    ensure_dir(out_dir)
    # Convert to pandas for plotting
    daily = daily_ddf.compute()
    daily = daily.sort_values("date")

    for y in y_cols:
        if y not in daily.columns:
            continue
        plot = daily.hvplot.line(x="date", y=y, title=f"{title_prefix}: {y}", width=900, height=400)
        hv.save(plot, os.path.join(out_dir, f"daily_{y}.html"))

def plot_spatial_mean(spatial_ddf, value_col, out_dir, title=None):
    ensure_dir(out_dir)
    spatial = spatial_ddf.compute()

    # Quick raster-style plot using hvplot points (coarse bins)
    plot = spatial.hvplot.scatter(
        x="lon_bin", y="lat_bin", c=value_col, cmap="viridis",
        title=title or f"Spatial mean: {value_col}",
        width=900, height=450, colorbar=True, alpha=0.8, size=5
    )
    hv.save(plot, os.path.join(out_dir, f"spatial_mean_{value_col}.html"))

def plot_corr_heatmap(corr_ddf, out_dir, title="Correlation heatmap"):
    ensure_dir(out_dir)
    corr = corr_ddf.compute()
    # Use hvplot heatmap
    corr_reset = corr.reset_index().melt(id_vars="index", var_name="variable", value_name="corr")
    plot = corr_reset.hvplot.heatmap(x="index", y="variable", C="corr", cmap="coolwarm", clim=(-1, 1), width=900, height=900, title=title)
    hv.save(plot, os.path.join(out_dir, "correlation_heatmap.html"))

# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    # Dask cluster
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        memory_limit=args.worker_mem,  # e.g., "8GB"
        dashboard_address=args.dashboard,  # e.g., ":8787"
    )
    client = Client(cluster)

    print("Reading CSV with Dask...")
    dtypes = None  # optionally enforce dtypes here for speed/consistency
    df = dd.read_csv(
        args.csv,
        blocksize=args.blocksize,  # e.g., "256MB"
        assume_missing=True,       # safer for large files with mixed types
        dtype=dtypes,
    )

    # Basic sanitation: drop fully empty columns and trim spaces in headers
    df.columns = [c.strip() for c in df.columns]
    # Exclude plotting of IDs but keep them
    excluded_plot_cols = set(ID_COLUMNS)

    # Parse time and add time fields
    df = infer_datetime(df, time_col=args.time_col)
    df = add_time_fields(df, time_col=args.time_col)

    # Convert CSV -> Parquet for fast re-use
    if args.parquet_out:
        print("Writing partitioned Parquet...")
        to_parquet_partitioned(df, args.parquet_out, partition_cols=["year", "month"])

    # Daily means for selected metrics
    print("Computing daily means...")
    daily_cols = [c for c in ["msl", "wspd", "zeta", "div", "S", "agree", "t2m"] if c in df.columns]
    daily_ddf = daily_means(df, time_col=args.time_col, numeric_cols=daily_cols)

    # Spatial binning at 1° (adjustable)
    print("Computing spatial means...")
    spatial_cols = [c for c in ["msl", "wspd", "zeta", "div", "S", "agree", "t2m"] if c in df.columns]
    spatial_ddf = latlon_hexbin(df, numeric_cols=spatial_cols, binsize_deg=args.bin_deg)

    # Correlation matrix on a sample
    print("Computing correlations (sampled)...")
    corr_cols = [c for c in NUMERIC_COLS_DEFAULT if c in df.columns and c not in excluded_plot_cols]
    corr_ddf = correlations(df, numeric_cols=corr_cols, sample_rows=args.corr_sample)

    # Plot outputs (HTML files)
    print("Plotting...")
    if daily_cols:
        plot_daily_series(daily_ddf, daily_cols, args.out_dir, title_prefix="Daily mean")
    if spatial_cols:
        # Plot 2–3 representative fields to keep output manageable
        for value_col in spatial_cols[:3]:
            plot_spatial_mean(spatial_ddf, value_col, args.out_dir, title=f"Spatial mean ({args.bin_deg}° bins): {value_col}")
    if len(corr_cols) > 3:
        plot_corr_heatmap(corr_ddf, args.out_dir)

    print(f"Done. HTML outputs in: {args.out_dir}")
    print("Tip: open the Dask dashboard for progress monitoring if enabled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse and graph 40GB weather CSV using Dask.")
    parser.add_argument("--csv", required=True, help="Path to the large CSV file.")
    parser.add_argument("--time-col", default="time", help="Time column name.")
    parser.add_argument("--out-dir", default="outputs", help="Directory to save plots.")
    parser.add_argument("--parquet-out", default="parquet_out", help="Directory to write partitioned Parquet.")
    parser.add_argument("--bin-deg", type=float, default=1.0, help="Spatial bin size in degrees.")
    parser.add_argument("--blocksize", default="256MB", help="Dask CSV blocksize for chunking.")
    parser.add_argument("--workers", type=int, default=4, help="Number of Dask workers.")
    parser.add_argument("--threads", type=int, default=2, help="Threads per worker.")
    parser.add_argument("--worker-mem", default="8GB", help="Memory limit per worker (e.g., 8GB).")
    parser.add_argument("--dashboard", default=None, help="Dashboard address (e.g., :8787) or None.")
    parser.add_argument("--corr-sample", type=int, default=5_000_000, help="Rows to sample for correlation.")
    args = parser.parse_args()
    main(args)