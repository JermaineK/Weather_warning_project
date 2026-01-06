#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storm_hourly_counts.py
Compute per-storm hourly point counts from a matches table and
optionally render a heatmap for quick inspection.

Agent: add storm-hourly counts + heatmap for mapping diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_any(path: str | Path) -> pd.DataFrame:
    low = str(path).lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _pick_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _to_time_h(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")


def _heatmap(pivot: pd.DataFrame, out_png: Path, title: str, vmax: Optional[float]) -> None:
    import matplotlib.pyplot as plt  # local import to avoid hard dependency in non-plot runs

    data = pivot.to_numpy()
    fig_w = max(10.0, min(20.0, 0.25 * data.shape[1]))
    fig_h = max(6.0, min(14.0, 0.4 * data.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmax=vmax)

    # ticks
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(s) for s in pivot.index], fontsize=8)
    if data.shape[1] > 1:
        tick_count = min(10, data.shape[1])
        idx = np.linspace(0, data.shape[1] - 1, tick_count, dtype=int)
        labels = [pivot.columns[i].strftime("%m-%d %H") for i in idx]
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Storm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="points per hour")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _timeline(counts: pd.DataFrame, out_png: Path, title: str, top_storms: int, min_total: int) -> None:
    import matplotlib.pyplot as plt  # local import to avoid hard dependency in non-plot runs

    totals = counts.groupby("storm_id")["points"].sum().sort_values(ascending=False)
    keep = totals.loc[totals >= int(min_total)].head(int(top_storms)).index.tolist()
    subset = counts.loc[counts["storm_id"].isin(keep)]
    if subset.empty:
        print("[storm-hourly] no storms meet min-total for timeline; skipping plot.")
        return

    storms = keep
    n = len(storms)
    fig_h = max(4.0, min(18.0, 1.4 * n))
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, storms):
        sub = subset.loc[subset["storm_id"] == sid]
        ax.plot(sub["time_h"], sub["points"], color="#1f77b4", lw=1.2, marker="o", ms=3)
        ax.set_ylabel(str(sid), rotation=0, labelpad=40, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[-1].set_xlabel("Hour (UTC)")
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-storm hourly point counts and heatmap.")
    ap.add_argument("--matches", required=True, help="CSV/Parquet with storm_id + time columns.")
    ap.add_argument("--out-csv", required=True, help="Output CSV for hourly counts.")
    ap.add_argument("--out-png", default=None, help="Optional heatmap PNG output.")
    ap.add_argument("--storm-col", default=None, help="Storm id column override.")
    ap.add_argument("--time-col", default=None, help="Time column override.")
    ap.add_argument("--top-storms", type=int, default=20, help="Max storms to include in heatmap.")
    ap.add_argument("--min-total", type=int, default=1, help="Min total points to include in heatmap.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--vmax", type=float, default=None, help="Optional heatmap vmax.")
    ap.add_argument("--plot-kind", choices=["heatmap", "timeline"], default="heatmap", help="Plot style for PNG output.")
    args = ap.parse_args()

    df = _read_any(args.matches)
    if df.empty:
        raise SystemExit("[storm-hourly] matches file is empty.")

    storm_col = args.storm_col or _pick_col(df, ["storm_id", "sid", "name", "storm_name"])
    time_col = args.time_col or _pick_col(df, ["time_h", "object_time", "track_time", "time", "seed_time"])
    if not storm_col or not time_col:
        raise SystemExit("[storm-hourly] missing storm_id or time column in matches.")

    tmp = pd.DataFrame(
        {
            "storm_id": df[storm_col].astype(str),
            "time_h": _to_time_h(df[time_col]),
        }
    ).dropna(subset=["storm_id", "time_h"])

    if tmp.empty:
        raise SystemExit("[storm-hourly] no valid storm/time rows after parsing.")

    counts = tmp.groupby(["storm_id", "time_h"]).size().rename("points").reset_index()
    counts = counts.sort_values(["storm_id", "time_h"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_csv, index=False)
    print(f"[storm-hourly] wrote {len(counts):,} rows -> {out_csv}")

    if args.out_png:
        title = args.title or "Storm hourly point counts"
        if args.plot_kind == "timeline":
            _timeline(counts, Path(args.out_png), title, args.top_storms, args.min_total)
            print(f"[storm-hourly] wrote timeline -> {args.out_png}")
        else:
            totals = counts.groupby("storm_id")["points"].sum().sort_values(ascending=False)
            keep = totals.loc[totals >= int(args.min_total)].head(int(args.top_storms)).index
            subset = counts.loc[counts["storm_id"].isin(keep)]
            if subset.empty:
                print("[storm-hourly] no storms meet min-total for heatmap; skipping plot.")
                return 0
            pivot = subset.pivot_table(index="storm_id", columns="time_h", values="points", fill_value=0)
            # order by total points desc
            order = pivot.sum(axis=1).sort_values(ascending=False).index
            pivot = pivot.loc[order]
            _heatmap(pivot, Path(args.out_png), title, args.vmax)
            print(f"[storm-hourly] wrote heatmap -> {args.out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
