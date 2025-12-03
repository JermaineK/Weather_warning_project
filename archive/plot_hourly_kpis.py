#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hourly_kpis.py — v2
Turn an hourly KPI CSV into plots + a Markdown summary you can paste into a report.

Inputs
------
--csv <path>                 Hourly KPIs (from hourly_metrics.py or similar)
--out-dir <dir>              Where to write plots + summaries (default: results/plots)
--title <str>                Optional title prefix for figures

Outputs
-------
<out-dir>/
  kpi_coverage.png
  kpi_active.png
  kpi_clusters.png              (if n_clusters present)
  kpi_summary.md                (Markdown snippet for your report)
  kpi_summary.csv               (one row per tag with headline stats)

Notes
-----
- Tolerant to column name quirks: uses _hour|hour|time for time; _tag optional.
- If multiple tags exist, plots one line per tag and summarizes per-tag.
- Trend = OLS slope of coverage vs time (per day units) *in %-points/day*.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _pick_time_col(df: pd.DataFrame) -> str:
    for c in ["_hour", "hour", "time"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find a time-like column (_hour|hour|time).")

def _ensure_datetime(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    return df

def _plot_line(df: pd.DataFrame, tcol: str, ycol: str, ylabel: str, out_path: Path, title: str, tags):
    plt.figure(figsize=(11, 4))
    for tag in tags:
        dd = df if tags == ["all"] else df[df["_tag"] == tag]
        if dd.empty or ycol not in dd.columns:
            continue
        plt.plot(dd[tcol], dd[ycol], label=(None if tags == ["all"] else tag))
    plt.title(f"{title} — {ylabel}" if title else ylabel)
    plt.xlabel("Time (UTC)")
    plt.ylabel(ylabel)
    if tags != ["all"]:
        plt.legend(frameon=False)
    plt.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def _per_tag_summary(df: pd.DataFrame, tcol: str, tag: str | None):
    dd = df if tag is None else df[df["_tag"] == tag]
    if dd.empty:
        return None
    # coverage stats
    cov = dd["coverage"].astype(float) if "coverage" in dd.columns else pd.Series(dtype=float)
    cov_mean = float(np.nanmean(cov)) if len(cov) else np.nan
    cov_med  = float(np.nanmedian(cov)) if len(cov) else np.nan
    cov_p95  = float(np.nanpercentile(cov, 95)) if len(cov) else np.nan

    # active max
    act_max = int(np.nanmax(dd["active"])) if "active" in dd.columns and len(dd["active"]) else np.nan

    # simple linear trend (%-points/day) on coverage
    trend_ppd = np.nan
    if len(cov) >= 3 and np.isfinite(cov).any():
        t0 = dd[tcol].min()
        days = (dd[tcol] - t0).dt.total_seconds() / 86400.0
        # robust to NaNs
        m = np.isfinite(days) & np.isfinite(cov)
        if m.sum() >= 3:
            X = np.c_[np.ones(m.sum()), days[m]]
            y = cov[m]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            slope = float(beta[1])  # fraction/day
            trend_ppd = slope * 100.0

    # spikes/dips: z-score on coverage (only if enough data)
    top_spike, top_dip = (pd.NaT, np.nan), (pd.NaT, np.nan)
    if len(cov) >= 8:
        z = (cov - cov.mean()) / (cov.std(ddof=1) + 1e-12)
        # highest positive z and most negative z
        i_spk = int(np.nanargmax(z))
        i_dip = int(np.nanargmin(z))
        top_spike = (dd[tcol].iloc[i_spk], float(z.iloc[i_spk]))
        top_dip   = (dd[tcol].iloc[i_dip], float(z.iloc[i_dip]))

    return {
        "tag": (tag if tag is not None else "all"),
        "rows": int(len(dd)),
        "hours": int(dd[tcol].nunique()),
        "mean_coverage": cov_mean,
        "median_coverage": cov_med,
        "p95_coverage": cov_p95,
        "max_active": act_max,
        "trend_cov_pp_per_day": trend_ppd,
        "top_spike_time": top_spike[0],
        "top_spike_z": top_spike[1],
        "top_dip_time": top_dip[0],
        "top_dip_z": top_dip[1],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="results/plots")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    tcol = _pick_time_col(df)
    df = _ensure_datetime(df, tcol).sort_values(tcol)

    # normalize tags presence
    if "_tag" not in df.columns:
        df["_tag"] = "all"
    tags = sorted(df["_tag"].dropna().unique().tolist()) or ["all"]

    # Sanity: ensure canonical cols exist even if missing
    for c in ["coverage", "active"]:
        if c not in df.columns:
            df[c] = np.nan

    title = args.title or Path(args.csv).stem

    # Plots
    _plot_line(df, tcol, "coverage", "Coverage (fraction active)", out_dir / "kpi_coverage.png", title, tags)
    _plot_line(df, tcol, "active", "Active cells (count)", out_dir / "kpi_active.png", title, tags)
    if "n_clusters" in df.columns:
        _plot_line(df, tcol, "n_clusters", "Cluster count", out_dir / "kpi_clusters.png", title, tags)

    # Per-tag summaries
    rows = []
    for tag in tags:
        s = _per_tag_summary(df, tcol, tag if tags != ["all"] else None)
        if s:
            rows.append(s)
    summ = pd.DataFrame(rows)

    # Write CSV summary
    summ_path = out_dir / "kpi_summary.csv"
    summ.to_csv(summ_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # Markdown snippet for report
    md_path = out_dir / "kpi_summary.md"
    lines = []
    lines.append(f"# Hourly KPI Summary\n")
    lines.append(f"_Source_: `{args.csv}`\n")
    if args.title:
        lines.append(f"**Title**: {args.title}\n")
    lines.append("## Headline metrics by tag\n")
    lines.append("| tag | hours | mean cov | median | p95 | max active | trend (pp/day) | spike (UTC, z) | dip (UTC, z) |")
    lines.append("|-----|-------|----------:|-------:|----:|-----------:|---------------:|----------------|--------------|")
    for _, r in summ.sort_values("tag").iterrows():
        def fmt_time(x): 
            return (pd.to_datetime(x).strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "—")
        lines.append(
            f"| {r['tag']} | {int(r['hours'])} | "
            f"{r['mean_coverage']:.3f} | {r['median_coverage']:.3f} | {r['p95_coverage']:.3f} | "
            f"{('' if pd.isna(r['max_active']) else int(r['max_active']))} | "
            f"{(np.nan if pd.isna(r['trend_cov_pp_per_day']) else r['trend_cov_pp_per_day']):.2f} | "
            f\"{fmt_time(r['top_spike_time'])}, {r['top_spike_z']:.2f}\" | "
            f\"{fmt_time(r['top_dip_time'])}, {r['top_dip_z']:.2f}\" |"
        )
    lines.append("\n### Figure set\n")
    lines.append(f"- Coverage: `kpi_coverage.png`")
    lines.append(f"- Active cells: `kpi_active.png`")
    if "n_clusters" in df.columns:
        lines.append(f"- Cluster count: `kpi_clusters.png`")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[write] {out_dir/'kpi_coverage.png'}")
    print(f"[write] {out_dir/'kpi_active.png'}")
    if "n_clusters" in df.columns:
        print(f"[write] {out_dir/'kpi_clusters.png'}")
    print(f"[write] {summ_path} (rows={len(summ)})")
    print(f"[write] {md_path}  ← drop this into your final report")

if __name__ == "__main__":
    main()