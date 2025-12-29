#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transition_sanity_report.py

Agent: lightweight diagnostics for state transition features.
Outputs storm rates by transition class, pulse counts, and mutual info rankings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

pd.options.mode.copy_on_write = True

try:
    import pyarrow.dataset as ds  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ds = None


def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _sample_table(path: str, columns: List[str], n: int) -> pd.DataFrame:
    if _is_parquet(path) and ds is not None:
        scanner = ds.dataset(path).scanner(columns=columns if columns else None, limit=n)
        tbl = scanner.to_table()
        return tbl.to_pandas()
    df = pd.read_parquet(path, columns=columns if columns else None) if _is_parquet(path) else pd.read_csv(path, usecols=columns or None, low_memory=False)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df


def _storm_rate_by_class(df: pd.DataFrame, label: str) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    if "transition_class" not in df.columns or label not in df.columns:
        return rates
    df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    for cls, g in df.groupby("transition_class"):
        if pd.isna(cls):
            continue
        rates[f"class_{int(cls)}_storm_rate"] = float(g[label].mean())
    return rates


def _pulse_rates(df: pd.DataFrame, label: str, col: str, thr: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if col not in df.columns or label not in df.columns:
        return out
    df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    high = df[col] >= thr
    if high.any():
        out[f"{col}_high_storm_rate"] = float(df.loc[high, label].mean())
    low = ~high
    if low.any():
        out[f"{col}_low_storm_rate"] = float(df.loc[low, label].mean())
    return out


def _mutual_info(df: pd.DataFrame, label: str, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols or label not in df.columns:
        return pd.DataFrame(columns=["feature", "mi"])
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    mi = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    return pd.DataFrame({"feature": cols, "mi": mi}).sort_values("mi", ascending=False)


def parse_args():
    ap = argparse.ArgumentParser(description="Sanity stats for transition features.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--table", required=True, help="Input table with transitions + labels.")
    ap.add_argument("--label", default="storm", help="Label column.")
    ap.add_argument("--sample-rows", type=int, default=500_000, help="Rows to sample for diagnostics.")
    ap.add_argument("--out-md", default=None, help="Optional markdown report path.")
    return ap.parse_args()


def main():
    args = parse_args()
    cols = ["transition_class", "G", "E", "dG_1h", "dE_1h", "G_pulse_count_96h", "E_pulse_count_96h", "G_persist_24h", args.label, "SFI", "S3"]
    df = _sample_table(args.table, cols, args.sample_rows)
    if df.empty:
        raise SystemExit("No data available for sanity report.")

    lines: List[str] = []
    rates = _storm_rate_by_class(df, args.label)
    lines.append("## Storm rate by transition class")
    if rates:
        for k, v in rates.items():
            lines.append(f"- {k}: {v:.4f}")
    else:
        lines.append("- transition_class or label missing")

    lines.append("\n## Pulse thresholds")
    pulse_stats = {}
    pulse_stats.update(_pulse_rates(df, args.label, "G_pulse_count_96h", thr=1))
    pulse_stats.update(_pulse_rates(df, args.label, "E_pulse_count_96h", thr=1))
    if pulse_stats:
        for k, v in pulse_stats.items():
            lines.append(f"- {k}: {v:.4f}")
    else:
        lines.append("- pulse columns missing")

    lines.append("\n## Mutual information (top)")
    mi_df = _mutual_info(df, args.label, ["G", "E", "dG_1h", "dE_1h", "G_persist_24h", "SFI", "S3"])
    if not mi_df.empty:
        for _, row in mi_df.head(10).iterrows():
            lines.append(f"- {row['feature']}: {row['mi']:.4f}")
    else:
        lines.append("- insufficient columns for MI")

    report = "\n".join(lines)
    print(report)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(report, encoding="utf-8")
        print(f"[sanity] report -> {args.out_md}")


if __name__ == "__main__":
    main()
