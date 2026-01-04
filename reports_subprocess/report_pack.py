#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_pack.py

Agent: build report-pack tables (health, feature stats, object skill, directionality, associations).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None  # type: ignore

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import config_normalize
from utils.run_naming import make_run_dir
from pipeline_contracts import contract_for, order_steps


def _table_kind(path: str | Path) -> Optional[str]:
    low = str(path).lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return "parquet"
    if low.endswith(".csv.gz"):
        return "csv.gz"
    if low.endswith(".csv"):
        return "csv"
    return None


def _is_parquet(path: str | Path) -> bool:
    return _table_kind(path) == "parquet"


def _read_any(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    kind = _table_kind(path)
    if kind == "parquet":
        return pd.read_parquet(path, columns=columns)
    if kind in {"csv", "csv.gz"}:
        return pd.read_csv(path, low_memory=False, usecols=columns, compression="infer")
    raise SystemExit(f"[report-pack] unsupported table format for {path}")


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data_"
    df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def _count_csv_rows(path: Path) -> int:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    try:
        row_idx = -1
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for row_idx, _ in enumerate(fh):
                pass
        return max(0, row_idx)
    except Exception as exc:
        raise SystemExit(f"[report-pack] failed to count CSV rows for {path}: {exc}")


def _scan_parquet_rows(pf: "pq.ParquetFile", batch_size: int = 200_000) -> int:
    total = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[]):
        total += len(batch)
    return total


def _parquet_row_counts(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if pq is None:
        return None, None
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"[report-pack] failed to open parquet file for {path}: {exc}")
    meta = pf.metadata
    meta_rows = int(meta.num_rows) if meta is not None else None
    counted_rows = None
    if meta is not None:
        try:
            counted_rows = int(sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups)))
        except Exception:
            counted_rows = None
    if not meta_rows:
        # Agent: fallback to a row-group scan when metadata is missing/zero.
        try:
            counted_rows = _scan_parquet_rows(pf)
        except Exception as exc:
            raise SystemExit(f"[report-pack] failed to scan parquet rows for {path}: {exc}")
    if counted_rows is None:
        counted_rows = meta_rows if meta_rows is not None else 0
    return meta_rows, counted_rows


def _row_counts(path: Path) -> Tuple[Optional[int], Optional[int]]:
    kind = _table_kind(path)
    if kind == "parquet":
        return _parquet_row_counts(path)
    if kind in {"csv", "csv.gz"}:
        return None, _count_csv_rows(path)
    try:
        return None, 1 if path.exists() and path.stat().st_size > 0 else 0
    except Exception:
        return None, 0


def _row_count(path: Path) -> int:
    _, counted = _row_counts(path)
    return int(counted or 0)


def _peek_columns(path: Path) -> List[str]:
    kind = _table_kind(path)
    if kind == "parquet" and pq is not None:
        return list(pq.ParquetFile(path).schema.names)
    if kind in {"csv", "csv.gz"}:
        try:
            return list(pd.read_csv(path, nrows=0, compression="infer").columns)
        except Exception as exc:
            raise SystemExit(f"[report-pack] failed to read CSV header for {path}: {exc}")
    return []


def _minmax_columns(path: Path, cols: List[str], max_rows: int = 200_000) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {c: (None, None) for c in cols}
    if not path.exists():
        return out
    kind = _table_kind(path)
    try:
        available = _peek_columns(path)
    except Exception:
        return out
    want = [c for c in cols if c in available]
    if not want:
        return out
    if kind == "parquet" and pq is not None:
        pf = pq.ParquetFile(path)
        seen = 0
        for batch in pf.iter_batches(columns=want, batch_size=max_rows):
            df = batch.to_pandas()
            for c in want:
                if c not in df.columns:
                    continue
                vals = pd.to_numeric(df[c], errors="coerce")
                if c == "time":
                    vals = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
                vmin = vals.min()
                vmax = vals.max()
                cur_min, cur_max = out.get(c, (None, None))
                out[c] = (
                    vmin if cur_min is None or (pd.notna(vmin) and vmin < cur_min) else cur_min,
                    vmax if cur_max is None or (pd.notna(vmax) and vmax > cur_max) else cur_max,
                )
            seen += len(df)
            if seen >= max_rows:
                break
        return out
    if kind not in {"csv", "csv.gz"}:
        return out
    try:
        df = pd.read_csv(path, usecols=want, nrows=max_rows, low_memory=False, compression="infer")
    except ValueError:
        return out
    for c in want:
        if c not in df.columns:
            continue
        if c == "time":
            vals = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
        else:
            vals = pd.to_numeric(df[c], errors="coerce")
        out[c] = (vals.min(), vals.max())
    return out


def _collect_output_specs(cfg: Dict[str, Any]) -> List[Tuple[str, str, Path, List[str]]]:
    outputs: List[Tuple[str, str, Path, List[str]]] = []
    for section in cfg:
        if not isinstance(cfg.get(section), dict):
            continue
        sec = cfg.get(section, {})
        if not sec.get("enabled"):
            continue
        steps = sec.get("steps") or []
        if section == "seeds":
            steps = _seed_step_dicts(sec)
        if section == "report" and not steps:
            legacy = {k: v for k, v in sec.items() if k not in ("enabled", "steps")}
            legacy.setdefault("mode", "summary")
            steps = [legacy]
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict) or step.get("enabled") is False:
                continue
            mode = str(step.get("mode", ""))
            if section == "seeds":
                seed_specs = _seed_output_specs(step)
                if seed_specs:
                    outputs.extend(seed_specs)
                    continue
            contract = contract_for(section, mode)
            if not contract:
                continue
            expected_cols = contract.expected_output_columns(step)
            for spec in contract.outputs:
                path, _ = spec.resolve(step)
                if path:
                    outputs.append((section, mode, Path(path), expected_cols))
    return outputs


def _seed_step_dicts(sec: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_steps = [
        ("from-alerts", sec.get("from_alerts")),
        ("proto-outcomes", sec.get("outcomes")),
        ("gse-tracks", sec.get("gse_tracks")),
        ("starts-vs-tracks", sec.get("starts")),
        ("analyze", sec.get("analyze")),
    ]
    steps: List[Dict[str, Any]] = []
    for name, cfg in raw_steps:
        if cfg is None:
            continue
        step = {"mode": name, **cfg}
        steps.append(step)
    return steps


def _seed_output_specs(step: Dict[str, Any]) -> List[Tuple[str, str, Path, List[str]]]:
    mode = str(step.get("mode", ""))
    out_dir = step.get("out_dir") or step.get("out-dir")
    run_name = step.get("run_name") or step.get("run-name")
    if not out_dir or not run_name:
        return []
    base = Path(out_dir) / str(run_name)
    outputs: List[Tuple[str, str, Path, List[str]]] = []

    if mode == "from-alerts":
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_union_byhour.csv"), ["time", "lat", "lon"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_starts_byhour.csv"), ["lat", "lon", "time_start"]))
        return outputs

    if mode == "starts-vs-tracks":
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_seed_starts_points.csv"), ["time_h", "lat", "lon"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_seed_patches.csv"), ["time_h", "patch_id", "lat_cen", "lon_cen"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_seed_track_matches.csv"), ["time_h", "storm_id", "d_km_min"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_seed_summary.txt"), []))
        return outputs

    if mode == "proto-outcomes":
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_track_points.csv"), ["time", "lat", "lon", "track_id"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_tracks.csv"), ["track_id", "start_time", "end_time"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_conversion_rates.csv"), ["horizon_h", "conv_to_TS_rate"]))
        return outputs

    if mode == "gse-tracks":
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_gse_track_points.csv"), ["time", "lat", "lon", "track_id"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_gse_track_summaries.csv"), ["track_id", "start_time", "end_time"]))
        outputs.append(("seeds", mode, base.with_name(f"{base.name}_gse_state_transitions.csv"), ["from_state", "to_state", "count"]))
        return outputs

    return outputs


def _find_step_path(cfg: Dict[str, Any], section: str, mode: str, keys: Sequence[str]) -> Optional[str]:
    sec = cfg.get(section, {}) if isinstance(cfg.get(section), dict) else {}
    steps = sec.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("mode", "")).strip() != mode:
            continue
        for k in keys:
            if k in step and step.get(k):
                return str(step.get(k))
    return None


def _infer_features_path(cfg: Dict[str, Any]) -> Optional[str]:
    if not cfg:
        return None
    for section, mode, keys in [
        ("data_stage", "state-transitions", ("outfile", "out")),
        ("data_stage", "add-ids", ("outfile", "out")),
        ("features", "join-labels-grid", ("out", "outfile")),
        ("features", "gka-ms", ("outfile", "out")),
    ]:
        path = _find_step_path(cfg, section, mode, keys)
        if path and Path(path).exists():
            return path
    return None


def _resolve_label_col(df: pd.DataFrame, label_col: str) -> str:
    if label_col != "auto":
        return label_col
    for cand in ["pregen", "storm", "near_storm", "y_viable"]:
        if cand in df.columns:
            vals = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            if vals.sum() > 0:
                return cand
    for cand in ["pregen", "storm", "near_storm", "y_viable"]:
        if cand in df.columns:
            return cand
    return "storm"


def _sample_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    hashed = hash_pandas_object(df, index=False, hash_key=str(seed))
    order = np.argsort(hashed.to_numpy())
    keep = order[: max_rows]
    return df.iloc[keep].copy()


def _feature_stats(df: pd.DataFrame, features: List[str], label_col: str) -> pd.DataFrame:
    rows = []
    for split_name, mask in [
        ("all", pd.Series(True, index=df.index)),
        ("positive", pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0),
        ("negative", pd.to_numeric(df[label_col], errors="coerce").fillna(0) == 0),
    ]:
        subset = df.loc[mask]
        for feat in features:
            if feat not in subset.columns:
                continue
            vals = pd.to_numeric(subset[feat], errors="coerce")
            miss = float(vals.isna().mean())
            v = vals.dropna()
            if v.empty:
                rows.append(
                    {
                        "feature": feat,
                        "split": split_name,
                        "mean": np.nan,
                        "std": np.nan,
                        "q05": np.nan,
                        "q50": np.nan,
                        "q95": np.nan,
                        "missing_frac": miss,
                        "n": int(len(subset)),
                    }
                )
                continue
            rows.append(
                {
                    "feature": feat,
                    "split": split_name,
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "q05": float(v.quantile(0.05)),
                    "q50": float(v.quantile(0.50)),
                    "q95": float(v.quantile(0.95)),
                    "missing_frac": miss,
                    "n": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def _association_scan(df: pd.DataFrame, features: List[str], label: str, subset_name: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).to_numpy()
    if y.size == 0:
        return pd.DataFrame()

    try:
        from sklearn.feature_selection import mutual_info_classif  # type: ignore
    except Exception:
        mutual_info_classif = None

    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce").fillna(0).to_numpy()
        if x.size == 0:
            continue
        # point-biserial via correlation
        if np.std(x) > 0:
            pb = float(np.corrcoef(x, y)[0, 1])
        else:
            pb = np.nan
        # Spearman
        try:
            sp = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        except Exception:
            sp = np.nan
        # MI
        mi = np.nan
        if mutual_info_classif is not None:
            try:
                mi = float(mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False, random_state=42)[0])
            except Exception:
                mi = np.nan
        rows.append(
            {
                "subset": subset_name,
                "feature": feat,
                "point_biserial": pb,
                "spearman": sp,
                "mutual_info": mi,
                "n": int(len(x)),
            }
        )
    return pd.DataFrame(rows)


def _parse_edges(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).replace(",", " ").split():
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    return vals


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n
    pearson = float(np.corrcoef(x, y)[0, 1])
    try:
        spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        spearman = float("nan")
    return pearson, spearman, n


def _effect_size_deciles(x: np.ndarray, y: np.ndarray, q: float = 0.1) -> Tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = np.nanquantile(x, q)
    hi = np.nanquantile(x, 1 - q)
    top = y[x >= hi]
    bot = y[x <= lo]
    top_mean = float(np.nanmean(top)) if top.size else float("nan")
    bot_mean = float(np.nanmean(bot)) if bot.size else float("nan")
    return top_mean, bot_mean, float(top_mean - bot_mean) if np.isfinite(top_mean) and np.isfinite(bot_mean) else float("nan")


def _filter_corr_features(
    features: List[str],
    label_col: str,
    alert_col: str,
    near_col: str,
    lead_col: str,
) -> List[str]:
    exclude = {
        label_col,
        alert_col,
        near_col,
        lead_col,
        "time",
        "lat",
        "lon",
        "ilat",
        "ilon",
        "storm",
        "storm_id",
        "storm_window",
        "storm_point",
    }
    exclude_prefixes = ("storm_", "label_", "target_")
    out = []
    for feat in features:
        if feat in exclude:
            continue
        if any(feat.startswith(pref) for pref in exclude_prefixes):
            continue
        out.append(feat)
    return out


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.feature_selection import mutual_info_classif  # type: ignore
    except Exception:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    try:
        return float(mutual_info_classif(x[mask].reshape(-1, 1), y[mask], discrete_features=False, random_state=42)[0])
    except Exception:
        return float("nan")


def _correlation_sweep(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    lead_col: str,
    lead_bins: List[float],
    lat_bands: List[float],
    max_missing: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty or target not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy()
    overall_rows: List[Dict[str, object]] = []
    by_lead_rows: List[Dict[str, object]] = []
    by_lat_rows: List[Dict[str, object]] = []

    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        missing_frac = float(np.mean(~np.isfinite(x)))
        row: Dict[str, object] = {
            "feature": feat,
            "missing_frac": missing_frac,
            "n_total": int(len(x)),
        }
        if missing_frac > max_missing:
            row["drop_reason"] = f"missing>{max_missing}"
            overall_rows.append(row)
            continue
        pearson, spearman, n = _safe_corr(x, y)
        top_mean, bot_mean, effect = _effect_size_deciles(x, y)
        row.update(
            {
                "pearson": pearson,
                "spearman": spearman,
                "mutual_info": _mutual_info(x, y),
                "n_used": n,
                "top_decile_rate": top_mean,
                "bottom_decile_rate": bot_mean,
                "effect_size": effect,
            }
        )
        overall_rows.append(row)

    overall = pd.DataFrame(overall_rows)

    if lead_col in df.columns and lead_bins and len(lead_bins) >= 2:
        lead_vals = pd.to_numeric(df[lead_col], errors="coerce")
        labels = [f"{lead_bins[i]}-{lead_bins[i+1]}" for i in range(len(lead_bins) - 1)]
        lead_bin = pd.cut(lead_vals, bins=lead_bins, labels=labels, include_lowest=False)
        df_lead = df.copy()
        df_lead["_lead_bin"] = lead_bin
        for bin_label, sub in df_lead.groupby("_lead_bin"):
            if bin_label is None or sub.empty:
                continue
            y_sub = pd.to_numeric(sub[target], errors="coerce").fillna(0).to_numpy()
            for feat in features:
                if feat not in sub.columns:
                    continue
                x = pd.to_numeric(sub[feat], errors="coerce").to_numpy()
                missing_frac = float(np.mean(~np.isfinite(x)))
                if missing_frac > max_missing:
                    by_lead_rows.append(
                        {
                            "lead_bin": str(bin_label),
                            "feature": feat,
                            "missing_frac": missing_frac,
                            "pearson": float("nan"),
                            "spearman": float("nan"),
                            "mutual_info": float("nan"),
                            "n_used": 0,
                        }
                    )
                    continue
                pearson, spearman, n = _safe_corr(x, y_sub)
                by_lead_rows.append(
                    {
                        "lead_bin": str(bin_label),
                        "feature": feat,
                        "missing_frac": missing_frac,
                        "pearson": pearson,
                        "spearman": spearman,
                        "mutual_info": _mutual_info(x, y_sub),
                        "n_used": n,
                    }
                )

    if "lat" in df.columns and lat_bands and len(lat_bands) >= 2:
        lat_vals = pd.to_numeric(df["lat"], errors="coerce")
        labels = [f"{lat_bands[i]}..{lat_bands[i+1]}" for i in range(len(lat_bands) - 1)]
        lat_bin = pd.cut(lat_vals, bins=lat_bands, labels=labels, include_lowest=True)
        df_lat = df.copy()
        df_lat["_lat_band"] = lat_bin
        for band, sub in df_lat.groupby("_lat_band"):
            if band is None or sub.empty:
                continue
            y_sub = pd.to_numeric(sub[target], errors="coerce").fillna(0).to_numpy()
            for feat in features:
                if feat not in sub.columns:
                    continue
                x = pd.to_numeric(sub[feat], errors="coerce").to_numpy()
                missing_frac = float(np.mean(~np.isfinite(x)))
                if missing_frac > max_missing:
                    by_lat_rows.append(
                        {
                            "lat_band": str(band),
                            "feature": feat,
                            "missing_frac": missing_frac,
                            "pearson": float("nan"),
                            "spearman": float("nan"),
                            "mutual_info": float("nan"),
                            "n_used": 0,
                        }
                    )
                    continue
                pearson, spearman, n = _safe_corr(x, y_sub)
                by_lat_rows.append(
                    {
                        "lat_band": str(band),
                        "feature": feat,
                        "missing_frac": missing_frac,
                        "pearson": pearson,
                        "spearman": spearman,
                        "mutual_info": _mutual_info(x, y_sub),
                        "n_used": n,
                    }
                )

    by_lead = pd.DataFrame(by_lead_rows)
    if not by_lead.empty:
        stab = (
            by_lead.groupby("feature")["spearman"]
            .agg(["std", "min", "max"])
            .rename(columns={"std": "spearman_lead_std", "min": "spearman_lead_min", "max": "spearman_lead_max"})
            .reset_index()
        )
        overall = overall.merge(stab, on="feature", how="left")

    by_lat = pd.DataFrame(by_lat_rows)
    return overall, by_lead, by_lat


def _correlation_by_month(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    max_missing: float,
) -> pd.DataFrame:
    if df.empty or target not in df.columns or "time" not in df.columns:
        return pd.DataFrame()
    t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    month = t.dt.to_period("M")
    df_mon = df.copy()
    df_mon["_month"] = month
    rows: List[Dict[str, object]] = []
    for mon, sub in df_mon.groupby("_month"):
        if mon is None or sub.empty:
            continue
        y_sub = pd.to_numeric(sub[target], errors="coerce").fillna(0).to_numpy()
        for feat in features:
            if feat not in sub.columns:
                continue
            x = pd.to_numeric(sub[feat], errors="coerce").to_numpy()
            missing_frac = float(np.mean(~np.isfinite(x)))
            if missing_frac > max_missing:
                rows.append(
                    {
                        "month": str(mon),
                        "feature": feat,
                        "missing_frac": missing_frac,
                        "pearson": float("nan"),
                        "spearman": float("nan"),
                        "mutual_info": float("nan"),
                        "n_used": 0,
                    }
                )
                continue
            pearson, spearman, n = _safe_corr(x, y_sub)
            rows.append(
                {
                    "month": str(mon),
                    "feature": feat,
                    "missing_frac": missing_frac,
                    "pearson": pearson,
                    "spearman": spearman,
                    "mutual_info": _mutual_info(x, y_sub),
                    "n_used": n,
                }
            )
    return pd.DataFrame(rows)


def _residualize(vec: np.ndarray, design: np.ndarray) -> np.ndarray:
    try:
        beta = np.linalg.lstsq(design, vec, rcond=None)[0]
        return vec - design @ beta
    except Exception:
        return vec


def _correlation_residualized(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    max_missing: float,
) -> pd.DataFrame:
    if df.empty or target not in df.columns:
        return pd.DataFrame()
    if not {"lat", "lon", "time"}.issubset(df.columns):
        return pd.DataFrame()
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    month = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.month
    month_dummies = pd.get_dummies(month, prefix="mon", dummy_na=False)
    rows: List[Dict[str, object]] = []
    base = pd.concat([lat, lon, month_dummies], axis=1)
    base_cols = base.columns.tolist()
    base = base.to_numpy(dtype=float)
    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy(dtype=float)
        missing_frac = float(np.mean(~np.isfinite(x)))
        row: Dict[str, object] = {
            "feature": feat,
            "missing_frac": missing_frac,
            "n_total": int(len(x)),
        }
        if missing_frac > max_missing:
            row["drop_reason"] = f"missing>{max_missing}"
            rows.append(row)
            continue
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(base).all(axis=1)
        n = int(mask.sum())
        if n < 3:
            row.update({"pearson": float("nan"), "spearman": float("nan"), "n_used": n})
            rows.append(row)
            continue
        design = base[mask]
        # Agent: include intercept plus lat/lon/month controls.
        design = np.column_stack([np.ones(design.shape[0]), design])
        x_res = _residualize(x[mask], design)
        y_res = _residualize(y[mask], design)
        pearson, spearman, n_used = _safe_corr(x_res, y_res)
        row.update(
            {
                "pearson": pearson,
                "spearman": spearman,
                "n_used": n_used,
                "controls": "lat,lon," + ",".join(str(c) for c in base_cols if str(c).startswith("mon_")),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _permute_within_strata(y: np.ndarray, strata: pd.Series, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    for key, idx in strata.groupby(strata).groups.items():
        if idx is None:
            continue
        idx_arr = np.asarray(idx, dtype=int)
        if idx_arr.size < 2:
            continue
        y_perm[idx_arr] = rng.permutation(y_perm[idx_arr])
    return y_perm


def _correlation_permtest(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    lat_bands: List[float],
    max_missing: float,
) -> pd.DataFrame:
    if df.empty or target not in df.columns or not lat_bands or len(lat_bands) < 2:
        return pd.DataFrame()
    if not {"lat", "time"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    lat_vals = pd.to_numeric(df["lat"], errors="coerce")
    labels = [f"{lat_bands[i]}..{lat_bands[i+1]}" for i in range(len(lat_bands) - 1)]
    lat_bin = pd.cut(lat_vals, bins=lat_bands, labels=labels, include_lowest=True)
    month = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.to_period("M")
    strata = pd.Series(lat_bin.astype(str)) + "|" + pd.Series(month.astype(str))
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy(dtype=float)
    y_perm = _permute_within_strata(y, strata, seed=42)
    rows: List[Dict[str, object]] = []
    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        missing_frac = float(np.mean(~np.isfinite(x)))
        if missing_frac > max_missing:
            rows.append(
                {
                    "feature": feat,
                    "missing_frac": missing_frac,
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                    "n_used": 0,
                }
            )
            continue
        pearson, spearman, n_used = _safe_corr(x, y_perm)
        rows.append(
            {
                "feature": feat,
                "missing_frac": missing_frac,
                "pearson": pearson,
                "spearman": spearman,
                "n_used": n_used,
            }
        )
    return pd.DataFrame(rows)


def _topk_by_metric(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    for metric in ["point_biserial", "spearman", "mutual_info"]:
        if metric not in df.columns:
            continue
        ranked = (
            df.copy()
            .assign(_abs=lambda d: d[metric].abs())
            .sort_values("_abs", ascending=False)
            .head(k)
            .drop(columns=["_abs"])
        )
        ranked["metric"] = metric
        out.append(ranked)
    return pd.concat(out, ignore_index=True) if out else df


def _directionality(matches: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or tracks.empty or "vmax" not in tracks.columns:
        return pd.DataFrame()
    if "match_is_primary" in matches.columns:
        matches = matches.loc[pd.to_numeric(matches["match_is_primary"], errors="coerce").fillna(0).astype(int) > 0]
    elif "match_rank" in matches.columns:
        matches = matches.loc[pd.to_numeric(matches["match_rank"], errors="coerce").fillna(0) <= 1]
    tracks = tracks.sort_values(["storm_id", "time"])
    tracks["vmax_next"] = tracks.groupby("storm_id")["vmax"].shift(-1)
    tracks["dvmax"] = tracks["vmax_next"] - tracks["vmax"]
    m = matches.merge(
        tracks[["storm_id", "time", "dvmax"]],
        left_on=["storm_id", "track_time"],
        right_on=["storm_id", "time"],
        how="left",
    )
    out = []
    for col in ["obj_axis_align_motion", "obj_corepull_align_motion"]:
        if col not in m.columns:
            continue
        vals = pd.to_numeric(m[col], errors="coerce")
        dv = pd.to_numeric(m["dvmax"], errors="coerce")
        ok = vals.notna() & dv.notna()
        if ok.any():
            corr = float(pd.Series(vals[ok]).corr(pd.Series(dv[ok]), method="spearman"))
            out.append({"metric": col, "spearman_dvmax": corr, "n": int(ok.sum())})
    if "obj_motion_bearing_deg" in m.columns and "obj_flow_bearing_deg" in m.columns:
        a = pd.to_numeric(m["obj_motion_bearing_deg"], errors="coerce").to_numpy()
        b = pd.to_numeric(m["obj_flow_bearing_deg"], errors="coerce").to_numpy()
        diff = ((a - b + 180.0) % 360.0) - 180.0
        diff = diff[np.isfinite(diff)]
        if diff.size:
            out.append(
                {
                    "metric": "motion_vs_flow",
                    "median_abs_deg": float(np.nanmedian(np.abs(diff))),
                    "p90_abs_deg": float(np.nanquantile(np.abs(diff), 0.9)),
                    "n": int(diff.size),
                }
            )
    return pd.DataFrame(out)


def _aoi_area_km2(area: Optional[str]) -> Optional[float]:
    if not area:
        return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in str(area).split(",")]
    except Exception:
        return None
    lat_mid = (latN + latS) / 2.0
    height_km = abs(latN - latS) * 111.32
    width_km = abs(lonE - lonW) * 111.32 * math.cos(math.radians(lat_mid))
    if not np.isfinite(height_km) or not np.isfinite(width_km) or height_km <= 0 or width_km <= 0:
        return None
    return float(height_km * width_km)


def _nearest_neighbor_km(lat: np.ndarray, lon: np.ndarray) -> Tuple[float, float]:
    n = len(lat)
    if n < 2:
        return float("nan"), float("nan")
    dmins = []
    for i in range(n):
        d = []
        for j in range(n):
            if i == j:
                continue
            d.append(_haversine_km(lat[i], lon[i], lat[j], lon[j]))
        dmins.append(min(d) if d else float("nan"))
    arr = np.asarray(dmins, dtype=float)
    return float(np.nanmean(arr)), float(np.nanmedian(arr))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(a, 0.0)))


def _object_fragmentation(objects: pd.DataFrame, area_km2: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if objects.empty or "obj_area_cells" not in objects.columns:
        return pd.DataFrame(), pd.DataFrame()
    areas = pd.to_numeric(objects["obj_area_cells"], errors="coerce")
    bins = [
        ("1", areas == 1),
        ("2-4", (areas >= 2) & (areas <= 4)),
        ("5-20", (areas >= 5) & (areas <= 20)),
        (">20", areas > 20),
    ]
    total = int(areas.notna().sum())
    frag_rows = []
    for label, mask in bins:
        count = int(mask.sum()) if total else 0
        frac = float(count / total) if total else float("nan")
        frag_rows.append({"bin": label, "count": count, "frac": frac})
    frag_df = pd.DataFrame(frag_rows)

    df = objects.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time", "obj_centroid_lat", "obj_centroid_lon"])
    df["hour"] = df["time"].dt.floor("h")
    rows = []
    for hr, g in df.groupby("hour", sort=True):
        lat = pd.to_numeric(g["obj_centroid_lat"], errors="coerce").to_numpy()
        lon = pd.to_numeric(g["obj_centroid_lon"], errors="coerce").to_numpy()
        mean_nn, median_nn = _nearest_neighbor_km(lat, lon)
        obj_count = int(len(g))
        area_max = float(pd.to_numeric(g["obj_area_cells"], errors="coerce").max()) if obj_count else float("nan")
        area_med = float(pd.to_numeric(g["obj_area_cells"], errors="coerce").median()) if obj_count else float("nan")
        density = float("nan")
        if area_km2 and area_km2 > 0:
            density = obj_count / (area_km2 / 10_000.0)
        rows.append(
            {
                "hour": hr,
                "object_count": obj_count,
                "area_max_cells": area_max,
                "area_median_cells": area_med,
                "nn_mean_km": mean_nn,
                "nn_median_km": median_nn,
                "density_per_10k_km2": density,
            }
        )
    hourly_df = pd.DataFrame(rows)
    return frag_df, hourly_df


def _safe_storm_id(val: object) -> str:
    txt = str(val).strip() or "storm"
    txt = re.sub(r"[^A-Za-z0-9_.-]+", "_", txt)
    return txt or "storm"


def _iso(ts_val: object) -> Optional[str]:
    if ts_val is None or (isinstance(ts_val, float) and np.isnan(ts_val)):
        return None
    try:
        return pd.Timestamp(ts_val).isoformat()
    except Exception:
        return None


def _fval(val: object) -> Optional[float]:
    try:
        v = float(val)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate report pack tables for a run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default=None, help="Output folder (default: results/reports/<run-name>).")
    ap.add_argument(
        "--run-root",
        default=None,
        help="Optional root for dated run folders (YYYYMMDD_runNNN).",
    )
    ap.add_argument(
        "--run-date",
        default=None,
        help="Optional date prefix for run folder (YYYYMMDD). Default: today (UTC).",
    )
    ap.add_argument("--config", default=None, help="Optional pipeline YAML for run-health checks.")
    ap.add_argument("--features", default=None, help="Labelled features table for stats/associations.")
    ap.add_argument("--label-col", default="auto")
    ap.add_argument("--alert-col", default="alert_final")
    ap.add_argument("--near-col", default="near_storm")
    ap.add_argument("--lead-col", default="t_to_storm_min_h")
    ap.add_argument("--lead-bins", default="0,24,48,72,120,240", help="Comma-separated lead-hour bin edges.")
    ap.add_argument("--lat-bands", default="-90,-60,-30,0,30,60,90", help="Comma-separated latitude band edges.")
    ap.add_argument("--corr-max-missing", type=float, default=0.5, help="Drop features above this missing fraction.")
    ap.add_argument("--objects", default="results/objects/objects_by_hour.parquet")
    ap.add_argument("--matches", default="results/matches/storm_object_matches.parquet")
    ap.add_argument("--tracks", default="data/tracks/tracks_subset.csv")
    ap.add_argument("--skill-by-lead", default=None)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assoc-topk", type=int, default=50)
    ap.add_argument("--health-scan-rows", type=int, default=200_000)
    ap.add_argument("--feature-cols", default=None, help="Comma-separated feature list override.")
    ap.add_argument("--chunk-rows", dest="chunk_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", dest="chunk_rows", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", dest="parquet_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    args = ap.parse_args()

    run_dir = None
    run_root = Path(args.run_root) if args.run_root else None
    out_dir_arg = Path(args.out_dir) if args.out_dir else None
    if run_root and (out_dir_arg is None or not out_dir_arg.is_absolute()):
        run_dir = make_run_dir(run_root, args.run_date)

    if out_dir_arg is not None:
        out_dir = out_dir_arg
        if run_dir and not out_dir.is_absolute():
            try:
                out_rel = out_dir.relative_to(run_root) if run_root else out_dir
            except Exception:
                out_rel = out_dir
            out_dir = run_dir / out_rel
    else:
        out_dir = run_dir or Path(f"results/reports/{args.run_name}")

    if run_dir:
        print(f"[report-pack] run folder -> {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- run health ---
    cfg_obj = {}
    if args.config:
        cfg = Path(args.config).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
            cfg_obj = yaml.safe_load(cfg) or {}
        except Exception:
            cfg_obj = json.loads(cfg) if cfg.strip() else {}
        cfg_obj, _ = config_normalize.normalize_config(cfg_obj)
    outputs = _collect_output_specs(cfg_obj) if cfg_obj else []
    rows = []
    for section, mode, path, expected in outputs:
        exists = path.exists()
        cols = _peek_columns(path) if exists and path.is_file() else []
        missing = [c for c in expected if c not in cols]
        spans = _minmax_columns(path, ["time", "lat", "lon"], max_rows=args.health_scan_rows) if exists and path.is_file() else {}
        rows_meta, rows_counted = _row_counts(path) if exists and path.is_file() else (None, 0)
        rows_warn = False
        if rows_meta is not None and rows_counted is not None and rows_meta != rows_counted:
            rows_warn = True
        rows_value = int(rows_counted or 0)
        rows.append(
            {
                "section": section,
                "mode": mode,
                "path": str(path),
                "exists": bool(exists),
                "rows": rows_value,
                "rows_metadata": int(rows_meta) if rows_meta is not None else None,
                "rows_counted": int(rows_counted) if rows_counted is not None else None,
                "rows_warn": bool(rows_warn),
                "missing_cols": ",".join(missing) if missing else "",
                "time_min": spans.get("time", (None, None))[0],
                "time_max": spans.get("time", (None, None))[1],
                "lat_min": spans.get("lat", (None, None))[0],
                "lat_max": spans.get("lat", (None, None))[1],
                "lon_min": spans.get("lon", (None, None))[0],
                "lon_max": spans.get("lon", (None, None))[1],
                "normalize_lon": (cfg_obj.get("defaults", {}) or {}).get("normalize_lon"),
                "area": (cfg_obj.get("defaults", {}) or {}).get("area"),
            }
        )
    seen_paths = {r.get("path") for r in rows if r.get("path")}

    def _append_extra_row(section: str, mode: str, path: Optional[Path], expected: List[str]) -> None:
        if path is None:
            return
        if str(path) in seen_paths:
            return
        exists = path.exists()
        cols = _peek_columns(path) if exists and path.is_file() else []
        missing = [c for c in expected if c not in cols]
        spans = _minmax_columns(path, ["time", "lat", "lon"], max_rows=args.health_scan_rows) if exists and path.is_file() else {}
        rows_meta, rows_counted = _row_counts(path) if exists and path.is_file() else (None, 0)
        rows_warn = False
        if rows_meta is not None and rows_counted is not None and rows_meta != rows_counted:
            rows_warn = True
        rows_value = int(rows_counted or 0)
        rows.append(
            {
                "section": section,
                "mode": mode,
                "path": str(path),
                "exists": bool(exists),
                "rows": rows_value,
                "rows_metadata": int(rows_meta) if rows_meta is not None else None,
                "rows_counted": int(rows_counted) if rows_counted is not None else None,
                "rows_warn": bool(rows_warn),
                "missing_cols": ",".join(missing) if missing else "",
                "time_min": spans.get("time", (None, None))[0],
                "time_max": spans.get("time", (None, None))[1],
                "lat_min": spans.get("lat", (None, None))[0],
                "lat_max": spans.get("lat", (None, None))[1],
                "lon_min": spans.get("lon", (None, None))[0],
                "lon_max": spans.get("lon", (None, None))[1],
                "normalize_lon": (cfg_obj.get("defaults", {}) or {}).get("normalize_lon"),
                "area": (cfg_obj.get("defaults", {}) or {}).get("area"),
            }
        )
        seen_paths.add(str(path))

    obj_contract = contract_for("report", "objects-by-hour")
    match_contract = contract_for("report", "object-matches")
    _append_extra_row(
        "report",
        "bundle.objects-by-hour",
        Path(args.objects) if args.objects else None,
        obj_contract.expected_output_columns({}) if obj_contract else [],
    )
    _append_extra_row(
        "report",
        "bundle.object-matches",
        Path(args.matches) if args.matches else None,
        match_contract.expected_output_columns({}) if match_contract else [],
    )
    run_health = pd.DataFrame(rows)
    run_health.to_parquet(out_dir / "run_health.parquet", index=False)

    # --- feature stats + associations + correlations ---
    feature_stats = None
    assoc_rows = []
    features_path = args.features
    if not features_path:
        inferred = _infer_features_path(cfg_obj)
        if inferred:
            features_path = inferred
    if features_path and Path(features_path).exists():
        features = _read_any(features_path)
        features = _sample_df(features, args.max_rows, args.seed)
        label_col = _resolve_label_col(features, args.label_col)
        if args.feature_cols:
            feats = [f.strip() for f in args.feature_cols.split(",") if f.strip()]
        else:
            exclude = {label_col, args.alert_col, args.near_col, args.lead_col}
            feats = [c for c in features.columns if c not in exclude]
        corr_feats = _filter_corr_features(feats, label_col, args.alert_col, args.near_col, args.lead_col)
        if label_col in features.columns:
            feature_stats = _feature_stats(features, feats, label_col)
            feature_stats.to_parquet(out_dir / "feature_stats.parquet", index=False)

            assoc_all = _association_scan(features, feats, label_col, "all")
            assoc_rows.append(assoc_all)

            if args.alert_col in features.columns:
                alerts = features.loc[pd.to_numeric(features[args.alert_col], errors="coerce").fillna(0) > 0]
                assoc_rows.append(_association_scan(alerts, feats, label_col, "alerts"))
            if args.near_col in features.columns:
                near = features.loc[pd.to_numeric(features[args.near_col], errors="coerce").fillna(0) > 0]
                assoc_rows.append(_association_scan(near, feats, label_col, "near_storm"))

            assoc = pd.concat([a for a in assoc_rows if a is not None and not a.empty], ignore_index=True)
            if not assoc.empty:
                assoc_out = []
                for subset in assoc["subset"].unique().tolist():
                    sub = assoc.loc[assoc["subset"] == subset]
                    assoc_out.append(_topk_by_metric(sub, int(args.assoc_topk)))
                assoc = pd.concat(assoc_out, ignore_index=True) if assoc_out else assoc
                assoc.to_parquet(out_dir / "feature_associations.parquet", index=False)

            lead_bins = sorted(dict.fromkeys(_parse_edges(args.lead_bins)))
            lat_bands = sorted(dict.fromkeys(_parse_edges(args.lat_bands)))
            corr_overall, corr_by_lead, corr_by_lat = _correlation_sweep(
                features,
                corr_feats,
                label_col,
                args.lead_col,
                lead_bins,
                lat_bands,
                float(args.corr_max_missing),
            )
            corr_by_month = _correlation_by_month(
                features,
                corr_feats,
                label_col,
                float(args.corr_max_missing),
            )
            corr_resid = _correlation_residualized(
                features,
                corr_feats,
                label_col,
                float(args.corr_max_missing),
            )
            corr_perm = _correlation_permtest(
                features,
                corr_feats,
                label_col,
                lat_bands,
                float(args.corr_max_missing),
            )
            if not corr_overall.empty:
                corr_overall.to_csv(out_dir / "correlations_overall.csv", index=False)
                top = corr_overall.copy()
                if "spearman" in top.columns:
                    top = top.assign(_abs=top["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
                (out_dir / "correlations_overall.md").write_text(_markdown_table(top, max_rows=25) + "\n", encoding="utf-8")
            if not corr_by_lead.empty:
                corr_by_lead.to_csv(out_dir / "correlations_by_lead.csv", index=False)
            if not corr_by_lat.empty:
                corr_by_lat.to_csv(out_dir / "correlations_by_lat.csv", index=False)
            if not corr_by_month.empty:
                corr_by_month.to_csv(out_dir / "correlations_by_month.csv", index=False)
            if not corr_resid.empty:
                corr_resid.to_csv(out_dir / "correlations_residualized.csv", index=False)
            if not corr_perm.empty:
                corr_perm.to_csv(out_dir / "correlations_permtest.csv", index=False)

    # --- skill-by-lead ---
    skill_path = args.skill_by_lead
    if not skill_path:
        for ext in ("parquet", "csv"):
            candidate = Path(f"results/metrics/{args.run_name}_viability_leads.{ext}")
            if candidate.exists():
                skill_path = str(candidate)
                break
    if skill_path and Path(skill_path).exists():
        skill = _read_any(skill_path)
        skill.to_parquet(out_dir / "skill_by_lead.parquet", index=False)
        skill.to_csv(out_dir / "skill_by_lead.csv", index=False)

    # --- object fragmentation + hourly stats ---
    if Path(args.objects).exists():
        objs = _read_any(args.objects)
        area_km2 = _aoi_area_km2((cfg_obj.get("defaults", {}) or {}).get("area"))
        frag_df, hourly_df = _object_fragmentation(objs, area_km2)
        if not frag_df.empty:
            frag_df.to_parquet(out_dir / "object_fragmentation.parquet", index=False)
        if not hourly_df.empty:
            hourly_df.to_parquet(out_dir / "object_hourly_stats.parquet", index=False)
        rejects_path = Path(args.objects).with_name("objects_rejects_by_hour.parquet")
        if rejects_path.exists():
            rejects = _read_any(rejects_path)
            rejects.to_parquet(out_dir / "object_rejects_by_hour.parquet", index=False)

    # --- object skill + directionality ---
    if Path(args.matches).exists() and Path(args.tracks).exists():
        matches = _read_any(args.matches)
        if not matches.empty:
            matches["track_time"] = pd.to_datetime(matches["track_time"], utc=True, errors="coerce").dt.tz_localize(None)
            matches["object_time"] = pd.to_datetime(matches["object_time"], utc=True, errors="coerce").dt.tz_localize(None)
        tracks = _read_any(args.tracks)
        if not tracks.empty:
            if "storm_id" not in tracks.columns and "sid" in tracks.columns:
                tracks = tracks.rename(columns={"sid": "storm_id"})
            if "storm_id" not in tracks.columns:
                tracks["storm_id"] = "storm"
            if "time" not in tracks.columns and "obs_time" in tracks.columns:
                tracks = tracks.rename(columns={"obs_time": "time"})
            tracks["time"] = pd.to_datetime(tracks["time"], utc=True, errors="coerce").dt.tz_localize(None)
            tracks = tracks.dropna(subset=["time"]).reset_index(drop=True)

        if not tracks.empty and not matches.empty:
            track_counts = tracks.groupby("storm_id")["time"].nunique().rename("track_points")
            matched_counts = matches.groupby("storm_id")["track_time"].nunique().rename("matched_points")
            obj_skill = pd.concat([track_counts, matched_counts], axis=1).fillna(0).reset_index()
            obj_skill["match_rate"] = obj_skill["matched_points"] / obj_skill["track_points"].replace(0, np.nan)
            obj_skill.to_parquet(out_dir / "object_skill.parquet", index=False)

            dir_df = _directionality(matches, tracks)
            if not dir_df.empty:
                dir_df.to_parquet(out_dir / "directionality.parquet", index=False)

            storms_dir = out_dir / "storms"
            storms_dir.mkdir(parents=True, exist_ok=True)
            for storm_id, t in tracks.groupby("storm_id"):
                t = t.sort_values("time")
                m = matches.loc[matches["storm_id"] == storm_id].copy()
                genesis = t["time"].min()
                end_time = t["time"].max()
                duration_h = (end_time - genesis).total_seconds() / 3600.0 if pd.notna(genesis) and pd.notna(end_time) else None
                vmax_max = _fval(pd.to_numeric(t.get("vmax", pd.Series(dtype=float)), errors="coerce").max()) if "vmax" in t.columns else None

                series = []
                lead_hours = []
                first_match_time = None
                first_match_lead = None
                if not m.empty:
                    m = m.sort_values("match_score", ascending=False)
                    idx = m.groupby("track_time", sort=True)["match_score"].idxmax()
                    m_best = m.loc[idx].sort_values("track_time")
                    for _, row in m_best.iterrows():
                        tt = row.get("track_time")
                        ot = row.get("object_time")
                        lead = None
                        if pd.notna(tt) and pd.notna(ot):
                            lead = (tt - ot).total_seconds() / 3600.0
                            lead_hours.append(float(lead))
                        series.append(
                            {
                                "track_time": _iso(tt),
                                "object_time": _iso(ot),
                                "match_score": _fval(row.get("match_score")),
                                "obj_score": _fval(row.get("obj_score")),
                                "d_km": _fval(row.get("d_km")),
                                "dt_hours": _fval(row.get("dt_hours")),
                                "track_bearing_deg": _fval(row.get("track_bearing_deg")),
                                "obj_motion_bearing_deg": _fval(row.get("obj_motion_bearing_deg")),
                                "obj_axis_align_motion": _fval(row.get("obj_axis_align_motion")),
                                "obj_corepull_align_motion": _fval(row.get("obj_corepull_align_motion")),
                                "vmax": _fval(row.get("vmax")),
                            }
                        )
                    if lead_hours:
                        first_match_lead = max(lead_hours)
                    if "object_time" in m_best.columns:
                        first_match_time = _iso(m_best["object_time"].min())

                summary = {
                    "storm_id": str(storm_id),
                    "genesis_time": _iso(genesis),
                    "end_time": _iso(end_time),
                    "duration_hours": _fval(duration_h),
                    "max_vmax": vmax_max,
                    "track_points": int(t["time"].nunique()),
                    "matched_points": int(m["track_time"].nunique()) if not m.empty else 0,
                    "first_match_time": first_match_time,
                    "first_match_lead_hours": _fval(first_match_lead),
                    "lead_hours": lead_hours,
                    "best_match_series": series,
                }
                out_path = storms_dir / f"{_safe_storm_id(storm_id)}.json"
                out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[report-pack] outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
