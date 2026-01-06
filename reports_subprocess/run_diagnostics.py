#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_diagnostics.py
Read-only diagnostics for a completed run folder and key artifacts.

Outputs:
  results/reports/<run_id>/diagnostics/diagnostics.json
  results/reports/<run_id>/diagnostics/diagnostics_summary.md
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_parquet(path: Path) -> bool:
    suffixes = "".join(path.suffixes[-2:]).lower()
    return suffixes == ".parquet" or path.suffix.lower() == ".parquet"


def _read_schema(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    if _is_parquet(path):
        try:
            import pyarrow.parquet as pq  # type: ignore

            schema = pq.ParquetFile(path).schema_arrow
            return [(field.name, str(field.type)) for field in schema]
        except Exception:
            try:
                cols = pd.read_parquet(path, columns=None).columns
                return [(c, "unknown") for c in cols]
            except Exception:
                return []
    try:
        head = pd.read_csv(path, nrows=1)
        return [(c, "unknown") for c in head.columns]
    except Exception:
        return []


def _schema_hash(schema: List[Tuple[str, str]]) -> str:
    raw = "\n".join([f"{k}:{v}" for k, v in schema])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_df(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    opener = gzip.open if path.suffix.lower().endswith("gz") else open
    rows = -1
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for rows, _ in enumerate(f):
            pass
    if rows < 0:
        return 0
    # subtract header
    return max(0, rows)


def _count_parquet_rows(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        meta_rows = pf.metadata.num_rows if pf.metadata else None
        counted = None
        if meta_rows is None or meta_rows == 0:
            counted = sum(rg.num_rows for rg in pf.metadata.row_groups) if pf.metadata else None
        return meta_rows, counted
    except Exception:
        return None, None


def _file_info(path: Path, max_hash_mb: int = 200) -> Dict[str, Optional[object]]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "rows_metadata": None,
            "rows_counted": None,
            "sha256": None,
            "provenance": None,
        }
    stat = path.stat()
    meta_rows = counted = None
    if _is_parquet(path):
        meta_rows, counted = _count_parquet_rows(path)
    else:
        counted = _count_csv_rows(path)
    sha = None
    size_mb = stat.st_size / (1024 * 1024)
    if size_mb <= max_hash_mb:
        sha = _sha256_file(path)
    prov = _find_provenance(path)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows_metadata": meta_rows,
        "rows_counted": counted,
        "sha256": sha,
        "provenance": prov,
    }


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_provenance(path: Path) -> Optional[str]:
    candidates = [
        path.with_suffix(path.suffix + ".provenance.json"),
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(".provenance.json"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _sample_parquet(path: Path, columns: Optional[List[str]], n: int, seed: int) -> pd.DataFrame:
    import pyarrow.parquet as pq  # type: ignore

    if n <= 0:
        return pd.DataFrame()
    pf = pq.ParquetFile(path)
    if pf.num_row_groups == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    groups = list(range(pf.num_row_groups))
    rng.shuffle(groups)
    samples = []
    for rg in groups:
        table = pf.read_row_group(rg, columns=columns)
        df = table.to_pandas()
        if df.empty:
            continue
        if len(df) > n:
            df = df.sample(n=min(n, len(df)), random_state=seed)
        samples.append(df)
        if sum(len(s) for s in samples) >= n:
            break
    if not samples:
        return pd.DataFrame()
    out = pd.concat(samples, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out.reset_index(drop=True)


def _sample_csv(path: Path, columns: Optional[List[str]], n: int, seed: int, chunksize: int = 200_000) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    sample = None
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, compression="infer", low_memory=False):
        if chunk.empty:
            continue
        if sample is None:
            take = min(n, len(chunk))
            sample = chunk.sample(n=take, random_state=seed)
        else:
            combined = pd.concat([sample, chunk], ignore_index=True)
            if len(combined) > n:
                sample = combined.sample(n=n, random_state=seed)
            else:
                sample = combined
        if sample is not None and len(sample) >= n and rng.random() < 0.1:
            # small chance to early stop once we have enough
            break
    return sample.reset_index(drop=True) if sample is not None else pd.DataFrame()


def _sample_table(path: Path, columns: Optional[List[str]], n: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if _is_parquet(path):
        return _sample_parquet(path, columns, n, seed)
    return _sample_csv(path, columns, n, seed)


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    if _is_parquet(path):
        meta_rows, counted = _count_parquet_rows(path)
        return int(counted or meta_rows or 0)
    return _count_csv_rows(path)


def _detect_cell_keys(cols: List[str]) -> List[str]:
    if "cell_id" in cols:
        return ["cell_id"]
    if {"ilat", "ilon"}.issubset(cols):
        return ["ilat", "ilon"]
    if {"lat", "lon"}.issubset(cols):
        return ["lat", "lon"]
    return []


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def _inventory_entry(
    name: str,
    path: Optional[Path],
    *,
    lead_col: str,
    sample_rows: int,
    seed: int,
) -> Dict[str, Any]:
    if path is None:
        return {"name": name, "path": None, "exists": False}
    if not path.exists():
        return {"name": name, "path": str(path), "exists": False}

    schema = _read_schema(path)
    cols = [c for c, _ in schema]
    rows = _row_count(path)
    cell_keys = _detect_cell_keys(cols)
    time_col = "time" if "time" in cols else None
    key_cols = [c for c in ([time_col] if time_col else []) + cell_keys if c]
    if lead_col in cols:
        key_cols.append(lead_col)

    sample = _sample_table(path, key_cols or None, min(sample_rows, max(1, rows)), seed)
    if time_col and time_col in sample.columns:
        tvals = pd.to_datetime(sample[time_col], errors="coerce")
        time_min = tvals.min()
        time_max = tvals.max()
        uniq_time = _safe_nunique(tvals)
    else:
        time_min = None
        time_max = None
        uniq_time = None

    uniq_cell = None
    if cell_keys and all(c in sample.columns for c in cell_keys):
        uniq_cell = int(sample[cell_keys].drop_duplicates().shape[0])

    uniq_keys: Dict[str, int] = {}
    for c in key_cols:
        if c in sample.columns:
            uniq_keys[c] = _safe_nunique(sample[c])

    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "rows": rows,
        "time_min": None if time_min is None or pd.isna(time_min) else str(time_min),
        "time_max": None if time_max is None or pd.isna(time_max) else str(time_max),
        "unique_time": uniq_time,
        "cell_keys": cell_keys,
        "unique_cell": uniq_cell,
        "unique_keys": uniq_keys,
        "lead_col": lead_col if lead_col in cols else None,
        "schema_hash": _schema_hash(schema),
    }


def _write_sample_table(
    src: Path,
    dst: Path,
    columns: Optional[List[str]],
    n: int,
    seed: int,
) -> None:
    if not src.exists():
        return
    sample = _sample_table(src, columns, n, seed)
    if sample.empty:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in {".parquet", ".parq", ".pq"}:
        sample.to_parquet(dst, index=False)
    else:
        sample.to_csv(dst, index=False)


def _parse_run_name(run_dir: Path) -> Optional[str]:
    report_json = list(run_dir.glob("*_report.json"))
    if not report_json:
        return None
    try:
        data = json.loads(report_json[0].read_text(encoding="utf-8"))
        return data.get("run_summary", {}).get("run_name")
    except Exception:
        return None


def _pick_latest(glob_pat: str) -> Optional[Path]:
    paths = list(Path().glob(glob_pat))
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _split_cols(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _lead_from_name(name: str) -> Optional[int]:
    m = re.search(r"lead(\d+)", name)
    return int(m.group(1)) if m else None


def _detect_lead_hashes(
    paths: List[Path],
    lead_col: str,
    label_col: str,
    key_cols: List[str],
    sample_rows: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    results: List[Dict[str, object]] = []
    if not paths:
        return results, "no training tables provided"

    for path in paths:
        if not path.exists():
            continue
        schema = _read_schema(path)
        cols = [c for c, _ in schema]
        lead_val = _lead_from_name(path.name)
        base_cols = set(cols)
        keep_cols = [c for c in cols if c in base_cols]
        df = _sample_table(path, keep_cols, sample_rows, seed)
        if df.empty:
            continue
        if lead_val is None and lead_col in df.columns:
            # try to infer lead from data if possible
            lv = pd.to_numeric(df[lead_col], errors="coerce").dropna().unique()
            lead_val = int(lv[0]) if len(lv) == 1 else None
        feature_cols = [c for c in df.columns if c not in key_cols + [label_col, lead_col]]
        feature_df = df[feature_cols].copy() if feature_cols else pd.DataFrame()
        label_df = df[[label_col]].copy() if label_col in df.columns else pd.DataFrame()
        if key_cols:
            key_cols_present = [c for c in key_cols if c in df.columns]
            if key_cols_present:
                df = df.sort_values(key_cols_present).reset_index(drop=True)
                feature_df = feature_df.sort_index().reset_index(drop=True)
                if not label_df.empty:
                    label_df = label_df.sort_index().reset_index(drop=True)
        results.append(
            {
                "path": str(path),
                "lead": lead_val,
                "schema_hash": _schema_hash(schema),
                "feature_hash": _hash_df(feature_df),
                "label_hash": _hash_df(label_df),
            }
        )
    return results, None


def _detect_lead_hashes_from_single(
    path: Path,
    lead_col: str,
    label_col: str,
    key_cols: List[str],
    sample_rows: int,
    seed: int,
    max_leads: int = 20,
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    if not path.exists():
        return [], "training table missing"
    schema = _read_schema(path)
    cols = [c for c, _ in schema]
    if lead_col not in cols:
        return [], f"lead column '{lead_col}' not found"
    try:
        if _is_parquet(path):
            lead_vals = pd.read_parquet(path, columns=[lead_col])[lead_col]
        else:
            lead_vals = pd.read_csv(path, usecols=[lead_col], compression="infer", low_memory=False)[lead_col]
        lead_vals = pd.to_numeric(lead_vals, errors="coerce").dropna().unique().tolist()
    except Exception:
        return [], "failed to read lead column"
    lead_vals = sorted([int(x) for x in lead_vals if pd.notna(x)])[:max_leads]
    results: List[Dict[str, object]] = []
    for lead in lead_vals:
        if _is_parquet(path):
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            samples = []
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                df = table.to_pandas()
                df = df.loc[pd.to_numeric(df[lead_col], errors="coerce") == lead]
                if df.empty:
                    continue
                if len(df) > sample_rows:
                    df = df.sample(n=sample_rows, random_state=seed)
                samples.append(df)
                if sum(len(s) for s in samples) >= sample_rows:
                    break
            df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        else:
            df = pd.DataFrame()
            for chunk in pd.read_csv(path, compression="infer", low_memory=False, chunksize=200_000):
                chunk = chunk.loc[pd.to_numeric(chunk[lead_col], errors="coerce") == lead]
                if chunk.empty:
                    continue
                if len(chunk) > sample_rows:
                    chunk = chunk.sample(n=sample_rows, random_state=seed)
                df = pd.concat([df, chunk], ignore_index=True)
                if len(df) >= sample_rows:
                    break
        if df.empty:
            continue
        feature_cols = [c for c in df.columns if c not in key_cols + [label_col, lead_col]]
        feature_df = df[feature_cols].copy() if feature_cols else pd.DataFrame()
        label_df = df[[label_col]].copy() if label_col in df.columns else pd.DataFrame()
        key_cols_present = [c for c in key_cols if c in df.columns]
        if key_cols_present:
            df = df.sort_values(key_cols_present).reset_index(drop=True)
            feature_df = feature_df.sort_index().reset_index(drop=True)
            if not label_df.empty:
                label_df = label_df.sort_index().reset_index(drop=True)
        results.append(
            {
                "path": str(path),
                "lead": lead,
                "schema_hash": _schema_hash(schema),
                "feature_hash": _hash_df(feature_df),
                "label_hash": _hash_df(label_df),
            }
        )
    return results, None


def _cols_hash(cols: List[str]) -> str:
    raw = "\n".join(cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stats_hash(df: pd.DataFrame, cols: List[str]) -> str:
    lines: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s = s[np.isfinite(s.to_numpy())]
        if s.empty:
            continue
        lines.append(
            f"{c}:{float(s.mean()):.6g}:{float(s.std(ddof=0)):.6g}:{float(s.min()):.6g}:{float(s.max()):.6g}"
        )
    raw = "\n".join(lines)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _lead_fingerprints_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, str]:
    cols_hash = _cols_hash(feature_cols)
    stats_hash = _stats_hash(df, feature_cols)
    sample_hash = _hash_df(df[feature_cols].copy()) if feature_cols else ""
    return {"columns_hash": cols_hash, "stats_hash": stats_hash, "sample_hash": sample_hash}


def _lead_fingerprints_from_tables(
    paths: List[Path],
    lead_col: str,
    label_col: str,
    key_cols: List[str],
    sample_rows: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    results: List[Dict[str, Any]] = []
    if not paths:
        return results, "no training tables provided"
    for path in paths:
        if not path.exists():
            continue
        schema = _read_schema(path)
        cols = [c for c, _ in schema]
        lead_val = _lead_from_name(path.name)
        df = _sample_table(path, cols or None, sample_rows, seed)
        if df.empty:
            continue
        if lead_val is None and lead_col in df.columns:
            lv = pd.to_numeric(df[lead_col], errors="coerce").dropna().unique()
            lead_val = int(lv[0]) if len(lv) == 1 else None
        feature_cols = [c for c in df.columns if c not in key_cols + [label_col, lead_col]]
        fps = _lead_fingerprints_from_df(df, feature_cols)
        results.append(
            {
                "path": str(path),
                "lead_h": lead_val,
                "feature_count": len(feature_cols),
                "columns_hash": fps["columns_hash"],
                "stats_hash": fps["stats_hash"],
                "sample_hash": fps["sample_hash"],
            }
        )
    return results, None


def _lead_fingerprints_from_single(
    path: Path,
    lead_col: str,
    label_col: str,
    key_cols: List[str],
    sample_rows: int,
    seed: int,
    max_leads: int = 20,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return [], "training table missing"
    schema = _read_schema(path)
    cols = [c for c, _ in schema]
    if lead_col not in cols:
        return [], f"lead column '{lead_col}' not found"
    try:
        if _is_parquet(path):
            lead_vals = pd.read_parquet(path, columns=[lead_col])[lead_col]
        else:
            lead_vals = pd.read_csv(path, usecols=[lead_col], compression="infer", low_memory=False)[lead_col]
        lead_vals = pd.to_numeric(lead_vals, errors="coerce").dropna().unique().tolist()
    except Exception:
        return [], "failed to read lead column"
    lead_vals = sorted([int(x) for x in lead_vals if pd.notna(x)])[:max_leads]
    results: List[Dict[str, Any]] = []
    for lead in lead_vals:
        if _is_parquet(path):
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            samples = []
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                df = table.to_pandas()
                df = df.loc[pd.to_numeric(df[lead_col], errors="coerce") == lead]
                if df.empty:
                    continue
                if len(df) > sample_rows:
                    df = df.sample(n=sample_rows, random_state=seed)
                samples.append(df)
                if sum(len(s) for s in samples) >= sample_rows:
                    break
            df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        else:
            df = pd.DataFrame()
            for chunk in pd.read_csv(path, compression="infer", low_memory=False, chunksize=200_000):
                chunk = chunk.loc[pd.to_numeric(chunk[lead_col], errors="coerce") == lead]
                if chunk.empty:
                    continue
                if len(chunk) > sample_rows:
                    chunk = chunk.sample(n=sample_rows, random_state=seed)
                df = pd.concat([df, chunk], ignore_index=True)
                if len(df) >= sample_rows:
                    break
        if df.empty:
            continue
        feature_cols = [c for c in df.columns if c not in key_cols + [label_col, lead_col]]
        fps = _lead_fingerprints_from_df(df, feature_cols)
        results.append(
            {
                "path": str(path),
                "lead_h": lead,
                "feature_count": len(feature_cols),
                "columns_hash": fps["columns_hash"],
                "stats_hash": fps["stats_hash"],
                "sample_hash": fps["sample_hash"],
            }
        )
    return results, None


def _overlap_count(
    train_path: Path,
    val_path: Path,
    key_cols: List[str],
    max_exact_rows: int,
) -> Tuple[Optional[int], str]:
    if not train_path.exists() or not val_path.exists():
        return None, "missing"
    def _read_keys(path: Path) -> pd.DataFrame:
        if _is_parquet(path):
            return pd.read_parquet(path, columns=key_cols)
        return pd.read_csv(path, usecols=key_cols, compression="infer", low_memory=False)

    train_df = _read_keys(train_path)
    val_df = _read_keys(val_path)
    if train_df.empty or val_df.empty:
        return 0, "empty"
    # exact check if manageable
    if len(train_df) <= max_exact_rows and len(val_df) <= max_exact_rows:
        merged = train_df.merge(val_df, on=key_cols, how="inner")
        return int(len(merged)), "exact"

    # approximate overlap
    smaller, larger = (train_df, val_df) if len(train_df) <= len(val_df) else (val_df, train_df)
    key_set = set(tuple(x) for x in smaller[key_cols].itertuples(index=False, name=None))
    overlap = sum(tuple(x) in key_set for x in larger[key_cols].itertuples(index=False, name=None))
    return int(overlap), "approx"


def _lead_label_stats(
    path: Path,
    lead_col: str,
    label_col: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    schema = _read_schema(path)
    available = {c for c, _ in schema}
    if lead_col not in available or label_col not in available:
        return pd.DataFrame()
    cols = [c for c in (lead_col, label_col) if c]
    if not cols:
        return pd.DataFrame()
    try:
        if _is_parquet(path):
            df = pd.read_parquet(path, columns=cols)
        else:
            df = pd.read_csv(path, usecols=cols, compression="infer", low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or lead_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame()
    lead_vals = pd.to_numeric(df[lead_col], errors="coerce")
    y = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0).astype(int)
    stats = (
        pd.DataFrame({"lead_h": lead_vals, "y": y})
        .dropna(subset=["lead_h"])
        .groupby("lead_h", sort=True)["y"]
        .agg(pos_count="sum", n_rows="count")
        .reset_index()
    )
    if not stats.empty:
        stats["pos_frac"] = stats["pos_count"] / stats["n_rows"].replace(0, np.nan)
    return stats


def _slowtick_identical(slowtick_dir: Path) -> Tuple[Optional[bool], str]:
    cov_path = slowtick_dir / "coverage_timeseries.csv"
    if not cov_path.exists():
        return None, "missing coverage_timeseries.csv"
    try:
        df = pd.read_csv(cov_path)
        if df.empty or "lead_h" not in df.columns:
            return None, "missing lead_h"
        piv = df.pivot_table(index="time", columns="lead_h", values="coverage", aggfunc="mean")
        if piv.shape[1] <= 1:
            return None, "insufficient leads"
        ref = piv.iloc[:, 0]
        for col in piv.columns[1:]:
            if not np.allclose(ref.fillna(0).to_numpy(), piv[col].fillna(0).to_numpy()):
                return False, "coverage differs by lead"
        return True, "coverage identical across leads"
    except Exception as exc:
        return None, f"error: {exc}"


def _time_shift_test(
    path: Path,
    label_col: str,
    lead_col: str,
    key_cols: List[str],
    features_used: List[str],
    sample_rows: int,
    seed: int,
    shift_hours: int = 72,
    max_features: int = 50,
) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "skip", "reason": "training table missing"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return {"status": "skip", "reason": "sklearn missing"}

    schema = _read_schema(path)
    cols = [c for c, _ in schema]
    time_col = "time" if "time" in cols else None
    if not time_col:
        return {"status": "skip", "reason": "time column missing"}
    if label_col not in cols:
        return {"status": "skip", "reason": f"label column '{label_col}' missing"}

    if features_used:
        feat_cols = [c for c in features_used if c in cols]
    else:
        exclude = set(key_cols + [label_col, lead_col])
        feat_cols = [c for c in cols if c not in exclude]
    if not feat_cols:
        return {"status": "skip", "reason": "no feature columns found"}
    feat_cols = feat_cols[:max_features]

    use_cols = list(dict.fromkeys([time_col, *key_cols, label_col, *feat_cols]))
    df = _sample_table(path, use_cols, sample_rows, seed)
    if df.empty:
        return {"status": "skip", "reason": "empty sample"}

    dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
    if dup_cols:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    y = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0).astype(int).to_numpy()

    def _fit_eval(din: pd.DataFrame) -> Dict[str, float]:
        X = din[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        yv = (pd.to_numeric(din[label_col], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
        if len(np.unique(yv)) < 2:
            return {"auc": float("nan"), "prauc": float("nan")}
        X_train, X_val, y_train, y_val = train_test_split(
            X, yv, test_size=0.2, random_state=seed, stratify=yv
        )
        model = LogisticRegression(max_iter=200, n_jobs=1)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        return {
            "auc": float(roc_auc_score(y_val, prob)),
            "prauc": float(average_precision_score(y_val, prob)),
        }

    base_metrics = _fit_eval(df)
    results = {"baseline": base_metrics, "shifts": {}}
    if dup_cols:
        results["dropped_duplicate_columns"] = dup_cols

    keys = [c for c in key_cols if c in df.columns]
    if time_col not in keys:
        keys.append(time_col)
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {"status": "skip", "reason": "no join keys for shift test"}

    df_feat = df[keys + feat_cols].copy()
    df_lab = df[keys + [label_col]].copy()
    for shift in (+shift_hours, -shift_hours):
        df_shift = df_feat.copy()
        df_shift[time_col] = df_shift[time_col] + pd.to_timedelta(int(shift), unit="h")
        merged = df_lab.merge(df_shift, on=keys, how="inner")
        if merged.empty:
            results["shifts"][str(shift)] = {"rows": 0, "auc": float("nan"), "prauc": float("nan")}
            continue
        metrics = _fit_eval(merged)
        results["shifts"][str(shift)] = {"rows": int(len(merged)), **metrics}

    def _delta(metric: str, shift_key: str) -> float:
        base = results["baseline"].get(metric)
        sh = results["shifts"].get(shift_key, {}).get(metric)
        if base is None or sh is None or not np.isfinite(base) or not np.isfinite(sh):
            return float("nan")
        return float(sh - base)

    delta_pos = _delta("auc", str(shift_hours))
    delta_neg = _delta("auc", str(-shift_hours))
    delta_pr_pos = _delta("prauc", str(shift_hours))
    delta_pr_neg = _delta("prauc", str(-shift_hours))
    results["delta_auc"] = {"plus": delta_pos, "minus": delta_neg}
    results["delta_prauc"] = {"plus": delta_pr_pos, "minus": delta_pr_neg}

    unchanged = (
        np.isfinite(delta_pos) and np.isfinite(delta_neg)
        and abs(delta_pos) < 0.02 and abs(delta_neg) < 0.02
        and np.isfinite(delta_pr_pos) and np.isfinite(delta_pr_neg)
        and abs(delta_pr_pos) < 0.02 and abs(delta_pr_neg) < 0.02
    )
    results["status"] = "fail" if unchanged else "pass"
    results["unchanged"] = bool(unchanged)
    return results


def _summarize_checks(checks: List[Dict[str, object]]) -> Dict[str, int]:
    counts = {"fail": 0, "warn": 0, "pass": 0, "skip": 0}
    for c in checks:
        status = str(c.get("status") or "skip").lower()
        if status not in counts:
            status = "skip"
        counts[status] += 1
    return counts


def _write_diagnostics_report(
    path: Path,
    *,
    run_name: str,
    run_dir: Path,
    checks: List[Dict[str, object]],
    artifacts: Dict[str, List[Dict[str, Optional[object]]]],
    extra_paths: Dict[str, Optional[Path]],
) -> None:
    counts = _summarize_checks(checks)
    lines: List[str] = []
    lines.append(f"# Diagnostics report: {run_name}")
    lines.append("")
    lines.append(f"Run folder: `{run_dir}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Issue | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    for c in checks:
        status = c.get("status")
        lines.append(f"| {c.get('issue')} | {status} | {c.get('evidence')} |")
    lines.append("")
    lines.append("## Check counts")
    lines.append("")
    lines.append(f"- pass: {counts['pass']}")
    lines.append(f"- warn: {counts['warn']}")
    lines.append(f"- fail: {counts['fail']}")
    lines.append(f"- skip: {counts['skip']}")
    lines.append("")
    lines.append("## Artifact inventory")
    lines.append("")
    for name, items in artifacts.items():
        lines.append(f"- {name}: {len(items)} file(s)")
    lines.append("")
    lines.append("## Diagnostics outputs")
    lines.append("")
    for label, p in extra_paths.items():
        if p:
            lines.append(f"- {label}: `{p}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only diagnostics for a completed run.")
    ap.add_argument("--run-dir", required=True, help="Run folder under results/reports/<run_id>.")
    ap.add_argument("--run-name", default=None, help="Run name (used for alerts/seedmaps defaults).")
    ap.add_argument("--alerts-dir", default="results/alerts", help="Directory with alerts_* files.")
    ap.add_argument("--alerts", nargs="*", default=None, help="Explicit alerts files (overrides autodetect).")
    ap.add_argument("--alerts-base", default=None, help="Explicit base alerts file (optional).")
    ap.add_argument("--alerts-thr", default=None, help="Explicit throttled alerts file (optional).")
    ap.add_argument("--alerts-final", default=None, help="Explicit final alerts file (optional).")
    ap.add_argument("--base-grid", nargs="*", default=None, help="Base labelled grid parquet(s).")
    ap.add_argument("--base-state", nargs="*", default=None, help="Base state parquet(s).")
    ap.add_argument("--predictions", default=None, help="Predictions parquet/csv.")
    ap.add_argument("--objects", default=None, help="Objects parquet.")
    ap.add_argument("--objects-by-hour", default=None, help="Objects-by-hour parquet.")
    ap.add_argument("--state-table", default=None, help="State table parquet/csv (optional).")
    ap.add_argument("--lookup-panel", default=None, help="Lookup panel parquet/csv (optional).")
    ap.add_argument("--matches-table", default=None, help="Matches table parquet/csv (optional).")
    ap.add_argument("--train-table", default=None, help="Unified training table (parquet/csv).")
    ap.add_argument("--train-tables-glob", nargs="*", default=None, help="Glob(s) for per-lead training tables.")
    ap.add_argument("--train-keys", default=None, help="Train keys table (parquet/csv).")
    ap.add_argument("--val-keys", default=None, help="Validation keys table (parquet/csv).")
    ap.add_argument("--metrics-json", default=None, help="Metrics JSON with feature list.")
    ap.add_argument("--feature-metadata", default=None, help="Optional feature metadata JSON.")
    ap.add_argument("--slowtick-dir", default=None, help="Slowtick output directory.")
    ap.add_argument("--label-col", default="y_viable", help="Label column name for training.")
    ap.add_argument("--lead-col", default="lead_h", help="Lead column name (if present).")
    ap.add_argument("--key-cols", default="time,ilat,ilon", help="Comma-separated key columns.")
    ap.add_argument("--sample-rows", type=int, default=50_000, help="Sample size for hashing.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    ap.add_argument("--max-overlap-rows", type=int, default=5_000_000, help="Max rows for exact overlap check.")
    ap.add_argument("--max-hash-mb", type=int, default=200, help="Max file size for sha256 hashing.")
    ap.add_argument("--out-dir", default=None, help="Override diagnostics output directory.")
    ap.add_argument("--diagnostics-artifacts-dir", default=None, help="Directory for small diagnostics artifacts.")
    ap.add_argument("--time-shift-hours", type=int, default=72, help="Hours to shift features in time-shift test.")
    ap.add_argument("--skip-time-shift", action="store_true", help="Skip time-shift diagnostics.")
    ap.add_argument(
        "--fail-on-checks",
        action="store_true",
        help="Exit non-zero if any check is marked as fail.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_name = args.run_name or _parse_run_name(run_dir) or "unknown_run"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path("results/diagnostics")
    project_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.diagnostics_artifacts_dir) if args.diagnostics_artifacts_dir else (project_dir / "diagnostics_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    key_cols = _split_cols(args.key_cols)
    label_col = args.label_col
    lead_col = args.lead_col

    # Default artifact detection
    def _first_existing(cands: List[str]) -> Optional[Path]:
        for c in cands:
            p = Path(c)
            if p.exists():
                return p
        return None

    alerts_paths = []
    if args.alerts:
        alerts_paths = [Path(p) for p in args.alerts]
    else:
        alerts_dir = Path(args.alerts_dir)
        if run_name and run_name != "unknown_run":
            alerts_paths = list(alerts_dir.glob(f"alerts_{run_name}_final.*"))
        if not alerts_paths:
            alerts_paths = list(alerts_dir.glob("alerts_*_final.*"))

    alerts_base = Path(args.alerts_base) if args.alerts_base else _first_existing(
        [
            f"{args.alerts_dir}/alerts_{run_name}_base.parquet",
            f"{args.alerts_dir}/alerts_{run_name}_base.csv.gz",
            f"{args.alerts_dir}/alerts_{run_name}_base.csv",
        ]
    )
    alerts_thr = Path(args.alerts_thr) if args.alerts_thr else _first_existing(
        [
            f"{args.alerts_dir}/alerts_{run_name}_thr.parquet",
            f"{args.alerts_dir}/alerts_{run_name}_thr.csv.gz",
            f"{args.alerts_dir}/alerts_{run_name}_thr.csv",
        ]
    )
    alerts_final = Path(args.alerts_final) if args.alerts_final else _first_existing(
        [
            f"{args.alerts_dir}/alerts_{run_name}_final.parquet",
            f"{args.alerts_dir}/alerts_{run_name}_final.csv.gz",
            f"{args.alerts_dir}/alerts_{run_name}_final.csv",
        ]
    )

    base_grid_paths = [Path(p) for p in (args.base_grid or [])]
    base_state_paths = [Path(p) for p in (args.base_state or [])]
    if not base_grid_paths:
        cand = _pick_latest("data/grid_labelled_*_id.parquet")
        if cand:
            base_grid_paths = [cand]
    if not base_state_paths:
        cand = _pick_latest("data/grid_labelled_*_state.parquet")
        if cand:
            base_state_paths = [cand]

    predictions_path = Path(args.predictions) if args.predictions else _pick_latest("results/predictions*.parquet")
    objects_path = Path(args.objects) if args.objects else Path("results/objects/objects.parquet")
    objects_by_hour_path = Path(args.objects_by_hour) if args.objects_by_hour else Path("results/objects/objects_by_hour.parquet")
    train_table_path = Path(args.train_table) if args.train_table else _pick_latest("data/grid_train_*_targets.parquet")
    state_table_path = Path(args.state_table) if args.state_table else _pick_latest("data/*_state*.parquet")
    lookup_panel_path = Path(args.lookup_panel) if args.lookup_panel else _pick_latest("data/*_slim*.parquet")
    matches_table_path = Path(args.matches_table) if args.matches_table else _pick_latest("results/matches/*.parquet")

    train_tables: List[Path] = []
    if args.train_tables_glob:
        for pat in args.train_tables_glob:
            train_tables.extend([Path(p) for p in Path().glob(pat)])
    else:
        train_tables.extend([Path(p) for p in Path().glob("data/grid_train_*_lead*.parquet")])

    metrics_json = Path(args.metrics_json) if args.metrics_json else _pick_latest("models/*metrics.json")
    feature_metadata = Path(args.feature_metadata) if args.feature_metadata else None
    slowtick_dir = Path(args.slowtick_dir) if args.slowtick_dir else (run_dir / "slowtick")

    # Artifact info
    artifacts: Dict[str, List[Dict[str, Optional[object]]]] = {}
    artifacts["alerts"] = [_file_info(p, max_hash_mb=args.max_hash_mb) for p in alerts_paths]
    if alerts_base:
        artifacts["alerts_base"] = [_file_info(alerts_base, max_hash_mb=args.max_hash_mb)]
    if alerts_thr:
        artifacts["alerts_thr"] = [_file_info(alerts_thr, max_hash_mb=args.max_hash_mb)]
    if alerts_final:
        artifacts["alerts_final"] = [_file_info(alerts_final, max_hash_mb=args.max_hash_mb)]
    artifacts["base_grid"] = [_file_info(p, max_hash_mb=args.max_hash_mb) for p in base_grid_paths]
    artifacts["base_state"] = [_file_info(p, max_hash_mb=args.max_hash_mb) for p in base_state_paths]
    if predictions_path:
        artifacts["predictions"] = [_file_info(predictions_path, max_hash_mb=args.max_hash_mb)]
    artifacts["objects"] = [_file_info(objects_path, max_hash_mb=args.max_hash_mb)]
    artifacts["objects_by_hour"] = [_file_info(objects_by_hour_path, max_hash_mb=args.max_hash_mb)]
    if state_table_path:
        artifacts["state_table"] = [_file_info(state_table_path, max_hash_mb=args.max_hash_mb)]
    if lookup_panel_path:
        artifacts["lookup_panel"] = [_file_info(lookup_panel_path, max_hash_mb=args.max_hash_mb)]
    if matches_table_path:
        artifacts["matches_table"] = [_file_info(matches_table_path, max_hash_mb=args.max_hash_mb)]
    if train_table_path:
        artifacts["train_table"] = [_file_info(train_table_path, max_hash_mb=args.max_hash_mb)]
    if train_tables:
        artifacts["train_tables"] = [_file_info(p, max_hash_mb=args.max_hash_mb) for p in train_tables]

    # Phase 0: schema inventory
    inventory_items = [
        ("train_table", train_table_path),
        ("predictions", predictions_path),
        ("alerts_base", alerts_base),
        ("alerts_thr", alerts_thr),
        ("alerts_final", alerts_final),
        ("state_table", state_table_path),
        ("lookup_panel", lookup_panel_path),
        ("objects_by_hour", objects_by_hour_path),
        ("matches_table", matches_table_path),
    ]
    schema_inventory = [
        _inventory_entry(name, path, lead_col=lead_col, sample_rows=args.sample_rows, seed=args.seed)
        for name, path in inventory_items
        if path is not None
    ]
    (project_dir / "schema_inventory.json").write_text(
        json.dumps(schema_inventory, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # Small sample artifacts for quick inspection
    sample_candidates = ["time", "lat", "lon", "ilat", "ilon", "cell_id", lead_col, label_col]
    for name, path in inventory_items:
        if path is None or not path.exists():
            continue
        cols = [c for c, _ in _read_schema(path)]
        keep_cols = [c for c in sample_candidates if c in cols]
        out_path = artifacts_dir / f"{name}_sample.parquet"
        _write_sample_table(path, out_path, keep_cols or None, n=min(2000, args.sample_rows), seed=args.seed)

    # Lead hash detection
    lead_hashes = []
    lead_hash_note = None
    if train_tables:
        lead_hashes, lead_hash_note = _detect_lead_hashes(
            train_tables, lead_col, label_col, key_cols, args.sample_rows, args.seed
        )
    elif train_table_path and train_table_path.exists():
        lead_hashes, lead_hash_note = _detect_lead_hashes_from_single(
            train_table_path, lead_col, label_col, key_cols, args.sample_rows, args.seed
        )

    lead_fingerprints = []
    lead_fp_note = None
    if train_tables:
        lead_fingerprints, lead_fp_note = _lead_fingerprints_from_tables(
            train_tables, lead_col, label_col, key_cols, args.sample_rows, args.seed
        )
    elif train_table_path and train_table_path.exists():
        lead_fingerprints, lead_fp_note = _lead_fingerprints_from_single(
            train_table_path, lead_col, label_col, key_cols, args.sample_rows, args.seed
        )
    (project_dir / "lead_feature_fingerprints.json").write_text(
        json.dumps({"note": lead_fp_note, "fingerprints": lead_fingerprints}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # Label stats per lead
    lead_label_stats = pd.DataFrame()
    if train_tables:
        stats_rows = []
        for p in train_tables:
            lead_val = _lead_from_name(p.name)
            stats = _lead_label_stats(p, lead_col, label_col)
            if stats.empty and lead_val is not None:
                stats = pd.DataFrame({"lead_h": [lead_val], "pos_count": [np.nan], "n_rows": [np.nan]})
            if lead_val is not None:
                stats["lead_h"] = lead_val
            stats_rows.append(stats)
        if stats_rows:
            lead_label_stats = pd.concat(stats_rows, ignore_index=True)
    elif train_table_path and train_table_path.exists():
        lead_label_stats = _lead_label_stats(train_table_path, lead_col, label_col)
    if not lead_label_stats.empty:
        lead_label_stats.to_parquet(project_dir / "lead_label_stats.parquet", index=False)

    # Train/val overlap (auto-discover if not explicitly provided)
    overlap_count = None
    overlap_mode = "missing"
    overlap_rows = []
    if args.train_keys and args.val_keys:
        overlap_count, overlap_mode = _overlap_count(
            Path(args.train_keys), Path(args.val_keys), key_cols, args.max_overlap_rows
        )
        overlap_rows.append(
            {
                "train_path": str(args.train_keys),
                "val_path": str(args.val_keys),
                "overlap": overlap_count,
                "mode": overlap_mode,
                "key_cols": ",".join(key_cols),
            }
        )
    else:
        train_keys = sorted(Path("models").glob("*_train_keys.*"))
        val_keys = sorted(Path("models").glob("*_val_keys.*"))
        val_map = {p.name.replace("_val_keys", ""): p for p in val_keys}
        for tr in train_keys:
            key = tr.name.replace("_train_keys", "")
            vl = val_map.get(key)
            if not vl:
                continue
            tr_cols = [c for c, _ in _read_schema(tr)]
            vl_cols = [c for c, _ in _read_schema(vl)]
            common = [c for c in tr_cols if c in vl_cols]
            keys = []
            if "time" in common:
                keys.append("time")
            if "cell_id" in common:
                keys.append("cell_id")
            elif {"ilat", "ilon"}.issubset(common):
                keys.extend(["ilat", "ilon"])
            elif {"lat", "lon"}.issubset(common):
                keys.extend(["lat", "lon"])
            if lead_col in common:
                keys.append(lead_col)
            if not keys:
                overlap_rows.append(
                    {
                        "train_path": str(tr),
                        "val_path": str(vl),
                        "overlap": None,
                        "mode": "missing_keys",
                        "key_cols": "",
                    }
                )
                continue
            ov, mode = _overlap_count(tr, vl, keys, args.max_overlap_rows)
            overlap_rows.append(
                {
                    "train_path": str(tr),
                    "val_path": str(vl),
                    "overlap": ov,
                    "mode": mode,
                    "key_cols": ",".join(keys),
                }
            )
        if overlap_rows:
            overlaps = [r for r in overlap_rows if r.get("overlap") is not None]
            if overlaps:
                overlap_count = max(int(r["overlap"]) for r in overlaps)
                overlap_mode = "auto"

    if overlap_rows:
        pd.DataFrame(overlap_rows).to_parquet(project_dir / "split_overlap.parquet", index=False)

    # Forbidden columns
    forbidden_patterns = [
        "storm", "storm_id", "storm_name", "storm_window", "storm_point",
        "vmax", "dist_to_track", "t_to_", "t_to_storm", "time", "row_id",
    ]
    features_used: List[str] = []
    if metrics_json and Path(metrics_json).exists():
        try:
            metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
            features_used = (
                metrics.get("features", [])
                or metrics.get("meta_features", [])
                or metrics.get("train", {}).get("features", [])
            )
        except Exception:
            features_used = []
    if not features_used and train_table_path and train_table_path.exists():
        schema = _read_schema(train_table_path)
        cols = [c for c, _ in schema]
        features_used = [c for c in cols if c not in key_cols + [label_col, lead_col]]

    forbidden_hits = []
    if features_used:
        for c in features_used:
            for pat in forbidden_patterns:
                if pat in c:
                    forbidden_hits.append(c)
                    break

    # Feature metadata causality
    causality_violations = []
    if feature_metadata and feature_metadata.exists():
        try:
            meta = json.loads(feature_metadata.read_text(encoding="utf-8"))
            for feat in features_used:
                info = meta.get(feat)
                if info is None:
                    continue
                if info.get("past_only") is False:
                    causality_violations.append(feat)
        except Exception:
            pass

    # Slowtick identical check
    slowtick_identical, slowtick_note = _slowtick_identical(slowtick_dir)

    # Time-shift test (Phase 3.1)
    time_shift_result = None
    if not args.skip_time_shift and train_table_path:
        time_shift_result = _time_shift_test(
            Path(train_table_path),
            label_col=label_col,
            lead_col=lead_col,
            key_cols=key_cols,
            features_used=features_used,
            sample_rows=args.sample_rows,
            seed=args.seed,
            shift_hours=args.time_shift_hours,
        )
        (project_dir / "time_shift_test.json").write_text(
            json.dumps(time_shift_result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    # Assemble checks
    checks = []

    # Lead column presence
    lead_missing_paths = []
    lead_required = False
    if train_tables:
        lead_required = True
        for p in train_tables:
            cols = [c for c, _ in _read_schema(p)]
            if lead_col not in cols:
                lead_missing_paths.append(str(p))
    elif train_table_path and train_table_path.exists():
        cols = [c for c, _ in _read_schema(train_table_path)]
        lead_required = ("t_to_storm_min_h" in cols) or ("lead" in label_col.lower())
        if lead_required and lead_col not in cols:
            lead_missing_paths.append(str(train_table_path))

    if lead_required:
        for name, path in (
            ("predictions", predictions_path),
            ("alerts_base", alerts_base),
            ("alerts_thr", alerts_thr),
            ("alerts_final", alerts_final),
        ):
            if path is None or not path.exists():
                continue
            cols = [c for c, _ in _read_schema(path)]
            if lead_col not in cols:
                lead_missing_paths.append(f"{name}:{path}")
        if lead_missing_paths:
            checks.append({
                "issue": "lead_h missing",
                "detected": True,
                "status": "fail",
                "evidence": f"missing lead_h in {len(lead_missing_paths)} table(s)",
            })
        else:
            checks.append({
                "issue": "lead_h missing",
                "detected": False,
                "status": "pass",
                "evidence": "lead_h present in required tables",
            })

    # Per-lead fingerprints
    if lead_fingerprints:
        fp_keys = {(d.get("columns_hash"), d.get("stats_hash"), d.get("sample_hash")) for d in lead_fingerprints}
        feat_identical = len(fp_keys) == 1
        label_identical = False
        if lead_hashes:
            label_hashes = {d.get("label_hash") for d in lead_hashes if d.get("label_hash")}
            label_identical = len(label_hashes) == 1
        elif not lead_label_stats.empty:
            label_identical = len(lead_label_stats[["pos_count", "pos_frac"]].dropna().drop_duplicates()) <= 1

        if feat_identical and label_identical:
            checks.append({
                "issue": "identical per-lead inputs",
                "detected": True,
                "status": "fail",
                "evidence": "feature fingerprints + label stats identical across leads",
            })
        elif feat_identical and not label_identical:
            checks.append({
                "issue": "identical per-lead inputs",
                "detected": False,
                "status": "pass",
                "evidence": "features identical, labels differ across leads",
            })
        else:
            checks.append({
                "issue": "identical per-lead inputs",
                "detected": False,
                "status": "warn",
                "evidence": "feature fingerprints differ across leads",
            })
    else:
        checks.append({
            "issue": "identical per-lead inputs",
            "detected": None,
            "status": "skip",
            "evidence": lead_fp_note or lead_hash_note or "no lead tables detected",
        })

    if overlap_count is not None:
        checks.append({
            "issue": "train/val overlap",
            "detected": overlap_count > 0,
            "status": "fail" if overlap_count > 0 else "pass",
            "evidence": f"overlap={overlap_count} ({overlap_mode})",
        })
    else:
        checks.append({
            "issue": "train/val overlap",
            "detected": None,
            "status": "skip",
            "evidence": "train_keys/val_keys not provided",
        })

    if forbidden_hits:
        checks.append({
            "issue": "forbidden/leaky columns in features",
            "detected": True,
            "status": "fail",
            "evidence": ", ".join(sorted(set(forbidden_hits))[:10]),
        })
    else:
        checks.append({
            "issue": "forbidden/leaky columns in features",
            "detected": False if features_used else None,
            "status": "pass" if features_used else "skip",
            "evidence": "no forbidden columns found" if features_used else "no feature list available",
        })

    if feature_metadata and feature_metadata.exists():
        if causality_violations:
            checks.append({
                "issue": "non-causal features in training",
                "detected": True,
                "status": "fail",
                "evidence": ", ".join(sorted(set(causality_violations))[:10]),
            })
        else:
            checks.append({
                "issue": "non-causal features in training",
                "detected": False,
                "status": "pass",
                "evidence": "no past_only=False features found",
            })
    else:
        checks.append({
            "issue": "non-causal features in training",
            "detected": None,
            "status": "skip",
            "evidence": "feature metadata not provided",
        })

    if slowtick_identical is True:
        checks.append({
            "issue": "slowtick identical across leads",
            "detected": True,
            "status": "warn",
            "evidence": slowtick_note,
        })
    elif slowtick_identical is False:
        checks.append({
            "issue": "slowtick identical across leads",
            "detected": False,
            "status": "pass",
            "evidence": slowtick_note,
        })
    else:
        checks.append({
            "issue": "slowtick identical across leads",
            "detected": None,
            "status": "skip",
            "evidence": slowtick_note or "slowtick data missing",
        })

    if time_shift_result:
        status = time_shift_result.get("status")
        checks.append({
            "issue": "time-shift test",
            "detected": status == "fail",
            "status": "fail" if status == "fail" else "pass",
            "evidence": "performance unchanged under shift" if status == "fail" else "metrics moved under shift",
        })
    else:
        checks.append({
            "issue": "time-shift test",
            "detected": None,
            "status": "skip",
            "evidence": "time-shift test not run",
        })

    # Join audit check
    join_audit_path = project_dir / "join_audit.json"
    fallback_join_audit = Path("results/diagnostics/join_audit.json")
    if not join_audit_path.exists() and fallback_join_audit.exists():
        join_audit_path = fallback_join_audit
        try:
            (project_dir / "join_audit.json").parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fallback_join_audit, project_dir / "join_audit.json")
        except Exception:
            pass
    join_fail = None
    join_note = "join audit missing"
    if join_audit_path.exists():
        try:
            entries = json.loads(join_audit_path.read_text(encoding="utf-8"))
            if isinstance(entries, dict):
                entries = entries.get("entries", [])
            flags = [
                e for e in entries
                if e.get("many_to_many") or e.get("row_explosion")
            ]
            join_fail = len(flags) > 0
            join_note = f"flagged={len(flags)}"
        except Exception:
            join_fail = None
            join_note = "join audit unreadable"
    checks.append({
        "issue": "join audit",
        "detected": join_fail,
        "status": "fail" if join_fail else "pass" if join_fail is False else "skip",
        "evidence": join_note,
    })

    # Build summary
    diag = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "generated_at": _utc_now(),
        "artifacts": artifacts,
        "lead_hashes": lead_hashes,
        "checks": checks,
    }

    out_json = out_dir / "diagnostics.json"
    out_md = out_dir / "diagnostics_summary.md"
    out_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Diagnostics summary for {run_name}")
    lines.append("")
    lines.append("| Issue | Detected? | Status | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    for c in checks:
        detected = c.get("detected")
        det_txt = "yes" if detected is True else "no" if detected is False else "n/a"
        lines.append(f"| {c.get('issue')} | {det_txt} | {c.get('status')} | {c.get('evidence')} |")
    lines.append("")
    lines.append("## Artifacts")
    for name, items in artifacts.items():
        lines.append(f"- {name}: {len(items)} file(s)")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    slowtick_summary = None
    if slowtick_dir and slowtick_dir.exists():
        candidate = slowtick_dir / "slowtick_summary.csv"
        if candidate.exists():
            slowtick_summary = project_dir / "slowtick_by_lead.csv"
            try:
                pd.read_csv(candidate).to_csv(slowtick_summary, index=False)
            except Exception:
                slowtick_summary = None

    report_path = project_dir / "diagnostics_report.md"
    extra_paths = {
        "schema_inventory": project_dir / "schema_inventory.json",
        "lead_feature_fingerprints": project_dir / "lead_feature_fingerprints.json",
        "lead_label_stats": project_dir / "lead_label_stats.parquet"
        if (project_dir / "lead_label_stats.parquet").exists()
        else None,
        "split_overlap": project_dir / "split_overlap.parquet" if (project_dir / "split_overlap.parquet").exists() else None,
        "time_shift_test": project_dir / "time_shift_test.json" if (project_dir / "time_shift_test.json").exists() else None,
        "join_audit": project_dir / "join_audit.json" if (project_dir / "join_audit.json").exists() else None,
        "slowtick_by_lead": slowtick_summary,
        "diagnostics_artifacts": artifacts_dir,
    }
    _write_diagnostics_report(
        report_path,
        run_name=run_name,
        run_dir=run_dir,
        checks=checks,
        artifacts=artifacts,
        extra_paths=extra_paths,
    )

    print(f"[diagnostics] wrote {out_json}")
    print(f"[diagnostics] wrote {out_md}")
    print(f"[diagnostics] wrote {report_path}")
    if slowtick_summary:
        print(f"[diagnostics] wrote {slowtick_summary}")

    if args.fail_on_checks:
        fail_count = sum(1 for c in checks if str(c.get("status")).lower() == "fail")
        if fail_count:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
