#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting_v2.py

Agent: generate a human-readable Markdown + JSON report plus per-storm pages.
Outputs are written into a dated run folder unless --out-dir is provided.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import config_normalize
from utils.run_naming import make_run_dir


def _is_parquet(path: Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq", ".pqt"))


def _read_any(path: Path, columns: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if _is_parquet(path):
        df = pd.read_parquet(path, columns=columns)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, usecols=columns, nrows=nrows, low_memory=False)


def _peek_columns(path: Path) -> List[str]:
    if not path.exists():
        return []
    if _is_parquet(path):
        try:
            import pyarrow.parquet as pq  # type: ignore

            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return []
    try:
        return list(pd.read_csv(path, nrows=0, compression="infer").columns)
    except Exception:
        return []


def _read_text_safe(path: Optional[Path]) -> str:
    if not path:
        return ""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return ""
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _exists_nonempty(path: Optional[Path]) -> bool:
    return bool(path and path.exists() and path.stat().st_size > 0)


def _row_count(path: Path) -> int:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            meta_rows = int(pq.ParquetFile(path).metadata.num_rows)
            if meta_rows:
                return meta_rows
        except Exception:
            pass
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            return int(sum(len(b) for b in pf.iter_batches(batch_size=200_000, columns=[])))
        except Exception:
            return 0
    if ext in {".csv", ".csv.gz"}:
        opener = gzip.open if ext == ".csv.gz" else open
        try:
            row_idx = -1
            with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
                for row_idx, _ in enumerate(fh):
                    pass
            return max(0, row_idx)
        except Exception:
            return 0
    return 0


def _alert_dir_snapshot(alerts_dir: Path) -> Dict[str, Any]:
    alert_files = 0
    rows_sampled = 0
    if alerts_dir.exists():
        alert_paths = list(alerts_dir.glob("alerts_*.csv*")) + list(alerts_dir.glob("alerts_*.parquet"))
        for p in alert_paths:
            alert_files += 1
            try:
                if _is_parquet(p):
                    c = pd.read_parquet(p, columns=None)
                else:
                    c = pd.read_csv(p, nrows=1000, low_memory=False)
                rows_sampled += min(len(c), 1000)
            except Exception:
                continue
    return {
        "alerts_dir": str(alerts_dir),
        "alert_files": alert_files,
        "rows_sampled": rows_sampled,
    }


def _storm_timeseries_snapshot(path: Optional[Path]) -> Dict[str, Any] | None:
    if not _exists_nonempty(path):
        return None
    try:
        sample = _read_any(Path(path), nrows=1000)
    except Exception:
        return None
    if sample is None or sample.empty:
        return None
    time_col = next((c for c in sample.columns if c in ("time", "obs_time", "valid_time")), None)
    id_col = next((c for c in sample.columns if c in ("storm_id", "sid", "name", "storm")), None)
    summary: Dict[str, Any] = {
        "path": str(path),
        "sample_rows": int(len(sample)),
    }
    if id_col:
        summary["unique_storms_sample"] = int(sample[id_col].nunique())
    if time_col:
        t_parsed = pd.to_datetime(sample[time_col], errors="coerce", utc=True).dt.tz_convert(None)
        summary["time_min_sample"] = t_parsed.min()
        summary["time_max_sample"] = t_parsed.max()
    return summary


def _parse_horizons(raw: str) -> List[int]:
    vals: List[int] = []
    try:
        vals = [int(h.strip()) for h in str(raw).split(",") if h.strip()]
    except Exception:
        vals = []
    return vals or [24, 48, 72, 120]


def _viability_conversion_snapshot(path: Optional[Path], horizons: List[int]) -> pd.DataFrame:
    if not _exists_nonempty(path):
        return pd.DataFrame()
    try:
        df = _read_any(Path(path), columns=["t_to_storm_min_h", "y_viable"])
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty or "t_to_storm_min_h" not in df.columns:
        return pd.DataFrame()
    lead_vals = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
    yv = pd.to_numeric(df.get("y_viable", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rows = []
    start = 0.0
    for h in horizons:
        mask = lead_vals.gt(start) & lead_vals.le(float(h))
        total = int(mask.sum())
        subset = yv.loc[mask]
        pos = float(subset.sum()) if total > 0 else 0.0
        rate = float(subset.mean()) if total > 0 else np.nan
        rows.append(
            {
                "horizon": f"({start},{h}]",
                "rows": total,
                "y_viable_mean": rate if pd.notna(rate) else np.nan,
                "positives": pos,
            }
        )
        start = float(h)
    tail_mask = lead_vals.gt(start)
    if tail_mask.any():
        subset = yv.loc[tail_mask]
        total = int(tail_mask.sum())
        pos = float(subset.sum()) if total > 0 else 0.0
        rate = float(subset.mean()) if total > 0 else np.nan
        rows.append(
            {
                "horizon": f"> {start}",
                "rows": total,
                "y_viable_mean": rate if pd.notna(rate) else np.nan,
                "positives": pos,
            }
        )
    return pd.DataFrame(rows)


def _viability_thresholds_snapshot(path: Optional[Path]) -> pd.DataFrame:
    if not _exists_nonempty(path):
        return pd.DataFrame()
    try:
        tbl = _read_any(Path(path))
    except Exception:
        return pd.DataFrame()
    if tbl is None or tbl.empty:
        return pd.DataFrame()
    for cand in ("lead_h", "lead", "lead_hours"):
        if cand in tbl.columns:
            tbl = tbl.rename(columns={cand: "lead_h"})
            break
    keep = [
        c
        for c in [
            "lead_h",
            "thr_Fbeta",
            "thr_fbeta",
            "thr_F1",
            "Fbeta",
            "F1",
            "precision",
            "recall",
            "coverage",
        ]
        if c in tbl.columns
    ]
    return tbl[keep].head(12) if keep else tbl.head(12)


def _viability_sweep_diag(path: Optional[Path]) -> pd.DataFrame:
    if not _exists_nonempty(path):
        return pd.DataFrame()
    try:
        tbl = _read_any(Path(path))
    except Exception:
        return pd.DataFrame()
    if tbl is None or tbl.empty:
        return pd.DataFrame()
    for cand in ("lead_h", "lead", "lead_hours"):
        if cand in tbl.columns:
            tbl = tbl.rename(columns={cand: "lead_h"})
            break
    diag_cols = [
        "lead_h",
        "n_rows",
        "positives",
        "pos_frac",
        "prob_q01",
        "prob_q50",
        "prob_q99",
        "feasible_count",
        "feasible_frac",
    ]
    keep = [c for c in diag_cols if c in tbl.columns]
    return tbl[keep].head(12) if keep else pd.DataFrame()


def _viability_metrics_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    if not _exists_nonempty(path):
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    lines: List[str] = []
    if isinstance(raw, dict):
        for split in ("train", "val", "overall"):
            if split in raw and isinstance(raw[split], dict):
                part = raw[split]
                auc = part.get("roc_auc")
                ap = part.get("avg_precision") or part.get("pr_auc")
                brier = part.get("brier")
                thr = part.get("opt_threshold") or part.get("thr_Fbeta")
                pieces = [f"{split.title():<6}"]
                if auc is not None:
                    pieces.append(f"AUC={float(auc):.3f}")
                if ap is not None:
                    pieces.append(f"PRAUC={float(ap):.3f}")
                if brier is not None:
                    pieces.append(f"Brier={float(brier):.4f}")
                if thr is not None:
                    pieces.append(f"thr={thr}")
                if len(pieces) > 1:
                    lines.append("  " + "  ".join(pieces))
        if not lines:
            base = raw.get("overall", raw)
            for k, v in base.items():
                if isinstance(v, (int, float, str)) and len(lines) < 8:
                    lines.append(f"{k}: {v}")
    return {"lines": lines, "raw": raw}


def _proto_outcomes_snapshot(path: Optional[Path]) -> pd.DataFrame:
    if not _exists_nonempty(path):
        return pd.DataFrame()
    try:
        tbl = _read_any(Path(path), nrows=500)
    except Exception:
        return pd.DataFrame()
    if tbl is None or tbl.empty:
        return pd.DataFrame()
    return tbl.head(20)


def _slowtick_union_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    if not _exists_nonempty(path):
        return {}
    try:
        seeds = _read_any(Path(path))
    except Exception:
        return {}
    if seeds is None or seeds.empty:
        return {}
    prob_col = None
    for c in ("prob_max", "prob_viable", "prob"):
        if c in seeds.columns:
            prob_col = c
            break
    if prob_col is None:
        return {}
    hi = seeds[pd.to_numeric(seeds[prob_col], errors="coerce") >= 0.5]
    if not {"slow_cos", "slow_sin"}.issubset(seeds.columns) or hi.empty:
        return {}
    phase = np.arctan2(
        pd.to_numeric(hi["slow_sin"], errors="coerce"),
        pd.to_numeric(hi["slow_cos"], errors="coerce"),
    )
    phase_hours = (phase % (2 * np.pi)) * 24.0 / (2 * np.pi)
    bins = np.arange(0, 25, 3)
    hist, _ = np.histogram(phase_hours, bins=bins)
    return {
        "high_prob_count": int(len(hi)),
        "phase_mean_h": float(np.nanmean(phase_hours)),
        "phase_std_h": float(np.nanstd(phase_hours)),
        "bins_h": bins.tolist(),
        "hist": hist.tolist(),
    }


def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(raw) or {}
    except Exception:
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}


def _safe_storm_id(val: object) -> str:
    txt = str(val).strip() or "storm"
    out = []
    for ch in txt:
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    return safe or "storm"


def _coerce_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _slowtick_dir(out_dir: Path, tables_dir: Path) -> Optional[Path]:
    for cand in (out_dir / "slowtick", tables_dir / "slowtick"):
        if cand.exists():
            return cand
    return None


def _file_signature(path: Path, max_hash_mb: float = 50.0) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_mb": round(size / 1e6, 2),
        "mtime": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
    }
    if size <= max_hash_mb * 1e6:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data_"
    df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (np.datetime64,)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.timedelta64,)):
        return str(pd.Timedelta(obj))
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def _quantiles(vals: pd.Series, qs: Iterable[float]) -> Dict[str, float]:
    out = {}
    v = pd.to_numeric(vals, errors="coerce").dropna()
    for q in qs:
        out[f"q{int(q * 100):02d}"] = float(v.quantile(q)) if not v.empty else float("nan")
    out["n"] = int(v.size)
    return out


def _haversine_km(lat1: float | np.ndarray, lon1: float | np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dphi = lat2r - lat1r
    dlambda = lon2r - lon1r
    a = np.sin(dphi / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.maximum(a, 0.0)))


def _object_hour_gaps(objects: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Dict[str, Any]:
    if objects.empty or start is None or end is None:
        return {}
    if "time" not in objects.columns:
        return {}
    tmin = pd.to_datetime(start, utc=True, errors="coerce").tz_localize(None)
    tmax = pd.to_datetime(end, utc=True, errors="coerce").tz_localize(None)
    if pd.isna(tmin) or pd.isna(tmax):
        return {}
    times = _coerce_time(objects["time"]).dropna().dt.floor("h")
    if times.empty:
        return {}
    hours_with = pd.DatetimeIndex(sorted(times.unique()))
    full = pd.date_range(tmin.floor("h"), tmax.floor("h"), freq="h")
    missing = full.difference(hours_with)
    gaps: List[Dict[str, Any]] = []
    if not missing.empty:
        start_gap = missing[0]
        prev = missing[0]
        for ts in missing[1:]:
            if ts == prev + pd.Timedelta(hours=1):
                prev = ts
                continue
            gaps.append({"start": start_gap, "end": prev})
            start_gap = ts
            prev = ts
        gaps.append({"start": start_gap, "end": prev})
    gap_rows = []
    for g in gaps:
        gstart = g["start"]
        gend = g["end"]
        length = int((gend - gstart).total_seconds() / 3600.0) + 1
        gap_rows.append(
            {
                "start": gstart,
                "end": gend,
                "hours": length,
            }
        )
    gap_rows = sorted(gap_rows, key=lambda x: x["hours"], reverse=True)
    return {
        "hours_total": int(len(full)),
        "hours_with_objects": int(len(hours_with)),
        "hours_missing_objects": int(max(0, len(full) - len(hours_with))),
        "missing_ranges": gap_rows[:5],
    }


def _pick_track_cols(df: pd.DataFrame) -> Dict[str, str]:
    def _first(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    time_col = _first(["time", "obs_time", "datetime", "valid_time"])
    lat_col = _first(["lat", "latitude"])
    lon_col = _first(["lon", "longitude"])
    id_col = _first(["storm_id", "sid", "name"])
    if not time_col or not lat_col or not lon_col:
        return {}
    return {"time": time_col, "lat": lat_col, "lon": lon_col, "id": id_col or "storm_id"}


def _lead_bin_table(vals: pd.Series, bins: List[float], labels: List[str], positive_only: bool = False) -> pd.DataFrame:
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if positive_only:
        v = v[v >= 0]
    if v.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"lead_h": v})
    df["lead_bin"] = pd.cut(df["lead_h"], bins=bins, labels=labels, include_lowest=True)
    out = df.groupby("lead_bin")["lead_h"].agg(["count", "median", "mean"]).reset_index()
    return out


def _lead_metrics_from_objects(
    objects: pd.DataFrame,
    tracks: pd.DataFrame,
    radius_km: float,
    cat1_threshold: float,
) -> Dict[str, Any]:
    if objects.empty or tracks.empty:
        return {}
    obj = objects.copy()
    time_col = "time" if "time" in obj.columns else None
    if time_col is None:
        return {}
    if {"obj_centroid_lat", "obj_centroid_lon"}.issubset(obj.columns):
        lat_col, lon_col = "obj_centroid_lat", "obj_centroid_lon"
    elif {"lat", "lon"}.issubset(obj.columns):
        lat_col, lon_col = "lat", "lon"
    else:
        return {}
    obj["time"] = _coerce_time(obj[time_col])
    obj["lat"] = pd.to_numeric(obj[lat_col], errors="coerce")
    obj["lon"] = pd.to_numeric(obj[lon_col], errors="coerce")
    obj = obj.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
    if obj.empty:
        return {}

    tr = tracks.copy()
    colmap = _pick_track_cols(tr)
    if not colmap:
        return {}
    tr = tr.rename(columns={colmap["time"]: "time", colmap["lat"]: "lat", colmap["lon"]: "lon"})
    if colmap["id"] in tr.columns:
        tr = tr.rename(columns={colmap["id"]: "storm_id"})
    elif "storm_id" not in tr.columns:
        tr["storm_id"] = "storm"
    tr["time"] = _coerce_time(tr["time"])
    tr["lat"] = pd.to_numeric(tr["lat"], errors="coerce")
    tr["lon"] = pd.to_numeric(tr["lon"], errors="coerce")
    tr = tr.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
    if tr.empty:
        return {}

    vmax_col = None
    for cand in ["vmax", "usa_wind", "wind", "maxwind", "vmax_kts", "vmax_kt"]:
        if cand in tr.columns:
            vmax_col = cand
            break

    rows: List[Dict[str, Any]] = []
    for storm_id, t in tr.groupby("storm_id"):
        t = t.sort_values("time").reset_index(drop=True)
        times = t["time"].to_numpy()
        lats = t["lat"].to_numpy(dtype=float)
        lons = t["lon"].to_numpy(dtype=float)
        if not len(times):
            continue
        genesis = t["time"].min()
        cat1_time = None
        if vmax_col:
            vmax_vals = pd.to_numeric(t[vmax_col], errors="coerce")
            hit = vmax_vals >= float(cat1_threshold)
            if hit.any():
                cat1_time = t.loc[hit, "time"].min()
        lat_mid = float(np.nanmean(lats)) if np.isfinite(lats).any() else 0.0
        lat_pad = radius_km / 111.0
        lon_pad = radius_km / max(1e-6, 111.0 * math.cos(math.radians(lat_mid)))
        lat_min = float(np.nanmin(lats)) - lat_pad
        lat_max = float(np.nanmax(lats)) + lat_pad
        lon_min = float(np.nanmin(lons)) - lon_pad
        lon_max = float(np.nanmax(lons)) + lon_pad
        obj_sub = obj.loc[
            (obj["lat"] >= lat_min)
            & (obj["lat"] <= lat_max)
            & (obj["lon"] >= lon_min)
            & (obj["lon"] <= lon_max)
            & (obj["time"] <= t["time"].max())
        ]
        if obj_sub.empty:
            continue
        for row in obj_sub.itertuples(index=False):
            t0 = row.time
            if pd.isna(t0):
                continue
            mask = times >= np.datetime64(t0)
            if not mask.any():
                continue
            tf = times[mask]
            lf = lats[mask]
            lof = lons[mask]
            dists = _haversine_km(float(row.lat), float(row.lon), lf, lof)
            if dists.size == 0:
                continue
            idx = int(np.nanargmin(dists))
            t_star = tf[idx]
            min_dist = float(dists[idx])
            lead_closest = float((pd.Timestamp(t_star) - t0).total_seconds() / 3600.0)
            lead_radius = float("nan")
            within = dists <= float(radius_km)
            if within.any():
                j = int(np.argmax(within))
                lead_radius = float((pd.Timestamp(tf[j]) - t0).total_seconds() / 3600.0)
            lead_genesis = float((genesis - t0).total_seconds() / 3600.0) if pd.notna(genesis) else float("nan")
            lead_cat1 = float((cat1_time - t0).total_seconds() / 3600.0) if pd.notna(cat1_time) else float("nan")
            rows.append(
                {
                    "storm_id": storm_id,
                    "object_time": t0,
                    "object_lat": float(row.lat),
                    "object_lon": float(row.lon),
                    "lead_to_closest_h": lead_closest,
                    "lead_to_radius_h": lead_radius,
                    "lead_to_genesis_h": lead_genesis,
                    "lead_to_cat1_h": lead_cat1,
                    "min_dist_km": min_dist,
                }
            )
    if not rows:
        return {}
    lead_df = pd.DataFrame(rows)
    bins = [-1e9, 0, 24, 48, 72, 120, 240, 1e9]
    labels = ["<=0", "0-24", "24-48", "48-72", "72-120", "120-240", ">240"]
    summary = {
        "closest_bins": _lead_bin_table(lead_df["lead_to_closest_h"], bins, labels, positive_only=True),
        "radius_bins": _lead_bin_table(lead_df["lead_to_radius_h"], bins, labels, positive_only=True),
        "genesis_bins": _lead_bin_table(lead_df["lead_to_genesis_h"], bins, labels, positive_only=True),
        "cat1_bins": _lead_bin_table(lead_df["lead_to_cat1_h"], bins, labels, positive_only=True),
        "n_objects": int(len(lead_df)),
    }
    return summary


def _alert_stats(alerts: pd.DataFrame, flag_col: str, time_col: Optional[str] = None) -> Dict[str, Any]:
    if alerts.empty or flag_col not in alerts.columns:
        return {}
    if time_col is None:
        time_col = "time"
        if time_col not in alerts.columns:
            for cand in ["valid_time", "datetime", "forecast_time"]:
                if cand in alerts.columns:
                    time_col = cand
                    break
    if time_col not in alerts.columns:
        return {}
    time_vals = _coerce_time(alerts[time_col])
    valid = time_vals.notna()
    if not valid.any():
        return {"hours": 0, "alerts": 0}
    flags = pd.to_numeric(alerts[flag_col], errors="coerce").fillna(0).astype(int)
    flags = flags.loc[valid]
    time_h = time_vals.loc[valid].dt.floor("h")
    mask = flags > 0
    if mask.any():
        per_hour = time_h.loc[mask].value_counts()
    else:
        per_hour = pd.Series(dtype=int)
    if per_hour.empty:
        return {"hours": int(time_h.nunique()), "alerts": 0}
    top_hours = {}
    for k, v in per_hour.sort_values(ascending=False).head(5).to_dict().items():
        if isinstance(k, pd.Timestamp) and pd.notna(k):
            key = k.isoformat()
        else:
            key = str(k)
        top_hours[key] = int(v)
    stats = {
        "hours": int(time_h.nunique()),
        "alerts": int(flags.sum()),
        "per_hour_q50": float(per_hour.quantile(0.50)),
        "per_hour_q90": float(per_hour.quantile(0.90)),
        "per_hour_q99": float(per_hour.quantile(0.99)),
        "top_hours": top_hours,
    }
    return stats


def _alert_stats_from_path(path: Path, flag_col: str) -> Dict[str, Any]:
    cols = _peek_columns(path)
    if not cols:
        return {}
    time_col = next((c for c in ("time", "valid_time", "datetime", "forecast_time") if c in cols), None)
    if time_col is None or flag_col not in cols:
        return {}
    try:
        df = _read_any(path, columns=[time_col, flag_col])
    except Exception:
        return {}
    if df.empty:
        return {}
    return _alert_stats(df, flag_col, time_col=time_col)


def _object_stats(objects: pd.DataFrame) -> Dict[str, Any]:
    if objects.empty or "time" not in objects.columns:
        return {}
    df = objects.copy()
    df["time"] = _coerce_time(df["time"])
    df = df.dropna(subset=["time"])
    df["time_h"] = df["time"].dt.floor("h")
    per_hour = df.groupby("time_h")["object_id"].nunique()
    area_q = _quantiles(df.get("obj_area_cells", pd.Series(dtype=float)), [0.05, 0.5, 0.95])
    compact = None
    if "obj_score_topk_mean" in df.columns and "obj_score_mean" in df.columns:
        mean = pd.to_numeric(df["obj_score_mean"], errors="coerce")
        topk = pd.to_numeric(df["obj_score_topk_mean"], errors="coerce")
        comp = topk / mean.replace(0, np.nan)
        compact = _quantiles(comp, [0.05, 0.5, 0.95])
    return {
        "hours": int(df["time_h"].nunique()),
        "objects": int(df["object_id"].nunique()),
        "objects_per_hour_q50": float(per_hour.quantile(0.50)) if not per_hour.empty else 0.0,
        "objects_per_hour_q90": float(per_hour.quantile(0.90)) if not per_hour.empty else 0.0,
        "area_cells": area_q,
        "compactness": compact or {},
    }


def _match_skill(matches: pd.DataFrame) -> Dict[str, Any]:
    if matches.empty:
        return {}
    df = matches.copy()
    df["track_time"] = _coerce_time(df["track_time"])
    df["object_time"] = _coerce_time(df["object_time"])
    df = df.dropna(subset=["track_time", "object_time"])
    df["lead_h"] = (df["track_time"] - df["object_time"]).dt.total_seconds() / 3600.0
    dist = pd.to_numeric(df.get("d_km", pd.Series(dtype=float)), errors="coerce")
    lead_bins = [-1e9, 0, 24, 48, 72, 120, 240, 1e9]
    labels = ["<=0", "0-24", "24-48", "48-72", "72-120", "120-240", ">240"]
    df["lead_bin"] = pd.cut(df["lead_h"], bins=lead_bins, labels=labels)
    lead_stats = df.groupby("lead_bin")["d_km"].agg(["count", "median", "mean"]).reset_index()
    score = pd.to_numeric(df.get("match_score", pd.Series(dtype=float)), errors="coerce")
    thresholds = [float(score.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95) if score.notna().any()]
    cov_prec = []
    total_tracks = df["track_time"].nunique()
    for thr in thresholds:
        sub = df.loc[score >= thr]
        close = sub.loc[pd.to_numeric(sub.get("d_km", pd.Series(dtype=float)), errors="coerce") <= 50.0]
        coverage = close["track_time"].nunique() / total_tracks if total_tracks else 0.0
        precision = close.shape[0] / sub.shape[0] if len(sub) else 0.0
        cov_prec.append({"threshold": thr, "coverage": coverage, "precision": precision, "n": int(len(sub))})
    return {
        "distance_quantiles": _quantiles(dist, [0.05, 0.5, 0.95]),
        "lead_bins": lead_stats,
        "coverage_precision": cov_prec,
    }


def _storm_pages(
    matches: pd.DataFrame,
    tracks: pd.DataFrame,
    out_dir: Path,
    match_top_n: int,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    if tracks.empty:
        return summaries
    tracks = tracks.copy()
    if "storm_id" not in tracks.columns and "sid" in tracks.columns:
        tracks = tracks.rename(columns={"sid": "storm_id"})
    if "storm_id" not in tracks.columns:
        tracks["storm_id"] = "storm"
    tracks["time"] = _coerce_time(tracks["time"])
    tracks = tracks.dropna(subset=["time"])
    matches = matches.copy()
    if not matches.empty:
        matches["track_time"] = _coerce_time(matches["track_time"])
        matches["object_time"] = _coerce_time(matches["object_time"])
    for storm_id, t in tracks.groupby("storm_id"):
        t = t.sort_values("time")
        m = matches.loc[matches["storm_id"] == storm_id].copy() if not matches.empty else pd.DataFrame()
        if "match_rank" in m.columns:
            m = m.loc[pd.to_numeric(m["match_rank"], errors="coerce") <= int(match_top_n)]
        genesis = t["time"].min()
        end_time = t["time"].max()
        duration_h = (end_time - genesis).total_seconds() / 3600.0 if pd.notna(genesis) and pd.notna(end_time) else None

        lead_hist = ""
        dir_summary = ""
        if not m.empty:
            lead = (m["track_time"] - m["object_time"]).dt.total_seconds() / 3600.0
            bins = [0, 24, 48, 72, 120, 240, 1e9]
            hist, edges = np.histogram(lead.dropna(), bins=bins)
            lead_lines = []
            for i in range(len(hist)):
                lead_lines.append(f"{int(edges[i])}-{int(edges[i+1])}h: {int(hist[i])}")
            lead_hist = ", ".join(lead_lines)
            if "track_bearing_deg" in m.columns and "obj_motion_bearing_deg" in m.columns:
                a = pd.to_numeric(m["track_bearing_deg"], errors="coerce").to_numpy()
                b = pd.to_numeric(m["obj_motion_bearing_deg"], errors="coerce").to_numpy()
                diff = ((a - b + 180.0) % 360.0) - 180.0
                diff = diff[np.isfinite(diff)]
                if diff.size:
                    dir_summary = f"Median |bearing diff|: {float(np.nanmedian(np.abs(diff))):.1f}°"

        storm_md = [
            f"# Storm {storm_id}",
            "",
            f"- Track points: {int(t['time'].nunique())}",
            f"- Time span: {genesis} -> {end_time}",
            f"- Duration (h): {duration_h:.1f}" if duration_h is not None else "- Duration (h): n/a",
            f"- Matches (top {match_top_n}): {int(len(m))}" if not m.empty else f"- Matches (top {match_top_n}): 0",
        ]
        if lead_hist:
            storm_md.append(f"- Lead histogram: {lead_hist}")
        if dir_summary:
            storm_md.append(f"- Direction comparison: {dir_summary}")
        out_path = out_dir / f"{_safe_storm_id(storm_id)}.md"
        out_path.write_text("\n".join(storm_md) + "\n", encoding="utf-8")
        summaries.append(
            {
                "storm_id": str(storm_id),
                "track_points": int(t["time"].nunique()),
                "matches": int(len(m)),
                "duration_h": float(duration_h) if duration_h is not None else None,
                "page": str(out_path),
            }
        )
    return summaries


def _weird_correlations(tables_dir: Path) -> pd.DataFrame:
    corr = tables_dir / "correlations_overall.csv"
    if corr.exists():
        df = pd.read_csv(corr)
        if df.empty or "feature" not in df.columns:
            return pd.DataFrame()
        patterns = ["SFI", "knee", "shear", "CAPE", "cape", "pdrop", "radial", "sph_"]
        mask = df["feature"].astype(str).apply(lambda x: any(p in x for p in patterns))
        df = df.loc[mask].copy() if mask.any() else df.copy()
        if "spearman" in df.columns:
            df = df.assign(_abs=df["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
        return df.head(20)

    assoc = tables_dir / "feature_associations.parquet"
    if not assoc.exists():
        return pd.DataFrame()
    df = pd.read_parquet(assoc)
    if df.empty or "feature" not in df.columns:
        return pd.DataFrame()
    patterns = ["SFI", "knee", "shear", "CAPE", "cape", "pdrop", "radial", "sph_"]
    mask = df["feature"].astype(str).apply(lambda x: any(p in x for p in patterns))
    if not mask.any():
        return df.head(0)
    df = df.loc[mask].copy()
    if "spearman" in df.columns:
        df = df.assign(_abs=df["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
    return df.head(20)


def _read_blocked(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        blocked = data.get("blocked", [])
        return blocked if isinstance(blocked, list) else []
    except Exception:
        return []


def _read_stage_manifest(run_name: str) -> pd.DataFrame:
    safe_run = str(run_name).strip().replace(" ", "_")
    path = Path("results/runs") / safe_run / "manifests" / "stages.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    stages = data.get("stages", {})
    rows: List[Dict[str, Any]] = []
    for stage, info in stages.items():
        outputs = info.get("outputs", []) if isinstance(info, dict) else []
        output_paths = "; ".join([o.get("path", "") for o in outputs if o.get("path")])
        rows_vals = [o.get("rows") for o in outputs if isinstance(o, dict) and o.get("rows") is not None]
        rows_val = int(rows_vals[0]) if rows_vals else None
        rows.append(
            {
                "stage": stage,
                "status": info.get("status"),
                "input_fingerprint": info.get("input_fingerprint"),
                "output_path": output_paths,
                "rows": rows_val,
                "health_ok": info.get("health_ok"),
            }
        )
    return pd.DataFrame(rows)


def _track_objects_from_manifest(stage_manifest: pd.DataFrame) -> Dict[str, Any]:
    if stage_manifest.empty or "stage" not in stage_manifest.columns:
        return {}
    row = stage_manifest.loc[stage_manifest["stage"] == "training.track-objects"]
    if row.empty:
        return {}
    info = row.iloc[0].to_dict()
    out_paths = str(info.get("output_path") or "")
    pick = ""
    for part in out_paths.split(";"):
        cand = part.strip()
        if not cand:
            continue
        if "objects" in cand and cand.endswith((".parquet", ".csv", ".csv.gz")):
            pick = cand
            if cand.endswith("objects.parquet") or cand.endswith("objects.csv") or cand.endswith("objects.csv.gz"):
                break
    return {
        "path": pick or out_paths,
        "rows": info.get("rows"),
        "health_ok": info.get("health_ok"),
    }


def _collect_model_paths(cfg: Any, exts: Tuple[str, ...]) -> List[Path]:
    found: List[Path] = []

    def _walk(obj: Any):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, str):
            p = Path(obj)
            if p.suffix.lower() in exts:
                found.append(p)

    _walk(cfg)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Markdown/JSON reporting pack for a run.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default=None, help="Output folder for MD/JSON (default: dated run folder).")
    ap.add_argument("--run-root", default=None, help="Root for dated run folders (default: results/reports).")
    ap.add_argument("--run-date", default=None, help="Override YYYYMMDD for run folder.")
    ap.add_argument("--tables-dir", default=None, help="Report-pack tables directory.")
    ap.add_argument("--config", default=None, help="Optional pipeline YAML for run summary.")
    ap.add_argument("--alerts", default=None, help="Alerts table for alert stats.")
    ap.add_argument("--objects", default=None, help="Objects-by-hour table for object stats.")
    ap.add_argument("--matches", default=None, help="Object matches table for skill stats.")
    ap.add_argument("--tracks", default=None, help="IBTrACS tracks table for storm pages.")
    ap.add_argument("--blocked", default=None, help="Optional blocked JSON emitted by report preflight.")
    ap.add_argument("--match-top-n", type=int, default=1, help="Top-N matches per hour to consider.")
    ap.add_argument("--alert-flag-col", default="alert_final")
    ap.add_argument("--lead-radius-km", type=float, default=150.0, help="Radius for lead-to-radius metrics.")
    ap.add_argument("--cat1-threshold", type=float, default=64.0, help="Track vmax threshold for Cat1+ timing.")
    ap.add_argument("--seed-summary", default=None, help="Optional seed summary text to embed.")
    ap.add_argument("--seed-analysis", default=None, help="Optional seed analysis text to embed.")
    ap.add_argument("--alerts-dir", default=None, help="Optional alerts directory for snapshot counts.")
    ap.add_argument(
        "--conversion-csv",
        "--include-conversion",
        dest="conversion_csv",
        default=None,
        help="Optional conversion/proto-outcomes CSV to embed.",
    )
    ap.add_argument(
        "--viability-targets",
        default="data/grid_train_gse_panel_targets.parquet",
        help="Panel with t_to_storm_min_h / y_viable for conversion snapshot.",
    )
    ap.add_argument(
        "--viability-metrics",
        default="models/viability_model_metrics.json",
        help="Viability model metrics JSON (optional).",
    )
    ap.add_argument(
        "--viability-thresholds",
        default="results/sweeps/viability_best_thresholds.csv",
        help="Per-lead viability thresholds CSV (optional).",
    )
    ap.add_argument("--storm-timeseries", default=None, help="Optional storm-centric time-series panel.")
    ap.add_argument("--seed-union", default=None, help="Optional seed union table for slow-tick snapshot.")
    ap.add_argument(
        "--viability-horizons",
        default="24,48,72,120",
        help="Comma-separated horizons (hours) for conversion snapshot.",
    )
    ap.add_argument("--ibtracs", default=None, help="Optional IBTrACS subset for reference.")
    ap.add_argument("--ibtracs-area", default=None, help="Optional AOI string used for IBTrACS overlays.")
    ap.add_argument(
        "--ibtracs-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Longitude frame used for IBTrACS overlays.",
    )
    ap.add_argument(
        "--extras",
        default="{}",
        help='JSON dict of extra label->path entries to surface in the report.',
    )
    ap.add_argument("--write-txt", action="store_true", help="Also write a .txt report copy.")
    ap.add_argument("--txt-only", action="store_true", help="Write .txt only (skip MD/JSON).")
    argv = []
    skip = False
    raw = sys.argv[1:]
    for i, tok in enumerate(raw):
        if skip:
            skip = False
            continue
        if tok == "--ibtracs-normalize-lon" and i + 1 < len(raw):
            argv.append(f"--ibtracs-normalize-lon={raw[i+1].strip()}")
            skip = True
        elif tok.startswith("--ibtracs-normalize-lon="):
            lhs, rhs = tok.split("=", 1)
            argv.append(f"{lhs}={rhs.strip()}")
        else:
            argv.append(tok)
    args = ap.parse_args(argv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(args.run_root) if args.run_root else Path("results/reports")
        out_dir = make_run_dir(root, args.run_date)

    safe_run = args.run_name.strip().replace(" ", "_")
    tables_dir = Path(args.tables_dir) if args.tables_dir else (out_dir / f"{safe_run}_tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{safe_run}_report.md"
    json_path = out_dir / f"{safe_run}_report.json"
    storms_dir = out_dir / "storms"

    # Resolve inputs
    alerts_path = None
    if args.alerts:
        alerts_path = Path(args.alerts)
    else:
        alerts_path = _first_existing(
            [
                Path(f"results/alerts/alerts_{args.run_name}_final.parquet"),
                Path(f"results/alerts/alerts_{args.run_name}_final.csv.gz"),
                Path(f"results/alerts/alerts_{args.run_name}_final.csv"),
            ]
        )

    objects_path = Path(args.objects) if args.objects else _first_existing([Path("results/objects/objects_by_hour.parquet")])
    matches_path = Path(args.matches) if args.matches else _first_existing([Path("results/matches/storm_object_matches.parquet")])
    tracks_path = Path(args.tracks) if args.tracks else _first_existing([Path("data/tracks/tracks_subset.csv")])

    cfg_path = Path(args.config) if args.config else _first_existing([Path("config/pipeline.yaml")])
    cfg = _read_yaml_or_json(cfg_path) if cfg_path else {}
    cfg, _ = config_normalize.normalize_config(cfg)

    blocked_path = Path(args.blocked) if args.blocked else None
    if blocked_path is None:
        cand = tables_dir / "blocked.json"
        if cand.exists():
            blocked_path = cand
        else:
            cand = out_dir / "blocked.json"
            if cand.exists():
                blocked_path = cand
    blocked_entries = _read_blocked(blocked_path)

    seed_summary_path = Path(args.seed_summary) if args.seed_summary else None
    if seed_summary_path is None and args.run_name:
        cand = Path(f"results/seedmaps/{args.run_name}_seed_summary.txt")
        if cand.exists():
            seed_summary_path = cand
    seed_analysis_path = Path(args.seed_analysis) if args.seed_analysis else None
    if seed_analysis_path is None and args.run_name:
        cand = Path(f"results/seedmaps/{args.run_name}_seed_analysis.txt")
        if cand.exists():
            seed_analysis_path = cand
    conversion_csv_path = Path(args.conversion_csv) if args.conversion_csv else None
    if conversion_csv_path is None and args.run_name:
        cand = Path(f"results/seedmaps/{args.run_name}_conversion_rates.csv")
        if cand.exists():
            conversion_csv_path = cand
    seed_union_path = Path(args.seed_union) if args.seed_union else None
    if seed_union_path is None and args.run_name:
        for cand in (
            Path(f"results/seedmaps/{args.run_name}_union_byhour.parquet"),
            Path(f"results/seedmaps/{args.run_name}_union_byhour.csv"),
        ):
            if cand.exists():
                seed_union_path = cand
                break

    seed_summary_text = _read_text_safe(seed_summary_path)
    seed_analysis_text = _read_text_safe(seed_analysis_path)

    alerts_snapshot = None
    if args.alerts_dir:
        alerts_snapshot = _alert_dir_snapshot(Path(args.alerts_dir))

    storm_ts_snapshot = _storm_timeseries_snapshot(Path(args.storm_timeseries)) if args.storm_timeseries else None

    horizons = _parse_horizons(args.viability_horizons)
    viability_conversion = _viability_conversion_snapshot(Path(args.viability_targets), horizons) if args.viability_targets else pd.DataFrame()
    viability_thresholds = _viability_thresholds_snapshot(Path(args.viability_thresholds)) if args.viability_thresholds else pd.DataFrame()
    viability_sweep_diag = _viability_sweep_diag(Path(args.viability_thresholds)) if args.viability_thresholds else pd.DataFrame()
    viability_metrics = _viability_metrics_snapshot(Path(args.viability_metrics)) if args.viability_metrics else {}
    proto_outcomes = _proto_outcomes_snapshot(conversion_csv_path)
    slowtick_union = _slowtick_union_snapshot(seed_union_path)

    ibtracs_ref_path = Path(args.ibtracs) if args.ibtracs else None
    if ibtracs_ref_path is None and args.tracks:
        ibtracs_ref_path = Path(args.tracks)
    ibtracs_ref = None
    if _exists_nonempty(ibtracs_ref_path):
        size_mb = ibtracs_ref_path.stat().st_size / 1e6
        rows = _row_count(ibtracs_ref_path)
        ibtracs_ref = {
            "path": str(ibtracs_ref_path),
            "rows": rows,
            "size_mb": round(size_mb, 1),
        }
    elif ibtracs_ref_path:
        ibtracs_ref = {"path": str(ibtracs_ref_path), "rows": 0, "size_mb": 0.0}

    try:
        extras = json.loads(args.extras) if args.extras else {}
        if not isinstance(extras, dict):
            extras = {}
    except Exception:
        extras = {}

    # Run summary
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
    run_summary = {
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "time_span": {
            "start": defaults.get("start"),
            "end": defaults.get("end"),
        },
        "area": defaults.get("area"),
        "normalize_lon": defaults.get("normalize_lon"),
        "table_format": cfg.get("table_format"),
    }

    # Model signatures (best effort)
    model_paths_raw = _collect_model_paths(cfg, (".pkl", ".joblib", ".json"))
    seen = set()
    model_paths: List[Path] = []
    for p in model_paths_raw:
        if str(p) in seen:
            continue
        seen.add(str(p))
        model_paths.append(p)
    run_summary["model_signatures"] = [_file_signature(p) for p in model_paths]

    # Data health
    run_health_path = tables_dir / "run_health.parquet"
    run_health = _read_any(run_health_path) if run_health_path.exists() else pd.DataFrame()
    if not run_health.empty and {"time_min", "time_max"}.issubset(run_health.columns):
        tmin = pd.to_datetime(run_health["time_min"], errors="coerce").min()
        tmax = pd.to_datetime(run_health["time_max"], errors="coerce").max()
        if pd.notna(tmin) or pd.notna(tmax):
            run_summary["time_span"] = {
                "start": tmin.isoformat() if pd.notna(tmin) else run_summary["time_span"].get("start"),
                "end": tmax.isoformat() if pd.notna(tmax) else run_summary["time_span"].get("end"),
            }
    run_start = pd.to_datetime(run_summary["time_span"].get("start"), errors="coerce")
    run_end = pd.to_datetime(run_summary["time_span"].get("end"), errors="coerce")

    # Alert stats
    alert_stats = _alert_stats_from_path(alerts_path, args.alert_flag_col) if alerts_path else {}

    # Object stats
    objects = _read_any(objects_path) if objects_path else pd.DataFrame()
    objects_sig = _file_signature(objects_path) if objects_path else {}
    if objects_path and objects_path.exists():
        objects_sig["rows"] = _row_count(objects_path)
    obj_stats = _object_stats(objects) if not objects.empty else {}
    obj_hours = _object_hour_gaps(objects, run_start, run_end) if not objects.empty else {}

    frag_df = pd.DataFrame()
    hourly_obj = pd.DataFrame()
    rejects_df = pd.DataFrame()
    frag_path = tables_dir / "object_fragmentation.parquet"
    if frag_path.exists():
        frag_df = _read_any(frag_path)
    hourly_path = tables_dir / "object_hourly_stats.parquet"
    if hourly_path.exists():
        hourly_obj = _read_any(hourly_path)
    rejects_path = tables_dir / "object_rejects_by_hour.parquet"
    if rejects_path.exists():
        rejects_df = _read_any(rejects_path)

    storm_hourly = pd.DataFrame()
    storm_hourly_path = tables_dir / "seed_storm_hourly_counts.csv"
    if storm_hourly_path.exists():
        storm_hourly = _read_any(storm_hourly_path)

    # Match skill
    matches = _read_any(matches_path) if matches_path else pd.DataFrame()
    match_skill = _match_skill(matches) if not matches.empty else {}

    # Tracks (for storm pages)
    tracks = _read_any(tracks_path) if tracks_path else pd.DataFrame()
    storm_pages = _storm_pages(matches, tracks, storms_dir, args.match_top_n) if not tracks.empty else []
    lead_metrics = _lead_metrics_from_objects(objects, tracks, float(args.lead_radius_km), float(args.cat1_threshold)) if not objects.empty and not tracks.empty else {}

    # Correlations
    corr_overall = pd.DataFrame()
    corr_overall_path = tables_dir / "correlations_overall.csv"
    if corr_overall_path.exists():
        corr_overall = _read_any(corr_overall_path)
    corr_df = _weird_correlations(tables_dir)

    # Viability skill-by-lead (from report_pack)
    skill_by_lead = pd.DataFrame()
    skill_path = tables_dir / "skill_by_lead.parquet"
    if not skill_path.exists():
        skill_path = tables_dir / "skill_by_lead.csv"
    if skill_path.exists():
        skill_by_lead = _read_any(skill_path)

    # Agent: surface slow-tick diagnostics in the report without feeding any model logic.
    slowtick_summary = pd.DataFrame()
    slowtick_spectrum = pd.DataFrame()
    slowtick_knee = pd.DataFrame()
    slowtick_parity = pd.DataFrame()
    slowtick_dir = _slowtick_dir(out_dir, tables_dir)
    if slowtick_dir:
        summary_path = slowtick_dir / "slowtick_summary.csv"
        if summary_path.exists():
            slowtick_summary = _read_any(summary_path)
        spectrum_path = slowtick_dir / "spectrum_summary.csv"
        if spectrum_path.exists():
            slowtick_spectrum = _read_any(spectrum_path)
            if not slowtick_spectrum.empty and "peak_freq" in slowtick_spectrum.columns:
                freq = pd.to_numeric(slowtick_spectrum["peak_freq"], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    slowtick_spectrum = slowtick_spectrum.assign(
                        peak_period_h=np.where(freq > 0, 1.0 / freq, np.nan),
                    )
        knee_path = slowtick_dir / "knee_fit.csv"
        if knee_path.exists():
            slowtick_knee = _read_any(knee_path)
        parity_path = slowtick_dir / "parity_summary.csv"
        if parity_path.exists():
            slowtick_parity = _read_any(parity_path)

    # Markdown report
    lines: List[str] = []
    lines.append(f"# Run Report - {args.run_name}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Output folder: {out_dir}")
    lines.append(f"- Time span: {run_summary['time_span'].get('start')} -> {run_summary['time_span'].get('end')}")
    lines.append(f"- AOI: {run_summary.get('area')}")
    lines.append(f"- Lon normalize: {run_summary.get('normalize_lon')}")
    if run_summary.get("table_format"):
        lines.append(f"- Table format: {run_summary.get('table_format')}")
    if run_summary.get("model_signatures"):
        lines.append("- Model artifacts:")
        for sig in run_summary["model_signatures"]:
            if not sig.get("exists"):
                lines.append(f"  - {sig.get('path')} (missing)")
            else:
                sha = sig.get("sha256")
                sha_txt = f" sha256={sha[:12]}..." if sha else ""
                lines.append(f"  - {sig.get('path')} ({sig.get('size_mb')} MB){sha_txt}")

    stage_manifest = _read_stage_manifest(args.run_name)
    if not stage_manifest.empty and "stage" in stage_manifest.columns:
        stage_manifest = stage_manifest.sort_values("stage")
    track_objects_info = _track_objects_from_manifest(stage_manifest)

    lines.append("")
    lines.append("## Stage Cache Summary")
    if not stage_manifest.empty:
        cols = [
            c
            for c in [
                "stage",
                "status",
                "input_fingerprint",
                "output_path",
                "rows",
                "health_ok",
            ]
            if c in stage_manifest.columns
        ]
        lines.append(_markdown_table(stage_manifest[cols], max_rows=50))
    else:
        lines.append("_No stage manifest found._")

    if seed_summary_text:
        lines.append("")
        lines.append("## Seed Summary")
        lines.append("```text")
        lines.extend(seed_summary_text.strip().splitlines())
        lines.append("```")

    if seed_analysis_text:
        lines.append("")
        lines.append("## Seed Analysis Notes")
        lines.append("```text")
        lines.extend(seed_analysis_text.strip().splitlines())
        lines.append("```")

    if alerts_snapshot:
        lines.append("")
        lines.append("## Alerts Snapshot")
        lines.append(f"- Alerts directory: {alerts_snapshot.get('alerts_dir')}")
        lines.append(f"- Alert files seen: {alerts_snapshot.get('alert_files')}")
        lines.append(f"- Sampled rows read: {alerts_snapshot.get('rows_sampled')}")

    if ibtracs_ref:
        lines.append("")
        lines.append("## IBTrACS Reference")
        lines.append(f"- Path: {ibtracs_ref.get('path')}")
        lines.append(f"- Rows: {ibtracs_ref.get('rows')}")
        lines.append(f"- Size MB: {ibtracs_ref.get('size_mb')}")
        if args.ibtracs_area:
            lines.append(f"- AOI: {args.ibtracs_area}")
        if args.ibtracs_normalize_lon:
            lines.append(f"- Lon frame: {args.ibtracs_normalize_lon}")

    if storm_ts_snapshot:
        lines.append("")
        lines.append("## Storm Time-Series Snapshot")
        lines.append(f"- Path: {storm_ts_snapshot.get('path')}")
        lines.append(f"- Sample rows: {storm_ts_snapshot.get('sample_rows')}")
        if storm_ts_snapshot.get("unique_storms_sample") is not None:
            lines.append(f"- Unique storms (sample): {storm_ts_snapshot.get('unique_storms_sample')}")
        if storm_ts_snapshot.get("time_min_sample") is not None or storm_ts_snapshot.get("time_max_sample") is not None:
            lines.append(
                f"- Time span (sample): {storm_ts_snapshot.get('time_min_sample')} -> {storm_ts_snapshot.get('time_max_sample')}"
            )

    if not viability_conversion.empty:
        lines.append("")
        lines.append("## Viability Conversion Snapshot (t_to_storm_min_h)")
        lines.append(_markdown_table(viability_conversion, max_rows=20))

    if not viability_thresholds.empty:
        lines.append("")
        lines.append("## Viability Thresholds (sweep)")
        lines.append(_markdown_table(viability_thresholds, max_rows=20))
    if not viability_sweep_diag.empty:
        lines.append("")
        lines.append("## Viability Sweep Diagnostics")
        lines.append(_markdown_table(viability_sweep_diag, max_rows=20))

    if viability_metrics:
        lines.append("")
        lines.append("## Viability Metrics")
        metric_lines = viability_metrics.get("lines") or []
        if metric_lines:
            lines.append("```text")
            lines.extend(metric_lines)
            lines.append("```")

    if not skill_by_lead.empty:
        lines.append("")
        lines.append("## Viability Skill by Lead")
        lines.append(_markdown_table(skill_by_lead, max_rows=20))

    if not proto_outcomes.empty:
        lines.append("")
        lines.append("## Proto Outcomes / Conversion Rates")
        lines.append(_markdown_table(proto_outcomes, max_rows=20))

    lines.append("")
    lines.append("## Data Health")
    if not run_health.empty:
        cols = [
            c
            for c in [
                "section",
                "mode",
                "path",
                "exists",
                "rows_metadata",
                "rows_counted",
                "rows",
                "rows_warn",
                "missing_cols",
            ]
            if c in run_health.columns
        ]
        lines.append(_markdown_table(run_health[cols], max_rows=30))
        if "rows_warn" in run_health.columns and run_health["rows_warn"].fillna(False).any():
            lines.append("- Row-count mismatch detected; check rows_metadata vs rows_counted.")
    else:
        lines.append("_No run_health table found._")

    if blocked_entries:
        lines.append("")
        lines.append("## BLOCKED")
        for item in blocked_entries:
            step = item.get("step", "report")
            reason = item.get("reason", "")
            detail = item.get("detail", "")
            path = item.get("path", "")
            prod = item.get("produced_by", "")
            msg = f"- {step}: {reason} {detail}".strip()
            if path:
                msg += f" ({path})"
            if prod:
                msg += f" | produced_by={prod}"
            lines.append(msg)

    lines.append("")
    lines.append("## Alert Stats")
    if alert_stats:
        lines.append(f"- Alerts: {alert_stats.get('alerts')}")
        lines.append(f"- Hours: {alert_stats.get('hours')}")
        lines.append(
            f"- Alerts/hour p50={alert_stats.get('per_hour_q50'):.1f} "
            f"p90={alert_stats.get('per_hour_q90'):.1f} "
            f"p99={alert_stats.get('per_hour_q99'):.1f}"
        )
        lines.append(f"- Top hours: {alert_stats.get('top_hours')}")
    else:
        lines.append("_No alerts stats available._")

    lines.append("")
    lines.append("## Slow-tick Diagnostics (observational)")
    lines.append("_Diagnostic only; does not influence pulse definitions or training._")
    if slowtick_dir and (not slowtick_summary.empty or not slowtick_spectrum.empty or not slowtick_knee.empty or not slowtick_parity.empty):
        if not slowtick_summary.empty:
            lines.append("Summary:")
            lines.append(_markdown_table(slowtick_summary, max_rows=20))
        if not slowtick_spectrum.empty:
            lines.append("")
            lines.append("Spectrum summary (24h +/-20% band):")
            lines.append(_markdown_table(slowtick_spectrum, max_rows=20))
        if not slowtick_knee.empty:
            lines.append("")
            lines.append("Knee fit:")
            lines.append(_markdown_table(slowtick_knee, max_rows=10))
        if not slowtick_parity.empty:
            lines.append("")
            lines.append("Hemispheric parity:")
            lines.append(_markdown_table(slowtick_parity, max_rows=20))
    else:
        lines.append("_No slow-tick diagnostics found._")

    if slowtick_union:
        lines.append("")
        lines.append("## Slow-tick Snapshot (seed union)")
        lines.append(f"- High-probability seeds (prob >= 0.5): {slowtick_union.get('high_prob_count')}")
        lines.append(f"- Mean slow-phase (h): {slowtick_union.get('phase_mean_h'):.2f}")
        lines.append(f"- Std slow-phase (h): {slowtick_union.get('phase_std_h'):.2f}")
        bins = slowtick_union.get("bins_h") or []
        hist = slowtick_union.get("hist") or []
        if bins and hist and len(bins) == len(hist) + 1:
            lines.append("  Phase distribution (3h bins):")
            for k in range(len(hist)):
                lines.append(f"  - {bins[k]:2.0f}-{bins[k+1]:2.0f} h : {int(hist[k]):7d}")

    lines.append("")
    lines.append("## Object Stats")
    if objects_sig:
        src_path = objects_sig.get("path")
        if objects_sig.get("exists"):
            rows = objects_sig.get("rows")
            rows_txt = f"{int(rows):,}" if isinstance(rows, (int, np.integer)) else "?"
            sha = objects_sig.get("sha256")
            sha_txt = f" sha256={sha[:12]}..." if sha else ""
            lines.append(f"- Objects source: {src_path} (rows={rows_txt}{sha_txt})")
        else:
            lines.append(f"- Objects source: {src_path} (missing)")
    if track_objects_info and objects_sig:
        track_path = track_objects_info.get("path") or ""
        track_rows = track_objects_info.get("rows")
        track_health = track_objects_info.get("health_ok")
        if track_path and str(track_path) != str(objects_sig.get("path")):
            if track_health is False or (isinstance(track_rows, (int, np.integer)) and track_rows == 0):
                lines.append(
                    f"- Note: training.track-objects output {track_path} rows={track_rows} health_ok={track_health}; "
                    f"report uses {objects_sig.get('path')}."
                )
    if obj_stats:
        lines.append(f"- Objects: {obj_stats.get('objects')}")
        lines.append(f"- Hours: {obj_stats.get('hours')}")
        lines.append(
            f"- Objects/hour p50={obj_stats.get('objects_per_hour_q50'):.1f} "
            f"p90={obj_stats.get('objects_per_hour_q90'):.1f}"
        )
        if obj_hours:
            lines.append(
                f"- Hours total={obj_hours.get('hours_total')} "
                f"with_objects={obj_hours.get('hours_with_objects')} "
                f"missing={obj_hours.get('hours_missing_objects')}"
            )
            gaps = obj_hours.get("missing_ranges") or []
            if gaps:
                gap_txt = [
                    f"{g['start'].isoformat()} -> {g['end'].isoformat()} ({g['hours']}h)"
                    for g in gaps
                    if isinstance(g, dict)
                ]
                if gap_txt:
                    lines.append(f"- Missing-hour ranges (top 5): {', '.join(gap_txt)}")
        area = obj_stats.get("area_cells", {})
        if area:
            lines.append(f"- Area cells q05={area.get('q05'):.1f} q50={area.get('q50'):.1f} q95={area.get('q95'):.1f}")
        comp = obj_stats.get("compactness", {})
        if comp:
            lines.append(
                f"- Compactness q05={comp.get('q05'):.2f} q50={comp.get('q50'):.2f} q95={comp.get('q95'):.2f}"
            )
    else:
        lines.append("_No object stats available._")

    lines.append("")
    lines.append("## Object Diagnostics")
    if not frag_df.empty:
        lines.append("Fragmentation by area:")
        lines.append(_markdown_table(frag_df, max_rows=10))
    if not hourly_obj.empty:
        nn_med = None
        dens_med = None
        if "nn_median_km" in hourly_obj.columns:
            nn_med = float(pd.to_numeric(hourly_obj["nn_median_km"], errors="coerce").median())
        if "density_per_10k_km2" in hourly_obj.columns:
            dens_med = float(pd.to_numeric(hourly_obj["density_per_10k_km2"], errors="coerce").median())
        if nn_med is not None:
            lines.append(f"- Median nearest-neighbor distance (km): {nn_med:.1f}")
        if dens_med is not None:
            lines.append(f"- Median object density per 10k km^2: {dens_med:.2f}")
    if not rejects_df.empty and "rejected_small" in rejects_df.columns:
        rej_med = float(pd.to_numeric(rejects_df["rejected_small"], errors="coerce").median())
        lines.append(f"- Median rejected small objects/hour: {rej_med:.1f}")
    if frag_df.empty and hourly_obj.empty and rejects_df.empty:
        lines.append("_No object diagnostics available._")

    lines.append("")
    lines.append("## Storm Hourly Seed Counts")
    if not storm_hourly.empty:
        agg = (
            storm_hourly.groupby("storm_id")["points"]
            .agg(total_points="sum", hours_with_points="count", max_per_hour="max")
            .reset_index()
            .sort_values("total_points", ascending=False)
        )
        lines.append(_markdown_table(agg, max_rows=15))
        heatmap_path = out_dir / "maps" / "seed_storm_hourly_counts.png"
        if heatmap_path.exists():
            lines.append(f"- Heatmap: {heatmap_path}")
    else:
        lines.append("_No storm hourly counts available._")

    lines.append("")
    lines.append("## Lead-time (primary, future-only)")
    if lead_metrics:
        lines.append("- Primary KPI: computed without match-time tolerance (future-only lead windows).")
        lines.append(f"- Objects evaluated: {lead_metrics.get('n_objects')}")
        closest_bins = lead_metrics.get("closest_bins")
        if isinstance(closest_bins, pd.DataFrame) and not closest_bins.empty:
            lines.append("Lead to closest approach (future-only):")
            lines.append(_markdown_table(closest_bins, max_rows=20))
        radius_bins = lead_metrics.get("radius_bins")
        if isinstance(radius_bins, pd.DataFrame) and not radius_bins.empty:
            lines.append("")
            lines.append(f"Lead to enter radius ({args.lead_radius_km} km):")
            lines.append(_markdown_table(radius_bins, max_rows=20))
        genesis_bins = lead_metrics.get("genesis_bins")
        if isinstance(genesis_bins, pd.DataFrame) and not genesis_bins.empty:
            lines.append("")
            lines.append("Lead to genesis (future-only):")
            lines.append(_markdown_table(genesis_bins, max_rows=20))
        cat1_bins = lead_metrics.get("cat1_bins")
        if isinstance(cat1_bins, pd.DataFrame) and not cat1_bins.empty:
            lines.append("")
            lines.append(f"Lead to Cat1+ (vmax >= {args.cat1_threshold}):")
            lines.append(_markdown_table(cat1_bins, max_rows=20))
    else:
        lines.append("_No lead-time metrics available._")

    lines.append("")
    lines.append("## Match Skill (time-tolerant)")
    lines.append("- Coincidence metric using time-tolerant matching (see match stage).")
    if match_skill:
        dist_q = match_skill.get("distance_quantiles", {})
        lines.append(
            f"- Distance km q05={dist_q.get('q05'):.1f} q50={dist_q.get('q50'):.1f} q95={dist_q.get('q95'):.1f}"
        )
        lead_stats = match_skill.get("lead_bins")
        if isinstance(lead_stats, pd.DataFrame) and not lead_stats.empty:
            lines.append("")
            lines.append("Lead-bin distance summary:")
            lines.append(_markdown_table(lead_stats, max_rows=20))
        cov = match_skill.get("coverage_precision", [])
        if cov:
            cov_df = pd.DataFrame(cov)
            lines.append("")
            lines.append("Coverage vs precision by threshold (proxy, d_km <= 50):")
            lines.append(_markdown_table(cov_df, max_rows=10))
    else:
        lines.append("_No match skill available._")

    lines.append("")
    lines.append("## Storm-by-storm Pages")
    if storm_pages:
        lines.append(_markdown_table(pd.DataFrame(storm_pages), max_rows=50))
    else:
        lines.append("_No storm pages generated._")

    lines.append("")
    lines.append("## Correlations Sweep")
    if corr_overall is not None and not corr_overall.empty:
        top = corr_overall.copy()
        if "spearman" in top.columns:
            top = top.assign(_abs=top["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
        lines.append(_markdown_table(top, max_rows=20))
        by_lead = tables_dir / "correlations_by_lead.csv"
        by_lat = tables_dir / "correlations_by_lat.csv"
        by_month = tables_dir / "correlations_by_month.csv"
        by_resid = tables_dir / "correlations_residualized.csv"
        by_perm = tables_dir / "correlations_permtest.csv"
        if by_lead.exists():
            lines.append(f"- By lead: {by_lead}")
        if by_lat.exists():
            lines.append(f"- By latitude: {by_lat}")
        if by_month.exists():
            lines.append(f"- By month: {by_month}")
        if by_resid.exists():
            lines.append(f"- Residualized: {by_resid}")
        if by_perm.exists():
            lines.append(f"- Permutation test: {by_perm}")
        if by_resid.exists():
            resid = _read_any(by_resid)
            if not resid.empty and "spearman" in resid.columns:
                resid = resid.assign(_abs=resid["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
                lines.append("")
                lines.append("Top residualized correlations:")
                lines.append(_markdown_table(resid, max_rows=15))
        if by_perm.exists():
            perm = _read_any(by_perm)
            if not perm.empty and "spearman" in perm.columns:
                perm = perm.assign(_abs=perm["spearman"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
                lines.append("")
                lines.append("Permutation test (top correlations after shuffle):")
                lines.append(_markdown_table(perm, max_rows=15))
        lines.append("")
        lines.append("_Note: multiple testing applies; treat these as hypotheses._")
    elif corr_df is not None and not corr_df.empty:
        lines.append(_markdown_table(corr_df, max_rows=20))
        lines.append("")
        lines.append("_Note: multiple testing applies; treat these as hypotheses._")
    else:
        lines.append("_No correlation table found._")

    if extras:
        lines.append("")
        lines.append("## Extra Artifacts")
        for k, v in extras.items():
            lines.append(f"- {k}: {v}")

    if args.write_txt or args.txt_only:
        txt_path = out_dir / f"{safe_run}_report.txt"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_json = {
        "run_summary": run_summary,
        "stage_cache_summary": stage_manifest.to_dict(orient="records") if not stage_manifest.empty else [],
        "data_health": run_health.to_dict(orient="records") if not run_health.empty else [],
        "blocked": blocked_entries,
        "seed_summary_text": seed_summary_text,
        "seed_analysis_text": seed_analysis_text,
        "alerts_snapshot": alerts_snapshot or {},
        "ibtracs_reference": ibtracs_ref or {},
        "storm_timeseries_snapshot": storm_ts_snapshot or {},
        "viability_conversion": viability_conversion.to_dict(orient="records") if not viability_conversion.empty else [],
        "viability_thresholds": viability_thresholds.to_dict(orient="records") if not viability_thresholds.empty else [],
        "viability_sweep_diagnostics": viability_sweep_diag.to_dict(orient="records") if not viability_sweep_diag.empty else [],
        "viability_metrics": viability_metrics.get("raw") if viability_metrics else {},
        "viability_skill_by_lead": skill_by_lead.to_dict(orient="records") if not skill_by_lead.empty else [],
        "proto_outcomes": proto_outcomes.to_dict(orient="records") if not proto_outcomes.empty else [],
        "slowtick_union": slowtick_union,
        "extras": extras,
        "alert_stats": alert_stats,
        "slowtick": {
            "dir": str(slowtick_dir) if slowtick_dir else None,
            "summary": slowtick_summary.to_dict(orient="records") if not slowtick_summary.empty else [],
            "spectrum": slowtick_spectrum.to_dict(orient="records") if not slowtick_spectrum.empty else [],
            "knee_fit": slowtick_knee.to_dict(orient="records") if not slowtick_knee.empty else [],
            "parity": slowtick_parity.to_dict(orient="records") if not slowtick_parity.empty else [],
        },
        "object_stats": obj_stats,
        "object_hour_gaps": obj_hours,
        "objects_signature": objects_sig,
        "track_objects_manifest": track_objects_info,
        "storm_hourly_counts": storm_hourly.to_dict(orient="records") if not storm_hourly.empty else [],
        "match_skill": {
            "distance_quantiles": match_skill.get("distance_quantiles", {}) if match_skill else {},
            "lead_bins": match_skill.get("lead_bins").to_dict(orient="records") if isinstance(match_skill.get("lead_bins"), pd.DataFrame) else [],
            "coverage_precision": match_skill.get("coverage_precision", []) if match_skill else [],
        },
        "lead_metrics": {
            "n_objects": lead_metrics.get("n_objects") if lead_metrics else None,
            "closest_bins": lead_metrics.get("closest_bins").to_dict(orient="records") if isinstance(lead_metrics.get("closest_bins"), pd.DataFrame) else [],
            "radius_bins": lead_metrics.get("radius_bins").to_dict(orient="records") if isinstance(lead_metrics.get("radius_bins"), pd.DataFrame) else [],
            "genesis_bins": lead_metrics.get("genesis_bins").to_dict(orient="records") if isinstance(lead_metrics.get("genesis_bins"), pd.DataFrame) else [],
            "cat1_bins": lead_metrics.get("cat1_bins").to_dict(orient="records") if isinstance(lead_metrics.get("cat1_bins"), pd.DataFrame) else [],
        },
        "storm_pages": storm_pages,
    }
    if not args.txt_only:
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(report_json, indent=2, ensure_ascii=True, default=_json_default),
            encoding="utf-8",
        )

        print(f"[reporting-v2] wrote {md_path}")
        print(f"[reporting-v2] wrote {json_path}")
    if args.write_txt or args.txt_only:
        print(f"[reporting-v2] wrote {out_dir / f'{safe_run}_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
