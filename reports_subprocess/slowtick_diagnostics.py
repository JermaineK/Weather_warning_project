#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slowtick_diagnostics.py

Post-process throttled/denoised/base alert grids to quantify:
  1) Knee law: coverage vs lead ~ L^(-2p) with bootstrap CI
  2) Hemispheric parity: N vs S coverage asymmetry with CI
  3) Slow-tick ridge: hourly spectrum peak near diurnal/slow band

Upgrades:
  - Anti-meridian-safe AOI crop
  - Robust file selection (stage preference + latest mtime)
  - Tolerant CSV read + consistent tz-naive UTC
  - Guardrails (--min-hours-per-lead, NaN handling)
  - Exposed bootstrap reps (--bootstrap-B)
  - Optional gap-filling for FFT and hemi time-series export
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------- shared helpers ----------

def _read_csv_tolerant(path: Path, usecols=None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    CSV reader that tolerates weird lines and can optionally limit rows.
    """
    try:
        return pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            usecols=usecols,
            nrows=nrows,
        )
    except Exception:
        return pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            engine="python",
            on_bad_lines="warn",
            usecols=usecols,
            nrows=nrows,
        )


def _is_parquet(path: Path) -> bool:
    suffixes = "".join(path.suffixes[-2:]).lower()
    return suffixes == ".parquet" or path.suffix.lower() == ".parquet"


def _read_table_tolerant(path: Path, usecols=None, nrows: Optional[int] = None) -> pd.DataFrame:
    if _is_parquet(path):
        try:
            return pd.read_parquet(path, columns=usecols)
        except Exception:
            return pd.read_parquet(path)
    return _read_csv_tolerant(path, usecols=usecols, nrows=nrows)


def _peek_columns(path: Path) -> List[str]:
    if _is_parquet(path):
        try:
            import pyarrow.parquet as pq  # type: ignore

            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            try:
                return list(pd.read_parquet(path, columns=None).columns)
            except Exception:
                return []
    try:
        head = _read_csv_tolerant(path, nrows=1)
        return list(head.columns)
    except Exception:
        return []


def _try_parse_time_raw(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)

    # direct ISO-ish
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)

    # user format
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass

    # numeric epoch
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = float(np.nanmedian(num))
        unit = "ms" if np.isfinite(mid) and mid > 1e11 else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)

    # last resort: infer format
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # default: -180..180
    return ((x + 180) % 360) - 180


def _parse_area(aoi: Optional[str]):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE


def _crop_aoi(df: pd.DataFrame, aoi, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    if not aoi:
        return df
    latN, lonW, latS, lonE = aoi
    df = df[(df[lat_col] <= latN) & (df[lat_col] >= latS)]
    if lonW <= lonE:
        return df[(df[lon_col] >= lonW) & (df[lon_col] <= lonE)]
    # wrap across anti-meridian
    return df[(df[lon_col] >= lonW) | (df[lon_col] <= lonE)]


def _hash_series(series: pd.Series, max_items: int = 1_000_000) -> str:
    if series is None or len(series) == 0:
        return ""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    idx = series.index
    try:
        idx_vals = pd.to_datetime(idx, errors="coerce").view("int64").to_numpy(dtype=np.int64, copy=False)
    except Exception:
        idx_vals = pd.to_numeric(pd.Index(idx), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    n = min(len(vals), max_items)
    packed = np.column_stack([idx_vals[:n], vals[:n]]).astype(np.float64, copy=False)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _hash_hemi(df: pd.DataFrame, max_items: int = 1_000_000) -> str:
    if df is None or df.empty:
        return ""
    north = pd.to_numeric(df.get("north"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    south = pd.to_numeric(df.get("south"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    idx = df.index
    try:
        idx_vals = pd.to_datetime(idx, errors="coerce").view("int64").to_numpy(dtype=np.int64, copy=False)
    except Exception:
        idx_vals = pd.to_numeric(pd.Index(idx), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    n = min(len(idx_vals), max_items)
    packed = np.column_stack([idx_vals[:n], north[:n], south[:n]]).astype(np.float64, copy=False)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _duplicate_hashes(hash_map: Dict[int, str]) -> Dict[str, List[int]]:
    by_hash: Dict[str, List[int]] = {}
    for lead, h in hash_map.items():
        if not h:
            continue
        by_hash.setdefault(h, []).append(int(lead))
    return {h: sorted(leads) for h, leads in by_hash.items() if len(leads) > 1}


# ---------- file I/O ----------

def _pick_candidate(paths: List[Path], prefer: str) -> Optional[Path]:
    # prefer can be 'throttled' | 'denoised' | 'base'
    def stage_score(p: Path) -> int:
        n = p.name
        if "_throttled" in n:
            return 0
        if "_denoised" in n:
            return 1
        return 2

    if not paths:
        return None

    if prefer == "throttled":
        wanted = [p for p in paths if "_throttled" in p.name]
    elif prefer == "denoised":
        wanted = [p for p in paths if "_denoised" in p.name]
    else:
        wanted = [p for p in paths if "_throttled" not in p.name and "_denoised" not in p.name]
    pool = wanted if wanted else paths
    pool = sorted(pool, key=lambda p: (stage_score(p), -p.stat().st_mtime))
    return pool[0] if pool else None


def _load_alert_table(
    path: Path,
    time_fmt: Optional[str],
    norm_lon: str,
    aoi: Optional[str],
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = _read_table_tolerant(path, usecols=usecols)

    # Ensure required columns exist
    if not {"time", "lat", "lon"}.issubset(df.columns):
        df = _read_table_tolerant(path)  # last resort full read

    # Normalize
    df["time"] = _try_parse_time_raw(df["time"], time_fmt)
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _norm_lon(df["lon"], norm_lon)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # AOI crop (anti-meridian safe)
    df = _crop_aoi(df, _parse_area(aoi), "lat", "lon")
    return df


def _load_threshold_map(path: Optional[str], thr_col: str) -> Dict[int, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = _read_table_tolerant(p)
    except Exception:
        return {}
    if df.empty:
        return {}

    lead_col = None
    for cand in ("lead_h", "lead", "lead_hours", "lead_hr"):
        if cand in df.columns:
            lead_col = cand
            break
    if lead_col is None:
        return {}

    thr_candidates = [
        thr_col,
        thr_col.lower(),
        "thr_Fbeta",
        "thr_fbeta",
        "thr_F1",
        "thr_f1",
        "opt_threshold",
        "thr",
        "threshold",
    ]
    thr_use = None
    for cand in thr_candidates:
        if cand in df.columns:
            thr_use = cand
            break
    if thr_use is None:
        return {}

    lead_vals = pd.to_numeric(df[lead_col], errors="coerce")
    thr_vals = pd.to_numeric(df[thr_use], errors="coerce")
    out: Dict[int, float] = {}
    for lead, thr in zip(lead_vals, thr_vals):
        if not np.isfinite(lead) or not np.isfinite(thr):
            continue
        out[int(lead)] = float(thr)
    return out


def _read_alert_for_lead(
    alerts_dir: Path,
    run: str,
    lead: int,
    requested_flag: str,
    time_fmt: Optional[str],
    norm_lon: str,
    aoi: Optional[str],
    prefer: str,
    debug: bool,
    fallback_path: Optional[Path],
    thresholds: Dict[int, float],
    prob_col: str,
    prob_quantile: Optional[float],
    fallback_df: Optional[pd.DataFrame],
    require_lead_col: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str], Dict[str, object]]:
    """
    Locate the preferred alerts file for a given lead, load it, normalize time/lat/lon,
    determine the effective flag column (with fallbacks), and return (df, path, eff_flag).
    """
    patterns = [
        f"alerts_{run}_lead{lead}_thr*.csv.gz",
        f"alerts_{run}_lead{lead}_thr*.csv",
        f"alerts_{run}_lead{lead}_thr*.parquet",
    ]
    cand: List[Path] = []
    for pat in patterns:
        cand.extend(alerts_dir.glob(pat))
    cand = sorted(cand)
    using_fallback = False
    if not cand:
        if fallback_path and fallback_path.exists():
            path = fallback_path
            using_fallback = True
            if debug:
                print(f"[slowtick][debug] lead={lead} using fallback {path.name}")
        else:
            return None, None, None
    else:
        path = _pick_candidate(cand, prefer)
        if path is None:
            return None, None, None

    # Peek columns
    if using_fallback and fallback_df is not None:
        cols = set(fallback_df.columns)
    else:
        cols = set(_peek_columns(path))

    # Optional: filter by lead_h if present to avoid identical-per-lead series
    lead_col = None
    for cand in ("lead_h", "lead", "lead_hours", "lead_hr"):
        if cand in cols:
            lead_col = cand
            break
    inferred_lead = False

    thr = thresholds.get(int(lead))
    lead_flag = None
    if requested_flag:
        lead_candidates = [
            f"{requested_flag}_lead{lead}",
            f"{requested_flag}_lead{lead}h",
            f"{requested_flag}_lead_{lead}",
            f"{requested_flag}_lead_{lead}h",
            f"{requested_flag}_{lead}",
            f"{requested_flag}_{lead}h",
        ]
        for cand in lead_candidates:
            if cand in cols:
                lead_flag = cand
                break

    use_prob_threshold = False

    # Enforce lead-aware filtering
    if lead_col is None:
        lead_in_name = f"lead{lead}" in str(path.name)
        if not lead_in_name and lead_flag is None and require_lead_col:
            raise SystemExit(
                f"[slowtick] lead_h column missing in {path.name}; cannot guarantee per-lead filtering."
            )
        # Allow explicit lead-only files or lead-specific flags by injecting lead_h
        lead_col = "lead_h"
        inferred_lead = True

    # Decide effective flag
    if lead_flag is not None:
        eff_flag = lead_flag
    else:
        eff_flag = requested_flag if requested_flag in cols else None
        if eff_flag is None:
            for alt in ("alert_throttled", "alert_final", "alert"):
                if alt in cols:
                    eff_flag = alt
                    break
        if eff_flag is None:
            eff_flag = requested_flag or "alert_final"

    usecols = [c for c in ("time", "lat", "lon", eff_flag, prob_col, lead_col) if c and c in cols] or None
    if using_fallback and fallback_df is not None:
        df = fallback_df
    else:
        df = _load_alert_table(path, time_fmt, norm_lon, aoi, usecols=usecols)
    if inferred_lead:
        df[lead_col] = float(lead)

    # If a lead column exists, slice to this lead's rows to prevent cross-lead mixing
    if lead_col and lead_col in df.columns:
        lead_vals = pd.to_numeric(df[lead_col], errors="coerce")
        mask = np.isfinite(lead_vals) & np.isclose(lead_vals.to_numpy(), float(lead), atol=0.01)
        if mask.any():
            df = df.loc[mask].copy()
        # If mask is empty, keep original (per-lead files may omit lead column)

    # Optional per-lead quantile thresholding on prob_col (overrides static thresholds)
    if prob_quantile is not None and prob_col in df.columns:
        prob_vals = pd.to_numeric(df[prob_col], errors="coerce")
        if prob_vals.notna().any():
            thr = float(prob_vals.quantile(prob_quantile))
        else:
            thr = None

    if thr is not None and prob_col in df.columns and lead_flag is None:
        use_prob_threshold = True
        eff_flag = f"_prob_thr_{lead}h"

    if use_prob_threshold:
        if prob_col in df.columns:
            prob = pd.to_numeric(df[prob_col], errors="coerce").fillna(-np.inf)
            df[eff_flag] = (prob >= float(thr)).astype(int)
        else:
            df[eff_flag] = 0
    else:
        if eff_flag not in df.columns:
            df[eff_flag] = 0
        df[eff_flag] = pd.to_numeric(df[eff_flag], errors="coerce").fillna(0).astype(int)

    if debug and not df.empty:
        tt = df["time"].dt.floor("h")
        print(
            f"[slowtick][debug] lead={lead} file={path.name} "
            f"| hours={tt.nunique()} time={tt.min()}->{tt.max()} "
            f"| lat={df['lat'].min():.2f}..{df['lat'].max():.2f} "
            f"| lon={df['lon'].min():.2f}..{df['lon'].max():.2f} "
            f"| flag_col={eff_flag}"
        )
        if lead_col and lead_col in df.columns:
            try:
                uniq = pd.to_numeric(df[lead_col], errors="coerce").dropna().unique()
                if len(uniq) <= 5:
                    print(f"[slowtick][debug] lead_col={lead_col} values={sorted(map(float, uniq))}")
                else:
                    print(f"[slowtick][debug] lead_col={lead_col} unique={len(uniq)}")
            except Exception:
                pass

    filter_mode = "lead_h_subset" if lead_col and not inferred_lead else "lead_flag_subset"
    if inferred_lead and f"lead{lead}" in str(path.name):
        filter_mode = "file_per_lead"
    meta = {
        "filter_mode": filter_mode,
        "lead_col": lead_col or "",
    }
    return df, path, eff_flag, meta


def _coverage_by_hour(df: pd.DataFrame, flag_col: str) -> pd.Series:
    tt = df["time"].dt.floor("h")
    return df.assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()


def _coverage_by_hour_hemi(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    tt = df["time"].dt.floor("h")
    north = df[df["lat"] >= 0].assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()
    south = df[df["lat"] < 0].assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()
    both = pd.concat({"north": north, "south": south}, axis=1).astype(float)
    return both.fillna(0.0)


def _format_coverage_pattern(pattern: str, lead: int) -> str:
    try:
        return pattern.format(lead=lead, lead_h=lead)
    except KeyError as exc:
        raise SystemExit(f"[slowtick] coverage-series-pattern missing key {exc}. Use {{lead}} or {{lead_h}}.")
    except Exception as exc:
        raise SystemExit(f"[slowtick] coverage-series-pattern error: {exc}")


def _coverage_series_candidates(series_dir: Path, lead: int, pattern: str) -> List[Path]:
    formatted = _format_coverage_pattern(pattern, lead)
    base = Path(formatted)
    if not base.is_absolute():
        base = series_dir / base
    if base.suffixes:
        return [base]
    base_str = str(base)
    return [
        Path(base_str + ".parquet"),
        Path(base_str + ".csv"),
        Path(base_str + ".csv.gz"),
    ]


def _load_coverage_series_file(
    path: Path,
    time_fmt: Optional[str],
) -> Tuple[pd.Series, Optional[pd.DataFrame], str, Optional[Tuple[str, str]]]:
    df = _read_table_tolerant(path)
    if df.empty:
        raise SystemExit(f"[slowtick] coverage series empty: {path}")
    if "time" not in df.columns:
        raise SystemExit(f"[slowtick] coverage series missing time column: {path}")
    df["time"] = _try_parse_time_raw(df["time"], time_fmt)
    df = df.dropna(subset=["time"]).copy()
    if df.empty:
        raise SystemExit(f"[slowtick] coverage series has no valid time rows: {path}")

    cov_col = None
    for cand in ("coverage", "cov", "mean_cov"):
        if cand in df.columns:
            cov_col = cand
            break
    if cov_col is None:
        raise SystemExit(f"[slowtick] coverage series missing coverage column: {path}")
    df[cov_col] = pd.to_numeric(df[cov_col], errors="coerce")
    cov = df.groupby("time", sort=True)[cov_col].mean().sort_index()

    hemi_cols: Optional[Tuple[str, str]] = None
    hemi: Optional[pd.DataFrame] = None
    if {"cov_north", "cov_south"}.issubset(df.columns):
        hemi_cols = ("cov_north", "cov_south")
        df["cov_north"] = pd.to_numeric(df["cov_north"], errors="coerce")
        df["cov_south"] = pd.to_numeric(df["cov_south"], errors="coerce")
        hemi = (
            df.groupby("time", sort=True)[["cov_north", "cov_south"]]
            .mean()
            .rename(columns={"cov_north": "north", "cov_south": "south"})
            .sort_index()
        )
    elif {"north", "south"}.issubset(df.columns):
        hemi_cols = ("north", "south")
        df["north"] = pd.to_numeric(df["north"], errors="coerce")
        df["south"] = pd.to_numeric(df["south"], errors="coerce")
        hemi = df.groupby("time", sort=True)[["north", "south"]].mean().sort_index()

    return cov, hemi, cov_col, hemi_cols


# ---------- stats helpers ----------

def _bootstrap_ci(values: np.ndarray, stat_fn, B=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return math.nan, (math.nan, math.nan)
    bs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        bs[b] = stat_fn(values[idx])
    lo = np.quantile(bs, alpha / 2)
    hi = np.quantile(bs, 1 - alpha / 2)
    return float(stat_fn(values)), (float(lo), float(hi))


def _fit_knee(logL, logCov):
    X = np.c_[np.ones_like(logL), logL]
    beta, *_ = np.linalg.lstsq(X, logCov, rcond=None)
    a, b = beta
    p = -b / 2.0
    return float(p), float(a), float(b)


def _knee_fit_with_bootstrap(leads, mean_cov, B=1000):
    L = np.asarray(leads, dtype=float)
    C = np.asarray(mean_cov, dtype=float)
    mask = np.isfinite(L) & np.isfinite(C) & (L > 0) & (C > 0)
    L, C = L[mask], C[mask]
    if len(L) < 2:
        return dict(p=math.nan, p_lo=math.nan, p_hi=math.nan, used_points=int(mask.sum()))
    logL = np.log(L)
    logC = np.log(C)
    p_hat, *_ = _fit_knee(logL, logC)

    rng = np.random.default_rng(123)
    ps = []
    for _ in range(B):
        idx = rng.integers(0, len(L), size=len(L))
        p_b, *_ = _fit_knee(logL[idx], logC[idx])
        ps.append(p_b)
    lo, hi = np.quantile(ps, [0.025, 0.975])
    return dict(p=float(p_hat), p_lo=float(lo), p_hi=float(hi), used_points=int(mask.sum()))


def _fft_peak(freqs, amps, target_per_h=24, band=0.20):
    f0 = 1.0 / target_per_h
    band_lo = f0 * (1 - band)
    band_hi = f0 * (1 + band)
    band_mask = (freqs >= band_lo) & (freqs <= band_hi)
    if not band_mask.any():
        return math.nan, math.nan
    k = np.argmax(amps[band_mask])
    f_peak = freqs[band_mask][k]
    a_peak = amps[band_mask][k]
    return float(f_peak), float(a_peak)


def _fill_small_gaps_hourly(series: pd.Series, max_gap_h=3) -> pd.Series:
    """Fill short NaN runs (<= max_gap_h) by linear interpolation; leave longer gaps."""
    s = series.copy()
    interp = s.interpolate(limit=max_gap_h, limit_direction="both")
    is_nan = s.isna().to_numpy()
    pad = np.r_[False, is_nan, False]
    edges = np.diff(pad.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for st, en in zip(starts, ends):
        run = en - st
        if run > max_gap_h:
            interp.iloc[st:en] = np.nan
    return interp


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Slow-tick diagnostics from throttled/denoised/base alerts.")
    ap.add_argument("--alerts-dir", required=True, help="Directory with alerts (throttled/denoised/base).")
    ap.add_argument("--run-name", required=True, help="Run name used in filenames (alerts_<run>_leadX_...).")
    ap.add_argument(
        "--coverage-series-dir",
        default=None,
        help=(
            "Optional directory with per-lead coverage series files "
            "(coverage_timeseries_lead_{lead}.parquet/csv). When set, alerts are not read."
        ),
    )
    ap.add_argument(
        "--coverage-series-pattern",
        default="coverage_timeseries_lead_{lead}",
        help=(
            "Per-lead coverage series filename pattern. Use {lead} or {lead_h}; "
            "omit extension to try .parquet/.csv/.csv.gz."
        ),
    )
    ap.add_argument("--fallback-alerts", default=None, help="Optional alerts file to use when per-lead files are missing.")
    ap.add_argument("--thresholds", default=None, help="Optional thresholds table (CSV/Parquet) for per-lead flags.")
    ap.add_argument("--threshold-col", default="thr_Fbeta", help="Threshold column to use in --thresholds.")
    ap.add_argument("--prob-col", default="prob_viable", help="Probability column for derived per-lead flags.")
    ap.add_argument(
        "--prob-quantile",
        type=float,
        default=None,
        help="Optional per-lead probability quantile for thresholds (overrides --thresholds).",
    )
    ap.add_argument(
        "--require-lead-col",
        action="store_true",
        help="Require a lead_h column (or explicit per-lead file) to enforce lead filtering.",
    )
    ap.add_argument("--leads", type=int, nargs="+", required=True, help="Lead hours to include.")
    ap.add_argument("--flag-col", default="alert_final", help="Preferred flag column (tries fallbacks).")
    ap.add_argument("--out-dir", default="results/slowtick", help="Output directory for CSVs/plots.")
    ap.add_argument("--time-format", default=None, help="Optional strftime for custom time parsing.")
    ap.add_argument(
        "--normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="none",
        help="Normalize longitudes in input files (default: none).",
    )
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument(
        "--prefer",
        choices=["throttled", "denoised", "base"],
        default="throttled",
        help="Which stage to prefer if multiple exist for a lead.",
    )
    ap.add_argument("--save-timeseries", action="store_true", help="Also save hourly coverage time series (global).")
    ap.add_argument(
        "--save-hemi-timeseries",
        action="store_true",
        help="Also save hourly N/S coverage time series.",
    )
    ap.add_argument("--bootstrap-B", type=int, default=1000, help="Bootstrap draws for knee/parity CIs.")
    ap.add_argument("--min-hours-per-lead", type=int, default=8, help="Skip leads with fewer hourly points.")
    ap.add_argument("--fft-gap-fill", type=int, default=2, help="Fill NaN gaps <= this many hours before FFT.")
    ap.add_argument("--cache-fallback", action="store_true", help="Cache fallback alerts in memory for reuse.")
    ap.add_argument(
        "--allow-identical",
        action="store_true",
        help="Allow identical per-lead coverage/parity series without failing (still logged).",
    )
    ap.add_argument(
        "--allow-const-flags",
        action="store_true",
        help="Allow constant per-lead flags without aborting (default: fail on const).",
    )
    ap.add_argument("--debug", action="store_true", help="Print file/range diagnostics.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = Path(args.alerts_dir)
    fallback_path = Path(args.fallback_alerts) if args.fallback_alerts else None

    thr_map = _load_threshold_map(args.thresholds, args.threshold_col)
    if args.thresholds and not thr_map:
        print("[slowtick] warning: thresholds provided but no lead thresholds parsed.")
    if args.prob_quantile is not None:
        if not (0.0 < args.prob_quantile < 1.0):
            raise SystemExit("--prob-quantile must be in (0,1).")

    fallback_df: Optional[pd.DataFrame] = None
    if args.cache_fallback and fallback_path and fallback_path.exists():
        cols = set(_peek_columns(fallback_path))
        candidate_cols = (
            "time",
            "lat",
            "lon",
            args.prob_col,
            args.flag_col,
            "alert_throttled",
            "alert_final",
            "alert",
        )
        usecols = [c for c in candidate_cols if c in cols] or None
        fallback_df = _load_alert_table(
            fallback_path,
            time_fmt=args.time_format,
            norm_lon=args.normalize_lon,
            aoi=args.area,
            usecols=usecols,
        )

    rows: List[Dict] = []
    cov_time: Dict[int, pd.Series] = {}
    cov_hemi_time: Dict[int, pd.DataFrame] = {}
    cov_hashes: Dict[int, str] = {}
    hemi_hashes: Dict[int, str] = {}

    # 1) coverage time series per lead
    use_series = bool(args.coverage_series_dir)
    series_dir = Path(args.coverage_series_dir) if args.coverage_series_dir else None
    if use_series and series_dir and not series_dir.exists():
        raise SystemExit(f"[slowtick] coverage-series-dir not found: {series_dir}")

    if use_series:
        # Agent: enforce per-lead coverage-series inputs when provided (no alerts parsing).
        for L in args.leads:
            candidates = _coverage_series_candidates(series_dir, L, args.coverage_series_pattern)
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                searched = ", ".join(str(p.name) for p in candidates)
                raise SystemExit(f"[slowtick] lead={L}: coverage series missing (searched {searched})")

            cov, hemi, cov_col, _ = _load_coverage_series_file(path, args.time_format)
            if cov.notna().sum() < max(4, args.min_hours_per_lead):
                print(f"[slowtick] lead={L}: too few hourly points ({cov.notna().sum()}); skipping.")
                continue

            cov = cov.sort_index()
            cov_time[L] = cov
            cov_hash = _hash_series(cov)
            cov_hashes[L] = cov_hash

            hemi_hash = ""
            if hemi is not None and not hemi.empty:
                hemi = hemi.sort_index()
                cov_hemi_time[L] = hemi
                hemi_hash = _hash_hemi(hemi)
                hemi_hashes[L] = hemi_hash

            rows.append(
                dict(
                    run_id=str(args.run_name),
                    lead_h=L,
                    hours=int(cov.notna().sum()),
                    mean_cov=float(np.nanmean(cov.values)),
                    file=path.name,
                    alerts_source_file="",
                    filter_mode="coverage_series",
                    n_rows_alerts_used=int(len(cov)),
                    time_min=str(cov.index.min()) if len(cov) else "",
                    time_max=str(cov.index.max()) if len(cov) else "",
                    flag_col="",
                    coverage_col=cov_col,
                    flag_mean=float("nan"),
                    flag_std=float("nan"),
                    flag_nonzero_frac=float("nan"),
                    flag_const=float("nan"),
                    series_hash=cov_hash,
                    hemi_hash=hemi_hash,
                    source="coverage_series",
                )
            )
    else:
        const_issues = []
        for L in args.leads:
            df, path, eff_flag, meta = _read_alert_for_lead(
                alerts_dir,
                args.run_name,
                L,
                args.flag_col,
                time_fmt=args.time_format,
                norm_lon=args.normalize_lon,
                aoi=args.area,
                prefer=args.prefer,
                debug=args.debug,
                fallback_path=fallback_path,
                thresholds=thr_map,
                prob_col=args.prob_col,
                prob_quantile=args.prob_quantile,
                fallback_df=fallback_df,
                require_lead_col=args.require_lead_col,
            )
            if df is None or df.empty or eff_flag is None:
                print(f"[slowtick] lead={L}: no usable alerts file; skipping.")
                continue

            flag_vals = pd.to_numeric(df[eff_flag], errors="coerce").fillna(0.0)
            flag_mean = float(flag_vals.mean()) if len(flag_vals) else float("nan")
            flag_std = float(flag_vals.std(ddof=0)) if len(flag_vals) else float("nan")
            flag_nonzero = float((flag_vals > 0).mean()) if len(flag_vals) else float("nan")
            flag_const = bool((np.isfinite(flag_std) and flag_std == 0.0) or flag_nonzero in (0.0, 1.0))
            if flag_const and not args.allow_const_flags:
                prob_vals = None
                if args.prob_col in df.columns:
                    prob_vals = pd.to_numeric(df[args.prob_col], errors="coerce")
                thr_used = thr_map.get(int(L))
                if args.prob_quantile is not None and prob_vals is not None and prob_vals.notna().any():
                    thr_used = float(prob_vals.quantile(args.prob_quantile))
                hist = None
                if prob_vals is not None:
                    vals = prob_vals.dropna().to_numpy()
                    if vals.size:
                        hist = np.histogram(vals, bins=10)
                const_issues.append(
                    dict(
                        lead_h=int(L),
                        flag_col=eff_flag,
                        prob_col=args.prob_col if args.prob_col in df.columns else None,
                        prob_min=float(np.nanmin(prob_vals)) if prob_vals is not None else float("nan"),
                        prob_max=float(np.nanmax(prob_vals)) if prob_vals is not None else float("nan"),
                        thr_used=thr_used,
                        flag_mean=flag_mean,
                        flag_nonzero_frac=flag_nonzero,
                        hist=hist,
                    )
                )

            coverage_col = eff_flag
            if flag_const and args.prob_col in df.columns:
                coverage_col = args.prob_col
                df[coverage_col] = pd.to_numeric(df[coverage_col], errors="coerce")
                print(
                    f"[slowtick] lead={L}: flag '{eff_flag}' is constant "
                    f"(mean={flag_mean:.4f}, frac>0={flag_nonzero:.4f}); using '{coverage_col}' for coverage."
                )

            cov = _coverage_by_hour(df, coverage_col)
            if cov.notna().sum() < max(4, args.min_hours_per_lead):
                print(f"[slowtick] lead={L}: too few hourly points ({cov.notna().sum()}); skipping.")
                continue

            hemi = _coverage_by_hour_hemi(df, coverage_col)
            cov_hash = _hash_series(cov)
            hemi_hash = _hash_hemi(hemi)
            cov_hashes[L] = cov_hash
            hemi_hashes[L] = hemi_hash
            cov_time[L] = cov.sort_index()
            cov_hemi_time[L] = hemi.sort_index()

            time_min = pd.to_datetime(df["time"], errors="coerce").min() if "time" in df.columns else None
            time_max = pd.to_datetime(df["time"], errors="coerce").max() if "time" in df.columns else None
            rows.append(
                dict(
                    run_id=str(args.run_name),
                    lead_h=L,
                    hours=int(cov.notna().sum()),
                    mean_cov=float(np.nanmean(cov.values)),
                    file=Path(path).name if path else "",
                    alerts_source_file=str(path.name) if path else "",
                    filter_mode=str(meta.get("filter_mode", "")),
                    n_rows_alerts_used=int(len(df)),
                    time_min=str(time_min) if time_min is not None else "",
                    time_max=str(time_max) if time_max is not None else "",
                    flag_col=eff_flag,
                    coverage_col=coverage_col,
                    flag_mean=flag_mean,
                    flag_std=flag_std,
                    flag_nonzero_frac=flag_nonzero,
                    flag_const=int(flag_const),
                    series_hash=cov_hash,
                    hemi_hash=hemi_hash,
                    source="alerts",
                )
            )

        # Agent: abort on constant per-lead flags to avoid meaningless slowtick summaries.
        if const_issues and not args.allow_const_flags:
            print("[slowtick] constant per-lead flag detected; aborting.", file=sys.stderr)
            for issue in const_issues:
                print(
                    f"[slowtick] lead={issue['lead_h']} flag={issue['flag_col']} "
                    f"prob_min={issue['prob_min']:.6f} prob_max={issue['prob_max']:.6f} "
                    f"thr={issue['thr_used']} nonzero_frac={issue['flag_nonzero_frac']:.4f}",
                    file=sys.stderr,
                )
                hist = issue.get("hist")
                if hist is not None:
                    counts, bins = hist
                    print(f"[slowtick] prob_hist bins={bins.tolist()} counts={counts.tolist()}", file=sys.stderr)
            raise SystemExit("constant per-lead flag; check thresholds/probability inputs")

    if not rows:
        print("[slowtick] no lead summaries produced; skipping diagnostics.")
        (out_dir / "slowtick_summary.csv").write_text(
            "run_id,lead_h,hours,mean_cov,file,alerts_source_file,filter_mode,n_rows_alerts_used,time_min,time_max,"
            "flag_col,coverage_col,flag_mean,flag_std,flag_nonzero_frac,flag_const,series_hash,hemi_hash,source\n",
            encoding="utf-8",
        )
        return 0
    summary = pd.DataFrame(rows).sort_values("lead_h")
    summary.to_csv(out_dir / "slowtick_summary.csv", index=False)

    # optional global time series
    if args.save_timeseries and cov_time:
        long_rows = []
        for L, s in cov_time.items():
            for t, v in s.items():
                long_rows.append({"lead_h": int(L), "time": pd.to_datetime(t), "coverage": float(v)})
        pd.DataFrame(long_rows).sort_values(["lead_h", "time"]).to_csv(
            out_dir / "coverage_timeseries.csv",
            index=False,
        )
        for L, s in cov_time.items():
            df_lead = pd.DataFrame(
                {
                    "time": pd.to_datetime(s.index),
                    "coverage": pd.to_numeric(s.values, errors="coerce"),
                }
            )
            hemi = cov_hemi_time.get(L)
            if hemi is not None and not hemi.empty:
                hemi_aligned = hemi.reindex(s.index)
                df_lead["cov_north"] = pd.to_numeric(hemi_aligned.get("north"), errors="coerce").to_numpy()
                df_lead["cov_south"] = pd.to_numeric(hemi_aligned.get("south"), errors="coerce").to_numpy()
            out_path = out_dir / f"coverage_timeseries_lead_{int(L)}.csv"
            df_lead.to_csv(out_path, index=False)

    if args.save_hemi_timeseries and cov_hemi_time:
        long_hemi = []
        for L, dfh in cov_hemi_time.items():
            for t, r in dfh.iterrows():
                long_hemi.append(
                    {
                        "lead_h": int(L),
                        "time": pd.to_datetime(t),
                        "cov_north": float(r.get("north", np.nan)),
                        "cov_south": float(r.get("south", np.nan)),
                    }
                )
        pd.DataFrame(long_hemi).sort_values(["lead_h", "time"]).to_csv(
            out_dir / "coverage_timeseries_hemi.csv",
            index=False,
        )

    dup_cov = _duplicate_hashes(cov_hashes)
    dup_hemi = _duplicate_hashes(hemi_hashes) if hemi_hashes else {}
    identical_issue = False
    if dup_cov:
        print(f"[slowtick] identical coverage series across leads: {dup_cov}")
        identical_issue = True
    if dup_hemi:
        print(f"[slowtick] identical hemispheric coverage series across leads: {dup_hemi}")
        identical_issue = True

    if 24 in cov_time and 240 in cov_time:
        s24 = cov_time[24]
        s240 = cov_time[240]
        if cov_hashes.get(24) == cov_hashes.get(240):
            print("[slowtick] coverage series identical for lead 24h and 240h.")
            identical_issue = True
        aligned = pd.DataFrame({"l24": s24, "l240": s240}).dropna()
        if len(aligned) > 3:
            d1 = aligned["l24"].diff().dropna()
            d2 = aligned["l240"].diff().dropna()
            if len(d1) and len(d2):
                corr = np.corrcoef(d1, d2)[0, 1]
                print(f"[slowtick] corr(dcov_24h, dcov_240h)={corr:.4f}")
                if np.isfinite(corr) and abs(corr) > 0.95:
                    identical_issue = True

    if not summary.empty and summary["hours"].nunique() == 1 and len(summary) > 1:
        hours_val = int(summary["hours"].iloc[0])
        print(f"[slowtick] identical per-lead hour counts detected (hours={hours_val}).")
        identical_issue = True

    if identical_issue and not args.allow_identical:
        raise SystemExit("[slowtick] identical per-lead series detected; rerun with --allow-identical to continue.")

    # 2) knee law fit on mean coverage vs lead
    if not summary.empty and (summary["mean_cov"] > 0).sum() >= 2:
        fit = _knee_fit_with_bootstrap(
            summary["lead_h"].values,
            summary["mean_cov"].values,
            B=int(args.bootstrap_B),
        )
        pd.DataFrame([fit]).to_csv(out_dir / "knee_fit.csv", index=False)

        # export fitted curve + plot
        Lx = np.array(summary["lead_h"].values, dtype=float)
        Cx = summary["mean_cov"].values
        plt.figure(figsize=(6.6, 4.2))
        plt.title("Knee law: coverage vs lead")
        plt.loglog(Lx, Cx, "o-", label="mean coverage")
        p = fit.get("p", math.nan)
        if np.isfinite(p):
            Lm, Cm = np.median(Lx), np.median(Cx)
            Cfit = Cm * (Lx / Lm) ** (-2 * p)
            pd.DataFrame({"lead_h": Lx, "coverage_fit": Cfit}).sort_values("lead_h").to_csv(
                out_dir / "knee_fit_curve.csv",
                index=False,
            )
            plt.loglog(
                Lx,
                Cfit,
                "--",
                label=f"fit: p={p:.3f} (95% [{fit['p_lo']:.3f},{fit['p_hi']:.3f}])",
            )
        plt.xlabel("Lead L (hours)")
        plt.ylabel("Coverage")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "knee_fit.png", dpi=160)
        plt.close()

    # 3) hemispheric parity
    parity_rows = []
    for L, hemi in cov_hemi_time.items():
        dif = (hemi["north"] - hemi["south"]).astype(float)
        dif = dif[np.isfinite(dif)]
        if dif.size == 0:
            continue
        mean_diff, (lo, hi) = _bootstrap_ci(
            dif.to_numpy(),
            np.mean,
            B=int(args.bootstrap_B),
            alpha=0.05,
        )
        parity_rows.append(
            dict(
                lead_h=int(L),
                mean_diff=float(mean_diff),
                ci_lo=float(lo),
                ci_hi=float(hi),
                n_hours=int(dif.size),
                series_hash=cov_hashes.get(int(L), ""),
                hemi_hash=hemi_hashes.get(int(L), ""),
            )
        )
        # quick plot
        plt.figure(figsize=(6.6, 3.2))
        plt.title(f"Hemispheric parity delta_cov = cov(N) - cov(S)  (lead {L}h)")
        plt.plot(hemi.index, dif, lw=0.7)
        plt.axhline(0, color="k", lw=0.8)
        plt.ylabel("delta coverage")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(out_dir / f"parity_lead{L}.png", dpi=160)
        plt.close()
    if parity_rows:
        pd.DataFrame(parity_rows).sort_values("lead_h").to_csv(
            out_dir / "parity_summary.csv",
            index=False,
        )

    # 4) slow-tick ridge via FFT of hourly coverage
    ridge_rows = []
    for L, cov in cov_time.items():
        y = cov.astype(float)
        if y.size < max(8, args.min_hours_per_lead):
            continue
        if args.fft_gap_fill and args.fft_gap_fill > 0:
            y = _fill_small_gaps_hourly(y, max_gap_h=int(args.fft_gap_fill))
        y = y - np.nanmean(y)
        yv = np.nan_to_num(y.to_numpy(), nan=0.0)
        w = np.hanning(yv.size)
        yf = np.fft.rfft(yv * w)
        freqs = np.fft.rfftfreq(yv.size, d=1.0)  # cycles per hour
        amps = np.abs(yf)

        f_peak, a_peak = _fft_peak(freqs, amps, target_per_h=24, band=0.20)

        # Diurnal power ratio: power in 24h+/-20% band / total low-f power (>=5h)
        f0 = 1 / 24.0
        band_mask = (freqs >= f0 * (1 - 0.2)) & (freqs <= f0 * (1 + 0.2))
        low_mask = (freqs > 0) & (freqs <= 0.2)
        if low_mask.any() and amps[low_mask].sum() > 0:
            diurnal_ratio = amps[band_mask].sum() / amps[low_mask].sum()
        else:
            diurnal_ratio = math.nan

        ridge_rows.append(
            dict(
                lead_h=int(L),
                peak_freq=f_peak,
                peak_amp=a_peak,
                diurnal_ratio=float(diurnal_ratio) if np.isfinite(diurnal_ratio) else np.nan,
                n_hours=int(y.size),
                series_hash=cov_hashes.get(int(L), ""),
            )
        )

        # spectrum plot (focus 0..0.2 cph ~ periods >=5h)
        plt.figure(figsize=(6.6, 3.6))
        plt.title(f"Hourly coverage spectrum (lead {L}h)")
        plt.plot(freqs, amps)
        plt.xlim(0, 0.2)
        plt.xlabel("Frequency (cycles per hour)")
        plt.ylabel("|FFT| (arbitrary)")
        if np.isfinite(f_peak):
            plt.axvline(f_peak, ls="--", alpha=0.6)
            plt.text(f_peak, 0.9 * amps.max(), f"{f_peak:.3f} cph", ha="left", va="top")
        plt.tight_layout()
        plt.savefig(out_dir / f"spectrum_lead{L}.png", dpi=160)
        plt.close()

    if ridge_rows:
        pd.DataFrame(ridge_rows).sort_values("lead_h").to_csv(
            out_dir / "spectrum_summary.csv",
            index=False,
        )

    print(f"[slowtick] Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
