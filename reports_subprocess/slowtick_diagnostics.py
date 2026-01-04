#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slowtick_diagnostics.py

Post-process throttled/denoised/base alert grids to quantify:
  1) Knee law: coverage vs lead ~ L^(-2p) with bootstrap CI
  2) Hemispheric parity: N vs S coverage asymmetry with CI
  3) Slow-tick ridge: hourly spectrum peak near diurnal/slow band

Upgrades:
  • Anti-meridian-safe AOI crop
  • Robust file selection (stage preference + latest mtime)
  • Tolerant CSV read + consistent tz-naive UTC
  • Guardrails (--min-hours-per-lead, NaN handling)
  • Exposed bootstrap reps (--bootstrap-B)
  • Optional gap-filling for FFT and hemi time-series export
"""

from __future__ import annotations

import argparse
import math
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
    fallback_df: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str]]:
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
    if thr is not None and prob_col in cols and lead_flag is None:
        use_prob_threshold = True

    # Decide effective flag
    if lead_flag is not None:
        eff_flag = lead_flag
    elif use_prob_threshold:
        eff_flag = f"_prob_thr_{lead}h"
    else:
        eff_flag = requested_flag if requested_flag in cols else None
        if eff_flag is None:
            for alt in ("alert_throttled", "alert_final", "alert"):
                if alt in cols:
                    eff_flag = alt
                    break
        if eff_flag is None:
            eff_flag = requested_flag or "alert_final"

    usecols = [c for c in ("time", "lat", "lon", eff_flag, prob_col) if c in cols] or None
    if using_fallback and fallback_df is not None:
        df = fallback_df
    else:
        df = _load_alert_table(path, time_fmt, norm_lon, aoi, usecols=usecols)

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

    return df, path, eff_flag


def _coverage_by_hour(df: pd.DataFrame, flag_col: str) -> pd.Series:
    tt = df["time"].dt.floor("h")
    return df.assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()


def _coverage_by_hour_hemi(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    tt = df["time"].dt.floor("h")
    north = df[df["lat"] >= 0].assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()
    south = df[df["lat"] < 0].assign(_t=tt).groupby("_t", sort=True)[flag_col].mean()
    both = pd.concat({"north": north, "south": south}, axis=1).astype(float)
    return both.fillna(0.0)


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
    """Fill short NaN runs (≤ max_gap_h) by linear interpolation; leave longer gaps."""
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
    ap.add_argument("--fallback-alerts", default=None, help="Optional alerts file to use when per-lead files are missing.")
    ap.add_argument("--thresholds", default=None, help="Optional thresholds table (CSV/Parquet) for per-lead flags.")
    ap.add_argument("--threshold-col", default="thr_Fbeta", help="Threshold column to use in --thresholds.")
    ap.add_argument("--prob-col", default="prob_viable", help="Probability column for derived per-lead flags.")
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
    ap.add_argument("--fft-gap-fill", type=int, default=2, help="Fill NaN gaps ≤ this many hours before FFT.")
    ap.add_argument("--cache-fallback", action="store_true", help="Cache fallback alerts in memory for reuse.")
    ap.add_argument("--debug", action="store_true", help="Print file/range diagnostics.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = Path(args.alerts_dir)
    fallback_path = Path(args.fallback_alerts) if args.fallback_alerts else None

    thr_map = _load_threshold_map(args.thresholds, args.threshold_col)
    if args.thresholds and not thr_map:
        print("[slowtick] warning: thresholds provided but no lead thresholds parsed.")

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

    # 1) coverage time series per lead
    for L in args.leads:
        df, path, eff_flag = _read_alert_for_lead(
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
            fallback_df=fallback_df,
        )
        if df is None or df.empty or eff_flag is None:
            print(f"[slowtick] lead={L}: no usable alerts file; skipping.")
            continue

        cov = _coverage_by_hour(df, eff_flag)
        if cov.notna().sum() < max(4, args.min_hours_per_lead):
            print(f"[slowtick] lead={L}: too few hourly points ({cov.notna().sum()}); skipping.")
            continue

        hemi = _coverage_by_hour_hemi(df, eff_flag)
        cov_time[L] = cov.sort_index()
        cov_hemi_time[L] = hemi.sort_index()

        rows.append(
            dict(
                lead_h=L,
                hours=int(cov.notna().sum()),
                mean_cov=float(np.nanmean(cov.values)),
                file=Path(path).name if path else "",
                flag_col=eff_flag,
            )
        )

    if not rows:
        print("[slowtick] no lead summaries produced; skipping diagnostics.")
        (out_dir / "slowtick_summary.csv").write_text("lead_h,hours,mean_cov,file,flag_col\n", encoding="utf-8")
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
            )
        )
        # quick plot
        plt.figure(figsize=(6.6, 3.2))
        plt.title(f"Hemispheric parity Δcov = cov(N) - cov(S)  (lead {L}h)")
        plt.plot(hemi.index, dif, lw=0.7)
        plt.axhline(0, color="k", lw=0.8)
        plt.ylabel("Δ coverage")
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

        # Diurnal power ratio: power in 24h±20% band / total low-f power (≥5h)
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
            )
        )

        # spectrum plot (focus 0..0.2 cph ~ periods ≥5h)
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
