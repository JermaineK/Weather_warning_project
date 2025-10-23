#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slowtick_diagnostics.py
Post-process throttled alert grids to quantify:
  1) Knee law: coverage vs lead ~ L^(-2p) with bootstrap CI
  2) Hemispheric parity: N vs S coverage asymmetry with CI
  3) Slow-tick ridge: hourly spectrum peak near diurnal/slow band

Inputs:
  --alerts-dir  directory with throttled/denoised/base alerts
  --run-name    run name used in alert filenames
  --leads       list of leads (hours) to include
  --flag-col    alert flag column (default: alert_final)
  --out-dir     directory for CSV/plots

Notes:
  • No geographic restriction: reads *all* lat/lon unless you explicitly pass --area.
  • Robust to differing flag columns per-file; will fall back to alert_throttled, alert_final, alert (in that order).
  • Supports lon normalization and optional AOI cropping.
"""

import argparse, os, math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- shared helpers ----------

def _try_parse_time_raw(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (mid and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def _parse_area(aoi: Optional[str]):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")


# ---------- file I/O ----------

def _read_alert_for_lead(alerts_dir: Path,
                         run: str,
                         lead: int,
                         requested_flag: str,
                         time_fmt: Optional[str],
                         norm_lon: str,
                         aoi: Optional[str],
                         debug: bool) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str]]:
    """
    Locate the preferred alerts file for a given lead, load it, normalize time/lat/lon,
    determine the effective flag column (with fallbacks), and return (df, path, eff_flag).
    """
    pattern = f"alerts_{run}_lead{lead}_thr*.csv.gz"
    cand = sorted(alerts_dir.glob(pattern))
    if not cand:
        return None, None, None

    # Preference: throttled → denoised → base
    prefer = [p for p in cand if p.name.endswith("_throttled.csv.gz")] \
             or [p for p in cand if p.name.endswith("_denoised.csv.gz")] \
             or cand
    path = prefer[0]

    # Peek to decide effective flag column
    try:
        head = pd.read_csv(path, nrows=1)
        cols = set(head.columns)
    except Exception:
        cols = set()

    eff_flag = requested_flag
    if eff_flag not in cols:
        for alt in ("alert_throttled", "alert_final", "alert"):
            if alt in cols:
                eff_flag = alt
                break

    # Read minimally-needed columns if possible
    usecols = None
    try:
        possible = {"time", "lat", "lon", eff_flag}
        usecols = [c for c in possible if c in cols] if cols else None
    except Exception:
        pass

    df = pd.read_csv(path, compression="infer", usecols=usecols).replace([np.inf, -np.inf], np.nan)

    # Time/coords/flag normalization
    df["time"] = _try_parse_time_raw(df["time"], time_fmt)
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _norm_lon(df["lon"], norm_lon)
    if eff_flag not in df.columns:
        # synthesize zero column if completely missing (should be rare)
        df[eff_flag] = 0
    df[eff_flag] = pd.to_numeric(df[eff_flag], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Optional AOI crop (default: no crop → global)
    a = _parse_area(aoi)
    if a:
        latN, lonW, latS, lonE = a
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)

    if debug and not df.empty:
        tt = df["time"].dt.floor("h")
        print(f"[slowtick][debug] lead={lead} file={path.name} "
              f"| hours={tt.nunique()} time={tt.min()}→{tt.max()} "
              f"| lat={df['lat'].min():.2f}..{df['lat'].max():.2f} "
              f"| lon={df['lon'].min():.2f}..{df['lon'].max():.2f} "
              f"| flag_col={eff_flag}")

    return df, path, eff_flag


def _coverage_by_hour(df: pd.DataFrame, flag_col: str) -> pd.Series:
    tt = df["time"].dt.floor("h")
    return df.assign(t=tt).groupby("t", sort=True)[flag_col].mean()


def _coverage_by_hour_hemi(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    tt = df["time"].dt.floor("h")
    north = df[df["lat"] >= 0].assign(t=tt).groupby("t", sort=True)[flag_col].mean()
    south = df[df["lat"]  < 0].assign(t=tt).groupby("t", sort=True)[flag_col].mean()
    both = pd.concat({"north": north, "south": south}, axis=1).fillna(0.0)
    return both


# ---------- stats helpers ----------

def _bootstrap_ci(values: np.ndarray, stat_fn, B=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return math.nan, (math.nan, math.nan)
    bs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        bs[b] = stat_fn(values[idx])
    lo = np.quantile(bs, alpha/2)
    hi = np.quantile(bs, 1 - alpha/2)
    return float(stat_fn(values)), (float(lo), float(hi))

def _fit_knee(logL, logCov):
    X = np.c_[np.ones_like(logL), logL]
    beta, *_ = np.linalg.lstsq(X, logCov, rcond=None)
    a, b = beta
    p = -b / 2.0
    return float(p), float(a), float(b)

def _knee_fit_with_bootstrap(leads, mean_cov, B=1000):
    mask = (np.array(leads) > 0) & (np.array(mean_cov) > 0)
    L = np.array(leads)[mask]
    C = np.array(mean_cov)[mask]
    if len(L) < 2:
        return dict(p=math.nan, p_lo=math.nan, p_hi=math.nan, used_points=int(mask.sum()))
    logL = np.log(L)
    logC = np.log(C)
    p_hat, a, b = _fit_knee(logL, logC)

    rng = np.random.default_rng(123)
    ps = []
    for _ in range(B):
        idx = rng.integers(0, len(L), size=len(L))
        p_b, *_ = _fit_knee(logL[idx], logC[idx])
        ps.append(p_b)
    ps = np.array(ps)
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


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Slow-tick diagnostics from throttled/denoised/base alerts.")
    ap.add_argument("--alerts-dir", required=True, help="Directory with alerts (throttled preferred).")
    ap.add_argument("--run-name", required=True, help="Run name used in filenames.")
    ap.add_argument("--leads", type=int, nargs="+", required=True, help="Lead hours to include.")
    ap.add_argument("--flag-col", default="alert_final", help="Preferred flag column (tries fallbacks automatically).")
    ap.add_argument("--out-dir", default="results/slowtick", help="Output directory for CSVs/plots.")
    ap.add_argument("--time-format", default=None, help="Optional strftime for custom time parsing.")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="Normalize longitudes in input files (default: none).")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--debug", action="store_true", help="Print file/range diagnostics.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = Path(args.alerts_dir)

    rows = []
    cov_time = {}
    cov_hemi_time = {}

    # 1) gather coverage time series per lead (global by default)
    for L in args.leads:
        df, path, eff_flag = _read_alert_for_lead(
            alerts_dir, args.run_name, L, args.flag_col,
            time_fmt=args.time_format, norm_lon=args.normalize_lon, aoi=args.area, debug=args.debug
        )
        if df is None or df.empty or eff_flag is None:
            print(f"[slowtick] missing/empty alerts for lead={L}; skipping.")
            continue

        cov = _coverage_by_hour(df, eff_flag)
        hemi = _coverage_by_hour_hemi(df, eff_flag)
        cov_time[L] = cov
        cov_hemi_time[L] = hemi

        rows.append(dict(lead_h=L, hours=int(len(cov)), mean_cov=float(cov.mean()), file=Path(path).name,
                         flag_col=eff_flag))

    summary = pd.DataFrame(rows).sort_values("lead_h")
    summary.to_csv(out_dir / "slowtick_summary.csv", index=False)

    # 2) knee law fit on mean coverage vs lead (log-log)
    if not summary.empty:
        fit = _knee_fit_with_bootstrap(summary["lead_h"].values, summary["mean_cov"].values, B=1000)
        pd.DataFrame([fit]).to_csv(out_dir / "knee_fit.csv", index=False)

        # plot: mean coverage vs lead + power-law fit
        plt.figure(figsize=(6.5, 4.2))
        plt.title("Knee law: coverage vs lead")
        plt.loglog(summary["lead_h"], summary["mean_cov"], "o-", label="mean coverage")
        p = fit["p"]
        if not math.isnan(p):
            Lx = np.array(summary["lead_h"].values, dtype=float)
            Cx = summary["mean_cov"].values
            Lm, Cm = np.median(Lx), np.median(Cx)
            Cfit = Cm * (Lx / Lm) ** (-2*p)
            plt.loglog(Lx, Cfit, "--", label=f"fit: p={p:.3f} (95% [{fit['p_lo']:.3f},{fit['p_hi']:.3f}])")
        plt.xlabel("Lead L (hours)")
        plt.ylabel("Coverage")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "knee_fit.png", dpi=160)
        plt.close()

    # 3) hemispheric parity (per hour diff, bootstrap CI on mean diff)
    parity_rows = []
    for L, hemi in cov_hemi_time.items():
        dif = (hemi["north"] - hemi["south"]).dropna().to_numpy()
        if dif.size == 0:
            continue
        mean_diff, (lo, hi) = _bootstrap_ci(dif, np.mean, B=1000, alpha=0.05)
        parity_rows.append(dict(lead_h=L, mean_diff=float(mean_diff), ci_lo=float(lo), ci_hi=float(hi),
                                n_hours=int(dif.size)))
        # quick plot
        plt.figure(figsize=(6.5, 3.2))
        plt.title(f"Hemispheric parity Δcov = cov(N) - cov(S)  (lead {L}h)")
        plt.plot(hemi.index, dif, lw=0.7)
        plt.axhline(0, color="k", lw=0.8)
        plt.ylabel("Δ coverage")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(out_dir / f"parity_lead{L}.png", dpi=160)
        plt.close()
    if parity_rows:
        pd.DataFrame(parity_rows).sort_values("lead_h").to_csv(out_dir / "parity_summary.csv", index=False)

    # 4) slow-tick ridge via FFT of hourly coverage
    ridge_rows = []
    for L, cov in cov_time.items():
        y = cov.to_numpy(dtype=float)
        if y.size < 8:
            continue
        y = y - y.mean()
        w = np.hanning(y.size)
        yf = np.fft.rfft(y * w)
        freqs = np.fft.rfftfreq(y.size, d=1.0)  # cycles per hour
        amps = np.abs(yf)
        f_peak, a_peak = _fft_peak(freqs, amps, target_per_h=24, band=0.20)
        ridge_rows.append(dict(lead_h=L, peak_freq=f_peak, peak_amp=a_peak, n_hours=int(y.size)))

        # spectrum plot (focus 0..0.2 cph ~ up to 5h period)
        plt.figure(figsize=(6.5, 3.6))
        plt.title(f"Hourly coverage spectrum (lead {L}h)")
        plt.plot(freqs, amps)
        plt.xlim(0, 0.2)
        plt.xlabel("Frequency (cycles per hour)")
        plt.ylabel("|FFT|")
        if not math.isnan(f_peak):
            plt.axvline(f_peak, ls="--", alpha=0.6)
            plt.text(f_peak, 0.9*amps.max(), f"{f_peak:.3f} cph", ha="left", va="top")
        plt.tight_layout()
        plt.savefig(out_dir / f"spectrum_lead{L}.png", dpi=160)
        plt.close()

    if ridge_rows:
        pd.DataFrame(ridge_rows).sort_values("lead_h").to_csv(out_dir / "spectrum_summary.csv", index=False)

    print(f"[slowtick] Wrote summaries to {out_dir}")

if __name__ == "__main__":
    main()