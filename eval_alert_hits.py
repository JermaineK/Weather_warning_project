#!/usr/bin/env python3
import argparse, sys, os
import numpy as np
import pandas as pd

# ---------- helpers (aligned with other scripts) ----------

CANDIDATE_TIME_COLS = ["time", "time_h", "datetime", "valid_time", "forecast_time"]

def _try_parse_time_raw(s: pd.Series, fmt: str | None) -> pd.Series:
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

def _parse_area(aoi: str | None):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")

def future_max_label_by_point(df: pd.DataFrame, target_col: str, hours: int) -> np.ndarray:
    """
    For each (lat,lon,time) row, set 1 if target_col==1 occurs within next `hours`
    for the SAME (lat,lon). Uses reversed rolling max over integer hour steps.
    """
    def _per_point(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["time"]).sort_values("time", kind="mergesort")
        s = pd.to_numeric(g[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        sr = s[::-1]
        r  = pd.Series(sr).rolling(window=int(hours), min_periods=1).max()
        win = r.shift(1).fillna(0).astype(int).to_numpy()[::-1]
        return pd.Series(win, index=g.index, name="y_future")

    out = (
        df[["time","lat","lon",target_col]]
          .groupby(["lat","lon"], sort=False, group_keys=False)
          .apply(_per_point)
    )
    return out.reindex(df.index).fillna(0).to_numpy(dtype=int)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate alert hits vs future labels at a given lead.")
    ap.add_argument("--labelled", required=True, help="Labelled grid CSV(.gz)/Parquet (with target column).")
    ap.add_argument("--alerts",   required=True, help="Alerts CSV(.gz) with flags for (time,lat,lon).")
    ap.add_argument("--lead-hours", type=int, required=True, help="Lead window (hours) for future label.")
    ap.add_argument("--target", default="pregen", help="Target column in labelled file (default: pregen).")
    ap.add_argument("--flag-col", default="alert_throttled",
                    help="Flag column in alerts (default: alert_throttled). Fallbacks: alert_final, alert.")
    ap.add_argument("--time-col", default="time", help="Time column (default: time).")
    ap.add_argument("--lat-col",  default="lat",  help="Latitude column (default: lat).")
    ap.add_argument("--lon-col",  default="lon",  help="Longitude column (default: lon).")
    ap.add_argument("--time-format", default=None, help="Optional strftime for custom time parsing.")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="Normalize longitudes for BOTH files (default: none).")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--save-metrics", default=None,
                    help="Optional CSV path to save a one-row metrics summary (precision, recall, F1, coverage, TP, FP, FN).")
    ap.add_argument("--debug", action="store_true", help="Print ranges and overlap diagnostics.")
    args = ap.parse_args()

    # Load labelled (CSV/Parquet)
    labelled_path = args.labelled
    if str(labelled_path).lower().endswith((".parquet", ".pq")):
        base = pd.read_parquet(labelled_path)
    else:
        base = pd.read_csv(labelled_path, compression="infer")
    base = base.replace([np.inf,-np.inf], np.nan)

    # Alerts
    alrt = pd.read_csv(args.alerts, compression="infer").replace([np.inf,-np.inf], np.nan)

    # Pick time column if missing
    if args.time_col not in base.columns:
        for c in CANDIDATE_TIME_COLS:
            if c in base.columns:
                args.time_col = c; break
    if args.time_col not in alrt.columns:
        for c in CANDIDATE_TIME_COLS:
            if c in alrt.columns:
                args.time_col = c; break

    # Validate essentials
    needed_base = [args.time_col, args.lat_col, args.lon_col, args.target]
    for c in needed_base:
        if c not in base.columns:
            sys.exit(f"[ERROR] labelled file missing column: {c}")

    # Pick/validate flag col with fallbacks
    flag_col = args.flag_col
    if flag_col not in alrt.columns:
        for alt in ["alert_final", "alert"]:
            if alt in alrt.columns:
                flag_col = alt; break
    if flag_col not in alrt.columns:
        sys.exit(f"[ERROR] alerts file missing flag column: {args.flag_col} (also tried 'alert_final','alert')")

    needed_alrt = [args.time_col, args.lat_col, args.lon_col, flag_col]
    for c in needed_alrt:
        if c not in alrt.columns:
            sys.exit(f"[ERROR] alerts file missing column: {c}")

    # Robust time parse to tz-naive UTC
    base_t = _try_parse_time_raw(base[args.time_col], args.time_format)
    alrt_t = _try_parse_time_raw(alrt[args.time_col], args.time_format)
    base = base.loc[base_t.notna()].copy(); base[args.time_col] = base_t[base_t.notna()]
    alrt = alrt.loc[alrt_t.notna()].copy(); alrt[args.time_col] = alrt_t[alrt_t.notna()]

    # Numeric cleanups and lon normalization
    base[args.lat_col] = pd.to_numeric(base[args.lat_col], errors="coerce")
    base[args.lon_col] = _norm_lon(base[args.lon_col], args.normalize_lon)
    alrt[args.lat_col] = pd.to_numeric(alrt[args.lat_col], errors="coerce")
    alrt[args.lon_col] = _norm_lon(alrt[args.lon_col], args.normalize_lon)
    alrt[flag_col]     = pd.to_numeric(alrt[flag_col], errors="coerce").fillna(0).astype(int)

    base = base.dropna(subset=[args.lat_col, args.lon_col, args.target]).reset_index(drop=True)
    alrt = alrt.dropna(subset=[args.lat_col, args.lon_col]).reset_index(drop=True)

    # Optional AOI crop
    aoi = _parse_area(args.area)
    if aoi:
        latN, lonW, latS, lonE = aoi
        base = base.loc[(base[args.lat_col] <= latN) & (base[args.lat_col] >= latS) &
                        (base[args.lon_col] >= lonW) & (base[args.lon_col] <= lonE)].reset_index(drop=True)
        alrt = alrt.loc[(alrt[args.lat_col] <= latN) & (alrt[args.lat_col] >= latS) &
                        (alrt[args.lon_col] >= lonW) & (alrt[args.lon_col] <= lonE)].reset_index(drop=True)

    # Deterministic order
    base.sort_values([args.time_col, args.lat_col, args.lon_col], inplace=True, kind="mergesort", ignore_index=True)
    alrt.sort_values([args.time_col, args.lat_col, args.lon_col], inplace=True, kind="mergesort", ignore_index=True)

    # Merge flags onto base (many base rows → ≤1 alert row)
    df = base[[args.time_col, args.lat_col, args.lon_col, args.target]].merge(
        alrt[[args.time_col, args.lat_col, args.lon_col, flag_col]],
        on=[args.time_col, args.lat_col, args.lon_col],
        how="left",
        validate="many_to_one"
    )
    df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)

    # Debug info
    if args.debug:
        def _rng(s): return (s.min(), s.max(), s.nunique())
        b_lo, b_hi, b_nh = _rng(df[args.time_col].dt.floor("h"))
        a_lo, a_hi, a_nh = _rng(alrt[args.time_col].dt.floor("h"))
        print(f"[eval][debug] labelled hours: {b_lo} → {b_hi} ({b_nh})")
        print(f"[eval][debug] alerts   hours: {a_lo} → {a_hi} ({a_nh})")
        print(f"[eval][debug] lat range: {df[args.lat_col].min():.3f} .. {df[args.lat_col].max():.3f}")
        print(f"[eval][debug] lon range: {df[args.lon_col].min():.3f} .. {df[args.lon_col].max():.3f}")

    # Build future label within lead window per (lat,lon)
    print(f"Preparing future labels for lead +{args.lead_hours}h …")
    y_future = future_max_label_by_point(
        df.rename(columns={args.time_col:"time", args.lat_col:"lat", args.lon_col:"lon"}),
        args.target,
        int(args.lead_hours)
    )

    # Metrics
    flags = df[flag_col].to_numpy(dtype=int)
    ytrue = y_future.astype(int)

    tp = int(((flags == 1) & (ytrue == 1)).sum())
    fp = int(((flags == 1) & (ytrue == 0)).sum())
    fn = int(((flags == 0) & (ytrue == 1)).sum())
    tn = int(((flags == 0) & (ytrue == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    cov  = flags.mean() if len(flags) else 0.0

    print(f"Lead +{args.lead_hours}h  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}  Coverage={cov:.3f}")

    # Optional metrics CSV
    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        pd.DataFrame([{
            "lead_h": int(args.lead_hours),
            "precision": float(prec),
            "recall": float(rec),
            "F1": float(f1),
            "coverage": float(cov),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "alerts_file": os.path.basename(args.alerts),
            "labelled_file": os.path.basename(args.labelled),
            "flag_col": flag_col,
            "target": args.target,
        }]).to_csv(args.save_metrics, index=False)
        print(f"[eval] wrote metrics → {args.save_metrics}")

if __name__ == "__main__":
    main()