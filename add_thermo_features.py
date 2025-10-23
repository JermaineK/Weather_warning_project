#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add thermodynamic & vertical-structure features (with safe fallbacks) to the labelled grid.

Priorities (cross-domain “equivalents”):
  1) Energy availability  → CAPE/CIN or lapse(T850–T500), near-surface buoyancy
  2) Transport & shear    → |V500 - V850|, low-level divergence, moisture convergence proxy
  3) Moisture reservoir   → specific humidity q850/q500, RH2m or TCWV if present
  4) Radiative forcing    → OLR if present (optional placeholder)
All features computed row-wise (no groupby-apply rolling), so no FutureWarnings.

Input:  CSV(.gz) with at least columns: time, lat, lon, plus whatever met fields you have.
Output: CSV(.gz) with new columns added and a brief on-screen summary.

Example:
  python add_thermo_features.py \
    --labelled data/grid_labelled_FMA_gka.csv.gz \
    --out      data/grid_labelled_FMA_gka_thermo.csv.gz
"""

import argparse
import gzip
import sys
import numpy as np
import pandas as pd

# ------- small utilities --------

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

def try_col(df, names):
    """Return first existing column name from list, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None

def safe_hypot(a, b):
    return np.sqrt(np.square(a) + np.square(b))

def dewpoint_to_rh(t2m_K, d2m_K):
    """Approx RH (Magnus). Inputs in Kelvin."""
    # Convert to °C
    t = t2m_K - 273.15
    td = d2m_K - 273.15
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp(a * t  / (b + t))
    e  = 6.1094 * np.exp(a * td / (b + td))
    rh = np.clip((e / es), 0.0, 1.5)  # allow slight >1 due to noise
    return rh

def print_present_or_skip(name, ok, extra=None):
    if ok:
        msg = f"  [+] {name}"
    else:
        msg = f"  [–] {name} (skipping)"
    if extra:
        msg += f"  {extra}"
    print(msg)

# ------- core computation --------

def add_thermo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols = []

    # ---------- Wind / Temp at pressure levels ----------
    u500 = try_col(out, ["u500", "u_500", "u_500hPa", "u500hPa"])
    v500 = try_col(out, ["v500", "v_500", "v_500hPa", "v500hPa"])
    u850 = try_col(out, ["u850", "u_850", "u_850hPa", "u850hPa"])
    v850 = try_col(out, ["v850", "v_850", "v_850hPa", "v850hPa"])

    T500 = try_col(out, ["t500", "T500", "temp500", "temp_500"])
    T850 = try_col(out, ["t850", "T850", "temp850", "temp_850"])

    q500 = try_col(out, ["q500", "q_500", "q_500hPa", "q500hPa", "shum500"])
    q850 = try_col(out, ["q850", "q_850", "q_850hPa", "q850hPa", "shum850"])

    # Near-surface
    t2m  = try_col(out, ["t2m", "T2M", "t2m_K", "t2m_k"])
    d2m  = try_col(out, ["d2m", "D2M", "d2m_K", "d2m_k"])  # dewpoint
    tcwv = try_col(out, ["tcwv", "TCWV", "pw", "precipitable_water"])
    olr  = try_col(out, ["olr", "OLR"])

    # Divergence already in your set?
    div  = try_col(out, ["div_mean", "div", "divergence"])

    print("\n== Thermo Feature Builder ==")
    print_present_or_skip("u/v @500 hPa", u500 and v500)
    print_present_or_skip("u/v @850 hPa", u850 and v850)
    print_present_or_skip("T @500 hPa",   bool(T500))
    print_present_or_skip("T @850 hPa",   bool(T850))
    print_present_or_skip("q @500 hPa",   bool(q500))
    print_present_or_skip("q @850 hPa",   bool(q850))
    print_present_or_skip("2m temp",      bool(t2m))
    print_present_or_skip("2m dewpoint",  bool(d2m))
    print_present_or_skip("TCWV",         bool(tcwv))
    print_present_or_skip("OLR",          bool(olr))
    print_present_or_skip("divergence",   bool(div))

    # 1) Vertical shear |V500 - V850|
    if u500 and v500 and u850 and v850:
        out["thermo_shear_850_500"] = safe_hypot(out[u500] - out[u850], out[v500] - out[v850])
        new_cols.append("thermo_shear_850_500")
    else:
        print_present_or_skip("thermo_shear_850_500", False)

    # 2) Lapse proxy ΔT = T850 - T500 (K)  → buoyancy sign
    if T850 and T500:
        out["thermo_lapse_850_500"] = out[T850] - out[T500]
        new_cols.append("thermo_lapse_850_500")
    else:
        print_present_or_skip("thermo_lapse_850_500", False)

    # 3) CAPE/CIN if already present; else buoyancy proxy CAPE*
    cape_col = try_col(out, ["cape", "CAPE"])
    cin_col  = try_col(out, ["cin", "CIN"])

    if cape_col:
        out["thermo_cape"] = out[cape_col]
        new_cols.append("thermo_cape")
    elif T850 and t2m:
        # Simple “excess surface buoyancy” proxy; dimensionless scale factor.
        g = 9.81
        capep = np.maximum(0.0, g * (out[t2m] - out[T850]) / (out[T850].replace(0, np.nan)))
        out["thermo_cape_proxy"] = capep.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        new_cols.append("thermo_cape_proxy")
        print_present_or_skip("thermo_cape_proxy", True, "(fallback)")
    else:
        print_present_or_skip("thermo_cape/proxy", False)

    if cin_col:
        out["thermo_cin"] = out[cin_col]
        new_cols.append("thermo_cin")

    # 4) Moisture reservoir: q850 preferred; otherwise RH2m or TCWV
    if q850:
        out["thermo_q850"] = out[q850]
        new_cols.append("thermo_q850")
    elif t2m and d2m:
        out["thermo_rh2m"] = dewpoint_to_rh(out[t2m], out[d2m])
        new_cols.append("thermo_rh2m")
        print_present_or_skip("thermo_rh2m", True, "(fallback from t2m,d2m)")
    elif tcwv:
        out["thermo_tcwv"] = out[tcwv]
        new_cols.append("thermo_tcwv")
        print_present_or_skip("thermo_tcwv", True, "(fallback)")

    # 5) Moisture flux convergence proxy: - q850 * div (units simplified)
    if q850 and div:
        out["thermo_mfc_proxy"] = - out[q850] * out[div]
        new_cols.append("thermo_mfc_proxy")
    elif div and t2m and d2m:
        # fallback: use RH2m as q proxy
        rh = dewpoint_to_rh(out[t2m], out[d2m])
        out["thermo_mfc_proxy"] = - rh * out[div]
        new_cols.append("thermo_mfc_proxy")
        print_present_or_skip("thermo_mfc_proxy", True, "(fallback via RH2m)")
    else:
        print_present_or_skip("thermo_mfc_proxy", False)

    # 6) Relative humidity gradient proxy (if explicit grads exist)
    # try various naming patterns in case you already had grads
    qx = try_col(out, ["q850_dx", "dqdx_850", "dq850_dx"])
    qy = try_col(out, ["q850_dy", "dqdy_850", "dq850_dy"])
    if qx and qy:
        out["thermo_gradq850"] = safe_hypot(out[qx], out[qy])
        new_cols.append("thermo_gradq850")
    else:
        print_present_or_skip("thermo_gradq850", False)

    # 7) OLR (optional pass-through if present)
    if olr:
        out["thermo_olr"] = out[olr]
        new_cols.append("thermo_olr")

    return out, new_cols

def summarize(df, new_cols, target_col="pregen"):
    print("\n== Thermo Feature Summary ==")
    print(f"Rows: {len(df):,}")
    if not new_cols:
        print("No new columns added.")
        return
    print(f"New columns added ({len(new_cols)}): {', '.join(new_cols)}")

    # Percentiles
    print("\nPercentiles (new features, p05/p50/p95):")
    pct = df[new_cols].quantile([0.05, 0.50, 0.95])
    with pd.option_context("display.float_format", "{:0.6f}".format):
        print(pct)

    if target_col in df.columns:
        print(f"\nPearson r vs. {target_col} (coincident):")
        y = df[target_col].astype(float).to_numpy()
        for c in new_cols:
            x = df[c].astype(float).to_numpy()
            # robust against degenerate std
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(np.nan_to_num(x), np.nan_to_num(y))[0,1])
            print(f"  {c:18s} r={r:+0.3f}")

def main():
    ap = argparse.ArgumentParser(description="Add thermodynamic & vertical-structure features (safe fallbacks).")
    ap.add_argument("--labelled", required=True, help="Input labelled CSV(.gz)")
    ap.add_argument("--out",      required=False, default=None, help="Output CSV(.gz); defaults to *_thermo.csv.gz")
    ap.add_argument("--target",   required=False, default="pregen", choices=["pregen","storm","near_storm"])
    args = ap.parse_args()

    in_path = args.labelled
    out_path = args.out or in_path.replace(".csv.gz", "_thermo.csv.gz").replace(".csv", "_thermo.csv.gz")

    # Read with parsed datetimes but do not coerce timezones
    df = pd.read_csv(in_path, parse_dates=["time"])
    # Ensure consistent dtypes for lat/lon (helpful downstream)
    if "lat" in df.columns: df["lat"] = df["lat"].astype(float)
    if "lon" in df.columns: df["lon"] = df["lon"].astype(float)

    out, new_cols = add_thermo_features(df)
    summarize(out, new_cols, target_col=args.target)

    # Write
    out.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {out_path}  | rows={len(out):,}")

if __name__ == "__main__":
    main()