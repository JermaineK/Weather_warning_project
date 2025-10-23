#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compute_gka_features.py

Adds lightweight "GKA-style" geometric features to an existing labelled grid file.

Inputs (expected columns, best-effort fallbacks):
- time (datetime), lat, lon
- S (scalar spiral score), relax, agree
- zeta_mean (or zeta), div_mean (or div)
- msl (mean sea level pressure)
- shear_proxy (optional)
- pregen (optional, used only for quick correlations in summary)

Outputs:
- Writes a CSV/CSV.GZ with new columns prefixed "gka_"
- Prints a compact summary to stdout (feature percentiles, correlations)

Usage example:
  python compute_gka_features.py --infile "data\\grid_labelled_FMA_phasecols.csv.gz" \
                                 --outfile "data\\grid_labelled_FMA_gka.csv.gz"
"""

import argparse
import sys
import numpy as np
import pandas as pd

NEW_COLS = [
    "gka_kappa",
    "gka_tau",
    "gka_parity_eta",
    "gka_A_overlap",
    "gka_F",
    "gka_msl_nd",
    "gka_knee_ratio",
]

def _get(df, primary, fallback=None):
    """Return df[primary] if it exists, else df[fallback] if given, else None."""
    if primary in df.columns:
        return df[primary]
    if fallback and fallback in df.columns:
        return df[fallback]
    return None


def add_gka_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a small set of GKA-inspired features using columns already in the project files.
    All outputs are appended to a copy of df and returned.
    """
    out = df.copy()

    # --- Base signals with graceful fallbacks ---
    S        = _get(out, "S")
    relax    = _get(out, "relax")
    agree    = _get(out, "agree")
    zeta     = _get(out, "zeta_mean", "zeta")
    divv     = _get(out, "div_mean", "div")
    msl      = _get(out, "msl")
    shear_p  = _get(out, "shear_proxy")
    S_mean3h = _get(out, "S_mean3h")

    # 1) Curvature / torsion proxies (vorticity & divergence)
    out["gka_kappa"] = (zeta.to_numpy(float) if zeta is not None else np.zeros(len(out), dtype=float))
    out["gka_tau"]   = (-(divv.to_numpy(float)) if divv is not None else np.zeros(len(out), dtype=float))

    # 2) Parity-odd channel eta: time-smoothed sign(zeta) per (lat,lon)
    if zeta is not None and "lat" in out.columns and "lon" in out.columns:
        z_vals = zeta.to_numpy(float)
        sgn = np.sign(z_vals)
        s = pd.Series(sgn, index=out.index)
        # 7-hour centered window (~±3h) per gridpoint
        eta = s.groupby([out["lat"], out["lon"]], sort=False)\
               .transform(lambda g: g.rolling(window=7, min_periods=1, center=True).mean())
        out["gka_parity_eta"] = eta.to_numpy()
    else:
        out["gka_parity_eta"] = np.zeros(len(out), dtype=float)

    # 3) Overlap / coherence proxy: reuse 'agree' (already a consensus-like score)
    out["gka_A_overlap"] = (agree.to_numpy(float) if agree is not None else np.zeros(len(out), dtype=float))

    # 4) Freedom of movement proxy: robust-normalized shear, then logistic squashing
    if shear_p is not None:
        shp = shear_p.to_numpy(float)
        med = float(np.median(shp))
        mad = float(np.mean(np.abs(shp - med))) + 1e-6
        z = (shp - med) / mad
        out["gka_F"] = 1.0 / (1.0 + np.exp(z))  # in (0,1)
    else:
        out["gka_F"] = np.zeros(len(out), dtype=float)

    # 5) Dimensionless MSL (median-centered, manual MAD)
    if msl is not None:
        msl_arr = msl.to_numpy(float)
        m_med = float(np.median(msl_arr))
        m_mad = float(np.mean(np.abs(msl_arr - m_med))) + 1e-6
        out["gka_msl_nd"] = (msl_arr - m_med) / m_mad
    else:
        out["gka_msl_nd"] = np.zeros(len(out), dtype=float)

    # 6) "Knee" proxy: ratio S / S_mean3h (if available)
    if (S is not None) and (S_mean3h is not None):
        denom = S_mean3h.to_numpy(float)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        out["gka_knee_ratio"] = S.to_numpy(float) / denom
    else:
        out["gka_knee_ratio"] = np.zeros(len(out), dtype=float)

    return out


def print_summary(df_before: pd.DataFrame, df_after: pd.DataFrame):
    added = [c for c in df_after.columns if c not in df_before.columns and c.startswith("gka_")]
    print(f"\n== GKA Feature Summary ==")
    print(f"Rows: {len(df_after):,}")
    print(f"New columns added ({len(added)}): {', '.join(added) if added else '(none)'}")

    if added:
        q = df_after[added].quantile([0.05, 0.5, 0.95])
        q.index = ["p05", "p50", "p95"]
        print("\nPercentiles (new features):")
        with pd.option_context("display.width", 120, "display.max_columns", 200):
            print(q)

    if "pregen" in df_after.columns and added:
        y = df_after["pregen"].astype(float).to_numpy()
        print("\nPearson r vs. pregen (coincident):")
        for c in added:
            x = df_after[c].to_numpy(float)
            r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
            print(f"  {c:16s}  r={r: .3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True, help="Input labelled CSV/CSV.GZ")
    ap.add_argument("--outfile", required=True, help="Output CSV/CSV.GZ with GKA features appended")
    args = ap.parse_args()

    df0 = pd.read_csv(args.infile, parse_dates=["time"])
    df1 = add_gka_features(df0)
    df1.to_csv(args.outfile, index=False)

    print(f"Wrote {args.outfile}  rows: {len(df1):,}")
    print_summary(df0, df1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)