# backfill_phase_features.py
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat_step_deg", type=float, default=0.25)
    ap.add_argument("--lon_step_deg", type=float, default=0.25)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["time"])
    need_base = ["time","lat","lon","S","agree","msl","zeta","S_mean3h","zeta_mean3h","div_mean3h"]
    missing = [c for c in need_base if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required base columns: {missing}")

    # Deterministic ordering
    df = df.sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    # ---- Static mappings (just renames/proxies) ----
    if "relax" not in df.columns:
        df["relax"] = df["S_mean3h"]
    if "zeta_mean" not in df.columns:
        df["zeta_mean"] = df["zeta_mean3h"]
    if "div_mean" not in df.columns:
        df["div_mean"] = df["div_mean3h"]

    # ---- Pressure gradient magnitude (msl_grad) without .apply ----
    # neighbor values aligned with the same index using groupby+shift
    lat_prev = df.groupby(["time","lon"], sort=False)["msl"].shift(1)
    lat_next = df.groupby(["time","lon"], sort=False)["msl"].shift(-1)
    lon_prev = df.groupby(["time","lat"], sort=False)["msl"].shift(1)
    lon_next = df.groupby(["time","lat"], sort=False)["msl"].shift(-1)

    # centered differences; fall back to one-sided at edges
    dmsl_dlat = (lat_next - lat_prev) / (2*args.lat_step_deg)
    dmsl_dlat = dmsl_dlat.fillna((df["msl"] - lat_prev) / args.lat_step_deg) \
                         .fillna((lon_next - df["msl"]) / args.lat_step_deg)

    dmsl_dlon = (lon_next - lon_prev) / (2*args.lon_step_deg)
    dmsl_dlon = dmsl_dlon.fillna((df["msl"] - lon_prev) / args.lon_step_deg) \
                         .fillna((lon_next - df["msl"]) / args.lon_step_deg)

    # convert deg→meters
    lat_rad = np.deg2rad(df["lat"].to_numpy())
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * np.cos(lat_rad)
    dmsl_dy = dmsl_dlat.to_numpy() / m_per_deg_lat
    dmsl_dx = dmsl_dlon.to_numpy() / np.clip(m_per_deg_lon, 1e-6, None)
    df["msl_grad"] = np.sqrt(dmsl_dx**2 + dmsl_dy**2)

    # ---- Time-derivatives per grid point (aligned via diff) ----
    df["dS_dt"]     = df.groupby(["lat","lon"], sort=False)["S"]     .diff().fillna(0.0)
    df["drelax_dt"] = df.groupby(["lat","lon"], sort=False)["relax"] .diff().fillna(0.0)
    df["dagree_dt"] = df.groupby(["lat","lon"], sort=False)["agree"] .diff().fillna(0.0)

    df.to_csv(args.out, index=False)
    print(
        f"Wrote {args.out} | rows: {len(df)} | new cols added: "
        f"{[c for c in ['relax','zeta_mean','div_mean','msl_grad','dS_dt','drelax_dt','dagree_dt'] if c in df.columns]}"
    )

if __name__ == "__main__":
    main()