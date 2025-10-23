# compute_spherical_feedback.py
import argparse
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Compute spherical-feedback features (center-ness, radial wind alignment, optional lightning density) and SFI.")
    ap.add_argument("--labelled", required=True, help="grid_labelled_*.csv.gz (needs: time, lat, lon, msl, u10, v10)")
    ap.add_argument("--lightning", default="", help="Optional CSV/GZ with columns: time,lat,lon[,flashes]; counts or 0/1")
    ap.add_argument("--neighbor-step", type=float, default=0.0, help="Grid step in degrees; 0=auto (median spacing)")
    ap.add_argument("--radius-cells", type=int, default=1, help="Neighborhood radius in grid cells (1 = 8-neighbors)")
    ap.add_argument("--lightning-radius", type=float, default=0.3, help="Radial search (deg) for lightning density")
    ap.add_argument("--lead-hours", type=int, default=24, help="For quick coincident corr to pregen (if present)")
    ap.add_argument("--out", default="results/spherical_feedback.csv.gz")
    return ap.parse_args()

def robust01(x):
    x = np.asarray(x, float)
    q1, q99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    return np.clip((x - q1) / (q99 - q1 + 1e-12), 0, 1)

def infer_step(vals):
    v = np.sort(np.unique(np.asarray(vals, float)))
    if len(v) < 3:
        return 0.25
    d = np.diff(v)
    # use median of positive diffs
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.25

def per_time_neighbors_block(sub, step, radius_cells):
    # Build coordinate index for this hour
    idx = sub.index.to_numpy()
    lats = sub["lat"].to_numpy()
    lons = sub["lon"].to_numpy()
    msl  = sub["msl"].to_numpy()
    u10  = sub["u10"].to_numpy()
    v10  = sub["v10"].to_numpy()

    pos = {(float(lats[i]), float(lons[i])): i for i in range(len(idx))}
    # precompute neighbor offsets
    offs = []
    r = radius_cells
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0:
                continue
            offs.append((dy*step, dx*step))

    # outputs
    center_score = np.zeros(len(idx), dtype=float)  # low vs mean neighbor low
    radial_align = np.zeros(len(idx), dtype=float)  # mean cos(theta) (signed)
    neigh_count  = np.zeros(len(idx), dtype=int)

    # compute per point
    for i in range(len(idx)):
        lat0, lon0 = lats[i], lons[i]
        m0 = msl[i]
        u, v = u10[i], v10[i]
        spd = np.hypot(u, v) + 1e-12

        sum_m = 0.0; n_m = 0
        sum_align = 0.0; n_a = 0

        for (dlat, dlon) in offs:
            j = pos.get((lat0 + dlat, lon0 + dlon))
            if j is None: 
                continue
            # neighbors same time
            sum_m += msl[j]; n_m += 1

            # radial direction from this point to neighbor (outward)
            # vector on lat/lon grid; we ignore metric scaling since stencil small
            rx = (lons[j] - lon0); ry = (lats[j] - lat0)
            rn = np.hypot(rx, ry)
            if rn > 0:
                # project wind onto radial (outward = +1, inward = -1)
                # wind vector at THIS point
                cos_th = (u * (rx / rn) + v * (ry / rn)) / spd
                sum_align += cos_th
                n_a += 1

        center_score[i] = (np.mean(msl[i:i+1]) - (sum_m / n_m)) if n_m > 0 else 0.0
        radial_align[i] = (sum_align / n_a) if n_a > 0 else 0.0
        neigh_count[i]  = n_m

    # normalize: we want "more center-like" = lower pressure than neighbors → positive score
    cs_norm = robust01(-center_score)  # negate so deep lows => higher normalized value
    # radial alignment: use magnitude and sign; store signed and absolute
    ra_signed = np.tanh(radial_align)     # keep sign but squash
    ra_abs    = np.abs(ra_signed)
    return cs_norm, ra_signed, ra_abs, neigh_count

def lightning_density_block(sub, lgt, radius_deg):
    # sub: rows for a given hour; lgt: lightning rows for same hour
    if lgt is None or lgt.empty:
        return np.zeros(len(sub), dtype=float)
    lat0 = sub["lat"].to_numpy()
    lon0 = sub["lon"].to_numpy()
    latL = lgt["lat"].to_numpy()
    lonL = lgt["lon"].to_numpy()
    # naive ball search (ok at ~5k points/hour); can be sped up with k-d tree if needed
    out = np.zeros(len(sub), dtype=float)
    for i in range(len(sub)):
        dlat = np.abs(latL - lat0[i])
        dlon = np.abs(lonL - lon0[i])
        hit = (dlat <= radius_deg) & (dlon <= radius_deg)
        if np.any(hit):
            # weight by 1 / (1 + distance) to get a soft density
            dist = np.hypot(dlat[hit], dlon[hit])
            out[i] = float(np.sum(1.0 / (1.0 + dist)))
    return out

def main():
    args = parse_args()
    print("== Spherical Feedback Features ==", flush=True)
    print(f"Labelled : {args.labelled}", flush=True)
    if args.lightning:
        print(f"Lightning: {args.lightning}", flush=True)
    print(f"Neighbor radius (cells): {args.radius_cells}", flush=True)

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    need = {"time","lat","lon","msl","u10","v10"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    lgt = None
    if args.lightning:
        lgt = pd.read_csv(args.lightning, parse_dates=["time"])
        if not {"time","lat","lon"} <= set(lgt.columns):
            raise ValueError("Lightning file must include: time, lat, lon [, flashes]")
        if "flashes" in lgt.columns:
            lgt = lgt[lgt["flashes"] > 0]
        lgt["time_hr"] = lgt["time"].dt.floor("h")

    # work hour-by-hour to avoid deprecations and memory spikes
    df["time_hr"] = df["time"].dt.floor("h")
    if args.neighbor_step <= 0:
        step_lat = infer_step(df["lat"].unique())
        step_lon = infer_step(df["lon"].unique())
        # assume square-ish
        step = float(np.median([step_lat, step_lon]))
    else:
        step = args.neighbor_step

    cs_list = []
    ra_s_list = []
    ra_a_list = []
    nd_list = []

    uniq_hours = pd.Index(df["time_hr"].unique())
    N = len(uniq_hours)
    for t_i, th in enumerate(uniq_hours):
        sub = df[df["time_hr"] == th].copy()
        cs, ra_s, ra_a, _ = per_time_neighbors_block(sub, step, args.radius_cells)
        cs_list.append(pd.Series(cs, index=sub.index))
        ra_s_list.append(pd.Series(ra_s, index=sub.index))
        ra_a_list.append(pd.Series(ra_a, index=sub.index))
        if lgt is not None:
            lsub = lgt[lgt["time_hr"] == th]
            dens = lightning_density_block(sub, lsub, args.lightning_radius)
            nd_list.append(pd.Series(dens, index=sub.index))
        if (t_i+1) % max(1, N//10) == 0:
            print(f"  … {t_i+1}/{N} hours", flush=True)

    df["sph_center"] = pd.concat(cs_list).sort_index()
    df["sph_radial_signed"] = pd.concat(ra_s_list).sort_index()
    df["sph_radial_abs"]    = pd.concat(ra_a_list).sort_index()
    if nd_list:
        dens_raw = pd.concat(nd_list).sort_index().to_numpy()
        df["sph_lightning_raw"] = dens_raw
        df["sph_lightning"]     = robust01(dens_raw)
    else:
        df["sph_lightning_raw"] = 0.0
        df["sph_lightning"]     = 0.0

    # Composite SFI (weights can be tuned)
    # inward or outward both indicate "shell" → use abs radial alignment
    sfi = 0.45*df["sph_center"].to_numpy() + 0.35*df["sph_radial_abs"].to_numpy() + 0.20*df["sph_lightning"].to_numpy()
    df["SFI"] = robust01(sfi)

    # quick correlation against coincident pregen, if available
    if "pregen" in df.columns:
        y = df["pregen"].astype(int).to_numpy()
        def rcol(c):
            x = df[c].to_numpy()
            x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
            y0 = (y - y.mean()) / (y.std() + 1e-12)
            return float(np.nanmean(x*y0))
        print("\nPearson r vs pregen (coincident):", flush=True)
        for c in ["sph_center","sph_radial_abs","sph_lightning","SFI"]:
            print(f"  {c:16s} r={rcol(c):+.3f}")

    # percentiles
    q = df[["sph_center","sph_radial_abs","sph_lightning","SFI"]].quantile([0.05,0.5,0.95])
    print("\nPercentiles (0.05/0.50/0.95):")
    print(q)

    # write
    keep = ["time","lat","lon","sph_center","sph_radial_signed","sph_radial_abs","sph_lightning_raw","sph_lightning","SFI"]
    df[keep].to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  | rows={len(df):,}", flush=True)

if __name__ == "__main__":
    main()