#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute_spherical_feedback.py — robust neighbor stencil + enhanced SFI (memory safe; no lightning/rain)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

pd.options.mode.copy_on_write = True

# ----------------------- CLI -----------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Compute spherical-feedback features per hour: center-ness from MSL, "
            "radial wind alignment, local variance of vort/div tension, local T2m anomaly, "
            "a thermo×shear coupling, and composite SFI indices."
        )
    )
    # friendly aliases (for features_manager passthrough)
    ap.add_argument("--infile",    dest="labelled", required=False, help="grid_labelled_*.{csv,parquet}[.gz]")
    ap.add_argument("--labelled",  dest="labelled", required=False, help="Alias of --infile")
    ap.add_argument("--outfile",   dest="out",      required=False, help="Output path")
    ap.add_argument("--out",       dest="out",      required=False, help="Alias of --outfile")

    # neighborhood geometry
    ap.add_argument("--neighbor-step", type=float, default=0.0,
                    help="Grid step in degrees; 0=auto (median spacing)")
    ap.add_argument("--radius-cells", type=int, default=1,
                    help="Neighborhood radius in grid cells (1=8-neighbors)")

    # lead correlation (optional; still supported if pregen exists)
    ap.add_argument("--lead-hours", type=int, default=24,
                    help="Optional lead window for quick corr against 'pregen' if present")

    # harmonize with pipeline knobs
    ap.add_argument("--normalize-lon", default="-180..180",
                    help="Accepts ' -180..180', '-180..180', '0..360', 'none'")
    ap.add_argument("--area", default=None,
                    help='Optional AOI "latN,lonW,latS,lonE" applied before processing')

    # weights for SFI2 (you can tune in YAML)
    ap.add_argument("--w-center", type=float, default=0.40, help="Weight for sph_center")
    ap.add_argument("--w-radial", type=float, default=0.25, help="Weight for sph_radial_abs")
    ap.add_argument("--w-vdrstd", type=float, default=0.20, help="Weight for sph_vdr_std")
    ap.add_argument("--w-pdrop",  type=float, default=0.15, help="Weight for pressure-drop term (-msl_d1h)")
    ap.add_argument("--w-thermo", type=float, default=0.00, help="Optional extra weight for thermo_shear (default 0)")

    args = ap.parse_args()
    if not args.labelled:
        ap.error("the following arguments are required: --infile/--labelled")
    if not args.out:
        p = Path(args.labelled)
        args.out = str(p.with_name("spherical_feedback.csv.gz"))
    return args

# ----------------------- I/O helpers -----------------------
def load_any(path: str, parse_time=True) -> pd.DataFrame:
    lower = str(path).lower()
    if lower.endswith((".parquet", ".pq", ".pqt")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            encoding_errors="replace",
            on_bad_lines="skip",
            parse_dates=["time"] if parse_time else None,
        )
    if parse_time and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet", ".pq", ".pqt")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if (low.endswith(".csv.gz") or p.suffix.lower()==".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ----------------------- small utils -----------------------
ALIASES = {
    "u":   ["u", "u10", "U10M"],
    "v":   ["v", "v10", "V10M"],
    "msl": ["msl", "sp", "mslp", "mean_sea_level_pressure", "MSL"],
    # optionals that we use if present
    "vdr": ["gka_vortdiv_ratio"],                      # local vort/div ratio (preferred)
    "t2m": ["t2m", "2m_temperature", "T2M", "t_2m"],
    "shear": ["shear_06km","bulk_shear_0_6km","shear06","shear_06",
              "shear_01km","bulk_shear_0_1km","shear01","shear_01",
              "shear10_def","shear_2d10","shear10","S3"],  # permissive
    "msl_d1h": ["msl_d1h"],
}

def bind_col(cols, choices):
    for c in choices:
        if c in cols:
            return c
    return None

def robust01(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    q1, q99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    den = (q99 - q1) if (q99 > q1) else 1.0
    return np.clip((x - q1) / (den + 1e-12), 0.0, 1.0)

def infer_step(vals):
    v = np.sort(np.unique(np.asarray(vals, float)))
    if len(v) < 3:
        return 0.25
    d = np.diff(v)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.25

def quantize(coord, step):
    return np.round(coord / max(step, 1e-9)).astype(np.int32)

def norm_mode(s: str | None) -> str:
    if s is None:
        return "-180..180"
    t = str(s).strip()
    if t in ("-180..180", "0..360", "none"):
        return t
    if t.replace(" ", "") == "-180..180":
        return "-180..180"
    if t.replace(" ", "") == "0..360":
        return "0..360"
    return "-180..180"

def wrap_lon_vec(lon, mode: str):
    x = np.asarray(lon, float)
    if mode == "none":
        return x
    if mode == "0..360":
        y = np.mod(x, 360.0)
        y[y >= 359.9995] = 0.0
        return y
    # default: -180..180
    return ((x + 180.0) % 360.0) - 180.0

def parse_area(aoi: str | None):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(z.strip()) for z in aoi.split(",")]
    return latN, lonW, latS, lonE

def crop_aoi(df, aoi):
    if not aoi:
        return df
    latN, lonW, latS, lonE = aoi
    return df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                  (df["lon"] >= lonW) & (df["lon"] <= lonE)].copy()

# ----------------------- core blocks -----------------------
def per_time_neighbors_block(
    sub: pd.DataFrame,
    step_lat: float,
    step_lon: float,
    radius_cells: int,
    ucol: str, vcol: str, mcol: str,
    vdr_col: str | None,
    t2m_col: str | None,
    shear_col: str | None,
    pdrop_col: str | None,
):
    """
    Memory-aware neighbor features for one hour's slice (sub).
    Returns dict of np.float32 arrays aligned to sub.index (same length as sub):
      - center_norm, radial_signed, radial_abs
      - vdr_std (neighborhood std of vort/div ratio)  [0 if vdr_col missing]
      - t2m_anom_local (cell temp minus neighborhood mean; robust01) [0 if t2m missing]
      - thermo_shear = robust01(shear) * robust01(max(t2m_anom_local, 0)) [0 if missing]
      - pdrop = robust01(-msl_d1h) [0 if missing]
    """
    lat = sub["lat"].to_numpy(dtype=np.float32, copy=False)
    lon = sub["lon"].to_numpy(dtype=np.float32, copy=False)
    msl = pd.to_numeric(sub[mcol], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    u10 = pd.to_numeric(sub[ucol], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    v10 = pd.to_numeric(sub[vcol], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    vdr = None
    if vdr_col and vdr_col in sub.columns:
        vdr = pd.to_numeric(sub[vdr_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    t2m = None
    if t2m_col and t2m_col in sub.columns:
        t2m = pd.to_numeric(sub[t2m_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    shear = None
    if shear_col and shear_col in sub.columns:
        shear = pd.to_numeric(sub[shear_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    pdrop_src = None
    if pdrop_col and pdrop_col in sub.columns:
        pdrop_src = pd.to_numeric(sub[pdrop_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(msl) & np.isfinite(u10) & np.isfinite(v10)
    pos_idx_ok = np.nonzero(ok)[0]
    nsub = len(sub)

    # preallocate outputs
    center_raw = np.zeros(nsub, dtype=np.float32)
    radial_raw = np.zeros(nsub, dtype=np.float32)
    vdr_std    = np.zeros(nsub, dtype=np.float32)  # neighborhood std
    t2m_anom   = np.zeros(nsub, dtype=np.float32)  # local anomaly vs neighbor mean

    # quantized grid map (only valid cells)
    if pos_idx_ok.size == 0:
        # still populate optional quick terms from scalars
        out = {
            "center_norm": robust01(-center_raw),
            "radial_signed": np.tanh(radial_raw).astype(np.float32),
            "radial_abs": np.abs(np.tanh(radial_raw)).astype(np.float32),
            "vdr_std": vdr_std,
            "t2m_anom_local": t2m_anom,
            "thermo_shear": np.zeros(nsub, dtype=np.float32),
            "pdrop": robust01(-pdrop_src) if pdrop_src is not None else np.zeros(nsub, dtype=np.float32),
        }
        return out

    qlat = quantize(lat[pos_idx_ok], step_lat)
    qlon = quantize(lon[pos_idx_ok], step_lon)
    where = {(int(qlat[i]), int(qlon[i])): i for i in range(len(pos_idx_ok))}

    r = int(radius_cells)
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if not (dy == 0 and dx == 0)]

    cosphi = np.clip(np.cos(np.deg2rad(lat[pos_idx_ok])), 1e-6, None).astype(np.float32, copy=False)

    for k, pos in enumerate(pos_idx_ok):
        qi, qj = int(qlat[k]), int(qlon[k])

        # center-ness from MSL (deeper than neighbors → positive after robust01(-·))
        m0 = msl[pos]

        # wind radial alignment
        u = u10[pos]; v = v10[pos]
        spd = np.hypot(u, v).astype(np.float32)
        if not np.isfinite(spd) or spd == 0.0:
            spd = np.float32(1e-12)

        # neighbor accumulators
        n_m = n_a = 0
        m_sum = 0.0
        align_sum = 0.0

        # extra accumulators for optional stats
        vdr_vals = [] if vdr is not None else None
        t2m_vals = [] if t2m is not None else None

        for (dy, dx) in offsets:
            j_local = where.get((qi + dy, qj + dx))
            if j_local is None:
                continue
            pos_n = pos_idx_ok[j_local]

            # pressure
            m_sum += msl[pos_n]; n_m += 1

            # radial alignment i -> j (dlon scaled by cosφ at i)
            dlat = (lat[pos_n] - lat[pos])
            dlon = (lon[pos_n] - lon[pos]) * cosphi[k]
            rn = np.hypot(dlat, dlon).astype(np.float32)
            if rn > 0.0:
                rx = dlon / rn
                ry = dlat / rn
                cos_th = (u * rx + v * ry) / spd
                align_sum += float(cos_th)
                n_a += 1

            if vdr_vals is not None:
                vv = vdr[pos_n]
                if np.isfinite(vv):
                    vdr_vals.append(vv)
            if t2m_vals is not None:
                tt = t2m[pos_n]
                if np.isfinite(tt):
                    t2m_vals.append(tt)

        if n_m > 0:
            center_raw[pos] = (m0 - (m_sum / n_m))
        if n_a > 0:
            radial_raw[pos] = (align_sum / n_a)

        if vdr_vals is not None and len(vdr_vals) >= 2:
            vdr_std[pos] = np.nanstd(np.asarray(vdr_vals, dtype=np.float32))
        if t2m_vals is not None and len(t2m_vals) > 0:
            t2m_anom[pos] = (t2m[pos] - (np.nanmean(np.asarray(t2m_vals, dtype=np.float32))))

    # normalize/squash primaries
    center_norm = robust01(-center_raw)
    radial_signed = np.tanh(radial_raw).astype(np.float32)
    radial_abs = np.abs(radial_signed).astype(np.float32)

    # robust scales for secondaries
    vdr_std_n = robust01(vdr_std) if vdr is not None else np.zeros(nsub, dtype=np.float32)
    t2m_anom_pos = np.maximum(t2m_anom, 0.0)
    t2m_anom_n = robust01(t2m_anom_pos) if t2m is not None else np.zeros(nsub, dtype=np.float32)
    shear_n = robust01(shear) if shear is not None else np.zeros(nsub, dtype=np.float32)
    thermo_shear = (t2m_anom_n * shear_n).astype(np.float32)

    pdrop = robust01(-pdrop_src) if pdrop_src is not None else np.zeros(nsub, dtype=np.float32)

    return {
        "center_norm": center_norm.astype(np.float32),
        "radial_signed": radial_signed,
        "radial_abs": radial_abs,
        "vdr_std": vdr_std_n.astype(np.float32),
        "t2m_anom_local": t2m_anom_n.astype(np.float32),
        "thermo_shear": thermo_shear,
        "pdrop": pdrop.astype(np.float32),
    }

def future_any_by_point(df, label_col, hours):
    out = np.zeros(len(df), dtype=np.int8)
    for (_, _), g in df.groupby(["lat", "lon"], sort=False):
        y = pd.to_numeric(g[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        rev = y[::-1]
        s = pd.Series(rev)
        fut = s.shift(1).rolling(window=int(hours), min_periods=1).max().fillna(0).astype(int).to_numpy()[::-1]
        out[g.index] = fut
    return out

# ----------------------- main -----------------------
def main():
    args = parse_args()
    lon_mode = norm_mode(args.normalize_lon)
    aoi = parse_area(args.area)

    print("== Spherical Feedback Features ==", flush=True)
    print(f"In       : {args.labelled}", flush=True)

    # Load once; then slim early
    df0 = load_any(args.labelled)
    cols = set(df0.columns)

    ucol = bind_col(cols, ALIASES["u"])
    vcol = bind_col(cols, ALIASES["v"])
    mcol = bind_col(cols, ALIASES["msl"])
    if None in (ucol, vcol, mcol) or not {"time","lat","lon"}.issubset(cols):
        found = sorted(list(cols))[:24]
        raise ValueError(f"Missing required wind/pressure columns (need u,v,msl aliases). Found head: {found}")

    # optional columns
    vdr_col   = bind_col(cols, ALIASES["vdr"])
    t2m_col   = bind_col(cols, ALIASES["t2m"])
    shear_col = bind_col(cols, ALIASES["shear"])
    pdrop_col = bind_col(cols, ALIASES["msl_d1h"])

    # normalize coordinates; slim projection
    keep_cols = ["time","lat","lon", ucol, vcol, mcol]
    for opt in (vdr_col, t2m_col, shear_col, pdrop_col, "pregen"):
        if isinstance(opt, str) and opt in df0.columns and opt not in keep_cols:
            keep_cols.append(opt)

    df = df0[keep_cols].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["lat"]  = pd.to_numeric(df["lat"], errors="coerce").astype(np.float32)
    df["lon"]  = wrap_lon_vec(pd.to_numeric(df["lon"], errors="coerce"), lon_mode).astype(np.float32)
    df[ucol]   = pd.to_numeric(df[ucol], errors="coerce").astype(np.float32)
    df[vcol]   = pd.to_numeric(df[vcol], errors="coerce").astype(np.float32)
    df[mcol]   = pd.to_numeric(df[mcol], errors="coerce").astype(np.float32)
    if vdr_col:   df[vdr_col]   = pd.to_numeric(df[vdr_col], errors="coerce").astype(np.float32)
    if t2m_col:   df[t2m_col]   = pd.to_numeric(df[t2m_col], errors="coerce").astype(np.float32)
    if shear_col: df[shear_col] = pd.to_numeric(df[shear_col], errors="coerce").astype(np.float32)
    if pdrop_col: df[pdrop_col] = pd.to_numeric(df[pdrop_col], errors="coerce").astype(np.float32)

    df = df.dropna(subset=["time","lat","lon"]).reset_index(drop=True)
    if aoi:
        df = crop_aoi(df, aoi)
    if df.empty:
        raise ValueError("No rows after normalization/AOI filter.")

    # hour bucket & sort
    df.sort_values(["time", "lat", "lon"], kind="mergesort", inplace=True, ignore_index=True)
    df["time_hr"] = pd.to_datetime(df["time"]).dt.floor("h")

    # step inference
    step_lat = args.neighbor_step if args.neighbor_step > 0 else infer_step(df["lat"].unique())
    step_lon = args.neighbor_step if args.neighbor_step > 0 else infer_step(df["lon"].unique())
    print(f"Radius(cells)={args.radius_cells}  Neighbor step={'auto' if args.neighbor_step<=0 else args.neighbor_step}°", flush=True)
    print(f"LonMode={lon_mode}  step_lat≈{step_lat:.4f}°  step_lon≈{step_lon:.4f}°", flush=True)

    # per-hour processing (memory-safe)
    parts = {
        "sph_center": [],
        "sph_radial_signed": [],
        "sph_radial_abs": [],
        "sph_vdr_std": [],
        "t2m_anom_local": [],
        "thermo_shear": [],
        "pdrop_nd": [],
    }

    hours = pd.Index(df["time_hr"].unique())
    N = len(hours)
    for k, th in enumerate(hours, start=1):
        mask = (df["time_hr"] == th)
        idx = df.index[mask]
        sub = df.loc[mask, ["lat","lon", ucol, vcol, mcol] +
                           ([vdr_col] if vdr_col else []) +
                           ([t2m_col] if t2m_col else []) +
                           ([shear_col] if shear_col else []) +
                           ([pdrop_col] if pdrop_col else [])]

        out = per_time_neighbors_block(
            sub, step_lat, step_lon, args.radius_cells,
            ucol, vcol, mcol, vdr_col, t2m_col, shear_col, pdrop_col
        )

        parts["sph_center"].append(pd.Series(out["center_norm"], index=idx))
        parts["sph_radial_signed"].append(pd.Series(out["radial_signed"], index=idx))
        parts["sph_radial_abs"].append(pd.Series(out["radial_abs"], index=idx))
        parts["sph_vdr_std"].append(pd.Series(out["vdr_std"], index=idx))
        parts["t2m_anom_local"].append(pd.Series(out["t2m_anom_local"], index=idx))
        parts["thermo_shear"].append(pd.Series(out["thermo_shear"], index=idx))
        parts["pdrop_nd"].append(pd.Series(out["pdrop"], index=idx))

        if (k % max(1, N // 10)) == 0 or k == N:
            print(f"  … {k}/{N} hours", flush=True)

    # stitch
    for key, lst in parts.items():
        df[key] = pd.concat(lst).sort_index().astype(np.float32)

    # SFI (original) — kept for continuity
    sfi = 0.45 * df["sph_center"].to_numpy(dtype=np.float32) \
        + 0.35 * df["sph_radial_abs"].to_numpy(dtype=np.float32) \
        + 0.20 * np.zeros(len(df), dtype=np.float32)  # no lightning
    df["SFI"] = robust01(sfi).astype(np.float32)

    # SFI2 (enhanced, tunable)
    wC, wR, wV, wP, wT = args.w_center, args.w_radial, args.w_vdrstd, args.w_pdrop, args.w_thermo
    mix = (wC * df["sph_center"].to_numpy(dtype=np.float32) +
           wR * df["sph_radial_abs"].to_numpy(dtype=np.float32) +
           wV * df["sph_vdr_std"].to_numpy(dtype=np.float32) +
           wP * df["pdrop_nd"].to_numpy(dtype=np.float32) +
           wT * df["thermo_shear"].to_numpy(dtype=np.float32))
    df["SFI2"] = robust01(mix).astype(np.float32)

    # quick correlations vs pregen (if present)
    if "pregen" in df.columns:
        y0 = pd.to_numeric(df["pregen"], errors="coerce").fillna(0).astype(int).to_numpy()
        def pearson(x):
            x = np.asarray(x, float)
            xm = np.nanmean(x); xs = np.nanstd(x) + 1e-12
            ym = y0.mean(); ys = y0.std() + 1e-12
            return float(np.nanmean(((x - xm)/xs) * ((y0 - ym)/ys)))
        print("\nPearson r vs pregen (coincident):", flush=True)
        for c in ("sph_center","sph_radial_abs","sph_vdr_std","pdrop_nd","thermo_shear","SFI","SFI2"):
            print(f"  {c:14s} r={pearson(df[c]):+.3f}")

        if args.lead_hours and args.lead_hours > 0:
            yL = future_any_by_point(df[["time","lat","lon","pregen"]], "pregen", int(args.lead_hours))
            ym = yL.mean(); ys = yL.std() + 1e-12
            print(f"\nPearson r vs pregen_future(+{args.lead_hours}h):", flush=True)
            for c in ("sph_center","sph_radial_abs","sph_vdr_std","pdrop_nd","thermo_shear","SFI","SFI2"):
                x = df[c].to_numpy()
                xm = np.nanmean(x); xs = np.nanstd(x) + 1e-12
                r = float(np.nanmean(((x - xm)/xs) * ((yL - ym)/ys)))
                print(f"  {c:14s} r={r:+.3f}")

    # small snapshot
    try:
        q = df[["sph_center","sph_radial_abs","sph_vdr_std","t2m_anom_local","pdrop_nd","thermo_shear","SFI","SFI2"]].quantile([0.05, 0.50, 0.95])
        print("\nPercentiles (0.05 / 0.50 / 0.95):")
        print(q)
    except Exception:
        pass

    # write slim artifact
    keep = [
        "time","lat","lon",
        "sph_center","sph_radial_signed","sph_radial_abs",
        "sph_vdr_std","t2m_anom_local","pdrop_nd","thermo_shear",
        "SFI","SFI2"
    ]
    write_any(args.out, df[keep])
    print(f"\nWrote {args.out}  | rows={len(df):,}", flush=True)

if __name__ == "__main__":
    main()