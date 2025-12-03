#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute_spherical_feedback.py - robust neighbor stencil + enhanced SFI (streaming/memory-safe; no lightning/rain)

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure repository root (containing utils/) is importable when run as a script.
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import io_common

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
    ap.add_argument("--infile", dest="labelled", required=False, help="grid_labelled_*.{csv,parquet}[.gz]")
    ap.add_argument("--labelled", dest="labelled", required=False, help="Alias of --infile")
    ap.add_argument("--outfile", dest="out", required=False, help="Output path")
    ap.add_argument("--out", dest="out", required=False, help="Alias of --outfile")

    # neighborhood geometry
    ap.add_argument("--neighbor-step", type=float, default=0.0,
                    help="Grid step in degrees; 0=auto (median spacing)")
    ap.add_argument("--radius-cells", type=int, default=1,
                    help="Neighborhood radius in grid cells (1=8-neighbors)")

    # harmonize with pipeline knobs
    ap.add_argument("--normalize-lon", default="-180..180",
                    help="Accepts ' -180..180', '-180..180', '0..360', 'none'")
    ap.add_argument("--area", default=None,
                    help='Optional AOI "latN,lonW,latS,lonE" applied before processing')
    ap.add_argument("--chunk-rows", type=int, default=0,
                    help="Rows per chunk when streaming input (auto if 0).")
    ap.add_argument("--chunksize", type=int, default=0,
                    help="Alias for --chunk-rows (pipeline compatibility).")
    ap.add_argument("--parquet-rows", type=int, default=0,
                    help="Preferred batch size when streaming parquet input/output.")
    # Kept for compatibility; lead correlations are not computed in this streamer.
    ap.add_argument("--lead-hours", type=int, default=24,
                    help="Accepted for backward compatibility; currently unused in streaming mode.")

    # weights for SFI2 (tunable)
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
def _detect_table_format(path: Path) -> str:
    suf = "".join(path.suffixes[-2:]).lower()
    if suf in {".parquet", ".pq", ".pqt"} or path.suffix.lower() in {".parquet", ".pq", ".pqt"}:
        return "parquet"
    if suf == ".csv.gz":
        return "csv.gz"
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".gz":
        return "csv.gz"
    raise SystemExit(f"[spherical] Unsupported table format for {path}")


def _peek_columns(path: Path) -> List[str]:
    fmt = _detect_table_format(path)
    if fmt == "parquet":
        pf = pq.ParquetFile(path)
        return pf.schema.names
    head = pd.read_csv(path, nrows=5, compression="infer", low_memory=False)
    return list(head.columns)


def _iter_input_chunks(path: Path, usecols: Sequence[str], chunk_rows: int, parquet_rows: int | None = None) -> Iterable[pd.DataFrame]:
    fmt = _detect_table_format(path)
    if fmt == "parquet":
        pf = pq.ParquetFile(path)
        batch_size = parquet_rows if parquet_rows and parquet_rows > 0 else chunk_rows
        for batch in pf.iter_batches(batch_size=batch_size, columns=list(usecols)):
            yield batch.to_pandas()
    else:
        kw = {
            "compression": "infer",
            "low_memory": False,
            "encoding_errors": "replace",
            "on_bad_lines": "skip",
            "usecols": list(usecols),
            "chunksize": chunk_rows,
            "parse_dates": ["time"],
        }
        kw = io_common._csv_kwargs(path, kw)  # type: ignore[attr-defined]
        kw.pop("engine", None)  # chunked reads need pandas engine
        kw.pop("dtype_backend", None)
        reader = io_common._read_csv_with_missing_date_guard(path, kw)  # type: ignore[attr-defined]
        chunks = reader if not isinstance(reader, pd.DataFrame) else [reader]
        for chunk in chunks:
            yield chunk


class _ChunkedWriter:
    """
    Stream-friendly writer to keep memory bounded.
    """
    def __init__(self, path: Path, overwrite: bool):
        self.path = path
        self.format = _detect_table_format(path)
        self._writer = None
        self._wrote_header = False
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise SystemExit(f"[spherical] exists and overwrite disabled: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        if self.format == "parquet":
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
            self._writer.write_table(table)
        else:
            comp = "gzip" if self.format == "csv.gz" else "infer"
            mode = "w" if not self._wrote_header else "a"
            df.to_csv(
                self.path,
                index=False,
                compression=comp,
                date_format="%Y-%m-%d %H:%M:%S",
                mode=mode,
                header=not self._wrote_header,
            )
            self._wrote_header = True

    def close(self):
        if self._writer is not None:
            self._writer.close()


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


def robust01_from_quantiles(x: np.ndarray, q1: float, q99: float) -> np.ndarray:
    den = (q99 - q1) if (q99 > q1) else 1.0
    return np.clip((np.asarray(x, float) - q1) / (den + 1e-12), 0.0, 1.0)


def _compute_quantiles_from_temp(path: Path, cols: Sequence[str], batch_size: int) -> dict[str, Tuple[float, float]]:
    """Exact 1/99 quantiles by fully scanning temp file columns (slower, no sampling)."""
    fmt = _detect_table_format(path)
    q = {}
    for col in cols:
        vals = []
        if fmt == "parquet":
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=[col], batch_size=batch_size):
                arr = batch.column(0).to_numpy()
                if arr.size:
                    vals.append(arr[np.isfinite(arr)])
        else:
            for chunk in pd.read_csv(
                path,
                usecols=[col],
                compression="infer",
                low_memory=False,
                encoding_errors="replace",
                on_bad_lines="skip",
                chunksize=batch_size,
            ):
                arr = pd.to_numeric(chunk[col], errors="coerce").to_numpy()
                if arr.size:
                    vals.append(arr[np.isfinite(arr)])

        if not vals:
            q[col] = (0.0, 1.0)
            continue
        allv = np.concatenate(vals)
        q[col] = tuple(np.nanpercentile(allv, [1, 99]).tolist())  # type: ignore[assignment]
        del allv, vals
    return q


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

        # center-ness from MSL (deeper than neighbors -> positive after robust01(-·))
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

        vdr_vals = [] if vdr is not None else None
        t2m_vals = [] if t2m is not None else None

        for (dy, dx) in offsets:
            j_local = where.get((qi + dy, qj + dx))
            if j_local is None:
                continue
            pos_n = pos_idx_ok[j_local]

            m_sum += msl[pos_n]; n_m += 1

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

    center_norm = robust01(-center_raw)
    radial_signed = np.tanh(radial_raw).astype(np.float32)
    radial_abs = np.abs(radial_signed).astype(np.float32)

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


# ----------------------- streaming helpers -----------------------
def _process_ready_hours(
    buf: pd.DataFrame,
    hours: Sequence[pd.Timestamp],
    writer: _ChunkedWriter,
    radius_cells: int,
    neighbor_step: float,
    ucol: str, vcol: str, mcol: str,
    vdr_col: str | None,
    t2m_col: str | None,
    shear_col: str | None,
    pdrop_col: str | None,
    wC: float, wR: float, wV: float, wP: float, wT: float,
    rows_written: int,
    hours_done: int,
) -> tuple[pd.DataFrame, int, int]:
    if len(hours) == 0:
        return buf, rows_written, hours_done

    step_lat = neighbor_step if neighbor_step > 0 else infer_step(buf["lat"].unique())
    step_lon = neighbor_step if neighbor_step > 0 else infer_step(buf["lon"].unique())

    for th in sorted(hours):
        mask = buf["time_hr"] == th
        if not mask.any():
            continue
        sub = buf.loc[mask, ["time","lat","lon", ucol, vcol, mcol] +
                           ([vdr_col] if vdr_col else []) +
                           ([t2m_col] if t2m_col else []) +
                           ([shear_col] if shear_col else []) +
                           ([pdrop_col] if pdrop_col else [])]

        out = per_time_neighbors_block(
            sub, step_lat, step_lon, radius_cells,
            ucol, vcol, mcol, vdr_col, t2m_col, shear_col, pdrop_col
        )

        res = pd.DataFrame({
            "time": sub["time"].to_numpy(),
            "lat": sub["lat"].to_numpy(),
            "lon": sub["lon"].to_numpy(),
            "sph_center": out["center_norm"],
            "sph_radial_signed": out["radial_signed"],
            "sph_radial_abs": out["radial_abs"],
            "sph_vdr_std": out["vdr_std"],
            "t2m_anom_local": out["t2m_anom_local"],
            "pdrop_nd": out["pdrop"],
            "thermo_shear": out["thermo_shear"],
        }, index=sub.index).sort_index()

        sfi_raw = (0.45 * res["sph_center"].to_numpy(dtype=np.float32) +
                   0.35 * res["sph_radial_abs"].to_numpy(dtype=np.float32))
        mix_raw = (wC * res["sph_center"].to_numpy(dtype=np.float32) +
                   wR * res["sph_radial_abs"].to_numpy(dtype=np.float32) +
                   wV * res["sph_vdr_std"].to_numpy(dtype=np.float32) +
                   wP * res["pdrop_nd"].to_numpy(dtype=np.float32) +
                   wT * res["thermo_shear"].to_numpy(dtype=np.float32))
        res["SFI_raw"] = sfi_raw.astype(np.float32)
        res["SFI2_raw"] = mix_raw.astype(np.float32)

        writer.write(res)
        rows_written += len(res)
        hours_done += 1
        if (hours_done % 10) == 0:
            print(f"  . hours processed={hours_done} rows_written={rows_written:,}", flush=True)

    buf = buf.loc[~buf["time_hr"].isin(hours)].reset_index(drop=True)
    return buf, rows_written, hours_done


# ----------------------- main -----------------------
def main():
    args = parse_args()
    lon_mode = norm_mode(args.normalize_lon)
    aoi = parse_area(args.area)
    in_path = Path(args.labelled)
    out_path = Path(args.out)

    fmt_out = _detect_table_format(out_path)
    if fmt_out == "parquet":
        base = out_path.name
        if base.endswith(".parquet"):
            base = base[:-len(".parquet")]
        tmp_path = out_path.with_name(f"{base}.tmp.parquet")
    elif fmt_out == "csv.gz":
        base = out_path.name
        if base.endswith(".csv.gz"):
            base = base[:-len(".csv.gz")]
        tmp_path = out_path.with_name(f"{base}.tmp.csv.gz")
    else:  # csv
        base = out_path.stem
        tmp_path = out_path.with_name(f"{base}.tmp.csv")

    in_fmt = _detect_table_format(in_path)
    csv_rows, parq_rows = io_common.recommend_chunk_rows()  # type: ignore[attr-defined]
    chunk_pref = args.chunk_rows or args.chunksize
    parquet_pref = args.parquet_rows
    chunk_rows = chunk_pref or (parq_rows if in_fmt == "parquet" else csv_rows)
    if chunk_rows <= 0:
        chunk_rows = parq_rows if in_fmt == "parquet" else csv_rows
    parquet_rows = parquet_pref or chunk_rows

    print("== Spherical Feedback Features ==", flush=True)
    print(f"In       : {in_path}", flush=True)
    print(f"Out (tmp): {tmp_path}", flush=True)
    print(f"Chunk rows: {chunk_rows:,}  LonMode: {lon_mode}", flush=True)

    cols_all = set(_peek_columns(in_path))
    ucol = bind_col(cols_all, ALIASES["u"])
    vcol = bind_col(cols_all, ALIASES["v"])
    mcol = bind_col(cols_all, ALIASES["msl"])
    if None in (ucol, vcol, mcol) or not {"time","lat","lon"}.issubset(cols_all):
        found = sorted(list(cols_all))[:24]
        raise ValueError(f"Missing required wind/pressure columns (need u,v,msl aliases). Found head: {found}")

    vdr_col   = bind_col(cols_all, ALIASES["vdr"])
    t2m_col   = bind_col(cols_all, ALIASES["t2m"])
    shear_col = bind_col(cols_all, ALIASES["shear"])
    pdrop_col = bind_col(cols_all, ALIASES["msl_d1h"])

    keep_cols = ["time","lat","lon", ucol, vcol, mcol]
    for opt in (vdr_col, t2m_col, shear_col, pdrop_col, "pregen"):
        if isinstance(opt, str) and opt in cols_all and opt not in keep_cols:
            keep_cols.append(opt)

    writer_tmp = _ChunkedWriter(tmp_path, overwrite=True)
    rows_written = 0
    hours_done = 0
    buf = pd.DataFrame()

    for chunk in _iter_input_chunks(in_path, keep_cols, chunk_rows, parquet_rows=parquet_rows):
        if chunk is None or len(chunk) == 0:
            continue
        chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce").dt.tz_localize(None)
        chunk["lat"]  = pd.to_numeric(chunk["lat"], errors="coerce").astype(np.float32)
        chunk["lon"]  = wrap_lon_vec(pd.to_numeric(chunk["lon"], errors="coerce"), lon_mode).astype(np.float32)
        chunk[ucol]   = pd.to_numeric(chunk[ucol], errors="coerce").astype(np.float32)
        chunk[vcol]   = pd.to_numeric(chunk[vcol], errors="coerce").astype(np.float32)
        chunk[mcol]   = pd.to_numeric(chunk[mcol], errors="coerce").astype(np.float32)
        if vdr_col:   chunk[vdr_col]   = pd.to_numeric(chunk[vdr_col], errors="coerce").astype(np.float32)
        if t2m_col:   chunk[t2m_col]   = pd.to_numeric(chunk[t2m_col], errors="coerce").astype(np.float32)
        if shear_col: chunk[shear_col] = pd.to_numeric(chunk[shear_col], errors="coerce").astype(np.float32)
        if pdrop_col: chunk[pdrop_col] = pd.to_numeric(chunk[pdrop_col], errors="coerce").astype(np.float32)

        chunk = chunk.dropna(subset=["time","lat","lon"])
        if aoi:
            chunk = crop_aoi(chunk, aoi)
        if chunk.empty:
            continue

        chunk["time_hr"] = pd.to_datetime(chunk["time"]).dt.floor("h")
        buf = pd.concat([buf, chunk], ignore_index=True)
        buf.sort_values(["time","lat","lon"], kind="mergesort", inplace=True, ignore_index=True)

        hours = buf["time_hr"].unique()
        if len(hours) > 1:
            ready_hours = hours[:-1]  # leave last hour in buffer in case spillover appears next chunk
            buf, rows_written, hours_done = _process_ready_hours(
                buf, ready_hours, writer_tmp,
                args.radius_cells, args.neighbor_step,
                ucol, vcol, mcol, vdr_col, t2m_col, shear_col, pdrop_col,
                args.w_center, args.w_radial, args.w_vdrstd, args.w_pdrop, args.w_thermo,
                rows_written, hours_done
            )

    if not buf.empty:
        remaining_hours = buf["time_hr"].unique()
        buf, rows_written, hours_done = _process_ready_hours(
            buf, remaining_hours, writer_tmp,
            args.radius_cells, args.neighbor_step,
            ucol, vcol, mcol, vdr_col, t2m_col, shear_col, pdrop_col,
            args.w_center, args.w_radial, args.w_vdrstd, args.w_pdrop, args.w_thermo,
            rows_written, hours_done
        )

    writer_tmp.close()
    if rows_written == 0:
        raise SystemExit("[spherical] No rows processed; aborting.")

    q = _compute_quantiles_from_temp(tmp_path, ["SFI_raw","SFI2_raw"], max(chunk_rows, 100_000))
    sfi_q1, sfi_q99 = q["SFI_raw"]
    mix_q1, mix_q99 = q["SFI2_raw"]
    print(f"Quantiles (exact scan): SFI q1={sfi_q1:.4f} q99={sfi_q99:.4f} | SFI2 q1={mix_q1:.4f} q99={mix_q99:.4f}", flush=True)

    writer_out = _ChunkedWriter(out_path, overwrite=True)
    keep_final = [
        "time","lat","lon",
        "sph_center","sph_radial_signed","sph_radial_abs",
        "sph_vdr_std","t2m_anom_local","pdrop_nd","thermo_shear",
        "SFI","SFI2"
    ]

    for chunk in _iter_input_chunks(tmp_path, keep_final + ["SFI_raw","SFI2_raw"], chunk_rows, parquet_rows=parquet_rows):
        if chunk is None or len(chunk) == 0:
            continue
        chunk["SFI"] = robust01_from_quantiles(chunk["SFI_raw"], sfi_q1, sfi_q99).astype(np.float32)
        chunk["SFI2"] = robust01_from_quantiles(chunk["SFI2_raw"], mix_q1, mix_q99).astype(np.float32)
        writer_out.write(chunk[keep_final])

    writer_out.close()
    tmp_path.unlink(missing_ok=True)
    print(f"\nWrote {out_path}  | rows={rows_written:,}", flush=True)


if __name__ == "__main__":
    main()
