#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_symmetry_features.py

Agent: opt-in symmetry/OAM-style descriptors on weather grid patches without changing core physics.

Build small polar patches around each (time, ilat, ilon) cell, then compute
parity-odd contrast, spiral harmonic contrast, and a log-log knee metric.
Defaults are conservative and off the main pipeline; enable explicitly.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # Prefer package-style import when run via features_manager
    from features_subprocess.symmetry_filters import (
        knee_on_loglog,
        parity_odd_contrast_ring,
        spiral_harmonic_contrast,
    )
except ModuleNotFoundError:
    # Fallback for direct execution from this folder
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from symmetry_filters import knee_on_loglog, parity_odd_contrast_ring, spiral_harmonic_contrast  # type: ignore

pd.options.mode.copy_on_write = True


def _is_parquet(path: str) -> bool:
    return Path(path).suffix.lower() in {".parquet", ".parq", ".pq"}


def _iter_batches(path: str, columns: Sequence[str] | None, chunk_rows: int | None) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if chunk_rows and chunk_rows > 0:
            try:
                import pyarrow.parquet as pq  # type: ignore

                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=int(chunk_rows), columns=list(columns) if columns else None):
                    yield batch.to_pandas()
                return
            except Exception:
                # Fallback to pandas whole-file read if chunked parquet fails
                pass
        yield pd.read_parquet(path, columns=list(columns) if columns else None)
        return
    if chunk_rows and chunk_rows > 0:
        for ch in pd.read_csv(path, usecols=list(columns) if columns else None, chunksize=int(chunk_rows), low_memory=False):
            yield ch
        return
    yield pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)


def _pick_intensity(cols: Sequence[str]) -> str | None:
    candidates = ["gka_kappa", "S", "wspd", "zeta", "msl"]
    for c in candidates:
        if c in cols:
            return c
    return None


def _precompute_offsets(radius_cells: int, n_radii: int, theta_bins: int) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Precompute neighbor offsets -> (di, dj, r_bin, theta_bin).
    Angles are computed on the integer grid; r bins are nearest to linspace(1..radius).
    """
    r_vals = np.linspace(1.0, float(radius_cells), n_radii)
    offsets: List[Tuple[int, int, int, int]] = []
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di == 0 and dj == 0:
                continue
            dist = math.hypot(di, dj)
            if dist == 0 or dist > radius_cells + 1e-6:
                continue
            r_bin = int(np.abs(r_vals - dist).argmin())
            theta = math.atan2(di, dj)
            theta_bin = int(math.floor(((theta + math.pi) / (2 * math.pi)) * theta_bins)) % theta_bins
            offsets.append((di, dj, r_bin, theta_bin))
    return r_vals, offsets


def _theta_shuffle(I_r_theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle theta dimension independently per radius."""
    out = np.empty_like(I_r_theta)
    for i in range(I_r_theta.shape[0]):
        perm = rng.permutation(I_r_theta.shape[1])
        out[i] = I_r_theta[i, perm]
    return out


def _compute_block(
    block: pd.DataFrame,
    value_col: str,
    offsets: List[Tuple[int, int, int, int]],
    r_vals: np.ndarray,
    theta_bins: int,
    null_baseline: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Compute symmetry metrics for one time slice.
    Returns column_name -> np.ndarray aligned to block.index order.
    """
    n = len(block)
    results: Dict[str, np.ndarray] = {
        "sym_eta_odd_med": np.full(n, np.nan, dtype=np.float32),
        "sym_eta_spiral_med": np.full(n, np.nan, dtype=np.float32),
        "sym_eta_knee": np.full(n, np.nan, dtype=np.float32),
        "sym_theta_coverage": np.zeros(n, dtype=np.float32),
        "sym_neighbors": np.zeros(n, dtype=np.float32),
    }
    if null_baseline:
        results["sym_eta_odd_null_med"] = np.full(n, np.nan, dtype=np.float32)
        results["sym_eta_spiral_null_med"] = np.full(n, np.nan, dtype=np.float32)
        results["sym_eta_knee_null"] = np.full(n, np.nan, dtype=np.float32)

    ilat = pd.to_numeric(block["ilat"], errors="coerce").to_numpy(dtype=np.int64)
    ilon = pd.to_numeric(block["ilon"], errors="coerce").to_numpy(dtype=np.int64)
    vals = pd.to_numeric(block[value_col], errors="coerce").to_numpy(dtype=np.float32)

    # Map (ilat, ilon) -> row index; drop NaNs to avoid noisy lookups.
    lookup: Dict[Tuple[int, int], int] = {}
    for idx, (ia, jo) in enumerate(zip(ilat, ilon)):
        if np.isfinite(vals[idx]):
            lookup[(int(ia), int(jo))] = idx

    n_r = len(r_vals)
    for idx in range(n):
        center = (int(ilat[idx]), int(ilon[idx]))
        sum_arr = np.zeros((n_r, theta_bins), dtype=np.float32)
        count_arr = np.zeros((n_r, theta_bins), dtype=np.int32)
        for di, dj, r_bin, theta_bin in offsets:
            nbr_idx = lookup.get((center[0] + di, center[1] + dj))
            if nbr_idx is None:
                continue
            val = vals[nbr_idx]
            if not np.isfinite(val):
                continue
            sum_arr[r_bin, theta_bin] += val
            count_arr[r_bin, theta_bin] += 1

        n_neighbors = int(count_arr.sum())
        results["sym_neighbors"][idx] = float(n_neighbors)
        if n_neighbors == 0:
            continue

        I = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr > 0)
        coverage = (count_arr > 0).mean()
        results["sym_theta_coverage"][idx] = float(coverage)

        eta_odd, _ = parity_odd_contrast_ring(I)
        eta_spiral = spiral_harmonic_contrast(I)
        knee = knee_on_loglog(r_vals, eta_odd)

        results["sym_eta_odd_med"][idx] = float(np.nanmedian(eta_odd))
        results["sym_eta_spiral_med"][idx] = float(np.nanmedian(eta_spiral))
        results["sym_eta_knee"][idx] = float(knee.r_knee)

        if null_baseline:
            I_null = _theta_shuffle(I, rng)
            eta_odd_null, _ = parity_odd_contrast_ring(I_null)
            eta_spiral_null = spiral_harmonic_contrast(I_null)
            knee_null = knee_on_loglog(r_vals, eta_odd_null)
            results["sym_eta_odd_null_med"][idx] = float(np.nanmedian(eta_odd_null))
            results["sym_eta_spiral_null_med"][idx] = float(np.nanmedian(eta_spiral_null))
            results["sym_eta_knee_null"][idx] = float(knee_null.r_knee)

    return results


def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise SystemExit("symmetry features require a 'time' column.")
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute opt-in symmetry (parity/spiral) metrics on grid patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="Input CSV/Parquet with time, lat, lon, ilat, ilon, and intensity column.")
    ap.add_argument("--out", default="results/symmetry_features.csv", help="Output CSV path.")
    ap.add_argument("--intensity-col", default=None, help="Column to treat as intensity (default: pick gka_kappa>S>wspd>zeta>msl).")
    ap.add_argument("--radius-cells", type=int, default=3, help="Patch radius in grid cells for neighbor sampling.")
    ap.add_argument("--n-radii", type=int, default=3, help="Number of radial bins between 1 and radius-cells.")
    ap.add_argument("--theta-bins", type=int, default=16, help="Number of angular bins.")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Optional chunked CSV reading (0=full read).")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Accepted for compatibility; unused.")
    ap.add_argument("--parquet_rows", type=int, default=None, help="Accepted for compatibility; unused.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows for debugging.")
    ap.add_argument("--null-baseline", action="store_true", help="Also compute theta-shuffled baseline metrics.")
    ap.add_argument("--write-parquet", action="store_true", help="Also write Parquet copy alongside CSV.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    args = ap.parse_args()

    if args.radius_cells < 1 or args.n_radii < 1 or args.theta_bins < 4:
        raise SystemExit("radius-cells, n-radii, and theta-bins must be positive (theta-bins >=4).")

    # Alias handling for orchestrator-provided flags
    if (args.chunk_rows == 0 or args.chunk_rows is None) and args.chunksize:
        args.chunk_rows = args.chunksize
    if (args.chunk_rows == 0 or args.chunk_rows is None) and args.parquet_rows:
        args.chunk_rows = args.parquet_rows

    needed_cols = {"time", "lat", "lon", "ilat", "ilon"}
    chunk_rows = args.chunk_rows if args.chunk_rows and args.chunk_rows > 0 else None

    r_vals, offsets = _precompute_offsets(int(args.radius_cells), int(args.n_radii), int(args.theta_bins))
    rng = np.random.default_rng(42)

    # Prepare outputs (streaming to avoid memory blowup)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_parquet_main = out_path.suffix.lower() in {".parquet", ".parq", ".pq"}
    csv_path = None if is_parquet_main else out_path
    parquet_path = out_path if is_parquet_main else (Path(str(out_path) + ".parquet") if args.write_parquet else None)

    if is_parquet_main:
        # Force parquet writing when the main output ends with parquet-like suffix
        args.write_parquet = True

    # Overwrite guards
    paths_to_check = [p for p in (csv_path, parquet_path) if p is not None]
    for p in paths_to_check:
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (pass --overwrite to replace)")
        if args.overwrite:
            p.unlink(missing_ok=True)

    parquet_writer = None
    if args.write_parquet and parquet_path is not None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            if is_parquet_main:
                raise SystemExit(f"pyarrow required for parquet output: {exc}")
            print(f"[warn] pyarrow not available for parquet output ({exc}); skipping parquet write.")
            args.write_parquet = False
        else:
            parquet_writer = None

    total_rows = 0
    first_csv = True
    chunk_idx = 0
    t0 = time.time()
    for chunk in _iter_batches(args.panel, columns=None, chunk_rows=chunk_rows):
        if args.max_rows is not None and len(chunk) > args.max_rows:
            chunk = chunk.sample(n=int(args.max_rows), random_state=42).reset_index(drop=True)

        missing = needed_cols - set(chunk.columns)
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")

        chunk = _ensure_time(chunk)

        intensity_col = args.intensity_col or _pick_intensity(chunk.columns)
        if intensity_col is None:
            raise SystemExit("Could not infer intensity column; pass --intensity-col (e.g., gka_kappa or S).")
        if intensity_col not in chunk.columns:
            raise SystemExit(f"Intensity column '{intensity_col}' not in input.")

        # Initialize result columns with NaN/0 defaults
        for col in [
            "sym_eta_odd_med",
            "sym_eta_spiral_med",
            "sym_eta_knee",
            "sym_theta_coverage",
            "sym_neighbors",
        ]:
            chunk[col] = np.nan
        if args.null_baseline:
            for col in ["sym_eta_odd_null_med", "sym_eta_spiral_null_med", "sym_eta_knee_null"]:
                chunk[col] = np.nan

        # Per-time slices keep neighbor search bounded
        for _, idx in chunk.groupby("time", sort=False).indices.items():
            block = chunk.loc[idx]
            res = _compute_block(
                block,
                value_col=intensity_col,
                offsets=offsets,
                r_vals=r_vals,
                theta_bins=int(args.theta_bins),
                null_baseline=args.null_baseline,
                rng=rng,
            )
            for col, arr in res.items():
                chunk.loc[idx, col] = arr

        chunk_idx += 1

        if csv_path is not None:
            # Stream append to CSV to avoid holding all chunks
            chunk.to_csv(csv_path, index=False, mode="a", header=first_csv, date_format="%Y-%m-%d %H:%M:%S")
            first_csv = False

        if args.write_parquet and parquet_path is not None:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if parquet_writer is None:
                # Lazy-init writer on first chunk so parquet output works even when only hinted earlier.
                parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
            parquet_writer.write_table(table)

        total_rows += len(chunk)
        elapsed_min = (time.time() - t0) / 60.0
        print(
            f"[write] chunk {chunk_idx} rows={len(chunk):,} total={total_rows:,} elapsed={elapsed_min:0.2f} min",
            flush=True,
        )

    if parquet_writer is not None:
        parquet_writer.close()

    if total_rows == 0:
        print("[exit] no rows processed.")
        return

    target_msg = parquet_path if csv_path is None else csv_path
    extra_msg = f" + parquet copy {parquet_path}" if csv_path is not None and parquet_path is not None else ""
    print(f"[write] symmetry features -> {target_msg} rows={total_rows:,}{extra_msg}")


if __name__ == "__main__":
    main()
