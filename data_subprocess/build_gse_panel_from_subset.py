#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_gse_panel_from_subset.py

Build a Geometry/Shear/Energy (GSE) panel from the ID-filtered subset.

Intent: light-touch scalar summaries of spiral structure (G), shear hostility (S),
and energy/fuel (E) without changing underlying maths. Uses existing columns when
present and falls back to NaN-safe zeros otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


# Agent: assemble compact G/S/E scalars from existing features (no new physics).

def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _iter_file(path: str, chunksize: Optional[int]) -> Iterable[pd.DataFrame]:
    if _is_parquet(path):
        if pq is None:
            yield pd.read_parquet(path)
            return
        pf = pq.ParquetFile(path)
        if chunksize and chunksize > 0:
            for batch in pf.iter_batches(batch_size=int(chunksize)):
                yield batch.to_pandas()
        else:
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                yield table.to_pandas()
        return

    if chunksize and chunksize > 0:
        for ch in pd.read_csv(path, low_memory=False, chunksize=int(chunksize)):
            yield ch
        return
    yield pd.read_csv(path, low_memory=False)


class _StreamWriter:
    def __init__(self, dest: Path):
        self.dest = dest
        self._mode = "parquet" if _is_parquet(str(dest)) else "csv"
        self._header_written = False
        self._parquet_writer = None
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        if self.dest.exists():
            print(f"[warn] output {self.dest} exists; overwriting.")
            self.dest.unlink()
        if self._mode == "parquet" and pq is None:
            raise SystemExit("pyarrow is required for parquet output. Install pyarrow or choose CSV.")

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self._mode == "parquet":
            assert pa is not None and pq is not None
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.dest, table.schema)
            self._parquet_writer.write_table(table)
            return
        comp = "gzip" if self.dest.name.lower().endswith(".gz") else "infer"
        df.to_csv(
            self.dest,
            index=False,
            mode="w" if not self._header_written else "a",
            header=not self._header_written,
            compression=comp,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        self._header_written = True

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_gse(chunk: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = chunk.copy()

    def s(col: Optional[str]) -> pd.Series:
        if col and col in df:
            return _num(df[col])
        return pd.Series(np.nan, index=df.index)

    # --- source columns (NaN-tolerant) ---
    gka_f = s(args.g_col)
    g_knee = s(args.knee_col)
    sfi = s(args.sfi_col)
    sfi2 = s(args.sfi2_col) if args.sfi2_col else s(args.sfi_col)

    shear_low = s(args.shear_low_col)
    shear_deep = s(args.shear_deep_col)
    shear_proxy = s(args.shear_proxy_col)
    strain_s = s(args.S_col)

    thermo = s(args.thermo_col)
    pdrop = s(args.pdrop_col)
    t2m_anom = s(args.t2m_anom_col)

    # --- G: geometry / structure ---
    # Missing pieces -> 0 contribution, not NaN.
    g_struct = (
        args.w_gka_F * gka_f.abs().fillna(0.0)
        + args.w_knee * g_knee.clip(lower=0).fillna(0.0)
        + args.w_sfi * sfi2.fillna(sfi).clip(lower=0).fillna(0.0)
    )

    # --- S: shear hostility ---
    shear_parts = []
    if shear_low.notna().any():
        shear_parts.append(shear_low)
    if shear_deep.notna().any():
        shear_parts.append(shear_deep)
    if shear_proxy.notna().any():
        shear_parts.append(shear_proxy)

    # If no explicit shear fields are populated, fall back to generic S/strain
    if not shear_parts and strain_s.notna().any():
        shear_parts.append(strain_s)

    if shear_parts:
        stacked = pd.concat(shear_parts, axis=1).fillna(0.0)
        s_shear = np.sqrt((stacked**2).sum(axis=1))
    else:
        s_shear = pd.Series(np.nan, index=df.index)

    # --- E: energy / fuel ---
    e_terms = []
    if thermo.notna().any():
        e_terms.append(thermo)
    if pdrop.notna().any():
        e_terms.append(pdrop)
    if t2m_anom.notna().any():
        e_terms.append(t2m_anom)
    if e_terms:
        e_energy = pd.concat(e_terms, axis=1).fillna(0).sum(axis=1)
    else:
        e_energy = pd.Series(0.0, index=df.index)

    df["G_struct"] = g_struct.astype("float32")
    df["S_shear"] = s_shear.astype("float32")
    df["E_energy"] = e_energy.astype("float32")

    # --- Mud feature block (S): reuse existing neighborhood metrics where available ---
    zeta = s("zeta")
    gka_dir = s("gka_dir_var")
    sph_vdr_std = s("sph_vdr_std")
    if sph_vdr_std.notna().any():
        s_zeta_var = sph_vdr_std
    elif zeta.notna().any():
        s_zeta_var = zeta.abs()
    else:
        s_zeta_var = pd.Series(np.nan, index=df.index)

    s_dir_var = gka_dir if gka_dir.notna().any() else pd.Series(np.nan, index=df.index)
    a_agree = (1.0 - gka_dir.clip(lower=0.0, upper=1.0)) if gka_dir.notna().any() else pd.Series(np.nan, index=df.index)

    df["S_zeta_var"] = s_zeta_var.astype("float32")
    df["S_dir_var"] = s_dir_var.astype("float32")
    df["A_agree"] = a_agree.astype("float32")

    # --- Same-time G×S interaction block ---
    eps = 1.0e-6
    g_safe = df["G_struct"].to_numpy(dtype="float32", copy=False)
    s_safe = df["S_shear"].to_numpy(dtype="float32", copy=False)
    a_safe = pd.to_numeric(df["A_agree"], errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)
    df["G_over_S"] = ((g_safe + eps) / (s_safe + eps)).astype("float32")
    df["S_over_G"] = ((s_safe + eps) / (g_safe + eps)).astype("float32")
    df["G_times_S"] = (g_safe * s_safe).astype("float32")
    df["G_times_S_disagree"] = (g_safe * s_safe * (1.0 - a_safe)).astype("float32")

    # --- Same-sign proxy (causal, same-time) ---
    chir = s("gka_chirality")
    if not chir.notna().any():
        chir = s("gka_parity_eta")
    if zeta.notna().any() and chir.notna().any():
        same_sign = (np.sign(zeta.fillna(0.0)) == np.sign(chir.fillna(0.0))).astype("float32")
    else:
        same_sign = pd.Series(0.0, index=df.index)
    df["same_sign"] = same_sign.astype("float32")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a Geometry/Shear/Energy panel from an ID-filtered subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--subset", required=True, help="ID-filtered subset file (CSV(.gz) or Parquet).")
    ap.add_argument("--out", required=True, help="Output panel file (CSV(.gz) or Parquet).")
    # Agent: accept overwrite flag for pipeline compatibility (writer overwrites by default).
    ap.add_argument("--overwrite", action="store_true", help="No-op; output is overwritten if present.")
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=400_000,
        dest="chunksize",
        help="Chunk size for streaming input (CSV rows or parquet batch size).",
    )
    ap.add_argument("--g-col", default="gka_F", help="Primary geometry column (strength/fitness).")
    ap.add_argument("--knee-col", default="gka_knee_ratio", help="Scale break / knee ratio column.")
    ap.add_argument("--sfi-col", default="SFI", help="SFI column name (fallback if SFI2 missing).")
    ap.add_argument("--sfi2-col", default="SFI2", help="SFI2 column name.")
    ap.add_argument("--shear-low-col", default="shear_low", help="Low-level shear column.")
    ap.add_argument("--shear-deep-col", default="shear_deep", help="Deep-layer shear column.")
    ap.add_argument("--shear-proxy-col", default="shear_proxy", help="Proxy shear column.")
    ap.add_argument("--S-col", default="S", help="Shear/strain magnitude column (fallback).")
    ap.add_argument("--thermo-col", default="thermo_shear", help="Thermo/shear composite column.")
    ap.add_argument("--pdrop-col", default="pdrop_nd", help="Normalised pressure drop column.")
    ap.add_argument("--t2m-anom-col", default="t2m_anom_local", help="2 m temperature anomaly column.")
    ap.add_argument("--w-gka-F", type=float, default=1.0, help="Weight for gka_F term in G_struct.")
    ap.add_argument("--w-knee", type=float, default=0.5, help="Weight for knee ratio term in G_struct.")
    ap.add_argument("--w-sfi", type=float, default=0.5, help="Weight for SFI/SFI2 term in G_struct.")
    args = ap.parse_args()

    writer = _StreamWriter(Path(args.out))

    total_rows = 0
    for i, chunk in enumerate(_iter_file(args.subset, args.chunksize), start=1):
        if chunk is None or chunk.empty:
            continue
        out_chunk = _compute_gse(chunk, args)
        writer.write(out_chunk)
        total_rows += len(out_chunk)
        print(f"[gse-panel] chunk {i}: wrote {len(out_chunk):,} rows (cum={total_rows:,})")

    writer.close()
    print(f"[done] wrote {total_rows:,} rows -> {args.out}")


if __name__ == "__main__":
    main()
