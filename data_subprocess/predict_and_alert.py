#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_and_alert.py

Agent: combine base + specialist models into P_final with alert-regime mask.
Streams large tables (pyarrow when available) and writes per-chunk outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as ds  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None
    ds = None
    pq = None


# ---------------------------------------------------------------------------#
# IO helpers                                                                 #
# ---------------------------------------------------------------------------#

def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _peek_columns(path: str | Path) -> List[str]:
    if _is_parquet(path):
        if ds is None:
            return list(pd.read_parquet(path, nrows=1).columns)
        return ds.dataset(path).schema.names  # type: ignore[arg-type]
    return list(pd.read_csv(path, nrows=2, low_memory=False).columns)


def _stream_frames(path: str, columns: Sequence[str], chunk_rows: int | None, parquet_rows: int | None) -> Iterable[pd.DataFrame]:
    if _is_parquet(path) and ds is not None:
        scanner = ds.dataset(path).scanner(columns=list(columns), batch_size=parquet_rows or chunk_rows or None)
        for batch in scanner.to_batches():
            yield batch.to_pandas()
        return
    if _is_parquet(path):
        yield pd.read_parquet(path, columns=list(columns))
        return
    kwargs = {"usecols": list(columns), "low_memory": False}
    if "time" in columns:
        kwargs["parse_dates"] = ["time"]
    for chunk in pd.read_csv(path, chunksize=chunk_rows or None, **kwargs):
        yield chunk


def _write_stream(path: str | Path, df: pd.DataFrame, writer, first: bool):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _is_parquet(p):
        if pa is None or pq is None:
            if not first and p.exists():
                raise SystemExit("pyarrow required for streaming parquet writes.")
            df.to_parquet(p, index=False)
            return writer, False
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(p, table.schema)
        writer.write_table(table)
        return writer, False
    mode = "w" if first else "a"
    header = first
    comp = "gzip" if str(p).lower().endswith(".gz") else "infer"
    df.to_csv(p, index=False, mode=mode, header=header, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
    return writer, False


# ---------------------------------------------------------------------------#
# Core                                                                       #
# ---------------------------------------------------------------------------#

def _select_features(columns: Sequence[str], feature_list: Sequence[str] | None, prefixes: Sequence[str], label_cols: Sequence[str]) -> List[str]:
    if feature_list:
        return [c for c in feature_list if c in columns]
    pref = tuple(prefixes)
    feats = [c for c in columns if c not in label_cols and c.startswith(pref)]
    return feats or [c for c in columns if c not in label_cols]


def _mask_regime(df: pd.DataFrame, base_prob: np.ndarray, args) -> np.ndarray:
    mask = base_prob >= args.base_thr
    if args.G_thr is not None and "G" in df.columns:
        mask |= pd.to_numeric(df["G"], errors="coerce").to_numpy() >= args.G_thr
    if args.SFI_thr is not None and "SFI" in df.columns:
        mask |= pd.to_numeric(df["SFI"], errors="coerce").to_numpy() >= args.SFI_thr
    return mask


def _predict_chunk(df: pd.DataFrame, features: Sequence[str], model, calibrator) -> np.ndarray:
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mdl = calibrator if calibrator is not None else model
    prob = mdl.predict_proba(X)[:, 1]
    return prob


def predict(path: str, args) -> None:
    out_path = Path(args.outfile)
    if out_path.exists() and not args.overwrite:
        print(f"[predict] output exists; skipping (use --overwrite): {out_path}")
        return
    if out_path.exists() and args.overwrite:
        try:
            out_path.unlink()
        except Exception:
            pass
    cols_all = _peek_columns(path)
    prefixes = [p.strip() for p in args.feature_prefixes.split(",") if p.strip()]
    label_cols = {args.time_col, "lat", "lon", args.label_col}
    features = _select_features(cols_all, args.feature_cols, prefixes, label_cols)
    if not features:
        raise SystemExit("No feature columns selected for prediction.")

    extra_cols = {"lead_h", "t_to_storm_min_h", "row_id", "cell_id", "ilat", "ilon"}
    columns_needed = {args.time_col, args.label_col, "lat", "lon", "G", "SFI", *features, *extra_cols}
    columns_needed = [c for c in columns_needed if c in cols_all]
    writer = None
    first = True

    def _require(path_str: str, name: str):
        p = Path(path_str)
        if not p.exists():
            raise SystemExit(f"[predict] missing {name}: {p}. Train the model first or point to an existing file.")
        return p

    base_model_path = _require(args.base_model, "base_model")
    base_model = joblib.load(base_model_path)
    base_calibrator = joblib.load(args.base_calibrator) if args.base_calibrator else None
    spec_model = None
    spec_calibrator = None
    if args.specialist_model:
        spec_model = joblib.load(_require(args.specialist_model, "specialist_model"))
    if args.specialist_calibrator:
        spec_calibrator = joblib.load(args.specialist_calibrator)

    total_rows = 0
    for chunk in _stream_frames(path, list(columns_needed), args.chunk_rows, args.parquet_rows):
        if chunk.empty:
            continue
        base_prob = _predict_chunk(chunk, features, base_model, base_calibrator)
        chunk["P_base"] = base_prob
        regime = _mask_regime(chunk, base_prob, args)
        if spec_model is not None:
            spec_df = chunk.loc[regime]
            if not spec_df.empty:
                spec_prob = _predict_chunk(spec_df, features, spec_model, spec_calibrator)
                chunk.loc[regime, "P_spec"] = spec_prob
        if "P_spec" in chunk.columns:
            blend = args.blend_weight
            chunk["P_final"] = np.where(
                regime,
                blend * chunk["P_spec"].fillna(chunk["P_base"]) + (1.0 - blend) * chunk["P_base"],
                chunk["P_base"],
            )
        else:
            chunk["P_final"] = chunk["P_base"]
        chunk["alert_mask"] = chunk["P_final"] >= args.alert_thr
        if "lead_h" not in chunk.columns and "t_to_storm_min_h" in chunk.columns:
            lead_vals = pd.to_numeric(chunk["t_to_storm_min_h"], errors="coerce")
            chunk["lead_h"] = np.where(np.isfinite(lead_vals), np.ceil(lead_vals), np.nan)
        writer, first = _write_stream(args.outfile, chunk, writer, first)
        total_rows += len(chunk)

    if writer is not None and hasattr(writer, "close"):
        writer.close()
    print(f"[predict] wrote {args.outfile} rows={total_rows:,}")


# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#

def parse_args():
    ap = argparse.ArgumentParser(
        description="Blend base + specialist predictions into P_final.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features", required=True, help="Feature table (parquet/csv).")
    ap.add_argument("--base-model", required=True, help="Base model path (joblib).")
    ap.add_argument("--base-calibrator", default=None, help="Optional calibrator for base model.")
    ap.add_argument("--specialist-model", default=None, help="Specialist model path (joblib).")
    ap.add_argument("--specialist-calibrator", default=None, help="Optional calibrator for specialist.")
    ap.add_argument("--feature-cols", nargs="*", default=None, help="Explicit feature columns.")
    ap.add_argument(
        "--feature-prefixes",
        default="gka_,sph_,SFI,S3,zeta,div,msl_,dG_,dE_,G_,E_",
        help="Prefixes for auto feature selection if --feature-cols not set.",
    )
    ap.add_argument("--label-col", default="storm")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--base-thr", type=float, default=0.6, help="Alert-regime threshold on P_base.")
    ap.add_argument("--alert-thr", type=float, default=0.6, help="Alert flag threshold on P_final.")
    ap.add_argument("--G-thr", type=float, default=0.85, help="Optional G threshold for regime mask.")
    ap.add_argument("--SFI-thr", type=float, default=0.85, help="Optional SFI threshold for regime mask.")
    ap.add_argument("--blend-weight", type=float, default=0.5, help="Weight for specialist when regime is true.")
    ap.add_argument("--outfile", required=True, help="Output predictions table.")
    ap.add_argument(
        "--chunk-rows",
        "--chunk_rows",
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunk size.",
    )
    ap.add_argument(
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=200_000,
        help="Parquet batch size.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return ap.parse_args()


def main():
    args = parse_args()
    predict(args.features, args)


if __name__ == "__main__":
    main()
