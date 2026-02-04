#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_train_subset_by_id.py

Two-pass memory-aware subset builder:

  1) Read ONLY a handful of columns (id + label/lead, plus any --small-extra-cols)
     from a huge labelled grid, decide which IDs are "interesting"
     (e.g., pre-storm funnel) + a sample of quiet IDs.

  2) Stream the full file (all features) in chunks and keep only rows whose ID is in
     that selected set. Write out a smaller training subset.

This lets you use ID to avoid loading the whole dataset into RAM at once.

Notes:
  • ID values are normalised via _to_python_id (int-or-str) in BOTH passes, so
    '1234' and 1234 match consistently even if CSV type inference differs.
  • If you use --positive-query / --quiet-query, make sure any referenced
    columns are included via --small-extra-cols.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


def _is_parquet(path: str) -> bool:
    low = path.lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _read_small_view(path: str, cols: List[str]) -> pd.DataFrame:
    """
    Read only a small set of columns from a big file.
    Supports parquet or CSV.
    """
    if _is_parquet(path):
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, usecols=cols, low_memory=False)


def _iter_full_file(path: str, chunksize: Optional[int] = None) -> Iterable[pd.DataFrame]:
    """
    Stream the full file (all columns) in chunks.

    - For CSV: use pandas chunksize.
    - For parquet: iterate batches via pyarrow if available (uses chunksize as
      batch size when provided), otherwise iterate row groups. Fallback: one
      big read (not ideal for huge files).
    """
    if _is_parquet(path):
        if pq is None:
            yield pd.read_parquet(path)
            return
        pf = pq.ParquetFile(path)
        if chunksize and chunksize > 0:
            for batch in pf.iter_batches(batch_size=int(chunksize)):
                yield batch.to_pandas()
            return
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg)
            yield batch.to_pandas()
        return

    if not chunksize or chunksize <= 0:
        yield pd.read_csv(path, low_memory=False)
        return

    it = pd.read_csv(path, low_memory=False, chunksize=int(chunksize))
    for ch in it:
        yield ch


class _SubsetWriter:
    """
    Streaming writer that supports CSV(.gz) and parquet.
    For parquet, uses ParquetWriter when available for chunked writes.
    """

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
            raise SystemExit("pyarrow is required for parquet output. Install pyarrow or use CSV(.gz).")

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

        comp = "infer"
        if self.dest.name.lower().endswith(".gz"):
            comp = "gzip"
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


def _mask_from_query(df: pd.DataFrame, query: str, label: str) -> pd.Series:
    try:
        mask = df.eval(query, engine="python")
    except Exception as exc:  # pragma: no cover - defensive error path
        raise SystemExit(f"Failed to evaluate {label} query '{query}': {exc}") from exc
    if mask is None:
        raise SystemExit(f"{label} query '{query}' returned None.")
    if not isinstance(mask, pd.Series):
        raise SystemExit(f"{label} query '{query}' did not return a Series.")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return mask.fillna(False)


def _to_python_id(val: object) -> Optional[object]:
    """
    Normalise an ID to a stable Python object:
      • NaN -> None (dropped)
      • numeric-like -> int(val)
      • otherwise -> str(val)
    """
    if pd.isna(val):
        return None
    try:
        return int(val)
    except Exception:
        return str(val)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a storm-focused training subset by selecting IDs from a small view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--labelled",
        required=True,
        help="Huge labelled grid (CSV or Parquet). Must contain id and label/lead cols.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output subset file (CSV(.gz) or Parquet).",
    )
    ap.add_argument(
        "--id-col",
        default="row_id",
        help="Unique ID column to filter on (default: row_id).",
    )
    ap.add_argument(
        "--lead-col",
        default="t_to_storm_min_h",
        help="Column used to define the pre-storm funnel (default: t_to_storm_min_h).",
    )
    ap.add_argument(
        "--max-lead-h",
        type=float,
        default=240.0,
        help="Max future-hours to treat as 'pre-storm' (default: 240).",
    )
    ap.add_argument(
        "--positive-query",
        default=None,
        help=(
            "Optional pandas eval query to define positives on the small view, "
            "e.g., 'pregen_h24 == 1'. If set, overrides the default 0..max-lead-h funnel. "
            "Columns referenced here must be loaded via --small-extra-cols."
        ),
    )
    ap.add_argument(
        "--quiet-query",
        default=None,
        help=(
            "Optional pandas eval query to define quiet/background rows. "
            "If omitted, quiet rows are those with lead outside (0, max-lead-h]. "
            "If positive-query is set and quiet-query is not, quiet rows are simply ~positive."
        ),
    )
    ap.add_argument(
        "--small-extra-cols",
        nargs="*",
        default=[],
        help=(
            "Additional columns to load in the small view so they can be used "
            "in --positive-query / --quiet-query (e.g., pregen_h24, G_struct)."
        ),
    )
    ap.add_argument(
        "--quiet-mult",
        type=float,
        default=2.0,
        help="Quiet rows per positive row (approximate; counted at row-level).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional final cap on total subset rows (0 = no cap).",
    )
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=500_000,
        dest="chunksize",
        help="Chunk size for streaming input (CSV rows or parquet batch size).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    # Agent: accept overwrite flag for pipeline compatibility (writer overwrites by default).
    ap.add_argument("--overwrite", action="store_true", help="No-op; output is overwritten if present.")
    args = ap.parse_args()

    src = args.labelled
    out_path = Path(args.out)
    id_col = args.id_col
    lead_col = args.lead_col

    # ----- PASS 1: SMALL VIEW (ID + LEAD + OPTIONAL EXTRAS) -----
    base_small_cols = [id_col, lead_col]
    extra_cols = [c for c in (args.small_extra_cols or []) if c not in base_small_cols]
    small_cols = base_small_cols + extra_cols

    print(f"[pass1] reading small view from {src}")
    print(f"[pass1] columns: {small_cols}")
    small = _read_small_view(src, small_cols)

    if id_col not in small.columns:
        raise SystemExit(f"ID column '{id_col}' not found. Columns: {list(small.columns)[:10]} ...")
    if lead_col not in small.columns and args.positive_query is None:
        raise SystemExit(
            f"Lead/label column '{lead_col}' not found and no --positive-query set. "
            f"Columns: {list(small.columns)[:10]} ..."
        )

    # Normalise ID column in the small view
    small_id_norm = small[id_col].apply(_to_python_id)
    if small_id_norm.isna().all():
        raise SystemExit(f"All values in id_col='{id_col}' are NaN/invalid after normalisation.")

    lead = pd.to_numeric(small[lead_col], errors="coerce") if lead_col in small.columns else None

    # ----- positives -----
    if args.positive_query:
        m_pre = _mask_from_query(small, args.positive_query, label="positive")
    else:
        # default funnel: 0 <= lead <= max_lead_h  (0 = now / already-storming)
        m_pre = (lead >= 0.0) & (lead <= float(args.max_lead_h))

    pre_ids = small_id_norm.loc[m_pre].dropna().to_numpy()
    n_pre_rows = int(m_pre.sum())
    n_pre_ids_unique = int(np.unique(pre_ids).size)
    print(f"[pass1] positive rows: {n_pre_rows:,}  unique IDs: {n_pre_ids_unique:,}")

    if pre_ids.size == 0:
        print("[warn] No positive rows found; nothing to do.")
        if lead is not None:
            lead_valid = pd.to_numeric(lead, errors="coerce")
            finite = lead_valid[np.isfinite(lead_valid)]
            within = finite[(finite >= 0) & (finite <= float(args.max_lead_h))]
            print(
                f"[diag] lead_col='{lead_col}': total={len(lead_valid):,} "
                f"finite={len(finite):,} finite<=max_lead_h={len(within):,}"
            )
            if len(finite):
                print(
                    "[diag] lead stats "
                    f"min={finite.min():.3f} med={finite.median():.3f} "
                    f"max={finite.max():.3f}"
                )
        for lbl in ("pregen", "storm_window", "storm_point", "near_storm"):
            if lbl in small.columns:
                vc = small[lbl].value_counts(dropna=False).head()
                print(f"[diag] {lbl} value_counts:\n{vc}")
        print("[diag] If you expect positives, verify t_to_storm_min_h and label columns upstream.")
        return

    # ----- quiet/background -----
    if args.quiet_query:
        m_quiet = _mask_from_query(small, args.quiet_query, label="quiet")
    elif args.positive_query:
        m_quiet = ~m_pre
    else:
        # default quiet: lead is NaN or > max_lead_h
        m_quiet = (lead.isna()) | (lead > float(args.max_lead_h))

    quiet_ids_all = small_id_norm.loc[m_quiet].dropna().to_numpy()
    n_quiet_rows = int(m_quiet.sum())
    n_quiet_ids_unique = int(np.unique(quiet_ids_all).size)
    print(f"[pass1] quiet candidate rows: {n_quiet_rows:,}  unique IDs: {n_quiet_ids_unique:,}")

    # ----- sample quiet at row-level -----
    rng = np.random.default_rng(args.seed)
    n_pre = pre_ids.size
    n_quiet_target = int(min(quiet_ids_all.size, args.quiet_mult * n_pre))
    print(
        f"[pass1] quiet target rows (row-level) = "
        f"min({quiet_ids_all.size:,}, {args.quiet_mult} * {n_pre:,}) = {n_quiet_target:,}"
    )

    if n_quiet_target > 0:
        quiet_sample = rng.choice(quiet_ids_all, size=n_quiet_target, replace=False)
    else:
        quiet_sample = np.array([], dtype=object)

    # ----- build final ID set -----
    selected_ids = np.concatenate([pre_ids, quiet_sample])
    selected_id_set: Set[object] = {
        v for v in (_to_python_id(x) for x in selected_ids) if v is not None
    }
    if args.max_rows and len(selected_id_set) > args.max_rows:
        orig_size = len(selected_id_set)
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(list(selected_id_set), size=args.max_rows, replace=False)
        selected_id_set = set(sampled)
        print(
            f"[pass1] max_rows cap: downsampled selected IDs to {len(selected_id_set):,} "
            f"(from {orig_size:,})"
        )
    print(
        f"[pass1] total selected IDs after combining positives + quiet (unique) = "
        f"{len(selected_id_set):,}"
    )

    # free small stuff
    del small, lead, pre_ids, quiet_ids_all, quiet_sample, selected_ids, small_id_norm

    # ----- PASS 2: STREAM FULL FILE AND FILTER BY ID -----
    writer = _SubsetWriter(out_path)
    total_kept = 0
    found_ids: Set[object] = set()

    print(f"[pass2] streaming full {src} and filtering by {id_col} in selected_id_set")
    for i, chunk in enumerate(_iter_full_file(src, chunksize=args.chunksize), start=1):
        if chunk is None or chunk.empty:
            continue
        if id_col not in chunk.columns:
            raise SystemExit(f"Chunk {i} missing id_col='{id_col}'. Got columns: {list(chunk.columns)[:10]} ...")

        # normalise IDs in this chunk to match selected_id_set semantics
        chunk_ids_norm = chunk[id_col].apply(_to_python_id)
        mask = chunk_ids_norm.isin(selected_id_set)

        sub = chunk.loc[mask].copy()
        kept_here = len(sub)
        if kept_here == 0:
            print(f"[pass2] chunk {i}: kept 0 rows (cum={total_kept:,})")
            continue

        writer.write(sub)
        total_kept += kept_here
        found_ids.update(sub[id_col].apply(_to_python_id))

        print(f"[pass2] chunk {i}: kept {kept_here:,} rows (cum={total_kept:,})")

        if len(found_ids) >= len(selected_id_set):
            print("[pass2] seen all selected IDs; stopping early.")
            break

    writer.close()
    print(f"[done] wrote {total_kept:,} rows -> {out_path}")


if __name__ == "__main__":
    main()
