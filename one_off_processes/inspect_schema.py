#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_schema.py — zero/low-IO schema & key-column scanner for huge Parquet/CSV(.gz).

Usage (from project root):
  python data_subprocess/inspect_schema.py --glob "data/*.parquet" "data/*.csv.gz" --out manifest/schema_manifest.json

What it reports per file:
  • path, type, bytes
  • for Parquet: row_count (from metadata), column names & Arrow dtypes
  • for CSV: header columns only (no row scan; cheap)
  • presence of label-ish columns: pregen, storm*, event, label, y, target
  • presence of lead columns: lead_h, lead, lead_hours
  • presence of time column(s): time, time_hr, datetime, valid_time
  • for Parquet only: a tiny (head) sample of unique lead values (cheap if column is dictionary-encoded)
"""

from __future__ import annotations
import argparse, json, os, sys, gzip
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

LABEL_CANDIDATES = [
    "pregen", "storm", "storm_flag", "storm_any", "storm_next",
    "storm_future", "storm_future_any", "event", "label", "y", "target"
]
LEAD_CANDIDATES  = ["lead_h", "lead", "lead_hours"]
TIME_CANDIDATES  = ["time", "time_hr", "datetime", "valid_time"]

def is_parquet(p: Path) -> bool:
    return p.suffix.lower() in {".parquet", ".parq", ".pq", ".pqt"}

def is_csv_like(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".csv") or n.endswith(".csv.gz") or n.endswith(".gz")

def list_parquet_columns_meta(p: Path) -> Dict[str, Any]:
    # Use pyarrow metadata to avoid materializing data pages
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(p))
        schema = pf.schema_arrow
        cols = [f.name for f in schema]
        dtypes = {f.name: str(f.type) for f in schema}
        num_rows = 0
        try:
            # row groups available? sum quickly
            num_rows = sum(r.num_rows for r in pf.metadata.row_group_metadata)
        except Exception:
            try:
                num_rows = pf.metadata.num_rows
            except Exception:
                num_rows = 0

        # try to pull distinct lead values cheaply (dictionary stats) for common names
        lead_name = next((c for c in LEAD_CANDIDATES if c in cols), None)
        lead_values_sample: List[Any] = []
        if lead_name:
            try:
                # read just that column with a tiny scan (pyarrow is columnar)
                arr = pf.read([lead_name]).column(0)
                # dictionary encoded? obtain unique quickly
                try:
                    # pyarrow >= 12: unique works without full conversion
                    u = arr.unique().to_pylist()
                except Exception:
                    u = pd.Series(arr.to_pandas()).dropna().unique().tolist()
                lead_values_sample = sorted({int(x) for x in u if isinstance(x, (int, float))})[:20]
            except Exception:
                pass

        return dict(
            type="parquet",
            cols=cols,
            dtypes=dtypes,
            rows=num_rows,
            lead_values_sample=lead_values_sample
        )
    except Exception as e:
        return dict(type="parquet", error=f"{e.__class__.__name__}: {e}", cols=[], dtypes={}, rows=0)

def list_csv_header(p: Path) -> Dict[str, Any]:
    # Header-only sniff; avoid row counting (expensive for 2–10 GB gz)
    try:
        head = pd.read_csv(p, nrows=0, compression="infer", low_memory=False)
        cols = list(head.columns)
        return dict(type="csv", cols=cols, dtypes={}, rows=None)
    except Exception as e:
        # fallback: manual header via gzip (first non-empty line)
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                line = f.readline()
                cols = [c.strip() for c in line.split(",")] if line else []
            return dict(type="csv", cols=cols, dtypes={}, rows=None, note="gzip-fallback")
        except Exception as e2:
            return dict(type="csv", error=f"{e.__class__.__name__}: {e} | {e2}", cols=[], dtypes={}, rows=None)

def detect_keys(cols: List[str]) -> Dict[str, Any]:
    cols_set = set(cols)
    found_label = None
    for c in LABEL_CANDIDATES:
        if c in cols_set:
            found_label = c; break
        # case-insensitive fallback
        lower_map = {x.lower(): x for x in cols}
        if c.lower() in lower_map:
            found_label = lower_map[c.lower()]; break

    found_lead = None
    for c in LEAD_CANDIDATES:
        if c in cols_set:
            found_lead = c; break
        lower_map = {x.lower(): x for x in cols}
        if c.lower() in lower_map:
            found_lead = lower_map[c.lower()]; break

    found_time = None
    for c in TIME_CANDIDATES:
        if c in cols_set:
            found_time = c; break
        lower_map = {x.lower(): x for x in cols}
        if c.lower() in lower_map:
            found_time = lower_map[c.lower()]; break

    return dict(label=found_label, lead=found_lead, time=found_time)

def main():
    ap = argparse.ArgumentParser(description="Schema/column inspector for huge Parquet/CSV(.gz).")
    ap.add_argument("--glob", nargs="+", required=True, help="One or more file globs, e.g. data/*.parquet data/*.csv.gz")
    ap.add_argument("--out", default="manifest/schema_manifest.json", help="Where to write JSON manifest")
    ap.add_argument("--print-cols", action="store_true", help="Also print full column lists to console")
    args = ap.parse_args()

    # Expand globs (stable unique list)
    files: List[Path] = []
    for pat in args.glob:
        files.extend([Path(s) for s in sorted({*Path().glob(pat)})])
    files = [p for p in files if p.exists()]

    manifest: Dict[str, Any] = {"files": []}

    for p in files:
        entry: Dict[str, Any] = {
            "path": str(p),
            "bytes": p.stat().st_size,
        }
        if is_parquet(p):
            info = list_parquet_columns_meta(p)
        elif is_csv_like(p):
            info = list_csv_header(p)
        else:
            entry["type"] = "other"
            manifest["files"].append(entry)
            continue

        entry.update(info)

        # augment with key detections
        if "cols" in info and info.get("cols"):
            keys = detect_keys(info["cols"])
            entry.update(keys)

        manifest["files"].append(entry)

        # console summary
        size_gb = entry["bytes"] / (1024**3)
        msg = f"[{entry.get('type','?'):<7}] {p.name:40s}  size={size_gb:6.2f} GB"
        if entry.get("type") == "parquet":
            msg += f"  rows≈{entry.get('rows')}"
        label = entry.get("label") or "-"
        lead  = entry.get("lead") or "-"
        tcol  = entry.get("time") or "-"
        msg += f"  | label={label}  lead={lead}  time={tcol}"
        print(msg)

        if args.print_cols and entry.get("cols"):
            # avoid spamming the terminal—truncate if huge
            cols = entry["cols"]
            if len(cols) > 120:
                print("   cols(120 of {}): {}".format(len(cols), ", ".join(cols[:120]) + ", ..."))
            else:
                print("   cols({}): {}".format(len(cols), ", ".join(cols)))

        # show sample leads (parquet only, if available)
        if entry.get("type") == "parquet" and entry.get("lead_values_sample"):
            print(f"   lead sample: {entry['lead_values_sample'][:20]}")

    # write manifest
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[write] schema manifest -> {outp}")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try: sys.stdout.close()
        except Exception: pass
        try: sys.stderr.close()
        except Exception: pass