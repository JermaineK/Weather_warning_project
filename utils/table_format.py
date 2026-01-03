from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from utils import io_common


@dataclass(frozen=True)
class TableFormat:
    key: str
    suffix: str


_FORMATS: Dict[str, TableFormat] = {
    "parquet": TableFormat("parquet", ".parquet"),
    "parq": TableFormat("parquet", ".parquet"),
    "pq": TableFormat("parquet", ".parquet"),
    "csv.gz": TableFormat("csv.gz", ".csv.gz"),
    "csv_gz": TableFormat("csv.gz", ".csv.gz"),
    "csv-gz": TableFormat("csv.gz", ".csv.gz"),
    "csv": TableFormat("csv.gz", ".csv.gz"),
}


def normalize_preference(pref: str | None) -> str | None:
    if pref is None:
        return None
    p = pref.strip().lower().lstrip(".")
    fmt = _FORMATS.get(p)
    return fmt.key if fmt else None


def _is_table_like(path: Path) -> bool:
    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffixes in {".csv.gz", ".parquet"}:
        return True
    return path.suffix.lower() in {".csv", ".parquet"}


def _swap_suffix(path: Path, suffix: str) -> Path:
    stem = path.name
    if stem.endswith(".csv.gz"):
        stem = stem[: -len(".csv.gz")]
    elif "." in stem:
        stem = path.stem
    return path.with_name(stem + suffix)


def target_path(path: Path, preferred: str | None) -> Path:
    if preferred is None or not _is_table_like(path):
        return path
    suffix = _FORMATS[preferred].suffix
    suffixes = "".join(path.suffixes[-2:]).lower()
    current_fmt = "parquet" if suffixes == ".parquet" or path.suffix.lower() == ".parquet" else "csv.gz"
    if current_fmt == preferred:
        return path
    return _swap_suffix(path, suffix)


def _chunked_csv_to_parquet(src: Path, dest: Path, chunk_size: int | None = None) -> bool:
    """
    Last-resort, low-memory CSV->Parquet conversion using pandas chunks.
    Keeps the conversion from materializing the entire table when Arrow CSV
    type inference hits malformed columns (e.g., mostly-null columns with a
    stray numeric value).
    """
    if chunk_size is None:
        csv_rows, parq_rows = io_common.recommend_chunk_rows()  # type: ignore[attr-defined]
        chunk_size = csv_rows
    try:
        import pandas as pd  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Agent: avoid full-table reads; reuse the same date parsing defaults as io_common.read_any.
    csv_kw = io_common._csv_kwargs(src, {"parse_dates": io_common.READ_DATE_COLS})  # type: ignore[attr-defined]
    csv_kw.pop("engine", None)
    csv_kw.pop("dtype_backend", None)
    csv_kw.pop("memory_map", None)
    csv_kw.setdefault("low_memory", True)
    csv_kw["chunksize"] = chunk_size

    reader = io_common._read_csv_with_missing_date_guard(src, csv_kw)  # type: ignore[attr-defined]
    writer = None
    written = 0
    try:
        # pandas returns a TextFileReader iterator when chunksize is set.
        chunks = reader if not isinstance(reader, pd.DataFrame) else [reader]
        for chunk in chunks:
            if chunk is None or len(chunk) == 0:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="snappy")
            writer.write_table(table)
            written += len(chunk)
            if written and written % 1_000_000 == 0:
                print(f"[table-format] chunked csv->parquet rows written: {written:,}")

        if writer is None:
            # Empty input: still emit a valid parquet shell.
            pq.write_table(pa.table({}), dest)
        return True
    except Exception as exc:
        print(f"[table-format] chunked csv->parquet failed for {src}: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False
    finally:
        if writer is not None:
            writer.close()


def _stream_convert(src: Path, dest: Path) -> bool:
    """
    Try a streaming format conversion that avoids loading the full table into
    pandas DataFrames. This is especially helpful for large .csv.gz inputs that
    otherwise blow out memory when we eagerly materialize them during table
    format rewrites.
    """
    src_suffixes = "".join(src.suffixes[-2:]).lower()
    dest_suffixes = "".join(dest.suffixes[-2:]).lower()

    if dest_suffixes == ".parquet" and src_suffixes in {".csv.gz", ".csv"}:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.csv as pacsv  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        read_opts = pacsv.ReadOptions(block_size=8 * 1024 * 1024)
        convert_opts = pacsv.ConvertOptions(
            timestamp_parsers=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"],
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        )

        writer = None
        try:
            with pacsv.open_csv(src, read_options=read_opts, convert_options=convert_opts) as reader:
                for batch in reader:
                    if writer is None:
                        writer = pq.ParquetWriter(dest, batch.schema, compression="snappy")
                    writer.write_batch(batch)

                if writer is None:
                    # Empty CSV: still emit a valid parquet shell.
                    schema = getattr(reader, "schema", None)
                    empty = pa.Table.from_batches([], schema=schema) if schema else pa.table({})
                    pq.write_table(empty, dest)
        except Exception as exc:
            print(f"[table-format] streaming csv->parquet failed for {src}: {exc}")
            return _chunked_csv_to_parquet(src, dest)
        finally:
            if writer is not None:
                writer.close()
        return True

    return False


def ensure_preferred_copy(path: Path, preferred: str | None, convert_existing: bool = True) -> Path:
    target = target_path(path, preferred)
    if preferred is None or target == path:
        return path

    if target.exists():
        if convert_existing and path.exists():
            try:
                src_mtime = path.stat().st_mtime
                dst_mtime = target.stat().st_mtime
                if src_mtime > dst_mtime:
                    print(f"[table-format] refreshing {target} from newer {path}")
                    try:
                        target.unlink()
                    except Exception:
                        pass
                    if _stream_convert(path, target):
                        return target
                    df = io_common.read_any(path)
                    io_common.write_any(str(target), df)
                    return target
            except Exception:
                pass
        return target

    if convert_existing and path.exists():
        if _stream_convert(path, target):
            return target
        df = io_common.read_any(path)
        io_common.write_any(str(target), df)
        return target

    return target


def rewrite_step_paths(step: Dict[str, object], preferred: str | None, convert_existing: bool = True) -> Dict[str, object]:
    if preferred is None:
        return step

    rewritten: Dict[str, object] = {}
    for k, v in step.items():
        if isinstance(v, str):
            p = Path(v)
            if _is_table_like(p):
                target = ensure_preferred_copy(p, preferred, convert_existing=convert_existing)
                rewritten[k] = str(target)
                continue
        rewritten[k] = v
    return rewritten
