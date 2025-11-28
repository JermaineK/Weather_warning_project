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


def ensure_preferred_copy(path: Path, preferred: str | None, convert_existing: bool = True) -> Path:
    target = target_path(path, preferred)
    if preferred is None or target == path:
        return path

    if target.exists():
        return target

    if convert_existing and path.exists():
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
