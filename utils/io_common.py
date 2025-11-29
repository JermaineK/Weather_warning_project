from __future__ import annotations
from pathlib import Path
import importlib.util
import inspect
import pandas as pd

READ_DATE_COLS = ("time", "seed_time", "obs_time", "match_time")

def _csv_kwargs(extra_kw):
    kw = dict(memory_map=True, low_memory=True)

    # opt in to Arrow-backed columns where possible; this keeps large numeric
    # tables off the Python heap and helps avoid tokenization OOMs in pipelines
    if "dtype_backend" in inspect.signature(pd.read_csv).parameters:
        kw["dtype_backend"] = "pyarrow"
    if importlib.util.find_spec("pyarrow") is not None:
        kw.setdefault("engine", "pyarrow")

    # pandas' pyarrow CSV engine does not support memory_map
    if kw.get("engine") == "pyarrow":
        kw.pop("memory_map", None)

    kw.update(extra_kw)

    usecols = kw.get("usecols")
    if "parse_dates" in kw:
        parse = kw.get("parse_dates")
        if parse is True:
            parse = list(READ_DATE_COLS)
        elif isinstance(parse, (tuple, set)):
            parse = list(parse)
        elif isinstance(parse, str):
            parse = [parse]
        parse = parse or []
        if usecols:
            parse = [c for c in parse if c in usecols]
        if parse:
            kw["parse_dates"] = parse
        else:
            kw.pop("parse_dates", None)
    elif READ_DATE_COLS:
        parse = [c for c in READ_DATE_COLS if not usecols or c in usecols]
        if parse:
            kw["parse_dates"] = parse
    return kw


def _read_csv_with_missing_date_guard(path: str, kw: dict) -> pd.DataFrame:
    """Read a CSV while gracefully handling absent date columns.

    Some downstream tables omit optional date columns like ``match_time`` or
    ``obs_time``. When ``parse_dates`` requests columns that are absent,
    pandas raises ``ValueError``. To keep IO resilient (and avoid re-reading
    decompressed gzip blobs), retry without ``parse_dates`` when that happens.
    """

    try:
        return pd.read_csv(path, **kw)
    except ValueError as err:
        if "parse_dates" in kw and "Missing column provided to 'parse_dates'" in str(err):
            stripped = dict(kw)
            stripped.pop("parse_dates", None)
            return pd.read_csv(path, **stripped)
        raise


def read_any(path: str, parse_dates: tuple[str,...]=READ_DATE_COLS, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet",".parq",".pq")):
        return pd.read_parquet(path, **kw)

    if "parse_dates" not in kw and parse_dates:
        kw["parse_dates"] = list(parse_dates)
    kw = _csv_kwargs(kw)

    # First attempt with the preferred engine (often Arrow). If we exhaust
    # memory, progressively fall back to lighter-weight parsing options.
    try:
        return _read_csv_with_missing_date_guard(path, kw)
    except TypeError:
        # Older pandas versions do not support dtype_backend; retry without it
        kw = dict(kw)
        kw.pop("dtype_backend", None)
        return _read_csv_with_missing_date_guard(path, kw)
    except MemoryError:
        # The Arrow CSV engine can exhaust memory when reading large gzip files;
        # fall back to the default pandas engine which streams decompression.
        kw = dict(kw)
        kw.pop("engine", None)
        kw.pop("dtype_backend", None)
        kw.setdefault("low_memory", True)
        return _read_csv_with_missing_date_guard(path, kw)

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower()==".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
