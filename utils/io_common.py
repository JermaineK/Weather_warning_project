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


def read_any(path: str, parse_dates: tuple[str,...]=READ_DATE_COLS, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet",".parq",".pq")):
        return pd.read_parquet(path, **kw)

    if "parse_dates" not in kw and parse_dates:
        kw["parse_dates"] = list(parse_dates)
    kw = _csv_kwargs(kw)

    def _read_csv_allow_missing_dates(kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except TypeError:
            # Older pandas versions do not support dtype_backend; retry without it
            fallback = dict(kwargs)
            fallback.pop("dtype_backend", None)
            return pd.read_csv(path, **fallback)
        except ValueError as e:
            if "Missing column provided to 'parse_dates'" in str(e):
                no_dates = dict(kwargs)
                no_dates.pop("parse_dates", None)
                return _read_csv_allow_missing_dates(no_dates)
            raise

    try:
        return _read_csv_allow_missing_dates(dict(kw))
    except (MemoryError, OSError):
        # The Arrow CSV engine can exhaust memory when reading large gzip files;
        # fall back to the default pandas engine which streams decompression.
        kw.pop("engine", None)
        kw.pop("dtype_backend", None)
        kw.pop("memory_map", None)
        kw.setdefault("low_memory", True)
        try:
            return _read_csv_allow_missing_dates(dict(kw))
        except (MemoryError, OSError, pd.errors.ParserError, ValueError):
            # If the C parser still fails with OOM or bad rows, drop to the
            # Python engine and stream the file in chunks to limit peak memory
            # usage. Also tolerate missing parse_dates columns.
            kw["engine"] = "python"
            kw.pop("chunksize", None)
            kw.pop("memory_map", None)
            try:
                iter_df = pd.read_csv(path, chunksize=200_000, **kw)
            except (ValueError, OSError) as e:
                if "Missing column provided to 'parse_dates'" in str(e):
                    kw.pop("parse_dates", None)
                    iter_df = pd.read_csv(path, chunksize=200_000, **kw)
                else:
                    raise
            return pd.concat(iter_df, ignore_index=True)

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower()==".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
