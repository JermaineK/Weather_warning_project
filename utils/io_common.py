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
        if isinstance(parse, str):
            parse = [parse]
        elif isinstance(parse, (tuple, set)):
            parse = list(parse)
        if isinstance(parse, list):
            if usecols:
                parse = [c for c in parse if c in usecols]
            if parse:
                kw["parse_dates"] = parse
            else:
                kw.pop("parse_dates", None)
        elif parse in (True, False):
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
        kw["parse_dates"] = parse_dates
    kw = _csv_kwargs(kw)

    # First attempt with the preferred engine (often Arrow). If we exhaust
    # memory, progressively fall back to lighter-weight parsing options.
    try:
        return pd.read_csv(path, **kw)
    except TypeError:
        kw.pop("dtype_backend", None)
        return pd.read_csv(path, **kw)
    except (MemoryError, pd.errors.ParserError):
        # Drop Arrow preferences and retry with pandas' C engine.
        retry_kw = dict(kw)
        retry_kw.pop("dtype_backend", None)
        retry_kw.pop("engine", None)
        retry_kw.setdefault("low_memory", True)
        try:
            return pd.read_csv(path, **retry_kw)
        except (MemoryError, pd.errors.ParserError):
            # Stream in chunks to avoid parser tokenization OOMs.
            stream_kw = dict(retry_kw)
            chunksize = stream_kw.pop("chunksize", None) or 200_000
            frames = pd.read_csv(path, chunksize=chunksize, iterator=True, **stream_kw)
            return pd.concat(frames, ignore_index=True)

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower()==".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")