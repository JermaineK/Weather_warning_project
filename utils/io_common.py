from __future__ import annotations
from pathlib import Path
import pandas as pd

READ_DATE_COLS = ("time", "seed_time", "obs_time", "match_time")

def read_any(path: str, parse_dates: tuple[str,...]=READ_DATE_COLS, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet",".parq",".pq")):
        return pd.read_parquet(path, **kw)
    return pd.read_csv(path, low_memory=False,
                       parse_dates=[c for c in parse_dates if c in (kw.get("usecols") or []) or True], **kw)

def write_any(path: str, df: pd.DataFrame) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if low.endswith(".csv.gz") or p.suffix.lower()==".gz" else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")