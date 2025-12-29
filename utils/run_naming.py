from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional


def make_run_dir(root: Path, date_str: Optional[str] = None) -> Path:
    """
    Create a new run directory under `root` with pattern YYYYMMDD_runNNN.
    """
    if date_str is None:
        date_str = datetime.now(UTC).strftime("%Y%m%d")

    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{date_str}_run"
    existing_nums: List[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        try:
            existing_nums.append(int(suffix))
        except ValueError:
            continue

    next_n = max(existing_nums) + 1 if existing_nums else 1
    run_dir = root / f"{prefix}{next_n:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir
