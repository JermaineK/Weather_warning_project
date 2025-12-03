from __future__ import annotations

"""
env_check.py - lightweight runtime probes to suggest chunk sizes.

Uses utils.io_common helpers so recommendations stay consistent across scripts.
"""

import os
from typing import Dict, Any

from utils import io_common


def summarize() -> Dict[str, Any]:
    """
    Return a small dict with available memory, recommended CSV/Parquet chunk rows,
    and CPU count. Chunk sizing reuses io_common.recommend_chunk_rows().
    """
    avail = io_common.available_memory_bytes()
    csv_rows, parq_rows = io_common.recommend_chunk_rows()
    return {
        "available_bytes": avail,
        "available_gb": (avail / 1e9) if avail is not None else None,
        "csv_rows": csv_rows,
        "parquet_rows": parq_rows,
        "cpu_count": os.cpu_count(),
    }


def main() -> int:
    info = summarize()
    print("[env-check]", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
