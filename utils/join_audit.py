#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_audit.py

Lightweight join/merge guardrails to catch many-to-many explosions and
unexpected row growth. Emits a JSON log for pipeline diagnostics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _count_dupe_keys(df: pd.DataFrame, keys: Sequence[str]) -> int:
    if df.empty or not keys:
        return 0
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return 0
    return int(df.duplicated(subset=list(keys)).sum())


def _row_explosion(join_type: str, left_rows: int, right_rows: int, out_rows: int) -> bool:
    join_type = (join_type or "").lower()
    if join_type in {"left", "left_outer"}:
        return out_rows > left_rows
    if join_type in {"right", "right_outer"}:
        return out_rows > right_rows
    if join_type in {"inner", "outer"}:
        return out_rows > max(left_rows, right_rows)
    return out_rows > max(left_rows, right_rows)


def build_entry(
    *,
    step: str,
    keys: Sequence[str],
    join_type: str,
    left_rows: int,
    right_rows: int,
    out_rows: int,
    left_dupe_keys: int,
    right_dupe_keys: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    many_to_many = left_dupe_keys > 0 and right_dupe_keys > 0
    return {
        "step": step,
        "generated_at": _utc_now(),
        "join_type": join_type,
        "keys": list(keys),
        "left_rows": int(left_rows),
        "right_rows": int(right_rows),
        "out_rows": int(out_rows),
        "left_dupe_keys": int(left_dupe_keys),
        "right_dupe_keys": int(right_dupe_keys),
        "many_to_many": bool(many_to_many),
        "row_explosion": bool(_row_explosion(join_type, left_rows, right_rows, out_rows)),
        "extra": extra or {},
    }


def append_entry(path: str | Path | None, entry: Dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                payload = loaded
            elif isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
                payload = loaded["entries"]
        except Exception:
            payload = []
    payload.append(entry)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def enforce(entry: Dict[str, Any], allow_many_to_many: bool, allow_row_explosion: bool) -> None:
    if entry.get("many_to_many") and not allow_many_to_many:
        raise SystemExit(
            f"[join-audit] many-to-many merge detected in {entry.get('step')} "
            f"(left_dupe={entry.get('left_dupe_keys')} right_dupe={entry.get('right_dupe_keys')})."
        )
    if entry.get("row_explosion") and not allow_row_explosion:
        raise SystemExit(
            f"[join-audit] row explosion in {entry.get('step')}: out_rows={entry.get('out_rows')} "
            f"(left_rows={entry.get('left_rows')}, right_rows={entry.get('right_rows')})."
        )

