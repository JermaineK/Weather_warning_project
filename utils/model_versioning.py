from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.run_naming import make_run_dir


# Agent: helper utilities for versioned model outputs + provenance tracking.


@dataclass(frozen=True)
class VersionedPaths:
    version_dir: Path
    version_id: str


def _safe_name(name: str | None) -> str:
    if not name:
        return ""
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    return safe


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            meta = pf.metadata
            meta_rows = int(meta.num_rows) if meta is not None else 0
            if meta_rows:
                return meta_rows
            if meta is not None:
                return int(sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups)))
            return int(sum(len(b) for b in pf.iter_batches(batch_size=200_000, columns=[])))
        except Exception:
            return 0
    if ext in {".csv", ".csv.gz"}:
        opener = gzip.open if ext == ".csv.gz" else open
        try:
            row_idx = -1
            with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
                for row_idx, _ in enumerate(fh):
                    pass
            return max(0, row_idx)
        except Exception:
            return 0
    try:
        return 1 if path.exists() and path.stat().st_size > 0 else 0
    except Exception:
        return 0


def file_signature(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return entry
    try:
        stat = path.stat()
        entry["size"] = int(stat.st_size)
        entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    except Exception:
        pass
    if path.is_file():
        entry["sha256"] = _sha256_file(path)
        entry["rows"] = _row_count(path)
    return entry


def git_commit(repo_root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
    except Exception:
        return None
    return None


def resolve_version_dir(
    model_dir: Path,
    run_name: str | None = None,
    run_id: str | None = None,
    allow_existing: bool = False,
) -> VersionedPaths:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    chosen = _safe_name(run_id) or _safe_name(run_name)
    if chosen:
        version_dir = model_dir / chosen
        if version_dir.exists() and not allow_existing:
            raise SystemExit(
                f"Versioned model folder already exists: {version_dir} "
                "(set --allow-existing-version to reuse)."
            )
        version_dir.mkdir(parents=True, exist_ok=True)
        return VersionedPaths(version_dir=version_dir, version_id=chosen)

    version_dir = make_run_dir(model_dir)
    return VersionedPaths(version_dir=version_dir, version_id=version_dir.name)


def write_provenance(
    path: Path,
    run_id: str | None,
    run_name: str | None,
    config_sha256: str | None,
    git_commit_hash: str | None,
    source_inputs: Iterable[Path],
    outputs: Iterable[Path] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "git_commit": git_commit_hash,
        "config_sha256": config_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "source_inputs": [file_signature(Path(p)) for p in source_inputs],
    }
    if outputs:
        payload["outputs"] = [file_signature(Path(p)) for p in outputs]
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def update_latest(model_dir: Path, version_dir: Path, artifacts: Iterable[Path]) -> Path:
    model_dir = Path(model_dir)
    latest_dir = model_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for src in artifacts:
        src_path = Path(src)
        if not src_path.exists() or not src_path.is_file():
            continue
        dst = latest_dir / src_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
    pointer = {
        "version_dir": str(version_dir),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    (model_dir / "latest.json").write_text(json.dumps(pointer, indent=2, ensure_ascii=True), encoding="utf-8")
    return latest_dir
