from __future__ import annotations

from typing import Any, Dict, List, Tuple


_NO_STRIP_KEYS = set()
_ALIAS_KEYS = {
    "normalize-lon": "normalize_lon",
    "keep-tmp": "keep_tmp",
    "chunk-rows": "chunk_rows",
    "parquet-rows": "parquet_rows",
    "out-dir": "out_dir",
    "report-pack-config": "report_pack_config",
    "report-pack-out-dir": "report_pack_out_dir",
}
_DROP_IF_EMPTY = {"time_format"}


def _is_empty(val: Any) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "")


def normalize_config(cfg: Any) -> tuple[Any, List[str]]:
    """
    Normalize config keys/values:
      - Replace '-' with '_' in dict keys.
      - Strip leading/trailing whitespace in string values.

    Returns (normalized_cfg, change_log).
    """
    changes: List[str] = []

    def _norm(obj: Any, path: str, skip_strip: bool = False) -> Any:
        if isinstance(obj, dict):
            out: Dict[Any, Any] = {}
            for key, val in obj.items():
                new_key = key
                if isinstance(key, bool):
                    new_key = "on" if key else "off"
                    changes.append(f"{path or 'config'}: key {key} -> '{new_key}' (yaml-bool key)")
                if isinstance(new_key, str):
                    if new_key in _ALIAS_KEYS:
                        mapped = _ALIAS_KEYS[new_key]
                        changes.append(f"{path or 'config'}: key '{new_key}' -> '{mapped}' (alias)")
                        new_key = mapped
                    else:
                        repl = new_key.replace("-", "_")
                        if repl != new_key:
                            changes.append(f"{path or 'config'}: key '{new_key}' -> '{repl}'")
                            new_key = repl
                new_path = f"{path}.{new_key}" if path else str(new_key)
                skip_child = isinstance(new_key, str) and new_key in _NO_STRIP_KEYS
                norm_val = _norm(val, new_path, skip_strip=skip_child)
                if isinstance(new_key, str) and new_key in _DROP_IF_EMPTY and _is_empty(norm_val):
                    changes.append(f"{new_path}: empty value removed")
                    continue
                if new_key in out:
                    if _is_empty(out[new_key]) and not _is_empty(norm_val):
                        out[new_key] = norm_val
                        changes.append(f"{new_path}: filled from alias key")
                    else:
                        changes.append(f"{new_path}: duplicate key ignored")
                    continue
                out[new_key] = norm_val
            return out
        if isinstance(obj, list):
            return [_norm(v, f"{path}[]", skip_strip=skip_strip) for v in obj]
        if isinstance(obj, str):
            if skip_strip:
                return obj
            stripped = obj.strip()
            if stripped != obj:
                changes.append(f"{path or 'config'}: stripped leading/trailing whitespace")
            key_leaf = path.split(".")[-1] if path else ""
            if key_leaf == "area" and stripped:
                parts = [p for p in stripped.split(",") if p.strip() != ""]
                if len(parts) != 4:
                    changes.append(f"{path}: invalid area format (expected 4 comma-separated values)")
            if key_leaf == "normalize_lon" and stripped:
                if stripped not in {"-180..180", "0..360", "none"}:
                    changes.append(f"{path}: unexpected normalize_lon '{stripped}'")
            return stripped
        return obj

    return _norm(cfg, ""), changes
