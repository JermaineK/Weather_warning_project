#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_pipeline.py — ultra-thin orchestrator (step-level skip support)

New:
- In the `features.steps` list, each step may include:
    enabled: false         # skip this step entirely
    skip_if_exists: true   # skip if output already exists (checks out/outfile/out_csv)

Updated:
- Dropped the runtime_subprocess section.
- Seeds now use the seeds_subprocess/seeds_tracks.py manager.
- Reports now use reports_subprocess/reports_and_maps_manager.py with step-style config.
"""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import shutil
import shlex
import re
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Tuple
from collections import Counter

from utils import table_format
from utils import env_check
from utils import config_normalize
from pipeline_contracts import (
    contract_for,
    order_steps,
    postflight_step,
    preflight_step,
    summarize_step,
)

try:
    import yaml  # type: ignore
except Exception:
    # Fallback shim: if PyYAML is not available, provide a minimal safe_load
    # that attempts to parse the config as JSON. This keeps the script usable
    # in environments without pyyaml while encouraging installation for full YAML support.
    import json

    class _YAMLShim:
        @staticmethod
        def safe_load(s):
            s = s or ""
            if not s.strip():
                return {}
            try:
                return json.loads(s)
            except Exception:
                print(
                    "Warning: failed to parse config as JSON; install 'pyyaml' for full YAML support (pip install pyyaml).",
                    file=sys.stderr,
                )
                return {}

    yaml = _YAMLShim()

HERE = Path(__file__).resolve().parent
PREFERRED_TABLE_FORMAT: str | None = None
ENV_HINTS: Dict[str, Any] | None = None
FORCE_KEEP_QUANTILE: float | None = None
AUTO_OVERWRITE_ON_INVALID: bool = True
CONFIG_SHA256: str | None = None
GIT_COMMIT: str | None = None
RUN_NAME: str | None = None
RUN_MANIFEST_PATH: Path | None = None
CONFIG_PATH: Path | None = None
CACHE_CFG: Dict[str, Any] = {}
LOG_PATH: Path | None = None
LOG_FH = None
PIPELINE_CODE_SHA256: str | None = None
RUN_LOCK_PATH: Path | None = None

# ---------------- shell helpers ----------------

def sh(cmd: List[Union[str, Path]], check: bool = True) -> int:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    repo_root = str(HERE)
    py_path = env.get("PYTHONPATH", "")
    if repo_root not in py_path.split(os.pathsep):
        env["PYTHONPATH"] = repo_root + (os.pathsep + py_path if py_path else "")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for s in self._streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                continue
        return False


def _close_log() -> None:
    global LOG_FH
    if LOG_FH is None:
        return
    try:
        LOG_FH.flush()
        LOG_FH.close()
    except Exception:
        pass
    LOG_FH = None


def _init_logging(run_name: str | None, log_file: str | None) -> None:
    global LOG_PATH, LOG_FH
    if LOG_FH is not None:
        return
    if log_file and str(log_file).strip().lower() in {"none", "off", "false"}:
        return
    safe_run = _safe_run_name(run_name)
    if log_file:
        path = Path(log_file)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = Path("results/runs") / safe_run / "logs" / f"{safe_run}_{ts}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH = path
    LOG_FH = path.open("w", encoding="utf-8")
    atexit.register(_close_log)
    sys.stdout = _Tee(sys.stdout, LOG_FH)
    sys.stderr = _Tee(sys.stderr, LOG_FH)
    print(f"[log] writing to {path}")


def _ensure_loky_cpu_count() -> None:
    """
    Silence joblib/loky physical-core warnings by setting a sane default.
    """
    if os.environ.get("LOKY_MAX_CPU_COUNT"):
        return
    count = os.cpu_count()
    if not count:
        return
    os.environ["LOKY_MAX_CPU_COUNT"] = str(count)
    print(f"[env] LOKY_MAX_CPU_COUNT={count} (logical cores)")


def _pid_alive(pid: int) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes  # pragma: no cover - platform-specific

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return None


def _release_run_lock() -> None:
    global RUN_LOCK_PATH
    if RUN_LOCK_PATH is None:
        return
    try:
        info = _load_json(RUN_LOCK_PATH)
        if info.get("pid") not in {None, os.getpid()}:
            return
        RUN_LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _acquire_run_lock(run_name: str | None, force: bool) -> None:
    if not run_name:
        return
    global RUN_LOCK_PATH
    safe_run = _safe_run_name(run_name)
    lock_path = Path("results/runs") / safe_run / "locks" / "run_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        info = _load_json(lock_path)
        pid = info.get("pid")
        alive = _pid_alive(pid) if isinstance(pid, int) else None
        if alive is True and not force:
            raise SystemExit(
                f"[lock] run already active for {safe_run} (pid {pid}). "
                "Use --force-lock to override."
            )
        if alive is None and not force:
            raise SystemExit(
                f"[lock] existing lock for {safe_run}, unable to verify pid {pid}. "
                "Use --force-lock to override."
            )
        try:
            lock_path.unlink()
        except Exception:
            pass
    payload = {
        "pid": os.getpid(),
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exe": sys.executable,
        "cmdline": " ".join(sys.argv),
    }
    _write_json(lock_path, payload)
    RUN_LOCK_PATH = lock_path
    atexit.register(_release_run_lock)


def _canonical_config_path(cfg_path: Path) -> Path:
    name = cfg_path.name.lower()
    if ".canonical." in name or name.endswith(".canonical.yaml") or name.endswith(".canonical.yml"):
        return cfg_path
    return cfg_path.with_suffix(".canonical.yaml")

def _is_autofix_path(cfg_path: Path) -> bool:
    name = cfg_path.name.lower()
    return name.endswith(".autofix.yaml") or name.endswith(".autofix.yml")

def _autofix_config_path(cfg_path: Path) -> Path:
    if _is_autofix_path(cfg_path):
        return cfg_path
    return cfg_path.with_suffix(".autofix.yaml")

def _flatten_kv(prefix: str, obj: Any) -> List[str]:
    """
    Turn a nested structure into CLI flags.
    - scalars:          {"thr": 0.8}                  -> ["--thr", "0.8"]
    - lists (scalars):  {"leads": [24,48]}            -> ["--leads", "24,48"]
    - dict nested:      {"pick": {"metric": "F1"}}    -> ["--pick.metric", "F1"]
    - truthy booleans:  {"write_parquet": True}       -> ["--write-parquet"]
    - falsy booleans:   skipped
    """
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_kv(key, v))
        return out
    if isinstance(obj, bool):
        if obj:
            out.append(f"--{prefix.replace('_','-')}")
        return out
    if isinstance(obj, (list, tuple)):
        if not obj:
            return out
        joined = ",".join(map(str, obj))
        out += [f"--{prefix.replace('_','-')}", joined]
        return out
    key_leaf = prefix.split(".")[-1]
    if isinstance(obj, str):
        val = obj.strip()
        if val == "":
            return out
        if key_leaf in {"normalize_lon", "normalize-lon", "area"}:
            flag = f"--{prefix.replace('_','-')}"
            out.append(f"{flag}={val}")
            return out
        out += [f"--{prefix.replace('_','-')}", val]
        return out
    out += [f"--{prefix.replace('_','-')}", str(obj)]
    return out


_STEP_CLI_SKIP_PREFIXES = ("health_", "health-")

def _step_cli_args(step: Dict[str, Any], skip_keys: Iterable[str] = ()) -> List[str]:
    # Agent: avoid passing health-only config keys to subprocess CLIs.
    skip = {k for k in skip_keys if k}
    payload: Dict[str, Any] = {}
    for k, v in step.items():
        if k in skip:
            continue
        if isinstance(k, str) and k.startswith(_STEP_CLI_SKIP_PREFIXES):
            continue
        payload[k] = v
    return _flatten_kv("", payload)


def _apply_table_format(step: Dict[str, Any], convert_existing: bool = True) -> Dict[str, Any]:
    if not PREFERRED_TABLE_FORMAT:
        return step
    return table_format.rewrite_step_paths(step, PREFERRED_TABLE_FORMAT, convert_existing=convert_existing)

def _apply_runtime_hints(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject recommended chunk sizes when a step has chunk-related keys unset.
    Keys recognized: chunk_rows, chunksize (only if already present), parquet_rows.
    """
    if ENV_HINTS is None:
        return step
    csv_rows = ENV_HINTS.get("csv_rows")
    parq_rows = ENV_HINTS.get("parquet_rows")

    def set_if_missing(key: str, val):
        if key not in step or step[key] in (None, "", 0):
            step[key] = val

    if csv_rows:
        set_if_missing("chunk_rows", csv_rows)
        if "chunksize" in step:
            set_if_missing("chunksize", csv_rows)
    if parq_rows:
        set_if_missing("parquet_rows", parq_rows)
    return step

def _maybe_force_keep_quantile(step: Dict[str, Any], section: str, mode: str | None = None) -> Dict[str, Any]:
    """
    Optional global override: force any keep-quantile style key to FORCE_KEEP_QUANTILE.
    """
    if FORCE_KEEP_QUANTILE is None or not isinstance(step, dict):
        return step
    out = dict(step)
    for k in list(step.keys()):
        if not isinstance(k, str):
            continue
        norm = k.replace("-", "_")
        if norm == "keep_quantile" and step.get(k) is not None:
            before = step.get(k)
            out[k] = FORCE_KEEP_QUANTILE
            ctx = f"{section}.{mode}" if mode else section
            print(f"[keep-quantile] force_keep_quantile active -> overriding {ctx} {k} {before} -> {FORCE_KEEP_QUANTILE}")
    return out

def _maybe_force_overwrite(step: Dict[str, Any], section: str, mode: str | None = None) -> Dict[str, Any]:
    """
    If auto-overwrite is enabled and the step already declares an overwrite key,
    flip it to True when we detect invalid existing outputs.
    """
    if not AUTO_OVERWRITE_ON_INVALID or not isinstance(step, dict):
        return step
    if "overwrite" not in step:
        return step
    if step.get("overwrite") is True:
        return step
    out = dict(step)
    out["overwrite"] = True
    ctx = f"{section}.{mode}" if mode else section
    print(f"[overwrite] {ctx}: forcing overwrite=true due to invalid output")
    return out


def _handle_invalid_existing_output(
    section: str,
    mode: str,
    step: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    ctx = f"{section}.{mode}"
    if step.get("overwrite") is True:
        return _maybe_force_overwrite(step, section, mode)
    if AUTO_OVERWRITE_ON_INVALID:
        return _maybe_force_overwrite(step, section, mode)
    raise SystemExit(
        f"[{section}] existing output failed validation for {ctx} ({exc}); "
        "set overwrite=true (or enable auto_overwrite_on_invalid) to rebuild."
    )

def _mgr(path_parts: Iterable[str]) -> Path:
    return HERE.joinpath(*path_parts).resolve()


def _resolve_script(path: Union[str, Path]) -> Path:
    """Resolve a script path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else HERE.joinpath(p).resolve()

# ---------------- common helper ----------------

def _candidate_out_path(step: Dict[str, Any]) -> Path | None:
    """
    Try to discover an output path field to check for existence.
    We check common keys used across our scripts.
    For out_dir we do NOT skip, since that's usually a folder.
    """
    for k in ("out", "outfile", "out_csv", "out_md", "out_png"):
        v = step.get(k)
        if isinstance(v, str) and v.strip():
            return Path(v)
    return None

def _should_skip_if_exists(step: Dict[str, Any]) -> bool:
    return bool(step.get("skip_if_exists")) and not bool(step.get("overwrite"))


def _progress(tag: str, idx: int, total: int, label: str) -> None:
    total = max(total, 1)
    print(f"[{tag}] step {idx}/{total}: {label}")


def _is_output_key(key: str) -> bool:
    key = key.lower()
    if key in {"out", "outfile", "outcomes"}:
        return key != "outcomes"
    if key.startswith(("out_", "out-", "outdir", "outfile", "outcsv")):
        return True
    if key.endswith(("_out", "-out", "_out_dir", "-out-dir", "_outdir", "-outdir")):
        return True
    if "_out_" in key or "-out-" in key:
        return True
    return False


def _output_paths(step: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for k, v in step.items():
        if not isinstance(v, str) or not v.strip():
            continue
        if _is_output_key(k):
            paths.append(Path(v))
    return paths


def _input_paths(step: Dict[str, Any]) -> tuple[list[Path], list[str]]:
    files: List[Path] = []
    globs: List[str] = []
    known_file_ext = {
        ".parquet",
        ".parq",
        ".pq",
        ".csv",
        ".gz",
        ".json",
        ".txt",
        ".md",
        ".png",
        ".gif",
        ".pkl",
        ".yaml",
        ".yml",
        ".nc",
    }
    known_path_prefixes = (
        "data/",
        "data\\",
        "results/",
        "results\\",
        "models/",
        "models\\",
        "config/",
        "config\\",
        "./",
        ".\\",
        "../",
        "..\\",
    )
    literal_keys = {
        "normalize_lon",
        "normalize-lon",
        "area",
        "hours",
        "quantiles",
        "lead_bins",
        "lat_bands",
        "leads",
        "lead_hours",
        "ablation_sets",
        "vars",
        "pl_vars",
        "pl_levels",
        "features",
        "past_windows",
        "lags",
        "include_prefixes",
        "drop_patterns",
        "keep_prefixes",
    }

    for k, v in step.items():
        if not isinstance(v, str) or not v.strip():
            continue
        key = k.lower()
        if key in {"mode", "enabled", "skip_if_exists", "recipe"}:
            continue
        if key in literal_keys:
            continue
        if _is_output_key(k):
            continue
        if "glob" in key:
            globs.append(v)
            continue

        vv = v.strip()
        p = Path(vv)
        suffix = p.suffix.lower()
        looks_like_path = (
            suffix in known_file_ext
            or "/" in vv
            or "\\" in vv
            or "*" in vv
            or "?" in vv
            or vv.startswith(known_path_prefixes)
        )
        if looks_like_path:
            files.append(p)

    return files, globs


def _safe_run_name(name: str | None) -> str:
    if not name:
        return "run"
    safe_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.":
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "run"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_stage_manifest_for_autofix(run_name: str | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not run_name:
        return {}, {}
    safe_run = _safe_run_name(run_name)
    path = Path("results/runs") / safe_run / "manifests" / "stages.json"
    data = _load_json(path)
    stages = data.get("stages", {})
    meta = {
        "pipeline_code_sha256": data.get("pipeline_code_sha256"),
        "config_sha256": data.get("config_sha256"),
        "git_commit": data.get("git_commit"),
        "log_path": data.get("log_path"),
    }
    return (stages if isinstance(stages, dict) else {}), meta


def _load_run_manifest(run_name: str | None) -> Dict[str, Any]:
    if not run_name:
        return {}
    safe_run = _safe_run_name(run_name)
    path = Path("results/runs") / safe_run / "manifests" / "stages.json"
    return _load_json(path)


def _autofix_stale_reason(
    section: str,
    mode: str,
    step: Dict[str, Any],
    stages: Dict[str, Any],
) -> str | None:
    if not step.get("skip_if_exists"):
        return None
    outputs = _output_paths(step)
    if outputs and not any(p.exists() for p in outputs):
        return None
    entry = stages.get(f"{section}.{mode}")
    if not isinstance(entry, dict):
        return None
    if entry.get("health_ok") is False:
        return "previous_health_failed"
    prev_fp = entry.get("input_fingerprint")
    try:
        curr_fp, _ = _step_fingerprint(section, mode, step)
    except Exception:
        curr_fp = None
    if prev_fp and curr_fp and prev_fp != curr_fp:
        return "input_fingerprint_mismatch"
    prev_outputs = {o.get("path") for o in entry.get("outputs", []) if isinstance(o, dict) and o.get("path")}
    curr_outputs = {str(p) for p in outputs} if outputs else set()
    if prev_outputs and curr_outputs and prev_outputs != curr_outputs:
        return "output_path_mismatch"
    return None


def _load_diagnostics_checks(
    run_name: str | None,
) -> tuple[List[Dict[str, Any]], Path | None, float | None]:
    reports_dir = Path("results/reports")
    if not reports_dir.exists():
        return [], None, None
    candidates = list(reports_dir.glob("*/diagnostics/diagnostics.json"))
    if not candidates:
        return [], None, None
    chosen = None
    if run_name:
        for p in candidates:
            data = _load_json(p)
            if data.get("run_name") == run_name:
                if chosen is None or p.stat().st_mtime > chosen.stat().st_mtime:
                    chosen = p
    if chosen is None:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    data = _load_json(chosen)
    checks = data.get("checks", [])
    try:
        diag_mtime = chosen.stat().st_mtime
    except Exception:
        diag_mtime = None
    return (checks if isinstance(checks, list) else []), chosen, diag_mtime


def _normalize_path_str(value: str) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _latest_log_path(run_name: str | None, manifest: Dict[str, Any]) -> Path | None:
    if manifest.get("log_path"):
        p = Path(str(manifest.get("log_path")))
        if p.exists():
            return p
    if not run_name:
        return None
    safe_run = _safe_run_name(run_name)
    log_dir = Path("results/runs") / safe_run / "logs"
    if not log_dir.exists():
        return None
    candidates = [p for p in log_dir.glob("*.log") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _cmd_to_section_mode(cmd_line: str) -> tuple[str, str] | None:
    if not cmd_line:
        return None
    try:
        tokens = shlex.split(cmd_line, posix=False)
    except Exception:
        tokens = cmd_line.split()
    if not tokens:
        return None
    script_idx = None
    for i, tok in enumerate(tokens):
        if str(tok).lower().endswith(".py"):
            script_idx = i
            break
    if script_idx is None:
        return None
    script = Path(tokens[script_idx]).name.lower()
    manager_map = {
        "features_manager.py": "features",
        "data_stage_manager.py": "data_stage",
        "training_manager.py": "training",
        "alerts_logic_manager.py": "alerts_logic",
        "sweep_manager.py": "sweep",
        "eval_manager.py": "eval",
        "seeds_tracks.py": "seeds",
        "reports_and_maps_manager.py": "report",
    }
    section = manager_map.get(script)
    if not section:
        return None
    mode = None
    for tok in tokens[script_idx + 1:]:
        if str(tok).startswith("-"):
            continue
        mode = str(tok)
        break
    if section == "report" and not mode:
        mode = "bundle"
    if not mode:
        return None
    return section, mode


def _parse_run_log_errors(log_path: Path | None) -> Dict[str, Any]:
    info = {
        "manager_failed": set(),
        "missing_paths": set(),
        "trace_cmds": [],
        "error_cmds": [],
    }
    if not log_path or not log_path.exists():
        return info
    mgr_fail = re.compile(r"\[manager\] STEP ([^\s:]+) FAILED", re.IGNORECASE)
    no_such = re.compile(r"No such file or directory: ['\\\"]([^'\\\"]+)['\\\"]", re.IGNORECASE)
    missing_re = re.compile(r"missing (?:required|file|input|glob)[^:]*:?\\s*(.+)$", re.IGNORECASE)
    cmd_re = re.compile(r"^\\$\\s+(.+)$")
    last_cmd = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip()
            m = cmd_re.match(line)
            if m:
                last_cmd = m.group(1)
            m = mgr_fail.search(line)
            if m:
                info["manager_failed"].add(m.group(1))
            m = no_such.search(line)
            if m:
                info["missing_paths"].add(m.group(1))
            if "missing" in line.lower():
                m = missing_re.search(line)
                if m:
                    cand = m.group(1).strip().strip("'\"")
                    if any(sep in cand for sep in ("\\", "/")):
                        info["missing_paths"].add(cand)
                if line.rstrip().endswith(")"):
                    m = re.search(r"\\(([^)]+)\\)$", line)
                    if m:
                        cand = m.group(1).strip().strip("'\"")
                        if any(sep in cand for sep in ("\\", "/")):
                            info["missing_paths"].add(cand)
            if "Traceback" in line and last_cmd:
                info["trace_cmds"].append(last_cmd)
            low = line.lower()
            if (("error:" in low) or ("exception" in low)) and last_cmd:
                info["error_cmds"].append(last_cmd)
    return info


def _stage_output_index(stages: Dict[str, Any]) -> Dict[str, set[str]]:
    index: Dict[str, set[str]] = {}
    for key, entry in stages.items():
        if not isinstance(entry, dict):
            continue
        for out in entry.get("outputs", []) or []:
            if not isinstance(out, dict):
                continue
            path = out.get("path")
            if not path:
                continue
            norm = _normalize_path_str(path)
            index.setdefault(norm, set()).add(key)
    return index


def _config_output_index(cfg: Dict[str, Any], sections: List[str]) -> Dict[str, set[str]]:
    index: Dict[str, set[str]] = {}
    for section in sections:
        for step in _section_steps_for_fix(cfg, section):
            if not isinstance(step, dict):
                continue
            mode = str(step.get("mode", "")).strip()
            if not mode:
                continue
            for p in _output_paths(step):
                norm = _normalize_path_str(str(p))
                index.setdefault(norm, set()).add(f"{section}.{mode}")
    return index


def _match_stage_outputs(output_index: Dict[str, set[str]], missing_path: str) -> List[str]:
    norm = _normalize_path_str(missing_path)
    matches: set[str] = set()
    if norm in output_index:
        matches.update(output_index[norm])
    for out_norm, keys in output_index.items():
        if norm.endswith(out_norm) or out_norm.endswith(norm):
            matches.update(keys)
        elif norm.startswith(out_norm) or out_norm.startswith(norm):
            matches.update(keys)
    return sorted(matches)


def _expand_upstream_deps(section: str, mode: str) -> List[str]:
    deps: set[str] = set()
    stack = [mode]
    while stack:
        curr = stack.pop()
        contract = contract_for(section, curr)
        if not contract or not contract.dependencies:
            continue
        for dep in contract.dependencies:
            if dep in deps:
                continue
            deps.add(dep)
            stack.append(dep)
    return sorted(deps)


def _collect_step_index(cfg: Dict[str, Any], sections: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for section in sections:
        steps = _section_steps_for_fix(cfg, section)
        for step in steps:
            if not isinstance(step, dict):
                continue
            mode = str(step.get("mode", "")).strip()
            if not mode:
                continue
            index[(section, mode)] = step
    return index


def _step_upstreams(section: str, mode: str, step: Dict[str, Any]) -> set[Tuple[str, str]]:
    upstream: set[Tuple[str, str]] = set()
    contract = contract_for(section, mode)
    if not contract:
        return upstream
    for dep in contract.dependencies or ():
        upstream.add((section, dep))
    for spec in contract.inputs:
        for prod in spec.produced_by or ():
            if "." not in prod:
                continue
            sec, mod = prod.split(".", 1)
            upstream.add((sec, mod))
    return upstream


def _build_downstream_index(
    cfg: Dict[str, Any],
    sections: List[str],
) -> Dict[Tuple[str, str], set[Tuple[str, str]]]:
    steps = _collect_step_index(cfg, sections)
    downstream: Dict[Tuple[str, str], set[Tuple[str, str]]] = {key: set() for key in steps}
    for (section, mode), step in steps.items():
        for upstream in _step_upstreams(section, mode, step):
            if upstream in steps:
                downstream.setdefault(upstream, set()).add((section, mode))
    return downstream


def _expand_downstream_deps(
    cfg: Dict[str, Any],
    sections: List[str],
    section: str,
    mode: str,
) -> List[Tuple[str, str]]:
    downstream = _build_downstream_index(cfg, sections)
    start = (section, mode)
    if start not in downstream:
        return []
    seen: set[Tuple[str, str]] = set()
    stack = [start]
    while stack:
        curr = stack.pop()
        for nxt in downstream.get(curr, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    seen.discard(start)
    return sorted(seen)


def _collect_last_run_error_reasons(
    cfg: Dict[str, Any],
    run_name: str | None,
    stages_manifest: Dict[str, Any],
    sections: List[str],
) -> Dict[Tuple[str, str], set[str]]:
    reasons: Dict[Tuple[str, str], set[str]] = {}
    def _add(section: str, mode: str, reason: str) -> None:
        reasons.setdefault((section, mode), set()).add(reason)

    for key, entry in stages_manifest.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("health_ok") is False:
            if "." in key:
                section, mode = key.split(".", 1)
                _add(section, mode, "last_run_health_failed")

    manifest = _load_run_manifest(run_name)
    log_path = _latest_log_path(run_name, manifest)
    log_info = _parse_run_log_errors(log_path)
    if log_info.get("manager_failed"):
        if _find_step_ref(cfg, "report", "bundle") is not None:
            for step in sorted(log_info.get("manager_failed", [])):
                _add("report", "bundle", f"last_run_manager_failed:{step}")
            _add("report", "bundle", "last_run_report_failure")

    output_index = _stage_output_index(stages_manifest)
    config_index = _config_output_index(cfg, sections) if sections else {}
    for path in sorted(log_info.get("missing_paths", [])):
        matches = _match_stage_outputs(output_index, path)
        if not matches and config_index:
            matches = _match_stage_outputs(config_index, path)
        for key in matches:
            if "." not in key:
                continue
            section, mode = key.split(".", 1)
            _add(section, mode, f"last_run_missing_input:{path}")

    for cmd in log_info.get("trace_cmds", []):
        mapped = _cmd_to_section_mode(cmd)
        if mapped:
            section, mode = mapped
            _add(section, mode, "last_run_traceback")
    for cmd in log_info.get("error_cmds", []):
        mapped = _cmd_to_section_mode(cmd)
        if mapped:
            section, mode = mapped
            _add(section, mode, "last_run_error")

    return reasons


def _apply_last_run_reasons(
    cfg: Dict[str, Any],
    run_name: str | None,
    stages_manifest: Dict[str, Any],
    sections: List[str],
    changes: List[str],
    add_overwrite: bool,
    log_mtime: float | None,
) -> set[Tuple[str, str]]:
    touched: set[Tuple[str, str]] = set()
    last_run_reasons = _collect_last_run_error_reasons(cfg, run_name, stages_manifest, sections)
    if not last_run_reasons:
        return touched
    slowtick_failed = False
    report_reasons = last_run_reasons.get(("report", "bundle"), set())
    if any(str(r).startswith("last_run_manager_failed:slowtick-diag") for r in report_reasons):
        slowtick_failed = True
    for (section, mode), reasons in sorted(last_run_reasons.items()):
        step_ref = _find_step_ref(cfg, section, mode)
        if step_ref is None:
            continue
        # Agent: skip stale log-driven rebuilds only if outputs are newer than the log.
        step_for_preflight = dict(step_ref)
        step_for_preflight = _apply_runtime_hints(step_for_preflight)
        step_for_preflight = _apply_table_format(step_for_preflight, convert_existing=False)
        pref = preflight_step(section, step_for_preflight)
        outputs = _output_paths(step_for_preflight)
        force_on_manager_fail = any(str(r).startswith("last_run_manager_failed:") for r in reasons)
        if outputs and any(p.exists() for p in outputs) and not force_on_manager_fail:
            out_mtime = _outputs_max_mtime(outputs) if log_mtime is not None else None
            if out_mtime is not None and log_mtime is not None and out_mtime >= log_mtime:
                health_ok, _ = _outputs_health(step_for_preflight, pref.expected_output_columns or [])
                if health_ok:
                    continue
            elif log_mtime is None:
                health_ok, _ = _outputs_health(step_for_preflight, pref.expected_output_columns or [])
                if health_ok:
                    continue
        reason = "last_run_error: " + ", ".join(sorted(reasons))
        if _mark_overwrite(
            cfg, section, mode, reason, changes, add_overwrite, stages_manifest, log_mtime, True
        ):
            touched.add((section, mode))
        needs_upstream = any(r.startswith("last_run_missing_input:") for r in reasons)
        if needs_upstream:
            for dep in _expand_upstream_deps(section, mode):
                if _mark_overwrite(
                    cfg,
                    section,
                    dep,
                    f"upstream of {section}.{mode} ({reason})",
                    changes,
                    add_overwrite,
                    stages_manifest,
                    log_mtime,
                    True,
                ):
                    touched.add((section, dep))
    if slowtick_failed:
        if _mark_overwrite(
            cfg,
            "alerts_logic",
            "viability-pipeline",
            "slowtick-diag failed; refresh alerts for lead-specific coverage",
            changes,
            add_overwrite,
            stages_manifest,
            log_mtime,
            True,
        ):
            touched.add(("alerts_logic", "viability-pipeline"))
    return touched


def _diagnostics_overwrite_targets(issue: str) -> List[Tuple[str, str]]:
    key = issue.strip().lower()
    mapping: Dict[str, List[Tuple[str, str]]] = {
        "lead_h missing": [
            ("data_stage", "viability-targets"),
            ("data_stage", "state-transitions"),
            ("training", "predict-alerts"),
            ("alerts_logic", "apply-thresholds"),
            ("alerts_logic", "throttle"),
            ("alerts_logic", "denoise"),
            ("alerts_logic", "viability-pipeline"),
        ],
        "identical per-lead inputs": [
            ("data_stage", "viability-targets"),
            ("data_stage", "state-transitions"),
            ("training", "train-base"),
        ],
        "train/val overlap": [
            ("training", "train-base"),
            ("training", "train-alert-specialist"),
            ("data_stage", "train-viability"),
        ],
        "forbidden/leaky columns in features": [
            ("training", "train-base"),
            ("training", "train-alert-specialist"),
            ("data_stage", "train-viability"),
        ],
        "non-causal features in training": [
            ("training", "train-base"),
            ("training", "train-alert-specialist"),
            ("data_stage", "train-viability"),
        ],
        "slowtick identical across leads": [
            ("alerts_logic", "apply-thresholds"),
            ("alerts_logic", "throttle"),
            ("alerts_logic", "denoise"),
            ("alerts_logic", "viability-pipeline"),
        ],
        "flow direction missing in matches": [
            ("report", "bundle"),
            ("report", "objects-by-hour"),
            ("report", "object-matches"),
        ],
        "time-shift test": [
            ("features", "join-labels-grid"),
            ("data_stage", "state-transitions"),
            ("training", "train-base"),
        ],
        "join audit": [
            ("features", "join-features"),
            ("features", "integrate-thermo"),
            ("features", "join-labels-grid"),
        ],
    }
    return mapping.get(key, [])


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _git_commit(repo_root: Path) -> str | None:
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


def _config_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pipeline_code_sha256(root: Path) -> str | None:
    uniq = _pipeline_code_paths(root)
    if not uniq:
        return None
    h = hashlib.sha256()
    for p in uniq:
        h.update(str(p).encode("utf-8"))
        try:
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        except Exception:
            continue
    return h.hexdigest()


def _pipeline_code_paths(root: Path) -> List[Path]:
    targets: List[Path] = [root / "run_pipeline.py", root / "pipeline_contracts.py"]
    for folder in [
        "fetch_subprocess",
        "features_subprocess",
        "data_subprocess",
        "alerts_logic_subprocess",
        "sweep_subprocess",
        "eval_subprocess",
        "seeds_subprocess",
        "reports_subprocess",
        "utils",
    ]:
        base = root / folder
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if "archive" in p.parts:
                continue
            if p.name.endswith(".bak"):
                continue
            targets.append(p)
    return sorted({p.resolve() for p in targets if p.exists()})


def _pipeline_code_mtime(root: Path) -> float | None:
    latest = None
    for p in _pipeline_code_paths(root):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    return latest


def _cache_enabled(section: str, mode: str, step: Dict[str, Any]) -> bool:
    if step.get("cache") is True:
        return True
    if not CACHE_CFG.get("enabled"):
        return False
    stages = CACHE_CFG.get("stages", set())
    sections = CACHE_CFG.get("sections", set())
    if stages and f"{section}.{mode}" in stages:
        return True
    if sections and section in sections:
        return True
    return False


def _expand_input_entries(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    files, globs = _input_paths(step)
    seen = set()
    entries: List[Dict[str, Any]] = []

    for g in globs:
        matches = sorted(Path().glob(g))
        if not matches:
            entries.append({"glob": g, "exists": False})
            continue
        for mp in matches:
            try:
                key = str(mp.resolve())
            except Exception:
                key = str(mp)
            if key in seen:
                continue
            seen.add(key)
            entries.append(_path_signature(mp))

    for p in files:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        entries.append(_path_signature(p))

    return entries


def _path_signature(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return entry
    try:
        stat = path.stat()
        entry["size"] = int(stat.st_size)
        entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass
    if path.is_file():
        entry["sha256"] = _sha256_file(path)
    return entry


def _path_max_mtime(path: Path, max_children: int = 2000) -> float | None:
    try:
        max_mtime = path.stat().st_mtime
    except Exception:
        return None
    if path.is_dir():
        count = 0
        try:
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                count += 1
                if count > max_children:
                    break
                try:
                    mtime = child.stat().st_mtime
                except Exception:
                    continue
                if mtime > max_mtime:
                    max_mtime = mtime
        except Exception:
            pass
    return max_mtime


def _outputs_max_mtime(paths: List[Path]) -> float | None:
    latest = None
    for p in paths:
        if not p.exists():
            continue
        mtime = _path_max_mtime(p)
        if mtime is None:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    return latest


def _autofix_step_outputs(
    section: str,
    mode: str,
    step: Dict[str, Any],
    stages_manifest: Dict[str, Any],
) -> List[Path]:
    outputs = _output_paths(step)
    if outputs:
        return outputs
    entry = stages_manifest.get(f"{section}.{mode}") if stages_manifest else None
    if isinstance(entry, dict):
        for out in entry.get("outputs", []) or []:
            if isinstance(out, dict) and out.get("path"):
                outputs.append(Path(str(out.get("path"))))
    return outputs


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _step_fingerprint(section: str, mode: str, step: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    params = {
        k: v for k, v in step.items()
        if k not in {"enabled", "skip_if_exists", "cache", "overwrite"}
    }
    inputs = _expand_input_entries(step)
    payload = {
        "section": section,
        "mode": mode,
        "params": params,
        "inputs": inputs,
        "config_sha256": CONFIG_SHA256,
        "git_commit": GIT_COMMIT,
        "pipeline_code_sha256": PIPELINE_CODE_SHA256,
        "table_format": PREFERRED_TABLE_FORMAT,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), inputs


def _cache_root() -> Path:
    root = CACHE_CFG.get("dir") or "cache"
    return Path(root)


def _cache_stage_dir(section: str, mode: str) -> Path:
    safe_mode = _safe_run_name(mode)
    return _cache_root() / section / safe_mode


def _cache_index_path(section: str, mode: str) -> Path:
    return _cache_stage_dir(section, mode) / "index.json"


def _load_cache_index(section: str, mode: str) -> Dict[str, Any]:
    return _load_json(_cache_index_path(section, mode))


def _save_cache_index(section: str, mode: str, payload: Dict[str, Any]) -> None:
    _write_json(_cache_index_path(section, mode), payload)


def _cacheable_outputs(step: Dict[str, Any]) -> List[Path]:
    outputs = _output_paths(step)
    cacheable: List[Path] = []
    for p in outputs:
        if p.suffix or (p.exists() and p.is_file()):
            cacheable.append(p)
    return cacheable


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


def _read_sample(path: Path, columns: List[str], nrows: int) -> Any:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()
    try:
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(columns=columns, batch_size=nrows):
                    return batch.to_pandas()
                return None
            except Exception:
                import pandas as pd  # type: ignore
                df = pd.read_parquet(path, columns=columns)
                return df.head(nrows) if nrows else df
        if ext in {".csv", ".csv.gz"}:
            import pandas as pd  # type: ignore
            return pd.read_csv(path, usecols=columns, nrows=nrows, low_memory=False, compression="infer")
    except Exception:
        return None
    return None


def _health_check_outputs(
    paths: List[Path],
    expected_cols: List[str],
    max_missing: float,
    scan_rows: int,
    missing_ok_cols: Sequence[str] | None = None,
) -> tuple[bool, List[Dict[str, Any]], str | None]:
    outputs_info: List[Dict[str, Any]] = []
    health_ok = True
    reason = None
    missing_ok = {str(c).strip() for c in (missing_ok_cols or []) if str(c).strip()}
    for p in paths:
        info: Dict[str, Any] = {"path": str(p)}
        if not p.exists():
            info["exists"] = False
            outputs_info.append(info)
            health_ok = False
            reason = reason or "missing_output"
            continue
        info["exists"] = True
        if p.is_dir():
            try:
                entry_count = sum(1 for _ in p.iterdir())
            except Exception:
                entry_count = None
            info["is_dir"] = True
            info["entry_count"] = entry_count
            if entry_count is not None and entry_count <= 0:
                health_ok = False
                reason = reason or "empty_dir"
            outputs_info.append(info)
            continue
        rows = _row_count(p)
        info["rows"] = rows
        if rows <= 0:
            health_ok = False
            reason = reason or "empty_output"
        if expected_cols and p.suffix.lower() in {".parquet", ".pq", ".pqt", ".parq", ".csv", ".gz"}:
            sample = _read_sample(p, expected_cols, scan_rows)
            if sample is None or sample.empty:
                health_ok = False
                reason = reason or "health_sample_empty"
            else:
                missing_cols = [col for col in expected_cols if col not in sample.columns]
                if missing_cols:
                    info["missing_cols"] = missing_cols
                    health_ok = False
                    reason = reason or "missing_columns"
                missing_fracs = {}
                for col in expected_cols:
                    if col not in sample.columns:
                        missing_fracs[col] = 1.0
                    else:
                        missing_fracs[col] = float(sample[col].isna().mean())
                info["missing_frac_max"] = max(missing_fracs.values()) if missing_fracs else None
                missing_vals = [v for k, v in missing_fracs.items() if k not in missing_ok]
                if missing_vals and max(missing_vals) > max_missing:
                    health_ok = False
                    reason = reason or "missing_frac"
                if missing_vals and all(val >= 1.0 for val in missing_vals):
                    health_ok = False
                    reason = reason or "all_nan"
        outputs_info.append(info)
    return health_ok, outputs_info, reason


def _record_stage_manifest(
    section: str,
    mode: str,
    status: str,
    fingerprint: str | None,
    outputs_info: List[Dict[str, Any]],
    cache_path: str | None,
    health_ok: bool | None,
) -> None:
    if RUN_MANIFEST_PATH is None:
        return
    payload = _load_json(RUN_MANIFEST_PATH)
    stages = payload.get("stages", {})
    key = f"{section}.{mode}"
    stages[key] = {
        "section": section,
        "mode": mode,
        "status": status,
        "input_fingerprint": fingerprint,
        "outputs": outputs_info,
        "cache_path": cache_path,
        "health_ok": health_ok,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["run_name"] = RUN_NAME
    payload["config_sha256"] = CONFIG_SHA256
    payload["git_commit"] = GIT_COMMIT
    payload["pipeline_code_sha256"] = PIPELINE_CODE_SHA256
    if LOG_PATH:
        payload["log_path"] = str(LOG_PATH)
    payload["stages"] = stages
    _write_json(RUN_MANIFEST_PATH, payload)


def _cache_restore(
    section: str,
    mode: str,
    fingerprint: str,
    expected_cols: List[str],
) -> tuple[bool, List[Dict[str, Any]], str | None, str | None]:
    index = _load_cache_index(section, mode)
    entry = index.get(fingerprint)
    if not entry or entry.get("bad"):
        return False, [], None, None
    cache_dir = Path(entry.get("cache_dir", ""))
    if not cache_dir.exists():
        entry["bad"] = True
        entry["bad_reason"] = "cache_dir_missing"
        index[fingerprint] = entry
        _save_cache_index(section, mode, index)
        return False, [], None, None
    manifest_path = cache_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    outputs = manifest.get("outputs", [])
    cache_paths = [Path(o.get("cache_path", "")) for o in outputs if o.get("cache_path")]
    health_ok, outputs_info, reason = _health_check_outputs(
        cache_paths,
        expected_cols,
        float(CACHE_CFG.get("health_max_missing", 0.05)),
        int(CACHE_CFG.get("health_scan_rows", 200_000)),
    )
    if not health_ok:
        entry["bad"] = True
        entry["bad_reason"] = reason or "health_failed"
        index[fingerprint] = entry
        _save_cache_index(section, mode, index)
        return False, outputs_info, str(cache_dir), entry.get("bad_reason")

    restore_overwrite = bool(CACHE_CFG.get("restore_overwrite", False))
    for out in outputs:
        src = Path(out.get("cache_path", ""))
        dst = Path(out.get("path", ""))
        if not src.exists():
            continue
        if dst.exists() and not restore_overwrite:
            continue
        if dst.exists() and restore_overwrite:
            backup = dst.with_suffix(dst.suffix + ".bak")
            try:
                shutil.move(str(dst), str(backup))
            except Exception:
                pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return True, outputs_info, str(cache_dir), None


def _cache_store(
    section: str,
    mode: str,
    fingerprint: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Path],
    expected_cols: List[str],
) -> tuple[bool, List[Dict[str, Any]], str | None]:
    health_ok, outputs_info, reason = _health_check_outputs(
        outputs,
        expected_cols,
        float(CACHE_CFG.get("health_max_missing", 0.05)),
        int(CACHE_CFG.get("health_scan_rows", 200_000)),
    )
    if not health_ok:
        index = _load_cache_index(section, mode)
        entry = index.get(fingerprint, {})
        entry["bad"] = True
        entry["bad_reason"] = reason or "health_failed"
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        index[fingerprint] = entry
        _save_cache_index(section, mode, index)
        return False, outputs_info, None

    stage_dir = _cache_stage_dir(section, mode)
    stage_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = stage_dir / fingerprint
    if cache_dir.exists():
        alt = stage_dir / f"{fingerprint}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        cache_dir = alt
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_entries: List[Dict[str, Any]] = []
    for p in outputs:
        if not p.exists() or not p.is_file():
            continue
        rel = str(p).replace(":", "")
        dst = cache_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        output_entries.append({"path": str(p), "cache_path": str(dst)})

    manifest = {
        "section": section,
        "mode": mode,
        "input_fingerprint": fingerprint,
        "inputs": inputs,
        "outputs": output_entries,
        "health_ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(cache_dir / "manifest.json", manifest)

    index = _load_cache_index(section, mode)
    index[fingerprint] = {
        "cache_dir": str(cache_dir),
        "health_ok": True,
        "bad": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache_index(section, mode, index)
    return True, outputs_info, str(cache_dir)


def _cache_precheck(
    section: str,
    mode: str,
    step: Dict[str, Any],
    expected_cols: List[str],
) -> Dict[str, Any] | None:
    if not _cache_enabled(section, mode, step):
        return None
    fingerprint, inputs = _step_fingerprint(section, mode, step)
    outputs = _cacheable_outputs(step)
    if not outputs:
        return {"enabled": False, "fingerprint": fingerprint, "inputs": inputs, "outputs": outputs}
    hit, outputs_info, cache_path, _ = _cache_restore(section, mode, fingerprint, expected_cols)
    if hit:
        _record_stage_manifest(
            section,
            mode,
            "cache_hit",
            fingerprint,
            outputs_info,
            cache_path,
            True,
        )
        return {"hit": True, "fingerprint": fingerprint}
    return {"hit": False, "fingerprint": fingerprint, "inputs": inputs, "outputs": outputs}


def _outputs_health(step: Dict[str, Any], expected_cols: List[str]) -> tuple[bool | None, List[Dict[str, Any]]]:
    outputs = _cacheable_outputs(step) or _output_paths(step)
    if not outputs:
        return None, []
    # Agent: allow per-step health thresholds to avoid false rebuilds.
    max_missing = step.get("health_max_missing", step.get("health-max-missing"))
    scan_rows = step.get("health_scan_rows", step.get("health-scan-rows"))
    missing_ok = step.get("health_missing_ok_cols", step.get("health-missing-ok-cols", []))
    if isinstance(missing_ok, str):
        missing_ok = [c.strip() for c in missing_ok.split(",") if c.strip()]
    elif missing_ok is None:
        missing_ok = []
    else:
        missing_ok = [str(c).strip() for c in missing_ok if str(c).strip()]
    if max_missing in (None, ""):
        max_missing = CACHE_CFG.get("health_max_missing", 0.05)
    try:
        max_missing = float(max_missing)
    except Exception:
        max_missing = float(CACHE_CFG.get("health_max_missing", 0.05))
    if scan_rows in (None, ""):
        scan_rows = CACHE_CFG.get("health_scan_rows", 200_000)
    try:
        scan_rows = int(scan_rows)
    except Exception:
        scan_rows = int(CACHE_CFG.get("health_scan_rows", 200_000))
    health_ok, outputs_info, _ = _health_check_outputs(
        outputs,
        expected_cols,
        max_missing,
        scan_rows,
        missing_ok,
    )
    return health_ok, outputs_info

def _read_columns(path: Path) -> List[str] | None:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()
    try:
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                return list(pq.ParquetFile(path).schema.names)
            except Exception:
                import pandas as pd  # type: ignore

                return list(pd.read_parquet(path, columns=None).columns)
        if ext in {".csv", ".csv.gz"}:
            import pandas as pd  # type: ignore

            return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[describe] failed reading columns for {path}: {exc}")
    return None


def _describe_path(tag: str, role: str, path: Path) -> None:
    pp = path if path.is_absolute() else Path.cwd() / path
    if not pp.exists():
        print(f"[{tag} {role}] {path} (missing)")
        return
    if pp.is_dir():
        count = Counter(child.is_dir() for child in pp.iterdir())
        files = sum(1 for child in pp.iterdir() if child.is_file())
        print(f"[{tag} {role}] {path} (dir: {files} files, {count[True]} dirs)")
        return
    cols = _read_columns(pp)
    if cols:
        print(f"[{tag} {role}] {path} columns={cols}")
    else:
        print(f"[{tag} {role}] {path} (exists; column listing unavailable)")


def _describe_outputs(tag: str, step: Dict[str, Any]) -> None:
    paths = _output_paths(step)
    for p in paths:
        _describe_path(tag, "output", p)


def _describe_inputs(tag: str, step: Dict[str, Any]) -> None:
    files, globs = _input_paths(step)

    for g in globs:
        matches = sorted(Path().glob(g))
        if not matches:
            print(f"[{tag} input] glob={g} (no matches)")
            continue
        print(f"[{tag} input] glob={g} -> {len(matches)} matches (showing up to 3)")
        for mp in matches[:3]:
            _describe_path(tag, "input", mp)

    for f in files:
        _describe_path(tag, "input", f)


def _normalize_steps(section: str, steps: List[Dict[str, Any]], convert_existing: bool = True) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for step in steps:
        if step is None:
            continue
        step = _apply_runtime_hints(step)
        step = _apply_table_format(step, convert_existing=convert_existing)
        step = _maybe_force_keep_quantile(step, section, step.get("mode", section))
        normalized.append(step)
    return normalized


def _print_plan(section: str, steps: List[Dict[str, Any]]) -> None:
    if not steps:
        return
    print(f"\n[plan] {section} order:")
    for step in steps:
        mode = str(step.get("mode", "(unknown)"))
        pref = preflight_step(section, step)
        msg = summarize_step(section, step, pref)
        print(f"  - {msg}")
        if pref.errors:
            for err in pref.errors:
                print(f"      ! {err}")

def _dry_run_section(section: str, steps: List[Dict[str, Any]]) -> None:
    if not steps:
        return
    normalized = _normalize_steps(section, steps, convert_existing=False)
    has_modes = all(isinstance(s, dict) and ("mode" in s) for s in normalized)
    ordered = order_steps(section, normalized) if (normalized and has_modes) else normalized
    _print_plan(section, ordered)
    errors = 0
    for step in ordered:
        pref = preflight_step(section, step)
        errors += len(pref.errors)
    if errors:
        raise SystemExit(1)


def validate_pipeline(cfg: Dict[str, Any], sections: List[str]) -> None:
    """
    Dependency-aware preflight: allow missing inputs if produced by enabled upstream steps,
    otherwise fail fast with a clear error that points to the disabled producer.
    """
    check_sections = [s for s in sections if s in {"features", "data_stage", "training", "alerts_logic", "eval", "seeds", "report"}]
    steps_by_section: Dict[str, List[Dict[str, Any]]] = {}
    enabled_ids: set[str] = set()
    step_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    errors: List[str] = []

    def _enabled_steps(section: str) -> List[Dict[str, Any]]:
        sec = cfg.get(section, {}) if isinstance(cfg.get(section), dict) else {}
        if not sec.get("enabled"):
            return []
        if section == "seeds":
            raw = _seed_step_dicts(sec, convert_existing=False)
        elif section == "report":
            raw = sec.get("steps") or []
            if not raw and sec.get("enabled"):
                legacy = {k: v for k, v in sec.items() if k not in ("enabled", "steps")}
                legacy.setdefault("mode", "summary")
                raw = [legacy]
        else:
            raw = sec.get("steps") or []
        normalized = _normalize_steps(section, [s for s in raw if s is not None], convert_existing=False)
        has_modes = all(isinstance(s, dict) and ("mode" in s) for s in normalized)
        try:
            ordered = order_steps(section, normalized) if (normalized and has_modes) else normalized
        except SystemExit as exc:
            errors.append(f"[validate] {section}: dependency error: {exc}")
            ordered = normalized
        return [s for s in ordered if isinstance(s, dict)]

    for section in check_sections:
        steps = _enabled_steps(section)
        steps_by_section[section] = steps
        for step in steps:
            if step.get("enabled") is False:
                continue
            mode = str(step.get("mode", section)).strip()
            enabled_ids.add(f"{section}.{mode}")
            step_lookup[(section, mode)] = step

    def _producer_will_run(produced: Iterable[str]) -> bool:
        for prod in produced:
            if "." not in prod:
                continue
            sec, mode = prod.split(".", 1)
            cfg_step = step_lookup.get((sec, mode))
            if not cfg_step or cfg_step.get("enabled") is False:
                continue
            if cfg_step.get("overwrite") is True:
                return True
            if not cfg_step.get("skip_if_exists"):
                return True
            if cfg.get("auto_overwrite_on_invalid", False):
                return True
        return False

    for section, steps in steps_by_section.items():
        for step in steps:
            if step.get("enabled") is False:
                continue
            mode = str(step.get("mode", section)).strip()
            label = f"{section}.{mode}"
            pref = preflight_step(section, step)
            for issue in pref.input_issues:
                produced = list(issue.spec.produced_by or ())
                upstream_enabled = any(p in enabled_ids for p in produced)
                handled = False
                if issue.reason in {"missing_file", "missing_glob", "empty_file", "empty_glob"} and upstream_enabled:
                    # Produced by an enabled upstream step; defer validation until after it runs.
                    handled = True
                if issue.reason == "missing_columns" and upstream_enabled and _producer_will_run(produced):
                    handled = True
                if handled:
                    continue
                if produced:
                    disabled = [p for p in produced if p not in enabled_ids]
                    if disabled:
                        errors.append(
                            f"[validate] {label} needs {issue.used_key or issue.path or issue.spec.path_keys}; "
                            f"produced by {', '.join(disabled)} (currently disabled)."
                        )
                        handled = True
                if issue.reason == "missing_columns" and not handled:
                    errors.append(
                        f"[validate] {label} missing required columns in {issue.path}; "
                        f"expected from {', '.join(produced) if produced else 'upstream output'}."
                    )
                    handled = True
                if not handled:
                    errors.append(
                        f"[validate] {label} input issue ({issue.reason}): {issue.used_key or issue.path or issue.spec.path_keys}."
                    )
            if not pref.input_issues and pref.errors:
                for err in pref.errors:
                    if err not in errors:
                        errors.append(f"[validate] {err}")

    if errors:
        print("\n[pipeline-validate] blocking issues detected:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)

def _section_steps_for_fix(cfg: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    sec = cfg.get(section, {}) if isinstance(cfg.get(section), dict) else {}
    if section == "seeds":
        steps: List[Dict[str, Any]] = []
        mapping = [
            ("from-alerts", "from_alerts"),
            ("proto-outcomes", "outcomes"),
            ("gse-tracks", "gse_tracks"),
            ("starts-vs-tracks", "starts"),
            ("analyze", "analyze"),
        ]
        for mode, key in mapping:
            cfg_step = sec.get(key)
            if isinstance(cfg_step, dict):
                steps.append({"mode": mode, **cfg_step})
        return steps
    steps = sec.get("steps") or []
    if section == "report" and sec.get("enabled") and not steps:
        legacy = {k: v for k, v in sec.items() if k not in ("enabled", "steps")}
        legacy.setdefault("mode", "summary")
        steps = [legacy]
    return [s for s in steps if isinstance(s, dict)]


def _find_step_ref(cfg: Dict[str, Any], section: str, mode: str) -> Dict[str, Any] | None:
    sec = cfg.get(section, {}) if isinstance(cfg.get(section), dict) else {}
    if section == "seeds":
        mapping = {
            "from-alerts": "from_alerts",
            "proto-outcomes": "outcomes",
            "gse-tracks": "gse_tracks",
            "starts-vs-tracks": "starts",
            "analyze": "analyze",
        }
        key = mapping.get(mode)
        if key and isinstance(sec.get(key), dict):
            return sec.get(key)
        return None
    steps = sec.get("steps") or []
    for step in steps:
        if isinstance(step, dict) and str(step.get("mode", "")).strip() == mode:
            return step
    return None


def _ensure_section_enabled(
    cfg: Dict[str, Any],
    section: str,
    changes: List[str],
    reason: str,
) -> None:
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        return
    if sec.get("enabled") is True:
        return
    sec["enabled"] = True
    changes.append(f"enabled {section} ({reason})")


def _enable_step(
    cfg: Dict[str, Any],
    section: str,
    mode: str,
    changes: List[str],
    reason: str,
) -> bool:
    target = _find_step_ref(cfg, section, mode)
    if target is None:
        return False
    if target.get("enabled") is False:
        target["enabled"] = True
        changes.append(f"enabled {section}.{mode} ({reason})")
    _ensure_section_enabled(cfg, section, changes, f"required by {section}.{mode}")
    return True


def _parse_autofix_rebuild(raw: Any) -> List[Tuple[str, str]]:
    if raw is None:
        return []
    items: List[str] = []
    if isinstance(raw, (list, tuple, set)):
        for entry in raw:
            if entry is None:
                continue
            if isinstance(entry, dict):
                sec = str(entry.get("section", "")).strip()
                mode = str(entry.get("mode", "")).strip()
                if sec and mode:
                    items.append(f"{sec}.{mode}")
                continue
            items.append(str(entry))
    else:
        items.append(str(raw))

    out: List[Tuple[str, str]] = []
    for token in items:
        if not token:
            continue
        for part in str(token).replace(";", ",").split(","):
            tok = part.strip()
            if not tok:
                continue
            sep = None
            for cand in (".", ":", "/"):
                if cand in tok:
                    sep = cand
                    break
            if not sep:
                print(f"[autofix] warning: rebuild target '{tok}' missing section separator.", file=sys.stderr)
                continue
            sec, mode = [t.strip() for t in tok.split(sep, 1)]
            if not sec or not mode:
                print(f"[autofix] warning: rebuild target '{tok}' invalid.", file=sys.stderr)
                continue
            out.append((sec, mode))
    return out


def _parse_autofix_scope(raw: Any) -> str:
    if not raw:
        return "self"
    val = str(raw).strip().lower()
    if val in {"self", "upstream", "downstream", "both"}:
        return val
    print(f"[autofix] warning: rebuild_scope '{raw}' invalid; using 'self'.", file=sys.stderr)
    return "self"


def _mark_overwrite(
    cfg: Dict[str, Any],
    section: str,
    mode: str,
    reason: str,
    changes: List[str],
    add_overwrite: bool,
    stages_manifest: Dict[str, Any] | None = None,
    ref_mtime: float | None = None,
    guard_freshness: bool = False,
    allow_skip_toggle: bool = True,
    force_enable: bool = False,
) -> bool:
    target = _find_step_ref(cfg, section, mode)
    if target is None:
        return False
    if target.get("enabled") is False:
        if not force_enable:
            return False
        target["enabled"] = True
        changes.append(f"enabled {section}.{mode} ({reason})")
    _ensure_section_enabled(cfg, section, changes, f"required by {section}.{mode}")
    # Reporting manager does not support an --overwrite CLI flag.
    # For report steps, force re-run via skip_if_exists=false instead.
    if section == "report":
        changed = False
        if target.get("overwrite") is True:
            target["overwrite"] = False
            changes.append(f"overwrite {section}.{mode}=false ({reason}; report steps use skip_if_exists)")
            changed = True
        if allow_skip_toggle and (target.get("skip_if_exists") is True or "skip_if_exists" not in target):
            target["skip_if_exists"] = False
            changes.append(f"skip_if_exists {section}.{mode}=false ({reason})")
            changed = True
        return changed
    has_overwrite = "overwrite" in target
    if target.get("skip_if_exists") is False and not has_overwrite and not add_overwrite:
        return False
    if guard_freshness and ref_mtime is not None:
        outputs = _autofix_step_outputs(section, mode, target, stages_manifest or {})
        out_mtime = _outputs_max_mtime(outputs)
        if out_mtime is not None and out_mtime >= ref_mtime:
            print(
                f"[autofix] skip overwrite {section}.{mode}: outputs newer than "
                f"{_format_ts(ref_mtime)} (outputs {_format_ts(out_mtime)})."
            )
            return False
    if has_overwrite or add_overwrite:
        if target.get("overwrite") is True:
            return False
        target["overwrite"] = True
        changes.append(f"overwrite {section}.{mode}=true ({reason})")
        return True
    if allow_skip_toggle:
        if target.get("skip_if_exists") is True or "skip_if_exists" not in target:
            target["skip_if_exists"] = False
            changes.append(f"skip_if_exists {section}.{mode}=false ({reason})")
            return True
    print(
        f"[autofix] {section}.{mode}: {reason} but no overwrite key; "
        "use --autofix-add-overwrite or set overwrite=true in config."
    )
    return False


def _autofix_config(
    cfg: Dict[str, Any],
    sections: List[str],
    out_path: Path,
    add_overwrite: bool = False,
    manual_rebuild: List[Tuple[str, str]] | None = None,
    rebuild_scope: str | None = None,
    use_diagnostics_rules: bool | None = None,
    rebuild_on_change: bool | None = None,
) -> None:
    # Agent: tighten autofix rebuild rules and allow explicit manual targets without extra overwrite args.
    changes: List[str] = []
    run_name = cfg.get("run_name") or RUN_NAME
    autofix_cfg = cfg.get("autofix", {}) if isinstance(cfg.get("autofix"), dict) else {}
    if not add_overwrite and bool(autofix_cfg.get("add_overwrite", False)):
        add_overwrite = True
    cfg_targets = _parse_autofix_rebuild(autofix_cfg.get("rebuild"))
    manual_targets = list(cfg_targets)
    if manual_rebuild:
        manual_targets.extend(manual_rebuild)
    manual_scope = _parse_autofix_scope(rebuild_scope or autofix_cfg.get("rebuild_scope"))
    if use_diagnostics_rules is None:
        use_diagnostics_rules = bool(autofix_cfg.get("use_diagnostics_rules", False))
    if rebuild_on_change is None:
        rebuild_on_change = bool(autofix_cfg.get("rebuild_on_change", False))
    log_priority = bool(autofix_cfg.get("log_priority_over_diagnostics", True))
    only_rebuild_failed = bool(autofix_cfg.get("only_rebuild_failed", False))
    clear_overwrite_on_success = bool(autofix_cfg.get("clear_overwrite_on_success", False)) or only_rebuild_failed

    stages_manifest, stages_meta = _load_stage_manifest_for_autofix(run_name)
    manifest = _load_run_manifest(run_name)
    log_path = _latest_log_path(run_name, manifest)
    log_mtime = log_path.stat().st_mtime if log_path and log_path.exists() else None
    code_mtime = _pipeline_code_mtime(HERE)
    config_mtime = None
    if CONFIG_PATH and CONFIG_PATH.exists():
        try:
            config_mtime = CONFIG_PATH.stat().st_mtime
        except Exception:
            config_mtime = None
    code_or_config_mtime = None
    for ts in (code_mtime, config_mtime):
        if ts is None:
            continue
        code_or_config_mtime = ts if code_or_config_mtime is None else max(code_or_config_mtime, ts)
    code_changed = bool(stages_meta.get("pipeline_code_sha256")) and bool(PIPELINE_CODE_SHA256) and (
        stages_meta.get("pipeline_code_sha256") != PIPELINE_CODE_SHA256
    )
    config_changed = bool(stages_meta.get("config_sha256")) and bool(CONFIG_SHA256) and (
        stages_meta.get("config_sha256") != CONFIG_SHA256
    )
    if code_changed or config_changed:
        if rebuild_on_change:
            print(
                f"[autofix] upstream change detected: "
                f"code_changed={code_changed} config_changed={config_changed}"
            )
        else:
            print("[autofix] upstream change detected; rebuild_on_change=false (ignored)")
            code_changed = False
            config_changed = False
    diag_checks, diag_path, diag_mtime = _load_diagnostics_checks(run_name)
    log_marked: set[Tuple[str, str]] = set()
    diag_marked: set[Tuple[str, str]] = set()
    manual_marked: set[Tuple[str, str]] = set()
    stale_marked: set[Tuple[str, str]] = set()
    if log_priority:
        log_marked = _apply_last_run_reasons(
            cfg, run_name, stages_manifest, sections, changes, add_overwrite, log_mtime
        )
    if use_diagnostics_rules and diag_checks:
        diag_fails = [c for c in diag_checks if str(c.get("status")).lower() == "fail"]
        if diag_fails and diag_path:
            print(f"[autofix] diagnostics source: {diag_path}")
        skip_diag = set()
        if log_priority and log_mtime and diag_mtime and log_mtime >= diag_mtime:
            skip_diag = set(log_marked)
            if skip_diag:
                print("[autofix] log is newer than diagnostics; skipping diag overrides for log-flagged steps.")
        for chk in diag_fails:
            issue = str(chk.get("issue") or "").strip()
            if not issue:
                continue
            for section, mode in _diagnostics_overwrite_targets(issue):
                if (section, mode) in skip_diag:
                    continue
                diag_marked.add((section, mode))
                _mark_overwrite(
                    cfg,
                    section,
                    mode,
                    f"diagnostics: {issue}",
                    changes,
                    add_overwrite,
                    stages_manifest,
                    diag_mtime,
                    False,
                )
            if issue.lower() == "flow direction missing in matches":
                bundle = _find_step_ref(cfg, "report", "bundle")
                if isinstance(bundle, dict):
                    preferred_flow = "data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet"
                    current_flow = str(bundle.get("objects_flow_source") or "").strip()
                    if Path(preferred_flow).exists() and current_flow != preferred_flow:
                        bundle["objects_flow_source"] = preferred_flow
                        changes.append("set report.bundle.objects_flow_source=data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet (diagnostics: flow direction missing in matches)")
                    cells_out = str(bundle.get("cells_out") or "").strip()
                    if not cells_out:
                        bundle["cells_out"] = "results/objects/object_cells_by_hour.parquet"
                        changes.append("set report.bundle.cells_out=results/objects/object_cells_by_hour.parquet (diagnostics: flow direction missing in matches)")

    if manual_targets:
        uniq_targets: List[Tuple[str, str]] = []
        seen_targets: set[Tuple[str, str]] = set()
        for target in manual_targets:
            if target in seen_targets:
                continue
            seen_targets.add(target)
            uniq_targets.append(target)
        for section, mode in uniq_targets:
            if _find_step_ref(cfg, section, mode) is None:
                print(f"[autofix] warning: manual rebuild target not found: {section}.{mode}")
                continue
            _enable_step(cfg, section, mode, changes, "manual rebuild")
            manual_marked.add((section, mode))
            _mark_overwrite(
                cfg,
                section,
                mode,
                "manual_rebuild",
                changes,
                add_overwrite,
                stages_manifest,
                None,
                False,
                True,
                True,
            )
            if manual_scope in {"upstream", "both"}:
                for dep in _expand_upstream_deps(section, mode):
                    _enable_step(cfg, section, dep, changes, f"manual rebuild upstream of {section}.{mode}")
                    manual_marked.add((section, dep))
                    _mark_overwrite(
                        cfg,
                        section,
                        dep,
                        f"manual_rebuild upstream of {section}.{mode}",
                        changes,
                        add_overwrite,
                        stages_manifest,
                        None,
                        False,
                        True,
                        True,
                    )
            if manual_scope in {"downstream", "both"}:
                for dep_section, dep_mode in _expand_downstream_deps(cfg, sections, section, mode):
                    _enable_step(cfg, dep_section, dep_mode, changes, f"manual rebuild downstream of {section}.{mode}")
                    manual_marked.add((dep_section, dep_mode))
                    _mark_overwrite(
                        cfg,
                        dep_section,
                        dep_mode,
                        f"manual_rebuild downstream of {section}.{mode}",
                        changes,
                        add_overwrite,
                        stages_manifest,
                        None,
                        False,
                        True,
                        True,
                    )

    if not log_priority:
        _apply_last_run_reasons(cfg, run_name, stages_manifest, sections, changes, add_overwrite, log_mtime)
    pre_add_ids_stale = False
    pre_add_ids_stale_due_to_issue = False
    pre_add_sections = {"fetch", "features"}

    for section in sections:
        steps = _section_steps_for_fix(cfg, section)
        if not steps:
            continue

        # Enable in-section dependencies for enabled steps before ordering.
        active_modes = {
            str(s.get("mode", "")).strip()
            for s in steps
            if isinstance(s, dict) and s.get("enabled") is not False and str(s.get("mode", "")).strip()
        }
        changed = True
        while changed:
            changed = False
            for step in steps:
                if not isinstance(step, dict):
                    continue
                mode = str(step.get("mode", "")).strip()
                if not mode or mode not in active_modes:
                    continue
                contract = contract_for(section, mode)
                if not contract or not contract.dependencies:
                    continue
                for dep in contract.dependencies:
                    if dep in active_modes:
                        continue
                    target = _find_step_ref(cfg, section, dep)
                    if target is None:
                        continue
                    active_modes.add(dep)
                    if target.get("enabled") is False:
                        target["enabled"] = True
                        cfg.setdefault(section, {})["enabled"] = True
                        changes.append(f"enabled {section}.{dep} (dependency for {section}.{mode})")
                    changed = True

        try:
            ordered = order_steps(section, steps)
        except SystemExit:
            ordered = steps

        for step in ordered:
            mode = str(step.get("mode", "")).strip()
            if step.get("enabled") is False:
                continue
            step_for_preflight = dict(step)
            step_for_preflight = _apply_runtime_hints(step_for_preflight)
            step_for_preflight = _apply_table_format(step_for_preflight, convert_existing=False)
            pref = preflight_step(section, step_for_preflight)
            health_failed = False
            outputs = _output_paths(step_for_preflight)
            if outputs and any(p.exists() for p in outputs):
                health_ok, _ = _outputs_health(step_for_preflight, pref.expected_output_columns or [])
                if health_ok is False:
                    health_failed = True
            stale_reason = _autofix_stale_reason(section, mode, step_for_preflight, stages_manifest)
            if stale_reason == "previous_health_failed" and not health_failed:
                stale_reason = None
            if stale_reason is None and step_for_preflight.get("skip_if_exists"):
                outputs = _output_paths(step_for_preflight)
                if outputs and any(p.exists() for p in outputs):
                    if code_changed and config_changed:
                        stale_reason = "pipeline_code_or_config_changed"
                    elif code_changed:
                        stale_reason = "pipeline_code_changed"
                    elif config_changed:
                        stale_reason = "config_changed"
            if health_failed:
                stale_reason = "output_health_failed"
            if (section in pre_add_sections or (section == "data_stage" and mode == "add-ids")) and (pref.errors or pref.input_issues):
                pre_add_ids_stale = True
                pre_add_ids_stale_due_to_issue = True
            if stale_reason:
                stale_ref = None
                if stale_reason in {"pipeline_code_changed", "config_changed", "pipeline_code_or_config_changed"}:
                    stale_ref = code_or_config_mtime
                guard_freshness = stale_reason != "output_health_failed"
                mark_add_overwrite = add_overwrite
                stale_marked.add((section, mode))
                _mark_overwrite(
                    cfg,
                    section,
                    mode,
                    f"stale: {stale_reason}",
                    changes,
                    mark_add_overwrite,
                    stages_manifest,
                    stale_ref,
                    guard_freshness,
                )
                if section in pre_add_sections or (section == "data_stage" and mode == "add-ids"):
                    pre_add_ids_stale = True
                    if stale_reason == "output_health_failed":
                        pre_add_ids_stale_due_to_issue = True
            if (code_changed or config_changed) and (section in pre_add_sections or (section == "data_stage" and mode == "add-ids")):
                pre_add_ids_stale = True
            for issue in pref.input_issues:
                missing_cols = getattr(issue, "missing_cols", None) or []
                force_overwrite = issue.reason in {"missing_columns", "empty_file", "empty_glob"}
                for prod in issue.spec.produced_by or ():
                    if "." not in prod:
                        continue
                    prod_section, prod_mode = prod.split(".", 1)
                    target = _find_step_ref(cfg, prod_section, prod_mode)
                    if target is None:
                        continue
                    if target.get("enabled") is False:
                        target["enabled"] = True
                        changes.append(f"enabled {prod_section}.{prod_mode}")
                    _ensure_section_enabled(cfg, prod_section, changes, f"dependency for {section}.{mode}")
                    if force_overwrite:
                        _mark_overwrite(
                            cfg,
                            prod_section,
                            prod_mode,
                            f"upstream of {section}.{mode} ({issue.reason})",
                            changes,
                            add_overwrite,
                            stages_manifest,
                        )
                        for dep in _expand_upstream_deps(prod_section, prod_mode):
                            _mark_overwrite(
                                cfg,
                                prod_section,
                                dep,
                                f"upstream of {prod_section}.{prod_mode} ({issue.reason})",
                                changes,
                                add_overwrite,
                                stages_manifest,
                            )
    if pre_add_ids_stale:
        guard_freshness = not pre_add_ids_stale_due_to_issue
        stale_marked.add(("features", "join-labels-grid"))
        _mark_overwrite(
            cfg,
            "features",
            "join-labels-grid",
            "upstream stale before add-ids",
            changes,
            add_overwrite,
            stages_manifest,
            code_or_config_mtime,
            guard_freshness,
        )
        stale_marked.add(("data_stage", "add-ids"))
        _mark_overwrite(
            cfg,
            "data_stage",
            "add-ids",
            "upstream stale before add-ids",
            changes,
            add_overwrite,
            stages_manifest,
            code_or_config_mtime,
            guard_freshness,
        )

    if clear_overwrite_on_success:
        keep_overwrite = set().union(log_marked, diag_marked, manual_marked, stale_marked)
        for section in sections:
            steps = _section_steps_for_fix(cfg, section)
            if not steps:
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                mode = str(step.get("mode", "")).strip()
                if not mode:
                    continue
                if (section, mode) in keep_overwrite:
                    continue
                target = _find_step_ref(cfg, section, mode)
                if not target:
                    continue
                if target.get("overwrite") is True:
                    # Only clear overwrite for steps that are not flagged and not stale.
                    step_for_preflight = dict(step)
                    step_for_preflight = _apply_runtime_hints(step_for_preflight)
                    step_for_preflight = _apply_table_format(step_for_preflight, convert_existing=False)
                    stale_reason = _autofix_stale_reason(section, mode, step_for_preflight, stages_manifest)
                    if stale_reason:
                        continue
                    target["overwrite"] = False
                    if "skip_if_exists" in target and target.get("skip_if_exists") is not True:
                        target["skip_if_exists"] = True
                    changes.append(f"overwrite {section}.{mode}=false (healthy)")

    if not changes:
        print("[autofix] no changes applied.")
        return

    if hasattr(yaml, "safe_dump"):
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        import json
        out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[autofix] wrote {out_path} with {len(changes)} change(s):")
    for line in changes:
        print(f"[autofix] {line}")

def _seed_step_dicts(sec: Dict[str, Any], convert_existing: bool = True) -> List[Dict[str, Any]]:
    raw_steps = [
        ("from-alerts", sec.get("from_alerts")),
        ("proto-outcomes", sec.get("outcomes")),
        ("gse-tracks", sec.get("gse_tracks")),
        ("starts-vs-tracks", sec.get("starts")),
        ("analyze", sec.get("analyze")),
    ]
    steps: List[Dict[str, Any]] = []
    for name, cfg in raw_steps:
        if cfg is None:
            continue
        cfg = _apply_table_format(cfg, convert_existing=convert_existing)
        cfg = _maybe_force_keep_quantile(cfg, "seeds", name)
        steps.append({"mode": name, **cfg})
    return steps

# ---------------- section runners ----------------

def run_fetch(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["fetch_subprocess", "fetch_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_runtime_hints(step)
        step = _apply_table_format(step)
        step = _maybe_force_keep_quantile(step, "fetch", step.get("mode", "ibtracs"))
        mode = str(step.get("mode", "ibtracs"))
        _progress("fetch", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[fetch] skip (disabled): {mode}")
            continue
        expected_cols: List[str] = []
        cache_ctx = _cache_precheck("fetch", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("fetch", step)
            continue
        if _should_skip_if_exists(step):
            health_ok, outputs_info = _outputs_health(step, expected_cols)
            if health_ok:
                print(f"[fetch] skip (exists, health ok): {mode}")
                _describe_outputs("fetch", step)
                cache_path = None
                if cache_ctx and cache_ctx.get("outputs"):
                    stored, cache_outputs_info, cache_path = _cache_store(
                        "fetch",
                        mode,
                        cache_ctx["fingerprint"],
                        cache_ctx.get("inputs", []),
                        cache_ctx["outputs"],
                        expected_cols,
                    )
                    if stored:
                        outputs_info = cache_outputs_info or outputs_info
                _record_stage_manifest(
                    "fetch",
                    mode,
                    "reuse_existing",
                    cache_ctx["fingerprint"] if cache_ctx else None,
                    outputs_info,
                    cache_path,
                    health_ok,
                )
                continue
            if health_ok is False:
                print(f"[fetch] existing output failed validation; re-running step.", file=sys.stderr)
                step = _handle_invalid_existing_output("fetch", mode, step, SystemExit("health_check_failed"))

        _describe_inputs("fetch", step)
        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists"))
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("fetch", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "fetch",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "fetch",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_features(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["features_subprocess", "features_manager.py"])
    raw_steps = [s for s in sec.get("steps", []) if s is not None]
    normalized = _normalize_steps("features", raw_steps)
    ordered = order_steps("features", normalized)
    _print_plan("features", ordered)

    total = len(ordered)
    for idx, step in enumerate(ordered, 1):
        mode = str(step.get("mode", "build"))
        _progress("features", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[features] skip (disabled): {mode}")
            continue

        contract = contract_for("features", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []

        cache_ctx = _cache_precheck("features", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("features", step)
            continue

        # Skip-if-exists still requires schema validation so downstream checks are meaningful.
        if _should_skip_if_exists(step):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                try:
                    postflight_step("features", step, expected_cols)
                    print(f"[features] skip (exists, schema ok): {mode} -> {outp}")
                    _describe_outputs("features", step)
                    health_ok, outputs_info = _outputs_health(step, expected_cols)
                    cache_path = None
                    if cache_ctx and cache_ctx.get("outputs"):
                        stored, cache_outputs_info, cache_path = _cache_store(
                            "features",
                            mode,
                            cache_ctx["fingerprint"],
                            cache_ctx.get("inputs", []),
                            cache_ctx["outputs"],
                            expected_cols,
                        )
                        if stored:
                            outputs_info = cache_outputs_info or outputs_info
                    _record_stage_manifest(
                        "features",
                        mode,
                        "reuse_existing",
                        cache_ctx["fingerprint"] if cache_ctx else None,
                        outputs_info,
                        cache_path,
                        health_ok,
                    )
                    continue
                except SystemExit as exc:
                    print(f"[features] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                    step = _handle_invalid_existing_output("features", mode, step, exc)

        pref = preflight_step("features", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[features] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)

        _describe_inputs("features", step)
        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists"))
        sh([sys.executable, str(mgr), mode, *args])
        postflight_step("features", step, pref.expected_output_columns or expected_cols)
        _describe_outputs("features", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "features",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "features",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_data_stage(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["data_subprocess", "data_stage_manager.py"])
    raw_steps = [s for s in sec.get("steps", []) if s is not None]
    normalized = _normalize_steps("data_stage", raw_steps)
    ordered = order_steps("data_stage", normalized)
    _print_plan("data_stage", ordered)

    total = len(ordered)
    for idx, step in enumerate(ordered, 1):
        mode = str(step.get("mode", "stage"))
        _progress("data_stage", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[data_stage] skip (disabled): {mode}")
            continue

        contract = contract_for("data_stage", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []

        cache_ctx = _cache_precheck("data_stage", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("data_stage", step)
            continue

        if _should_skip_if_exists(step):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                try:
                    postflight_step("data_stage", step, expected_cols)
                    print(f"[data_stage] skip (exists, schema ok): {mode} -> {outp}")
                    _describe_outputs("data_stage", step)
                    health_ok, outputs_info = _outputs_health(step, expected_cols)
                    cache_path = None
                    if cache_ctx and cache_ctx.get("outputs"):
                        stored, cache_outputs_info, cache_path = _cache_store(
                            "data_stage",
                            mode,
                            cache_ctx["fingerprint"],
                            cache_ctx.get("inputs", []),
                            cache_ctx["outputs"],
                            expected_cols,
                        )
                        if stored:
                            outputs_info = cache_outputs_info or outputs_info
                    _record_stage_manifest(
                        "data_stage",
                        mode,
                        "reuse_existing",
                        cache_ctx["fingerprint"] if cache_ctx else None,
                        outputs_info,
                        cache_path,
                        health_ok,
                    )
                    continue
                except SystemExit as exc:
                    print(f"[data_stage] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                    step = _handle_invalid_existing_output("data_stage", mode, step, exc)

        pref = preflight_step("data_stage", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[data_stage] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)

        _describe_inputs("data_stage", step)
        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists"))
        sh([sys.executable, str(mgr), mode, *args])
        postflight_step("data_stage", step, pref.expected_output_columns)
        _describe_outputs("data_stage", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "data_stage",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "data_stage",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_training(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["data_subprocess", "training_manager.py"])
    raw_steps = [s for s in sec.get("steps", []) if s is not None]
    normalized = _normalize_steps("training", raw_steps)
    ordered = order_steps("training", normalized)
    _print_plan("training", ordered)

    total = len(ordered)
    for idx, step in enumerate(ordered, 1):
        mode = str(step.get("mode", "train-base"))
        _progress("training", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[training] skip (disabled): {mode}")
            continue

        contract = contract_for("training", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []

        cache_ctx = _cache_precheck("training", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("training", step)
            continue

        if _should_skip_if_exists(step):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                try:
                    postflight_step("training", step, expected_cols)
                    print(f"[training] skip (exists, schema ok): {mode} -> {outp}")
                    _describe_outputs("training", step)
                    health_ok, outputs_info = _outputs_health(step, expected_cols)
                    cache_path = None
                    if cache_ctx and cache_ctx.get("outputs"):
                        stored, cache_outputs_info, cache_path = _cache_store(
                            "training",
                            mode,
                            cache_ctx["fingerprint"],
                            cache_ctx.get("inputs", []),
                            cache_ctx["outputs"],
                            expected_cols,
                        )
                        if stored:
                            outputs_info = cache_outputs_info or outputs_info
                    _record_stage_manifest(
                        "training",
                        mode,
                        "reuse_existing",
                        cache_ctx["fingerprint"] if cache_ctx else None,
                        outputs_info,
                        cache_path,
                        health_ok,
                    )
                    continue
                except SystemExit as exc:
                    print(f"[training] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                    step = _handle_invalid_existing_output("training", mode, step, exc)

        pref = preflight_step("training", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[training] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)

        _describe_inputs("training", step)
        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists"))
        sh([sys.executable, str(mgr), mode, *args])
        postflight_step("training", step, pref.expected_output_columns or expected_cols)
        _describe_outputs("training", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "training",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "training",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_sweep(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["sweep_subprocess", "sweep_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_runtime_hints(step)
        step = _apply_table_format(step)
        step = _maybe_force_keep_quantile(step, "sweep", step.get("mode", "run"))
        mode = str(step.get("mode", "run"))
        _progress("sweep", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[sweep] skip (disabled): {mode}")
            continue
        expected_cols: List[str] = []
        cache_ctx = _cache_precheck("sweep", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("sweep", step)
            continue
        if _should_skip_if_exists(step):
            health_ok, outputs_info = _outputs_health(step, expected_cols)
            if health_ok:
                print(f"[sweep] skip (exists, health ok): {mode}")
                _describe_outputs("sweep", step)
                cache_path = None
                if cache_ctx and cache_ctx.get("outputs"):
                    stored, cache_outputs_info, cache_path = _cache_store(
                        "sweep",
                        mode,
                        cache_ctx["fingerprint"],
                        cache_ctx.get("inputs", []),
                        cache_ctx["outputs"],
                        expected_cols,
                    )
                    if stored:
                        outputs_info = cache_outputs_info or outputs_info
                _record_stage_manifest(
                    "sweep",
                    mode,
                    "reuse_existing",
                    cache_ctx["fingerprint"] if cache_ctx else None,
                    outputs_info,
                    cache_path,
                    health_ok,
                )
                continue
            if health_ok is False:
                print(f"[sweep] existing output failed validation; re-running step.", file=sys.stderr)
                step = _handle_invalid_existing_output("sweep", mode, step, SystemExit("health_check_failed"))

        _describe_inputs("sweep", step)
        if mode == "chain":
            recipe = step.get("recipe", "run+pick")
            args = _step_cli_args(step, ("mode", "recipe", "enabled"))
            sh([sys.executable, str(mgr), "chain", recipe, *args])
        else:
            args = _step_cli_args(step, ("mode", "enabled"))
            sh([sys.executable, str(mgr), "tool", mode, *args])
        _describe_outputs("sweep", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "sweep",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "sweep",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_score(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    scorer = _mgr(["grid_score.py"])
    jobs = [j for j in sec.get("jobs", []) if j is not None]
    total = len(jobs)
    for idx, job in enumerate(jobs, 1):
        job = _apply_runtime_hints(job)
        job = _apply_table_format(job)
        job = _maybe_force_keep_quantile(job, "score", job.get("mode", "score"))
        _progress("score", idx, total, job.get("mode", "score"))
        if job.get("enabled") is False:
            print(f"[score] skip (disabled): {job.get('mode')}")
            continue
        expected_cols: List[str] = []
        mode = str(job.get("mode", "score"))
        cache_ctx = _cache_precheck("score", mode, job, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("score", job)
            continue
        if _should_skip_if_exists(job):
            health_ok, outputs_info = _outputs_health(job, expected_cols)
            if health_ok:
                print(f"[score] skip (exists, health ok): {mode}")
                _describe_outputs("score", job)
                cache_path = None
                if cache_ctx and cache_ctx.get("outputs"):
                    stored, cache_outputs_info, cache_path = _cache_store(
                        "score",
                        mode,
                        cache_ctx["fingerprint"],
                        cache_ctx.get("inputs", []),
                        cache_ctx["outputs"],
                        expected_cols,
                    )
                    if stored:
                        outputs_info = cache_outputs_info or outputs_info
                _record_stage_manifest(
                    "score",
                    mode,
                    "reuse_existing",
                    cache_ctx["fingerprint"] if cache_ctx else None,
                    outputs_info,
                    cache_path,
                    health_ok,
                )
                continue
            if health_ok is False:
                print(f"[score] existing output failed validation; re-running step.", file=sys.stderr)
                job = _handle_invalid_existing_output("score", mode, job, SystemExit("health_check_failed"))

        _describe_inputs("score", job)
        args = _step_cli_args(job, ("enabled",))
        sh([sys.executable, str(scorer), *args])
        _describe_outputs("score", job)
        health_ok, outputs_info = _outputs_health(job, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "score",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "score",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_alerts_logic(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["alerts_logic_subprocess", "alerts_logic_manager.py"])
    raw_steps = [s for s in sec.get("steps", []) if s is not None]
    normalized = _normalize_steps("alerts_logic", raw_steps)
    ordered = order_steps("alerts_logic", normalized)
    _print_plan("alerts_logic", ordered)
    total = len(ordered)
    for idx, step in enumerate(ordered, 1):
        mode = str(step.get("mode", "denoise"))
        _progress("alerts_logic", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[alerts_logic] skip (disabled): {mode}")
            continue
        _describe_inputs("alerts_logic", step)

        contract = contract_for("alerts_logic", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []
        cache_ctx = _cache_precheck("alerts_logic", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("alerts_logic", step)
            continue

        if _should_skip_if_exists(step):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                try:
                    postflight_step("alerts_logic", step, expected_cols)
                    print(f"[alerts_logic] skip (exists, schema ok): {mode} -> {outp}")
                    _describe_outputs("alerts_logic", step)
                    health_ok, outputs_info = _outputs_health(step, expected_cols)
                    cache_path = None
                    if cache_ctx and cache_ctx.get("outputs"):
                        stored, cache_outputs_info, cache_path = _cache_store(
                            "alerts_logic",
                            mode,
                            cache_ctx["fingerprint"],
                            cache_ctx.get("inputs", []),
                            cache_ctx["outputs"],
                            expected_cols,
                        )
                        if stored:
                            outputs_info = cache_outputs_info or outputs_info
                    _record_stage_manifest(
                        "alerts_logic",
                        mode,
                        "reuse_existing",
                        cache_ctx["fingerprint"] if cache_ctx else None,
                        outputs_info,
                        cache_path,
                        health_ok,
                    )
                    continue
                except SystemExit as exc:
                    print(f"[alerts_logic] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                    step = _handle_invalid_existing_output("alerts_logic", mode, step, exc)

        pref = preflight_step("alerts_logic", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[alerts_logic] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)

        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists"))
        sh([sys.executable, str(mgr), mode, *args])
        postflight_step("alerts_logic", step, pref.expected_output_columns)
        _describe_outputs("alerts_logic", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "alerts_logic",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "alerts_logic",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_eval(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["eval_subprocess", "eval_manager.py"])
    raw_steps = [s for s in sec.get("steps", []) if s is not None]
    normalized = _normalize_steps("eval", raw_steps)
    ordered = order_steps("eval", normalized)
    _print_plan("eval", ordered)
    total = len(ordered)
    for idx, step in enumerate(ordered, 1):
        mode = str(step.get("mode", "hourly-rollup"))
        _progress("eval", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[eval] skip (disabled): {mode}")
            continue
        contract = contract_for("eval", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []

        cache_ctx = _cache_precheck("eval", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("eval", step)
            continue

        _describe_inputs("eval", step)
        pref = preflight_step("eval", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[eval] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)
        args = _step_cli_args(step, ("mode", "enabled"))
        sh([sys.executable, str(mgr), mode, *args])
        postflight_step("eval", step, pref.expected_output_columns)
        _describe_outputs("eval", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "eval",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "eval",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_seeds(sec: Dict[str, Any]) -> None:
    """
    Seeds/track tooling via the seeds_subprocess manager (seeds_tracks.py).
    Section shape (YAML):

      seeds:
        enabled: true
        from_alerts:
          enabled: true
          ... (CLI args)
        outcomes:
          enabled: true
          ...
        starts:
          enabled: true
          ...
        analyze:
          enabled: true
          ...
    """
    if not sec.get("enabled"):
        return
    tool = _mgr(["seeds_subprocess", "seeds_tracks.py"])

    steps = _seed_step_dicts(sec)
    total = len(steps)

    for idx, step in enumerate(steps, 1):
        name = str(step.get("mode", ""))
        cfg = dict(step)
        cfg.pop("mode", None)
        _progress("seeds", idx, total, name)
        if cfg.get("enabled", True) is False:
            print(f"[seeds] skip (disabled): {name}")
            continue
        contract = contract_for("seeds", name)
        expected_cols = contract.expected_output_columns(step) if contract else []
        cache_ctx = _cache_precheck("seeds", name, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("seeds", cfg)
            continue
        if _should_skip_if_exists(step):
            try:
                postflight_step("seeds", step, [])
                print(f"[seeds] skip (exists, schema ok): {name}")
                _describe_outputs("seeds", cfg)
                health_ok, outputs_info = _outputs_health(cfg, expected_cols)
                cache_path = None
                if cache_ctx and cache_ctx.get("outputs"):
                    stored, cache_outputs_info, cache_path = _cache_store(
                        "seeds",
                        name,
                        cache_ctx["fingerprint"],
                        cache_ctx.get("inputs", []),
                        cache_ctx["outputs"],
                        expected_cols,
                    )
                    if stored:
                        outputs_info = cache_outputs_info or outputs_info
                _record_stage_manifest(
                    "seeds",
                    name,
                    "reuse_existing",
                    cache_ctx["fingerprint"] if cache_ctx else None,
                    outputs_info,
                    cache_path,
                    health_ok,
                )
                continue
            except SystemExit as exc:
                print(f"[seeds] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                cfg = _handle_invalid_existing_output("seeds", name, cfg, exc)

        _describe_inputs("seeds", cfg)
        pref = preflight_step("seeds", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[seeds] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)
        args = _step_cli_args(cfg, ("enabled",))
        sh([sys.executable, str(tool), name, *args])
        postflight_step("seeds", step, pref.expected_output_columns)
        _describe_outputs("seeds", cfg)
        health_ok, outputs_info = _outputs_health(cfg, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "seeds",
                name,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "seeds",
            name,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

def run_report(sec: Dict[str, Any]) -> None:
    """
    Reports & maps via reports_subprocess/reports_and_maps_manager.py.

    New-style section:

      report:
        enabled: true
        steps:
          - mode: summary
            run_name: my_run
            ...
          - mode: sanity
            require_alerts: true
          - mode: maps
            union_csv: ...
            patches_csv: ...

    Back-compat:
      If no `steps` key is present, we treat `report` as a single
      `summary` step with the remaining keys as CLI args.
    """
    if not sec.get("enabled"):
        return
    mgr = _mgr(["reports_subprocess", "reports_and_maps_manager.py"])

    steps = sec.get("steps")
    if not steps:
        # Back-compat: treat whole block as a single summary step
        legacy_step = {k: v for k, v in sec.items() if k not in ("enabled", "steps")}
        legacy_step.setdefault("mode", "summary")
        steps = [legacy_step]

    raw_steps = [s for s in steps if s is not None]
    normalized = _normalize_steps("report", raw_steps)
    ordered = order_steps("report", normalized) if normalized else []
    _print_plan("report", ordered)

    for idx, step in enumerate(ordered, 1):
        if step is None:
            continue
        mode = str(step.get("mode", "summary"))
        _progress("report", idx, len(ordered), mode)
        if step.get("enabled") is False:
            print(f"[report] skip (disabled): {mode}")
            continue
        contract = contract_for("report", mode)
        expected_cols = contract.expected_output_columns(step) if contract else []
        cache_ctx = _cache_precheck("report", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("report", step)
            continue

        _describe_inputs("report", step)

        if _should_skip_if_exists(step):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                try:
                    postflight_step("report", step, expected_cols)
                    print(f"[report] skip (exists, schema ok): {mode} -> {outp}")
                    _describe_outputs("report", step)
                    health_ok, outputs_info = _outputs_health(step, expected_cols)
                    cache_path = None
                    if cache_ctx and cache_ctx.get("outputs"):
                        stored, cache_outputs_info, cache_path = _cache_store(
                            "report",
                            mode,
                            cache_ctx["fingerprint"],
                            cache_ctx.get("inputs", []),
                            cache_ctx["outputs"],
                            expected_cols,
                        )
                        if stored:
                            outputs_info = cache_outputs_info or outputs_info
                    _record_stage_manifest(
                        "report",
                        mode,
                        "reuse_existing",
                        cache_ctx["fingerprint"] if cache_ctx else None,
                        outputs_info,
                        cache_path,
                        health_ok,
                    )
                    continue
                except SystemExit as exc:
                    print(f"[report] existing output failed validation ({exc}); re-running step.", file=sys.stderr)
                    step = _handle_invalid_existing_output("report", mode, step, exc)

        pref = preflight_step("report", step)
        if pref.errors:
            for err in pref.errors:
                print(f"[report] preflight error: {err}", file=sys.stderr)
            raise SystemExit(1)

        args = _step_cli_args(step, ("mode", "enabled", "skip_if_exists", "overwrite"))
        if mode in {"bundle", "full", "all"}:
            sh([sys.executable, str(mgr), *args])
        else:
            sh([sys.executable, str(mgr), mode, *args])
        postflight_step("report", step, pref.expected_output_columns)
        _describe_outputs("report", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "report",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "report",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )


def run_misc(sec: Dict[str, Any]) -> None:
    """
    Optional runner for standalone scripts that are not part of a *_subprocess
    manager. Each step must provide a `script` path (relative to repo root or
    absolute) plus any CLI flags as key/value pairs.

    Example YAML:

      misc:
        enabled: true
        steps:
          - script: one_off_processes/predict_raw_scores.py
            enabled: true
            model: models/grid_logit_perlead.pkl
            features: data/grid_labelled_FMA_gka_realthermo.csv.gz
            out: results/predictions.csv.gz
            skip_if_exists: true
    """
    if not sec.get("enabled"):
        return

    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)

    for idx, step in enumerate(steps, 1):
        script = step.get("script")
        step = _apply_table_format(step)
        _progress("misc", idx, total, script or "(missing script)")
        if step.get("enabled") is False:
            print(f"[misc] skip (disabled): {script}")
            continue

        expected_cols: List[str] = []
        mode = str(script or "misc")
        cache_ctx = _cache_precheck("misc", mode, step, expected_cols)
        if cache_ctx and cache_ctx.get("hit"):
            _describe_outputs("misc", step)
            continue

        if _should_skip_if_exists(step):
            health_ok, outputs_info = _outputs_health(step, expected_cols)
            if health_ok:
                print(f"[misc] skip (exists, health ok): {script}")
                _describe_outputs("misc", step)
                cache_path = None
                if cache_ctx and cache_ctx.get("outputs"):
                    stored, cache_outputs_info, cache_path = _cache_store(
                        "misc",
                        mode,
                        cache_ctx["fingerprint"],
                        cache_ctx.get("inputs", []),
                        cache_ctx["outputs"],
                        expected_cols,
                    )
                    if stored:
                        outputs_info = cache_outputs_info or outputs_info
                _record_stage_manifest(
                    "misc",
                    mode,
                    "reuse_existing",
                    cache_ctx["fingerprint"] if cache_ctx else None,
                    outputs_info,
                    cache_path,
                    health_ok,
                )
                continue
            if health_ok is False:
                print(f"[misc] existing output failed validation; re-running step.", file=sys.stderr)
                step = _handle_invalid_existing_output("misc", mode, step, SystemExit("health_check_failed"))

        if not script:
            print("[misc] skip: missing 'script' path")
            continue

        path = _resolve_script(script)
        step = _maybe_force_keep_quantile(step, "misc", script)
        args = _step_cli_args(step, ("script", "enabled", "skip_if_exists"))
        sh([sys.executable, str(path), *args])
        _describe_outputs("misc", step)
        health_ok, outputs_info = _outputs_health(step, expected_cols)
        cache_path = None
        status = "executed"
        if cache_ctx and cache_ctx.get("outputs"):
            stored, cache_outputs_info, cache_path = _cache_store(
                "misc",
                mode,
                cache_ctx["fingerprint"],
                cache_ctx.get("inputs", []),
                cache_ctx["outputs"],
                expected_cols,
            )
            if stored:
                outputs_info = cache_outputs_info or outputs_info
                status = "rebuilt"
        _record_stage_manifest(
            "misc",
            mode,
            status,
            cache_ctx["fingerprint"] if cache_ctx else None,
            outputs_info,
            cache_path,
            health_ok,
        )

# ---------------- main ----------------

SECTION_ORDER = [
    "fetch",
    "features",
    "data_stage",
    "training",
    "sweep",
    "score",
    "alerts_logic",
    "eval",
    # runtime removed
    "seeds",
    "report",
    "misc",
]

def main() -> int:
    ap = argparse.ArgumentParser(description="Thin pipeline orchestrator")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (default: results/runs/<run_name>/logs/<run_name>_<UTC>.log). Use 'none' to disable.",
    )
    ap.add_argument("--no-lock", action="store_true", help="Disable the run lock guard.")
    ap.add_argument("--force-lock", action="store_true", help="Override an existing run lock.")
    ap.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Allow running with system Python even if .venv exists.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preflight only; do not execute steps.")
    ap.add_argument(
        "--autofix-config",
        nargs="?",
        const="auto",
        default=None,
        help="Write an updated config with enabled dependencies/overwrite fixes and exit. "
             "If no path is provided, writes <config>.autofix.yaml.",
    )
    ap.add_argument(
        "--autofix-add-overwrite",
        action="store_true",
        help="Allow autofix to add overwrite=true even if the step did not define it.",
    )
    ap.add_argument(
        "--autofix-rebuild",
        action="append",
        default=None,
        help="Manual rebuild targets for autofix (section.mode; repeatable or comma-separated).",
    )
    ap.add_argument(
        "--autofix-rebuild-scope",
        choices=["self", "upstream", "downstream", "both"],
        default=None,
        help="Scope for --autofix-rebuild targets (default: self).",
    )
    ap.add_argument(
        "--autofix-use-diagnostics",
        action="store_true",
        default=None,
        help="Allow diagnostics.json failures to trigger autofix rebuilds.",
    )
    ap.add_argument(
        "--autofix-rebuild-on-change",
        action="store_true",
        default=None,
        help="Allow code/config changes to trigger autofix rebuilds.",
    )
    ap.add_argument(
        "--autofix-refresh",
        action="store_true",
        help="Update the current config in place using autofix before running "
             "(enables dependencies/overwrite fixes). Defaults to true when using *.autofix.yaml.",
    )
    ap.add_argument(
        "--sections",
        default=",".join(SECTION_ORDER),
        help=f"Comma list to limit which sections run, in order. Default: {','.join(SECTION_ORDER)}"
    )
    ns = ap.parse_args()

    cfg_path = Path(ns.config).resolve()
    global CONFIG_PATH
    CONFIG_PATH = cfg_path
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}
    _init_logging(cfg.get("run_name"), ns.log_file)
    _ensure_loky_cpu_count()
    print(f"[python] exe={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix}")
    venv_bin = "Scripts" if os.name == "nt" else "bin"
    venv_exe = "python.exe" if os.name == "nt" else "python"
    venv_py = HERE / ".venv" / venv_bin / venv_exe
    if venv_py.exists():
        try:
            if venv_py.resolve() != Path(sys.executable).resolve():
                msg = f"[python] warning: .venv detected but active interpreter is {sys.executable}"
                print(msg)
                if not ns.allow_system_python:
                    raise SystemExit(
                        "[python] refusing to run with system interpreter while .venv exists "
                        "(use --allow-system-python to override)."
                    )
        except Exception:
            pass
    cfg, cfg_changes = config_normalize.normalize_config(cfg)
    if cfg_changes:
        print(f"[config] normalized {len(cfg_changes)} entries:")
        for line in cfg_changes:
            print(f"[config] {line}")
        if hasattr(yaml, "safe_dump"):
            try:
                canon_path = _canonical_config_path(cfg_path)
                canon_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                print(f"[config] canonical config -> {canon_path}")
            except Exception as exc:
                print(f"[config] warning: failed to write canonical config: {exc}")

    global PREFERRED_TABLE_FORMAT
    PREFERRED_TABLE_FORMAT = table_format.normalize_preference(cfg.get("table_format"))
    if PREFERRED_TABLE_FORMAT:
        print(f"[table-format] preference -> {PREFERRED_TABLE_FORMAT}")

    global CONFIG_SHA256
    CONFIG_SHA256 = _config_sha256(cfg_path)

    global GIT_COMMIT
    GIT_COMMIT = _git_commit(HERE)

    global PIPELINE_CODE_SHA256
    PIPELINE_CODE_SHA256 = _pipeline_code_sha256(HERE)
    if PIPELINE_CODE_SHA256:
        print(f"[pipeline] code sha256={PIPELINE_CODE_SHA256[:12]}...")

    global RUN_NAME
    RUN_NAME = cfg.get("run_name")
    if RUN_NAME:
        safe_run = _safe_run_name(RUN_NAME)
        global RUN_MANIFEST_PATH
        RUN_MANIFEST_PATH = Path("results/runs") / safe_run / "manifests" / "stages.json"
    if not ns.no_lock:
        # Agent: avoid duplicate runs and OOM by locking per run_name.
        _acquire_run_lock(RUN_NAME, force=bool(ns.force_lock))

    global CACHE_CFG
    cache_cfg = cfg.get("cache", {})
    if isinstance(cache_cfg, bool):
        cache_cfg = {"enabled": cache_cfg}
    if not isinstance(cache_cfg, dict):
        cache_cfg = {}
    cache_sections = set(cache_cfg.get("sections", []) or [])
    cache_stages = set(cache_cfg.get("stages", []) or [])
    CACHE_CFG = {
        "enabled": bool(cache_cfg.get("enabled", False)),
        "dir": cache_cfg.get("dir", "cache"),
        "sections": cache_sections,
        "stages": cache_stages,
        "health_max_missing": cache_cfg.get("health_max_missing", 0.05),
        "health_scan_rows": cache_cfg.get("health_scan_rows", 200_000),
        "restore_overwrite": cache_cfg.get("restore_overwrite", False),
    }

    wanted = [s.strip() for s in ns.sections.split(",") if s.strip()]
    order = [s for s in SECTION_ORDER if s in wanted]
    manual_rebuild = _parse_autofix_rebuild(ns.autofix_rebuild)

    if ns.autofix_config is not None:
        out_path = ns.autofix_config
        if out_path == "auto":
            out_path = str(_autofix_config_path(cfg_path))
        _autofix_config(
            cfg,
            order,
            Path(out_path),
            add_overwrite=bool(ns.autofix_add_overwrite),
            manual_rebuild=manual_rebuild,
            rebuild_scope=ns.autofix_rebuild_scope,
            use_diagnostics_rules=ns.autofix_use_diagnostics,
            rebuild_on_change=ns.autofix_rebuild_on_change,
        )
        return 0

    if ns.autofix_refresh or _is_autofix_path(cfg_path):
        # Agent: refresh autofix configs before validation to keep dependencies enabled.
        _autofix_config(
            cfg,
            order,
            cfg_path,
            add_overwrite=bool(ns.autofix_add_overwrite),
            manual_rebuild=manual_rebuild,
            rebuild_scope=ns.autofix_rebuild_scope,
            use_diagnostics_rules=ns.autofix_use_diagnostics,
            rebuild_on_change=ns.autofix_rebuild_on_change,
        )
        CONFIG_SHA256 = _config_sha256(cfg_path)

    global FORCE_KEEP_QUANTILE
    fq = cfg.get("force_keep_quantile")
    if fq is None and cfg.get("force_keep_all"):
        fq = 1.0
    if isinstance(fq, bool):
        FORCE_KEEP_QUANTILE = 1.0 if fq else None
    elif fq is not None:
        try:
            FORCE_KEEP_QUANTILE = min(1.0, max(0.0, float(fq)))
        except Exception:
            FORCE_KEEP_QUANTILE = None
    if FORCE_KEEP_QUANTILE is not None:
        print(f"[keep-quantile] force_keep_quantile -> {FORCE_KEEP_QUANTILE}")

    global AUTO_OVERWRITE_ON_INVALID
    auto_over = cfg.get("auto_overwrite_on_invalid", False)
    AUTO_OVERWRITE_ON_INVALID = bool(auto_over)

    global ENV_HINTS
    ENV_HINTS = env_check.summarize()
    if ENV_HINTS:
        avail = ENV_HINTS.get("available_gb")
        avail_txt = f"{float(avail):.2f}" if isinstance(avail, (int, float)) else "n/a"
        print(f"[env] available_gb={avail_txt} "
              f"csv_rows={ENV_HINTS.get('csv_rows')} parquet_rows={ENV_HINTS.get('parquet_rows')} "
              f"cpu_count={ENV_HINTS.get('cpu_count')}")

    workdir = cfg.get("workdir")
    if workdir:
        wd = Path(workdir).resolve()
        print(f"[cwd] -> {wd}")
        wd.mkdir(parents=True, exist_ok=True)
        os.chdir(wd)

    if ns.dry_run:
        if "fetch" in order:
            _dry_run_section("fetch", cfg.get("fetch", {}).get("steps", []) or [])
        if "features" in order:
            _dry_run_section("features", cfg.get("features", {}).get("steps", []) or [])
        if "data_stage" in order:
            _dry_run_section("data_stage", cfg.get("data_stage", {}).get("steps", []) or [])
        if "training" in order:
            _dry_run_section("training", cfg.get("training", {}).get("steps", []) or [])
        if "sweep" in order:
            _dry_run_section("sweep", cfg.get("sweep", {}).get("steps", []) or [])
        if "score" in order:
            _dry_run_section("score", cfg.get("score", {}).get("jobs", []) or [])
        if "alerts_logic" in order:
            _dry_run_section("alerts_logic", cfg.get("alerts_logic", {}).get("steps", []) or [])
        if "eval" in order:
            _dry_run_section("eval", cfg.get("eval", {}).get("steps", []) or [])
        if "seeds" in order:
            _dry_run_section("seeds", _seed_step_dicts(cfg.get("seeds", {}), convert_existing=False))
        if "report" in order:
            steps = cfg.get("report", {}).get("steps") or []
            if not steps and cfg.get("report", {}).get("enabled"):
                legacy = {k: v for k, v in cfg.get("report", {}).items() if k not in ("enabled", "steps")}
                legacy.setdefault("mode", "summary")
                steps = [legacy]
            _dry_run_section("report", steps)
        if "misc" in order:
            _dry_run_section("misc", cfg.get("misc", {}).get("steps", []) or [])
        print("\n[orchestrator] Dry run complete (no commands executed).")
        return 0

    validate_pipeline(cfg, order)

    if "fetch" in order:        run_fetch(cfg.get("fetch", {}))
    if "features" in order:     run_features(cfg.get("features", {}))
    if "data_stage" in order:   run_data_stage(cfg.get("data_stage", {}))
    if "training" in order:     run_training(cfg.get("training", {}))
    if "sweep" in order:        run_sweep(cfg.get("sweep", {}))
    if "score" in order:        run_score(cfg.get("score", {}))
    if "alerts_logic" in order: run_alerts_logic(cfg.get("alerts_logic", {}))
    if "eval" in order:         run_eval(cfg.get("eval", {}))
    if "seeds" in order:        run_seeds(cfg.get("seeds", {}))
    if "report" in order:       run_report(cfg.get("report", {}))
    if "misc" in order:         run_misc(cfg.get("misc", {}))

    print("\n[orchestrator] Complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
