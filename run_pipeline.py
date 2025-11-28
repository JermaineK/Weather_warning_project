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
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
from collections import Counter

from utils import table_format

try:
    import yaml
except Exception:
    print("Please `pip install pyyaml`.", file=sys.stderr)
    raise

HERE = Path(__file__).resolve().parent
PREFERRED_TABLE_FORMAT: str | None = None

# ---------------- shell helpers ----------------

def sh(cmd: List[Union[str, Path]], check: bool = True) -> int:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode

def _flatten_kv(prefix: str, obj: Any) -> List[str]:
    """
    Turn a nested structure into CLI flags.
    - scalars:          {"thr": 0.8}                  → ["--thr", "0.8"]
    - lists (scalars):  {"leads": [24,48]}            → ["--leads", "24,48"]
    - dict nested:      {"pick": {"metric": "F1"}}    → ["--pick.metric", "F1"]
    - truthy booleans:  {"write_parquet": True}       → ["--write-parquet"]
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
    out += [f"--{prefix.replace('_','-')}", str(obj)]
    return out


def _apply_table_format(step: Dict[str, Any]) -> Dict[str, Any]:
    if not PREFERRED_TABLE_FORMAT:
        return step
    return table_format.rewrite_step_paths(step, PREFERRED_TABLE_FORMAT, convert_existing=True)

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


def _progress(tag: str, idx: int, total: int, label: str) -> None:
    total = max(total, 1)
    print(f"[{tag}] step {idx}/{total}: {label}")


def _output_paths(step: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for k, v in step.items():
        if not isinstance(v, str) or not v.strip():
            continue
        key = k.lower()
        if key == "outcomes":
            continue
        if key == "out" or key.startswith("out_") or key.startswith("outfile") or key.startswith("outcsv"):
            paths.append(Path(v))
    return paths


def _input_paths(step: Dict[str, Any]) -> tuple[list[Path], list[str]]:
    files: List[Path] = []
    globs: List[str] = []

    for k, v in step.items():
        if not isinstance(v, str) or not v.strip():
            continue
        key = k.lower()
        if key in {"mode", "enabled", "skip_if_exists", "recipe"}:
            continue
        if key.startswith("out") or key.startswith("outfile"):
            continue
        if "glob" in key:
            globs.append(v)
            continue

        p = Path(v)
        looks_like_path = p.suffix or "/" in v or "\\" in v
        if looks_like_path:
            files.append(p)

    return files, globs


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
        print(f"[{tag} input] glob={g} → {len(matches)} matches (showing up to 3)")
        for mp in matches[:3]:
            _describe_path(tag, "input", mp)

    for f in files:
        _describe_path(tag, "input", f)

# ---------------- section runners ----------------

def run_fetch(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["fetch_subprocess", "fetch_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "ibtracs"))
        _progress("fetch", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[fetch] skip (disabled): {mode}")
            continue
        _describe_inputs("fetch", step)
        args = _flatten_kv("", {k: v for k, v in step.items()
                                if k not in ("mode", "enabled", "skip_if_exists")})
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("fetch", step)

def run_features(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["features_subprocess", "features_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "build"))
        _progress("features", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[features] skip (disabled): {mode}")
            continue
        _describe_inputs("features", step)
        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[features] skip (exists): {mode} → {outp}")
                _describe_outputs("features", step)
                continue
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("features", step)

def run_data_stage(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["data_subprocess", "data_stage_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "stage"))
        _progress("data_stage", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[data_stage] skip (disabled): {mode}")
            continue
        _describe_inputs("data_stage", step)
        args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("data_stage", step)

def run_sweep(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["sweep_subprocess", "sweep_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "run"))
        _progress("sweep", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[sweep] skip (disabled): {mode}")
            continue
        _describe_inputs("sweep", step)
        if mode == "chain":
            recipe = step.get("recipe", "run+pick")
            extra = {k: v for k, v in step.items() if k not in ("mode", "recipe","enabled")}
            args = _flatten_kv("", extra)
            sh([sys.executable, str(mgr), "chain", recipe, *args])
        else:
            args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
            sh([sys.executable, str(mgr), "tool", mode, *args])
        _describe_outputs("sweep", step)

def run_score(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    scorer = _mgr(["grid_score.py"])
    jobs = [j for j in sec.get("jobs", []) if j is not None]
    total = len(jobs)
    for idx, job in enumerate(jobs, 1):
        job = _apply_table_format(job)
        _progress("score", idx, total, job.get("mode", "score"))
        if job.get("enabled") is False:
            print(f"[score] skip (disabled): {job.get('mode')}")
            continue
        _describe_inputs("score", job)
        args = _flatten_kv("", {k:v for k,v in job.items() if k!="enabled"})
        sh([sys.executable, str(scorer), *args])
        _describe_outputs("score", job)

def run_alerts_logic(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["alerts_logic_subprocess", "alerts_logic_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "denoise"))
        _progress("alerts_logic", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[alerts_logic] skip (disabled): {mode}")
            continue
        _describe_inputs("alerts_logic", step)

        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[alerts_logic] skip (exists): {mode} → {outp}")
                _describe_outputs("alerts_logic", step)
                continue

        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("alerts_logic", step)

def run_eval(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["eval_subprocess", "eval_manager.py"])
    steps = [s for s in sec.get("steps", []) if s is not None]
    total = len(steps)
    for idx, step in enumerate(steps, 1):
        step = _apply_table_format(step)
        mode = str(step.get("mode", "hourly-rollup"))
        _progress("eval", idx, total, mode)
        if step.get("enabled") is False:
            print(f"[eval] skip (disabled): {mode}")
            continue
        _describe_inputs("eval", step)
        args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("eval", step)

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

    raw_steps = [
        ("from-alerts", sec.get("from_alerts")),
        ("proto-outcomes", sec.get("outcomes")),
        ("starts-vs-tracks", sec.get("starts")),
        ("analyze", sec.get("analyze")),
    ]
    steps = [(name, cfg) for name, cfg in raw_steps if cfg is not None]
    total = len(steps)

    for idx, (name, cfg) in enumerate(steps, 1):
        cfg = _apply_table_format(cfg)
        _progress("seeds", idx, total, name)
        if cfg.get("enabled", True) is False:
            print(f"[seeds] skip (disabled): {name}")
            continue
        _describe_inputs("seeds", cfg)
        args = _flatten_kv("", {k:v for k,v in cfg.items() if k!="enabled"})
        sh([sys.executable, str(tool), name, *args])
        _describe_outputs("seeds", cfg)

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

    for idx, step in enumerate(steps, 1):
        if step is None:
            continue
        step = _apply_table_format(step)
        mode = str(step.get("mode", "summary"))
        _progress("report", idx, len(steps), mode)
        if step.get("enabled") is False:
            print(f"[report] skip (disabled): {mode}")
            continue
        _describe_inputs("report", step)

        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[report] skip (exists): {mode} → {outp}")
                _describe_outputs("report", step)
                continue

        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])
        _describe_outputs("report", step)


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

        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[misc] skip (exists): {script} → {outp}")
                _describe_outputs("misc", step)
                continue

        if not script:
            print("[misc] skip: missing 'script' path")
            continue

        path = _resolve_script(script)
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("script", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(path), *args])
        _describe_outputs("misc", step)

# ---------------- main ----------------

SECTION_ORDER = [
    "fetch",
    "features",
    "data_stage",
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
        "--sections",
        default=",".join(SECTION_ORDER),
        help=f"Comma list to limit which sections run, in order. Default: {','.join(SECTION_ORDER)}"
    )
    ns = ap.parse_args()

    cfg_path = Path(ns.config).resolve()
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text) or {}

    global PREFERRED_TABLE_FORMAT
    PREFERRED_TABLE_FORMAT = table_format.normalize_preference(cfg.get("table_format"))
    if PREFERRED_TABLE_FORMAT:
        print(f"[table-format] preference → {PREFERRED_TABLE_FORMAT}")

    workdir = cfg.get("workdir")
    if workdir:
        wd = Path(workdir).resolve()
        print(f"[cwd] → {wd}")
        wd.mkdir(parents=True, exist_ok=True)
        os.chdir(wd)

    wanted = [s.strip() for s in ns.sections.split(",") if s.strip()]
    order = [s for s in SECTION_ORDER if s in wanted]

    if "fetch" in order:        run_fetch(cfg.get("fetch", {}))
    if "features" in order:     run_features(cfg.get("features", {}))
    if "data_stage" in order:   run_data_stage(cfg.get("data_stage", {}))
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