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

try:
    import yaml
except Exception:
    print("Please `pip install pyyaml`.", file=sys.stderr)
    raise

HERE = Path(__file__).resolve().parent

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

def _mgr(path_parts: Iterable[str]) -> Path:
    return HERE.joinpath(*path_parts).resolve()

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

# ---------------- section runners ----------------

def run_fetch(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["fetch_subprocess", "fetch_manager.py"])
    for step in sec.get("steps", []):
        if step is None or step.get("enabled") is False:
            continue
        mode = str(step.get("mode", "ibtracs"))
        args = _flatten_kv("", {k: v for k, v in step.items()
                                if k not in ("mode", "enabled", "skip_if_exists")})
        sh([sys.executable, str(mgr), mode, *args])

def run_features(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["features_subprocess", "features_manager.py"])
    for step in sec.get("steps", []):
        if step is None:
            continue
        if step.get("enabled") is False:
            print(f"[features] skip (disabled): {step.get('mode')}")
            continue
        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[features] skip (exists): {step.get('mode')} → {outp}")
                continue
        mode = str(step.get("mode", "build"))
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])

def run_data_stage(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["data_subprocess", "data_stage_manager.py"])
    for step in sec.get("steps", []):
        if step is None or step.get("enabled") is False:
            continue
        mode = str(step.get("mode", "stage"))
        args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
        sh([sys.executable, str(mgr), mode, *args])

def run_sweep(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["sweep_subprocess", "sweep_manager.py"])
    for step in sec.get("steps", []):
        if step is None or step.get("enabled") is False:
            continue
        mode = str(step.get("mode", "run"))
        if mode == "chain":
            recipe = step.get("recipe", "run+pick")
            extra = {k: v for k, v in step.items() if k not in ("mode", "recipe","enabled")}
            args = _flatten_kv("", extra)
            sh([sys.executable, str(mgr), "chain", recipe, *args])
        else:
            args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
            sh([sys.executable, str(mgr), "tool", mode, *args])

def run_score(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    scorer = _mgr(["grid_score.py"])
    for job in sec.get("jobs", []):
        if job is None or job.get("enabled") is False:
            continue
        args = _flatten_kv("", {k:v for k,v in job.items() if k!="enabled"})
        sh([sys.executable, str(scorer), *args])

def run_alerts_logic(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["alerts_logic_subprocess", "alerts_logic_manager.py"])
    for step in sec.get("steps", []):
        if step is None:
            continue
        if step.get("enabled") is False:
            print(f"[alerts_logic] skip (disabled): {step.get('mode')}")
            continue

        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[alerts_logic] skip (exists): {step.get('mode')} → {outp}")
                continue

        mode = str(step.get("mode", "denoise"))
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])

def run_eval(sec: Dict[str, Any]) -> None:
    if not sec.get("enabled"):
        return
    mgr = _mgr(["eval_subprocess", "eval_manager.py"])
    for step in sec.get("steps", []):
        if step is None or step.get("enabled") is False:
            continue
        mode = str(step.get("mode", "hourly-rollup"))
        args = _flatten_kv("", {k: v for k, v in step.items() if k not in ("mode","enabled")})
        sh([sys.executable, str(mgr), mode, *args])

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

    A = sec.get("from_alerts")
    if A and A.get("enabled", True):
        sh([sys.executable, str(tool), "from-alerts",
            *_flatten_kv("", {k:v for k,v in A.items() if k!="enabled"})])

    B = sec.get("outcomes")
    if B and B.get("enabled", True):
        sh([sys.executable, str(tool), "proto-outcomes",
            *_flatten_kv("", {k:v for k,v in B.items() if k!="enabled"})])

    C = sec.get("starts")
    if C and C.get("enabled", True):
        sh([sys.executable, str(tool), "starts-vs-tracks",
            *_flatten_kv("", {k:v for k,v in C.items() if k!="enabled"})])

    D = sec.get("analyze")
    if D and D.get("enabled", True):
        sh([sys.executable, str(tool), "analyze",
            *_flatten_kv("", {k:v for k,v in D.items() if k!="enabled"})])

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

    for step in steps:
        if step is None:
            continue
        if step.get("enabled") is False:
            print(f"[report] skip (disabled): {step.get('mode')}")
            continue

        if step.get("skip_if_exists"):
            outp = _candidate_out_path(step)
            if outp and outp.exists():
                print(f"[report] skip (exists): {step.get('mode')} → {outp}")
                continue

        mode = str(step.get("mode", "summary"))
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items()
             if k not in ("mode", "enabled", "skip_if_exists")}
        )
        sh([sys.executable, str(mgr), mode, *args])

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

    print("\n[orchestrator] Complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())