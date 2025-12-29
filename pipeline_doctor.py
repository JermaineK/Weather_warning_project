#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_doctor.py

Preflight-only checker for the orchestrated pipeline. Validates step order,
required inputs, and expected schemas without running any heavy computation.
Usage:
    python pipeline_doctor.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils import table_format
from utils import env_check
from utils import config_normalize
from pipeline_contracts import order_steps, preflight_step, summarize_step

try:
    import yaml  # type: ignore
except Exception:
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
                print("Warning: failed to parse config as JSON; install 'pyyaml' for full YAML support (pip install pyyaml).", file=sys.stderr)
                return {}

    yaml = _YAMLShim()


# -------------------- helpers (local copy of run_pipeline shims) --------------------
ENV_HINTS: Dict[str, Any] | None = None
PREFERRED_TABLE_FORMAT: str | None = None
FORCE_KEEP_QUANTILE: float | None = None


def _apply_table_format(step: Dict[str, Any], convert_existing: bool = False) -> Dict[str, Any]:
    if not PREFERRED_TABLE_FORMAT:
        return step
    return table_format.rewrite_step_paths(step, PREFERRED_TABLE_FORMAT, convert_existing=convert_existing)


def _apply_runtime_hints(step: Dict[str, Any]) -> Dict[str, Any]:
    if ENV_HINTS is None:
        return step
    csv_rows = ENV_HINTS.get("csv_rows")
    parq_rows = ENV_HINTS.get("parquet_rows")

    def set_if_missing(key: str, val):
        if key not in step or step[key] in (None, "", 0):
            step[key] = val

    if csv_rows:
        set_if_missing("chunk_rows", csv_rows)
        set_if_missing("chunk-rows", csv_rows)
        set_if_missing("chunksize", csv_rows)
    if parq_rows:
        set_if_missing("parquet_rows", parq_rows)
        set_if_missing("parquet-rows", parq_rows)
    return step


def _maybe_force_keep_quantile(step: Dict[str, Any], section: str, mode: str | None = None) -> Dict[str, Any]:
    if FORCE_KEEP_QUANTILE is None or not isinstance(step, dict):
        return step
    out = dict(step)
    for k in list(step.keys()):
        norm = k.replace("-", "_")
        if norm == "keep_quantile" and step.get(k) is not None:
            before = step.get(k)
            out[k] = FORCE_KEEP_QUANTILE
            ctx = f"{section}.{mode}" if mode else section
            print(f"[keep-quantile] force_keep_quantile active -> overriding {ctx} {k} {before} -> {FORCE_KEEP_QUANTILE}")
    return out


def _normalize_steps(section: str, steps: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for step in steps:
        if step is None:
            continue
        step = _apply_runtime_hints(step)
        step = _apply_table_format(step, convert_existing=False)
        step = _maybe_force_keep_quantile(step, section, step.get("mode", section))
        normalized.append(step)
    return normalized


def _print_section(section: str, steps: List[Dict[str, Any]]) -> int:
    if not steps:
        return 0
    try:
        ordered = order_steps(section, steps)
    except SystemExit as exc:
        print(f"[pipeline-doctor] {section} dependency error: {exc}", file=sys.stderr)
        return 1
    print(f"\n[pipeline-doctor] {section} plan:")
    errors = 0
    for step in ordered:
        pref = preflight_step(section, step)
        msg = summarize_step(section, step, pref)
        print(f"  - {msg}")
        for err in pref.errors:
            errors += 1
            print(f"      ! {err}")
    return errors


def _seed_step_dicts(sec: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        cfg = _apply_table_format(cfg, convert_existing=False)
        cfg = _maybe_force_keep_quantile(cfg, "seeds", name)
        steps.append({"mode": name, **cfg})
    return steps


# -------------------- main --------------------
SECTION_ORDER = [
    "fetch",
    "features",
    "data_stage",
    "training",
    "sweep",
    "score",
    "alerts_logic",
    "eval",
    "seeds",
    "report",
    "misc",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline preflight checker (no execution).")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument(
        "--sections",
        default=",".join(SECTION_ORDER),
        help=f"Comma list to limit sections. Default: {','.join(SECTION_ORDER)}",
    )
    ns = ap.parse_args()

    cfg_path = Path(ns.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg, cfg_changes = config_normalize.normalize_config(cfg)
    if cfg_changes:
        print(f"[config] normalized {len(cfg_changes)} entries:")
        for line in cfg_changes:
            print(f"[config] {line}")

    global PREFERRED_TABLE_FORMAT
    PREFERRED_TABLE_FORMAT = table_format.normalize_preference(cfg.get("table_format"))
    if PREFERRED_TABLE_FORMAT:
        print(f"[table-format] preference -> {PREFERRED_TABLE_FORMAT}")

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

    global ENV_HINTS
    ENV_HINTS = env_check.summarize()
    if ENV_HINTS:
        print(f"[env] available_gb={ENV_HINTS.get('available_gb'):.2f} "
              f"csv_rows={ENV_HINTS.get('csv_rows')} parquet_rows={ENV_HINTS.get('parquet_rows')} "
              f"cpu_count={ENV_HINTS.get('cpu_count')}")

    workdir = cfg.get("workdir")
    if workdir:
        wd = Path(workdir).resolve()
        print(f"[cwd] -> {wd}")
        wd.mkdir(parents=True, exist_ok=True)
        import os
        os.chdir(wd)

    wanted = [s.strip() for s in ns.sections.split(",") if s.strip()]
    sections = [s for s in SECTION_ORDER if s in wanted]

    total_errors = 0
    if "features" in sections:
        steps = _normalize_steps("features", cfg.get("features", {}).get("steps", []) or [])
        total_errors += _print_section("features", steps)
    if "data_stage" in sections:
        steps = _normalize_steps("data_stage", cfg.get("data_stage", {}).get("steps", []) or [])
        total_errors += _print_section("data_stage", steps)
    if "training" in sections:
        steps = _normalize_steps("training", cfg.get("training", {}).get("steps", []) or [])
        total_errors += _print_section("training", steps)
    if "alerts_logic" in sections:
        steps = _normalize_steps("alerts_logic", cfg.get("alerts_logic", {}).get("steps", []) or [])
        total_errors += _print_section("alerts_logic", steps)
    if "eval" in sections:
        steps = _normalize_steps("eval", cfg.get("eval", {}).get("steps", []) or [])
        total_errors += _print_section("eval", steps)
    if "seeds" in sections:
        steps = _seed_step_dicts(cfg.get("seeds", {}))
        total_errors += _print_section("seeds", steps)
    if "report" in sections:
        rep = cfg.get("report", {})
        steps = rep.get("steps") or []
        if rep.get("enabled") and not steps:
            legacy = {k: v for k, v in rep.items() if k not in ("enabled", "steps")}
            legacy.setdefault("mode", "summary")
            steps = [legacy]
        steps = _normalize_steps("report", steps)
        total_errors += _print_section("report", steps)

    if total_errors:
        print(f"\n[pipeline-doctor] found {total_errors} blocking issue(s).")
        return 1
    print("\n[pipeline-doctor] preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
