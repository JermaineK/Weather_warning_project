#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alerts_logic_manager.py

Thin front door for the alert-shaping tools in this directory.

Current tools (script filenames in brackets):
  - viability-pipeline               [viability_pipeline.py]
  - apply / apply-rules              [apply_rules.py]
  - rule-applier                     [rule_applier.py]
  - rule-miner                       [rule_miner.py]
  - leadband-mine                    [leadband_rule_miner.py]
  - phase-rules                      [phase_rules.py]
  - apply-thresholds / apply-thr     [apply_thresholds.py]
  - blend-apply-norm                 [blend_apply_norm.py]      (legacy / experimental)
  - blend-sweep                      [blend_sweep.py]           (legacy / experimental)
  - throttle                         [throttle_by_percentile.py]
  - denoise                          [denoise_alerts.py]

The manager itself does no heavy lifting: it just picks the script and
forwards all remaining arguments as-is, so pipeline.yaml can treat this
as a single entry point.

Examples
--------

# Simple rules on a labelled grid
python alerts_logic_manager.py apply-rules \
  --labelled data/grid_labelled.csv.gz \
  --rules "gka_knee_ratio|pos|1.2" \
  --out results/alerts_rules.csv.gz

# Lead-band rule mining
python alerts_logic_manager.py leadband-mine \
  --labelled data/grid_labelled.csv.gz \
  --target pregen \
  --out results/leadband_rules.csv

# Phase comparison / rule candidates
python alerts_logic_manager.py phase-rules \
  --labelled data/grid_labelled.csv.gz \
  --build models/build_l1.pkl \
  --relax models/relax_l1.pkl \
  --outdir results/phase_rules

# Apply a trained bundle + threshold to produce base alerts
python alerts_logic_manager.py apply-thresholds \
  --labelled data/grid_labelled_FMA_gka.parquet \
  --model models/grid_logit_perlead.pkl \
  --lead-hours 72 \
  --thr 0.08 \
  --out results/alerts/base_lead72_thr0.08.parquet

# Throttle alerts per-hour by quantile
python alerts_logic_manager.py throttle \
  --alerts results/alerts_raw.csv.gz \
  --keep-quantile 0.9 \
  --out results/alerts_throttled.csv.gz

# Spatial + temporal denoise of an alert grid
python alerts_logic_manager.py denoise \
  --alerts results/alerts_throttled.csv.gz \
  --persist-hours 3 \
  --min-neighbors 3 \
  --out results/alerts_denoised.csv.gz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent

# Canonical subcommand -> script filename.
# Dash/underscore aliases are handled in find_script, so this map can stay small.
SCRIPT_MAP: Dict[str, str] = {
    "viability-pipeline": "viability_pipeline.py",
    # spiral-conditioned genesis trigger (geom-val early-warning model)
    "genesis-trigger":    "genesis_trigger.py",
    # core rule appliers / discovery
    "apply":              "apply_rules.py",
    "apply-rules":        "apply_rules.py",
    "rule-applier":       "rule_applier.py",
    "rule-miner":         "rule_miner.py",
    "leadband-mine":      "leadband_rule_miner.py",
    "phase-rules":        "phase_rules.py",

    # base alerts directly from a trained bundle + thresholds
    "apply-thresholds":   "apply_thresholds.py",
    "apply-thr":          "apply_thresholds.py",

    # blending (kept as legacy / experimental)
    "blend-apply-norm":   "blend_apply_norm.py",
    "blend-sweep":        "blend_sweep.py",

    # throttle / denoise
    "throttle":           "throttle_by_percentile.py",
    "denoise":            "denoise_alerts.py",
}


def find_script(cmd: str) -> Path:
    """
    Resolve a subcommand name to a script in this directory.

    - First: look up cmd in SCRIPT_MAP.
    - Then: try dash/underscore normalisations.
    - Finally: try treating the token itself as a basename and append .py.
    """
    # 1) direct mapping
    mapped = SCRIPT_MAP.get(cmd)
    if mapped:
        p = (HERE / mapped).resolve()
        if p.exists():
            return p

    # 2) normalise dash/underscore variants via the map
    variants = {
        cmd,
        cmd.replace("-", "_"),
        cmd.replace("_", "-"),
    }
    for v in variants:
        mapped = SCRIPT_MAP.get(v)
        if mapped:
            p = (HERE / mapped).resolve()
            if p.exists():
                return p

    # 3) fall back to file-name guessing
    file_candidates = []
    for v in variants:
        if v.endswith(".py"):
            file_candidates.append(v)
        else:
            file_candidates.append(v + ".py")

    for fn in file_candidates:
        p = (HERE / fn).resolve()
        if p.exists():
            return p

    tried = [SCRIPT_MAP.get(cmd, "<no direct map>")] + file_candidates
    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{cmd}' in {HERE}\n"
        f"Tried: {tried}"
    )


def build_forward_args(_ns: argparse.Namespace) -> list[str]:
    """
    Manager-owned flags -> forwarded argv.

    Intentionally minimal: we currently don't own any flags that need to be
    forwarded, but this helper is kept for future toggles if needed.
    """
    return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Alerts logic manager: thin wrapper around the per-step scripts "
            "(rules, leadband mining, phase rules, bundle application, blending, throttling, denoise)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "tool",
        help=(
            "Which alerts-logic tool to run "
            "(e.g. apply-rules, rule-miner, leadband-mine, phase-rules, "
            "apply-thresholds, blend-apply-norm, blend-sweep, throttle, denoise). "
            "Dash/underscore variants are accepted."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without running it.",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to auto-fill common defaults for viability alerts (apply-thr/throttle/denoise).",
    )

    # Everything unknown is forwarded to the target script.
    args, extra = ap.parse_known_args()

    def _has_flag(seq: list[str], flag: str) -> bool:
        return any(x == flag or x.startswith(flag + "=") for x in seq)

    def _apply_run_defaults(tool: str, run_name: str | None, argv: list[str]) -> list[str]:
        if not run_name:
            return argv
        out = list(argv)

        def ensure(flag: str, *vals: str):
            if _has_flag(out, flag):
                return
            out.extend([flag, *map(str, vals)])

        tool_norm = tool.replace("_", "-")
        if tool_norm in {"apply-thresholds", "apply-thr"}:
            ensure("--labelled", "data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet")
            ensure("--model", "models/viability_model.pkl")
            ensure("--metrics-json", "models/viability_model_metrics.json")
            ensure("--prob-col", "prob_viable")
            ensure("--flag-col", "alert_base")
            ensure("--normalize-lon", "-180..180")
            if not _has_flag(out, "--out"):
                ensure("--out", f"results/alerts/alerts_{run_name}_base.parquet")
        elif tool_norm == "throttle":
            ensure("--normalize-lon", "-180..180")
            ensure("--prob-col", "prob_viable")
            ensure("--flag-col", "alert_base")
            if not _has_flag(out, "--alerts"):
                ensure("--alerts", f"results/alerts/alerts_{run_name}_base.parquet")
            if not _has_flag(out, "--out"):
                ensure("--out", f"results/alerts/alerts_{run_name}_thr.parquet")
        elif tool_norm == "denoise":
            ensure("--flag-col", "alert_base")
            ensure("--flag-out", "alert_final")
            ensure("--score-col", "prob_viable")
            if not _has_flag(out, "--alerts"):
                ensure("--alerts", f"results/alerts/alerts_{run_name}_thr.parquet")
            if not _has_flag(out, "--out"):
                ensure("--out", f"results/alerts/alerts_{run_name}_final.parquet")
        elif tool_norm == "viability-pipeline":
            ensure("--run-name", run_name)
        return out

    script = find_script(args.tool)
    fwd = build_forward_args(args)
    extra = _apply_run_defaults(args.tool, args.run_name, list(extra))
    cmd = [sys.executable, str(script), *fwd, *extra]

    print(f"\n$ {' '.join(map(str, cmd))}")
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
