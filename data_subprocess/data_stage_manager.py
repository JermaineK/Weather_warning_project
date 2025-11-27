#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_stage_manager.py

Thin front door for data staging + heuristic scoring + IBTrACS matching +
threshold feature mining + probability prediction + per-lead logit training.

Current tools (script filenames in brackets):

  - stage-data        [stage_data.py]
  - score-heuristic   [thresholds_and_alerts.py]
  - ibtracs-match     [ibtracs_match.py]
  - thresholds-scan   [thresholds_scan.py]
  - predict-prob      [predict_storm_probability.py]
  - train-logit       [train_per_lead_logit.py]

This manager does no heavy lifting itself; it just selects the script
and forwards the remaining CLI arguments as-is. It is designed to be
called from run_pipeline.py, e.g.:

  python data_subprocess/data_stage_manager.py stage-data      --config ...
  python data_subprocess/data_stage_manager.py score-heuristic --scoring-src ...
  python data_subprocess/data_stage_manager.py ibtracs-match   --alerts ... --tracks ...
  python data_subprocess/data_stage_manager.py thresholds-scan --alerts-with-targets ...
  python data_subprocess/data_stage_manager.py predict-prob    --scoring-src ...
  python data_subprocess/data_stage_manager.py train-logit     --labelled ... --out ...

Dash/underscore variants are accepted via normalisation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent

SCRIPT_MAP: Dict[str, str] = {
    "stage-data":      "stage_data.py",
    "score-heuristic": "thresholds_and_alerts.py",
    "ibtracs-match":   "ibtracs_match.py",
    "thresholds-scan": "thresholds_scan.py",
    "predict-prob":    "predict_storm_probability.py",
    "train-logit":     "train_per_lead_logit.py",
}

ALIASES: Dict[str, str] = {
    # convenience / backwards-compat
    "stage":      "stage-data",
    "stage_data": "stage-data",

    "score":           "score-heuristic",
    "heuristic":       "score-heuristic",
    "heuristic-score": "score-heuristic",

    "ibtracs":         "ibtracs-match",
    "match":           "ibtracs-match",

    "scan":            "thresholds-scan",
    "thr-scan":        "thresholds-scan",

    "predict":         "predict-prob",
    "predict_prob":    "predict-prob",
    "prob":            "predict-prob",

    # new trainer aliases
    "train":                 "train-logit",
    "train_logit":           "train-logit",
    "train-perlead-logit":   "train-logit",
    "train_per_lead_logit":  "train-logit",
    "perlead-logit":         "train-logit",
}


def normalize_tool(name: str) -> str:
    name = name.strip()
    if name in SCRIPT_MAP:
        return name
    if name in ALIASES:
        return ALIASES[name]
    # allow dash/underscore drift
    alt = name.replace("_", "-")
    if alt in SCRIPT_MAP:
        return alt
    if alt in ALIASES:
        return ALIASES[alt]
    return name


def find_script(cmd: str) -> Path:
    """
    Resolve a subcommand name to a script in this directory.

    Priority:
      1) SCRIPT_MAP via normalize_tool
      2) Guess "<cmd>.py", "<cmd_with_dashes>.py", "<cmd_with_underscores>.py"
    """
    tool = normalize_tool(cmd)

    mapped = SCRIPT_MAP.get(tool)
    if mapped:
        p = (HERE / mapped).resolve()
        if p.exists():
            return p

    # fallbacks: try literal and dash/underscore variants as filenames
    variants = {
        tool,
        tool.replace("-", "_"),
        tool.replace("_", "-"),
    }
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

    tried = [SCRIPT_MAP.get(tool, "<no direct map>")] + file_candidates
    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{cmd}' in {HERE}\n"
        f"Tried: {tried}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Data-stage manager: staging, heuristic scoring, IBTrACS matching, "
            "threshold feature mining, probability prediction, and per-lead logit training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "tool",
        help=(
            "Which data-stage tool to run "
            "(stage-data, score-heuristic, ibtracs-match, thresholds-scan, "
            "predict-prob, train-logit). "
            "Dash/underscore variants and simple aliases are accepted."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without running it.",
    )

    args, extra = ap.parse_known_args()

    script = find_script(args.tool)
    cmd = [sys.executable, str(script), *extra]

    print(f"\n$ {' '.join(map(str, cmd))}")
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
