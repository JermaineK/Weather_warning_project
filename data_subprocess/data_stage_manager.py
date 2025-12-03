#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_stage_manager.py

Thin front door for data staging + ID subset building + GSE panel/targets +
viability training. Legacy heuristic/prob scoring is archived.

Current tools (script filenames in brackets):

  - stage-data          [stage_data.py]
  - subset-by-id        [build_train_subset_by_id.py]
  - gse-panel           [build_gse_panel_from_subset.py]
  - gse-lagged          [build_gse_lagged.py]
  - slowtick-features   [build_slowtick_features.py]
  - viability-targets   [build_viability_targets.py]
  - train-viability     [train_viability_model.py]
  - storm-timeseries    [build_storm_timeseries_by_id.py]
  - join-labels-grid    [join_labels_grid.py]

This manager does no heavy lifting itself; it just selects the script
and forwards the remaining CLI arguments as-is. It is designed to be
called from run_pipeline.py, e.g.:

  python data_subprocess/data_stage_manager.py stage-data        --config ...
  python data_subprocess/data_stage_manager.py subset-by-id      --labelled ... --out ...
  python data_subprocess/data_stage_manager.py gse-panel         --subset ... --out ...
  python data_subprocess/data_stage_manager.py gse-lagged        --panel ... --out ...
  python data_subprocess/data_stage_manager.py slowtick-features --panel ... --out ...
  python data_subprocess/data_stage_manager.py viability-targets --panel ... --out ...
  python data_subprocess/data_stage_manager.py train-viability   --train ... --model-out ...
  python data_subprocess/data_stage_manager.py storm-timeseries  --labelled-with-id ... --tracks ...
  python data_subprocess/data_stage_manager.py join-labels-grid  --features ... --labels ... --out ...

Dash/underscore variants are accepted via normalisation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent

SCRIPT_MAP: Dict[str, str] = {
    "stage-data":        "stage_data.py",
    "subset-by-id":      "build_train_subset_by_id.py",
    "gse-panel":         "build_gse_panel_from_subset.py",
    "gse-lagged":        "build_gse_lagged.py",
    "slowtick-features": "build_slowtick_features.py",
    "viability-targets": "build_viability_targets.py",
    "train-viability":   "train_viability_model.py",
    "storm-timeseries":  "build_storm_timeseries_by_id.py",
    "join-labels-grid":  "join_labels_grid.py",
    "lookup-panel":      "build_lookup_panel.py",
}

ALIASES: Dict[str, str] = {
    # convenience / backwards-compat
    "stage":      "stage-data",
    "stage_data": "stage-data",

    # subset builder
    "subset":             "subset-by-id",
    "train-subset":       "subset-by-id",
    "subset_by_id":       "subset-by-id",
    "build-train-subset": "subset-by-id",

    # GSE panel
    "panel":         "gse-panel",
    "gse":           "gse-panel",
    "gse_panel":     "gse-panel",

    # lagged GSE
    "gse-lag":       "gse-lagged",
    "gse_lagged":    "gse-lagged",
    "lagged-gse":    "gse-lagged",

    # slowtick features
    "slowtick-feat":     "slowtick-features",
    "slowtick_feat":     "slowtick-features",

    # viability targets
    "targets":             "viability-targets",
    "viability":           "viability-targets",
    "viability_targets":   "viability-targets",

    # viability trainer
    "train-viability-model": "train-viability",
    "viability-train":       "train-viability",

    # storm timeseries
    "storm-ts":          "storm-timeseries",
    "storm_timeseries":  "storm-timeseries",

    # label joiner
    "join-labels":       "join-labels-grid",
    "labels":            "join-labels-grid",
    "join_labels_grid":  "join-labels-grid",
    # lookup panel
    "lookup":            "lookup-panel",
    "lookup_panel":      "lookup-panel",

    # diagnostics
}


OPTION_ALIASES: Dict[str, Dict[str, str]] = {
    # join-labels-grid expects underscore flags; run_pipeline emits hyphenated ones.
    "join-labels-grid": {
        "--storm-radius-deg": "--storm_radius_deg",
        "--storm-time-h":     "--storm_time_h",
        "--near-radius-deg":  "--near_radius_deg",
        "--near-time-h":      "--near_time_h",
    },
}


def rewrite_args(tool: str, argv: List[str]) -> List[str]:
    """
    Map hyphenated flags back to the underscore variants expected by the script.
    Supports both '--flag value' and '--flag=value' forms.
    """
    fmap = OPTION_ALIASES.get(tool)
    if not fmap:
        return argv

    out: List[str] = []
    for tok in argv:
        if tok.startswith("--"):
            if "=" in tok:
                flag, val = tok.split("=", 1)
                out.append(f"{fmap.get(flag, flag)}={val}")
                continue
            out.append(fmap.get(tok, tok))
            continue
        out.append(tok)
    return out


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
            "Data-stage manager: staging, ID subsets, GSE panel/targets, viability training. "
            "Legacy heuristic/prob scoring scripts are archived."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "tool",
        help=(
            "Which data-stage tool to run "
            "(stage-data, subset-by-id, gse-panel, viability-targets, "
            "gse-lagged, slowtick-features, train-viability, storm-timeseries, join-labels-grid). "
            "Dash/underscore variants and simple aliases are accepted."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without running it.",
    )

    args, extra = ap.parse_known_args()

    tool = normalize_tool(args.tool)
    rewritten = rewrite_args(tool, extra)

    # join-labels-grid now lives under features_subprocess; forward there to keep
    # data-stage configs working without duplicating logic.
    if tool == "join-labels-grid":
        feat_mgr = (HERE.parent / "features_subprocess" / "features_manager.py").resolve()
        cmd = [sys.executable, str(feat_mgr), "join-labels-grid", *rewritten]
    else:
        script = find_script(tool)
        cmd = [sys.executable, str(script), *rewritten]

    print(f"\n$ {' '.join(map(str, cmd))}")
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
