#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_manager.py

Thin orchestration layer for sweep/search utilities:

  - sweep_runner.py                        → "run"
  - sweep_gate_runner.py                   → "gate"
  - best_f1.py                             → "best-f1"
  - find_best_f1_thresholds_constrained.py → "best-constrained"
  - pick_best_from_sweep.py                → "pick"

You can also run a small chain in one go:
  - chain "run+pick"  → run sweep_runner then pick_best_from_sweep
  - chain "gate+pick" → run sweep_gate_runner then pick_best_from_sweep
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent

SCRIPT_MAP: Dict[str, str] = {
    "run":              "sweep_runner.py",
    "gate":             "sweep_gate_runner.py",
    "best-f1":          "best_f1.py",
    "best-constrained": "find_best_f1_thresholds_constrained.py",
    "pick":             "pick_best_from_sweep.py",
}

ALIASES: Dict[str, str] = {
    "sweep":       "run",
    "gates":       "gate",
    "best":        "best-f1",
    "constrained": "best-constrained",
}

def _norm_tool(name: str) -> str:
    n = name.strip().lower()
    if n in SCRIPT_MAP:
        return n
    if n in ALIASES:
        return ALIASES[n]
    # forgiving dash/underscore differences
    n2 = n.replace("-", "_")
    for k in list(SCRIPT_MAP) + list(ALIASES):
        if k.replace("-", "_") == n2:
            return ALIASES.get(k, k)
    return n

def _find_script(tool: str) -> Path:
    t = _norm_tool(tool)
    if t in SCRIPT_MAP:
        p = (HERE / SCRIPT_MAP[t]).resolve()
        if p.exists():
            return p
    # last-ditch: try “as-is”
    candidates = [
        tool + ".py",
        tool.replace("-", "_") + ".py",
        tool.replace("_", "-") + ".py",
    ]
    for fn in candidates:
        p = (HERE / fn).resolve()
        if p.exists():
            return p
    tried = [SCRIPT_MAP.get(t, "<no direct map>")] + candidates
    raise FileNotFoundError(f"Could not locate a script for '{tool}' in {HERE}\nTried: {tried}")

def _run(tool: str, extra: List[str], quiet: bool, dry: bool) -> int:
    script = _find_script(tool)
    cmd = [sys.executable, str(script), *extra]
    if not quiet:
        print(f"\n$ {' '.join(map(str, cmd))}")
    if dry:
        return 0
    return subprocess.run(cmd).returncode

def _split_prefixed_args(prefix: str, all_extra: List[str]) -> List[str]:
    """
    Extract arguments that start with '--<prefix>.' and strip that prefix.

    Example:
      all_extra = ["--run.out", "results/x", "--pick.metric", "F1", "--foo", "bar"]
      _split_prefixed_args("run", all_extra)  -> ["--out", "results/x"]
    """
    out: List[str] = []
    skip_next = False
    for i, tok in enumerate(all_extra):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith(f"--{prefix}."):
            base = "--" + tok.split(".", 1)[1]
            out.append(base)
            # forward the next token too if it is a value (not another flag)
            if i + 1 < len(all_extra) and not str(all_extra[i + 1]).startswith("--"):
                out.append(all_extra[i + 1])
                skip_next = True
    return out

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep manager (run/gate/best/pick + simple chain)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # direct tool runner
    p_tool = sub.add_parser("tool", help="Run a single tool (run, gate, best-f1, best-constrained, pick)")
    p_tool.add_argument("name", choices=sorted(set(list(SCRIPT_MAP.keys()) + list(ALIASES.keys()))))
    p_tool.add_argument("--quiet", action="store_true")
    p_tool.add_argument("--dry-run", action="store_true")
    p_tool.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments passed through to the tool")

    # simple chains
    p_chain = sub.add_parser("chain", help="Run a small chain like run+pick or gate+pick")
    p_chain.add_argument("recipe", choices=["run+pick", "gate+pick"])
    p_chain.add_argument("--quiet", action="store_true")
    p_chain.add_argument("--dry-run", action="store_true")
    p_chain.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Pass namespaced args with prefixes:\n"
            "  --run.X …  for sweep_runner\n"
            "  --gate.X … for sweep_gate_runner\n"
            "  --pick.X … for pick_best_from_sweep\n"
            "Example:\n"
            "  sweep_manager.py chain run+pick \\\n"
            "    --run.labelled data/grid_labelled.csv.gz \\\n"
            "    --run.model models/grid_logit.pkl \\\n"
            "    --pick.csv results/sweep_summary.csv \\\n"
            "    --pick.save results/best_sweep_thresholds.csv\n"
        ),
    )

    # convenience shortcuts so you can call without “tool”
    for short in SCRIPT_MAP:
        sp = sub.add_parser(short, help=f"Shortcut for '{short}' tool")
        sp.add_argument("--quiet", action="store_true")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("extra", nargs=argparse.REMAINDER)

    ns = ap.parse_args()

    # Shortcut paths (mode equals tool name)
    if ns.mode in SCRIPT_MAP:
        return _run(ns.mode, ns.extra, ns.quiet, ns.dry_run)

    if ns.mode == "tool":
        tool = _norm_tool(ns.name)
        return _run(tool, ns.extra, ns.quiet, ns.dry_run)

    # Chain recipes
    if ns.mode == "chain":
        recipe = ns.recipe
        quiet = ns.quiet
        dry = ns.dry_run

        if recipe == "run+pick":
            run_args  = _split_prefixed_args("run",  ns.extra)
            pick_args = _split_prefixed_args("pick", ns.extra)
            rc = _run("run", run_args, quiet, dry)
            if rc != 0:
                return rc
            return _run("pick", pick_args, quiet, dry)

        if recipe == "gate+pick":
            gate_args = _split_prefixed_args("gate", ns.extra)
            pick_args = _split_prefixed_args("pick", ns.extra)
            rc = _run("gate", gate_args, quiet, dry)
            if rc != 0:
                return rc
            return _run("pick", pick_args, quiet, dry)

    # Shouldn’t reach here
    return 2

if __name__ == "__main__":
    sys.exit(main())