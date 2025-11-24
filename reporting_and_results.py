#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reporting_and_results.py — thin orchestrator for end-of-pipeline reporting.

Subcommands → script it calls (in reports_subprocess/):
  summary        → report_generate_summary.py
  sanity         → report_sanity_checks.py
  debug-snaps    → report_debug_snaps.py
  maps           → report_make_maps.py

Everything after the subcommand is forwarded unchanged.
"""

from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path
from typing import List, Dict

SUBPROC_DIR_DEFAULT = "reports_subprocess"
CANDIDATES: Dict[str, List[str]] = {
    "summary":     ["report_generate_summary.py", "generate_summary.py"],
    "sanity":      ["report_sanity_checks.py", "sanity_checks.py"],
    "debug-snaps": ["report_debug_snaps.py", "debug_snaps.py"],
    "maps":        ["report_make_maps.py", "make_maps.py"],
}

def find_script(name: str, folder: Path) -> Path:
    tried = []
    for fn in CANDIDATES.get(name, []):
        p = folder / fn
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not locate a script for '{name}' in {folder}\nTried: {tried}"
    )

def main():
    ap = argparse.ArgumentParser(
        description="Reporting/results orchestrator (delegates to reports_subprocess/*.py).",
        add_help=False
    )
    ap.add_argument("subcommand", choices=list(CANDIDATES.keys()))
    ap.add_argument("--scripts-dir", default=SUBPROC_DIR_DEFAULT)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER)
    ns, unknown = ap.parse_known_args()

    scripts_dir = Path(ns.scripts_dir)
    script = find_script(ns.subcommand, scripts_dir)
    forwarded = []
    if unknown: forwarded += unknown
    if ns.passthrough: forwarded += ns.passthrough

    cmd = [ns.python_exe, str(script), *forwarded]
    print(f"[report] ➜ {script.name}")
    print(f"[report] cwd={os.getcwd()}")
    if forwarded:
        print(f"[report] args={' '.join(forwarded)}")

    if ns.dry_run:
        print("[report] DRY RUN")
        return

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()