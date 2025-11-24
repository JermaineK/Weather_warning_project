#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
grid_score.py — thin orchestrator that defers to the 5 scoring subprocesses.

Available subcommands (→ script it calls):
  predict-raw            → predict_raw_scores.py
  per-lead               → score_per_lead.py
  per-lead-logit-bundle  → score_per_lead_logit_bundle.py
  all-leads-logit        → score_all_leads_logit.py
  apply-thresholds       → apply_thresholds.py

Everything after the subcommand is passed through to the target script unchanged.

Examples:
  python grid_score.py predict-raw --features data/features.parquet --model models/x.joblib --out results/raw.parquet
  python grid_score.py per-lead --features data/features.parquet --bundle models/perlead.pkl --leads 1..240 --out-dir results/alerts
  python grid_score.py apply-thresholds --in results/alerts --thr-map '{"24":0.06,"72":0.04}'
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict

# Canonical subcommands → candidate filenames (first match wins)
SUBPROCESS_CANDIDATES: Dict[str, List[str]] = {
    "predict-raw": [
        "predict_raw_scores.py", "predict-raw-scores.py", "predictraw.py"
    ],
    "per-lead": [
        "score_per_lead.py", "score-per-lead.py"
    ],
    "per-lead-logit-bundle": [
        "score_per_lead_logit_bundle.py", "score-per-lead-logit-bundle.py", "score_per_lead_bundle.py"
    ],
    "all-leads-logit": [
        "score_all_leads_logit.py", "score-all-leads-logit.py"
    ],
    "apply-thresholds": [
        "apply_thresholds.py", "apply-thresholds.py"
    ],
}

def find_script(subcmd: str, scripts_dir: Path) -> Path:
    candidates = SUBPROCESS_CANDIDATES.get(subcmd, [])
    tried: List[str] = []
    for name in candidates:
        p = scripts_dir / name
        tried.append(str(p))
        if p.exists():
            return p
    # Helpful error with what we looked for
    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{subcmd}' in {scripts_dir}\n"
        f"Tried: {candidates}"
    )

def main():
    p = argparse.ArgumentParser(
        description="Grid scoring orchestrator (delegates to score_subprocess/*.py).",
        add_help=False  # let downstream scripts own --help; we keep a small front help
    )
    # We only parse the few flags *we* care about; leave the rest to the subprocess.
    p.add_argument("subcommand", choices=list(SUBPROCESS_CANDIDATES.keys()))
    p.add_argument("--scripts-dir", default="score_subprocess",
                   help="Folder containing the 5 scoring scripts (default: score_subprocess)")
    p.add_argument("--python-exe", default=sys.executable,
                   help="Python interpreter to invoke the subprocess (default: current)")
    p.add_argument("--dry-run", action="store_true", help="Print the command that would run, then exit.")
    p.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER,
                   help="Arguments after -- are forwarded verbatim to the target script")

    # Parse only our known args; leave the rest intact (so users can omit -- separator)
    ns, unknown = p.parse_known_args()

    scripts_dir = Path(ns.scripts_dir)
    script_path = find_script(ns.subcommand, scripts_dir)

    # Build argv for the child: python <script> <unknown + passthrough>
    forwarded: List[str] = []
    if unknown:
        forwarded += unknown
    if ns.passthrough:
        forwarded += ns.passthrough

    cmd = [ns.python_exe, str(script_path), *forwarded]

    print(f"[grid_score] ➜ {script_path.name}")
    print(f"[grid_score] cwd={os.getcwd()}")
    print(f"[grid_score] python={ns.python_exe}")
    if forwarded:
        print(f"[grid_score] args={' '.join(forwarded)}")
    else:
        print(f"[grid_score] args=(none)")

    if ns.dry_run:
        print("[grid_score] DRY RUN: not executing")
        return

    # Run child; stream output live; raise on non-zero
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Bubble up with friendly message; preserve the return code
        print(f"[grid_score] Subprocess failed (exit={e.returncode}).", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()