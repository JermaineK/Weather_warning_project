#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeds_tracks.py — lightweight launcher

This CLI dispatches subcommands to smaller, focused scripts living in the same
folder as this file (typically: seeds_subprocess). It keeps your top-level
neat and lets each tool evolve independently.

Current subcommands
-------------------
  analyze          → analyze_seeds.py      (seed diagnostics / summaries)
  from-alerts      → from_alerts.py        (build hourly proto-seeds from alerts)
  proto-outcomes   → proto_outcomes.py     (link proto-tracks and label outcomes vs IBTrACS)
  starts-vs-tracks → starts_vs_tracks.py   (seed starts → patches → best-track lead-time diagnostics)

Examples
--------
  python seeds_tracks.py analyze \
    --seeds results/seedmaps/coral_sea_demo_union_byhour.csv \
    --out-dir results/seedmaps

  python seeds_tracks.py from-alerts \
    --alerts "results/alerts/alerts_*_thr*.csv.gz" \
    --out-dir results/seedmaps \
    --run-name coral_sea_demo

  python seeds_tracks.py proto-outcomes \
    --seeds results/seedmaps/coral_sea_demo_union_byhour.csv \
    --ibtracs data/tracks/tracks_subset.csv \
    --out-dir results/seedmaps \
    --run-name coral_sea_demo

  python seeds_tracks.py starts-vs-tracks \
    --seeds results/seedmaps/coral_sea_demo_union_byhour.csv \
    --tracks data/tracks/tracks_subset.csv \
    --out-dir results/seedmaps
"""

from __future__ import annotations

import argparse
import shutil
import sys
import subprocess
from pathlib import Path
from typing import List

# ---------- subcommand → candidate script filenames ----------
CANDIDATES = {
    "analyze": [
        "analyze_seeds.py",
        "analyze.py",
        "seeds_analyze.py",
    ],
    "from-alerts": [
        "from_alerts.py",
        "seeds_from_alerts.py",
        "seeds_from_alerts_v2.py",
    ],
    "proto-outcomes": [
        "proto_outcomes.py",
        "seed_proto_tracks_outcomes.py",
        "proto_tracks_outcomes.py",
    ],
    "starts-vs-tracks": [
        "starts_vs_tracks.py",
        "seed_starts_vs_tracks.py",
    ],
}

# If None, default to the directory containing this file.
DEFAULT_SUBDIR: str | None = None


def resolve_scripts_dir(user_arg: str | None) -> Path:
    """
    Resolve the directory that contains the sub-scripts.

    - If --scripts-dir is provided, use that.
    - Otherwise, default to the directory containing this launcher.
      (If DEFAULT_SUBDIR is set, treat it as a subfolder of that.)
    """
    if user_arg:
        return Path(user_arg).expanduser().resolve()

    here = Path(__file__).resolve().parent
    if DEFAULT_SUBDIR:
        return (here / DEFAULT_SUBDIR).resolve()
    return here


def find_script(subcommand: str, scripts_dir: Path) -> Path:
    """
    Map a subcommand to an actual script file in scripts_dir.

    Resolution order:
      1) CANDIDATES[subcommand] list, in order.
      2) Generic names: <subcommand>.py and seeds_<subcommand>.py (with '-'→'_').
      3) Case-insensitive match over *.py in scripts_dir for all candidates above.
    """
    # 1) explicit mapping
    for fname in CANDIDATES.get(subcommand, []):
        p = scripts_dir / fname
        if p.exists():
            return p

    # 2) generic patterns
    base = subcommand.replace("-", "_")
    generic = [
        f"{base}.py",
        f"seeds_{base}.py",
    ]
    for fname in generic:
        p = scripts_dir / fname
        if p.exists():
            return p

    # 3) last-resort, case-insensitive search over the union of candidate names
    wanted = CANDIDATES.get(subcommand, []) + generic
    lower_targets = {n.lower() for n in wanted}
    for p in scripts_dir.glob("*.py"):
        if p.name.lower() in lower_targets:
            return p

    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{subcommand}' in {scripts_dir}\n"
        f"Tried: {wanted}"
    )


def build_cmd(python: str, script_path: Path, passthrough: List[str]) -> List[str]:
    # Use list form for robust quoting on Windows
    return [python, str(script_path), *passthrough]


def main():
    top = argparse.ArgumentParser(
        description="Seed/Track Toolkit (launcher for modular subprocess scripts).",
        add_help=True,
    )
    # global options
    top.add_argument(
        "--scripts-dir",
        default=None,
        help=(
            "Folder containing sub-scripts. "
            "Default: the directory containing this file (typically seeds_subprocess)."
        ),
    )
    top.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )

    # first positional: subcommand
    top.add_argument(
        "subcommand",
        choices=sorted(CANDIDATES.keys()),
        help="Which tool to run.",
    )

    # collect the rest verbatim for the sub-script
    top.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected sub-script.",
    )

    ns = top.parse_args()

    scripts_dir = resolve_scripts_dir(ns.scripts_dir)
    if not scripts_dir.exists():
        raise SystemExit(f"[error] scripts dir not found: {scripts_dir}")

    script_path = find_script(ns.subcommand, scripts_dir)

    # Strip an optional leading '--' that argparse may leave in REMAINDER
    passthrough = ns.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    py = sys.executable or shutil.which("python") or "python"
    cmd = build_cmd(py, script_path, passthrough)

    print(f"[launch] {ns.subcommand} -> {script_path}")
    if ns.dry_run:
        print("[dry-run]", cmd)
        return

    # Inherit stdout/stderr; propagate exit code
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise SystemExit(res.returncode)
    except FileNotFoundError as e:
        # Usually means sys.executable/python missing, or script disappeared
        print(f"[error] {e}")
        raise SystemExit(127)


if __name__ == "__main__":
    main()