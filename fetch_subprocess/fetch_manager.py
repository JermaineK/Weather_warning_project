#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_manager.py

Front door for all data-fetch and track-preparation scripts:

  - era5_fetch_cds.py              -> "era5"        (single-levels; now with safe default --vars)
  - era5_fetch_cds.py              -> "era5-pl"     (pressure-levels preset: u/v at 1000..500 hPa)
  - era5_fetch_cds.py              -> "era5-both"   (single-levels + pressure-levels in one run)
  - era5_merge_singlelevels.py     -> "era5-merge-single" (merge per-month single-level .nc into union files)
  - ibtracs_fetch.py               -> "ibtracs"
  - prepare_besttrack_intensity.py -> "intensity"

Environment-configurable defaults (optional)
--------------------------------------------
FETCH_AREA         e.g. "-10,135,-25,155"
FETCH_HOURS        e.g. "0..23"
FETCH_PL_LEVELS    e.g. "1000,925,850,700,500"
FETCH_PL_VARS      e.g. "u v"
FETCH_SINGLE_VARS  e.g. "u10 v10 msl t2m d2m tcwv tp sshf slhf divergence vorticity"

Downstream override rule: your CLI flags come after the preset, so they win.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent

# ---------- Script routing ----------

SCRIPT_MAP: Dict[str, str] = {
    "era5":               "era5_fetch_cds.py",
    "era5-pl":            "era5_fetch_cds.py",
    "era5-both":          "era5_fetch_cds.py",
    "era5-merge-single":  "era5_merge_singlelevels.py",
    "ibtracs":            "ibtracs_fetch.py",
    "intensity":          "prepare_besttrack_intensity.py",
}

ALIASES: Dict[str, str] = {
    "int": "intensity",
    "era5-fetch":     "era5",
    "reanalysis":     "era5",
    "tracks":         "ibtracs",
    "ibtracs-fetch":  "ibtracs",
    "bt":             "intensity",
    "besttrack":      "intensity",
    "prepare":        "intensity",
    # handy shorthands
    "pl":             "era5-pl",
    "both":           "era5-both",
    "merge-single":   "era5-merge-single",
}

def normalize_tool(name: str) -> str:
    n = name.strip()
    if n in SCRIPT_MAP:
        return n
    if n in ALIASES:
        return ALIASES[n]
    alt = n.replace("_", "-")
    if alt in SCRIPT_MAP:
        return alt
    alt2 = n.replace("-", "_")
    if alt2 in SCRIPT_MAP:
        return alt2
    return n

def find_script(tool: str) -> Path:
    tool = normalize_tool(tool)
    if tool in SCRIPT_MAP:
        p = (HERE / SCRIPT_MAP[tool]).resolve()
        if p.exists():
            return p
    candidates = [
        tool + ".py",
        tool.replace("-", "_") + ".py",
        tool.replace("_", "-") + ".py",
    ]
    for fn in candidates:
        p = (HERE / fn).resolve()
        if p.exists():
            return p
    tried = [SCRIPT_MAP.get(tool, "<no direct map>")] + candidates
    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{tool}' in {HERE}\nTried: {tried}"
    )

# ---------- Preset wiring ----------

def split_words(val: str | None) -> List[str]:
    if not val:
        return []
    # allow spaces or commas
    raw = [s.strip() for s in val.replace(",", " ").split()]
    return [s for s in raw if s]

def build_preset_args(tool: str, extra_args: List[str] | None = None) -> List[str]:
    """
    Returns a list of CLI args to PREPEND for the selected tool.
    Downstream user-supplied args come after and therefore override these.
    """
    tool_norm = normalize_tool(tool)
    extra_args = extra_args or []

    # Env-driven defaults (optional)
    area   = os.getenv("FETCH_AREA")                # e.g. "-10,135,-25,155"
    hours  = os.getenv("FETCH_HOURS")               # e.g. "0..23"
    pl_lv  = os.getenv("FETCH_PL_LEVELS", "1000,925,850,700,500")
    pl_vs  = os.getenv("FETCH_PL_VARS",   "u v")
    # Safe default single-levels for your downstream thermo & features:
    s_vs   = os.getenv(
        "FETCH_SINGLE_VARS",
        # Minimal but complete set for your downstream thermo + shear + GKA features:
        "u10 v10 msl t2m d2m tcwv tp sshf slhf divergence vorticity",
    )

    preset: List[str] = []

    def maybe_common():
        # Only add area/hours if user didn't explicitly choose them later.
        # We still pass them early so user CLI can override.
        if area:
            preset.extend(["--area", area])
        if hours:
            preset.extend(["--hours", hours])

    if tool_norm == "era5":
        # Single-levels only, with safe defaults so first-time users don’t hit --vars errors.
        maybe_common()
        s_vars = split_words(s_vs)
        if s_vars:
            preset.extend(["--vars", *s_vars])

    elif tool_norm == "era5-pl":
        # Pressure-levels only (u, v @ default levels) for bulk-shear harvests
        maybe_common()
        preset.extend(["--dataset", "reanalysis-era5-pressure-levels"])
        pl_vars = split_words(pl_vs)
        if pl_vars:
            preset.extend(["--pl-vars", *pl_vars])
        preset.extend(["--pl-levels", pl_lv])
        preset.extend(["--pl-suffix", "uv"])  # tidy filenames

    elif tool_norm == "era5-both":
        # Single-levels + pressure-levels in one run
        maybe_common()
        s_vars = split_words(s_vs)
        if s_vars:
            preset.extend(["--vars", *s_vars])
        pl_vars = split_words(pl_vs)
        if pl_vars:
            preset.extend(["--pl-vars", *pl_vars])
        preset.extend(["--pl-levels", pl_lv])
        preset.extend(["--pl-suffix", "uv"])

    elif tool_norm == "era5-merge-single":
        # No presets; you’ll pass --in-glob/--out-dir/--suffix etc. directly.
        pass

    elif tool_norm == "ibtracs":
        # Enforce a consistent lon frame for downstream processing if the user
        # doesn’t explicitly override it on the CLI.
        has_norm = any(arg.startswith("--normalize-lon") for arg in extra_args)
        if not has_norm:
            preset.extend(["--normalize-lon", "-180..180"])

    # Other tools (intensity, etc.) don't need presets: they already have sane defaults.
    return preset

# ---------- Main ----------

def main():
    choices = sorted(set(list(SCRIPT_MAP.keys()) + list(ALIASES.keys())))
    ap = argparse.ArgumentParser(
        description="Fetch manager for ERA5, IBTrACS, and intensity preparation scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("tool", choices=choices, help="Which fetch tool (or preset) to run")
    ap.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    ap.add_argument("--quiet", action="store_true", help="Append --quiet to downstream tool if supported")
    ap.add_argument("--print-preset", action="store_true",
                    help="Print the resolved preset args for this tool (for debugging)")

    ns, extra = ap.parse_known_args()

    script = find_script(ns.tool)
    cmd = [sys.executable, str(script)]

    # Preset arguments (prepended); user's extra args come after and override as needed
    preset_args = build_preset_args(ns.tool, extra)
    if ns.print_preset:
        print("[preset]", " ".join(map(str, preset_args)))
    if preset_args:
        cmd.extend(preset_args)

    if ns.quiet:
        cmd.append("--quiet")
    cmd.extend(extra)

    print(f"\n$ {' '.join(map(str, cmd))}")
    if ns.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())