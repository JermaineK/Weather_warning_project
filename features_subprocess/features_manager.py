"""
features_manager.py - delegator + friendly flag translator.

All real work is done by sibling scripts. This manager chooses the tool and
passes CLI args through, with optional per-mode rewriting so you can use
nice hyphenated flags in YAML/CLI while core scripts keep their own API.

Modes -> scripts
  build               -> build_features_grid.py
  patch               -> features_patch.py
  gka                 -> compute_gka_features.py
  gka-ms              -> compute_gka_multiscale.py
  integrate-thermo    -> integrate_era5_thermo.py
  spherical-feedback  -> compute_spherical_feedback.py
  spherical           -> compute_spherical_feedback.py (alias)
  bulk-shear          -> features_bulk_shear.py
  join-features       -> features_join_features.py
  join-labels-grid    -> join_labels_grid.py
  join                -> join_labels_grid.py (alias)
  add-ids             -> add_row_id.py
  id                  -> add_row_id.py (alias)
  symmetry            -> compute_symmetry_features.py
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------
ROUTING: Dict[str, str] = {
    "build":              "build_features_grid.py",
    "patch":              "features_patch.py",
    "gka":                "compute_gka_features.py",
    "gka-ms":             "compute_gka_multiscale.py",
    "symmetry":           "compute_symmetry_features.py",
    "integrate-thermo":   "integrate_era5_thermo.py",
    "spherical-feedback": "compute_spherical_feedback.py",
    "spherical":          "compute_spherical_feedback.py",  # alias
    "bulk-shear":         "features_bulk_shear.py",
    "join-features":      "features_join_features.py",
    "join-labels-grid":   "join_labels_grid.py",
    "join":               "join_labels_grid.py",  # legacy alias
    "add-ids":            "add_row_id.py",
    "id":                 "add_row_id.py",
}

# --------------------------------------------------------------------
# Per-mode flag rewrite tables
#   Key: manager mode
#   Value: mapping from user-facing flag -> script-expected flag
# --------------------------------------------------------------------
FLAG_MAPS: Dict[str, Dict[str, str]] = {
    # Most other tools already speak kebab; no rewrites needed.
    "build": {},
    "patch": {},
    "gka": {},
    "gka-ms": {},
    "symmetry": {},
    "integrate-thermo": {},
    "spherical-feedback": {},
    "bulk-shear": {},
    "join-features": {},
    "join-labels-grid": {
        "--storm-radius-deg": "--storm_radius_deg",
        "--storm-time-h":     "--storm_time_h",
        "--near-radius-deg":  "--near_radius_deg",
        "--near-time-h":      "--near_time_h",
        "--pregen-future-h":  "--pregen_future_h",
    },
    "join": {
        "--storm-radius-deg": "--storm_radius_deg",
        "--storm-time-h":     "--storm_time_h",
        "--near-radius-deg":  "--near_radius_deg",
        "--near-time-h":      "--near_time_h",
        "--pregen-future-h":  "--pregen_future_h",
    },
    "add-ids": {},
    "id": {},
}

# For certain modes/flags, trim spaces after commas so merges-on lists are tidy.
COMPACT_CSV: Dict[str, List[str]] = {
    "join-features": ["--on"],  # e.g., --on "time, lat, lon" -> "time,lat,lon"
}

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _rewrite_args(mode: str, argv: List[str]) -> List[str]:
    """
    Rewrite flags according to FLAG_MAPS for this mode.
    Supports both '--flag value' and '--flag=value'.
    """
    fmap = FLAG_MAPS.get(mode, {})
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            # handle --flag=value
            if "=" in tok:
                f, val = tok.split("=", 1)
                out.append(fmap.get(f, f) + "=" + val)
                i += 1
                continue
            # handle --flag value
            mapped = fmap.get(tok, tok)
            out.append(mapped)
            # carry through a value token if present and not a flag
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        # positional / stray token
        out.append(tok)
        i += 1
    return out

def _compact_csv_if_needed(mode: str, argv: List[str]) -> List[str]:
    """
    For flags that should be CSVs, remove spaces after commas so downstream
    parsers that split on commas don't inherit spaces.
    Works with both '--flag value' and '--flag=value'.
    """
    targets = set(COMPACT_CSV.get(mode, []))
    if not targets:
        return argv

    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and "=" in tok:
            f, val = tok.split("=", 1)
            if f in targets:
                val = ",".join([p.strip() for p in val.split(",")])
            out.append(f"{f}={val}")
            i += 1
            continue
        if tok in targets:
            out.append(tok)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                val = argv[i + 1]
                val = ",".join([p.strip() for p in val.split(",")])
                out.append(val)
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    return out

def _run(cmd: List[str]) -> int:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Feature pipeline front-door delegator")
    parser.add_argument("mode", choices=sorted(ROUTING.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Print rewritten command and exit")
    ns, passthrough = parser.parse_known_args()

    script = (HERE / ROUTING[ns.mode]).resolve()
    if not script.exists():
        parser.error(f"Missing script for mode '{ns.mode}': {script}")

    # 1) Rewrite flags if this mode needs it
    args1 = _rewrite_args(ns.mode, passthrough)
    # 2) Compact CSV-ish values (e.g., join-features --on)
    args2 = _compact_csv_if_needed(ns.mode, args1)

    cmd = [sys.executable, str(script), *args2]
    print("# features_manager ->", script.name)
    if ns.dry_run:
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        return 0

    rc = _run(cmd)
    return rc

if __name__ == "__main__":
    sys.exit(main())
