#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_storm_timeseries_by_id.py

Placeholder for storm-centred G/S/E time series aggregation.
Not implemented yet; kept as a stub to document CLI shape.
"""

from __future__ import annotations

import argparse
import sys


# Agent: stub only. Implement aggregation when slow-tick analysis is ready.


def main() -> None:
    ap = argparse.ArgumentParser(
        description="(Stub) Build storm-centred time series of G/S/E fields around track points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labelled-with-id", help="Rich grid with row_id and features.", required=False)
    ap.add_argument("--tracks", help="Storm tracks table (CSV/Parquet).", required=False)
    ap.add_argument("--out", help="Output time series panel.", required=False)
    ap.add_argument("--t-before", type=float, default=240.0, help="Hours before reference time.")
    ap.add_argument("--t-after", type=float, default=48.0, help="Hours after reference time.")
    ap.add_argument("--radius-deg", type=float, default=2.0, help="Radius (deg) around track point.")
    ap.add_argument("--ref-kind", default="genesis", help="Reference time kind: genesis|max-int.")
    args = ap.parse_args()

    print(
        "[stub] build_storm_timeseries_by_id.py is not implemented yet. "
        "When ready, enable aggregation for slow-tick analysis."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
