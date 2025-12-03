#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper to expose join_labels_grid via the features manager while reusing the
canonical implementation in data_subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Ensure repo root is importable then defer to the shared implementation.
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from data_subprocess.join_labels_grid import main as _main  # type: ignore

    return _main()


if __name__ == "__main__":
    sys.exit(main())
