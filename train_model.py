#!/usr/bin/env python3
"""Compatibility entrypoint for model training.

`edge_ml.py` is now the canonical trainer implementation. This wrapper preserves
legacy invocations (dashboard/manual scripts) while delegating all logic to
`edge_ml.main()`.
"""

from edge_ml import main


if __name__ == "__main__":
    raise SystemExit(main())
