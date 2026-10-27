"""Pytest configuration for the repository root.

The application package lives at the repository root (``app/``) rather than
under ``src/``, so the package is not importable by default when pytest is
run from the repository root. Without this file, ``pytest`` fails at
collection time with ``ModuleNotFoundError: No module named 'app'`` — only
``python -m pytest`` worked, because that form happens to put the current
directory on ``sys.path``.

Adding the repository root to ``sys.path`` makes the documented command
(``pytest``) work, so the test suite runs the same way locally and in CI.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
