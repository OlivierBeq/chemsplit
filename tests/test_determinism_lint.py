"""No bare ``np.argmax``/``np.argmin`` (or ``numpy.argmax``/``numpy.argmin``) anywhere
in ``chemsplit/`` outside ``chemsplit/determinism.py`` itself. Every tie-break in the library
must resolve to the smallest index, and the only place allowed to implement that rule is
``determinism.argmax_tiebreak``/``argmin_tiebreak``.
"""

from __future__ import annotations

import re
from pathlib import Path

_PATTERN = re.compile(r"\b(?:np|numpy)\.arg(?:max|min)\s*\(")

_ALLOWED_FILES = {"chemsplit/determinism.py"}


def _chemsplit_root() -> Path:
    return Path(__file__).resolve().parent.parent / "chemsplit"


def test_no_bare_argmax_argmin_outside_determinism():
    root = _chemsplit_root()
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).as_posix()
        if rel in _ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    assert not violations, (
        "bare np.argmax/np.argmin found outside chemsplit/determinism.py (route through "
        "determinism.argmax_tiebreak/argmin_tiebreak instead):\n" + "\n".join(violations)
    )
