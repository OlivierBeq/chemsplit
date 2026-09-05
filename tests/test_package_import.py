"""performance budget: bare `import chemsplit` must be lightweight and must not import RDKit.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    return result.stdout


def test_bare_import_is_fast_and_rdkit_free():
    out = _run(
        "import sys, time\n"
        "t0 = time.time()\n"
        "import chemsplit\n"
        "dt = time.time() - t0\n"
        "print(dt)\n"
        "print('rdkit' in sys.modules)\n"
        "print('sklearn' in sys.modules)\n"
    )
    lines = out.strip().splitlines()
    dt = float(lines[0])
    rdkit_loaded = lines[1] == "True"
    sklearn_loaded = lines[2] == "True"
    assert not rdkit_loaded, "bare `import chemsplit` must not import rdkit"
    assert not sklearn_loaded, "bare `import chemsplit` must not import sklearn either"
    # Generous margin over this project's literal 400ms: this asserts chemsplit's own code doesn't
    # add meaningful overhead, not a strict environment-independent wall-clock guarantee (an
    # unrelated slow site/usercustomize hook, or filesystem latency in a given sandbox, is outside
    # chemsplit's control) -- see chemsplit/__init__.py's module docstring for the full discussion.
    assert dt < 1.0, f"bare `import chemsplit` took {dt:.3f}s, expected well under 1s"


def test_all_public_names_resolve():
    out = _run(
        "import chemsplit\n"
        "missing = [n for n in chemsplit.__all__ if not hasattr(chemsplit, n)]\n"
        "print(missing)\n"
    )
    assert out.strip() == "[]"


def test_dir_matches_all():
    import chemsplit

    assert sorted(dir(chemsplit)) == sorted(set(chemsplit.__all__))


def test_unknown_attribute_raises_attribute_error():
    import chemsplit

    with pytest.raises(AttributeError, match="not_a_real_name"):
        _ = chemsplit.not_a_real_name


def test_lazy_attribute_resolves_and_caches():
    import chemsplit

    cls1 = chemsplit.ButinaSplitter
    cls2 = chemsplit.ButinaSplitter
    assert cls1 is cls2
    assert cls1.__name__ == "ButinaSplitter"
