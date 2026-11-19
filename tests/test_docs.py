"""Tutorial code must run and print what the docs say it prints (see docs/run_docs.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
_spec = importlib.util.spec_from_file_location("run_docs", DOCS / "run_docs.py")
assert _spec is not None and _spec.loader is not None
run_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_docs)


@pytest.mark.parametrize(
    "page",
    sorted((DOCS / "tutorials").glob("*.md")),
    ids=lambda p: p.stem,
)
def test_tutorial(page: Path):
    errors = run_docs.check_page(page)
    assert not errors, "\n\n".join(errors) + "\n\nRun `python docs/run_docs.py --update` if intended."
