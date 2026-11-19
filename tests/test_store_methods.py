"""FragmentStore method forms must expose the same parameters as the module functions."""

from __future__ import annotations

import inspect

import pytest

import pyvplot as vp

PAIRS = [
    "vplot",
    "to_matrix",
    "cut_site_footprint",
    "footprint_score",
    "extract_events",
    "insert_size_distribution",
    "phasogram",
    "frip",
    "qc_summary",
]


@pytest.mark.parametrize("name", PAIRS)
def test_method_signature_matches_function(name: str):
    fn_params = list(inspect.signature(getattr(vp, name)).parameters.values())[1:]  # drop store
    meth_params = list(inspect.signature(getattr(vp.FragmentStore, name)).parameters.values())[1:]
    assert [(p.name, p.default) for p in meth_params] == [(p.name, p.default) for p in fn_params]
