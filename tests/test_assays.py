"""Tests for assay configuration presets and registry."""

from __future__ import annotations

import pytest

from pyvplot.assays import (
    ASSAY_PRESETS,
    AssayConfig,
    get_assay,
    list_assays,
    register_assay,
)


def test_builtin_presets_exist_and_shifts_match_physics():
    names = list_assays()
    for expected in ("atac", "chec", "chip_exo", "cutrun", "cuttag", "generic", "mnase"):
        assert expected in names

    atac = get_assay("atac")
    assert atac.left_shift == 4
    assert atac.right_shift == -5
    assert atac.paired_end is True
    assert "nfr" in atac.size_bins

    chec = get_assay("chec")
    assert chec.left_shift == 0
    assert chec.right_shift == 0
    assert chec.default_feature == "cuts"
    assert "tf" in chec.size_bins

    exo = get_assay("chip_exo")
    assert exo.paired_end is False
    assert exo.default_feature == "cuts_stranded"


def test_aliases_and_passthrough():
    assert get_assay("ATAC-seq") == get_assay("atac")
    assert get_assay("ChEC-seq") == get_assay("chec")
    assert get_assay("CUT&RUN") == get_assay("cutrun")
    assert get_assay("CUT&Tag") == get_assay("cuttag")
    assert get_assay("ChIP-exo") == get_assay("chip_exo")
    assert get_assay("MNase-seq") == get_assay("mnase")

    cfg = get_assay("chec")
    assert get_assay(cfg) is cfg


def test_overrides_and_serialization_roundtrip():
    base = get_assay("atac")
    custom = base.with_overrides(left_shift=5, right_shift=-4, name="atac_custom")
    assert custom.left_shift == 5
    assert custom.right_shift == -4
    assert custom.name == "atac_custom"

    restored = AssayConfig.from_dict(custom.to_dict())
    assert restored == custom


def test_validation_and_register():
    with pytest.raises(ValueError, match="Invalid default_feature"):
        AssayConfig(name="bad", default_feature="invalid")

    with pytest.raises(ValueError, match="Invalid size bin"):
        AssayConfig(name="bad_bin", size_bins={"oops": (200, 100)})

    with pytest.raises(ValueError, match="Unknown assay preset"):
        get_assay("nonexistent_assay")

    custom = AssayConfig(name="test_custom_assay", left_shift=2, right_shift=-2)
    try:
        register_assay(custom)
        assert get_assay("test_custom_assay") == custom
        with pytest.raises(ValueError, match="already exists"):
            register_assay(custom)
        updated = custom.with_overrides(left_shift=3)
        register_assay(updated, overwrite=True)
        assert get_assay("test_custom_assay").left_shift == 3
    finally:
        ASSAY_PRESETS.pop("test_custom_assay", None)


def test_feature_lists_stay_in_sync():
    from pyvplot import locus
    from pyvplot.assays import VALID_FEATURES
    from pyvplot.extract import _FEATURE_CODES

    # every feature the kernels understand is a valid assay default, and the other way round
    assert set(_FEATURE_CODES) == set(VALID_FEATURES)
    assert set(locus._FEATURES) <= set(VALID_FEATURES)
    cfg = AssayConfig(name="x", default_feature="cuts_proximal")
    assert cfg.default_feature == "cuts_proximal"
