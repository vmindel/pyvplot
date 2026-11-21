"""Tests for bedGraph, BigWig, and pymisha track export helpers."""

from __future__ import annotations

import gzip
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pyvplot.export import to_bedgraph, to_bigwig, to_binned_track, to_pymisha
from pyvplot.store import FragmentStore


def _make_export_store() -> FragmentStore:
    # 3 fragments on chr1:
    # [10, 35) (len 25, spans bins 1=[10,20), 2=[20,30), 3=[30,40) with overlaps 10, 10, 5)
    # [12, 18) (len 6, inside bin 1 with overlap 6)
    # [100, 250) (len 150, nucleosome)
    return FragmentStore(
        starts={"chr1": np.array([10, 12, 100], dtype=np.int32)},
        ends={"chr1": np.array([35, 18, 250], dtype=np.int32)},
        assay="chec",
    )


def test_to_binned_track_coverage_midpoint_and_cuts():
    store = _make_export_store()

    # Coverage conservation: sum(value * bin_size) MUST equal exact total fragment base pairs!
    cov_df = to_binned_track(store, feature="coverage", bin_size=10, normalize=None)
    total_bp = float(np.sum(cov_df["value"] * (cov_df["end"] - cov_df["start"])))
    expected_bp = float(np.sum(store.lengths()))
    assert pytest.approx(total_bp, rel=1e-9) == expected_bp

    # Verify exact partial bin coverage for [10, 35) + [12, 18) in bin [10, 20): (10 + 6) / 10 = 1.6
    bin_10 = cov_df.loc[cov_df["start"] == 10, "value"].iloc[0]
    assert pytest.approx(float(bin_10), rel=1e-9) == 1.6
    # Bin [30, 40) gets 5/10 = 0.5
    bin_30 = cov_df.loc[cov_df["start"] == 30, "value"].iloc[0]
    assert pytest.approx(float(bin_30), rel=1e-9) == 0.5

    # Size-stratified midpoint track using named assay size bin ("mononucleosome": 130-200)
    nuc_df = to_binned_track(
        store, feature="midpoint", len_range="mononucleosome", bin_size=10, normalize=None
    )
    assert len(nuc_df) == 1
    assert int(nuc_df["start"].iloc[0]) == 170  # midpoint of [100, 250) is 175 -> bin [170, 180)
    assert float(nuc_df["value"].iloc[0]) == 1.0


def test_to_bedgraph_plain_and_gzip(tmp_path: Path):
    store = _make_export_store()

    bg = to_bedgraph(store, tmp_path / "track.bedGraph", feature="cuts", bin_size=10, normalize=None, track_name="chec_cuts")
    lines = bg.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("track type=bedGraph")
    # 3 fragments * 2 cuts = 6 total cuts across bins
    total_val = sum(float(line.split("\t")[3]) for line in lines[1:])
    assert pytest.approx(total_val, rel=1e-9) == 6.0

    bg_gz = to_bedgraph(store, tmp_path / "track.bedGraph.gz", feature="midpoint", bin_size=10, normalize="cpm")
    with gzip.open(bg_gz, "rt", encoding="utf-8") as fh:
        gz_lines = fh.read().strip().splitlines()
    assert len(gz_lines) == 3  # midpoints at 22, 15, 175 -> bins 10, 20, 170


def test_to_pymisha_mock(monkeypatch: pytest.MonkeyPatch):
    store = _make_export_store()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    fake_pm = types.ModuleType("pymisha")
    fake_pm.gtrack_create_sparse = lambda *a, **kw: calls.append(("sparse", a, kw))  # type: ignore[attr-defined]
    fake_pm.gtrack_create_dense = lambda *a, **kw: calls.append(("dense", a, kw))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymisha", fake_pm)

    to_pymisha(store, "chec.tf_cuts", feature="cuts", len_range=(1, 100), bin_size=10, mode="sparse", description="TF cuts")
    to_pymisha(store, "chec.nuc_mid", feature="midpoint", len_range="mononucleosome", bin_size=10, mode="dense")

    assert calls[0][0] == "sparse"
    assert calls[0][1][0] == "chec.tf_cuts"
    assert calls[1][0] == "dense"
    assert calls[1][2]["binsize"] == 10


def test_to_bigwig_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = _make_export_store()
    written: dict[str, object] = {}

    class DummyBW:
        def addHeader(self, header):
            written["header"] = header

        def addEntries(self, chroms, starts, ends, values):
            written["entries"] = (chroms, starts, ends, values)

        def close(self):
            written["closed"] = True

    fake_bw = types.ModuleType("pyBigWig")
    fake_bw.open = lambda path, mode: DummyBW()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyBigWig", fake_bw)

    out = to_bigwig(store, tmp_path / "out.bw", chrom_sizes={"chr1": 1000}, feature="midpoint", bin_size=10, normalize=None)
    assert out == tmp_path / "out.bw"
    assert written["header"] == [("chr1", 1000)]
    assert written["closed"] is True
    assert len(written["entries"][1]) == 3  # type: ignore[index]


def test_to_bigwig_warns_about_chromosomes_missing_from_chrom_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = FragmentStore(
        starts={"chr1": np.array([10, 12], dtype=np.int32), "chrUn": np.array([5], dtype=np.int32)},
        ends={"chr1": np.array([35, 18], dtype=np.int32), "chrUn": np.array([60], dtype=np.int32)},
        assay="chec",
    )
    written: dict[str, object] = {}

    class DummyBW:
        def addHeader(self, header):
            pass

        def addEntries(self, chroms, starts, ends, values):
            written.setdefault("chroms", set()).update(chroms)  # type: ignore[union-attr]

        def close(self):
            pass

    fake_bw = types.ModuleType("pyBigWig")
    fake_bw.open = lambda path, mode: DummyBW()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyBigWig", fake_bw)

    with pytest.warns(UserWarning, match=r"missing from chrom_sizes \('chrUn'\)"):
        to_bigwig(store, tmp_path / "o.bw", chrom_sizes={"chr1": 1000}, feature="midpoint", bin_size=10)
    assert written["chroms"] == {"chr1"}

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        to_bigwig(store, tmp_path / "o2.bw", chrom_sizes={"chr1": 1000}, feature="midpoint", bin_size=10, chroms=["chr1"])
