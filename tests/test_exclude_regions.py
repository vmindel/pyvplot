"""Blacklist removal: FragmentStore.exclude_regions and the exclude_regions= ingestion option."""

from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyvplot as vp
from pyvplot.store import FragmentStore, merge_intervals
from test_store import _write_synthetic_paired_bam


def _random_store(n: int = 3000, seed: int = 0) -> FragmentStore:
    rng = np.random.default_rng(seed)
    out_s, out_e = {}, {}
    for chrom in ("chr1", "chr2"):
        s = rng.integers(0, 50_000, n)
        out_s[chrom], out_e[chrom] = s, s + rng.integers(10, 400, n)
    return FragmentStore.from_arrays(out_s, out_e, assay="chec")


def _reference_keep(store: FragmentStore, regions: pd.DataFrame, mode: str) -> dict[str, np.ndarray]:
    """Brute force: per fragment, test every region."""
    keep = {}
    for chrom in store.chroms:
        s, e = store.starts[chrom].astype(int), store.ends[chrom].astype(int)
        hit = np.zeros(len(s), dtype=bool)
        for r in regions[regions["chrom"] == chrom].itertuples():
            if mode == "overlap":
                hit |= (s < r.end) & (e > r.start)
            else:
                mid = (s + e) // 2
                hit |= (mid >= r.start) & (mid < r.end)
        keep[chrom] = ~hit
    return keep


@pytest.mark.parametrize("mode", ["overlap", "midpoint"])
def test_matches_brute_force_with_overlapping_and_adjacent_regions(mode: str):
    store = _random_store()
    rng = np.random.default_rng(1)
    starts = rng.integers(0, 50_000, 60)
    regions = pd.DataFrame(
        {"chrom": rng.choice(["chr1", "chr2", "chrX"], 60), "start": starts, "end": starts + rng.integers(1, 800, 60)}
    )
    # an exactly adjacent pair and a nested one
    regions = pd.concat(
        [regions, pd.DataFrame({"chrom": ["chr1"] * 3, "start": [100, 200, 120], "end": [200, 300, 150]})],
        ignore_index=True,
    )
    out = store.exclude_regions(regions, mode=mode)
    ref = _reference_keep(store, regions, mode)
    for chrom in store.chroms:
        np.testing.assert_array_equal(out.starts[chrom], store.starts[chrom][ref[chrom]])
        np.testing.assert_array_equal(out.ends[chrom], store.ends[chrom][ref[chrom]])
    assert out.n_fragments == sum(int(v.sum()) for v in ref.values())
    assert out.n_fragments < store.n_fragments


def test_overlap_and_midpoint_boundaries():
    store = FragmentStore.from_arrays({"chr1": np.array([0, 100, 200, 300])}, {"chr1": np.array([100, 200, 300, 400])})
    region = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [250]})
    # [100,200) touches the region only at the half-open edge: kept. [200,300) overlaps: dropped.
    assert store.exclude_regions(region).starts["chr1"].tolist() == [0, 100, 300]
    # midpoints are 50, 150, 250, 350: 250 is outside [200, 250)
    assert store.exclude_regions(region, mode="midpoint").starts["chr1"].tolist() == [0, 100, 200, 300]
    region = pd.DataFrame({"chrom": ["chr1"], "start": [199], "end": [201]})
    assert store.exclude_regions(region).starts["chr1"].tolist() == [0, 300]  # one base into each neighbour


def test_bed_path_plain_and_gzip_and_dataframe_aliases(tmp_path: Path):
    store = _random_store()
    text = "# comment\ntrack name=x\nchr1\t1000\t5000\tname\t0\t+\nchr2\t0\t2000\n"
    plain, gz = tmp_path / "bl.bed", tmp_path / "bl.bed.gz"
    plain.write_text(text)
    with gzip.open(gz, "wt") as fh:
        fh.write(text)
    frame = pd.DataFrame({"Chr": ["chr1", "chr2"], "Start": [1000, 0], "END": [5000, 2000]})
    ref = store.exclude_regions(plain)
    for other in (gz, str(plain), frame):
        got = store.exclude_regions(other)
        for chrom in store.chroms:
            np.testing.assert_array_equal(got.starts[chrom], ref.starts[chrom])
    assert ref.n_fragments < store.n_fragments
    assert ref.metadata["exclude_regions"][0]["source"] == str(plain)
    assert store.exclude_regions(frame).metadata["exclude_regions"][0]["source"] == "DataFrame"


def test_input_store_is_untouched_and_metadata_records_the_removal():
    store = _random_store()
    n = store.n_fragments
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10_000]})
    out = store.exclude_regions(regions)
    assert store.n_fragments == n and "exclude_regions" not in store.metadata
    rec = out.metadata["exclude_regions"][0]
    assert rec["n_before"] == n and rec["n_removed"] == n - out.n_fragments > 0 and rec["n_regions"] == 1
    # chromosomes without regions are returned unchanged
    np.testing.assert_array_equal(out.starts["chr2"], store.starts["chr2"])
    # chained calls append
    again = out.exclude_regions(pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [5_000]}))
    assert [r["n_regions"] for r in again.metadata["exclude_regions"]] == [1, 1]


def test_cpm_follows_the_filtered_depth():
    store, anchors = vp.datasets.simulate_chec(n_sites=40, seed=3)
    blacklist = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [int(store.ends["chr1"].max()) // 2]})
    out = store.exclude_regions(blacklist)
    v_all = store.vplot(anchors, flank=100, pos_bin=5, len_bin=5)
    v_out = out.vplot(anchors, flank=100, pos_bin=5, len_bin=5)
    assert v_out.n_fragments == out.n_fragments < v_all.n_fragments


def test_stranded_store_keeps_strands_aligned():
    store = FragmentStore.from_arrays(
        {"chr1": np.array([10, 20, 30, 40])}, {"chr1": np.array([11, 21, 31, 41])},
        assay="chip_exo", strands={"chr1": np.array([1, -1, 1, -1])},
    )
    out = store.exclude_regions(pd.DataFrame({"chrom": ["chr1"], "start": [15], "end": [35]}))
    assert out.starts["chr1"].tolist() == [10, 40]
    assert out.strands is not None and out.strands["chr1"].tolist() == [1, -1]


def test_empty_regions_and_empty_store_are_fine():
    store = _random_store(50)
    empty = pd.DataFrame({"chrom": [], "start": [], "end": []})
    out = store.exclude_regions(empty)
    assert out.n_fragments == store.n_fragments
    nothing = FragmentStore.from_arrays({"chr1": np.array([], dtype=int)}, {"chr1": np.array([], dtype=int)})
    assert nothing.exclude_regions(pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5]})).n_fragments == 0


def test_chromosome_naming_mismatch_warns_but_partial_overlap_does_not():
    store = _random_store(100)
    with pytest.warns(UserWarning, match="chromosome naming"):
        out = store.exclude_regions(pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10**6]}))
    assert out.n_fragments == store.n_fragments
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.exclude_regions(pd.DataFrame({"chrom": ["chr1", "chrUn"], "start": [0, 0], "end": [10, 10]}))


@pytest.mark.parametrize(
    "bad,match",
    [
        (pd.DataFrame({"chrom": ["chr1"], "start": [5]}), "'start' and 'end'"),
        (pd.DataFrame({"start": [5], "end": [9]}), "chrom"),
        (pd.DataFrame({"chrom": ["chr1"], "start": [9], "end": [5]}), "end < start"),
    ],
)
def test_bad_region_tables_are_rejected(bad: pd.DataFrame, match: str):
    with pytest.raises(ValueError, match=match):
        _random_store(10).exclude_regions(bad)
    with pytest.raises(ValueError, match="Invalid mode"):
        _random_store(10).exclude_regions(pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1]}), mode="contained")


def test_bed_header_error_names_the_file(tmp_path: Path):
    bed = tmp_path / "hdr.bed"
    bed.write_text("chrom\tstart\tend\nchr1\t1\t2\n")
    with pytest.raises(ValueError, match=r"hdr\.bed:1"):
        _random_store(10).exclude_regions(bed)


def test_merge_intervals():
    s, e = merge_intervals(np.array([300, 0, 5, 10, 100, 100]), np.array([310, 5, 8, 20, 120, 110]))
    # [0,5) and [5,8) touch, [10,20) is separate, [100,120) absorbs [100,110), [300,310) is separate
    assert s.tolist() == [0, 10, 100, 300] and e.tolist() == [8, 20, 120, 310]
    assert all(len(a) == 0 for a in merge_intervals(np.array([]), np.array([])))


def test_save_load_keeps_the_record(tmp_path: Path):
    out = _random_store().exclude_regions(pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [9000]}))
    loaded = FragmentStore.load(out.save(tmp_path / "s.npz"))
    assert loaded.metadata["exclude_regions"] == out.metadata["exclude_regions"]


def test_from_bam_and_from_fragments_bed_apply_the_blacklist_at_load(tmp_path: Path):
    bam = tmp_path / "t.bam"
    _write_synthetic_paired_bam(bam)
    plain = FragmentStore.from_bam(bam, assay="chec", exclude_chroms={"chrM"})
    chrom = plain.chroms[0]
    lo, hi = int(plain.starts[chrom].min()), int(plain.ends[chrom].max())
    bl = pd.DataFrame({"chrom": [chrom], "start": [lo], "end": [hi]})
    filtered = FragmentStore.from_bam(bam, assay="chec", exclude_chroms={"chrM"}, exclude_regions=bl)
    assert len(filtered.starts[chrom]) == 0 and filtered.n_fragments == plain.n_fragments - len(plain.starts[chrom])
    mid = FragmentStore.from_bam(bam, assay="chec", exclude_chroms={"chrM"}, exclude_regions=bl, exclude_mode="midpoint")
    assert mid.n_fragments == filtered.n_fragments

    frag_bed = tmp_path / "f.bed"
    frag_bed.write_text("chr1\t100\t180\nchr1\t1000\t1100\nchr1\t5000\t5100\n")
    kept = FragmentStore.from_fragments_bed(
        frag_bed, exclude_regions=pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [1050]})
    )
    assert kept.starts["chr1"].tolist() == [100, 5000]
