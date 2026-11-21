"""Tests for k-mer cleavage bias estimation, matrix correction, and control normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvplot.bias import KmerBiasTable, normalize_by_control, reverse_complement
from pyvplot.matrices import SignalMatrix
from pyvplot.store import FragmentStore


def test_reverse_complement_and_table_loading(tmp_path: Path):
    assert reverse_complement("ACGTNNac") == "gtNNACGT"

    tsv = tmp_path / "bias.tsv"
    tsv.write_text("kmer\tscore\nAAAA\t2.0\nTTTT\t0.5\nACGT\t1.0\n", encoding="utf-8")
    tbl = KmerBiasTable.from_table(tsv, normalize=False)
    assert tbl.k == 4
    assert tbl["AAAA"] == 2.0
    assert tbl["TTTT"] == 0.5
    assert tbl["ACGT"] == 1.0


def test_empirical_kmer_bias_estimation_from_cuts():
    """Plant a strong 4-mer cleavage preference at 'GATC' and verify KmerBiasTable.from_cuts recovers it."""
    rng = np.random.default_rng(42)
    bases = np.array(list("ACGT"))
    seq = "".join(rng.choice(bases, size=20_000))

    # Find every occurrence of 'GATC' and place heavy cleavage cuts at its center (offset 2)
    starts: list[int] = []
    ends: list[int] = []
    for i in range(50, len(seq) - 200):
        if seq[i : i + 4] == "GATC":
            cut_pos = i + 2
            for _ in range(15):
                starts.append(cut_pos)
                ends.append(cut_pos + 80)

    # Also add uniform background cuts
    bg_starts = rng.integers(50, len(seq) - 200, size=1000)
    starts.extend(bg_starts.tolist())
    ends.extend((bg_starts + 80).tolist())

    store = FragmentStore(
        starts={"chr1": np.asarray(starts, dtype=np.int32)},
        ends={"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )

    bias = KmerBiasTable.from_cuts(store, {"chr1": seq}, k=4, cut_offset=2, pseudocount=1.0)
    # 'GATC' (palindromic!) must have the highest cleavage bias weight of all 256 4-mers
    gatc_score = bias["GATC"]
    assert gatc_score == float(np.max(bias.weights))
    assert gatc_score > 3.0


def test_bias_correction_removes_sequence_artifact():
    # Construct a 4-mer bias table where 'GATC' has 4x higher cleavage propensity
    bias = KmerBiasTable.from_dict({"GATC": 4.0}, cut_offset=2, normalize=False)

    # Sequence of length 20 with 'GATC' at positions 8..12 (cut_offset=2 -> base 10 has bias 4.0)
    seq = "AAAAAAAA" + "GATC" + "AAAAAAAA"
    exp = bias.score_sequence(seq)
    assert exp[10] == 4.0
    assert exp[5] == 1.0

    # Observed signal is purely proportional to bias (no true TF signal: 4.0 at base 10, 1.0 elsewhere)
    mat = SignalMatrix(
        values=exp[None, :].copy(),
        region_index=np.array(["a0"]),
        upstream=10,
        downstream=10,
        bin_size=1,
        feature="cuts_plus",
    )

    corrected = bias.correct_matrix(mat, [seq], preserve_scale=True)
    # After bias correction, all 20 positions should be flat (uniform)!
    assert np.std(corrected.values[0]) < 1e-4
    # Total signal sum preserved
    assert pytest.approx(float(np.sum(corrected.values)), rel=1e-6) == float(np.sum(mat.values))


def test_normalize_by_control_matrix_and_array():
    m_sig = SignalMatrix(
        values=np.array([[10.0, 20.0], [5.0, 5.0]]),
        region_index=np.array([0, 1]),
        upstream=10,
        downstream=10,
        bin_size=10,
    )
    m_ctrl = SignalMatrix(
        values=np.array([[5.0, 5.0], [5.0, 5.0]]),
        region_index=np.array([0, 1]),
        upstream=10,
        downstream=10,
        bin_size=10,
    )

    res_log2 = normalize_by_control(m_sig, m_ctrl, mode="log2fc", pseudocount=0.0)
    assert isinstance(res_log2, SignalMatrix)
    np.testing.assert_allclose(res_log2.values, [[1.0, 2.0], [0.0, 0.0]])

    arr_sub = normalize_by_control(m_sig.values, m_ctrl.values, mode="subtract")
    np.testing.assert_allclose(arr_sub, [[5.0, 15.0], [0.0, 0.0]])


def _random_genome(n: int = 300_000, seed: int = 0) -> str:
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n))


def _boundaries_with_dinucleotide(seq: str, dinuc: str) -> np.ndarray:
    """Boundaries b (between base b-1 and b) whose flanking bases read ``dinuc`` on the forward strand."""
    return np.array([b for b in range(1, len(seq)) if seq[b - 1 : b + 1] == dinuc])


def test_from_cuts_learns_right_end_preference_in_reverse_frame():
    """A cutter that only cuts at AC (reading 5'->3') cuts forward GT on the opposite strand.

    Left ends at forward-AC boundaries and right ends at forward-GT boundaries must both
    teach the table that AC is preferred.
    """
    rng = np.random.default_rng(1)
    seq = _random_genome()
    lens = lambda n: rng.integers(40, 120, n)  # noqa: E731

    left = np.sort(_boundaries_with_dinucleotide(seq, "AC")[:15_000])
    left_store = FragmentStore.from_arrays({"c": left}, {"c": left + lens(len(left))})
    right = np.sort(_boundaries_with_dinucleotide(seq, "GT")[:15_000])
    right_store = FragmentStore.from_arrays({"c": right - lens(len(right))}, {"c": right})

    for store in (left_store, right_store):
        table = KmerBiasTable.from_cuts(store, {"c": seq}, k=2, cut_offset=1)
        assert table["AC"] == float(np.max(table.weights))
        assert table["AC"] > 5.0


def test_from_cuts_stranded_minus_reads_use_their_own_base():
    """Stranded 5' ends: a minus read whose 5' base sits right after an RC-frame AC cut."""
    seq = _random_genome(seed=3)
    # Forward-strand AC boundary b: the cut is before base b. Minus-strand AC means forward
    # GT at (b-1, b); the minus read's 5' base is b-1.
    pos = np.sort(_boundaries_with_dinucleotide(seq, "GT")[:15_000] - 1)
    store = FragmentStore.from_arrays(
        {"c": pos}, {"c": pos + 1}, assay="chip_exo", strands={"c": np.full(len(pos), -1)}
    )
    table = KmerBiasTable.from_cuts(store, {"c": seq}, k=2, cut_offset=1)
    assert table["AC"] == float(np.max(table.weights))


def _oriented_scores(table: KmerBiasTable, seq: str, frame: str) -> np.ndarray:
    return table.expected_matrix([seq], frame=frame)[0]


@pytest.mark.parametrize("feature,frame", [("cuts_plus", "forward"), ("cuts_minus", "reverse"), ("cuts", "both")])
def test_correct_matrix_flattens_its_own_expectation_per_feature(feature: str, frame: str):
    rng = np.random.default_rng(4)
    seq = "".join(rng.choice(list("ACGT"), 60))
    table = KmerBiasTable.from_dict(
        {a + b: float(1 + 3 * (a + b == "AC") + 2 * (a + b == "GT")) for a in "ACGT" for b in "ACGT"},
        cut_offset=1,
    )
    exp = _oriented_scores(table, seq, frame)
    assert exp.std() > 0.1
    mat = SignalMatrix(
        values=exp[None, :].copy(), region_index=np.array(["a"]), upstream=30, downstream=30,
        bin_size=1, feature=feature,
    )
    out = table.correct_matrix(mat, [seq])
    assert np.std(out.values[0]) < 1e-4


def test_reverse_frame_matches_cuts_of_the_opposite_strand():
    """Bin i of a right-end profile scores the reverse-strand k-mer across boundary i."""
    table = KmerBiasTable.from_dict({"AC": 5.0}, cut_offset=1, normalize=False)
    seq = "TTTTGTTTT"  # forward 'GT' spans boundary 5 (bases 4,5); RC reading gives 'AC'
    exp = table.score_sequence_reverse(seq)
    assert exp[5] == 5.0
    assert np.count_nonzero(exp != 1.0) == 1
    # a '-' anchor sees the same sequence reverse-complemented
    assert table.score_sequence_reverse(reverse_complement(seq), strand="-")[5] == 5.0


def test_correct_matrix_rejects_non_cut_features():
    table = KmerBiasTable.from_dict({"AC": 2.0}, cut_offset=1)
    mat = SignalMatrix(
        values=np.ones((1, 10)), region_index=np.array(["a"]), upstream=5, downstream=5,
        bin_size=1, feature="midpoint",
    )
    with pytest.raises(ValueError, match="cut-site"):
        table.correct_matrix(mat, ["A" * 10])


def _depth_scaled_vplots():
    from pyvplot.vplot import VPlot

    rng = np.random.default_rng(0)
    base = rng.poisson(40, (6, 8)).astype(float)
    kw = dict(pos_edges=np.arange(9.0), len_edges=np.arange(7.0), n_anchors=1_000)
    sig = base.copy()
    sig[:, :4] *= 4  # a true 4x gain in the left half
    # 50M-fragment libraries, so one raw count is ~2e-5 CPM per anchor
    return (
        VPlot(counts=sig, n_fragments=50_000_000, **kw),
        VPlot(counts=base, n_fragments=50_000_000, **kw),
    )


def test_normalize_by_control_default_pseudocount_is_scale_aware():
    sig, ctrl = _depth_scaled_vplots()
    raw = normalize_by_control(sig, ctrl, mode="log2fc")
    cpm = normalize_by_control(sig.normalize("cpm"), ctrl.normalize("cpm"), mode="log2fc")
    # Same biology whatever the units: the default must not depend on them
    np.testing.assert_allclose(cpm.counts, raw.counts, atol=0.05)
    assert 1.5 < np.median(raw.counts[:, :4]) < 2.5
    # An explicit raw-count pseudocount swamps CPM-scaled values; that is what the default avoids
    swamped = normalize_by_control(sig.normalize("cpm"), ctrl.normalize("cpm"), mode="log2fc", pseudocount=1.0)
    assert np.abs(swamped.counts).max() < 0.01


def test_normalize_by_control_rejects_mixed_units():
    sig, ctrl = _depth_scaled_vplots()
    with pytest.raises(ValueError, match="different units"):
        normalize_by_control(sig.normalize("cpm"), ctrl)
    m = SignalMatrix(values=np.ones((1, 4)), region_index=np.array([0]), upstream=2, downstream=2,
                     bin_size=1, n_fragments=1_000_000)
    with pytest.raises(ValueError, match="different units"):
        normalize_by_control(m.normalize("cpm"), m)
    # matrices: default pseudocount is half a raw count in the matrix's units
    cpm = m.normalize("cpm")
    out = normalize_by_control(cpm, cpm, mode="ratio")
    np.testing.assert_allclose(out.values, 1.0)


def test_kmer_counting_is_independent_of_chunk_size():
    from pyvplot.bias import _count_kmers

    rng = np.random.default_rng(5)
    seq = "".join(rng.choice(list("ACGTN"), 20_000, p=[0.24, 0.24, 0.24, 0.24, 0.04]))
    k_starts = rng.integers(-10, 20_010, 5_000)
    ref_bg, ref_obs = _count_kmers(seq, 3, k_starts, chunk=10**9)
    for chunk in (1, 7, 999, 8_192):
        bg, obs = _count_kmers(seq, 3, k_starts, chunk=chunk)
        np.testing.assert_array_equal(bg, ref_bg)
        np.testing.assert_array_equal(obs, ref_obs)
    # k-mers touching N are skipped, and out-of-range starts are ignored
    assert ref_bg.sum() < len(seq) - 2
    assert ref_obs.sum() <= ((k_starts >= 0) & (k_starts < len(seq) - 2)).sum()
