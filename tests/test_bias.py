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
        feature="cuts",
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
