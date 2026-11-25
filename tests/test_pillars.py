"""Tests for multi-length cut-pillar and de-novo motif footprint scanning and pygenomic_tracks DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

import pyvplot as vp


def _make_synthetic_chec_store() -> vp.FragmentStore:
    """Create a synthetic paired-end ChEC-seq locus with two protected 16-bp motifs
    at chr1:1200-1216 and chr1:1260-1276 flanked by multi-length cleavage pillars.
    """
    rng = np.random.default_rng(42)
    starts = []
    ends = []

    # Motif 1 at [1200, 1216) -> left pillar at 1199, right pillar at 1216
    # Motif 2 at [1260, 1276) -> left pillar at 1259, right pillar at 1276
    pillar_positions = [1199, 1216, 1259, 1276]

    # 1. Spanning short TF fragments across Motif 1 ([1199, 1216]) and Motif 2 ([1259, 1276])
    # and inter-motif fragments [1199, 1276], [1216, 1259]
    for _ in range(80):
        starts.append(1199 + int(rng.integers(-1, 2)))
        ends.append(1259 + int(rng.integers(-1, 2)))
    for _ in range(80):
        starts.append(1216 + int(rng.integers(-1, 2)))
        ends.append(1276 + int(rng.integers(-1, 2)))
    for _ in range(60):
        starts.append(1199 + int(rng.integers(-1, 2)))
        ends.append(1276 + int(rng.integers(-1, 2)))

    # 2. Multi-length cuts anchored at each pillar across lengths 25..220 bp
    for p in pillar_positions:
        for fl in range(25, 225, 10):
            for _ in range(4):
                # Left-anchored fragment starting at p (unless it cuts inside motif core)
                if p in (1199, 1259):
                    starts.append(p - fl + int(rng.integers(-1, 2)))
                    ends.append(p + int(rng.integers(-1, 2)))
                else:
                    starts.append(p + int(rng.integers(-1, 2)))
                    ends.append(p + fl + int(rng.integers(-1, 2)))

    return vp.FragmentStore.from_arrays(
        {"chr1": np.array(starts, dtype=np.int32)},
        {"chr1": np.array(ends, dtype=np.int32)},
        assay="chec",
    )


def test_scan_locus_pillars_and_track_dfs():
    store = _make_synthetic_chec_store()
    scan = store.scan_locus_pillars("chr1:1100-1400")

    assert scan.chrom == "chr1"
    assert scan.start == 1100
    assert scan.end == 1400
    assert len(scan.pos) == 300
    assert scan.protection_2d.shape == (len(scan.footprint_widths), 300)

    # Check called pillars include the 4 synthetic pillar sites (within +-2 bp)
    assert len(scan.pillars) >= 4
    assert {"chrom", "start", "end", "pos", "score", "strand", "pillar_score", "n_len_bins"}.issubset(
        scan.pillars.columns
    )
    for target_p in (1199, 1216, 1259, 1276):
        assert np.min(np.abs(scan.pillars["pos"].to_numpy() - target_p)) <= 2

    # Check called motifs recover the two protected troughs centered near 1208 and 1268
    assert len(scan.motifs) >= 2
    assert {"chrom", "start", "end", "center", "width", "name", "score", "strand", "protection_score"}.issubset(
        scan.motifs.columns
    )
    centers = scan.motifs["center"].to_numpy()
    assert np.min(np.abs(centers - 1208)) <= 3
    assert np.min(np.abs(centers - 1268)) <= 3

    # Check pygenomic_tracks DataFrame outputs
    tracks = scan.to_pygenomic_tracks(bin_size=1)
    for key in ("pillar_score", "protection_score", "raw_log2_protection", "tf_coverage", "cuts", "pillars", "motifs"):
        df = tracks[key]
        assert isinstance(df, pd.DataFrame)
        assert {"chrom", "start", "end"}.issubset(df.columns)
        assert (df["end"] > df["start"]).all()

    # Check binned track and drop_zeros
    binned = scan.to_track_df("protection_score", bin_size=10, drop_zeros=True)
    assert len(binned) > 0
    assert list(binned.columns) == ["chrom", "start", "end", "value", "score"]
    assert (binned["value"] != 0.0).all()


def test_scan_genome_pillars():
    store = _make_synthetic_chec_store()
    motifs_df, pillars_df = store.scan_genome_pillars(return_pillars=True, min_bin_frags=5)

    assert len(motifs_df) >= 2
    assert len(pillars_df) >= 4
    assert {"chrom", "start", "end", "center", "width", "protection_score", "composite_score"}.issubset(
        motifs_df.columns
    )
    assert {"chrom", "start", "end", "pos", "pillar_score", "n_len_bins"}.issubset(pillars_df.columns)
