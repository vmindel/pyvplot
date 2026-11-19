"""examples/bam_to_vplot.py runs end to end on a BAM built from simulated fragments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

import pyvplot as vp

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "examples" / "bam_to_vplot.py"


def _write_bam(store: vp.FragmentStore, path: Path, chrom_len: int) -> None:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": chrom_len}]}
    segs = []
    for i, (s, e) in enumerate(zip(store.starts["chr1"].tolist(), store.ends["chr1"].tolist(), strict=True)):
        rl, tl = min(36, e - s), e - s
        for is_r1, start, rev, tlen, mate in ((True, s, False, tl, e - rl), (False, e - rl, True, -tl, s)):
            a = pysam.AlignedSegment()
            a.query_name = f"q{i}"
            a.query_sequence = "A" * rl
            a.query_qualities = pysam.qualitystring_to_array("I" * rl)
            a.reference_id = a.next_reference_id = 0
            a.mapping_quality = 60
            a.cigar = [(0, rl)]
            a.is_paired = a.is_proper_pair = True
            a.is_read1, a.is_read2 = is_r1, not is_r1
            a.is_reverse, a.mate_is_reverse = rev, not rev
            a.reference_start, a.next_reference_start, a.template_length = start, mate, tlen
            segs.append(a)
    segs.sort(key=lambda r: r.reference_start)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for a in segs:
            out.write(a)


@pytest.mark.filterwarnings("ignore:.*zero expected background:UserWarning")
def test_bam_to_vplot_script(tmp_path):
    sim, anchors = vp.datasets.simulate_chec(n_sites=20, seed=0)
    bam = tmp_path / "sim.bam"
    _write_bam(sim, bam, chrom_len=int(sim.ends["chr1"].max()) + 1000)
    bed = tmp_path / "motifs.bed"
    pd.DataFrame(
        {
            "chrom": anchors["chrom"],
            "start": anchors["center"] - 5,
            "end": anchors["center"] + 5,
            "name": anchors.index,
            "score": 0,
            "strand": anchors["strand"],
        }
    ).to_csv(bed, sep="\t", header=False, index=False)

    spec = importlib.util.spec_from_file_location("bam_to_vplot", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "res" / "sim"
    mod.main(["--bam", str(bam), "--motifs", str(bed), "--out", str(out)])

    for suffix in (".frags.npz", ".qc.json", ".vplot.png", ".vplot.tsv.gz", ".footprint.png",
                   ".footprint.tsv.gz", ".scores.tsv"):
        assert Path(str(out) + suffix).exists(), suffix

    qc = json.loads(Path(str(out) + ".qc.json").read_text())
    assert qc["n_fragments"] == sim.filter(min_len=10, max_len=1000).n_fragments
    tidy = pd.read_csv(str(out) + ".vplot.tsv.gz", sep="\t")
    assert list(tidy.columns) == ["pos", "length", "value"] and tidy["value"].sum() > 0
    scores = pd.read_csv(str(out) + ".scores.tsv", sep="\t", index_col="motif")
    assert list(scores.index) == list(anchors.index)
    assert np.isfinite(scores["protection_score"]).all()

    # Second run reuses the cached store.
    mod.main(["--bam", str(bam), "--motifs", str(bed), "--out", str(out)])


def test_your_data_page_embeds_the_script():
    page = (ROOT / "docs" / "tutorials" / "your-data.md").read_text()
    assert SCRIPT.read_text() in page, "docs/tutorials/your-data.md is out of sync with examples/bam_to_vplot.py"
