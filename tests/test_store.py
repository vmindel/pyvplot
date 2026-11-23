"""Tests for FragmentStore ingestion (BAM, BED, BEDPE), caching (.npz, .parquet), and filtering."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from pyvplot.store import FragmentStore


def _write_synthetic_paired_bam(path: Path) -> None:
    """Create a synthetic paired-end BAM containing forward-R1, reverse-R1, low-MAPQ, dup, and chrM pairs."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [
            {"SN": "chr1", "LN": 100_000},
            {"SN": "chr2", "LN": 100_000},
            {"SN": "chrM", "LN": 16_571},
        ],
    }

    def _make_pair(
        qname: str,
        tid: int,
        frag_start: int,
        frag_end: int,
        r1_reverse: bool = False,
        mapq: int = 60,
        is_dup: bool = False,
        is_secondary: bool = False,
    ) -> list[pysam.AlignedSegment]:
        tlen = frag_end - frag_start
        read_len = min(35, tlen)

        r1 = pysam.AlignedSegment()
        r1.query_name = qname
        r1.query_sequence = "A" * read_len
        r1.query_qualities = pysam.qualitystring_to_array("I" * read_len)
        r1.reference_id = tid
        r1.next_reference_id = tid
        r1.mapping_quality = mapq
        r1.cigar = [(0, read_len)]
        r1.is_paired = True
        r1.is_proper_pair = True
        r1.is_read1 = True
        r1.is_duplicate = is_dup
        r1.is_secondary = is_secondary

        r2 = pysam.AlignedSegment()
        r2.query_name = qname
        r2.query_sequence = "A" * read_len
        r2.query_qualities = pysam.qualitystring_to_array("I" * read_len)
        r2.reference_id = tid
        r2.next_reference_id = tid
        r2.mapping_quality = mapq
        r2.cigar = [(0, read_len)]
        r2.is_paired = True
        r2.is_proper_pair = True
        r2.is_read2 = True
        r2.is_duplicate = is_dup
        r2.is_secondary = is_secondary

        if not r1_reverse:
            r1.is_reverse = False
            r1.mate_is_reverse = True
            r1.reference_start = frag_start
            r1.next_reference_start = frag_end - read_len
            r1.template_length = tlen

            r2.is_reverse = True
            r2.mate_is_reverse = False
            r2.reference_start = frag_end - read_len
            r2.next_reference_start = frag_start
            r2.template_length = -tlen
        else:
            r1.is_reverse = True
            r1.mate_is_reverse = False
            r1.reference_start = frag_end - read_len
            r1.next_reference_start = frag_start
            r1.template_length = -tlen

            r2.is_reverse = False
            r2.mate_is_reverse = True
            r2.reference_start = frag_start
            r2.next_reference_start = frag_end - read_len
            r2.template_length = tlen

        return [r1, r2]

    segments: list[pysam.AlignedSegment] = []
    # 1. chr1 forward R1: [1000, 1100)
    segments.extend(_make_pair("q1", 0, 1000, 1100, r1_reverse=False))
    # 2. chr1 reverse R1: [2000, 2150)
    segments.extend(_make_pair("q2", 0, 2000, 2150, r1_reverse=True))
    # 3. chr1 exact coordinate duplicate of q1 (without SAM dup flag)
    segments.extend(_make_pair("q3", 0, 1000, 1100, r1_reverse=False, is_dup=False))
    # 4. chr1 SAM-flagged duplicate: [3000, 3080)
    segments.extend(_make_pair("q4", 0, 3000, 3080, r1_reverse=False, is_dup=True))
    # 5. chr1 low MAPQ (should be filtered at min_mapq=20)
    segments.extend(_make_pair("q5", 0, 4000, 4120, mapq=5))
    # 6. chr1 secondary alignment (should always be excluded)
    segments.extend(_make_pair("q6", 0, 5000, 5120, is_secondary=True))
    # 7. chr2 short fragment [500, 560)
    segments.extend(_make_pair("q7", 1, 500, 560, r1_reverse=True))
    # 8. chrM fragment [100, 250)
    segments.extend(_make_pair("q8", 2, 100, 250, r1_reverse=False))

    segments.sort(key=lambda r: (r.reference_id, r.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for seg in segments:
            out.write(seg)


def _write_synthetic_chip_exo_bam(path: Path) -> None:
    """Create a synthetic single-end ChIP-exo BAM with + and - strand reads."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 50_000}],
    }
    reads = []
    # Forward read covering [1000, 1036) -> 5' end at 1000 (+1)
    r_plus = pysam.AlignedSegment()
    r_plus.query_name = "exo_plus"
    r_plus.query_sequence = "A" * 36
    r_plus.query_qualities = pysam.qualitystring_to_array("I" * 36)
    r_plus.reference_id = 0
    r_plus.reference_start = 1000
    r_plus.cigar = [(0, 36)]
    r_plus.mapping_quality = 60
    r_plus.is_reverse = False
    reads.append(r_plus)

    # Reverse read covering [1020, 1056) -> 5' end at 1055 (-1)
    r_minus = pysam.AlignedSegment()
    r_minus.query_name = "exo_minus"
    r_minus.query_sequence = "A" * 36
    r_minus.query_qualities = pysam.qualitystring_to_array("I" * 36)
    r_minus.reference_id = 0
    r_minus.reference_start = 1020
    r_minus.cigar = [(0, 36)]
    r_minus.mapping_quality = 60
    r_minus.is_reverse = True
    reads.append(r_minus)

    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads:
            out.write(r)


def test_from_bam_paired_chec_and_atac_shifts(tmp_path: Path):
    bam_path = tmp_path / "paired.bam"
    _write_synthetic_paired_bam(bam_path)

    # ChEC-seq (no shift, no dedup, exclude chrM)
    chec_r1 = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=False, exclude_chroms={"chrM"}, pairing="read1"
    )
    chec_mate = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=False, exclude_chroms={"chrM"}, pairing="mate"
    )

    # Both pairing strategies must agree exactly
    assert chec_r1.chroms == chec_mate.chroms
    for c in chec_r1.chroms:
        np.testing.assert_array_equal(chec_r1.starts[c], chec_mate.starts[c])
        np.testing.assert_array_equal(chec_r1.ends[c], chec_mate.ends[c])

    # Expect q1 (1000-1100), q3 (1000-1100), q2 (2000-2150), q4 (3000-3080) on chr1; q7 (500-560) on chr2
    assert "chrM" not in chec_r1.starts
    np.testing.assert_array_equal(chec_r1.starts["chr1"], [1000, 1000, 2000, 3000])
    np.testing.assert_array_equal(chec_r1.ends["chr1"], [1100, 1100, 2150, 3080])
    np.testing.assert_array_equal(chec_r1.starts["chr2"], [500])
    np.testing.assert_array_equal(chec_r1.ends["chr2"], [560])
    assert chec_r1.max_len["chr1"] == 150
    assert chec_r1.n_fragments == 5

    # With dedup=True: drops q4 (SAM 0x400) AND drops coordinate-duplicate q3
    chec_dedup = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=True, exclude_chroms={"chrM"}
    )
    np.testing.assert_array_equal(chec_dedup.starts["chr1"], [1000, 2000])
    np.testing.assert_array_equal(chec_dedup.ends["chr1"], [1100, 2150])

    # ATAC-seq applies (+4, -5) shift once during ingestion
    atac = FragmentStore.from_bam(
        bam_path, assay="atac", min_mapq=20, dedup=True, exclude_chroms={"chrM"}
    )
    np.testing.assert_array_equal(atac.starts["chr1"], [1004, 2004])
    np.testing.assert_array_equal(atac.ends["chr1"], [1095, 2145])
    np.testing.assert_array_equal(atac.starts["chr2"], [504])
    np.testing.assert_array_equal(atac.ends["chr2"], [555])


def test_from_bam_chip_exo_single_end(tmp_path: Path):
    bam_path = tmp_path / "exo.bam"
    _write_synthetic_chip_exo_bam(bam_path)

    store = FragmentStore.from_bam(bam_path, assay="chip_exo", min_mapq=20)
    assert store.is_stranded is True
    assert store.n_fragments == 2
    np.testing.assert_array_equal(store.starts["chr1"], [1000, 1055])
    np.testing.assert_array_equal(store.ends["chr1"], [1001, 1056])
    assert store.strands is not None
    np.testing.assert_array_equal(store.strands["chr1"], [1, -1])


def test_from_fragments_bed_tsv_gz_and_bedpe(tmp_path: Path):
    # 10x / ArchR style .frag.tsv.gz
    frag_path = tmp_path / "sample.frag.tsv.gz"
    with gzip.open(frag_path, "wt", encoding="utf-8") as fh:
        fh.write("# header comment\n")
        fh.write("chr1\t500\t600\tBARCODE1\t2\n")
        fh.write("chr1\t200\t280\tBARCODE2\t1\n")
        fh.write("chr2\t1000\t1200\tBARCODE1\t3\n")

    s_nocount = FragmentStore.from_fragments_bed(frag_path, assay="atac", use_count=False)
    assert s_nocount.n_fragments == 3
    np.testing.assert_array_equal(s_nocount.starts["chr1"], [200, 500])
    np.testing.assert_array_equal(s_nocount.ends["chr1"], [280, 600])

    s_count = FragmentStore.from_fragments_bed(frag_path, assay="atac", use_count=True)
    assert s_count.n_fragments == 6
    np.testing.assert_array_equal(s_count.starts["chr1"], [200, 500, 500])

    # BEDPE (auto-detected)
    bedpe_path = tmp_path / "pairs.bedpe"
    bedpe_path.write_text(
        "chr1\t100\t150\tchr1\t200\t250\treadA\t60\t+\t-\n"
        "chr1\t400\t450\tchr1\t300\t350\treadB\t60\t-\t+\n"
        "chr1\t100\t150\tchr2\t200\t250\tdiscordant\t60\t+\t-\n",
        encoding="utf-8",
    )
    s_bedpe = FragmentStore.from_fragments_bed(bedpe_path, assay="chec")
    assert s_bedpe.n_fragments == 2
    np.testing.assert_array_equal(s_bedpe.starts["chr1"], [100, 300])
    np.testing.assert_array_equal(s_bedpe.ends["chr1"], [250, 450])


@pytest.mark.parametrize("fmt", ["npz", "npz_compressed", "parquet"])
def test_save_and_load_roundtrip(tmp_path: Path, fmt: str):
    store = FragmentStore(
        starts={"chr1": np.array([300, 100, 100]), "chr2": np.array([50])},
        ends={"chr1": np.array([450, 180, 220]), "chr2": np.array([150])},
        assay="chec",
        metadata={"sample": "WT_rep1"},
    )

    if fmt == "npz":
        out = store.save(tmp_path / "cache.npz", compressed=False)
    elif fmt == "npz_compressed":
        out = store.save(tmp_path / "cache_c.npz", compressed=True)
    else:
        out = store.save(tmp_path / "cache.parquet")

    loaded = FragmentStore.load(out)
    assert loaded.assay == store.assay
    assert loaded.n_fragments == store.n_fragments
    assert loaded.chroms == store.chroms
    assert loaded.metadata == store.metadata
    for c in store.chroms:
        np.testing.assert_array_equal(loaded.starts[c], store.starts[c])
        np.testing.assert_array_equal(loaded.ends[c], store.ends[c])
        assert loaded.max_len[c] == store.max_len[c]


@pytest.mark.parametrize("ext", [".npz", ".parquet"])
def test_save_and_load_stranded_roundtrip(tmp_path: Path, ext: str):
    store = FragmentStore(
        starts={"chr1": np.array([100, 200])},
        ends={"chr1": np.array([101, 201])},
        strands={"chr1": np.array([1, -1], dtype=np.int8)},
        assay="chip_exo",
    )
    out = store.save(tmp_path / f"exo{ext}")
    loaded = FragmentStore.load(out)
    assert loaded.is_stranded is True
    assert loaded.strands is not None
    np.testing.assert_array_equal(loaded.strands["chr1"], store.strands["chr1"])


def test_filter_merge_and_dataframe_conversion():
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "start": [100, 200, 300, 50],
            "end": [150, 360, 550, 100],
        }
    )
    store = FragmentStore.from_dataframe(df, assay="chec")
    assert store.n_fragments == 4

    tf_only = store.filter(min_len=20, max_len=120, chroms={"chr1"})
    assert tf_only.chroms == ["chr1"]
    assert tf_only.n_fragments == 1
    np.testing.assert_array_equal(tf_only.lengths(), [50])

    merged = FragmentStore.merge([tf_only, store.filter(chroms={"chr2"})])
    assert merged.n_fragments == 2
    out_df = merged.to_dataframe()
    assert list(out_df.columns) == ["chrom", "start", "end"]
    assert len(out_df) == 2


@pytest.mark.parametrize("name", ["cache.foo", "cache"])
def test_save_explicit_npz_returns_an_existing_path(tmp_path: Path, name: str):
    store = FragmentStore(starts={"chr1": np.array([1, 5])}, ends={"chr1": np.array([10, 20])})
    out = store.save(tmp_path / name, format="npz")
    assert out.exists() and out.suffix == ".npz"
    assert FragmentStore.load(out).n_fragments == 2
    assert [p.name for p in tmp_path.iterdir()] == [out.name]


def test_from_bam_threads_do_not_change_the_result(tmp_path: Path):
    bam = tmp_path / "t.bam"
    _write_synthetic_paired_bam(bam)
    one = FragmentStore.from_bam(bam, assay="chec", exclude_chroms={"chrM"})
    four = FragmentStore.from_bam(bam, assay="chec", exclude_chroms={"chrM"}, threads=4)
    assert one.chroms == four.chroms and one.n_fragments > 0
    for c in one.chroms:
        np.testing.assert_array_equal(one.starts[c], four.starts[c])
        np.testing.assert_array_equal(one.ends[c], four.ends[c])


def test_chrom_buffers_are_compact_and_exact():
    from pyvplot.store import _ChromBuffers

    buf = _ChromBuffers(stranded=True)
    buf.add("chr1", 5, 6, -1)
    buf.add("chr1", 7, 8, 1, count=3)
    buf.add("chr2", 2_000_000_000, 2_000_000_001, 1)
    starts, ends, strands = buf.to_numpy()
    assert starts["chr1"].dtype == np.int32 and strands is not None
    assert starts["chr1"].tolist() == [5, 7, 7, 7] and ends["chr1"].tolist() == [6, 8, 8, 8]
    assert strands["chr1"].tolist() == [-1, 1, 1, 1] and starts["chr2"][0] == 2_000_000_000
    assert buf.starts["chr1"].itemsize == 4  # not a list of Python ints
    with pytest.raises(OverflowError):
        buf.add("chr3", 2**31, 2**31 + 1)


def test_from_bam_ingest_stats_read1(tmp_path: Path):
    bam_path = tmp_path / "paired.bam"
    _write_synthetic_paired_bam(bam_path)
    store = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=False, exclude_chroms={"chrM"}, pairing="read1"
    )
    st = store.metadata["ingest_stats"]
    # 8 pairs = 16 records: q6 secondary (2 flag), 7 mate-2 records of the rest (pairing),
    # q5 R1 low MAPQ, q8 R1 on chrM, and 5 fragments parsed.
    assert st["records_read"] == 16
    assert st["skipped_flag"] == 2
    assert st["skipped_pairing"] == 7
    assert st["skipped_mapq"] == 1
    assert st["skipped_chrom"] == 1
    assert st["skipped_length"] == 0
    assert st["fragments_parsed"] == 5
    assert st["removed_coord_dedup"] == 0 and st["removed_blacklist"] == 0
    assert st["fragments_kept"] == store.n_fragments == 5
    assert "seconds" not in st  # only timed when verbose
    skipped = sum(v for k, v in st.items() if k.startswith("skipped_"))
    assert skipped + st["fragments_parsed"] == st["records_read"]


def test_from_bam_ingest_stats_mate_dedup_blacklist(tmp_path: Path):
    bam_path = tmp_path / "paired.bam"
    _write_synthetic_paired_bam(bam_path)
    mate = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=False, exclude_chroms={"chrM"}, pairing="mate"
    )
    st = mate.metadata["ingest_stats"]
    assert (st["skipped_flag"], st["skipped_pairing"], st["skipped_mapq"], st["skipped_chrom"]) == (2, 0, 2, 2)
    assert st["fragments_parsed"] == 5 and st["unpaired_mates"] == 0

    blacklist = pd.DataFrame({"chrom": ["chr1"], "start": [1990], "end": [2010]})
    dd = FragmentStore.from_bam(
        bam_path, assay="chec", min_mapq=20, dedup=True, exclude_chroms={"chrM"},
        exclude_regions=blacklist,
    )
    st = dd.metadata["ingest_stats"]
    assert st["skipped_flag"] == 4  # q6 secondary + q4 SAM duplicate, both mates
    assert st["fragments_parsed"] == 4  # q1, q2, q3, q7
    assert st["removed_coord_dedup"] == 1  # q3 duplicates q1
    assert st["removed_blacklist"] == 1  # q2 [2000, 2150) overlaps the region
    assert st["fragments_kept"] == dd.n_fragments == 2


def test_from_bam_verbose_reports_and_matches_silent(tmp_path: Path, capsys):
    bam_path = tmp_path / "paired.bam"
    _write_synthetic_paired_bam(bam_path)
    kw = dict(assay="chec", min_mapq=20, dedup=False, exclude_chroms={"chrM"})
    quiet = FragmentStore.from_bam(bam_path, **kw)
    assert capsys.readouterr().err == ""
    loud = FragmentStore.from_bam(bam_path, verbose=True, **kw)
    err = capsys.readouterr().err
    assert "[pyvplot.from_bam" in err
    assert "now on chr1" in err and "now on chr2" in err
    assert "records read" in err and "fragments kept" in err and "skipped: MAPQ" in err
    for c in quiet.chroms:
        np.testing.assert_array_equal(quiet.starts[c], loud.starts[c])
        np.testing.assert_array_equal(quiet.ends[c], loud.ends[c])
    assert loud.metadata["ingest_stats"]["seconds"] >= 0

    loud.save(tmp_path / "s.frags.npz")
    back = FragmentStore.load(tmp_path / "s.frags.npz")
    assert back.metadata["ingest_stats"]["records_read"] == 16


def test_from_bam_ingest_stats_chip_exo(tmp_path: Path):
    bam_path = tmp_path / "exo.bam"
    _write_synthetic_chip_exo_bam(bam_path)
    store = FragmentStore.from_bam(bam_path, assay="chip_exo", min_mapq=20, verbose=True)
    st = store.metadata["ingest_stats"]
    assert st["fragments_kept"] == store.n_fragments
    skipped = sum(v for k, v in st.items() if k.startswith("skipped_"))
    assert skipped + st["fragments_parsed"] == st["records_read"]


def test_merge_keeps_input_metadata(tmp_path: Path):
    bam_path = tmp_path / "paired.bam"
    _write_synthetic_paired_bam(bam_path)
    blacklist = pd.DataFrame({"chrom": ["chr1"], "start": [1990], "end": [2010]})
    a = FragmentStore.from_bam(bam_path, assay="chec", exclude_chroms={"chrM"})
    b = FragmentStore.from_bam(bam_path, assay="chec", exclude_chroms={"chrM"}, exclude_regions=blacklist)
    merged = FragmentStore.merge([a, b], dedup=True)
    inputs = merged.metadata["merged_inputs"]
    assert [m["ingest_stats"]["fragments_kept"] for m in inputs] == [5, 4]
    assert "exclude_regions" not in inputs[0] and inputs[1]["exclude_regions"][0]["n_removed"] == 1
    assert merged.metadata["merged_from"] == [str(bam_path), str(bam_path)]
    assert merged.metadata["removed_coord_dedup"] == 9 - merged.n_fragments
    merged.save(tmp_path / "m.frags.npz")
    back = FragmentStore.load(tmp_path / "m.frags.npz")
    assert back.metadata["merged_inputs"][1]["ingest_stats"]["removed_blacklist"] == 1
    assert "removed_coord_dedup" not in FragmentStore.merge([a, b]).metadata


def test_from_bam_verbose_name_sorted_does_not_log_every_switch(tmp_path: Path, capsys):
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6", "SO": "queryname"}, "SQ": [{"SN": c, "LN": 100_000} for c in ("chr1", "chr2")]}
    bam_path = tmp_path / "interleaved.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out:
        for i in range(200):
            tid, s = i % 2, 1000 + 10 * i  # chromosome alternates every pair
            for r1 in (True, False):
                a = pysam.AlignedSegment()
                a.query_name = f"q{i:04d}"
                a.reference_id = a.next_reference_id = tid
                a.reference_start, a.next_reference_start = (s, s + 100) if r1 else (s + 100, s)
                a.cigarstring = "50M"
                a.query_sequence = "A" * 50
                a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                a.mapping_quality = 40
                a.template_length = 150 if r1 else -150
                a.flag = 0x1 | 0x2 | (0x40 | 0x20 if r1 else 0x80 | 0x10)
                out.write(a)
    store = FragmentStore.from_bam(bam_path, assay="chec", verbose=True)
    err = capsys.readouterr().err
    assert store.n_fragments == 200
    assert err.count("now on chr1") == 1 and err.count("now on chr2") == 1
    assert err.count("[pyvplot.from_bam") <= 6  # header, 2 chromosomes, finished, done
