"""pyvplot: fragment-length geometry around genomic anchors for ChEC-seq, ATAC-seq, MNase-seq and relatives.

Typical use::

    import pyvplot as vp

    store = vp.FragmentStore.from_bam("sample.bam", assay="chec")
    v = store.vplot("motifs.bed", flank=300, pos_bin=5, len_bin=5)
    v.normalize("expected").plot()
"""

from pyvplot import datasets
from pyvplot.assays import (
    ASSAY_PRESETS,
    AssayConfig,
    get_assay,
    list_assays,
    register_assay,
    resolve_len_range,
)
from pyvplot.bias import (
    KmerBiasTable,
    extract_anchor_sequences,
    normalize_by_control,
    reverse_complement,
)
from pyvplot.export import to_bedgraph, to_bigwig, to_binned_track, to_pymisha
from pyvplot.extract import (
    HAS_NUMBA,
    NormalizedAnchors,
    extract_events,
    normalize_anchors,
)
from pyvplot.locus import LocusVPlot, locus_vplot
from pyvplot.matrices import (
    CutSiteFootprint,
    SignalMatrix,
    cut_site_footprint,
    footprint_score,
    to_matrix,
)
from pyvplot.qc import (
    PhasogramResult,
    estimate_nrl,
    frip,
    insert_size_distribution,
    phasogram,
    qc_summary,
)
from pyvplot.store import FragmentStore
from pyvplot.vplot import VPlot, VPlotCollection, vplot

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ASSAY_PRESETS",
    "AssayConfig",
    "CutSiteFootprint",
    "FragmentStore",
    "HAS_NUMBA",
    "KmerBiasTable",
    "LocusVPlot",
    "NormalizedAnchors",
    "PhasogramResult",
    "SignalMatrix",
    "VPlot",
    "VPlotCollection",
    "cut_site_footprint",
    "datasets",
    "estimate_nrl",
    "extract_anchor_sequences",
    "extract_events",
    "footprint_score",
    "frip",
    "get_assay",
    "insert_size_distribution",
    "list_assays",
    "locus_vplot",
    "normalize_anchors",
    "normalize_by_control",
    "phasogram",
    "qc_summary",
    "register_assay",
    "resolve_len_range",
    "reverse_complement",
    "to_bedgraph",
    "to_bigwig",
    "to_binned_track",
    "to_matrix",
    "to_pymisha",
    "vplot",
]
