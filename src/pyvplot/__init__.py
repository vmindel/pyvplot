"""pyvplot: Fast, assay-aware fragment geometry, V-plots, cut-site footprints, and nucleosome phasing."""

from pyvplot.assays import (
    ASSAY_PRESETS,
    AssayConfig,
    get_assay,
    list_assays,
    register_assay,
)
from pyvplot.bias import KmerBiasTable, normalize_by_control, reverse_complement
from pyvplot.export import to_bedgraph, to_bigwig, to_binned_track, to_pymisha
from pyvplot.extract import (
    HAS_NUMBA,
    NormalizedAnchors,
    extract_events,
    normalize_anchors,
)
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
    "NormalizedAnchors",
    "PhasogramResult",
    "SignalMatrix",
    "VPlot",
    "VPlotCollection",
    "cut_site_footprint",
    "estimate_nrl",
    "extract_events",
    "footprint_score",
    "frip",
    "get_assay",
    "insert_size_distribution",
    "list_assays",
    "normalize_anchors",
    "normalize_by_control",
    "phasogram",
    "qc_summary",
    "register_assay",
    "reverse_complement",
    "to_bedgraph",
    "to_bigwig",
    "to_binned_track",
    "to_matrix",
    "to_pymisha",
    "vplot",
]
