"""Assay physics presets for paired-end and single-end chromatin profiling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

VALID_FEATURES = frozenset(
    {"midpoint", "cuts", "cuts_plus", "cuts_minus", "cuts_stranded", "coverage"}
)


@dataclass(frozen=True)
class AssayConfig:
    """Configuration encapsulating assay-specific cleavage geometry and size bands.

    Attributes
    ----------
    name:
        Canonical preset name (e.g., ``"chec"``, ``"atac"``, ``"mnase"``).
    left_shift:
        Offset (in bp) added to the 0-based fragment ``start`` (or forward-strand
        5' read end for single-end assays). For ATAC-seq Tn5 correction this is ``+4``.
    right_shift:
        Offset (in bp) added to the 0-based half-open fragment ``end`` (or
        reverse-strand 5' read end). For ATAC-seq Tn5 correction this is ``-5``.
    default_feature:
        Default geometric feature extracted for matrices and profiles:
        ``"midpoint"``, ``"cuts"``, ``"cuts_plus"``, ``"cuts_minus"``,
        ``"cuts_stranded"``, or ``"coverage"``.
    paired_end:
        Whether fragments are paired-end intervals ``[start, end)`` (``True``)
        or stranded single-end 5' cleavage events (``False``, e.g., ChIP-exo).
    size_bins:
        Named ``(min_len, max_len)`` fragment-length regimes (inclusive, bp)
        used for QC band summaries and size-stratified analyses.
    description:
        Human-readable summary of the assay preset.
    """

    name: str
    left_shift: int = 0
    right_shift: int = 0
    default_feature: str = "midpoint"
    paired_end: bool = True
    size_bins: dict[str, tuple[int, int]] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.default_feature not in VALID_FEATURES:
            raise ValueError(
                f"Invalid default_feature {self.default_feature!r}. "
                f"Expected one of {sorted(VALID_FEATURES)}."
            )
        # Validate size_bins ordering
        for label, bounds in self.size_bins.items():
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(
                    f"Invalid size bin {label!r}: {bounds!r}. Expected (min_len, max_len) with min_len <= max_len."
                )

    def with_overrides(self, **kwargs: Any) -> AssayConfig:
        """Return a modified copy of this assay configuration."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "left_shift": int(self.left_shift),
            "right_shift": int(self.right_shift),
            "default_feature": self.default_feature,
            "paired_end": bool(self.paired_end),
            "size_bins": {k: [int(v[0]), int(v[1])] for k, v in self.size_bins.items()},
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssayConfig:
        """Deserialize configuration from a dictionary."""
        raw_bins = data.get("size_bins", {})
        size_bins = {k: (int(v[0]), int(v[1])) for k, v in raw_bins.items()}
        return cls(
            name=str(data["name"]),
            left_shift=int(data.get("left_shift", 0)),
            right_shift=int(data.get("right_shift", 0)),
            default_feature=str(data.get("default_feature", "midpoint")),
            paired_end=bool(data.get("paired_end", True)),
            size_bins=size_bins,
            description=str(data.get("description", "")),
        )


ASSAY_PRESETS: dict[str, AssayConfig] = {
    "chec": AssayConfig(
        name="chec",
        left_shift=0,
        right_shift=0,
        default_feature="cuts",
        paired_end=True,
        size_bins={
            "tf": (20, 120),
            "mononucleosome": (130, 200),
        },
        description="ChEC-seq: tethered MNase cleavage ends and TF/nucleosome fragments.",
    ),
    "atac": AssayConfig(
        name="atac",
        left_shift=4,
        right_shift=-5,
        default_feature="cuts",
        paired_end=True,
        size_bins={
            "nfr": (10, 100),
            "mononucleosome": (180, 247),
            "dinucleosome": (315, 473),
        },
        description="ATAC-seq: Tn5 (+4/-5) shifted cleavage sites and nucleosome-free/nucleosomal fragments.",
    ),
    "mnase": AssayConfig(
        name="mnase",
        left_shift=0,
        right_shift=0,
        default_feature="midpoint",
        paired_end=True,
        size_bins={
            "subnucleosome": (20, 120),
            "nucleosome": (135, 165),
        },
        description="MNase-seq: nucleosome dyad midpoints and sub-nucleosomal particles.",
    ),
    "cutrun": AssayConfig(
        name="cutrun",
        left_shift=0,
        right_shift=0,
        default_feature="midpoint",
        paired_end=True,
        size_bins={
            "short": (20, 120),
            "nucleosome": (150, 200),
        },
        description="CUT&RUN: pA-MNase targeted cleavage fragments.",
    ),
    "cuttag": AssayConfig(
        name="cuttag",
        left_shift=0,
        right_shift=0,
        default_feature="cuts",
        paired_end=True,
        size_bins={
            "short": (20, 120),
            "nucleosome": (150, 200),
        },
        description="CUT&Tag: pA-Tn5 targeted tagmentation fragments.",
    ),
    "chip_exo": AssayConfig(
        name="chip_exo",
        left_shift=0,
        right_shift=0,
        default_feature="cuts_stranded",
        paired_end=False,
        size_bins={},
        description="ChIP-exo / ChIP-nexus: strand-separated single-end 5' exonuclease stop sites.",
    ),
    "generic": AssayConfig(
        name="generic",
        left_shift=0,
        right_shift=0,
        default_feature="midpoint",
        paired_end=True,
        size_bins={},
        description="Generic paired-end fragments with no endpoint shift.",
    ),
}

_ASSAY_ALIASES: dict[str, str] = {
    "chec-seq": "chec",
    "chec_seq": "chec",
    "checseq": "chec",
    "atac-seq": "atac",
    "atac_seq": "atac",
    "atacseq": "atac",
    "mnase-seq": "mnase",
    "mnase_seq": "mnase",
    "mnaseseq": "mnase",
    "cut&run": "cutrun",
    "cut_run": "cutrun",
    "cut-run": "cutrun",
    "cutandrun": "cutrun",
    "cut&tag": "cuttag",
    "cut_tag": "cuttag",
    "cut-tag": "cuttag",
    "cutandtag": "cuttag",
    "chip-exo": "chip_exo",
    "chipexo": "chip_exo",
    "chip-nexus": "chip_exo",
    "chip_nexus": "chip_exo",
}


def get_assay(assay: str | AssayConfig) -> AssayConfig:
    """Resolve an assay name (or alias) or pass through an :class:`AssayConfig`."""
    if isinstance(assay, AssayConfig):
        return assay
    if not isinstance(assay, str):
        raise TypeError(f"Expected str or AssayConfig, got {type(assay).__name__}.")
    key = assay.strip().lower()
    canonical = _ASSAY_ALIASES.get(key, key)
    if canonical not in ASSAY_PRESETS:
        available = ", ".join(sorted(ASSAY_PRESETS))
        raise ValueError(f"Unknown assay preset {assay!r}. Available presets: {available}.")
    return ASSAY_PRESETS[canonical]


def resolve_len_range(
    assay: str | AssayConfig,
    len_range: tuple[int, int] | str | None,
) -> tuple[int, int] | None:
    """Turn a fragment-length filter into an inclusive ``(min_len, max_len)`` tuple.

    ``len_range`` is either ``None`` (no filter), an explicit ``(min_len, max_len)``
    tuple, or the name of one of the assay's size bins (``"tf"``, ``"nfr"``,
    ``"mononucleosome"``, ...). Both bounds are inclusive.
    """
    if len_range is None:
        return None
    if isinstance(len_range, str):
        cfg = get_assay(assay)
        if len_range not in cfg.size_bins:
            available = ", ".join(sorted(cfg.size_bins)) or "(none)"
            raise ValueError(
                f"Unknown size bin {len_range!r} for assay {cfg.name!r}. "
                f"Available size bins: {available}. Or pass an explicit (min_len, max_len) tuple."
            )
        return cfg.size_bins[len_range]
    if len(len_range) != 2:
        raise ValueError(f"Expected (min_len, max_len) tuple, got {len_range!r}.")
    lo, hi = int(len_range[0]), int(len_range[1])
    if lo > hi:
        raise ValueError(f"Invalid len_range {len_range!r}: min_len > max_len.")
    return (lo, hi)


def list_assays() -> list[str]:
    """Return sorted canonical names of all registered assay presets."""
    return sorted(ASSAY_PRESETS)


def register_assay(config: AssayConfig, overwrite: bool = False) -> None:
    """Register a custom :class:`AssayConfig` preset globally."""
    key = config.name.strip().lower()
    if key in ASSAY_PRESETS and not overwrite:
        raise ValueError(
            f"Assay preset {key!r} already exists. Pass overwrite=True to replace it."
        )
    ASSAY_PRESETS[key] = config
