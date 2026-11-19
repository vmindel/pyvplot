"""Small simulated datasets for tutorials, tests and trying the API without real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyvplot.store import FragmentStore


def simulate_chec(
    n_sites: int = 50,
    tf_fragments: int = 30,
    nucleosome_fragments: int = 80,
    background_per_site: int = 100,
    tf_len: tuple[int, int] = (45, 80),
    nucleosome_len: tuple[int, int] = (140, 170),
    tf_jitter: float = 5.0,
    nucleosome_spread: int = 400,
    site_spacing: int = 2_000,
    chrom: str = "chr1",
    assay: str = "chec",
    seed: int = 0,
) -> tuple[FragmentStore, pd.DataFrame]:
    """Simulate paired-end fragments around TF binding sites.

    Each site gets ``tf_fragments`` short fragments whose midpoints sit on the
    site (normal jitter, sd ``tf_jitter`` bp) and ``nucleosome_fragments``
    nucleosome-sized fragments with midpoints uniform in
    ``+/- nucleosome_spread`` bp. ``n_sites * background_per_site`` background
    fragments are spread uniformly over the simulated chromosome, with lengths
    uniform in 30-400 bp, so V-plot flanks are not empty.
    Set ``tf_fragments=0`` to simulate an unbound condition; with the same
    ``seed`` and ``n_sites`` both runs return identical anchors.

    Parameters
    ----------
    n_sites:
        Number of binding sites, spaced ``site_spacing`` bp apart.
    tf_fragments, nucleosome_fragments:
        Fragments per site of each class.
    background_per_site:
        Background fragments per site, placed anywhere on the chromosome.
    tf_len, nucleosome_len:
        Length range (bp, half-open) of each fragment class.
    tf_jitter:
        Standard deviation (bp) of TF fragment midpoints around the site.
    nucleosome_spread:
        Half-width (bp) of the region where nucleosome midpoints are placed.
    site_spacing:
        Distance (bp) between neighbouring sites.
    chrom:
        Chromosome name used for fragments and anchors.
    assay:
        Assay preset for the returned store.
    seed:
        Random seed.

    Returns
    -------
    store : FragmentStore
        Simulated fragments.
    anchors : pandas.DataFrame
        One row per site with ``chrom``, ``center`` and a random ``strand``,
        indexed ``site_0``, ``site_1``, ...

    Examples
    --------
    >>> wt, anchors = vp.datasets.simulate_chec(seed=1)
    >>> ko, _ = vp.datasets.simulate_chec(tf_fragments=0, seed=2)
    """
    rng = np.random.default_rng(seed)
    margin = nucleosome_spread + max(nucleosome_len[1], tf_len[1]) + 1_000
    centers = margin + site_spacing * np.arange(n_sites, dtype=np.int64)
    chrom_len = int(centers[-1] + margin) if n_sites > 0 else margin
    # Drawn first so that runs differing only in fragment counts share anchors.
    strands = rng.choice(["+", "-"], n_sites)

    def _frags(mids: np.ndarray, len_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        lens = rng.integers(len_range[0], len_range[1], size=len(mids))
        starts = np.round(mids).astype(np.int64) - lens // 2
        return starts, starts + lens

    parts = []
    if tf_fragments > 0:
        mids = np.repeat(centers, tf_fragments) + rng.normal(0.0, tf_jitter, n_sites * tf_fragments)
        parts.append(_frags(mids, tf_len))
    if nucleosome_fragments > 0:
        offs = rng.integers(-nucleosome_spread, nucleosome_spread, n_sites * nucleosome_fragments)
        parts.append(_frags(np.repeat(centers, nucleosome_fragments) + offs, nucleosome_len))
    n_bg = n_sites * background_per_site
    if n_bg > 0:
        mids = rng.integers(500, chrom_len - 500, n_bg).astype(np.float64)
        parts.append(_frags(mids, (30, 400)))

    if parts:
        starts = np.concatenate([p[0] for p in parts])
        ends = np.concatenate([p[1] for p in parts])
    else:
        starts = ends = np.empty(0, dtype=np.int64)

    store = FragmentStore.from_arrays({chrom: starts}, {chrom: ends}, assay=assay)
    store.metadata["source"] = "pyvplot.datasets.simulate_chec"
    anchors = pd.DataFrame(
        {
            "chrom": chrom,
            "center": centers,
            "strand": strands,
        },
        index=[f"site_{i}" for i in range(n_sites)],
    )
    return store, anchors
