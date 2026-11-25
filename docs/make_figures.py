"""Generate publication-quality documentation figures into docs/img/ using pyvplot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyvplot as vp
from pyvplot.plotting import (
    plot_cut_footprint_matplotlib,
    plot_insert_size_matplotlib,
    plot_phasogram_matplotlib,
    plot_vplot_matplotlib,
)

IMG_DIR = Path(__file__).parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _simulate_vplot_locus(
    tf_bound: bool = True,
    n_anchors: int = 80,
    seed: int = 42,
) -> tuple[vp.FragmentStore, pd.DataFrame]:
    """Simulate realistic paired-end cleavage geometry around TF motifs.

    When ``tf_bound=True``, cleavage is blocked inside the protected motif core
    ``[-14, +14)`` and enriched just outside its boundaries, producing a classic
    V-shaped fragment distribution with flanking phased nucleosomes at +/-145 bp.
    When ``tf_bound=False`` (mutant), TF protection disappears and a central
    nucleosome invades the vacated motif.
    """
    rng = np.random.default_rng(seed)
    centers = np.arange(5_000, 5_000 + n_anchors * 3_000, 3_000)
    anchors = pd.DataFrame(
        {
            "chrom": ["chr1"] * n_anchors,
            "center": centers,
            "strand": rng.choice(["+", "-"], size=n_anchors),
        },
        index=[f"motif_{i}" for i in range(n_anchors)],
    )

    starts: list[int] = []
    ends: list[int] = []

    for c in centers:
        if tf_bound:
            # 1. Classic V-plot TF-spanning fragments: left cut just upstream of -14,
            #    right cut just downstream of +14 (exponential decay away from boundary)
            n_tf = 45
            left_off = -14 - rng.exponential(scale=18.0, size=n_tf)
            right_off = 14 + rng.exponential(scale=18.0, size=n_tf)
            for lo, ro in zip(left_off, right_off, strict=True):
                s = int(round(c + lo))
                e = int(round(c + ro))
                if 25 <= (e - s) < 230:
                    starts.append(s)
                    ends.append(e)

            # 2. Adjacent linker-cleaved fragments forming the outer V arms
            n_arm = 35
            for sign in (-1, 1):
                near = sign * (14 + rng.exponential(scale=6.0, size=n_arm))
                flen = rng.uniform(35, 160, size=n_arm)
                if sign < 0:
                    s_arr = c + near - flen
                    e_arr = c + near
                else:
                    s_arr = c + near
                    e_arr = c + near + flen
                for s_f, e_f in zip(s_arr, e_arr, strict=True):
                    starts.append(int(round(s_f)))
                    ends.append(int(round(e_f)))

            # 3. Well-phased -1/+1 and -2/+2 flanking nucleosomes
            for nuc_pos, n_nuc, sd in ((-150, 75, 14), (150, 75, 14), (-335, 35, 22), (335, 35, 22)):
                mids = c + rng.normal(nuc_pos, sd, size=n_nuc)
                fls = rng.normal(152, 9, size=n_nuc)
                for m, fl in zip(mids, fls, strict=True):
                    fl_i = int(round(np.clip(fl, 125, 195)))
                    s_i = int(round(m - fl_i / 2))
                    starts.append(s_i)
                    ends.append(s_i + fl_i)
        else:
            # Mutant: no TF protection, nucleosome invades center + fuzzy flanking nucleosomes
            for nuc_pos, n_nuc, sd in ((0, 95, 20), (-185, 55, 25), (185, 55, 25)):
                mids = c + rng.normal(nuc_pos, sd, size=n_nuc)
                fls = rng.normal(152, 10, size=n_nuc)
                for m, fl in zip(mids, fls, strict=True):
                    fl_i = int(round(np.clip(fl, 125, 195)))
                    s_i = int(round(m - fl_i / 2))
                    starts.append(s_i)
                    ends.append(s_i + fl_i)

        # Uniform background across both short and nucleosomal lengths
        n_bg_short = 25
        mids_s = c + rng.uniform(-340, 340, size=n_bg_short)
        fls_s = rng.uniform(30, 125, size=n_bg_short)
        for m, fl in zip(mids_s, fls_s, strict=True):
            fl_i = int(round(fl))
            s_i = int(round(m - fl_i / 2))
            starts.append(s_i)
            ends.append(s_i + fl_i)

        n_bg_nuc = 65
        mids_n = c + rng.uniform(-340, 340, size=n_bg_nuc)
        fls_n = rng.normal(154, 14, size=n_bg_nuc)
        for m, fl in zip(mids_n, fls_n, strict=True):
            fl_i = int(round(np.clip(fl, 125, 215)))
            s_i = int(round(m - fl_i / 2))
            starts.append(s_i)
            ends.append(s_i + fl_i)

    store = vp.FragmentStore(
        starts={"chr1": np.asarray(starts, dtype=np.int32)},
        ends={"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )
    return store, anchors


def make_vplot_figures() -> None:
    wt, anchors = _simulate_vplot_locus(tf_bound=True, n_anchors=90, seed=101)
    mut, _ = _simulate_vplot_locus(tf_bound=False, n_anchors=90, seed=202)

    vp_wt = vp.vplot(wt, anchors, flank=260, min_len=25, max_len=215, pos_bin=2, len_bin=2)
    vp_mut = vp.vplot(mut, anchors, flank=260, min_len=25, max_len=215, pos_bin=2, len_bin=2)

    vp_wt_cpm = vp_wt.normalize("cpm").smooth((1.2, 1.2))
    vp_wt_oe = vp_wt.normalize("expected", expected_from="flank", pseudocount=0.25).smooth((1.2, 1.2))
    vp_diff = vp_mut.diff(vp_wt, mode="log2fc", pseudocount=15.0, smooth_sigma=(1.6, 1.6))

    # 1. Hero 2-panel figure for index.md and quickstart.md
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    plot_vplot_matplotlib(
        vp_wt_oe,
        ax=axes[0],
        cmap="magma",
        title="WT ChEC-seq V-plot (Observed / Expected)",
        cbar_label="O / E enrichment",
    )
    plot_vplot_matplotlib(
        vp_diff,
        ax=axes[1],
        title="Differential V-plot: Mutant vs. WT",
        cbar_label="log2FC (Mut / WT)",
    )
    fig.savefig(IMG_DIR / "vplot_hero.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Normalization comparison 3-panel figure for vplots-and-diff.md
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    plot_vplot_matplotlib(
        vp_wt_cpm,
        ax=axes[0],
        cmap="magma",
        title="CPM (nucleosomes dominate)",
        cbar_label="CPM / anchor",
    )
    plot_vplot_matplotlib(
        vp_wt_oe,
        ax=axes[1],
        cmap="magma",
        title="Row-wise expected norm (TF V rescued)",
        cbar_label="O / E",
    )
    plot_vplot_matplotlib(
        vp_diff,
        ax=axes[2],
        title="Differential Mut vs. WT",
        cbar_label="log2FC",
    )
    fig.savefig(IMG_DIR / "vplot_norm_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Faceted plotnine VPlotCollection
    wt_b, anchors_b = _simulate_vplot_locus(tf_bound=True, n_anchors=60, seed=303)
    mut_b, _ = _simulate_vplot_locus(tf_bound=False, n_anchors=60, seed=404)
    vp_wt_a = vp.vplot(wt, anchors, flank=240, min_len=30, max_len=210, pos_bin=4, len_bin=4).normalize("expected", pseudocount=0.3).smooth((1.0, 1.0))
    vp_mut_a = vp.vplot(mut, anchors, flank=240, min_len=30, max_len=210, pos_bin=4, len_bin=4).normalize("expected", pseudocount=0.3).smooth((1.0, 1.0))
    vp_wt_b = vp.vplot(wt_b, anchors_b, flank=240, min_len=30, max_len=210, pos_bin=4, len_bin=4).normalize("expected", pseudocount=0.3).smooth((1.0, 1.0))
    vp_mut_b = vp.vplot(mut_b, anchors_b, flank=240, min_len=30, max_len=210, pos_bin=4, len_bin=4).normalize("expected", pseudocount=0.3).smooth((1.0, 1.0))

    coll = (
        vp.VPlotCollection()
        .add(vp_wt_a, tf="CTCF", condition="WT")
        .add(vp_mut_a, tf="CTCF", condition="KO")
        .add(vp_wt_b, tf="Reb1", condition="WT")
        .add(vp_mut_b, tf="Reb1", condition="KO")
    )
    import plotnine as p9

    p = coll.plot_plotnine(facet="tf ~ condition", cmap="magma", limits=(0.0, 10.0), fill_label="O / E") + p9.theme(
        figure_size=(7.5, 5.2),
        panel_spacing=0.06,
    )
    p.save(IMG_DIR / "vplot_plotnine_facet.png", dpi=180, verbose=False)


def make_footprint_and_matrix_figure() -> None:
    wt, anchors = _simulate_vplot_locus(tf_bound=True, n_anchors=90, seed=101)

    fp = wt.cut_site_footprint(anchors, flank=80, bin_size=1, len_range="tf", split_strands=True, agg="mean")
    mat_tf = wt.to_matrix(anchors, flank=350, bin_size=5, feature="midpoint", len_range="tf").normalize("cpm")
    mat_nuc = wt.to_matrix(anchors, flank=350, bin_size=5, feature="midpoint", len_range="mononucleosome").normalize("cpm")

    prof_tf = mat_tf.mean_profile(ci=0.95)
    prof_nuc = mat_nuc.mean_profile(ci=0.95)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)

    # Left: 1-bp strand-separated cleavage cut-site footprint
    plot_cut_footprint_matplotlib(
        fp,
        ax=axes[0],
        title="1-bp Strand-Separated Cut-Site Footprint (20–120 bp)",
    )
    axes[0].axvspan(-14, 14, color="#ffd92f", alpha=0.22, label="Protected motif core")
    axes[0].legend(frameon=False, fontsize=8)

    # Right: Size-stratified TF vs. nucleosome midpoint profiles with 95% CI
    ax = axes[1]
    ax.plot(prof_tf["pos"], prof_tf["mean"], color="#d95f02", linewidth=1.6, label="TF fragments (20–120 bp)")
    ax.fill_between(prof_tf["pos"], prof_tf["ci_low"], prof_tf["ci_high"], color="#d95f02", alpha=0.2)
    ax.plot(prof_nuc["pos"], prof_nuc["mean"], color="#1b9e77", linewidth=1.6, label="Mononucleosomes (130–200 bp)")
    ax.fill_between(prof_nuc["pos"], prof_nuc["ci_low"], prof_nuc["ci_high"], color="#1b9e77", alpha=0.2)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Distance from anchor center (bp)")
    ax.set_ylabel("Mean midpoint signal (CPM)")
    ax.set_title("Size-Stratified Midpoint Profiles (95% CI)")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(IMG_DIR / "cut_site_footprint.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_qc_and_phasogram_figures() -> None:
    # 1. Realistic ATAC-seq insert-size distribution (NFR + mono + di + tri nucleosomes)
    rng = np.random.default_rng(77)
    nfr = rng.gamma(shape=7.0, scale=8.5, size=24_000)
    mono = rng.normal(loc=202.0, scale=22.0, size=14_000)
    di = rng.normal(loc=392.0, scale=32.0, size=5_500)
    tri = rng.normal(loc=580.0, scale=42.0, size=1_500)
    raw_lens = np.concatenate([nfr, mono, di, tri])
    lens = np.round(raw_lens[(raw_lens >= 20) & (raw_lens <= 700)]).astype(np.int32)
    starts = np.arange(1_000, 1_000 + len(lens) * 20, 20, dtype=np.int32)
    atac = vp.FragmentStore({"chr1": starts}, {"chr1": starts + lens}, assay="atac")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    plot_insert_size_matplotlib(atac, ax=axes[0], min_len=20, max_len=650, log_y=False, title="ATAC-seq Insert Size (Linear)")
    plot_insert_size_matplotlib(atac, ax=axes[1], min_len=20, max_len=650, log_y=True, title="ATAC-seq Insert Size (Log10)")
    fig.savefig(IMG_DIR / "insert_size_qc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Nucleosome phasogram & NRL recovery (true NRL = 185 bp)
    rng2 = np.random.default_rng(2026)
    s_list: list[int] = []
    e_list: list[int] = []
    for arr_idx in range(45):
        base = 5_000 + arr_idx * 5_000
        for k in range(8):
            dyad_c = base + k * 185
            for _ in range(12):
                dyad = int(round(dyad_c + rng2.normal(0.0, 10.0)))
                fl = int(rng2.integers(144, 156))
                s_list.append(dyad - fl // 2)
                e_list.append(dyad - fl // 2 + fl)
    mnase = vp.FragmentStore(
        {"chr1": np.asarray(s_list, dtype=np.int32)},
        {"chr1": np.asarray(e_list, dtype=np.int32)},
        assay="mnase",
    )
    ph = mnase.phasogram(max_lag=900, smooth_sigma=6.0)

    fig, ax = plot_phasogram_matplotlib(
        ph,
        figsize=(7.5, 4.0),
        title=f"Nucleosome Phasogram (NRL = {ph.nrl:.1f} bp, R² = {ph.r_squared:.4f})",
    )
    fig.savefig(IMG_DIR / "phasogram_nrl.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_bias_figure() -> None:
    """Illustrate how KmerBiasTable removes strong enzyme sequence artifacts around a motif."""
    rng = np.random.default_rng(123)
    # 4-mer table where 'GATC' is cut 5x more frequently than average
    bias = vp.KmerBiasTable.from_dict({"GATC": 5.0, "ATAT": 2.5}, cut_offset=2, normalize=False)

    # 80-bp anchor window with a real protected motif at [-10, +10) AND an artifactual 'GATC'
    # hotspot at offset -20 and +18
    motif_seq = list("".join(rng.choice(list("ACGT"), size=80)))
    motif_seq[18:22] = list("GATC")
    motif_seq[56:60] = list("GATC")
    motif_seq[32:36] = list("ATAT")
    seq_str = "".join(motif_seq)

    exp = bias.score_sequence(seq_str)
    pos = np.arange(-40, 40) + 0.5
    # True underlying accessibility: protected dip at [-10, +10), elevated shoulders at +/-12..25
    true_signal = 1.0 + 1.2 * np.exp(-0.5 * ((np.abs(pos) - 18.0) / 5.0) ** 2)
    true_signal[np.abs(pos) < 10.0] = 0.25
    observed = true_signal * exp

    mat = vp.SignalMatrix(
        values=observed[None, :],
        region_index=np.array(["site_0"]),
        upstream=40,
        downstream=40,
        bin_size=1,
        feature="cuts_plus",
    )
    corrected = bias.correct_matrix(mat, [seq_str], preserve_scale=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True, sharey=True)
    axes[0].plot(pos, observed, color="#b2182b", linewidth=1.5, label="Observed cuts (sequence-biased)")
    axes[0].axvspan(-10, 10, color="#ffd92f", alpha=0.22, label="Protected motif")
    axes[0].set_title("Before K-mer Bias Correction")
    axes[0].set_xlabel("Distance from motif center (bp)")
    axes[0].set_ylabel("Cut-site signal")
    axes[0].set_ylim(-0.3, 13.8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[1].plot(pos, corrected.values[0], color="#1b9e77", linewidth=1.6, label="Bias-corrected cuts")
    axes[1].axvspan(-10, 10, color="#ffd92f", alpha=0.22, label="Protected motif")
    axes[1].set_title("After KmerBiasTable.correct_matrix()")
    axes[1].set_xlabel("Distance from motif center (bp)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.savefig(IMG_DIR / "bias_correction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_signal_heatmap_figure() -> None:
    """Per-anchor heatmap of TF-sized midpoints: strongly, weakly and un-bound sites."""
    parts = [
        vp.datasets.simulate_chec(n_sites=100, tf_fragments=tf, chrom=chrom, seed=seed)
        for tf, chrom, seed in ((30, "chrA", 1), (8, "chrB", 2), (0, "chrC", 3))
    ]
    store = vp.FragmentStore.merge([s for s, _ in parts])
    anchors = pd.concat([a.rename(index=lambda i, c=a["chrom"].iat[0]: f"{c}_{i}") for _, a in parts])
    mat = store.to_matrix(anchors, flank=400, bin_size=10, feature="midpoint", len_range="tf").normalize("cpm")
    fig, _ = mat.plot(title="TF-sized fragment midpoints (20-120 bp)", figsize=(4.2, 6.0))
    fig.savefig(IMG_DIR / "signal_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pillar_tutorial_store(
    sites: tuple[tuple[int, int], ...] = ((5000, 5016), (5300, 5312)), seed: int = 3
) -> vp.FragmentStore:
    """The store simulated in docs/tutorials/pillar-scanning.md: background plus bound sites."""
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for _ in range(3000):
        s = int(rng.integers(4000, 6000))
        starts.append(s)
        ends.append(s + int(rng.integers(20, 250)))
    for lo, hi in sites:
        for _ in range(80):
            starts.append(lo - int(rng.integers(25, 220)))
            ends.append(lo)
            starts.append(hi)
            ends.append(hi + int(rng.integers(25, 220)))
        for _ in range(40):
            starts.append(lo)
            ends.append(hi + int(rng.integers(20, 80)))
    order = np.argsort(starts)
    return vp.FragmentStore.from_arrays(
        {"chr1": np.asarray(starts)[order].astype(np.int32)},
        {"chr1": np.asarray(ends)[order].astype(np.int32)},
        assay="chec",
    )


def make_pillar_figures() -> None:
    frags = _pillar_tutorial_store()
    scan = frags.scan_locus_pillars("chr1:4800-5500")
    fig, _ = scan.plot(store=frags, figsize=(11.0, 8.0))
    fig.savefig(IMG_DIR / "pillar_scan_locus.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Calibration: the tutorial's background-only control at the sample's background depth
    rng_c = np.random.default_rng(12)
    c_starts = np.sort(rng_c.integers(4000, 6000, 3000))
    control = vp.FragmentStore.from_arrays(
        {"chr1": c_starts.astype(np.int32)},
        {"chr1": (c_starts + rng_c.integers(20, 250, 3000)).astype(np.int32)},
        assay="chec",
    )
    loose = dict(min_protection_score=0.0, min_raw_protection=0.0, min_bin_frags=5)
    s = frags.scan_genome_pillars(**loose)["protection_score"].to_numpy()
    c = control.scan_genome_pillars(**loose)["protection_score"].to_numpy()
    thr = np.linspace(0.0, 2.2, 221)
    n_s = np.array([(s >= t).sum() for t in thr])
    n_c = np.array([(c >= t).sum() for t in thr])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.step(thr, n_s, where="post", color="#257de0", lw=2.0)
    ax.step(thr, n_c, where="post", color="#f47230", lw=2.0)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax.axvline(1.0, color="#666666", lw=1.0, ls="--")
    ax.text(1.03, 30, "chosen threshold 1.0:\ncontrol has no calls left", fontsize=8.5, color="#191919", va="top")
    ax.text(1.25, 2.25, "sample (2 bound sites)", fontsize=8.5, color="#191919", va="bottom")
    first_zero = float(thr[np.argmax(n_c == 0)])
    ax.text(first_zero - 0.02, 0.45, "background-only control", fontsize=8.5, color="#191919", ha="right", va="center")
    ax.set_xlabel("min_protection_score")
    ax.set_ylabel("motif calls at or above threshold")
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(IMG_DIR / "pillar_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_vplot_figures()
    make_signal_heatmap_figure()
    make_footprint_and_matrix_figure()
    make_qc_and_phasogram_figures()
    make_bias_figure()
    make_pillar_figures()
    files = sorted(p.name for p in IMG_DIR.glob("*.png"))
    print(f"Generated {len(files)} figures in {IMG_DIR}: {', '.join(files)}")


if __name__ == "__main__":
    main()
