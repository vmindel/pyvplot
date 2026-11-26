"""Step 5: de-novo motifs from genome-wide footprint calls (no model, no hits as input).

Input: ``out/genome_calls_user.parquet`` from 05_shipped.py (``scan_genome_pillars`` with the
profile caller and the Fig2 settings). Three discovery routes on the top calls:

1. pystreme (STREME-style) on 30-bp windows at each call's profile peak, against
   (a) dinucleotide shuffles, (b) local flanks 60 bp either side (same regions, not
   footprinted), (c) GC-matched windows from active regions without calls. (c) asks for
   enrichment beyond composition, which matters because MNase barely cuts GC runs.
2. MoDISco-style: modiscolite on 200-bp windows with contributions = one-hot x the clipped
   1-bp protection profile, so seqlets are protected stretches clustered by sequence.
3. moodspy: every discovered motif is compared (PCC) with the model's TF-MoDISco CWMs
   (modisco_results.h5) and annotated against HOCOMOCO v12 core (pystreme.annotate).

GPU job (short-gpu); see run_06.sh.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

import pyvplot as vp  # noqa: E402

N_TOP = int(os.environ.get("N_TOP", 20000))
N_MODISCO = int(os.environ.get("N_MODISCO", 10000))
HALF = 15
H12 = "/home/labs/barkailab/vovam/H12CORE_meme_format.meme"
USER = dict(len_range=(20, 300), flank_mode="either", footprint_widths=range(6, 21), motif_width_range=(6, 20))
T0 = time.time()
OUT = C.OUT / "denovo"
OUT.mkdir(exist_ok=True)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def seqs_for(pm, chroms, starts, ends):
    out = []
    df = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    for chrom, g in df.groupby("chrom", sort=False):
        s = pm.gseq_extract(g[["chrom", "start", "end"]].reset_index(drop=True))
        out.append(pd.Series([x.upper() for x in s], index=g.index))
    return pd.concat(out).sort_index().tolist()


def to_ppm(pwm):
    a = np.asarray(pwm, dtype=float)
    if a.shape[0] != 4:
        a = a.T
    if (a < 0).any() or not np.allclose(a.sum(0), 1, atol=1e-3):
        a = 0.25 * np.power(2.0, a)
        a = a / a.sum(0, keepdims=True)
    return a


def main():
    import h5py
    import moodspy as mpy
    import pymisha as pm
    import torch
    from pystreme import MotifDiscovery, SequenceStore, annotate, summary
    from pystreme.control import Explicit, GCMatched

    pm.gsetroot(C.GROOT)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    calls = pd.read_parquet(C.OUT / "genome_calls_user.parquet")
    calls = calls[calls["chrom"].str.match(r"^chr[0-9XY]+$")]
    regions = C.load_regions()
    in_tile = np.zeros(len(calls), bool)
    for chrom, g in regions.groupby("chrom"):
        m = (calls["chrom"] == chrom).to_numpy()
        c = calls["center"].to_numpy()[m]
        i = np.clip(np.searchsorted(g["start"].to_numpy(), c, side="right") - 1, 0, len(g) - 1)
        in_tile[np.flatnonzero(m)] = (c >= g["start"].to_numpy()[i]) & (c < g["end"].to_numpy()[i])
    calls["in_ism_tile"] = in_tile
    top = calls.sort_values("composite_score", ascending=False).head(N_TOP).reset_index(drop=True)
    log(f"{len(calls)} genome-wide calls; top {len(top)}: {top['in_ism_tile'].mean():.2%} inside ISM tiles; "
        f"widths {top['width'].value_counts().head(5).to_dict()}")

    c = top["center"].to_numpy()
    pos = seqs_for(pm, top["chrom"], c - HALF, c + HALF)
    flank = seqs_for(pm, np.r_[top["chrom"], top["chrom"]], np.r_[c - 60 - HALF, c + 60 - HALF],
                     np.r_[c - 60 + HALF, c + 60 + HALF])
    # GC pool: random positions in the same active windows (+-300 bp of calls) >= 40 bp from any call
    rng = np.random.default_rng(0)
    cand_c = np.repeat(c, 3) + rng.integers(-300, 300, 3 * len(c))
    cand_chr = np.repeat(top["chrom"].to_numpy(), 3)
    allc = {ch: np.sort(g["center"].to_numpy()) for ch, g in calls.groupby("chrom")}
    keep = []
    for i, (ch, x) in enumerate(zip(cand_chr, cand_c)):
        a = allc[ch]
        j = np.searchsorted(a, x)
        d = min(abs(x - a[max(0, j - 1)]), abs(x - a[min(len(a) - 1, j)]))
        keep.append(d >= 40)
    keep = np.asarray(keep)
    pool = seqs_for(pm, cand_chr[keep], cand_c[keep] - HALF, cand_c[keep] + HALF)
    log(f"positives {len(pos)}, flank controls {len(flank)}, GC pool {len(pool)}")

    gc = lambda ss: float(np.mean([(s.count("G") + s.count("C")) / max(1, len(s)) for s in ss]))  # noqa: E731
    stats = {"n_calls": int(len(calls)), "n_top": int(len(top)), "frac_top_in_ism_tile": float(top["in_ism_tile"].mean()),
             "gc_pos": gc(pos), "gc_flank": gc(flank), "gc_pool": gc(pool)}
    log(stats)

    disc = MotifDiscovery.from_sequences(pos, device=dev)
    runs = {}
    for name, ctrl in (("shuffle", None), ("flank", Explicit(flank)),
                       ("gcmatched", GCMatched(pool=SequenceStore.from_sequences(pool, device=dev)))):
        ms = disc.fit(n_motifs=8, widths=range(6, 16), control=ctrl, seed=0, verbose=True)
        runs[name] = ms
        summary(ms).to_csv(OUT / f"streme_{name}_summary.csv", index=False)
        if ms:
            annotate(ms, H12, top=3).to_csv(OUT / f"streme_{name}_h12.csv", index=False)
        log(name, summary(ms)[["consensus", "width", "n_sites", "holdout_logp", "central_logp"]].to_string() if ms else "none")

    # ---- MoDISco-style on protection-weighted one-hot
    mtop = top.head(N_MODISCO)
    store = vp.FragmentStore.load(C.STORE)
    W2 = 100
    oh = np.zeros((len(mtop), 2 * W2, 4), np.float32)
    hyp = np.zeros_like(oh)
    seqs = seqs_for(pm, mtop["chrom"], mtop["center"] - W2, mtop["center"] + W2)
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, (r, s) in enumerate(zip(mtop.itertuples(), seqs)):
        sc = store.scan_locus_pillars((r.chrom, int(r.center) - W2 - 150, int(r.center) + W2 + 150), **USER)
        prof = np.clip(sc.protection_score[150:150 + 2 * W2], 0, None)
        for j, b in enumerate(s):
            k = lut.get(b)
            if k is not None:
                oh[i, j, k] = 1
        hyp[i] = prof[:, None]
    del store
    log("modisco inputs", oh.shape)
    patterns_out = []
    try:
        from modiscolite.tfmodisco import TFMoDISco
        pos_p, _neg_p = TFMoDISco(one_hot=oh, hypothetical_contribs=hyp, sliding_window_size=11, flank_size=5,
                                  trim_to_window_size=20, initial_flank_to_add=3, final_flank_to_add=0,
                                  max_seqlets_per_metacluster=20000, target_seqlet_fdr=0.2, verbose=True)
        for k, p in enumerate(pos_p or []):
            patterns_out.append({"name": f"fp_modisco_{k}", "ppm": np.asarray(p.sequence).T, "n_seqlets": len(p.seqlets)})
    except Exception as e:  # report, keep the rest
        log("modisco failed:", repr(e))
    log("footprint-modisco patterns", [(p["name"], p["n_seqlets"]) for p in patterns_out])

    # ---- compare everything to the model's CWMs (PPM) with moodspy
    model = []
    with h5py.File(C.ATTR_DIR / "modisco_results.h5", "r") as f:
        for grp in ("pos_patterns", "neg_patterns"):
            if grp not in f:
                continue
            for pn in f[grp]:
                ppm = np.asarray(f[grp][pn]["sequence"][:]).T
                model.append(mpy.Motif(ppm, name=f"{grp[:3]}_{pn.replace('pattern_', '')}", kind="probability"))
    queries = [mpy.Motif(to_ppm(m.pwm), name=f"streme_{run}_{i}_{m.consensus}", kind="probability")
               for run, ms in runs.items() for i, m in enumerate(ms)]
    queries += [mpy.Motif(p["ppm"], name=p["name"], kind="probability") for p in patterns_out]
    comp = None
    if queries:
        sim = mpy.compare_motifs(queries, targets=model, method="PCC", min_mean_ic=0.0)
        sim = pd.DataFrame(np.asarray(sim.values if hasattr(sim, "values") else sim), index=[q.name for q in queries], columns=[m.name for m in model])
        sim.to_csv(OUT / "vs_model_modisco_pcc.csv")
        comp = pd.DataFrame({"best_model_pattern": sim.idxmax(1), "pcc": sim.max(1)})
        comp.to_csv(OUT / "vs_model_best.csv")
        log("best model match per motif\n" + comp.round(3).to_string())
    with open(OUT / "denovo.pkl", "wb") as fh:
        pickle.dump({"stats": stats, "streme": {k: [(m.consensus, m.pwm, m.holdout_logp, m.n_sites) for m in v]
                                                for k, v in runs.items()},
                     "fp_modisco": patterns_out}, fh)
    with open(OUT / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)

    # ---- logos
    try:
        import matplotlib
        matplotlib.use("Agg")
        from moodspy.plotting import view_motifs
        if queries:
            fig = view_motifs(queries, kind="ICM", align=False)
            fig.savefig(OUT / "denovo_logos.png", dpi=110, bbox_inches="tight")
        fig = view_motifs(model[:12], kind="ICM", align=False)
        fig.savefig(OUT / "model_modisco_logos.png", dpi=110, bbox_inches="tight")
    except Exception as e:
        log("logo plotting failed:", repr(e))
    log("done")


if __name__ == "__main__":
    main()
