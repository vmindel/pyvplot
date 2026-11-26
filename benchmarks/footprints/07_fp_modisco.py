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
    from modiscolite.tfmodisco import TFMoDISco

    pm.gsetroot(C.GROOT)
    calls = pd.read_parquet(C.OUT / "genome_calls_user.parquet")
    calls = calls[calls["chrom"].str.match(r"^chr[0-9XY]+$")]
    mtop = calls.sort_values("composite_score", ascending=False).head(N_MODISCO).reset_index(drop=True)
    store = vp.FragmentStore.load(C.STORE)
    W2 = 100
    oh = np.zeros((len(mtop), 2 * W2, 4), np.float32)
    hyp = np.zeros_like(oh)
    seqs = seqs_for(pm, mtop["chrom"], mtop["center"] - W2, mtop["center"] + W2)
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, (r, s) in enumerate(zip(mtop.itertuples(), seqs)):
        sc = store.scan_locus_pillars((r.chrom, int(r.center) - W2 - 150, int(r.center) + W2 + 150), **USER)
        prof = sc.protection_score[150:150 + 2 * W2]  # signed: footprints > 0, pillars < 0
        for j, b in enumerate(s):
            k = lut.get(b)
            if k is not None:
                oh[i, j, k] = 1
        hyp[i] = prof[:, None]
    del store
    log("modisco inputs", oh.shape, "frac>0", float((hyp[..., 0] > 0).mean()))
    pos_p, neg_p = TFMoDISco(one_hot=oh, hypothetical_contribs=hyp, sliding_window_size=11, flank_size=5,
                             trim_to_window_size=20, initial_flank_to_add=3, final_flank_to_add=0,
                             max_seqlets_per_metacluster=20000, target_seqlet_fdr=0.2, verbose=True)
    pats = [{"name": f"fp_modisco_{k}", "ppm": np.asarray(p.sequence).T, "n_seqlets": len(p.seqlets)}
            for k, p in enumerate(pos_p or [])]
    log("footprint-modisco patterns", [(p["name"], p["n_seqlets"]) for p in pats])
    model = []
    with h5py.File(C.ATTR_DIR / "modisco_results.h5", "r") as f:
        for grp in ("pos_patterns", "neg_patterns"):
            for pn in f.get(grp, {}):
                model.append(mpy.Motif(np.asarray(f[grp][pn]["sequence"][:]).T, name=f"{grp[:3]}_{pn.replace('pattern_', '')}",
                                       kind="probability"))
    queries = [mpy.Motif(p["ppm"], name=f'{p["name"]} (n={p["n_seqlets"]})', kind="probability") for p in pats]
    if queries:
        sim = mpy.compare_motifs(queries, targets=model, method="PCC", min_mean_ic=0.0)
        sim = pd.DataFrame(np.asarray(sim.values if hasattr(sim, "values") else sim),
                           index=[q.name for q in queries], columns=[m.name for m in model])
        best = pd.DataFrame({"best_model_pattern": sim.idxmax(axis=1), "pcc": sim.max(axis=1),
                             "consensus": [mpy.iupac_consensus(q) for q in queries]})
        best.to_csv(OUT / "fp_modisco_vs_model.csv")
        log("best model match\n" + best.round(3).to_string())
        import matplotlib
        matplotlib.use("Agg")
        from moodspy.plotting import view_motifs
        view_motifs(queries[:12], kind="ICM", align=False).savefig(OUT / "fp_modisco_logos.png", dpi=110, bbox_inches="tight")
    with open(OUT / "fp_modisco.pkl", "wb") as fh:
        pickle.dump(pats, fh)
    log("done")


if __name__ == "__main__":
    main()
