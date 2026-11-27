"""De novo motifs in high- vs low-occupancy footprint calls.

Top protection calls (genome_calls_user.parquet, from 05_shipped.py) are split into
occupancy tertiles with ``pyvplot.occupancy_profile(score="log2")`` (bias-ms,
feat/occupancy-score, merged in 89c8df4). If occupancy
separates binding from cleavage preference, KLF should concentrate in the high tertile and
MNase signatures (CTAG) / bare GC in the low one.

Per tertile: frequency of fixed k-mer families in the 30-bp window at the call centre,
GC, overlap with FiNeMo hits, and pystreme discovery against shuffles and GC-matched windows.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

import pyvplot as vp  # noqa: E402
from pyvplot import occupancy_profile  # noqa: E402

N_TOP = int(os.environ.get("N_TOP", 60000))
HALF = 15
H12 = "/home/labs/barkailab/vovam/H12CORE_meme_format.meme"
OUT = C.OUT / "denovo_occ"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
FAMILIES = {  # both strands searched
    "KLF_GGGYGKGG": "GGG[CT]G[GT]GG",
    "SP_GGGGCGGGG": "GGGGCGGG",
    "AP1_TGASTCA": "TGA[CG]TCA",
    "TEAD_CATTCC": "CATTCC",
    "FOX_TRTTTAC": "T[AG]TTTAC",
    "MNase_CTAG": "CTAG",
    "MNase_CCTAGG": "CCTAGG",
}
RC = str.maketrans("ACGT", "TGCA")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def has(seq, pat):
    """Pattern on either strand: search the sequence and its reverse complement."""
    return re.search(pat, seq) is not None or re.search(pat, seq.translate(RC)[::-1]) is not None


def seqs_for(pm, df, lo, hi):
    out = pd.Series(index=df.index, dtype=object)
    for chrom, g in df.groupby("chrom", sort=False):
        iv = pd.DataFrame({"chrom": chrom, "start": g["center"] + lo, "end": g["center"] + hi})
        out.loc[g.index] = [s.upper() for s in pm.gseq_extract(iv.reset_index(drop=True))]
    return out.tolist()


def main():
    import pymisha as pm
    import torch
    from pystreme import MotifDiscovery, SequenceStore, annotate, summary
    from pystreme.control import GCMatched

    pm.gsetroot(C.GROOT)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", "pyvplot from", vp.__file__)
    calls = pd.read_parquet(C.OUT / "genome_calls_user.parquet")
    calls = calls[calls["chrom"].str.match(r"^chr[0-9XY]+$")]
    top = calls.sort_values("composite_score", ascending=False).head(N_TOP).reset_index(drop=True)
    store = vp.FragmentStore.load(C.STORE)
    occ = np.empty(len(top))
    for i, r in enumerate(top.itertuples()):
        occ[i] = occupancy_profile(store, (r.chrom, int(r.center), int(r.center) + 1), score="log2")[0]
    top["occupancy_log2"] = occ
    del store
    log("occupancy done", pd.Series(occ).describe().round(3).to_dict())
    top["tertile"] = pd.qcut(top["occupancy_log2"].rank(method="first"), 3, labels=["low", "mid", "high"])
    top["seq"] = seqs_for(pm, top, -HALF, HALF)

    hits = C.load_hits()
    hits = hits[hits["pattern"].str.startswith("pos_")]
    hc = {ch: np.sort(((g["start"] + g["end"]) / 2).to_numpy()) for ch, g in hits.groupby("chrom")}

    def near_hit(r):
        a = hc.get(r.chrom)
        if a is None or len(a) == 0:
            return False
        j = np.searchsorted(a, r.center)
        return min(abs(r.center - a[max(0, j - 1)]), abs(r.center - a[min(len(a) - 1, j)])) <= 4

    top["near_finemo_hit"] = [near_hit(r) for r in top.itertuples()]
    rows = []
    for t, g in top.groupby("tertile", observed=True):
        row = {"tertile": t, "n": len(g), "occ_median": float(g["occupancy_log2"].median()),
               "protection_median": float(g["protection_score"].median()),
               "tf_cov_median": float(g["tf_coverage"].median()),
               "gc": float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in g["seq"]])),
               "frac_near_finemo_hit": float(g["near_finemo_hit"].mean())}
        for name, pat in FAMILIES.items():
            row[name] = float(np.mean([has(s, pat) for s in g["seq"]]))
        rows.append(row)
    fam = pd.DataFrame(rows)
    fam.to_csv(OUT / "tertile_families.csv", index=False)
    log("tertiles\n" + fam.round(3).to_string(index=False))
    top.drop(columns=["seq"]).to_parquet(OUT / "top_calls_occupancy.parquet")

    # GC-matched pool: random 30-bp windows +-300 bp from calls, >= 40 bp from any call
    rng = np.random.default_rng(0)
    cand = top.sample(min(len(top), 30000), random_state=0)[["chrom", "center"]].copy()
    cand["center"] = cand["center"] + rng.integers(-300, 300, len(cand))
    allc = {ch: np.sort(g["center"].to_numpy()) for ch, g in calls.groupby("chrom")}
    ok = []
    for r in cand.itertuples():
        a = allc[r.chrom]
        j = np.searchsorted(a, r.center)
        ok.append(min(abs(r.center - a[max(0, j - 1)]), abs(r.center - a[min(len(a) - 1, j)])) >= 40)
    pool = seqs_for(pm, cand[np.asarray(ok)], -HALF, HALF)
    pool_store = SequenceStore.from_sequences(pool, device=dev)
    res = {}
    for t in ("high", "low"):
        seqs = top.loc[top["tertile"] == t, "seq"].tolist()
        disc = MotifDiscovery.from_sequences(seqs, device=dev)
        for cname, ctrl in (("shuffle", None), ("gcmatched", GCMatched(pool=pool_store))):
            ms = disc.fit(n_motifs=6, widths=range(6, 16), control=ctrl, seed=0)
            s = summary(ms)
            s.to_csv(OUT / f"streme_{t}_{cname}.csv", index=False)
            if ms:
                ann = annotate(ms, H12, top=1)
                ann.to_csv(OUT / f"streme_{t}_{cname}_h12.csv", index=False)
                s = s.assign(h12=ann.groupby("query")["target"].first().reindex(s["name"]).to_numpy()
                             if "name" in s else None)
            res[f"{t}_{cname}"] = s
            cols = [c for c in ("consensus", "width", "n_sites", "holdout_logp", "central_logp", "h12") if c in s]
            log(t, cname, "\n" + (s[cols].to_string() if len(s) else "none"))
    with open(OUT / "summary.json", "w") as fh:
        json.dump({k: v.to_dict(orient="records") for k, v in res.items()} | {"tertiles": rows}, fh, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
