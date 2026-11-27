"""Step 4b: is the KLF5 footprint occupancy or MNase sequence preference, and what is the ceiling?

1. Same-sequence control: the commonest oriented pattern_0 9-mer, aggregated (oriented cut
   density, flank-normalised) at (a) FiNeMo hits carrying it, (b) occurrences inside ISM
   tiles with no hit within 10 bp, (c) occurrences off-peak (>= 2 kb from any tile).
   Same sequence -> same cleavage preference; depth of depletion that differs is binding.
2. Sequence-only scorer: GC fraction in a 10-bp window, evaluated like the other scorers.
3. Site-level AUROC: score at pattern_0 hit centres vs non-hit positions >= 10 bp from any
   hit, in TF-coverage decile bins (fixes the clustered per-bp label issue).
4. Per-site ceiling: core/flank cut ratio at top-coverage pattern_0 hits vs matched nulls.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
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

spec = importlib.util.spec_from_file_location("s02", HERE / "02_scorers.py")
S02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S02)

T0 = time.time()
FL = 60
RC = str.maketrans("ACGTN", "TGCAN")
CHROM_SIZES = dict(l.split()[:2] for l in open("/home/labs/barkailab/vovam/mm10/chrom_sizes.txt"))
CHROM_SIZES = {k: int(v) for k, v in CHROM_SIZES.items()}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def revcomp(s):
    return s.translate(RC)[::-1]


def oriented_profile(cut, starts, ends, strands, w):
    """Mean oriented cut count at boundaries rel = -FL .. w+FL (0 = motif start boundary)."""
    rel = np.arange(-FL, w + FL + 1)
    pos = np.where(strands[:, None] == "+", starts[:, None] + rel[None, :], ends[:, None] - rel[None, :])
    ok = (pos >= 0) & (pos < len(cut))
    v = np.where(ok, cut[np.clip(pos, 0, len(cut) - 1)], 0.0)
    return v.sum(0), ok.sum(0)


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main():
    import pymisha as pm
    pm.gsetroot(C.GROOT)
    regions = C.load_regions()
    hits = C.assign_hits(C.load_hits(), regions)
    z = np.load(C.OUT / "attr_regions.npz")
    attr = {int(k[1:]): z[k] for k in z.files}
    with open(C.OUT / "baseline_scans.pkl", "rb") as fh:
        base = pickle.load(fh)
    log("inputs")

    # ---- sequences of regions (+pad) for GC scorer and k-mer search
    pad = S02.PADC
    seqs = {}
    for chrom, g in regions.groupby("chrom"):
        iv = pd.DataFrame({"chrom": chrom, "start": (g["start"] - pad).clip(lower=0), "end": g["end"] + pad})
        for rid, s in zip(g["region_id"], pm.gseq_extract(iv)):
            seqs[int(rid)] = s.upper()
    log("region seqs")

    # ---- commonest oriented 9-mer at pattern_0 hits
    p0 = hits[hits["pattern"] == "pos_0"].copy()
    p0 = p0[(p0["end"] - p0["start"]) == 9]
    st = regions["start"].to_numpy()

    def hit_seq(h):
        s = seqs[h.region_id][h.start - st[h.region_id] + pad: h.end - st[h.region_id] + pad]
        return s if h.strand == "+" else revcomp(s)
    p0["seq"] = [hit_seq(h) for h in p0.itertuples()]
    kmer_counts = p0["seq"].value_counts()
    kmer = kmer_counts.index[0]
    log("top pattern_0 9-mers", kmer_counts.head(8).to_dict())

    # ---- cut arrays per chrom (all lengths 20-300, both ends) and site sets
    store = vp.FragmentStore.load(C.STORE)
    tiles_by_chrom = {c: g for c, g in regions.groupby("chrom")}
    hit_iv = {c: g for c, g in hits.groupby("chrom")}
    agg = {k: [np.zeros(9 + 2 * FL + 1), np.zeros(9 + 2 * FL + 1)] for k in ("hit_kmer", "tile_nohit", "offpeak", "hit_all_p0")}
    n_sites = {k: 0 for k in agg}
    ceiling_rows = []
    rng = np.random.default_rng(0)
    pat = re.compile(f"(?=({kmer}|{revcomp(kmer)}))")
    off_chroms = ["chr19", "chr18", "chr17", "chr16", "chr15"]
    for chrom in store.chroms:
        if chrom not in tiles_by_chrom:
            continue
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        k = (fl >= 20) & (fl < 300)
        size = int(E.max()) + 2
        cut = (np.bincount(S[k], minlength=size) + np.bincount(E[k], minlength=size)).astype(np.float32)
        tf = (fl >= 20) & (fl <= 120)
        covd = np.bincount(S[tf], minlength=size + 1) - np.bincount(E[tf], minlength=size + 1)
        cov = np.cumsum(covd)[:-1].astype(np.float32)
        rg = tiles_by_chrom[chrom]
        hs = hit_iv.get(chrom, pd.DataFrame(columns=hits.columns))
        hit_mask = np.zeros(size, bool)
        for a, b in zip(hs["start"].to_numpy(), hs["end"].to_numpy()):
            hit_mask[max(0, a - 10): b + 10] = True
        tile_mask = np.zeros(size, bool)
        for a, b in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            tile_mask[max(0, a - 2000): b + 2000] = True

        # (a) hits carrying the kmer, and all pattern_0 hits
        ph = p0[p0["chrom"] == chrom]
        for key, sub in (("hit_kmer", ph[ph["seq"] == kmer]), ("hit_all_p0", ph)):
            if len(sub):
                sm, cn = oriented_profile(cut, sub["start"].to_numpy(), sub["end"].to_numpy(), sub["strand"].to_numpy(), 9)
                agg[key][0] += sm; agg[key][1] += cn; n_sites[key] += len(sub)
        # (b) kmer in tiles without a hit nearby
        tsites = []
        for r in rg.itertuples():
            s = seqs[r.region_id]
            for m in pat.finditer(s):
                g0 = r.start - pad + m.start()
                if g0 < r.start or g0 + 9 > r.end:
                    continue
                if hit_mask[g0:g0 + 9].any():
                    continue
                tsites.append((g0, "+" if m.group(1) == kmer else "-"))
        if tsites:
            a = np.array([t[0] for t in tsites]); sd = np.array([t[1] for t in tsites])
            sm, cn = oriented_profile(cut, a, a + 9, sd, 9)
            agg["tile_nohit"][0] += sm; agg["tile_nohit"][1] += cn; n_sites["tile_nohit"] += len(a)
        # (c) off-peak occurrences on a few chromosomes
        if chrom in off_chroms:
            clen = min(size - 1, CHROM_SIZES[chrom])
            seq = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [0], "end": [clen]}))[0].upper()
            occ = [(m.start(), "+" if m.group(1) == kmer else "-") for m in pat.finditer(seq)]
            occ = [(a, sd) for a, sd in occ if FL < a < size - FL - 10 and not tile_mask[a]]
            if occ:
                a = np.array([o[0] for o in occ]); sd = np.array([o[1] for o in occ])
                sm, cn = oriented_profile(cut, a, a + 9, sd, 9)
                agg["offpeak"][0] += sm; agg["offpeak"][1] += cn; n_sites["offpeak"] += len(a)
            del seq
        # (4) per-site ceiling: core (interior 8 boundaries) vs flanks (12 bp each side)
        cs = np.concatenate([[0.0], np.cumsum(cut, dtype=np.float64)])
        for h in ph.itertuples():
            s0, e0 = h.start, h.end
            core = cs[e0] - cs[s0 + 1]
            fla = (cs[s0 + 1] - cs[s0 - 12]) + (cs[e0 + 13] - cs[e0])
            ceiling_rows.append((chrom, s0, float(core), float(fla), float(cov[(s0 + e0) // 2]), h.hit_coefficient, 1))
        # matched null: random tile positions >= 10 bp from hits
        cand = []
        for r in rg.itertuples():
            xs = rng.integers(r.start + 15, r.end - 25, size=3)
            cand.extend(int(x) for x in xs if not hit_mask[x:x + 9].any())
        for s0 in cand:
            e0 = s0 + 9
            core = cs[e0] - cs[s0 + 1]
            fla = (cs[s0 + 1] - cs[s0 - 12]) + (cs[e0 + 13] - cs[e0])
            ceiling_rows.append((chrom, s0, float(core), float(fla), float(cov[(s0 + e0) // 2]), np.nan, 0))
        del cut, cs, cov, covd, hit_mask, tile_mask
        log(chrom, {k: n_sites[k] for k in n_sites})
    del store

    rel = np.arange(-FL, 9 + FL + 1)
    far = (np.abs(rel - 4.5) > 30)
    seqcheck = {}
    for k, (sm, cn) in agg.items():
        prof = sm / np.maximum(cn, 1)
        prof = prof / prof[far].mean()
        seqcheck[k] = {"n": n_sites[k], "core_over_flank": float(prof[(rel >= 1) & (rel <= 8)].mean()),
                       "profile": prof.tolist()}
    log("seqcheck", {k: (v["n"], round(v["core_over_flank"], 3)) for k, v in seqcheck.items()})

    ceil = pd.DataFrame(ceiling_rows, columns=["chrom", "start", "core", "flank", "cov", "coef", "is_hit"])
    ceil["ratio"] = np.log2((ceil["core"] / 8 + 0.25) / (ceil["flank"] / 26 + 0.25))
    ceil["cov_dec"] = pd.qcut(ceil["cov"].rank(method="first"), 10, labels=False)
    ceiling = []
    for d, g in ceil.groupby("cov_dec"):
        hh, nn = g[g.is_hit == 1]["ratio"], g[g.is_hit == 0]["ratio"]
        ceiling.append({"cov_decile": int(d), "cov_median": float(g["cov"].median()), "n_hit": len(hh), "n_null": len(nn),
                        "hit_frac_ratio_lt_-1": float((hh < -1).mean()), "null_frac_ratio_lt_-1": float((nn < -1).mean()),
                        "hit_median_ratio": float(hh.median()), "null_median_ratio": float(nn.median()),
                        "auroc_neg_ratio": auroc(-hh.to_numpy(), -nn.to_numpy())})
    log("ceiling", json.dumps(ceiling, indent=0))

    # ---- GC scorer + site-level AUROC for profiles
    gc = {}
    for r in regions.itertuples():
        s = np.frombuffer(seqs[r.region_id].encode(), dtype=np.uint8)
        isgc = ((s == ord("G")) | (s == ord("C"))).astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(isgc)])
        n = r.end - r.start
        c = np.arange(n) + pad  # boundary index in padded seq
        gc[r.region_id] = ((csum[np.clip(c + 5, 0, len(s))] - csum[np.clip(c - 5, 0, len(s))]) / 10.0).astype(np.float32)
    profiles = {"gc10": gc,
                "baseline_default": {i: base[i]["default_prof"] for i in base},
                "baseline_user": {i: base[i]["user_prof"] for i in base},
                "tf_coverage": {i: base[i]["cov"] for i in base}}
    cov = profiles["tf_coverage"]
    terc = json.load(open(C.OUT / "baseline_metrics.json"))["cov_terciles"]
    labels = C.make_labels(regions, hits)
    res = {"seqcheck": seqcheck, "ceiling": ceiling, "kmer": kmer, "top_kmers": kmer_counts.head(10).to_dict()}
    res["gc10_perbp"] = C.eval_profile(gc, regions, attr, labels, cov, terc)
    pk = {}
    from pyvplot.pillars import _find_peaks_1d
    for i, p in gc.items():
        q = _find_peaks_1d(p.astype(np.float64), min_height=0.0, min_distance=10)
        pk[i] = np.c_[st[i] + q, p[q] + 1e-6 * rng.random(len(q))]
    res["gc10_calls"] = C.eval_calls(pk, regions, hits)

    # site-level AUROC in coverage deciles
    anyhit = {i: np.zeros(r, bool) for i, r in zip(regions["region_id"], regions["end"] - regions["start"])}
    for h in hits.itertuples():
        a = max(0, h.start - st[h.region_id] - 10)
        anyhit[h.region_id][a: h.end - st[h.region_id] + 10] = True
    pos_idx = [(h.region_id, h.start - st[h.region_id] + 4, h.split) for h in p0.itertuples()]
    neg_idx = []
    for r in regions.itertuples():
        ok = np.flatnonzero(~anyhit[r.region_id] & ~np.isnan(attr[r.region_id]))
        if len(ok):
            for x in rng.choice(ok, size=min(4, len(ok)), replace=False):
                neg_idx.append((r.region_id, int(x), r.split))
    pos_cov = np.array([cov[i][x] for i, x, _ in pos_idx])
    neg_cov = np.array([cov[i][x] for i, x, _ in neg_idx])
    edges = np.quantile(pos_cov, np.linspace(0, 1, 11))
    pb, nb = np.clip(np.digitize(pos_cov, edges[1:-1]), 0, 9), np.clip(np.digitize(neg_cov, edges[1:-1]), 0, 9)
    psplit = np.array([s for *_, s in pos_idx]); nsplit = np.array([s for *_, s in neg_idx])
    site = {}
    for name, prof in profiles.items():
        ps = np.array([prof[i][x] for i, x, _ in pos_idx], dtype=np.float64)
        ns = np.array([prof[i][x] for i, x, _ in neg_idx], dtype=np.float64)
        rows = []
        for b in range(10):
            m1, m0 = (pb == b) & (psplit == "test"), (nb == b) & (nsplit == "test")
            rows.append(auroc(ps[m1], ns[m0]))
        site[name] = {"auroc_by_cov_decile_test": rows, "mean": float(np.nanmean(rows)),
                      "cov_edges": edges.tolist()}
    res["site_auroc"] = site
    log("site auroc", {k: (round(v["mean"], 3), [round(x, 3) for x in v["auroc_by_cov_decile_test"]]) for k, v in site.items()})
    log("gc10 perbp", res["gc10_perbp"]["test"])
    with open(C.OUT / "checks_metrics.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    ceil.to_parquet(C.OUT / "ceiling_sites.parquet")
    log("done")


if __name__ == "__main__":
    main()
