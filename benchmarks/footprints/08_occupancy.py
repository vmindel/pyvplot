"""Option 3: occupancy from short fragments around the site, a signal cleavage bias does not make.

ChEC's readout of binding is tethered cleavage around the factor: short (TF-size) fragments
whose midpoints sit on the site and whose two ends bracket it. Scores at base ``c``:

* ``mid80`` / ``mid120``: midpoints of 20-80 / 20-120 bp fragments in c +- 2, against their
  mean density at c +- [20, 150] (local background); ``_r`` = shrunk log2 ratio, ``_llr`` =
  signed Poisson LLR (depth-aware).
* ``brk``: fragments of 20-120 bp with one end in [c-60, c-5] and the other in [c+5, c+60]
  (they bracket a ~10-bp site), against the same quantity at c +- [40, 150].
* ``summit25``: midpoints of 20-120 bp fragments in c +- 25 against the same background. A
  coarse "near a summit" detector, the occupancy counterpart of GC for depletion: a win of the
  bp scores must beat it.

Evaluation (dev chosen, test reported):

* site AUROC of pattern_0 hit centres vs non-hit positions, by coverage quintile and within
  coverage x GC bins (GC alone 0.52, depletion profile 0.70 there);
* offset from each pattern_0 hit centre to the nearest score peak (bp resolution or not);
* sequence check: sum observed and expected over all copies of the top pattern_0 9-mers at
  hits, in tiles without a hit and off-peak, then divide; plus the oriented midpoint
  aggregate off-peak (does an uncuttable core alone pull midpoints onto it?);
* re-ranking: the default profile's peaks (bp positions) scored by each occupancy score,
  through ``eval_calls``, against the profile's own score.
"""

from __future__ import annotations

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
from pyvplot.pillars import _find_peaks_1d  # noqa: E402

T0 = time.time()
CHROM_SIZES = {l.split()[0]: int(l.split()[1]) for l in open(Path(C.GROOT) / "chrom_sizes.txt")}
OFF_CHROMS = ["chr19", "chr18", "chr17", "chr16", "chr15"]
RC = str.maketrans("ACGTN", "TGCAN")
FL = 60


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def revcomp(s):
    return s.translate(RC)[::-1]


def wsum(cs, c, lo, hi):
    """Sum of arr[c+lo .. c+hi] (inclusive) from its cumsum ``cs`` (len n+1)."""
    n = len(cs) - 1
    return cs[np.clip(c + hi + 1, 0, n)] - cs[np.clip(c + lo, 0, n)]


def signed_llr(obs, exp):
    exp = np.maximum(exp, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(obs > 0, obs * np.log(np.maximum(obs, 1e-12) / exp), 0.0) - (obs - exp)
    return np.where(obs >= exp, t, -t)


class ChromSignals:
    """Per-chromosome cumulative sums of midpoint and bracket counts."""

    def __init__(self, S, E, size):
        fl = E - S
        mid = (S + E) // 2
        self.cs = {}
        for name, (lo, hi) in {"m80": (20, 80), "m120": (20, 120)}.items():
            k = (fl >= lo) & (fl <= hi)
            self.cs[name] = np.concatenate([[0.0], np.cumsum(np.bincount(mid[k], minlength=size)[:size], dtype=np.float64)])
        # bracket: positions c with S in [c-60, c-5] and E in [c+5, c+60]  ->  c in [max(S+5, E-60), min(E-5, S+60)]
        k = (fl >= 20) & (fl <= 120)
        a = np.maximum(S[k] + 5, E[k] - 60)
        b = np.minimum(E[k] - 5, S[k] + 60)
        ok = b >= a
        d = np.bincount(a[ok], minlength=size + 1)[: size + 1] - np.bincount(b[ok] + 1, minlength=size + 1)[: size + 1]
        brk = np.cumsum(d)[:size].astype(np.float64)
        self.cs["brk"] = np.concatenate([[0.0], np.cumsum(brk)])

    def scores(self, c):
        out = {}
        for name, key in (("mid80", "m80"), ("mid120", "m120")):
            cs = self.cs[key]
            obs = wsum(cs, c, -2, 2)
            bg = (wsum(cs, c, -150, 150) - wsum(cs, c, -20, 20)) / 260.0
            exp = 5 * bg
            out[f"{name}_r"] = np.log2((obs + 1) / (exp + 1))
            out[f"{name}_llr"] = signed_llr(obs, exp)
            if name == "mid120":
                obs25 = wsum(cs, c, -25, 25)
                bg25 = (wsum(cs, c, -150, 150) - obs25) / 250.0
                out["summit25_r"] = np.log2((obs25 + 1) / (51 * bg25 + 1))
                out["summit25_llr"] = signed_llr(obs25, 51 * bg25)
        cs = self.cs["brk"]
        obs = wsum(cs, c, 0, 0)
        bg = (wsum(cs, c, -150, -40) + wsum(cs, c, 40, 150)) / 222.0
        out["brk_r"] = np.log2((obs + 1) / (bg + 1))
        out["brk_llr"] = signed_llr(obs, bg)
        return out

    def raw(self, c):
        """Observed and expected counts (for aggregate-then-divide)."""
        res = {}
        for key, (lo, hi, blo, bhi, bex, nb) in {"m120": (-2, 2, -150, 150, 20, 260.0)}.items():
            cs = self.cs[key]
            res[f"{key}_obs"] = wsum(cs, c, lo, hi)
            res[f"{key}_exp"] = 5 * (wsum(cs, c, blo, bhi) - wsum(cs, c, -bex, bex)) / nb
        cs = self.cs["brk"]
        res["brk_obs"] = wsum(cs, c, 0, 0)
        res["brk_exp"] = (wsum(cs, c, -150, -40) + wsum(cs, c, 40, 150)) / 222.0
        return res


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
    store = vp.FragmentStore.load(C.STORE)
    st = regions["start"].to_numpy()
    log("inputs")

    p0 = hits[(hits["pattern"] == "pos_0") & ((hits["end"] - hits["start"]) == 9)].copy()
    seq_full = {}
    for chrom in sorted(set(regions["chrom"])):
        seq_full[chrom] = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [0], "end": [CHROM_SIZES[chrom]]}))[0].upper()
    p0["seq"] = [seq_full[h.chrom][h.start:h.end] if h.strand == "+" else revcomp(seq_full[h.chrom][h.start:h.end])
                 for h in p0.itertuples()]
    kmers = p0["seq"].value_counts().index[:4].tolist()
    log("kmers", kmers)

    names = None
    prof = {}
    seqraw = {}  # (kmer, set) -> summed obs/exp
    offpeak_mid = {}  # (kmer|random) -> oriented midpoint aggregate
    rng = np.random.default_rng(0)
    for chrom, g in regions.groupby("chrom"):
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        size = CHROM_SIZES[chrom] + 1
        sig = ChromSignals(S, E, size)
        for r in g.itertuples():
            c = np.arange(r.start, r.end)
            sc = sig.scores(c)
            names = list(sc)
            prof[r.region_id] = {k: v.astype(np.float32) for k, v in sc.items()}
        # sequence check sets
        hs = hits[hits["chrom"] == chrom]
        hit_mask = np.zeros(size, bool)
        for a, b in zip(hs["start"].to_numpy(), hs["end"].to_numpy()):
            hit_mask[max(0, a - 10): b + 10] = True
        tile = np.zeros(size, bool)
        near = np.zeros(size, bool)
        for a, b in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            tile[a:b] = True
            near[max(0, a - 2000): b + 2000] = True
        seq = seq_full[chrom]
        sets_all = []
        for km in kmers:
            pat = re.compile(f"(?=({km}|{revcomp(km)}))")
            occ = np.array([m.start() for m in pat.finditer(seq)], dtype=np.int64)
            occ = occ[(occ > 200) & (occ < size - 200)]
            if len(occ) == 0:
                continue
            strand = np.array(["+" if seq[o:o + 9] == km else "-" for o in occ])
            in_tile = tile[occ] & tile[occ + 8]
            has_hit = np.array([hit_mask[o:o + 9].any() for o in occ])
            sph = p0[(p0["chrom"] == chrom) & (p0["seq"] == km)]
            sets = {"hit": (sph["start"].to_numpy(), sph["strand"].to_numpy()),
                    "tile_nohit": (occ[in_tile & ~has_hit], strand[in_tile & ~has_hit])}
            if chrom in OFF_CHROMS:
                sel = ~near[occ]
                sets["offpeak"] = (occ[sel], strand[sel])
            sets_all.append((km, sets))
        if chrom in OFF_CHROMS:
            ok = np.flatnonzero(~near[200:size - 200]) + 200
            rnd = rng.choice(ok, size=20000, replace=False)
            sets_all.append(("random", {"offpeak": (rnd, rng.choice(["+", "-"], len(rnd)))}))
        tr = np.flatnonzero(tile & ~hit_mask)
        if len(tr):
            rnd = rng.choice(tr, size=min(5000, len(tr)), replace=False)
            sets_all.append(("random", {"tile_nohit": (rnd, rng.choice(["+", "-"], len(rnd)))}))
        cs_m = sig.cs["m120"]
        for km, sets in sets_all:
            for sname, (s0, sd) in sets.items():
                if len(s0) == 0:
                    continue
                cen = s0 + 4
                raw = sig.raw(cen)
                acc = seqraw.setdefault((km, sname), {"n": 0, **{k: 0.0 for k in raw}})
                acc["n"] += len(s0)
                for k, v in raw.items():
                    acc[k] += float(v.sum())
                if sname == "offpeak":
                    rel = np.arange(-FL, FL + 1)
                    pos = np.where(sd[:, None] == "+", cen[:, None] + rel[None, :], cen[:, None] - rel[None, :])
                    v = (cs_m[pos + 1] - cs_m[pos]).sum(0)
                    a = offpeak_mid.setdefault(km, np.zeros(len(rel)))
                    a += v
        del sig
        log(chrom)
    del store

    # ---- evaluation
    cov = {i: base[i]["cov"] for i in base}
    terc = json.load(open(C.OUT / "baseline_metrics.json"))["cov_terciles"]
    labels = C.make_labels(regions, hits)
    gc = {}
    for r in regions.itertuples():
        s = np.frombuffer(seq_full[r.chrom][r.start - 5: r.end + 5].encode(), dtype=np.uint8)
        isgc = ((s == 71) | (s == 67)).astype(np.float64)
        csg = np.concatenate([[0.0], np.cumsum(isgc)])
        n = r.end - r.start
        gc[r.region_id] = ((csg[np.arange(n) + 10] - csg[np.arange(n)]) / 10.0).astype(np.float32)
    profiles = {"gc10": gc, "tf_coverage": cov,
                "profile_default": {i: base[i]["default_prof"] for i in base},
                "profile_user": {i: base[i]["user_prof"] for i in base}}
    for nm in names:
        profiles[nm] = {i: prof[i][nm] for i in prof}

    anyhit = {i: np.zeros(r, bool) for i, r in zip(regions["region_id"], regions["end"] - regions["start"])}
    for h in hits.itertuples():
        a = max(0, h.start - st[h.region_id] - 10)
        anyhit[h.region_id][a: h.end - st[h.region_id] + 10] = True
    site = {}
    for split in ("dev", "test"):
        pos_idx = [(h.region_id, h.start - st[h.region_id] + 4) for h in p0.itertuples() if h.split == split]
        neg_idx = []
        for r in regions[regions["split"] == split].itertuples():
            ok = np.flatnonzero(~anyhit[r.region_id] & ~np.isnan(attr[r.region_id]))
            for x in (rng.choice(ok, size=min(6, len(ok)), replace=False) if len(ok) else []):
                neg_idx.append((r.region_id, int(x)))
        pc = np.array([cov[i][x] for i, x in pos_idx]); nc = np.array([cov[i][x] for i, x in neg_idx])
        pg = np.array([gc[i][x] for i, x in pos_idx]); ng = np.array([gc[i][x] for i, x in neg_idx])
        ce = np.quantile(pc, np.linspace(0, 1, 6))[1:-1]
        pcb, ncb = np.digitize(pc, ce), np.digitize(nc, ce)
        gbins = [0.55, 0.65, 0.75, 0.85]
        pgb, ngb = np.digitize(pg, gbins), np.digitize(ng, gbins)
        for name, pr in profiles.items():
            ps = np.array([pr[i][x] for i, x in pos_idx], float)
            ns = np.array([pr[i][x] for i, x in neg_idx], float)
            cell, wts = [], []
            for b in range(5):
                for g_ in range(5):
                    m1, m0 = (pcb == b) & (pgb == g_), (ncb == b) & (ngb == g_)
                    a = auroc(ps[m1], ns[m0])
                    if not np.isnan(a):
                        cell.append(a); wts.append(m1.sum())
            site.setdefault(name, {})[split] = {
                "cov_x_gc": float(np.average(cell, weights=wts)),
                "by_cov_quintile": [auroc(ps[pcb == b], ns[ncb == b]) for b in range(5)]}
    log("site auroc (cov x gc; dev / test)\n" + "\n".join(
        f"  {k:18s} {v['dev']['cov_x_gc']:.3f} / {v['test']['cov_x_gc']:.3f}   by cov (test) "
        f"{[round(x, 3) for x in v['test']['by_cov_quintile']]}" for k, v in site.items()))

    # offset hit centre -> nearest local max within +-12 (test split)
    offsets = {}
    for name, pr in profiles.items():
        if name in ("tf_coverage",):
            continue
        o = []
        for h in p0[p0["split"] == "test"].itertuples():
            c0 = h.start - st[h.region_id] + 4
            a, b = c0 - 12, c0 + 13
            if a < 0 or b > len(pr[h.region_id]):
                continue
            o.append((a + int(np.argmax(pr[h.region_id][a:b])) - c0) * (1 if h.strand == "+" else -1))
        o = np.asarray(o)
        offsets[name] = {"median": float(np.median(o)), "frac_within3": float((np.abs(o) <= 3).mean()),
                         "frac_at_window_edge": float((np.abs(o) >= 12).mean())}
    log("offsets\n" + "\n".join(f"  {k:18s} {v}" for k, v in offsets.items()))

    # sequence check: aggregate then divide
    seqcheck = []
    for (km, sname), a in sorted(seqraw.items()):
        seqcheck.append({"kmer": km, "set": sname, "n": a["n"],
                         "mid120_obs_over_exp": a["m120_obs"] / max(a["m120_exp"], 1e-9),
                         "brk_obs_over_exp": a["brk_obs"] / max(a["brk_exp"], 1e-9),
                         "mid120_obs_per_site": a["m120_obs"] / a["n"]})
    log("seqcheck\n" + pd.DataFrame(seqcheck).round(3).to_string(index=False))
    offmid = {k: (v / v[np.abs(np.arange(-FL, FL + 1)) > 30].mean()).tolist() for k, v in offpeak_mid.items()}
    log("offpeak midpoint aggregate at centre (/far flank):", {k: round(v[FL], 3) for k, v in offmid.items()})

    # re-ranking: default profile peaks (positions) scored by each occupancy score
    rerank = {}
    peaks = {}
    for i, p in profiles["profile_default"].items():
        peaks[i] = _find_peaks_1d(p.astype(np.float64), min_height=0.0, min_distance=10)
    for name in ["profile_default", "gc10", *names]:
        pr = profiles[name]
        calls = {i: np.c_[st[i] + pk + 0.5, pr[i][pk].astype(float)] for i, pk in peaks.items()}
        rerank[name] = C.eval_calls(calls, regions, hits, n_thr=16)
    def at10(curve):
        c = min(curve, key=lambda x: abs(x["per_kb"] - 5))
        return f"{c['per_kb']:.1f}/kb prec {c['precision']:.3f} rec {c['recall_any']:.3f} rec_p0 {c['recall_p0']:.3f}"
    log("rerank profile peaks (~5/kb; dev | test)\n" + "\n".join(
        f"  {k:18s} {at10(v['dev'])} | {at10(v['test'])}" for k, v in rerank.items()))

    res = {"site_auroc": site, "offsets": offsets, "seqcheck": seqcheck, "offpeak_mid_profile": offmid,
           "rerank": rerank, "kmers": kmers}
    with open(C.OUT / "occupancy_metrics.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(C.OUT / "occupancy_profiles.pkl", "wb") as fh:
        pickle.dump({k: {i: prof[i][k] for i in prof} for k in ("mid120_r", "brk_r", "mid120_llr")}, fh)
    log("done")


if __name__ == "__main__":
    main()
