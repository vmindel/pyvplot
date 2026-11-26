"""Bias table fitted with KLF motif matches masked, and the shipped occupancy track.

09 showed that off-peak KLF-like 9-mers are still weakly bound, so a k-mer table fitted on
off-peak cuts partly learns that protection as "sequence preference". Here every table is
fitted with ``KmerBiasTable.from_cuts(..., exclude=...)`` on the chromosomes NOT used for the
9-mer check (chr15-19 held out):

* ``k6_off``: ISM tiles +- 2 kb and pyvplot active windows +- 500 bp excluded (as in 04);
* ``k6_off_klf1`` / ``k6_off_klf5``: also every 9-mer whose KLF PWM score (pattern_0, both
  strands) is in the top 1% / 5% of all 4**9 9-mers, +- 10 bp;
* ``k6_off_placebo``: also as many 9-mers as klf1 per GC count, drawn from those below their
  GC count's median PWM score, +- 10 bp. Masking KLF-like 9-mers also removes GC-rich
  context; the placebo removes the same context without the motif, so only klf1 - placebo
  differences can be read as removed binding. Compare against ``k6_off``;
* ``k6_all`` (reference only): the 04 table, ``bias_k6.pkl``, fitted on all chromosomes
  including the chr15-19 positions the 9-mer check scores (in-sample there).

Checks:

1. Held-out 9-mer test (chr15-19, strict off-peak, as 09): for KLF-like (top 1% PWM within a
   GC count) vs below-median 9-mers, observed core/flank depletion against each table's
   expected depletion. If masking removes absorbed binding, the masked tables expect less
   depletion at KLF-like 9-mers and leave more of it as residual.
2. Profile benchmark through the public API (``scan_locus_pillars(expected_cuts=...)``,
   default settings, ``caller="profile"``): calls vs FiNeMo hits (eval_calls), per-bp AUPRC,
   site AUROC within coverage x GC bins.
3. ``LocusPillarScan.occupancy_score`` (pyvplot's ``occupancy_profile``) in the same site
   AUROC: it should reproduce 08's ``brk_llr``.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
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
from pyvplot.pillars import find_active_windows  # noqa: E402

spec = importlib.util.spec_from_file_location("s09", HERE / "09_kmer_binding_vs_bias.py")
S09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S09)

T0 = time.time()
K = S09.K
CHROM_SIZES = S09.CHROM_SIZES
OFF_CHROMS = S09.OFF_CHROMS
MAIN_OUT = S09.MAIN_OUT
PAD = 150
KLF_PAD = 10
G: dict = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def klf_scores():
    """Per 9-mer code: KLF PWM score (max over strands), GC count, and whether it is canonical
    (``kmer_codes`` reports min(code, revcomp code))."""
    import h5py
    with h5py.File(C.ATTR_DIR / "modisco_results.h5", "r") as f:
        ppm = np.asarray(f["pos_patterns"]["pattern_0"]["sequence"][:])[19:28]
    pwm = np.log2((ppm + 0.01) / (1.04 * 0.25))
    codes = np.arange(4 ** K, dtype=np.int64)
    dig = np.stack([(codes >> (2 * (K - 1 - j))) & 3 for j in range(K)], 1)
    rdig = (3 - dig)[:, ::-1]
    fwd = pwm[np.arange(K)[None, :], dig].sum(1)
    rc = pwm[np.arange(K)[None, :], rdig].sum(1)
    rcode = (rdig * (4 ** np.arange(K - 1, -1, -1))[None, :]).sum(1)
    gc = ((dig == 1) | (dig == 2)).sum(1)
    return np.maximum(fwd, rc), gc, codes <= rcode


def mask_sets(score9, gc9, canon):
    """Boolean sets over canonical 9-mer codes: top 1% / 5% KLF PWM, and a placebo with the
    klf1 set's GC-count mix (same number of distinct 9-mers per GC count) drawn from 9-mers
    below the median PWM score of their GC count."""
    sets = {"klf1": canon & (score9 >= np.quantile(score9, 0.99)),
            "klf5": canon & (score9 >= np.quantile(score9, 0.95))}
    rng = np.random.default_rng(1)
    plc = np.zeros_like(canon)
    for g in np.unique(gc9):
        n_g = int((sets["klf1"] & (gc9 == g)).sum())
        if n_g == 0:
            continue
        pool = np.flatnonzero(canon & (gc9 == g) & (score9 < np.median(score9[canon & (gc9 == g)])))
        plc[rng.choice(pool, size=min(n_g, len(pool)), replace=False)] = True
    sets["placebo"] = plc
    return sets


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def scan_region(args):
    rid, chrom, s, e = args
    L = max(0, s - PAD)
    R = e + PAD
    out = {}
    for name, fn in G["exp_fns"].items():
        sc = G["store"].scan_locus_pillars((chrom, L, R), expected_cuts=fn, min_protection_score=0.0,
                                           min_raw_protection=0.0, len_range=(20, 300))
        m = sc.motifs
        m = m[(m["center"] >= s) & (m["center"] < e)]
        out[name] = (sc.protection_score[s - L: e - L].astype(np.float32),
                     np.c_[m["center"].to_numpy(dtype=float), m["protection_score"].to_numpy(dtype=float)])
        if name == "none":
            out["occupancy_score"] = sc.occupancy_score[s - L: e - L].astype(np.float32)
            out["occ_at_calls"] = m["occupancy_score"].to_numpy(dtype=float)
    return rid, out


def main():
    import pymisha as pm
    pm.gsetroot(C.GROOT)
    regions = C.load_regions()
    hits = C.assign_hits(C.load_hits(), regions)
    store = vp.FragmentStore.load(C.STORE)
    score9, gc9, canon = klf_scores()
    sets = mask_sets(score9, gc9, canon)
    thr = {"klf1": float(np.quantile(score9, 0.99)), "klf5": float(np.quantile(score9, 0.95))}
    log("inputs; KLF PWM thresholds", thr, "| distinct canonical 9-mers per set", {k: int(v.sum()) for k, v in sets.items()},
        "| klf1 GC mix", np.bincount(gc9[sets["klf1"]], minlength=10).tolist(),
        "| placebo GC mix", np.bincount(gc9[sets["placebo"]], minlength=10).tolist())

    # ---- sequences and exclusion masks
    seqs, excl = {}, {v: [] for v in ("off", *(f"off_{s}" for s in sets))}
    frac = {}
    have = all((C.OUT / f"bias_k6_{v}.pkl").exists() for v in excl)
    for chrom in store.chroms:
        if chrom not in CHROM_SIZES:
            continue
        seq = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [0], "end": [CHROM_SIZES[chrom]]}))[0].upper()
        seqs[chrom] = seq
        if have:
            continue
        n = len(seq)
        base = np.zeros(n, bool)
        rg = regions[regions["chrom"] == chrom]
        for a, b in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            base[max(0, a - 2000): b + 2000] = True
        for a, b in find_active_windows(store.starts[chrom], store.ends[chrom]):
            base[max(0, a - 500): b + 500] = True
        codes = S09.kmer_codes(seq)
        masks = {"off": base}
        for v, member in sets.items():
            p = np.flatnonzero((codes >= 0) & member[np.maximum(codes, 0)])
            d = (np.bincount(np.maximum(p - KLF_PAD, 0), minlength=n + 1)
                 - np.bincount(np.minimum(p + K + KLF_PAD, n), minlength=n + 1))
            masks[f"off_{v}"] = base | (np.cumsum(d[:n]) > 0)
        del codes
        for v, mk in masks.items():
            a, b = runs(mk)
            excl[v].append(pd.DataFrame({"chrom": chrom, "start": a, "end": b}))
            frac.setdefault(v, [0, 0])
            frac[v][0] += int(mk.sum())
            frac[v][1] += n
        log(chrom, {v: f"{m.mean():.3f}" for v, m in masks.items()})
    train = [c for c in seqs if c not in OFF_CHROMS]
    tables = {}
    if have:
        for v in excl:
            with open(C.OUT / f"bias_k6_{v}.pkl", "rb") as fh:
                tables[f"k6_{v}"] = pickle.load(fh)
        frac = json.load(open(C.OUT / "bias_k6_masked_fraction.json"))
        log("loaded existing tables")
    else:
        excl = {v: pd.concat(x, ignore_index=True) for v, x in excl.items()}
        with open(C.OUT / "bias_k6_masked_fraction.json", "w") as fh:
            json.dump(frac, fh)
        for v, ex in excl.items():
            t = vp.KmerBiasTable.from_cuts(store, seqs, k=6, chroms=train, exclude=ex)
            tables[f"k6_{v}"] = t
            with open(C.OUT / f"bias_k6_{v}.pkl", "wb") as fh:
                pickle.dump(t, fh)
            log(f"k6_{v}", t, f"{len(ex)} excluded intervals")
        del excl
    log("masked fraction (all chromosomes)", {v: round(a / b, 4) for v, (a, b) in frac.items()})
    with open(MAIN_OUT / "bias_k6.pkl", "rb") as fh:
        tables["k6_all"] = pickle.load(fh)
    w = {k: t.weights for k, t in tables.items()}
    from scipy.stats import spearmanr
    log("weight Spearman vs k6_off", {k: round(float(spearmanr(w["k6_off"], x).statistic), 4) for k, x in w.items()})
    # GC-rich 6-mers: does masking raise their weight (less "MNase avoids GC")?
    gc6 = np.array([sum(((c >> (2 * (5 - j))) & 3) in (1, 2) for j in range(6)) for c in range(4 ** 6)])
    log("mean weight by 6-mer GC count", {k: [round(float(x[gc6 == g].mean()), 3) for g in range(7)] for k, x in w.items()})

    # ---- 1. held-out 9-mer test on chr15-19
    nk = 4 ** K
    acc = {k: np.zeros(nk) for k in ("n", "core", "flank")}
    for t in tables:
        acc[f"ec_{t}"] = np.zeros(nk)
        acc[f"ef_{t}"] = np.zeros(nk)
    for chrom in OFF_CHROMS:
        seq = seqs[chrom]
        size = CHROM_SIZES[chrom] + 1
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        k = (fl >= 20) & (fl < 300)
        cut = (np.bincount(S[k], minlength=size) + np.bincount(E[k], minlength=size))[:size].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(cut)])
        ok = np.ones(size, bool)
        rg = regions[regions["chrom"] == chrom]
        for a, b in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            ok[max(0, a - 2000): b + 2000] = False
        for a, b in find_active_windows(store.starts[chrom], store.ends[chrom]):
            ok[max(0, a - 500): b + 500] = False
        ok[:300] = False
        ok[size - 300:] = False
        codes = S09.kmer_codes(seq)
        s0 = np.flatnonzero(ok[: len(codes)] & (codes >= 0))
        s0 = s0[ok[s0 + K - 1]]
        cd = codes[s0]
        del codes
        acc["n"] += np.bincount(cd, minlength=nk)
        acc["core"] += np.bincount(cd, weights=cs[s0 + 9] - cs[s0 + 1], minlength=nk)
        acc["flank"] += np.bincount(cd, weights=(cs[s0 + 1] - cs[s0 - 12]) + (cs[s0 + 22] - cs[s0 + 9]), minlength=nk)
        for t, tab in tables.items():
            ev = vp.pillar_expected_cuts(tab, {chrom: seq})(chrom, 0, size)
            es = np.concatenate([[0.0], np.cumsum(ev)])
            acc[f"ec_{t}"] += np.bincount(cd, weights=es[s0 + 9] - es[s0 + 1], minlength=nk)
            acc[f"ef_{t}"] += np.bincount(cd, weights=(es[s0 + 1] - es[s0 - 12]) + (es[s0 + 22] - es[s0 + 9]), minlength=nk)
            del ev, es
        log("9-mer check", chrom, f"{len(s0) / 1e6:.1f}M positions")
        del cut, cs
    keep = np.flatnonzero(acc["n"] >= 30)
    df = pd.DataFrame({"code": keep, "klf_pwm": score9[keep], **{k: v[keep] for k, v in acc.items()}})
    df["gc"] = [sum(ch in "GC" for ch in S09.decode(int(c))) for c in keep]
    rows = []
    for g, d in df.groupby("gc"):
        if len(d) < 50:
            continue
        q = d["klf_pwm"].quantile([0.5, 0.99])
        for lab, sub in (("top1pct", d[d.klf_pwm >= q[0.99]]), ("below_median", d[d.klf_pwm < q[0.5]])):
            row = {"gc": int(g), "set": lab, "n_kmers": len(sub), "n_sites": float(sub.n.sum()),
                   "obs_depl": float((sub.core.sum() / 8) / (sub.flank.sum() / 26))}
            for t in tables:
                e_ = float((sub[f"ec_{t}"].sum() / 8) / (sub[f"ef_{t}"].sum() / 26))
                row[f"exp_{t}"] = e_
                row[f"resid_{t}"] = row["obs_depl"] / e_
            rows.append(row)
    kcheck = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    log("held-out 9-mer depletion (obs, expected per table, obs/exp)\n" + kcheck.round(3).to_string(index=False))

    # ---- 2./3. profile benchmark over ISM tiles
    G["store"] = store
    G["exp_fns"] = {"none": None, **{t: vp.pillar_expected_cuts(tab, seqs) for t, tab in tables.items()}}
    jobs = list(zip(regions["region_id"], regions["chrom"], regions["start"], regions["end"]))
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", 8))) as pool:
        res = dict(pool.imap_unordered(scan_region, jobs, chunksize=64))
    log("scanned", len(res))

    z = np.load(C.OUT / "attr_regions.npz")
    attr = {int(k[1:]): z[k] for k in z.files}
    with open(C.OUT / "baseline_scans.pkl", "rb") as fh:
        base = pickle.load(fh)
    cov = {i: base[i]["cov"] for i in base}
    terc = json.load(open(C.OUT / "baseline_metrics.json"))["cov_terciles"]
    labels = C.make_labels(regions, hits)
    st = regions["start"].to_numpy()
    results = {"thresholds": thr, "masked_fraction": {v: a / b for v, (a, b) in frac.items()},
               "kmer_check": kcheck.to_dict(orient="records"), "calls": {}, "profile": {}}
    for name in G["exp_fns"]:
        calls = {i: res[i][name][1] for i in res}
        cm = C.eval_calls(calls, regions, hits, n_thr=16)
        results["calls"][name] = cm
        results["profile"][name] = C.eval_profile({i: res[i][name][0] for i in res}, regions, attr, labels, cov, terc)
        c = min(cm["test"], key=lambda x: abs(x["per_kb"] - 10))
        log(name, "calls@~10/kb test", json.dumps(c), "| per-bp", json.dumps(results["profile"][name]["test"]))

    # site AUROC within coverage x GC bins (as 08)
    gc = {}
    for r in regions.itertuples():
        s = np.frombuffer(seqs[r.chrom][r.start - 5: r.end + 5].encode(), dtype=np.uint8)
        isgc = ((s == 71) | (s == 67)).astype(np.float64)
        csg = np.concatenate([[0.0], np.cumsum(isgc)])
        n = r.end - r.start
        gc[r.region_id] = ((csg[np.arange(n) + 10] - csg[np.arange(n)]) / 10.0).astype(np.float32)
    profiles = {"gc10": gc, "occupancy_score": {i: res[i]["occupancy_score"] for i in res},
                **{f"protection_{n}": {i: res[i][n][0] for i in res} for n in G["exp_fns"]}}
    p0 = hits[(hits["pattern"] == "pos_0") & ((hits["end"] - hits["start"]) == 9)]
    anyhit = {i: np.zeros(r, bool) for i, r in zip(regions["region_id"], regions["end"] - regions["start"])}
    for h in hits.itertuples():
        a = max(0, h.start - st[h.region_id] - 10)
        anyhit[h.region_id][a: h.end - st[h.region_id] + 10] = True
    rng = np.random.default_rng(0)
    site = {}
    from sklearn.metrics import roc_auc_score

    def auroc(p, q):
        if len(p) < 20 or len(q) < 20:
            return float("nan")
        return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))

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
            site.setdefault(name, {})[split] = {"cov_x_gc": float(np.average(cell, weights=wts)),
                                                "overall": auroc(ps, ns)}
    results["site_auroc"] = site
    log("site AUROC (cov x gc; dev / test; overall test)\n" + "\n".join(
        f"  {k:22s} {v['dev']['cov_x_gc']:.3f} / {v['test']['cov_x_gc']:.3f}   {v['test']['overall']:.3f}"
        for k, v in site.items()))

    # occupancy at calls: are calls near a hit more occupied than calls away from hits?
    occ = np.concatenate([res[i]["occ_at_calls"] for i in res])
    log("occupancy_score at default calls: quantiles", np.round(np.quantile(occ, [0.1, 0.5, 0.9]), 2).tolist())
    with open(C.OUT / "masked_bias_metrics.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
