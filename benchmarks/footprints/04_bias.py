"""Step 4c: self-calibrated MNase sequence bias from the KLF5 data's own off-peak cuts.

* K-mer cleavage tables (k=6, k=8) from cuts outside ISM tiles +/- 2 kb and outside any
  active window (pyvplot ``find_active_windows``): masked bases are N, so masked k-mers are
  excluded from both observed and background counts. No GFP-MNase.
* Acceptance test: oriented observed / expected cut density at the same 9-mers in hits,
  tile-no-hit and off-peak sites. Off-peak core should return to ~1 after correction.
* Bias-corrected depletion scorer (binomial LLR with expected-weighted core/flank sizes),
  evaluated like the others, plus site AUROC within coverage x GC bins (signal beyond sequence).
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
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
from pyvplot.pillars import _find_peaks_1d, find_active_windows  # noqa: E402

spec = importlib.util.spec_from_file_location("s02", HERE / "02_scorers.py")
S02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S02)

T0 = time.time()
FL = 60
PADC = S02.PADC
RC = str.maketrans("ACGTN", "TGCAN")
CHROM_SIZES = {l.split()[0]: int(l.split()[1]) for l in open(Path(C.GROOT) / "chrom_sizes.txt")}
OFF_CHROMS = ["chr19", "chr18", "chr17", "chr16", "chr15"]
WIDTHS = np.arange(6, 21)
G: dict = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def revcomp(s):
    return s.translate(RC)[::-1]


def binom_llr_q(core, flank, q0):
    """Signed binomial LLR that the core share of (core+flank) cuts is below q0."""
    n = core + flank
    with np.errstate(divide="ignore", invalid="ignore"):
        qh = np.where(n > 0, core / np.maximum(n, 1e-12), q0)
        t1 = np.where(core > 0, core * np.log(np.maximum(qh, 1e-300) / q0), 0.0)
        t2 = np.where(flank > 0, flank * np.log(np.maximum(1 - qh, 1e-300) / (1 - q0)), 0.0)
    llr = t1 + t2
    return np.where(qh < q0, llr, -llr)


def score_exp(cuts, expc, f, mode, widths):
    n = len(cuts)
    cs = np.concatenate([[0.0], np.cumsum(cuts, dtype=np.float64)])
    es = np.concatenate([[0.0], np.cumsum(expc, dtype=np.float64)])
    out = np.zeros((len(widths), n))
    c = np.arange(n)
    for wi, w in enumerate(widths):
        x = c - w // 2
        valid = (x - f >= 0) & (x + w + f + 1 <= n)
        xx = np.clip(x, f, n - w - f - 1)

        def seg(arr, a, b):
            return arr[b] - arr[a]
        co, ce = seg(cs, xx + 1, xx + w), seg(es, xx + 1, xx + w)
        lo, le = seg(cs, xx - f, xx + 1), seg(es, xx - f, xx + 1)
        ro, re_ = seg(cs, xx + w, xx + w + f + 1), seg(es, xx + w, xx + w + f + 1)
        if mode == "min":
            s = np.minimum(binom_llr_q(co, lo, ce / (ce + le)), binom_llr_q(co, ro, ce / (ce + re_)))
        else:
            s = binom_llr_q(co, lo + ro, ce / (ce + le + re_))
        out[wi] = np.where(valid, s, 0.0)
    return out


def run_region(rid):
    r = G["regions"].iloc[rid]
    n = int(r.end - r.start)
    cuts = G["cuts"][rid]
    res = {}
    for name, (tab, f, mode, widths) in G["variants"].items():
        expc = np.ones_like(cuts) if tab is None else G["exp"][tab][rid]
        s2 = score_exp(cuts, expc, f, mode, widths)
        bi = np.argmax(s2, axis=0)
        prof = s2[bi, np.arange(len(cuts))][PADC:PADC + n]
        bw = widths[bi][PADC:PADC + n]
        pk = _find_peaks_1d(prof, min_height=0.0, min_distance=10)
        cen = r.start + pk - bw[pk] // 2 + bw[pk] / 2.0
        res[name] = (prof.astype(np.float32), np.c_[cen, prof[pk]])
    return rid, res


def oriented(arr, starts, ends, strands, w):
    rel = np.arange(-FL, w + FL + 1)
    pos = np.where(strands[:, None] == "+", starts[:, None] + rel[None, :], ends[:, None] - rel[None, :])
    ok = (pos >= 0) & (pos < len(arr))
    return np.where(ok, arr[np.clip(pos, 0, len(arr) - 1)], 0.0).sum(0)


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
    log("inputs")

    # ---- genome sequence, masked copy for bias estimation
    seq_full, seq_mask = {}, {}
    n_masked = 0
    for chrom in store.chroms:
        if chrom not in CHROM_SIZES:
            continue
        s = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [0], "end": [CHROM_SIZES[chrom]]}))[0].upper()
        seq_full[chrom] = s
        b = bytearray(s.encode())
        mask = np.zeros(len(b), bool)
        rg = regions[regions["chrom"] == chrom]
        for a, e in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            mask[max(0, a - 2000): e + 2000] = True
        for a, e in find_active_windows(store.starts[chrom], store.ends[chrom]):
            mask[max(0, a - 500): e + 500] = True
        arr = np.frombuffer(b, dtype=np.uint8).copy()
        arr[mask] = ord("N")
        seq_mask[chrom] = arr.tobytes().decode()
        n_masked += int(mask.sum())
    log(f"sequences; masked {n_masked / 1e6:.1f} Mb")

    tables = {}
    for k in (6, 8):
        t = vp.KmerBiasTable.from_cuts(store, seq_mask, k=k)
        tables[f"k{k}"] = t
        with open(C.OUT / f"bias_k{k}.pkl", "wb") as fh:
            pickle.dump(t, fh)
        log(t)
    w6 = tables["k6"].weights
    alph = "ACGT"
    def dec(i, k):
        return "".join(alph[(i >> (2 * (k - 1 - j))) & 3] for j in range(k))
    order = np.argsort(w6)
    log("k6 lowest", [(dec(i, 6), round(float(w6[i]), 3)) for i in order[:8]])
    log("k6 highest", [(dec(i, 6), round(float(w6[i]), 3)) for i in order[-8:]])
    exp_fns = {name: vp.pillar_expected_cuts(t, seq_full) for name, t in tables.items()}

    # ---- same-9-mer acceptance test (obs and exp, oriented)
    p0 = hits[(hits["pattern"] == "pos_0") & ((hits["end"] - hits["start"]) == 9)].copy()
    p0["seq"] = [seq_full[h.chrom][h.start:h.end] if h.strand == "+" else revcomp(seq_full[h.chrom][h.start:h.end])
                 for h in p0.itertuples()]
    kmers = p0["seq"].value_counts().index[:4].tolist()
    log("kmers", kmers)
    ncol = 9 + 2 * FL + 1
    acc = {}
    for chrom in store.chroms:
        if chrom not in seq_full or chrom not in set(regions["chrom"]):
            continue
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        kk = (fl >= 20) & (fl < 300)
        size = CHROM_SIZES[chrom] + 1
        cut = (np.bincount(S[kk], minlength=size) + np.bincount(E[kk], minlength=size))[:size].astype(np.float64)
        exps = {name: fn(chrom, 0, size) for name, fn in exp_fns.items()}
        hs = hits[hits["chrom"] == chrom]
        hit_mask = np.zeros(size, bool)
        for a, b in zip(hs["start"].to_numpy(), hs["end"].to_numpy()):
            hit_mask[max(0, a - 10): b + 10] = True
        rg = regions[regions["chrom"] == chrom]
        tile = np.zeros(size, bool)
        near = np.zeros(size, bool)
        for a, b in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            tile[a:b] = True
            near[max(0, a - 2000): b + 2000] = True
        seq = seq_full[chrom]
        for km in kmers:
            pat = re.compile(f"(?=({km}|{revcomp(km)}))")
            occ = np.array([m.start() for m in pat.finditer(seq)], dtype=np.int64)
            if len(occ) == 0:
                continue
            strand = np.array(["+" if seq[o:o + 9] == km else "-" for o in occ])
            occ_ok = (occ > FL + 20) & (occ < size - FL - 30)
            occ, strand = occ[occ_ok], strand[occ_ok]
            in_tile = tile[occ] & tile[occ + 8]
            has_hit = np.array([hit_mask[o:o + 9].any() for o in occ])
            sph = p0[(p0["chrom"] == chrom) & (p0["seq"] == km)]
            sets = {"hit": (sph["start"].to_numpy(), sph["strand"].to_numpy()),
                    "tile_nohit": (occ[in_tile & ~has_hit], strand[in_tile & ~has_hit])}
            if chrom in OFF_CHROMS:
                sel = ~near[occ]
                sets["offpeak"] = (occ[sel], strand[sel])
            for sname, (st_, sd) in sets.items():
                if len(st_) == 0:
                    continue
                key = (km, sname)
                a = acc.setdefault(key, {"n": 0, "obs": np.zeros(ncol), **{t: np.zeros(ncol) for t in exps}})
                a["n"] += len(st_)
                a["obs"] += oriented(cut, st_, st_ + 9, sd, 9)
                for t, ev in exps.items():
                    a[t] += oriented(ev, st_, st_ + 9, sd, 9)
        del cut, exps
        log("acc", chrom)
    rel = np.arange(-FL, 9 + FL + 1)
    far = np.abs(rel - 4.5) > 30
    core = (rel >= 1) & (rel <= 8)
    accept = []
    for (km, sname), a in acc.items():
        row = {"kmer": km, "set": sname, "n": a["n"]}
        o = a["obs"] / a["obs"][far].mean()
        row["obs_core"] = float(o[core].mean())
        for t in tables:
            e = a[t] / a[t][far].mean()
            row[f"exp_core_{t}"] = float(e[core].mean())
            r = (a["obs"] / np.maximum(a[t], 1e-9))
            r = r / r[far].mean()
            row[f"obs_over_exp_core_{t}"] = float(r[core].mean())
            row[f"obs_over_exp_profile_{t}"] = r.tolist()
        row["obs_profile"] = o.tolist()
        accept.append(row)
    log("accept", json.dumps([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if "profile" not in k} for r in accept], indent=0))

    # ---- per-region cuts + expected, corrected scorers
    cuts, expr = {}, {t: {} for t in tables}
    for chrom, g in regions.groupby("chrom"):
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        kk = (fl >= 20) & (fl < 300)
        size = CHROM_SIZES[chrom] + 1
        cut = (np.bincount(S[kk], minlength=size) + np.bincount(E[kk], minlength=size))[:size].astype(np.float32)
        for r in g.itertuples():
            a = r.start - PADC
            b = r.end + PADC
            cuts[r.region_id] = cut[a:b].astype(np.float64)
            for t, fn in exp_fns.items():
                expr[t][r.region_id] = fn(chrom, a, b)
    del store
    log("region cuts/exp")
    G.update(regions=regions, cuts=cuts, exp=expr, variants={
        "raw_pooled_f12": (None, 12, "pooled", WIDTHS),
        "k6_pooled_f12": ("k6", 12, "pooled", WIDTHS),
        "k6_min_f12": ("k6", 12, "min", WIDTHS),
        "k6_pooled_f12_w10": ("k6", 12, "pooled", np.array([10])),
        "k8_pooled_f12": ("k8", 12, "pooled", WIDTHS),
        "k8_pooled_f12_w8-12": ("k8", 12, "pooled", np.arange(8, 13)),
    })
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", 8))) as pool:
        out = dict(pool.imap_unordered(run_region, regions["region_id"].tolist(), chunksize=128))
    log("scored")

    # ---- evaluation
    cov = {i: base[i]["cov"] for i in base}
    terc = json.load(open(C.OUT / "baseline_metrics.json"))["cov_terciles"]
    labels = C.make_labels(regions, hits)
    st = regions["start"].to_numpy()
    gc = {}
    for r in regions.itertuples():
        s = np.frombuffer(seq_full[r.chrom][r.start - 5: r.end + 5].encode(), dtype=np.uint8)
        isgc = ((s == 71) | (s == 67)).astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(isgc)])
        n = r.end - r.start
        gc[r.region_id] = ((cs[np.arange(n) + 10] - cs[np.arange(n)]) / 10.0).astype(np.float32)
    profiles = {"gc10": gc, "baseline_user": {i: base[i]["user_prof"] for i in base},
                **{name: {i: out[i][name][0] for i in out} for name in G["variants"]}}
    results = {"accept": accept, "variants": {}}
    for name in G["variants"]:
        calls = {i: out[i][name][1] for i in out}
        results["variants"][name] = {"profile": C.eval_profile(profiles[name], regions, attr, labels, cov, terc),
                                     "calls": C.eval_calls(calls, regions, hits)}
    # site AUROC: coverage deciles, and coverage x GC bins
    rng = np.random.default_rng(0)
    anyhit = {i: np.zeros(r, bool) for i, r in zip(regions["region_id"], regions["end"] - regions["start"])}
    for h in hits.itertuples():
        a = max(0, h.start - st[h.region_id] - 10)
        anyhit[h.region_id][a: h.end - st[h.region_id] + 10] = True
    pos_idx = [(h.region_id, h.start - st[h.region_id] + 4) for h in p0.itertuples() if h.split == "test"]
    neg_idx = []
    for r in regions[regions["split"] == "test"].itertuples():
        ok = np.flatnonzero(~anyhit[r.region_id] & ~np.isnan(attr[r.region_id]))
        for x in rng.choice(ok, size=min(6, len(ok)), replace=False) if len(ok) else []:
            neg_idx.append((r.region_id, int(x)))
    pc = np.array([cov[i][x] for i, x in pos_idx]); nc = np.array([cov[i][x] for i, x in neg_idx])
    pg = np.array([gc[i][x] for i, x in pos_idx]); ng = np.array([gc[i][x] for i, x in neg_idx])
    ce = np.quantile(pc, np.linspace(0, 1, 6))[1:-1]
    pcb, ncb = np.digitize(pc, ce), np.digitize(nc, ce)
    pgb, ngb = np.digitize(pg, [0.55, 0.65, 0.75, 0.85]), np.digitize(ng, [0.55, 0.65, 0.75, 0.85])
    site = {}
    for name, prof in profiles.items():
        ps = np.array([prof[i][x] for i, x in pos_idx], float)
        ns = np.array([prof[i][x] for i, x in neg_idx], float)
        by_cov = [auroc(ps[pcb == b], ns[ncb == b]) for b in range(5)]
        cell, wts = [], []
        for b in range(5):
            for g_ in range(5):
                m1, m0 = (pcb == b) & (pgb == g_), (ncb == b) & (ngb == g_)
                a = auroc(ps[m1], ns[m0])
                if not np.isnan(a):
                    cell.append(a); wts.append(m1.sum())
        site[name] = {"by_cov_quintile": by_cov, "cov_x_gc_weighted": float(np.average(cell, weights=wts)),
                      "by_cov_top_quintile_gc": [auroc(ps[(pcb == 4) & (pgb == g_)], ns[(ncb == 4) & (ngb == g_)]) for g_ in range(5)]}
    results["site_auroc"] = site
    log("site auroc", json.dumps({k: {kk: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for kk, v in d.items()} for k, d in site.items()}, indent=0))
    with open(C.OUT / "bias_metrics.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(C.OUT / "bias_profiles.pkl", "wb") as fh:
        pickle.dump({name: {i: out[i][name] for i in out} for name in ("k6_pooled_f12", "k8_pooled_f12")}, fh)
    log("done")


if __name__ == "__main__":
    main()
