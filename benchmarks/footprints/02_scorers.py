"""Step 4a: candidate count-based footprint scorers vs the current profile/caller.

Scorer family: for each footprint centre boundary c and width w, the footprint occupies
bases [x, x+w) with x = c - w//2; core = cuts at the interior boundaries x+1..x+w-1,
flanks = cuts at the f+1 boundaries on each side (edges included, where the measured
pillars sit). Statistic: signed binomial log-likelihood ratio that the core cut rate is
below the flank rate ("llr"), or the shrunk log2 rate ratio ("ratio"). "min" requires
depletion against both flanks (FOS-like), "pooled" uses both flanks together. Profile =
max over widths 6..20; calls = profile peaks.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pyvplot as vp  # noqa: E402
from pyvplot.pillars import _find_peaks_1d  # noqa: E402

PADC = 60
WIDTHS = np.arange(6, 21)
LEN_CLASSES = {"all": (20, 300), "short": (20, 120)}
T0 = time.time()
CUTS: dict[int, dict[str, np.ndarray]] = {}
REG: pd.DataFrame | None = None


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def binom_llr(core, nc, flank, nf):
    """Signed LLR (nats) that core rate < flank rate given core+flank counts."""
    core = core.astype(np.float64)
    flank = flank.astype(np.float64)
    n = core + flank
    q0 = nc / (nc + nf)
    with np.errstate(divide="ignore", invalid="ignore"):
        qh = np.where(n > 0, core / np.maximum(n, 1e-12), q0)
        t1 = np.where(core > 0, core * np.log(np.maximum(qh, 1e-300) / q0), 0.0)
        t2 = np.where(flank > 0, flank * np.log(np.maximum(1 - qh, 1e-300) / (1 - q0)), 0.0)
    llr = t1 + t2
    return np.where(qh < q0, llr, -llr)


def score_2d(cuts, f, mode, stat, widths=WIDTHS):
    n = len(cuts)
    cs = np.concatenate([[0.0], np.cumsum(cuts, dtype=np.float64)])
    out = np.zeros((len(widths), n))
    c = np.arange(n)
    for wi, w in enumerate(widths):
        x = c - w // 2
        valid = (x - f >= 0) & (x + w + f + 1 <= n)
        xx = np.clip(x, f, n - w - f - 1)
        core = cs[xx + w] - cs[xx + 1]
        lf = cs[xx + 1] - cs[xx - f]
        rf = cs[xx + w + f + 1] - cs[xx + w]
        nc, nf = w - 1, f + 1
        if stat == "llr":
            if mode == "min":
                s = np.minimum(binom_llr(core, nc, lf, nf), binom_llr(core, nc, rf, nf))
            else:
                s = binom_llr(core, nc, lf + rf, 2 * nf)
        else:
            a = 0.5
            rc = core / nc
            if mode == "min":
                s = np.log2((np.minimum(lf, rf) / nf + a) / (rc + a))
            else:
                s = np.log2(((lf + rf) / (2 * nf) + a) / (rc + a))
        out[wi] = np.where(valid, s, 0.0)
    return out


VARIANTS = {}
for lc in LEN_CLASSES:
    for f in (6, 12):
        for mode in ("min", "pooled"):
            for stat in ("llr", "ratio"):
                VARIANTS[f"{stat}_{mode}_{lc}_f{f}"] = dict(lc=lc, f=f, mode=mode, stat=stat, widths=WIDTHS)
VARIANTS["llr_min_all_f6_w9"] = dict(lc="all", f=6, mode="min", stat="llr", widths=np.array([9]))
VARIANTS["llr_min_all_f6_w8-12"] = dict(lc="all", f=6, mode="min", stat="llr", widths=np.arange(8, 13))


def run_region(rid):
    r = REG.iloc[rid]
    n = int(r.end - r.start)
    res = {}
    for name, v in VARIANTS.items():
        cuts = CUTS[rid][v["lc"]]
        s2 = score_2d(cuts, v["f"], v["mode"], v["stat"], v["widths"])
        bi = np.argmax(s2, axis=0)
        prof_full = s2[bi, np.arange(len(cuts))]
        prof = prof_full[PADC:PADC + n]
        bw = v["widths"][bi][PADC:PADC + n]
        pk = _find_peaks_1d(prof, min_height=0.0, min_distance=10)
        cen = r.start + pk - bw[pk] // 2 + bw[pk] / 2.0
        res[name] = (prof.astype(np.float32), np.c_[cen, prof[pk]], bw.astype(np.int16))
    return rid, res


def main():
    global REG
    regions = C.load_regions()
    REG = regions
    hits = C.assign_hits(C.load_hits(), regions)
    z = np.load(C.OUT / "attr_regions.npz")
    attr = {int(k[1:]): z[k] for k in z.files}
    with open(C.OUT / "baseline_scans.pkl", "rb") as fh:
        base = pickle.load(fh)
    cov = {i: base[i]["cov"] for i in base}
    terc = json.load(open(C.OUT / "baseline_metrics.json"))["cov_terciles"]
    labels = C.make_labels(regions, hits)
    log("inputs loaded")

    store = vp.FragmentStore.load(C.STORE)
    for chrom, g in regions.groupby("chrom"):
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        size = int(E.max()) + 2
        arrs = {}
        for lc, (lo, hi) in LEN_CLASSES.items():
            k = (fl >= lo) & (fl < hi)
            arrs[lc] = (np.bincount(S[k], minlength=size) + np.bincount(E[k], minlength=size)).astype(np.float32)
        for r in g.itertuples():
            n = r.end - r.start + 2 * PADC
            a = r.start - PADC
            d = {}
            for lc, arr in arrs.items():
                v = np.zeros(n, np.float32)
                lo, hi = max(0, a), min(len(arr), a + n)
                v[lo - a:hi - a] = arr[lo:hi]
                d[lc] = v
            CUTS[r.region_id] = d
        del arrs
    del store
    log("cuts built")

    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", 8))) as pool:
        out = dict(pool.imap_unordered(run_region, regions["region_id"].tolist(), chunksize=128))
    log("scored")

    results = {}
    # baselines through the same evaluators
    bprof = {i: base[i]["default_prof"] for i in base}
    bcalls = {i: np.c_[base[i]["default_calls"][:, 2], base[i]["default_calls"][:, 3]] for i in base}
    results["baseline_default"] = {"profile": C.eval_profile(bprof, regions, attr, labels, cov, terc),
                                   "calls": C.eval_calls(bcalls, regions, hits)}
    pk_calls = {}
    for i, p in bprof.items():
        pk = _find_peaks_1d(p.astype(np.float64), min_height=0.0, min_distance=10)
        pk_calls[i] = np.c_[regions["start"].iloc[i] + pk, p[pk]]
    results["baseline_default_peaks"] = {"calls": C.eval_calls(pk_calls, regions, hits)}
    results["tf_coverage"] = {"profile": C.eval_profile(cov, regions, attr, labels, cov, terc)}
    log("baselines evaluated")
    for name in VARIANTS:
        prof = {i: out[i][name][0] for i in out}
        calls = {i: out[i][name][1] for i in out}
        results[name] = {"profile": C.eval_profile(prof, regions, attr, labels, cov, terc),
                         "calls": C.eval_calls(calls, regions, hits)}
        # width chosen at pattern_0 hit centres
        bw = []
        for h in hits[hits["pattern"] == "pos_0"].itertuples():
            c = (h.start + h.end) // 2 - regions["start"].iloc[h.region_id]
            bw.append(out[h.region_id][name][2][c])
        results[name]["best_width_at_p0"] = pd.Series(bw).value_counts().sort_index().to_dict()
        log(name, json.dumps(results[name]["profile"]["dev"]))
    with open(C.OUT / "scorers_metrics.json", "w") as fh:
        json.dump(results, fh, indent=1, default=int)
    with open(C.OUT / "scorers_profiles.pkl", "wb") as fh:
        pickle.dump({"llr_min_all_f6": {i: out[i]["llr_min_all_f6"] for i in out}}, fh)
    log("done")


if __name__ == "__main__":
    main()
