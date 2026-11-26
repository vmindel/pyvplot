"""Bias or binding at off-peak GC boxes? Every 9-mer, strict off-peak, stratified by GC.

Strict off-peak: >= 2 kb from any ISM tile and >= 500 bp from any pyvplot active window,
on five chromosomes. For each 9-mer occurrence (canonical: k-mer and its reverse complement
pooled): cuts at the 8 interior boundaries vs 12 bp of flank either side (depletion), and
20-120 bp fragment midpoints in the centre +- 2 vs c +- [20, 150] (occupancy). Sums are
aggregated per 9-mer before dividing.

If the 9-mers that look more like the KLF site (pattern_0 PWM from the model's TF-MoDISco,
trimmed as FiNeMo trims it) are more depleted and more midpoint-enriched than other 9-mers of
the SAME GC count, the signal follows the motif, i.e. binding. If it follows GC only, it is
cleavage preference. The k=6 KmerBiasTable's expected depletion is reported alongside.
"""

from __future__ import annotations

import json
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

T0 = time.time()
K = 9
CHROM_SIZES = {l.split()[0]: int(l.split()[1]) for l in open(Path(C.GROOT) / "chrom_sizes.txt")}
OFF_CHROMS = ["chr19", "chr18", "chr17", "chr16", "chr15"]
LUT = np.full(256, -1, np.int64)
for i, b in enumerate("ACGT"):
    LUT[ord(b)] = i
MAIN_OUT = Path("/home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def kmer_codes(seq: str):
    b = LUT[np.frombuffer(seq.encode(), np.uint8)]
    n = len(b) - K + 1
    code = np.zeros(n, np.int64)
    rc = np.zeros(n, np.int64)
    valid = np.ones(n, bool)
    for j in range(K):
        x = b[j:j + n]
        valid &= x >= 0
        code = (code << 2) | np.maximum(x, 0)
        rc = rc | ((3 - np.maximum(b[K - 1 - j:K - 1 - j + n], 0)) << (2 * (K - 1 - j)))
    # rc built so that rc encodes revcomp: position K-1-j of forward -> position j of rc
    return np.where(valid, np.minimum(code, rc), -1)


def decode(c):
    return "".join("ACGT"[(c >> (2 * (K - 1 - j))) & 3] for j in range(K))


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def main():
    import h5py
    import pymisha as pm
    pm.gsetroot(C.GROOT)
    regions = C.load_regions()
    store = vp.FragmentStore.load(C.STORE)
    nk = 4 ** K
    acc = {k: np.zeros(nk) for k in ("n", "core", "flank", "mid_obs", "mid_exp", "exp_core", "exp_flank")}
    with open(MAIN_OUT / "bias_k6.pkl", "rb") as fh:
        table = pickle.load(fh)
    for chrom in OFF_CHROMS:
        size = CHROM_SIZES[chrom] + 1
        seq = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [0], "end": [CHROM_SIZES[chrom]]}))[0].upper()
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        fl = E - S
        k = (fl >= 20) & (fl < 300)
        cut = (np.bincount(S[k], minlength=size) + np.bincount(E[k], minlength=size))[:size].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(cut)])
        m = (fl >= 20) & (fl <= 120)
        mid = np.bincount(((S[m] + E[m]) // 2), minlength=size)[:size].astype(np.float64)
        ms = np.concatenate([[0.0], np.cumsum(mid)])
        ok = np.ones(size, bool)
        rg = regions[regions["chrom"] == chrom]
        for a, b in zip(rg["start"].to_numpy(), rg["end"].to_numpy()):
            ok[max(0, a - 2000): b + 2000] = False
        for a, b in find_active_windows(store.starts[chrom], store.ends[chrom]):
            ok[max(0, a - 500): b + 500] = False
        ok[:300] = False
        ok[size - 300:] = False
        codes = kmer_codes(seq)
        s0 = np.flatnonzero(ok[: len(codes)] & (codes >= 0))
        s0 = s0[ok[s0 + K - 1]]
        cd = codes[s0]
        core = cs[s0 + 9] - cs[s0 + 1]
        flank = (cs[s0 + 1] - cs[s0 - 12]) + (cs[s0 + 22] - cs[s0 + 9])
        c = s0 + 4
        mobs = ms[c + 3] - ms[c - 2]
        mexp = 5 * ((ms[c + 151] - ms[c - 150]) - (ms[c + 21] - ms[c - 20])) / 260.0
        ev = vp.pillar_expected_cuts(table, {chrom: seq})(chrom, 0, size)
        es = np.concatenate([[0.0], np.cumsum(ev)])
        ecore = es[s0 + 9] - es[s0 + 1]
        eflank = (es[s0 + 1] - es[s0 - 12]) + (es[s0 + 22] - es[s0 + 9])
        for name, w in (("n", None), ("core", core), ("flank", flank), ("mid_obs", mobs), ("mid_exp", mexp),
                        ("exp_core", ecore), ("exp_flank", eflank)):
            acc[name] += np.bincount(cd, weights=w, minlength=nk)
        log(chrom, f"{len(s0) / 1e6:.1f}M strict off-peak 9-mer positions")
        del seq, cut, cs, mid, ms, codes, ev, es

    # KLF PWM (pattern_0 CWM-derived PPM, trimmed 19:28)
    with h5py.File(C.ATTR_DIR / "modisco_results.h5", "r") as f:
        ppm = np.asarray(f["pos_patterns"]["pattern_0"]["sequence"][:])[19:28]  # (9, 4) ACGT
    pwm = np.log2((ppm + 0.01) / (1.04 * 0.25))
    keep = np.flatnonzero(acc["n"] >= 30)
    rows = []
    for c in keep:
        s = decode(int(c))
        x = np.array([LUT[ord(ch)] for ch in s])
        xr = np.array([LUT[ord(ch)] for ch in revcomp(s)])
        score = max(pwm[np.arange(K), x].sum(), pwm[np.arange(K), xr].sum())
        rows.append((s, sum(ch in "GC" for ch in s), score, acc["n"][c], acc["core"][c], acc["flank"][c],
                     acc["mid_obs"][c], acc["mid_exp"][c], acc["exp_core"][c], acc["exp_flank"][c]))
    df = pd.DataFrame(rows, columns=["kmer", "gc", "klf_pwm", "n", "core", "flank", "mid_obs", "mid_exp", "exp_core", "exp_flank"])
    # core/flank per boundary: 8 core boundaries, 26 flank boundaries
    df["depl"] = np.log2(((df.core / 8) + 1e-3) / ((df.flank / 26) + 1e-3))
    df["exp_depl"] = np.log2((df.exp_core / 8) / (df.exp_flank / 26))
    df["resid_depl"] = df["depl"] - df["exp_depl"]
    df["mid_enr"] = np.log2((df.mid_obs + 0.5) / (df.mid_exp + 0.5))
    df["cuts_per_site"] = (df.core + df.flank) / df.n
    df.to_parquet(C.OUT / "kmer_offpeak.parquet")
    log(f"{len(df)} 9-mers with >= 30 strict off-peak copies")

    from scipy.stats import spearmanr
    out = []
    for g, d in df.groupby("gc"):
        if len(d) < 50:
            continue
        q = d["klf_pwm"].quantile([0.5, 0.99])
        top = d[d.klf_pwm >= q[0.99]]
        mid_ = d[d.klf_pwm < q[0.5]]
        row = {"gc": int(g), "n_kmers": len(d),
               "rho_pwm_depl": spearmanr(d.klf_pwm, d.depl).statistic,
               "rho_pwm_resid_depl": spearmanr(d.klf_pwm, d.resid_depl).statistic,
               "rho_pwm_mid": spearmanr(d.klf_pwm, d.mid_enr).statistic,
               "rho_pwm_cuts_per_site": spearmanr(d.klf_pwm, d.cuts_per_site).statistic}
        for lab, sub in (("top1pct", top), ("below_median", mid_)):
            core, flank = sub.core.sum(), sub.flank.sum()
            row[f"{lab}_depl"] = float((core / 8) / (flank / 26))
            row[f"{lab}_exp_depl"] = float((sub.exp_core.sum() / 8) / (sub.exp_flank.sum() / 26))
            row[f"{lab}_mid_enr"] = float(sub.mid_obs.sum() / sub.mid_exp.sum())
            row[f"{lab}_cuts_per_site"] = float((core + flank) / sub.n.sum())
            row[f"{lab}_example"] = ",".join(sub.sort_values("klf_pwm", ascending=False).kmer.head(3)) if lab == "top1pct" else ""
        out.append(row)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    log("by GC count\n" + res.drop(columns=["below_median_example"]).round(3).to_string(index=False))
    with open(C.OUT / "kmer_binding_vs_bias.json", "w") as fh:
        json.dump(res.to_dict(orient="records"), fh, indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
