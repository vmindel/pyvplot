"""Step 1-3 of the footprint benchmark: empirical KLF5 footprint + baseline metrics.

* Aggregate fragment ends (by length x oriented position) around FiNeMo pattern_0 hits
  and around random in-tile control sites.
* Scan every ISM region with the current ``scan_locus_pillars`` (two configs) and score
  the 1-bp profile and the pair caller against the FiNeMo hits and the ISM attribution.

Run through bsub (see run_01.sh); never on the login node.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pyvplot as vp  # noqa: E402

PAD = 150
FLANK = 100
MAX_LEN = 300
CONFIGS = {
    "default": dict(len_range=(20, 300)),
    "user": dict(len_range=(20, 300), len_bin=10, flank_mode="either",
                 footprint_widths=range(6, 21), motif_width_range=(6, 20)),
}
STORE: vp.FragmentStore | None = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- aggregate
def aggregate_ends(store, sites: pd.DataFrame, wmax: int):
    """Oriented fragment-end histograms around sites.

    Oriented boundary coordinate ``rel``: 0 = motif start boundary, ``w`` = motif end
    boundary, regardless of strand. Returns (left_end, right_end) arrays of shape
    (MAX_LEN, 2*FLANK + wmax + 1) indexed [fragment length, rel + FLANK].
    """
    ncol = 2 * FLANK + wmax + 1
    ml = np.zeros(MAX_LEN * ncol, dtype=np.int64)
    mr = np.zeros(MAX_LEN * ncol, dtype=np.int64)
    for chrom, g in sites.groupby("chrom", sort=False):
        if chrom not in store.starts:
            continue
        S = store.starts[chrom].astype(np.int64)
        E = store.ends[chrom].astype(np.int64)
        idx_l, idx_r = [], []
        for s, e, st in zip(g["start"].to_numpy(), g["end"].to_numpy(), g["strand"].to_numpy()):
            lo, hi = s - FLANK, e + FLANK
            i0 = np.searchsorted(S, lo - MAX_LEN)
            i1 = np.searchsorted(S, hi + 1)
            fs, fe = S[i0:i1], E[i0:i1]
            fl = fe - fs
            k = (fl > 0) & (fl < MAX_LEN)
            fs, fe, fl = fs[k], fe[k], fl[k]
            if st == "-":
                ol, orr = e - fe, e - fs
            else:
                ol, orr = fs - s, fe - s
            for o, acc in ((ol, idx_l), (orr, idx_r)):
                m = (o >= -FLANK) & (o <= FLANK + wmax)
                acc.append(fl[m] * ncol + (o[m] + FLANK))
        if idx_l:
            ml += np.bincount(np.concatenate(idx_l), minlength=MAX_LEN * ncol)
            mr += np.bincount(np.concatenate(idx_r), minlength=MAX_LEN * ncol)
    return ml.reshape(MAX_LEN, ncol), mr.reshape(MAX_LEN, ncol)


# ---------------------------------------------------------------- scanning
def scan_region(args):
    rid, chrom, s, e = args
    L = max(0, s - PAD)
    off = s - L
    n = e - s
    res = {"region_id": rid}
    for name, kw in CONFIGS.items():
        sc = STORE.scan_locus_pillars((chrom, L, e + PAD), **kw)
        sl = slice(off, off + n)
        res[f"{name}_prof"] = sc.protection_score[sl].astype(np.float32)
        m = sc.motifs
        if len(m):
            m = m[(m["center"] >= s) & (m["center"] < e)]
            res[f"{name}_calls"] = m[["start", "end", "center", "protection_score"]].to_numpy(dtype=np.float64)
        else:
            res[f"{name}_calls"] = np.empty((0, 4))
        if name == "default":
            res["cov"] = sc.tf_coverage[sl].astype(np.float32)
            res["pillar"] = sc.pillar_score[sl].astype(np.float32)
    return res


# ---------------------------------------------------------------- metrics
def auprc(y, s):
    from sklearn.metrics import average_precision_score
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def spearman(a, b):
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main():
    global STORE
    regions = C.load_regions()
    hits = C.load_hits()
    log(f"{len(regions)} merged ISM regions, {int((regions.end - regions.start).sum())} bp; {len(hits)} hits")

    attr_cache = C.OUT / "attr_regions.npz"
    if attr_cache.exists():
        z = np.load(attr_cache)
        attr = {int(k[1:]): z[k] for k in z.files}
    else:
        attr = C.load_attr(regions)
        np.savez_compressed(attr_cache, **{f"r{k}": v for k, v in attr.items()})
    log("attr loaded", len(attr))

    STORE = vp.FragmentStore.load(C.STORE)
    log("store", STORE)

    # assign hits to regions
    hits = hits[hits["pattern"].str.startswith("pos_")].copy()
    hits["c2"] = hits["start"] + hits["end"]  # 2 x center
    reg_by_chrom = {c: g for c, g in regions.groupby("chrom")}
    rid = np.full(len(hits), -1)
    for chrom, g in hits.groupby("chrom"):
        if chrom not in reg_by_chrom:
            continue
        rg = reg_by_chrom[chrom]
        i = np.searchsorted(rg["start"].to_numpy(), g["start"].to_numpy(), side="right") - 1
        ok = (i >= 0)
        i = np.clip(i, 0, len(rg) - 1)
        inside = ok & (g["end"].to_numpy() <= rg["end"].to_numpy()[i])
        rid[hits.index.get_indexer(g.index)] = np.where(inside, rg["region_id"].to_numpy()[i], -1)
    hits["region_id"] = rid
    hits = hits[hits["region_id"] >= 0].reset_index(drop=True)
    log("hits inside regions", len(hits), hits["pattern"].value_counts().head(8).to_dict())

    # ---- empirical footprint around pattern_0
    p0 = hits[hits["pattern"] == "pos_0"]
    w0 = int((p0["end"] - p0["start"]).mode().iloc[0])
    p0 = p0[(p0["end"] - p0["start"]) == w0]
    rng = np.random.default_rng(0)
    reg_len = (regions["end"] - regions["start"]).to_numpy()
    pick = rng.choice(len(regions), size=len(p0), p=reg_len / reg_len.sum())
    cs = regions["start"].to_numpy()[pick] + (rng.random(len(p0)) * (reg_len[pick] - w0)).astype(int)
    ctrl = pd.DataFrame({"chrom": regions["chrom"].to_numpy()[pick], "start": cs, "end": cs + w0,
                         "strand": rng.choice(["+", "-"], len(p0))})
    ml, mr = aggregate_ends(STORE, p0, w0)
    cl, cr = aggregate_ends(STORE, ctrl, w0)
    p0_strong = p0[p0["hit_coefficient"] >= p0["hit_coefficient"].quantile(0.75)]
    sl_, sr_ = aggregate_ends(STORE, p0_strong, w0)
    np.savez_compressed(C.OUT / "agg_pattern0.npz", left=ml, right=mr, ctrl_left=cl, ctrl_right=cr,
                        strong_left=sl_, strong_right=sr_, w=w0, flank=FLANK, n=len(p0), n_strong=len(p0_strong))
    log(f"aggregate done: {len(p0)} pattern_0 hits (w={w0})")

    # ---- scan every region
    jobs = list(zip(regions["region_id"], regions["chrom"], regions["start"], regions["end"]))
    with mp.get_context("fork").Pool(int(__import__("os").environ.get("NPROC", 8))) as pool:
        scans = {r["region_id"]: r for r in pool.imap_unordered(scan_region, jobs, chunksize=64)}
    log("scans done", len(scans))
    with open(C.OUT / "baseline_scans.pkl", "wb") as fh:
        __import__("pickle").dump(scans, fh)

    # ---- per-bp arrays over all regions
    lab_any, lab_p0 = {}, {}
    for r in regions.itertuples():
        n = r.end - r.start
        lab_any[r.region_id] = np.zeros(n, bool)
        lab_p0[r.region_id] = np.zeros(n, bool)
    for h in hits.itertuples():
        r = regions.iloc[h.region_id]
        a, b = h.start - r.start, h.end - r.start
        lab_any[h.region_id][a:b] = True
        if h.pattern == "pos_0":
            lab_p0[h.region_id][a:b] = True

    def cat(key, split=None):
        ids = regions["region_id"] if split is None else regions.loc[regions["split"] == split, "region_id"]
        if key in ("attr",):
            return np.concatenate([attr[i] for i in ids])
        if key == "lab_any":
            return np.concatenate([lab_any[i] for i in ids])
        if key == "lab_p0":
            return np.concatenate([lab_p0[i] for i in ids])
        return np.concatenate([scans[i][key] for i in ids])

    metrics: dict = {"n_regions": len(regions), "n_hits": len(hits)}
    cov_all = cat("cov", "dev")
    terc = np.quantile(cov_all[~np.isnan(cat("attr", "dev"))], [1 / 3, 2 / 3]).tolist()
    metrics["cov_terciles"] = terc
    for split in ("dev", "test"):
        A = cat("attr", split)
        ok = ~np.isnan(A)
        cov = cat("cov", split)[ok]
        tier = np.digitize(cov, terc)
        ya, y0 = cat("lab_any", split)[ok], cat("lab_p0", split)[ok]
        scores = {k: cat(f"{k}_prof", split)[ok] for k in CONFIGS}
        scores["tf_coverage"] = cov
        scores["neg_pillar"] = -cat("pillar", split)[ok]
        scores["random"] = np.random.default_rng(1).random(ok.sum())
        m = {"n_bp": int(ok.sum()), "frac_in_hit": float(ya.mean())}
        for name, s in scores.items():
            m[f"auprc_any_{name}"] = auprc(ya, s)
            m[f"auprc_p0_{name}"] = auprc(y0, s)
            for t in range(3):
                mt = tier == t
                m[f"auprc_any_{name}_cov{t}"] = auprc(ya[mt], s[mt])
        # per-region spearman vs attribution
        ids = regions.loc[regions["split"] == split, "region_id"]
        for name in list(CONFIGS) + ["cov"]:
            rho, rho_pos = [], []
            for i in ids:
                a = attr[i]
                k = ~np.isnan(a)
                if k.sum() < 100:
                    continue
                s = scans[i][f"{name}_prof" if name != "cov" else "cov"][k]
                rho.append(spearman(s, a[k]))
                rho_pos.append(spearman(s, np.maximum(a[k], 0)))
            m[f"rho_attr_{name}_median"] = float(np.nanmedian(rho))
            m[f"rho_attrpos_{name}_median"] = float(np.nanmedian(rho_pos))
        # caller precision / recall
        hs = hits[hits["split"] == split]
        for name in CONFIGS:
            calls = [(i, c) for i in ids for c in scans[i][f"{name}_calls"]]
            n_calls = len(calls)
            hit_c2 = {i: g["c2"].to_numpy() for i, g in hs.groupby("region_id")}
            prec_ok = 0
            matched = set()
            for i, c in calls:
                hc = hit_c2.get(i)
                if hc is None:
                    continue
                d = np.abs(hc - 2 * c[2]) / 2.0  # both in boundary coordinates
                j = np.flatnonzero(d <= 4)
                if len(j):
                    prec_ok += 1
            for h in hs.itertuples():
                cc = scans[h.region_id][f"{name}_calls"]
                if len(cc) and np.min(np.abs(2 * cc[:, 2] - h.c2)) / 2.0 <= 4:
                    matched.add(h.Index)
            hs_m = hs.index.isin(list(matched))
            m[f"calls_{name}_n"] = n_calls
            m[f"calls_{name}_per_kb"] = n_calls / (m["n_bp"] / 1000)
            m[f"calls_{name}_precision4"] = prec_ok / max(1, n_calls)
            m[f"calls_{name}_recall4_any"] = float(hs_m.mean())
            m[f"calls_{name}_recall4_p0"] = float(hs_m[hs["pattern"].to_numpy() == "pos_0"].mean())
            q = pd.qcut(hs["hit_coefficient"], 4, labels=False).to_numpy()
            for t in range(4):
                m[f"calls_{name}_recall4_coefq{t}"] = float(hs_m[q == t].mean())
        metrics[split] = m

    # ---- profile aggregate + offset around pattern_0 hits (all splits)
    nrel = 2 * FLANK + w0 + 1
    agg = {k: np.zeros(nrel) for k in list(CONFIGS) + ["attr", "cov"]}
    cnt = np.zeros(nrel)
    offsets = {k: [] for k in CONFIGS}
    for h in p0.itertuples():
        r = regions.iloc[h.region_id]
        rel = np.arange(-FLANK, FLANK + w0 + 1)
        # oriented boundary rel -> base index (profile index p == base L+p)
        pos = (h.start + rel) if h.strand == "+" else (h.end - rel)
        idx = pos - r.start
        ok = (idx >= 0) & (idx < r.end - r.start)
        cnt[ok] += 1
        for k in CONFIGS:
            agg[k][ok] += scans[h.region_id][f"{k}_prof"][idx[ok]]
        agg["cov"][ok] += scans[h.region_id]["cov"][idx[ok]]
        bidx = ((h.start + rel) if h.strand == "+" else (h.end - 1 - rel)) - r.start  # bases
        okb = (bidx >= 0) & (bidx < r.end - r.start)
        agg["attr"][okb] += np.nan_to_num(attr[h.region_id][bidx[okb]])
        c = (h.start + h.end) / 2.0
        lo, hi = int(np.floor(c - 12)), int(np.ceil(c + 12))
        a0, b0 = lo - r.start, hi - r.start
        if a0 >= 0 and b0 <= r.end - r.start:
            for k in CONFIGS:
                seg = scans[h.region_id][f"{k}_prof"][a0:b0]
                p = lo + int(np.argmax(seg))  # profile index = footprint-centre boundary
                offsets[k].append((p - c) * (1 if h.strand == "+" else -1))
    for k in agg:
        agg[k] = agg[k] / np.maximum(cnt, 1)
    for k in CONFIGS:
        o = np.asarray(offsets[k])
        metrics[f"offset_p0_{k}"] = {"median": float(np.median(o)), "mean": float(o.mean()),
                                    "frac_within3": float((np.abs(o) <= 3).mean()),
                                    "hist": np.histogram(o, bins=np.arange(-12.5, 13.5))[0].tolist()}
    np.savez_compressed(C.OUT / "agg_profile_pattern0.npz", **agg, cnt=cnt, w=w0, flank=FLANK,
                        **{f"offsets_{k}": np.asarray(v) for k, v in offsets.items()})

    with open(C.OUT / "baseline_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    log(json.dumps(metrics, indent=1))
    make_figures(w0)
    log("done")


def make_figures(w0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(C.OUT / "agg_pattern0.npz")
    rel = np.arange(-FLANK, FLANK + w0 + 1)
    far = np.abs(rel - w0 / 2) > 70
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for j, (k, ttl) in enumerate((("left", "fragment LEFT ends (oriented)"), ("right", "fragment RIGHT ends (oriented)"))):
        o = z[k].astype(float)
        c = z[f"ctrl_{k}"].astype(float)
        on = o / np.maximum(o[:, far].mean(axis=1, keepdims=True), 1e-9)
        cn = c / np.maximum(c[:, far].mean(axis=1, keepdims=True), 1e-9)
        lr = np.log2((on + 0.05) / (cn + 0.05))
        im = axes[0, j].imshow(lr[20:MAX_LEN], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2,
                               extent=[rel[0] - .5, rel[-1] + .5, 20, MAX_LEN])
        axes[0, j].axvline(0, color="k", lw=.5); axes[0, j].axvline(w0, color="k", lw=.5)
        axes[0, j].set_title(f"{ttl}: log2 enrichment vs control, flank-normalised")
        axes[0, j].set_ylabel("fragment length"); axes[0, j].set_xlim(-60, 60 + w0)
        fig.colorbar(im, ax=axes[0, j])
    ax = axes[1, 0]
    for lo, hi, col in ((20, 60, "#d62828"), (60, 100, "#f77f00"), (100, 150, "#2a9d8f"), (150, 300, "#264653")):
        for k, ls in (("left", "-"), ("right", "--")):
            o = z[k][lo:hi].sum(0).astype(float)
            ax.plot(rel, o / o[far].mean(), color=col, ls=ls, lw=1,
                    label=f"{lo}-{hi} bp {k} ends")
    ax.axvline(0, color="k", lw=.5); ax.axvline(w0, color="k", lw=.5)
    ax.set_xlim(-60, 60 + w0); ax.set_title("cut density around pattern_0 (oriented), / far flank")
    ax.legend(fontsize=7, ncol=2)
    zp = np.load(C.OUT / "agg_profile_pattern0.npz")
    ax = axes[1, 1]
    for k, col in (("default", "#1b4332"), ("user", "#52b788")):
        ax.plot(rel, zp[k], color=col, label=f"profile ({k})")
    ax2 = ax.twinx()
    ax2.plot(rel, zp["attr"], color="#7b2cbf", lw=1, label="ISM attr")
    ax.axvline(0, color="k", lw=.5); ax.axvline(w0, color="k", lw=.5)
    ax.set_xlim(-40, 40 + w0); ax.set_title("mean profile / attribution around pattern_0 (oriented)")
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    fig.savefig(C.OUT / "fig_footprint_pattern0.png", dpi=130)


if __name__ == "__main__":
    main()
