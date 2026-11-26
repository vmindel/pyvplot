"""Shared inputs for the KLF5 footprint benchmark.

Ground truth is model-derived: FiNeMo hits called on ISM attributions of the AlphaGenome
KLF5FULL fine-tune, and the ISM attribution track itself. Both exist only inside the ISM
tiles (summit +/- 200 bp of 64,404 peak tiles), so every metric is restricted to those
tiles: outside them "no hit" is not a negative.

Dev / test split by chromosome: tune on even chromosomes + chrX, report on odd ones.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("FP_OUT", ROOT / "out"))
OUT.mkdir(parents=True, exist_ok=True)

PAPER = Path("/home/labs/barkailab/vovam/KLFNov2026_paper")
STORE = PAPER / "data/frags/KLF5FULL.merged.frags.npz"
ATTR_DIR = Path("/home/labs/barkailab/vovam/projects/chec_alphagenome/results/attr/KLF5FULL/ism64k_top10k/count")
HITS = ATTR_DIR / "finemo/hits/hits_unique.tsv"
TILES = ATTR_DIR / "full/peaks.narrowPeak"
GROOT = "/home/labs/barkailab/vovam/mm10/"
ATTR_TRACK = "nmumg.chec.klf.agfull_ism64k.klf5full"
TILE_HALF = 200


def split_of(chrom: str) -> str:
    c = chrom.removeprefix("chr")
    return "test" if c.isdigit() and int(c) % 2 == 1 else "dev"


def load_regions() -> pd.DataFrame:
    """ISM tiles (summit +/- 200) merged into non-overlapping regions."""
    pk = pd.read_csv(TILES, sep="\t", header=None, usecols=[0, 1, 9], names=["chrom", "start", "summit"])
    c = pk["start"] + pk["summit"]
    win = pd.DataFrame({"chrom": pk["chrom"], "start": c - TILE_HALF, "end": c + TILE_HALF})
    win = win.sort_values(["chrom", "start"]).reset_index(drop=True)
    rows = []
    for chrom, g in win.groupby("chrom", sort=True):
        s0, e0 = None, None
        for s, e in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            if s0 is None:
                s0, e0 = s, e
            elif s <= e0:
                e0 = max(e0, e)
            else:
                rows.append((chrom, s0, e0))
                s0, e0 = s, e
        rows.append((chrom, s0, e0))
    reg = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    reg["split"] = reg["chrom"].map(split_of)
    reg["region_id"] = np.arange(len(reg))
    return reg


def load_hits() -> pd.DataFrame:
    h = pd.read_csv(HITS, sep="\t").rename(columns={"chr": "chrom"})
    h["pattern"] = h["motif_name"].str.replace("_patterns.pattern_", "_", regex=False)
    h["split"] = h["chrom"].map(split_of)
    return h


def load_attr(regions: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-region 1-bp ISM attribution (NaN where the track has no value)."""
    import pymisha as pm

    pm.gsetroot(GROOT)
    out: dict[int, np.ndarray] = {}
    starts = regions.set_index("region_id")["start"]
    for _, iv in regions.groupby("chrom", sort=False):  # one chromosome at a time: max_data_size
        iv = iv[["chrom", "start", "end", "region_id"]].copy()
        df = pm.gextract(ATTR_TRACK, intervals=iv, iterator=1, colnames=["attr"], intervals_join="intervals")
        for rid, g in df.groupby("region_id", sort=False):
            rid = int(rid)
            r = regions.iloc[rid]
            a = np.full(int(r.end - r.start), np.nan, dtype=np.float32)
            a[g["start"].to_numpy() - int(starts[rid])] = g["attr"].to_numpy(dtype=np.float32)
            out[rid] = a
    for rid, r in zip(regions["region_id"], (regions["end"] - regions["start"])):
        out.setdefault(int(rid), np.full(int(r), np.nan, dtype=np.float32))
    return out


# ---------------------------------------------------------------- evaluation helpers
def assign_hits(hits: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Positive-pattern hits fully inside an ISM region, with ``region_id`` and ``c2`` (2 x centre)."""
    hits = hits[hits["pattern"].str.startswith("pos_")].copy()
    hits["c2"] = hits["start"] + hits["end"]
    rid = np.full(len(hits), -1)
    pos = np.arange(len(hits))
    for chrom, rg in regions.groupby("chrom"):
        m = (hits["chrom"] == chrom).to_numpy()
        if not m.any():
            continue
        hs, he = hits["start"].to_numpy()[m], hits["end"].to_numpy()[m]
        i = np.clip(np.searchsorted(rg["start"].to_numpy(), hs, side="right") - 1, 0, len(rg) - 1)
        inside = (hs >= rg["start"].to_numpy()[i]) & (he <= rg["end"].to_numpy()[i])
        rid[pos[m]] = np.where(inside, rg["region_id"].to_numpy()[i], -1)
    hits["region_id"] = rid
    return hits[hits["region_id"] >= 0].reset_index(drop=True)


def make_labels(regions, hits):
    lab_any = {int(r.region_id): np.zeros(r.end - r.start, bool) for r in regions.itertuples()}
    lab_p0 = {k: v.copy() for k, v in lab_any.items()}
    st = regions["start"].to_numpy()
    for h in hits.itertuples():
        a, b = h.start - st[h.region_id], h.end - st[h.region_id]
        lab_any[h.region_id][a:b] = True
        if h.pattern == "pos_0":
            lab_p0[h.region_id][a:b] = True
    return lab_any, lab_p0


def _auprc(y, s):
    from sklearn.metrics import average_precision_score
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def eval_profile(prof, regions, attr, labels, cov, terc):
    """Per-bp AUPRC (any hit / pattern_0, by coverage tercile) and per-region Spearman vs ISM."""
    from scipy.stats import spearmanr
    lab_any, lab_p0 = labels
    out = {}
    for split in ("dev", "test"):
        ids = regions.loc[regions["split"] == split, "region_id"].to_numpy()
        A = np.concatenate([attr[i] for i in ids])
        ok = ~np.isnan(A)
        s = np.concatenate([prof[i] for i in ids])[ok].astype(np.float64)
        s = np.nan_to_num(s, nan=-1e9)
        ya = np.concatenate([lab_any[i] for i in ids])[ok]
        y0 = np.concatenate([lab_p0[i] for i in ids])[ok]
        tier = np.digitize(np.concatenate([cov[i] for i in ids])[ok], terc)
        m = {"auprc_any": _auprc(ya, s), "auprc_p0": _auprc(y0, s)}
        for t in range(3):
            m[f"auprc_any_cov{t}"] = _auprc(ya[tier == t], s[tier == t])
        rho = []
        for i in ids[:: max(1, len(ids) // 4000)]:  # subsample regions for speed
            a = attr[i]
            k = ~np.isnan(a)
            if k.sum() < 100 or np.std(prof[i][k]) == 0:
                continue
            rho.append(spearmanr(prof[i][k], np.maximum(a[k], 0)).statistic)
        m["rho_attrpos_median"] = float(np.nanmedian(rho))
        out[split] = m
    return out


def eval_calls(calls, regions, hits, tol=4.0, n_thr=12, seed=0):
    """Precision/recall curve of point calls against hit centres.

    ``calls``: {region_id: array (n, 2) of [centre boundary (genomic), score]}. A call is
    correct if a positive-pattern hit centre lies within ``tol`` bp; a hit is recalled if a
    call at or above the threshold lies within ``tol``. ``rand_precision`` places the same
    calls uniformly at random within their region (chance level for that call density).
    """
    rng = np.random.default_rng(seed)
    out = {}
    st = regions["start"].to_numpy()
    en = regions["end"].to_numpy()
    for split in ("dev", "test"):
        ids = regions.loc[regions["split"] == split, "region_id"].to_numpy()
        hs = hits[hits["split"] == split]
        hc = {i: g["c2"].to_numpy() / 2.0 for i, g in hs.groupby("region_id")}
        sc, ok, rok = [], [], []
        hit_best = []
        is_p0 = []
        for i in ids:
            c = calls.get(i)
            if c is None or len(c) == 0:
                c = np.empty((0, 2))
            h = hc.get(i, np.empty(0))
            if len(c):
                sc.append(c[:, 1])
                if len(h):
                    d = np.abs(c[:, :1] - h[None, :])
                    ok.append(d.min(1) <= tol)
                else:
                    ok.append(np.zeros(len(c), bool))
                rc = rng.uniform(st[i], en[i], len(c))
                rok.append((np.abs(rc[:, None] - h[None, :]).min(1) <= tol) if len(h) else np.zeros(len(c), bool))
        for i, g in hs.groupby("region_id"):
            c = calls.get(i)
            h = g["c2"].to_numpy() / 2.0
            if c is None or len(c) == 0:
                hit_best.append(np.full(len(h), -np.inf))
            else:
                d = np.abs(h[:, None] - c[None, :, 0]) <= tol
                hit_best.append(np.where(d, c[None, :, 1], -np.inf).max(1))
            is_p0.append(g["pattern"].to_numpy() == "pos_0")
        sc = np.concatenate(sc) if sc else np.empty(0)
        ok = np.concatenate(ok) if ok else np.empty(0, bool)
        rok = np.concatenate(rok) if rok else np.empty(0, bool)
        hb = np.concatenate(hit_best)
        p0 = np.concatenate(is_p0)
        nbp = float((en[ids] - st[ids]).sum())
        thr = np.quantile(sc, np.linspace(0, 0.98, n_thr)) if len(sc) else []
        curve = []
        for t in thr:
            k = sc >= t
            curve.append({"thr": float(t), "per_kb": float(k.sum() / nbp * 1000),
                          "precision": float(ok[k].mean()), "rand_precision": float(rok[k].mean()),
                          "recall_any": float((hb >= t).mean()), "recall_p0": float((hb[p0] >= t).mean())})
        out[split] = curve
    return out
