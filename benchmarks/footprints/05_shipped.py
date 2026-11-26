"""Step 4d: the shipped caller (``caller="profile"``) vs ``"pairs"``, through the public API,
plus genome-wide footprint calls for de-novo motif discovery (step 5).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pyvplot as vp  # noqa: E402

PAD = 150
CONFIGS = {
    "default": dict(len_range=(20, 300)),
    "user": dict(len_range=(20, 300), flank_mode="either", footprint_widths=range(6, 21), motif_width_range=(6, 20)),
}
STORE = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def scan_region(args):
    rid, chrom, s, e = args
    L = max(0, s - PAD)
    out = {}
    for name, kw in CONFIGS.items():
        for caller in ("profile", "pairs"):
            sc = STORE.scan_locus_pillars((chrom, L, e + PAD), caller=caller, min_protection_score=0.0,
                                          min_raw_protection=0.0, **kw)
            m = sc.motifs
            m = m[(m["center"] >= s) & (m["center"] < e)]
            cen = (m["start"].to_numpy() + m["end"].to_numpy()) / 2.0 if caller == "profile" else m["center"].to_numpy()
            out[f"{name}_{caller}"] = np.c_[cen, m["protection_score"].to_numpy()]
            if caller == "profile":
                out[f"{name}_{caller}_peak"] = np.c_[m["center"].to_numpy(), m["protection_score"].to_numpy()]
    return rid, out


def main():
    global STORE
    regions = C.load_regions()
    hits = C.assign_hits(C.load_hits(), regions)
    STORE = vp.FragmentStore.load(C.STORE)
    log("loaded")
    jobs = list(zip(regions["region_id"], regions["chrom"], regions["start"], regions["end"]))
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", 8))) as pool:
        res = dict(pool.imap_unordered(scan_region, jobs, chunksize=64))
    log("scanned")
    keys = next(iter(res.values())).keys()
    metrics = {}
    for k in keys:
        calls = {i: res[i][k] for i in res}
        metrics[k] = C.eval_calls(calls, regions, hits, n_thr=16)
        c = min(metrics[k]["test"], key=lambda x: abs(x["per_kb"] - 10))
        log(k, json.dumps(c))
    with open(C.OUT / "shipped_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)

    # genome-wide calls for de novo discovery (user's settings, default thresholds)
    gw = STORE.scan_genome_pillars(**CONFIGS["user"])
    gw.to_parquet(C.OUT / "genome_calls_user.parquet")
    log("genome-wide calls", len(gw), gw["width"].describe().to_dict())


if __name__ == "__main__":
    main()
