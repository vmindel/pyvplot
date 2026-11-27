"""Depth control for 11_denovo_occupancy: occupancy tertiles within TF-coverage quintiles.

Reads ``out/denovo_occ/top_calls_occupancy.parquet``. Splits it into tertiles by occupancy
inside each tf_coverage quintile, then pools low, mid and high across quintiles, so the
pooled tertiles have matched coverage. Reports per pooled tertile: median coverage, occupancy
and protection, GC, the fraction within 4 bp of a FiNeMo hit, and motif-family rates. Also
reports each quintile separately.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

spec = importlib.util.spec_from_file_location("s11", HERE / "11_denovo_occupancy.py")
S11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S11)


def summarise(g):
    row = {"n": len(g), "tf_cov_median": float(g["tf_coverage"].median()),
           "occ_median": float(g["occupancy_log2"].median()),
           "protection_median": float(g["protection_score"].median()),
           "gc": float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in g["seq"]])),
           "frac_near_finemo_hit": float(g["near_finemo_hit"].mean())}
    for name, pat in S11.FAMILIES.items():
        row[name] = float(np.mean([S11.has(s, pat) for s in g["seq"]]))
    return row


def main():
    import pymisha as pm
    pm.gsetroot(C.GROOT)
    top = pd.read_parquet(C.OUT / "denovo_occ" / "top_calls_occupancy.parquet")
    top["seq"] = S11.seqs_for(pm, top, -S11.HALF, S11.HALF)
    top["cov_q"] = pd.qcut(top["tf_coverage"].rank(method="first"), 5, labels=False)
    top["occ_t"] = top.groupby("cov_q")["occupancy_log2"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    lab = {0: "low", 1: "mid", 2: "high"}
    pooled = pd.DataFrame([{"tertile": lab[t], **summarise(g)} for t, g in top.groupby("occ_t")])
    per_q = pd.DataFrame([{"cov_q": q, "tertile": lab[t], **summarise(g)}
                          for (q, t), g in top.groupby(["cov_q", "occ_t"])])
    pooled.to_csv(C.OUT / "denovo_occ" / "tertiles_within_cov.csv", index=False)
    per_q.to_csv(C.OUT / "denovo_occ" / "tertiles_by_cov_quintile.csv", index=False)
    pd.set_option("display.width", 250)
    print("pooled (occupancy tertiles within coverage quintiles)\n" + pooled.round(3).T.to_string())
    cols = ["cov_q", "tertile", "tf_cov_median", "occ_median", "gc", "frac_near_finemo_hit", "KLF_GGGYGKGG", "MNase_CTAG"]
    print("\nper coverage quintile\n" + per_q[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
