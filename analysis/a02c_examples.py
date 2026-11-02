"""Regenerate the repetition-0 example files of the primary outcome (used by figures 4 and 7
and the worked example). Identical seeds to a02_risk_decision.py, so the rows reproduce the
repetition-0 rows of the main result files.

Run:  python a02c_examples.py
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a02_risk_decision import RES, run_rep  # noqa: E402

if __name__ == "__main__":
    bundle = os.path.join(RES, "_bundle_y_allergic_disease.pkl")
    jobs = [("MAR", 0, 1.0, bundle), ("MNAR", 0, 1.0, bundle)]
    with ProcessPoolExecutor(max_workers=2) as ex:
        for job, res in zip(jobs, ex.map(run_rep, jobs)):
            with open(os.path.join(RES, f"t1_examples_{job[0]}.pkl"), "wb") as fh:
                pickle.dump(res[5], fh)
            auc = [r for r in res[0] if r["score"] == "FLIP"][0]["auroc"]
            print(job[0], "rep 0 FLIP AUROC", round(auc, 4), "tau", round(res[5]["tau"], 4))
