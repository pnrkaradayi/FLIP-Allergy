"""
a05_dust.py
===========
Dataset 3. A household dust allergen assessment is a clinically orderable
'panel'. How much does it tell about the same person's specific IgE, compared
with the allergen panels it could replace? Its value of information for any
sIgE-based decision is bounded by this conditional information.

Reported for each sIgE with a matching dust molecule:
  R2 (10-fold cross-validated) of log sIgE on the 11-analyte dust block,
  AUROC of the dust block for a positive sIgE result,
  and, for reference, the AUROC from the other sIgE panels (latent model).

Run:  python a05_dust.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import NAME  # noqa: E402

RES = os.path.join(HERE, "results")
PAIRS = {"Der f 1": ("DSXDF1", "LBXID2"), "Der p 1": ("DSXDP1", "LBXID1"),
         "Fel d 1": ("DSXFD1", "LBXIE1"), "Can f 1": ("DSXCF1", "LBXIE5"),
         "Bla g 1": ("DSXBG1", "LBXII6"), "Bla g 2": ("DSXBG2", "LBXII6"),
         "Mus m 1": ("DSXMM1", "LBXE72"), "Rat n 1": ("DSXRN1", "LBXE74"),
         "Alt a 1": ("DSXAA1", "LBXIM6"), "Asp f 1": ("DSXAF1", "LBXIM3")}
DUST = ["DSXAA1", "DSXAF1", "DSXBG1", "DSXBG2", "DSXCF1", "DSXDF1", "DSXDP1", "DSXFD1",
        "DSXMM1", "DSXRN1", "DSXENX"]


def main():
    d1 = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    du = pd.read_csv(os.path.join(HERE, "data", "d3_dust.csv"))
    m = d1.merge(du, on="SEQN", how="inner")
    m = m[m[DUST].notna().all(axis=1)].copy()
    Xd = m[["lg_" + c for c in DUST]].values
    pred = pd.read_csv(os.path.join(RES, "sigma_prediction_by_analyte.csv"))
    ref = pred[pred.model == "factor"].set_index("analyte").pos_auroc
    kf = KFold(10, shuffle=True, random_state=0)
    rows = []
    for mol, (dv, sv) in PAIRS.items():
        y = m["lg_" + sv].values
        pos = (m["cens_" + sv].values == 0).astype(int)
        yhat = cross_val_predict(LinearRegression(), Xd, y, cv=kf)
        r2 = 1 - np.mean((y - yhat) ** 2) / np.var(y)
        ph = cross_val_predict(LogisticRegression(max_iter=2000), Xd, pos, cv=kf, method="predict_proba")[:, 1]
        r_pair = np.corrcoef(m["lg_" + dv], y)[0, 1]
        rows.append(dict(dust_molecule=mol, sIgE=NAME[sv], n=len(m), pearson_pair=r_pair, cv_r2_dust_block=r2,
                         auroc_dust_block=roc_auc_score(pos, ph), auroc_other_sIgE_panels=ref.get(NAME[sv], np.nan)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "dust_information.csv"), index=False)
    print(f"participants with all 11 dust analytes: {len(m):,}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
