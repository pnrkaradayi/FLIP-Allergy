"""Diagnostic: how well does the preposterior VOI predict the realised effect of
measuring each missing panel? (rep 0, MAR, test split; posterior decision rule)"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import ALL20, BLOCK_COLS, BLOCK_NAMES, COST, recorded_from_latent  # noqa: E402
from a02_risk_decision import expand, sample_block_masks  # noqa: E402

F = pickle.load(open(os.path.join(HERE, "results", "_bundle_y_allergic_disease.pkl"), "rb"))
E = pickle.load(open(os.path.join(HERE, "results", "t1_examples_MAR.pkl"), "rb"))
d = F["d"]
te = d[d.split == "test"].reset_index(drop=True)
Mb = E["Mb"]
tau, w, p = F["tau"], F["w"], len(ALL20)
Yrec = te[["lg_" + c for c in ALL20]].values
st0 = te[["cens_" + c for c in ALL20]].values.astype(int)
M = expand(Mb)
from flipx import covariate_design  # noqa: E402
Xl = covariate_design(te.RIDAGEYR, te.male, te.RIDRETH1)
Zfull = (te[F["feat"]].values - F["mu_sc"]) / F["sd_sc"]
cov_part = Zfull[:, p:] @ w[p:]
d_full = E["d_full"]

rows_all, blk_all = [], []
for j, b in enumerate(BLOCK_NAMES):
    r = np.flatnonzero(Mb[:, j])
    rows_all.append(r)
    blk_all.append(np.full(r.size, j))
R_ = np.concatenate(rows_all)
B_ = np.concatenate(blk_all)
st = st0[R_].copy()
Mb2 = Mb[R_].copy()
Mb2[np.arange(len(R_)), B_] = False
M2 = expand(Mb2)
st[M2] = 2
Zl = F["lat"]["factor"].conditional_draws(Yrec[R_], st, Xl[R_], 16, 30, 12, seed=7)
Rr = np.where(M2[:, None, :], recorded_from_latent(Zl), Yrec[R_][:, None, :])
e2 = F["b0"] + ((Rr - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + cov_part[R_][:, None]
p2 = (e2 >= tau).mean(1)
d2 = p2 >= 0.5
real_flip = (d2 != d_full[R_]).astype(float)
real_u = np.minimum(p2, 1 - p2)
pred = E["resid"][R_, B_]
df = pd.DataFrame(dict(row=R_, blk=B_, pred=pred, real_flip=real_flip, real_u=real_u))
print("Spearman(pred residual, realised posterior uncertainty):",
      round(spearmanr(df.pred, df.real_u).statistic, 3))
print("Spearman(pred residual, realised flip):", round(spearmanr(df.pred, df.real_flip).statistic, 3))
# per patient choices
uR = np.abs(w) / np.abs(w).sum()
kimval = np.array([uR[BLOCK_COLS[b]].sum() for b in BLOCK_NAMES])
g = df.groupby("row")
res = []
for row, q in g:
    if len(q) < 2:
        continue
    voi = q.loc[q.pred.idxmin()]
    kim = q.loc[q.blk.map(lambda j: -kimval[j]).idxmin()]
    oracle_u = q.real_u.min()
    res.append(dict(row=row, n=len(q), voi_flip=voi.real_flip, kim_flip=kim.real_flip,
                    voi_u=voi.real_u, kim_u=kim.real_u, oracle_u=oracle_u,
                    same=int(voi.blk == kim.blk), voi_blk=BLOCK_NAMES[int(voi.blk)],
                    kim_blk=BLOCK_NAMES[int(kim.blk)]))
res = pd.DataFrame(res)
print(res[["voi_flip", "kim_flip", "voi_u", "kim_u", "oracle_u", "same"]].mean().round(4))
print("VOI choices:", res.voi_blk.value_counts().to_dict())
print("Kim choices:", res.kim_blk.value_counts().to_dict())
print("Kim block order by sum |w|:", sorted(zip(BLOCK_NAMES, kimval.round(3)), key=lambda x: -x[1]))
# mean realised uncertainty by block among those missing it
print(df.groupby("blk").agg(pred=("pred", "mean"), real_u=("real_u", "mean"), real_flip=("real_flip", "mean"))
      .rename(index=dict(enumerate(BLOCK_NAMES))).round(4))
