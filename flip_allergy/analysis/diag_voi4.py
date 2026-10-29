exec(open('diag_voi.py', encoding='utf8').read().split('# per patient choices')[0])
import numpy as np
from flipx import preposterior_censored, FLOOR, preposterior_residual
st_now = st0.copy(); st_now[M] = 2
Zl0 = F["lat"]["factor"].conditional_draws(Yrec, st_now, Xl, 16, 30, 12, seed=3)
R0 = np.where(M[:, None, :], recorded_from_latent(Zl0), Yrec[:, None, :])
e0 = F["b0"] + ((R0 - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + cov_part[:, None]
for name, fn in (('linear', lambda e,R,c: preposterior_residual(e,R,tau)), ('censored', lambda e,R,c: preposterior_censored(e,R,FLOOR[c],tau))):
    resid = np.full(Mb.shape, np.inf)
    for j, b in enumerate(BLOCK_NAMES):
        r = np.flatnonzero(Mb[:, j]); c = BLOCK_COLS[b]
        resid[r, j] = fn(e0[r], R0[r][:, :, c], c)
    df['pred2'] = resid[df.row.values, df.blk.values]
    out=[]
    for row, q in df.groupby('row'):
        if len(q)<2: continue
        v = q.loc[q.pred2.idxmin()]
        out.append((v.real_u, v.real_flip, q.real_u.min()))
    o=np.array(out)
    print(name, 'spearman', round(spearmanr(df.pred2, df.real_u).statistic,3), 'VOI choice real_u', o[:,0].mean().round(4), 'flip', o[:,1].mean().round(4), 'oracle', o[:,2].mean().round(4))
    print(df.groupby('blk').pred2.mean().round(4).to_dict())
