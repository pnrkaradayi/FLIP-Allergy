import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '.')
exec(open('diag_voi.py', encoding='utf8').read().split('rows_all, blk_all')[0])
from flipx import preposterior_residual
st_now = st0.copy(); st_now[M] = 2
for nch, keep in ((16,12),(32,16),(48,24)):
    Zl0 = F["lat"]["factor"].conditional_draws(Yrec, st_now, Xl, nch, 30 if keep<=16 else 40, keep, seed=3)
    R0 = np.where(M[:, None, :], recorded_from_latent(Zl0), Yrec[:, None, :])
    e0 = F["b0"] + ((R0 - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + cov_part[:, None]
    resid = np.full(Mb.shape, np.inf)
    for j, b in enumerate(BLOCK_NAMES):
        r = np.flatnonzero(Mb[:, j])
        resid[r, j] = preposterior_residual(e0[r], R0[r][:, :, BLOCK_COLS[b]], tau)
    np.save(f'results/_diag_resid_{nch}x{keep}.npy', resid)
    print(nch, keep, 'done', flush=True)
