"""
a01_structure.py
================
Mechanistic layer: the co-sensitisation structure of the allergen panel.

1. Naive Pearson vs censoring-corrected latent correlation (training split).
2. Information redundancy of each analyte given the other nineteen.
3. Cross-reactivity factor model: loadings and parameter count.
4. Prespecified predictive comparison of four conditional models on the
   calibration split. Each panel block is hidden in turn and predicted from
   the others: naive Gaussian (censoring ignored), independence (covariates
   only), free latent covariance, cross-reactivity factor covariance.
5. Cross-modality transport: tetrachoric (latent) correlation of
   sensitisation in serum sIgE (NHANES 2005-2006) versus skin prick test
   (NHANES III) on the ten allergens the two panels share.

Run:  python a01_structure.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flipx import (ALL20, BLOCK_COLS, BLOCK_NAMES, CENS, FAMILIES, FAMILIES_V1,  # noqa: E402
                   NAME, CensoredMVN, covariate_design, family_mask,
                   recorded_from_latent)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPT = ["MIT", "CAT", "COC", "ALT", "RYE", "BER", "RAG", "THI", "WHI", "PEA"]
SPT_TO_SERUM = {"MIT": "LBXID2", "CAT": "LBXIE1", "COC": "LBXII6", "ALT": "LBXIM6",
                "RYE": "LBXIG5", "BER": "LBXIG2", "RAG": "LBXIW1", "THI": "LBXW11",
                "WHI": "LBXIT7", "PEA": "LBXF13"}


def load_d1():
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    Y = d[["lg_" + c for c in ALL20]].values
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    X = covariate_design(d.RIDAGEYR, d.male, d.RIDRETH1)
    return d, Y, st, X


def corr_of(S):
    sd = np.sqrt(np.diag(S))
    return S / np.outer(sd, sd)


class NaiveGaussian:
    """Gaussian regression on the RECORDED values, censoring ignored (v1 FLIP)."""

    def fit(self, Y, X):
        self.B = np.linalg.lstsq(X, Y, rcond=None)[0]
        R = Y - X @ self.B
        self.Sigma = R.T @ R / len(Y)
        return self

    def conditional_draws(self, Y, miss, X, K=160, seed=0):
        rng = np.random.default_rng(seed)
        Mu = X @ self.B
        out = np.repeat(Y[:, None, :], K, axis=1).astype(float)
        pats, inv = np.unique(miss, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        for k, pat in enumerate(pats):
            if not pat.any():
                continue
            rows = np.flatnonzero(inv == k)
            m, o = pat, ~pat
            A = np.linalg.solve(self.Sigma[np.ix_(o, o)], self.Sigma[np.ix_(o, m)]).T
            cond = Mu[np.ix_(rows, m)] + (Y[np.ix_(rows, o)] - Mu[np.ix_(rows, o)]) @ A.T
            L = np.linalg.cholesky(self.Sigma[np.ix_(m, m)] - A @ self.Sigma[np.ix_(o, m)])
            e = rng.standard_normal((rows.size, K, m.sum())) @ L.T
            out[np.ix_(rows, np.arange(K), np.flatnonzero(m))] = cond[:, None, :] + e
        return out


def crps_draws(draws, y):
    """CRPS estimator from draws: E|X - y| - 0.5 E|X - X'| (per row)."""
    t1 = np.abs(draws - y[:, None]).mean(1)
    s = np.sort(draws, axis=1)
    K = s.shape[1]
    i = np.arange(1, K + 1)
    t2 = (2 * (s * (2 * i - K - 1)[None, :]).sum(1) / (K * K)) / 2.0
    return t1 - t2


def main():
    os.makedirs(RES, exist_ok=True)
    d, Y, st, X = load_d1()
    tr, ca = (d.split == "train").values, (d.split == "calib").values
    names = [NAME[c] for c in ALL20]
    fmask = family_mask()

    # ---------------------------------------------------------- 1. correlation
    Rnaive = np.corrcoef(Y[tr], rowvar=False)
    t0 = time.time()
    models = {}
    for structure in ("free", "factor", "diagonal"):
        models[structure] = CensoredMVN(CENS, structure=structure, families_mask=fmask,
                                        n_iter=300, burn=100, seed=11).fit(Y[tr], st[tr], X[tr])
    models["factor_v1"] = CensoredMVN(CENS, structure="factor",
                                      families_mask=family_mask(families=FAMILIES_V1),
                                      n_iter=300, burn=100, seed=11).fit(Y[tr], st[tr], X[tr])
    print(f"latent models fitted in {time.time() - t0:.1f}s")
    Rlat = corr_of(models["free"].Sigma)
    Rfac = corr_of(models["factor"].Sigma)
    pd.DataFrame(Rnaive, index=names, columns=names).to_csv(os.path.join(RES, "corr_naive.csv"))
    pd.DataFrame(Rlat, index=names, columns=names).to_csv(os.path.join(RES, "corr_latent_free.csv"))
    pd.DataFrame(Rfac, index=names, columns=names).to_csv(os.path.join(RES, "corr_latent_factor.csv"))
    iu = np.triu_indices(len(ALL20), 1)
    pairs = pd.DataFrame({"a": [names[i] for i in iu[0]], "b": [names[j] for j in iu[1]],
                          "naive": Rnaive[iu], "latent": Rlat[iu], "factor": Rfac[iu]})
    pairs["attenuation"] = pairs.latent - pairs.naive
    pairs.sort_values("latent", ascending=False).to_csv(os.path.join(RES, "corr_pairs.csv"), index=False)
    print("\nstrongest latent pairs:")
    print(pairs.sort_values("latent", ascending=False).head(12).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nlatent minus naive over 190 pairs: median {pairs.attenuation.median():+.3f}, "
          f"{(pairs.attenuation > 0).mean():.1%} of pairs larger after correction")
    Rfac1 = corr_of(models["factor_v1"].Sigma)
    print(f"factor (v2) vs free off-diagonal RMSE {np.sqrt(np.mean((Rfac[iu] - Rlat[iu]) ** 2)):.4f}; "
          f"v1 {np.sqrt(np.mean((Rfac1[iu] - Rlat[iu]) ** 2)):.4f}")

    # ---------------------------------------------------------- 2. redundancy
    rows = []
    for nm, S in (("free", models["free"].Sigma), ("factor", models["factor"].Sigma)):
        P = np.linalg.inv(S)
        r2 = 1 - 1 / (np.diag(P) * np.diag(S))
        for j, c in enumerate(ALL20):
            blk = [b for b, v in BLOCK_COLS.items() if j in v][0]
            others = [k for k in range(len(ALL20)) if k not in BLOCK_COLS[blk]]
            Soo = S[np.ix_(others, others)]
            r2_out = float(S[j, others] @ np.linalg.solve(Soo, S[others, j]) / S[j, j])
            rows.append(dict(model=nm, analyte=NAME[c], block=blk, R2_given_all_others=r2[j],
                             R2_given_other_blocks=r2_out))
    red = pd.DataFrame(rows)
    red.to_csv(os.path.join(RES, "redundancy.csv"), index=False)
    print("\nredundancy (free latent covariance):")
    print(red[red.model == "free"].sort_values("R2_given_other_blocks", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------------------------------------------------- 3. factor model
    m = models["factor"]
    sd = np.sqrt(np.diag(m.Sigma))
    load = pd.DataFrame(m.Lam / sd[:, None], index=names, columns=list(FAMILIES))
    load["unique variance"] = m.Psi / sd ** 2
    load.to_csv(os.path.join(RES, "factor_loadings.csv"))
    npar_fac = int(fmask.sum()) + len(ALL20)
    print(f"\nfactor model parameters {npar_fac} (loadings + uniquenesses) vs "
          f"{len(ALL20) * (len(ALL20) + 1) // 2} for a free covariance")

    # ---------------------------------------------------------- 4. prediction
    naive = NaiveGaussian().fit(Y[tr], X[tr])
    Yc, stc, Xc = Y[ca], st[ca], X[ca]
    rec_true = Yc
    out = []
    for b in BLOCK_NAMES:
        cols = BLOCK_COLS[b]
        stat = stc.copy()
        stat[:, cols] = 2
        miss = np.zeros_like(stat, bool)
        miss[:, cols] = True
        preds = {}
        for nm in ("free", "factor", "factor_v1", "diagonal"):
            Z = models[nm].conditional_draws(Yc, stat, Xc, n_chains=16, n_sweeps=30, keep=10, seed=5)
            preds[nm] = recorded_from_latent(Z)[:, :, cols]
        preds["naive"] = naive.conditional_draws(Yc, miss, Xc, K=160, seed=5)[:, :, cols]
        for nm, P in preds.items():
            for q, j in enumerate(cols):
                y = rec_true[:, j]
                crps = crps_draws(P[:, :, q], y).mean()
                if j < 19:
                    pos = stc[:, j] == 0
                    ppos = (P[:, :, q] > CENS[j]).mean(1)
                    auc = roc_auc_score(pos, ppos) if 0 < pos.sum() < len(pos) else np.nan
                    pc = np.clip(ppos, 1e-3, 1 - 1e-3)
                    ll = -np.mean(pos * np.log(pc) + (~pos) * np.log(1 - pc))
                    bri = np.mean((ppos - pos) ** 2)
                    prev_pred, prev_obs = ppos.mean(), pos.mean()
                else:
                    auc = ll = bri = prev_pred = prev_obs = np.nan
                # randomised PIT on the reported scale (censored cells sit at the floor)
                rng_pit = np.random.default_rng(j)
                below = (P[:, :, q] < y[:, None] - 1e-9).mean(1)
                at = (np.abs(P[:, :, q] - y[:, None]) <= 1e-9).mean(1)
                pit = below + at * rng_pit.random(len(y))
                cov90 = np.mean((pit >= 0.05) & (pit <= 0.95))
                out.append(dict(block=b, analyte=NAME[ALL20[j]], model=nm, crps=crps,
                                pos_auroc=auc, pos_logloss=ll, pos_brier=bri,
                                pred_positive_rate=prev_pred, obs_positive_rate=prev_obs,
                                pit_central90=cov90))
        print(f"  predicted hidden block {b}", flush=True)
    pr = pd.DataFrame(out)
    pr.to_csv(os.path.join(RES, "sigma_prediction_by_analyte.csv"), index=False)
    summ = pr.groupby("model")[["crps", "pos_auroc", "pos_logloss", "pos_brier",
                                "pit_central90"]].mean()
    summ.to_csv(os.path.join(RES, "sigma_prediction_summary.csv"))
    print("\nhidden-block prediction, calibration split (mean over 20 analytes):")
    print(summ.round(4).to_string())

    # ---------------------------------------------------------- 5. transport
    d2 = pd.read_csv(os.path.join(HERE, "data", "d2_nhanes3_spt.csv"))
    st2 = np.where(d2[[f"pos_{k}" for k in SPT]].values == 1, 3, 1)
    X2 = covariate_design(d2.HSAGEIR, d2.male)
    tr2 = (d2.split == "train").values
    pm2 = CensoredMVN(np.zeros(10), structure="free", probit=True, n_iter=300, burn=100,
                      seed=3).fit(np.zeros((tr2.sum(), 10)), st2[tr2], X2[tr2])
    serum = [SPT_TO_SERUM[k] for k in SPT]
    st1 = np.where(d[["cens_" + c for c in serum]].values == 0, 3, 1)
    X1 = covariate_design(d.RIDAGEYR, d.male)
    pm1 = CensoredMVN(np.zeros(10), structure="free", probit=True, n_iter=300, burn=100,
                      seed=3).fit(np.zeros((tr.sum(), 10)), st1[tr], X1[tr])
    R1, R2 = corr_of(pm1.Sigma), corr_of(pm2.Sigma)
    iu10 = np.triu_indices(10, 1)
    rho = spearmanr(R1[iu10], R2[iu10]).statistic
    pear = np.corrcoef(R1[iu10], R2[iu10])[0, 1]
    nm10 = [NAME[c].replace("D. farinae", "Mite") for c in serum]
    tp = pd.DataFrame({"a": [nm10[i] for i in iu10[0]], "b": [nm10[j] for j in iu10[1]],
                       "serum_sIgE": R1[iu10], "skin_prick": R2[iu10]})
    tp.sort_values("serum_sIgE", ascending=False).to_csv(os.path.join(RES, "transport_pairs.csv"),
                                                         index=False)
    pd.DataFrame(R1, index=nm10, columns=nm10).to_csv(os.path.join(RES, "tetra_serum10.csv"))
    pd.DataFrame(R2, index=nm10, columns=nm10).to_csv(os.path.join(RES, "tetra_spt10.csv"))
    print(f"\ncross-modality structure (45 tetrachoric correlations): Spearman {rho:.3f}, "
          f"Pearson {pear:.3f}")
    print(tp.sort_values("serum_sIgE", ascending=False).head(10).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    with open(os.path.join(RES, "structure_summary.txt"), "w") as f:
        f.write(f"n_train={tr.sum()} n_calib={ca.sum()}\n")
        f.write(f"censored_allergen_cells_train={np.mean(st[tr][:, :19] == 1):.4f}\n")
        f.write(f"latent_minus_naive_median={pairs.attenuation.median():.4f}\n")
        f.write(f"frac_pairs_larger_after_correction={(pairs.attenuation > 0).mean():.4f}\n")
        f.write(f"factor_params={npar_fac} free_params={len(ALL20) * (len(ALL20) + 1) // 2}\n")
        f.write(f"factor_vs_free_offdiag_rmse={np.sqrt(np.mean((Rfac[iu] - Rlat[iu]) ** 2)):.4f}\n")
        f.write(f"transport_spearman={rho:.4f} transport_pearson={pear:.4f}\n")
    print("\nwrote", RES)


if __name__ == "__main__":
    main()
