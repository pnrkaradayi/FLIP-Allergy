"""
flipx.py
========
Core library for FLIP applied to allergen-specific IgE panels.

Model
-----
Each participant has p = 20 analytes (19 allergen-specific IgE + total IgE) on
the log10 scale. The latent concentration vector follows a multivariate
normal regression

    z_i = B' x_i + e_i,      e_i ~ N(0, Sigma),

where x_i holds always-available covariates (age, sex, race/ethnicity). The
assay reports z_ij exactly when it is above the lower limit of detection
c_j and otherwise only the bound z_ij <= c_j (left-censoring; 88% of allergen
cells in NHANES 2005-2006). Sigma is either free or constrained to a
cross-reactivity factor structure Sigma = Lambda Lambda' + Psi, in which each
factor loads only on the members of one cross-reactive protein family.

Fitting is by stochastic EM with Gibbs data augmentation: censored cells are
imputed from their truncated conditionals and the parameters are re-estimated
from the completed data. Deployment-time inference for a new participant
draws the censored and the not-ordered cells jointly by a blocked Gibbs
sampler, which is exact under the model (no moment matching).

Cell status codes used throughout
---------------------------------
    0  observed exactly
    1  observed but below the detection limit   (z <= c)
    2  not ordered / missing                    (unconstrained)
    3  known positive, value not used           (z > c; probit-type data)
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

# ----------------------------------------------------------------------------
# Panel definition (NHANES 2005-2006)
# ----------------------------------------------------------------------------
SIGE = ["LBXID2", "LBXID1", "LBXIE1", "LBXIE5", "LBXII6", "LBXIM6", "LBXF13",
        "LBXIF1", "LBXIF2", "LBXIW1", "LBXIG5", "LBXIG2", "LBXIT7", "LBXIT3",
        "LBXF24", "LBXIM3", "LBXW11", "LBXE72", "LBXE74"]
ALL20 = SIGE + ["LBXIGE"]
NAME = {"LBXID2": "D. farinae", "LBXID1": "D. pteronyssinus", "LBXIE1": "Cat",
        "LBXIE5": "Dog", "LBXII6": "Cockroach", "LBXIM6": "Alternaria",
        "LBXF13": "Peanut", "LBXIF1": "Egg", "LBXIF2": "Milk", "LBXIW1": "Ragweed",
        "LBXIG5": "Rye grass", "LBXIG2": "Bermuda grass", "LBXIT7": "Oak",
        "LBXIT3": "Birch", "LBXF24": "Shrimp", "LBXIM3": "Aspergillus",
        "LBXW11": "Russian thistle", "LBXE72": "Mouse", "LBXE74": "Rat",
        "LBXIGE": "Total IgE"}
CENS = np.array([np.log10(0.35)] * 19 + [np.log10(2.0)])     # detection limits
FLOOR = np.array([np.log10(0.25)] * 19 + [np.log10(1.41)])   # NHANES fill values

BLOCKS = {
    "mite": ["LBXID2", "LBXID1"],
    "pet": ["LBXIE1", "LBXIE5"],
    "pest": ["LBXII6", "LBXE72", "LBXE74"],
    "grass": ["LBXIG5", "LBXIG2"],
    "tree": ["LBXIT7", "LBXIT3"],
    "weed": ["LBXIW1", "LBXW11"],
    "mould": ["LBXIM6", "LBXIM3"],
    "food": ["LBXF13", "LBXIF1", "LBXIF2", "LBXF24"],
    "totige": ["LBXIGE"],
}
BLOCK_NAMES = list(BLOCKS)
BLOCK_COLS = {b: [ALL20.index(c) for c in v] for b, v in BLOCKS.items()}
COST = np.array([len(v) for v in BLOCKS.values()], float)   # one unit per assay

# Cross-reactive protein families (sparsity pattern of the structured model).
# FAMILIES_V1 is the prespecified set; FAMILIES (v2) adds two textbook
# molecular-allergology facts: the peanut PR-10 protein Ara h 8 (Bet v 1
# homologue) and the pollen/plant-food panallergens (profilins, cross-reactive
# carbohydrate determinants) shared by grass, tree and weed pollen and peanut.
FAMILIES_V1 = {
    "atopy (global)": ALL20,
    "mite groups 1/2": ["LBXID2", "LBXID1"],
    "PR-10 (Fagales)": ["LBXIT7", "LBXIT3"],
    "grass groups 1/5": ["LBXIG5", "LBXIG2"],
    "weed pollen": ["LBXIW1", "LBXW11"],
    "lipocalins/albumins": ["LBXIE1", "LBXIE5", "LBXE72", "LBXE74"],
    "tropomyosin": ["LBXF24", "LBXII6", "LBXID2", "LBXID1"],
    "moulds": ["LBXIM6", "LBXIM3"],
    "food storage proteins": ["LBXF13", "LBXIF1", "LBXIF2"],
}
FAMILIES = {("PR-10 (Bet v 1 homologues)" if k == "PR-10 (Fagales)" else k):
            (["LBXIT7", "LBXIT3", "LBXF13"] if k == "PR-10 (Fagales)" else v)
            for k, v in FAMILIES_V1.items()}
FAMILIES["pollen panallergens"] = ["LBXIG5", "LBXIG2", "LBXIT7", "LBXIT3",
                                   "LBXIW1", "LBXW11", "LBXF13"]


def family_mask(cols=ALL20, families=FAMILIES):
    k = list(families)
    M = np.zeros((len(cols), len(k)), bool)
    for c, key in enumerate(k):
        for v in families[key]:
            if v in cols:
                M[cols.index(v), c] = True
    return M


def covariate_design(age, male, race=None):
    """Always-available covariates for the latent mean."""
    a = np.asarray(age, float) / 10.0
    cols = [np.ones_like(a), a, a ** 2, np.log(a), np.asarray(male, float)]
    if race is not None:
        r = np.asarray(race)
        for lvl in (1, 2, 4, 5):            # reference: non-Hispanic white (3)
            cols.append((r == lvl).astype(float))
    return np.column_stack(cols)


# ----------------------------------------------------------------------------
# Truncated normal draws (vectorised, numerically safe in the tails)
# ----------------------------------------------------------------------------
def _draw_below(m, s, u, rng):
    """N(m, s^2) conditioned on z <= u."""
    a = np.clip((u - m) / s, -37.0, 37.0)
    logp = log_ndtr(a) + np.log(rng.random(np.shape(m)) + 1e-300)
    w = ndtri(np.exp(logp))
    w = np.where(np.isfinite(w), np.minimum(w, a), a)
    return m + s * w


def _draw_above(m, s, l, rng):
    """N(m, s^2) conditioned on z > l."""
    return m - s * (_draw_below(np.zeros_like(m), np.ones_like(m), (m - l) / s, rng))


# ----------------------------------------------------------------------------
# Factor analysis with a fixed zero pattern (EM on a covariance matrix)
# ----------------------------------------------------------------------------
def fa_em(S, mask, Lam=None, Psi=None, n_steps=50, min_psi=1e-3):
    """ML factor analysis S ~ Lam Lam' + diag(Psi) with Lam zero outside `mask`.

    Rubin & Thayer (1982) EM; rows of Lam are updated only on their allowed
    columns, which keeps the M-step exact under the sparsity constraint.
    """
    p, k = mask.shape
    if Lam is None:
        sd = np.sqrt(np.diag(S))
        Lam = mask * (0.3 * sd[:, None])
        Lam[:, 0] = mask[:, 0] * 0.6 * sd
    if Psi is None:
        Psi = np.clip(np.diag(S) - (Lam ** 2).sum(1), min_psi, None)
    for _ in range(n_steps):
        Sig = Lam @ Lam.T + np.diag(Psi)
        Beta = np.linalg.solve(Sig, Lam).T                  # k x p
        C = np.eye(k) - Beta @ Lam + Beta @ S @ Beta.T      # E[ff']
        SB = S @ Beta.T                                     # p x k
        Lnew = np.zeros_like(Lam)
        for i in range(p):
            A = np.flatnonzero(mask[i])
            if A.size:
                Lnew[i, A] = np.linalg.solve(C[np.ix_(A, A)] + 1e-9 * np.eye(A.size), SB[i, A])
        Psi = np.clip(np.diag(S) - 2 * (Lnew * SB).sum(1) + np.einsum("ij,jk,ik->i", Lnew, C, Lnew),
                      min_psi, None)
        Lam = Lnew
    return Lam, Psi


# ----------------------------------------------------------------------------
# Censored multivariate normal regression
# ----------------------------------------------------------------------------
def _safe_chol(C):
    """Cholesky factor; for a numerically indefinite matrix the eigenvalues are floored."""
    try:
        return np.linalg.cholesky(C + 1e-10 * np.eye(C.shape[0]))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh((C + C.T) / 2)
        w = np.maximum(w, 1e-8 * max(float(w.max()), 1e-12))
        return V * np.sqrt(w)


class CensoredMVN:
    """Multivariate normal regression with left/right censoring and missingness.

    structure : "free" | "factor" | "diagonal"
    probit    : if True, the latent scale is fixed to unit variance and the
                thresholds `cens` are estimated as part of B (binary data).
    """

    def __init__(self, cens, structure="free", families_mask=None, probit=False,
                 n_iter=200, burn=80, seed=0, probit_mask=None):
        self.cens = np.asarray(cens, float)
        self.structure = structure
        self.fmask = families_mask
        # probit_mask: boolean (p,) marking binary coordinates whose latent scale is
        # fixed to one (mixed continuous/binary data); probit=True marks all of them.
        self.probit_mask = (None if probit_mask is None and not probit
                            else np.ones(len(self.cens), bool) if probit_mask is None
                            else np.asarray(probit_mask, bool))
        self.probit = self.probit_mask is not None
        self.n_iter, self.burn = n_iter, burn
        self.rng = np.random.default_rng(seed)

    # -- fitting ---------------------------------------------------------
    def fit(self, Y, status, X, verbose=False):
        Y = np.asarray(Y, float).copy()
        status = np.asarray(status, int)
        X = np.asarray(X, float)
        n, p = Y.shape
        c = np.broadcast_to(self.cens, (n, p))
        Z = Y.copy()
        sd0 = np.nanstd(np.where(status == 0, Y, np.nan), 0)
        sd0 = np.where(np.isfinite(sd0) & (sd0 > 0), sd0, 1.0)
        Z = np.where(status == 1, c - 0.5 * sd0, Z)
        Z = np.where(status == 3, c + 0.5 * sd0, Z)
        Z = np.where(status == 2, np.nanmean(np.where(status == 0, Y, np.nan), 0), Z)
        Z = np.where(np.isfinite(Z), Z, 0.0)
        XtX_inv = np.linalg.inv(X.T @ X)
        Lam = Psi = None
        accB, accS, cnt = 0.0, 0.0, 0
        for it in range(self.n_iter):
            B = XtX_inv @ X.T @ Z
            R = Z - X @ B
            S = R.T @ R / n
            Sig, Lam, Psi = self._structure(S, Lam, Psi)
            if self.probit:
                d = np.where(self.probit_mask, np.sqrt(np.diag(Sig)), 1.0)
                Sig = Sig / np.outer(d, d)
                B = B / d[None, :]
                Z = Z / d[None, :]
                if Lam is not None:
                    Lam = Lam / d[:, None]
                    Psi = Psi / d ** 2
            Mu = X @ B
            Z = self._gibbs_sweep(Z, Mu, Sig, status, c, update_missing=True)
            if it >= self.burn:
                accB = accB + B
                accS = accS + Sig
                cnt += 1
            if verbose and it % 50 == 0:
                print(f"    SEM iteration {it}", flush=True)
        self.B = accB / cnt
        self.Sigma = accS / cnt
        if self.structure == "factor":
            self.Lam, self.Psi = fa_em(self.Sigma, self.fmask, Lam, Psi, n_steps=200)
            self.Sigma = self.Lam @ self.Lam.T + np.diag(self.Psi)
        self.Q = np.linalg.inv(self.Sigma)
        return self

    def _structure(self, S, Lam, Psi):
        if self.structure == "free":
            return S, None, None
        if self.structure == "diagonal":
            return np.diag(np.diag(S)), None, None
        Lam, Psi = fa_em(S, self.fmask, Lam, Psi, n_steps=8 if Lam is not None else 60)
        return Lam @ Lam.T + np.diag(Psi), Lam, Psi

    def _gibbs_sweep(self, Z, Mu, Sig, status, c, update_missing=True):
        """One sweep: censored cells one coordinate at a time, missing cells in
        blocks drawn exactly from their Gaussian conditional."""
        Q = np.linalg.inv(Sig)
        p = Z.shape[1]
        R = Z - Mu
        for j in range(p):
            rows = np.flatnonzero((status[:, j] == 1) | (status[:, j] == 3))
            if rows.size == 0:
                continue
            qj = Q[:, j]
            cm = Mu[rows, j] - (R[rows] @ qj - R[rows, j] * Q[j, j]) / Q[j, j]
            cs = 1.0 / np.sqrt(Q[j, j])
            below = status[rows, j] == 1
            newv = np.empty(rows.size)
            if below.any():
                newv[below] = _draw_below(cm[below], cs, c[rows[below], j], self.rng)
            if (~below).any():
                newv[~below] = _draw_above(cm[~below], cs, c[rows[~below], j], self.rng)
            Z[rows, j] = newv
            R[rows, j] = newv - Mu[rows, j]
        if update_missing:
            miss = status == 2
            if miss.any():
                pats, inv = np.unique(miss, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                for k, pat in enumerate(pats):
                    if not pat.any():
                        continue
                    rows = np.flatnonzero(inv == k)
                    m, o = pat, ~pat
                    if o.any():
                        A = np.linalg.solve(Sig[np.ix_(o, o)], Sig[np.ix_(o, m)]).T
                        cond = Mu[np.ix_(rows, m)] + (Z[np.ix_(rows, o)] - Mu[np.ix_(rows, o)]) @ A.T
                        Cc = Sig[np.ix_(m, m)] - A @ Sig[np.ix_(o, m)]
                    else:
                        cond = Mu[np.ix_(rows, m)]
                        Cc = Sig[np.ix_(m, m)]
                    L = _safe_chol(Cc)
                    Z[np.ix_(rows, m)] = cond + self.rng.standard_normal((rows.size, m.sum())) @ L.T
        return Z

    # -- deployment-time conditional draws -------------------------------
    def conditional_draws(self, Y, status, X, n_chains=24, n_sweeps=36, keep=12, seed=None):
        """Draws of the full latent vector given observed / censored / missing cells.

        Returns array (n, n_chains * keep, p). Cells with status 0 are returned
        at their observed values in every draw.
        """
        rng_save = self.rng
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        Y = np.asarray(Y, float)
        status = np.asarray(status, int)
        n, p = Y.shape
        Mu1 = np.asarray(X, float) @ self.B
        rep = np.repeat(np.arange(n), n_chains)
        Mu = Mu1[rep]
        st = status[rep]
        c = np.broadcast_to(self.cens, Mu.shape)
        sd = np.sqrt(np.diag(self.Sigma))
        Z = np.where(st == 0, Y[rep], Mu)
        Z = np.where(st == 1, np.minimum(Mu, c) - 0.3 * sd, Z)
        Z = np.where(st == 3, np.maximum(Mu, c) + 0.3 * sd, Z)
        Z = Z + np.where(st == 2, 0.5 * sd * self.rng.standard_normal(Z.shape), 0.0)
        out = np.empty((n, n_chains, keep, p))
        burn = n_sweeps - keep
        for s in range(n_sweeps):
            Z = self._gibbs_sweep(Z, Mu, self.Sigma, st, c, update_missing=True)
            if s >= burn:
                out[:, :, s - burn, :] = Z.reshape(n, n_chains, p)
        self.rng = rng_save
        return out.reshape(n, n_chains * keep, p)


def recorded_from_latent(Z, cens=CENS, floor=FLOOR):
    """Assay-scale log10 value that the laboratory would report."""
    return np.where(Z > cens, Z, floor)


# ----------------------------------------------------------------------------
# Decision-flip probability, calibration and guarantees
# ----------------------------------------------------------------------------
def flip_from_draws(eta_draws, tau, d_hat):
    """Fraction of completions whose decision differs from the issued one."""
    return ((eta_draws >= tau) != np.asarray(d_hat)[:, None]).mean(1)


def flip_gaussian(mu, sd, tau, d_hat):
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    z = np.where(d_hat, tau - mu, mu - tau) / np.maximum(sd, 1e-12)
    out = norm.cdf(z)
    deg = sd <= 1e-12
    out[deg] = ((mu[deg] >= tau) != np.asarray(d_hat)[deg]).astype(float)
    return out


class Calibrator:
    """Isotonic map from any score to an empirical flip probability."""

    def __init__(self, increasing=True):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip",
                                      increasing=increasing)

    def fit(self, s, f):
        self.iso.fit(np.asarray(s, float), np.asarray(f, float))
        return self

    def transform(self, s):
        return self.iso.predict(np.asarray(s, float))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    tot = 0.0
    for ch in np.array_split(order, n_bins):
        if len(ch):
            tot += len(ch) * abs(p[ch].mean() - y[ch].mean())
    return float(tot / len(p))


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper confidence bound."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


def ltt_threshold(score, loss, alpha, delta, n_grid=100, min_frac=0.02):
    """Learn-then-Test with fixed-sequence testing over a grid of acceptance
    thresholds (accept when score <= lambda). Returns the most liberal lambda
    certified at level (alpha, delta), or -inf if none.

    Guarantee: P( Pr[loss = 1 | accepted] <= alpha ) >= 1 - delta.
    """
    score = np.asarray(score, float)
    loss = np.asarray(loss, bool)
    if len(score) == 0:
        return -np.inf
    # start the fixed sequence where certification is possible at all: with
    # zero losses, n accepted rows certify alpha only if n >= log(delta)/log(1-alpha)
    n_min = np.log(delta) / np.log(1 - alpha)
    q0 = min(1.0, max(min_frac, 1.5 * n_min / len(score)))
    grid = np.unique(np.quantile(score, np.linspace(q0, 1.0, n_grid), method="higher"))
    best = -np.inf
    for lam in grid:
        acc = score <= lam
        if cp_upper(int(loss[acc].sum()), int(acc.sum()), delta) <= alpha:
            best = float(lam)
        else:
            break
    return best


def ltt_mondrian(score, loss, groups, alpha, delta, **kw):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    d = delta / max(len(uniq), 1)
    return {g: ltt_threshold(np.asarray(score)[groups == g], np.asarray(loss)[groups == g],
                             alpha, d, **kw) for g in uniq}


def apply_mondrian(score, groups, lam):
    groups = np.asarray(groups)
    thr = np.array([lam.get(g, -np.inf) for g in groups], float)
    return np.asarray(score, float) <= thr


def risk_coverage(score, flip, grid=None, rng=None):
    score = np.asarray(score, float)
    flip = np.asarray(flip, float)
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
    rng = np.random.default_rng(0) if rng is None else rng
    order = np.lexsort((rng.random(len(score)), score))
    return grid, np.array([flip[order[:max(int(round(c * len(score))), 1)]].mean() for c in grid])


def kim_score(M, w, sd):
    """Kim et al. (2026): sum over missing columns of |c_v| sd(x_v), normalised."""
    u = np.abs(np.asarray(w, float)) * np.asarray(sd, float)
    return np.asarray(M, float) @ (u / u.sum())


# ----------------------------------------------------------------------------
# Value of information from posterior draws
# ----------------------------------------------------------------------------
def preposterior_residual(eta_draws, block_draws, tau):
    """Expected flip probability after the block is measured.

    Within a participant's posterior draws, the linear predictor is regressed
    on the block's reported values; the fitted value is what measuring the
    block would reveal about eta and the residual SD is what would remain.
        E[FLIP'] = mean_k Phi( -|yhat_k - tau| / s_res )
    eta_draws  : (n, K)
    block_draws: (n, K, q)
    """
    n, K = eta_draws.shape
    Xb = np.concatenate([np.ones((n, K, 1)), block_draws], axis=2)
    XtX = np.einsum("nkq,nkr->nqr", Xb, Xb) + 1e-8 * np.eye(Xb.shape[2])[None]
    Xty = np.einsum("nkq,nk->nq", Xb, eta_draws)
    coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    yhat = np.einsum("nkq,nq->nk", Xb, coef)
    s = np.sqrt(np.maximum(((eta_draws - yhat) ** 2).mean(1), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = norm.cdf(-np.abs(yhat - tau) / s[:, None])
    u = np.where(s[:, None] > 1e-9, u, 0.0)
    return u.mean(1)


def preposterior_censored(eta_draws, block_draws, block_floor, tau, min_pos=12):
    """Censoring-aware preposterior: expected flip probability after the block
    is measured, for the posterior-majority decision rule.

    A panel result is either 'all analytes below the detection limit' (the
    common outcome) or 'at least one positive' with reported values. The two
    outcomes are handled separately within the participant's posterior draws:
        E[U'] = P(neg) * U(eta | neg) + P(pos) * E[U(eta | values, pos)]
    with U = min(P(eta >= tau), 1 - P(eta >= tau)). The negative branch is
    computed exactly from the draws; the positive branch regresses eta on the
    reported values within the positive draws (Gaussian residual), falling
    back to the empirical fraction when there are fewer than `min_pos` draws.
    Because the outcome partition refines the current information, the
    expected value never exceeds the current U (Jensen), so the gain is >= 0.
    """
    n, K = eta_draws.shape
    pos = (block_draws > np.asarray(block_floor)[None, None, :] + 1e-9).any(2)
    above = eta_draws >= tau
    npos = pos.sum(1)
    nneg = K - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        f_neg = np.where(nneg > 0, (above & ~pos).sum(1) / np.maximum(nneg, 1), 0.0)
        f_pos = np.where(npos > 0, (above & pos).sum(1) / np.maximum(npos, 1), 0.0)
    u_neg = np.minimum(f_neg, 1 - f_neg)
    u_pos = np.minimum(f_pos, 1 - f_pos)
    big = npos >= min_pos
    if big.any():
        W = pos[big].astype(float)
        Xb = np.concatenate([np.ones((big.sum(), K, 1)), block_draws[big]], axis=2)
        XtX = np.einsum("nk,nkq,nkr->nqr", W, Xb, Xb) + 1e-6 * np.eye(Xb.shape[2])[None]
        Xty = np.einsum("nk,nkq,nk->nq", W, Xb, eta_draws[big])
        coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        yhat = np.einsum("nkq,nq->nk", Xb, coef)
        res2 = (W * (eta_draws[big] - yhat) ** 2).sum(1) / np.maximum(W.sum(1) - Xb.shape[2], 1)
        s = np.sqrt(np.maximum(res2, 1e-12))
        u = norm.cdf(-np.abs(yhat - tau) / s[:, None])
        u_reg = (W * u).sum(1) / W.sum(1)
        u_pos[big] = np.minimum(u_reg, u_pos[big]) if False else u_reg
    return (nneg * u_neg + npos * u_pos) / K


def shapley_variance(eta_var_fn, blocks):
    """Exact Shapley split of a set function v(T) = var(nothing) - var(T measured)."""
    import itertools
    total = eta_var_fn(())
    phi = dict.fromkeys(blocks, 0.0)
    perms = list(itertools.permutations(blocks))
    cache = {}

    def v(T):
        key = frozenset(T)
        if key not in cache:
            cache[key] = total - eta_var_fn(tuple(sorted(T)))
        return cache[key]

    for perm in perms:
        prev, acc = 0.0, []
        for b in perm:
            acc.append(b)
            cur = v(acc)
            phi[b] += cur - prev
            prev = cur
    s = sum(phi.values())
    return {b: (phi[b] / len(perms)) for b in blocks}, total, s / len(perms)


def net_benefit(y, decision, t):
    y, d = np.asarray(y, int), np.asarray(decision, int)
    n = len(y)
    return float(((d == 1) & (y == 1)).sum() / n - t / (1 - t) * ((d == 1) & (y == 0)).sum() / n)
