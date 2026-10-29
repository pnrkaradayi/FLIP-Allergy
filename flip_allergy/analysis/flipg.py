"""
flipg.py
========
FLIP for general clinical tables (outside the allergen panel): mixed
continuous and binary test results, any classifier.

The joint distribution of the test results is a Gaussian copula: continuous
results enter through their normal scores (rank-based, fitted on the training
split), binary results through a thresholded latent normal with unit scale.
The latent vector is fitted with the same stochastic EM as the allergen model
(flipx.CensoredMVN, free covariance); deployment-time completions of the
unordered tests are blocked Gibbs draws, and every completion is passed
through the deployed classifier unchanged.

Status codes as in flipx: 0 observed exactly, 1 binary = 0 (z <= 0),
2 not ordered, 3 binary = 1 (z > 0).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtri
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV

from flipx import CensoredMVN, flip_from_draws


# ----------------------------------------------------------------------------
# Marginal transform (Gaussian copula)
# ----------------------------------------------------------------------------
class Copula:
    """Marginal transforms: normal scores for continuous columns ("cont"), 0/1
    for binary columns ("bin"), log10 for assay values censored below a
    detection limit ("cens"; reported values below the limit are set to the
    limit / sqrt(2), as NHANES does, and enter the latent model as bounds)."""

    def __init__(self, kinds, lods=None):
        if np.asarray(kinds).dtype == bool:                     # backward compatible: is_bin vector
            kinds = np.where(kinds, "bin", "cont")
        self.kinds = np.asarray(kinds)
        self.is_bin = self.kinds == "bin"
        self.is_cens = self.kinds == "cens"
        lods = np.ones(len(self.kinds)) if lods is None else np.asarray(lods, float)
        self.c = np.where(self.is_cens, np.log10(np.where(self.is_cens, lods, 1.0)), 0.0)
        self.fill = self.c - 0.5 * np.log10(2.0)

    def fit(self, X):
        X = np.asarray(X, float)
        self.ref = [np.sort(X[np.isfinite(X[:, j]), j]) if self.kinds[j] == "cont" else None
                    for j in range(X.shape[1])]
        return self

    def transform(self, X):
        """Feature space used by every classifier and imputer (NaN kept)."""
        X = np.asarray(X, float)
        T = np.full_like(X, np.nan)
        for j in range(X.shape[1]):
            ok = np.isfinite(X[:, j])
            if self.is_bin[j]:
                T[ok, j] = (X[ok, j] > 0.5).astype(float)
            elif self.is_cens[j]:
                v = X[ok, j]
                T[ok, j] = np.where(np.log10(np.maximum(v, 1e-12)) < self.c[j], self.fill[j],
                                    np.log10(np.maximum(v, 1e-12)))
            else:
                r = self.ref[j]
                lo = np.searchsorted(r, X[ok, j], side="left")
                hi = np.searchsorted(r, X[ok, j], side="right")
                u = np.clip((lo + hi) / 2.0 / len(r), 0.5 / len(r), 1 - 0.5 / len(r))
                T[ok, j] = ndtri(u)
        return T

    def latent_inputs(self, T):
        """(Y, status) for CensoredMVN from transformed features."""
        fin = np.isfinite(T)
        Y = np.where(fin, T, 0.0)
        st = np.where(fin, 0, 2)
        st = np.where(self.is_bin[None, :] & fin, np.where(T > 0.5, 3, 1), st)
        st = np.where(self.is_cens[None, :] & fin & (T < self.c[None, :]), 1, st)
        Y = np.where(self.is_bin[None, :], 0.0, Y)
        return Y, st

    def features_from_latent(self, Z):
        return np.where(self.is_bin, (Z > 0).astype(float),
                        np.where(self.is_cens, np.where(Z >= self.c, Z, self.fill), Z))


def fit_latent(cop, T, n_iter=300, burn=100, seed=0, X=None, fmask=None):
    """Free covariance, or the cross-reactivity factor structure when a family mask is given."""
    Y, st = cop.latent_inputs(T)
    mdl = CensoredMVN(cop.c, "free" if fmask is None else "factor", families_mask=fmask, n_iter=n_iter,
                      burn=burn, seed=seed, probit_mask=cop.is_bin if cop.is_bin.any() else None)
    return mdl.fit(Y, st, np.ones((len(T), 1)) if X is None else X)


def latent_draws(mdl, cop, T, n_chains=16, n_sweeps=30, keep=12, seed=0, X=None):
    """Completed feature vectors (n, D, p); observed cells kept as observed."""
    Y, st = cop.latent_inputs(T)
    Z = mdl.conditional_draws(Y, st, np.ones((len(T), 1)) if X is None else X, n_chains, n_sweeps, keep,
                              seed=seed)
    F = cop.features_from_latent(Z)
    obs = np.isfinite(T)
    return np.where(obs[:, None, :], T[:, None, :], F)


# ----------------------------------------------------------------------------
# Imputers used by the published pipelines (fitted on the training split)
# ----------------------------------------------------------------------------
def make_imputer(kind, seed=0):
    if kind == "missforest":
        return IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2,
                                                              random_state=seed, n_jobs=1),
                                max_iter=6, random_state=seed)
    if kind == "mice":
        return IterativeImputer(max_iter=10, random_state=seed)
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "mean":
        return SimpleImputer(strategy="mean")
    if kind == "median":
        return SimpleImputer(strategy="median")
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Classifiers: score() returns the quantity compared with the threshold tau
# ----------------------------------------------------------------------------
class Logistic:
    """L1 logistic regression on standardized features (Kim et al. 2026 style)."""
    name = "L1 logistic"

    def fit(self, T, y, seed=0):
        self.mu, self.sd = T.mean(0), np.where(T.std(0) > 0, T.std(0), 1.0)
        S = (T - self.mu) / self.sd
        m = LogisticRegressionCV(Cs=12, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                                 max_iter=5000, random_state=seed).fit(S, y)
        self.w, self.b0 = m.coef_.ravel(), float(m.intercept_[0])
        return self

    def score(self, T):
        return self.b0 + ((T - self.mu) / self.sd) @ self.w

    def kim_weights(self):
        """Kim et al.: spread of each feature's SHAP value; for a linear model on
        standardized features the SHAP SD of feature j is |w_j|."""
        return np.abs(self.w)


class Forest:
    """Random forest probability (the classifier of most published pipelines)."""
    name = "random forest"

    def __init__(self, n_estimators=300):
        self.n_estimators = n_estimators

    def fit(self, T, y, seed=0):
        self.m = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=2,
                                        random_state=seed, n_jobs=1).fit(T, y)
        return self

    def score(self, T):
        return self.m.predict_proba(T)[:, 1]

    def kim_weights(self):
        return self.m.feature_importances_


def threshold_at_fpr(score_neg, fpr=0.10):
    return float(np.quantile(score_neg, 1 - fpr))


def scores_on_draws(clf, F):
    n, D, p = F.shape
    return clf.score(F.reshape(n * D, p)).reshape(n, D)


# ----------------------------------------------------------------------------
# Amputation of whole tests
# ----------------------------------------------------------------------------
def amputate(T, blocks, p_order, rng, mech="MAR", strength=1.1, mnar=0.8):
    """Boolean (n, n_blocks): True = test not ordered.

    MAR: availability depends on a latent care intensity shared by all tests;
    MNAR: additionally on the patient's own (unobserved) result of that test.
    """
    n = len(T)
    care = rng.normal(size=n)
    Mb = np.zeros((n, len(blocks)), bool)
    for j, cols in enumerate(blocks):
        drive = strength * care
        if mech == "MNAR":
            z = np.nanmean(T[:, cols], 1)
            z = (z - np.nanmean(z)) / (np.nanstd(z) if np.nanstd(z) > 0 else 1.0)
            drive = drive + mnar * np.nan_to_num(z)
        lo, hi = -15.0, 15.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if expit(mid + drive).mean() > p_order[j]:
                hi = mid
            else:
                lo = mid
        Mb[:, j] = rng.random(n) >= expit((lo + hi) / 2 + drive)
    return Mb


def expand(Mb, blocks, p):
    M = np.zeros((len(Mb), p), bool)
    for j, cols in enumerate(blocks):
        M[np.ix_(np.flatnonzero(Mb[:, j]), cols)] = True
    return M


def mi_draws(T_train, Tobs, clf, n_imp=30, seed=0):
    """Multiple imputation (chained equations, posterior sampling) -> scores (n, n_imp)."""
    imp = IterativeImputer(max_iter=10, random_state=seed, sample_posterior=True).fit(T_train)
    out = []
    for k in range(n_imp):
        imp.random_state_ = np.random.RandomState(seed + 1000 + k)
        out.append(clf.score(imp.transform(Tobs)))
    return np.stack(out, 1)


__all__ = ["Copula", "fit_latent", "latent_draws", "make_imputer", "Logistic", "Forest",
           "threshold_at_fpr", "scores_on_draws", "amputate", "expand", "mi_draws", "flip_from_draws"]
