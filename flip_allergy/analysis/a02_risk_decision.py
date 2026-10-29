"""
a02_risk_decision.py
====================
Task 1. A deployed allergic-disease risk model (logistic regression on the
19 allergen-specific IgE, total IgE, age, sex, BMI and poverty-income ratio)
is applied to patients for whom some panels were not ordered. Which issued
decisions would change if every panel were ordered, can the error among the
decisions we accept be guaranteed, and which panel should be ordered next?

Design
------
* The model, the imputer and the latent conditional model are fitted on the
  COMPLETE training split, as a published model would be.
* Missing panels are created by amputation on the calibration and test
  splits (MAR: care intensity only; MNAR: additionally the patient's own
  unobserved values in that panel). Every participant has all 19 sIgE, so the
  full-data decision, and hence the true flip status, is known.
* The issued decision uses single imputation with a random-forest-type
  chained imputer (MissForest-like), as in Kim et al. 2026.

Run:  python a02_risk_decision.py [n_mar] [n_mnar] [outcome]
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import (ALL20, BLOCK_COLS, BLOCK_NAMES, CENS, COST, FAMILIES, FLOOR,  # noqa: E402
                   Calibrator, CensoredMVN, apply_mondrian, brier, covariate_design,
                   ece, family_mask, flip_from_draws, flip_gaussian, kim_score,
                   ltt_mondrian, ltt_threshold, net_benefit, preposterior_censored,
                   recorded_from_latent, risk_coverage)

RES = os.path.join(HERE, "results")
COVS = ["RIDAGEYR", "male", "BMXBMI", "INDFMPIR"]
FPR_TARGET = 0.10
ALPHAS = (0.01, 0.02, 0.05)
ALPHA, DELTA = 0.02, 0.10
P_ORDER = {"mite": .72, "pet": .72, "pest": .45, "grass": .68, "tree": .62,
           "weed": .62, "mould": .50, "food": .40, "totige": .85}
N_CHAINS, N_SWEEPS, KEEP = 16, 30, 12


# ----------------------------------------------------------------- amputation
def sample_block_masks(n, rng, mech, Y, scale=1.0):
    """(n, 9) True = panel not ordered. MAR: care intensity; MNAR: + own values."""
    care = rng.normal(size=n)
    Mb = np.zeros((n, len(BLOCK_NAMES)), bool)
    for j, b in enumerate(BLOCK_NAMES):
        drive = 1.1 * care
        if mech == "MNAR":
            z = Y[:, BLOCK_COLS[b]]
            z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1.0)
            drive = drive + 0.8 * z.mean(1)
        target = 1 - (1 - P_ORDER[b]) * scale
        lo, hi = -15.0, 15.0
        for _ in range(60):                       # bisection for the intercept
            mid = (lo + hi) / 2
            if expit(mid + drive).mean() > target:
                hi = mid
            else:
                lo = mid
        Mb[:, j] = rng.random(n) >= expit((lo + hi) / 2 + drive)
    return Mb


def expand(Mb):
    M = np.zeros((len(Mb), len(ALL20)), bool)
    for j, b in enumerate(BLOCK_NAMES):
        M[np.ix_(np.flatnonzero(Mb[:, j]), BLOCK_COLS[b])] = True
    return M


# ----------------------------------------------------------------- fitting
def fit_all(outcome):
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    tr = d[d.split == "train"]
    med = tr[COVS].median()
    for c in COVS:
        d[c] = d[c].fillna(med[c])
    tr = d[d.split == "train"]
    feat = ["lg_" + c for c in ALL20] + COVS
    Xtr = tr[feat].values.astype(float)
    mu_sc, sd_sc = Xtr.mean(0), Xtr.std(0)
    Ztr = (Xtr - mu_sc) / sd_sc
    ytr = tr[outcome].values
    clf = LogisticRegressionCV(Cs=12, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                               max_iter=5000).fit(Ztr, ytr)
    w, b0 = clf.coef_.ravel(), float(clf.intercept_[0])
    eta_tr = b0 + Ztr @ w
    tau = float(np.quantile(eta_tr[ytr == 0], 1 - FPR_TARGET))

    imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3,
                                                         random_state=0, n_jobs=1),
                           max_iter=5, random_state=0).fit(Ztr)
    imp_mi = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True).fit(Ztr)

    Yl = tr[["lg_" + c for c in ALL20]].values
    stl = tr[["cens_" + c for c in ALL20]].values.astype(int)
    Xl = covariate_design(tr.RIDAGEYR, tr.male, tr.RIDRETH1)
    lat = {"factor": CensoredMVN(CENS, "factor", family_mask(families=FAMILIES), n_iter=300,
                                 burn=100, seed=11).fit(Yl, stl, Xl),
           "free": CensoredMVN(CENS, "free", n_iter=300, burn=100, seed=11).fit(Yl, stl, Xl)}
    Cnaive = np.cov(Ztr, rowvar=False)
    te = d[d.split == "test"]
    auc_test = roc_auc_score(te[outcome], b0 + ((te[feat].values - mu_sc) / sd_sc) @ w)
    info = dict(outcome=outcome, tau=tau, t_prob=float(expit(tau)), auc_test_full=float(auc_test),
                n_nonzero=int((w != 0).sum()), prevalence_train=float(ytr.mean()))
    return dict(d=d, feat=feat, mu_sc=mu_sc, sd_sc=sd_sc, w=w, b0=b0, tau=tau, imp=imp,
                imp_mi=imp_mi, lat=lat, Cnaive=Cnaive, info=info, outcome=outcome)


# ----------------------------------------------------------------- one split
def prepare(F, df, Mb, seed):
    p = len(ALL20)
    M = expand(Mb)
    Yrec = df[["lg_" + c for c in ALL20]].values.astype(float)
    status = df[["cens_" + c for c in ALL20]].values.astype(int)
    status[M] = 2
    Zfull = (df[F["feat"]].values.astype(float) - F["mu_sc"]) / F["sd_sc"]
    Zobs = Zfull.copy()
    Zobs[:, :p][M] = np.nan
    w, b0, tau = F["w"], F["b0"], F["tau"]
    eta_hat = b0 + F["imp"].transform(Zobs) @ w
    eta_full = b0 + Zfull @ w
    d_hat, d_full = eta_hat >= tau, eta_full >= tau
    Xl = covariate_design(df.RIDAGEYR, df.male, df.RIDRETH1)
    cov_part = Zfull[:, p:] @ w[p:]

    def eta_from(Zlat):
        R = np.where(M[:, None, :], recorded_from_latent(Zlat), Yrec[:, None, :])
        return b0 + ((R - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + cov_part[:, None], R

    draws = {}
    for nm, mdl in F["lat"].items():
        Zl = mdl.conditional_draws(Yrec, status, Xl, N_CHAINS, N_SWEEPS, KEEP, seed=seed)
        e, R = eta_from(Zl)
        draws[nm] = (e, R if nm == "factor" else None)
        del Zl
    complete = ~M.any(1)
    if complete.any():
        assert np.allclose(draws["factor"][0][complete], eta_full[complete][:, None], atol=1e-8)

    # v1 FLIP: Gaussian closed form on the recorded standardized features
    C = F["Cnaive"]
    mu_g = np.empty(len(df))
    sd_g = np.zeros(len(df))
    Mfull = np.zeros_like(Zobs, bool)
    Mfull[:, :p] = M
    pats, inv = np.unique(Mfull, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    for k, pat in enumerate(pats):
        rows = np.flatnonzero(inv == k)
        m, o = pat, ~pat
        if not m.any():
            mu_g[rows] = b0 + Zfull[np.ix_(rows, o)] @ w[o]
            continue
        A = np.linalg.solve(C[np.ix_(o, o)], C[np.ix_(o, m)]).T
        cm = Zfull[np.ix_(rows, o)] @ A.T
        Sc = C[np.ix_(m, m)] - A @ C[np.ix_(o, m)]
        mu_g[rows] = b0 + Zfull[np.ix_(rows, o)] @ w[o] + cm @ w[m]
        sd_g[rows] = np.sqrt(max(float(w[m] @ Sc @ w[m]), 0.0))

    F["imp_mi"].random_state_ = np.random.RandomState(seed + 7)
    eta_mi = np.stack([b0 + F["imp_mi"].transform(Zobs) @ w for _ in range(30)], 1)

    sc = {
        "Fraction missing": M.mean(1),
        "Kim et al. 2026": kim_score(Mfull, w, np.ones(len(w))),
        "Margin only": -np.abs(eta_hat - tau),
        "Conditional SD only": sd_g,
        "FLIP v1 (Gaussian)": flip_gaussian(mu_g, sd_g, tau, d_hat),
        "Multiple imputation": flip_from_draws(eta_mi, tau, d_hat),
        "FLIP, free covariance": flip_from_draws(draws["free"][0], tau, d_hat),
        "FLIP": flip_from_draws(draws["factor"][0], tau, d_hat),
    }
    p_pos = (draws["factor"][0] >= tau).mean(1)
    d_bayes = p_pos >= 0.5
    return dict(M=M, Mb=Mb, Yrec=Yrec, status=status, Zfull=Zfull, Zobs=Zobs, eta_hat=eta_hat,
                eta_full=eta_full, d_hat=d_hat, d_full=d_full, flip=d_hat != d_full,
                draws=draws, scores=sc, y=df[F["outcome"]].values, Xl=Xl, cov_part=cov_part,
                p_pos=p_pos, d_bayes=d_bayes, flip_bayes=d_bayes != d_full,
                U_post=np.minimum(p_pos, 1 - p_pos))


PROB_SCORES = ["FLIP v1 (Gaussian)", "Multiple imputation", "FLIP, free covariance", "FLIP"]
LTT_SCORES = ["Fraction missing", "Kim et al. 2026", "Margin only", "Multiple imputation",
              "FLIP v1 (Gaussian)", "FLIP, free covariance", "FLIP"]


def run_rep(args):
    mech, rep, scale, bundle_path = args
    with open(bundle_path, "rb") as fh:
        F = pickle.load(fh)
    d = F["d"]
    ca, te = d[d.split == "calib"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    rng = np.random.default_rng(1000 * (mech == "MNAR") + 100 * int(scale * 10) + rep)
    Mb_c = sample_block_masks(len(ca), rng, mech, ca[["lg_" + c for c in ALL20]].values, scale)
    Mb_t = sample_block_masks(len(te), rng, mech, te[["lg_" + c for c in ALL20]].values, scale)
    C = prepare(F, ca, Mb_c, seed=rep * 2 + 1)
    C["draws"] = None
    T = prepare(F, te, Mb_t, seed=rep * 2 + 2)
    f, fc = T["flip"], C["flip"]
    tau = F["tau"]
    tag = dict(mech=mech, rep=rep, scale=scale)

    # calibration of every score to an empirical flip probability
    cal = {nm: Calibrator().fit(C["scores"][nm], fc) for nm in T["scores"]}
    calp = {nm: cal[nm].transform(T["scores"][nm]) for nm in T["scores"]}

    perf = []
    dlog = np.abs(T["eta_full"] - T["eta_hat"])
    for nm, s in T["scores"].items():
        cov, risk = risk_coverage(s, f)
        perf.append(dict(**tag, score=nm, flip_rate=float(f.mean()),
                         auroc=roc_auc_score(f, s), auprc=average_precision_score(f, s),
                         rho_dlogit=spearmanr(s, dlog).statistic, aurc=float(risk.mean()),
                         flip_at_50=float(risk[9]), flip_at_80=float(risk[15]),
                         brier_raw=brier(s, f) if nm in PROB_SCORES else np.nan,
                         ece_raw=ece(s, f) if nm in PROB_SCORES else np.nan,
                         brier_cal=brier(calp[nm], f), ece_cal=ece(calp[nm], f)))

    ltt = []
    gc, gt = np.minimum(C["Mb"].sum(1), 4), np.minimum(T["Mb"].sum(1), 4)
    for nm in LTT_SCORES:
        for a in ALPHAS:
            lam = ltt_threshold(C["scores"][nm], fc, a, DELTA)
            acc = T["scores"][nm] <= lam
            ltt.append(dict(**tag, score=nm, kind="marginal", alpha=a, coverage=float(acc.mean()),
                            realised=float(f[acc].mean()) if acc.any() else np.nan,
                            worst_group=np.nan))
        lamg = ltt_mondrian(C["scores"][nm], fc, gc, ALPHA, DELTA)
        acc = apply_mondrian(T["scores"][nm], gt, lamg)
        worst = [f[acc & (gt == g)].mean() for g in np.unique(gt) if (acc & (gt == g)).sum() >= 20]
        accm = T["scores"][nm] <= ltt_threshold(C["scores"][nm], fc, ALPHA, DELTA)
        worst_m = [f[accm & (gt == g)].mean() for g in np.unique(gt) if (accm & (gt == g)).sum() >= 20]
        ltt.append(dict(**tag, score=nm, kind="mondrian", alpha=ALPHA, coverage=float(acc.mean()),
                        realised=float(f[acc].mean()) if acc.any() else np.nan,
                        worst_group=float(max(worst)) if worst else np.nan,
                        worst_group_marginal=float(max(worst_m)) if worst_m else np.nan))

    # ---------------------------------------------------------- posterior (Bayes) decision
    bayes = dict(**tag, flip_imputed=float(f.mean()), flip_bayes=float(T["flip_bayes"].mean()),
                 flip_mi_majority=float((((T["scores"]["Multiple imputation"] > 0.5) ^ T["d_hat"])
                                         != T["d_full"]).mean()))
    fb, fbc = T["flip_bayes"], C["flip_bayes"]
    lam_post = {a: ltt_threshold(C["U_post"], fbc, a, DELTA) for a in ALPHAS}
    for a in ALPHAS:
        acc = T["U_post"] <= lam_post[a]
        ltt.append(dict(**tag, score="FLIP (posterior decision)", kind="marginal", alpha=a,
                        coverage=float(acc.mean()), realised=float(fb[acc].mean()) if acc.any() else np.nan,
                        worst_group=np.nan))
    lamg = ltt_mondrian(C["U_post"], fbc, gc, ALPHA, DELTA)
    acc = apply_mondrian(T["U_post"], gt, lamg)
    worst = [fb[acc & (gt == g)].mean() for g in np.unique(gt) if (acc & (gt == g)).sum() >= 20]
    accm = T["U_post"] <= lam_post[ALPHA]
    worst_m = [fb[accm & (gt == g)].mean() for g in np.unique(gt) if (accm & (gt == g)).sum() >= 20]
    ltt.append(dict(**tag, score="FLIP (posterior decision)", kind="mondrian", alpha=ALPHA,
                    coverage=float(acc.mean()), realised=float(fb[acc].mean()) if acc.any() else np.nan,
                    worst_group=float(max(worst)) if worst else np.nan,
                    worst_group_marginal=float(max(worst_m)) if worst_m else np.nan))

    # ---------------------------------------------------------- acquisition
    eta_d, R_d = T["draws"]["factor"]
    nb = len(BLOCK_NAMES)
    resid = np.full((len(te), nb), np.inf)
    for j, b in enumerate(BLOCK_NAMES):
        rows = np.flatnonzero(T["Mb"][:, j])
        if rows.size:
            resid[rows, j] = preposterior_censored(eta_d[rows], R_d[rows][:, :, BLOCK_COLS[b]],
                                                   FLOOR[BLOCK_COLS[b]], tau)
    has = T["Mb"].any(1)
    w = F["w"]
    p = len(ALL20)
    uR = np.abs(w) / np.abs(w).sum()
    kimv = np.where(T["Mb"], np.array([uR[BLOCK_COLS[b]].sum() for b in BLOCK_NAMES])[None, :], -np.inf)
    kim_panel = np.where(has, kimv.argmax(1), -1)
    rules = {"imputed": dict(now=T["scores"]["FLIP"], lam=ltt_threshold(C["scores"]["FLIP"], fc, ALPHA, DELTA),
                             flip0=f),
             "posterior": dict(now=T["U_post"], lam=lam_post[ALPHA], flip0=fb)}
    budget = 0.10 * len(te) * float(COST.mean())

    def after(chosen, rule):
        """Reveal the chosen panels, re-issue the decision under `rule`, recompute FLIP."""
        Zo = T["Zobs"].copy()
        st = T["status"].copy()
        Mb2 = T["Mb"].copy()
        for j, b in enumerate(BLOCK_NAMES):
            r = np.flatnonzero(chosen == j)
            if r.size:
                cols = BLOCK_COLS[b]
                Zo[np.ix_(r, cols)] = T["Zfull"][np.ix_(r, cols)]
                st[np.ix_(r, cols)] = te[["cens_" + ALL20[c] for c in cols]].values[r]
                Mb2[r, j] = False
        if rule == "imputed":
            d2 = (F["b0"] + F["imp"].transform(Zo) @ w) >= tau
        else:
            d2 = T["d_bayes"].copy()
        score2 = rules[rule]["now"].copy()
        r = np.flatnonzero(chosen >= 0)
        if r.size:
            M2 = expand(Mb2[r])
            Zl = F["lat"]["factor"].conditional_draws(T["Yrec"][r], st[r], T["Xl"][r],
                                                      N_CHAINS, N_SWEEPS, KEEP, seed=rep + 99)
            Rr = np.where(M2[:, None, :], recorded_from_latent(Zl), T["Yrec"][r][:, None, :])
            e2 = F["b0"] + ((Rr - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + T["cov_part"][r][:, None]
            p2 = (e2 >= tau).mean(1)
            if rule == "posterior":
                d2[r] = p2 >= 0.5
                score2[r] = np.minimum(p2, 1 - p2)
            else:
                score2[r] = flip_from_draws(e2, tau, d2[r])
        flip2 = d2 != T["d_full"]
        accepted = score2 <= rules[rule]["lam"]
        return dict(flip_after=float(flip2.mean()), certified=float(accepted.mean()),
                    certified_flip=float(flip2[accepted].mean()) if accepted.any() else np.nan,
                    spent=float(COST[chosen[chosen >= 0]].sum()))

    def greedy(order, panel):
        ch = np.full(len(te), -1)
        spent = 0.0
        for i in order:
            if panel[i] < 0:
                continue
            c = COST[panel[i]]
            if spent + c <= budget:
                ch[i] = panel[i]
                spent += c
        return ch

    acq = []
    rand_choice = np.array([rng.choice(np.flatnonzero(m)) if m.any() else -1 for m in T["Mb"]])
    voi_panel = np.where(has, np.where(np.isfinite(resid), resid, np.inf).argmin(1), -1)
    for rule, R in rules.items():
        base = dict(**tag, rule=rule, flip_before=float(R["flip0"].mean()))
        for nm, ch in (("Random panel", rand_choice), ("Kim: most important missing panel", kim_panel),
                       ("FLIP-VOI", voi_panel)):
            acq.append(dict(**base, setting="one panel per patient", policy=nm, **after(ch, rule)))
        gain = np.where(np.isfinite(resid), R["now"][:, None] - resid, -np.inf)
        ratio = gain / COST[None, :]
        best_blk = np.where(has, ratio.argmax(1), -1)
        best_ratio = np.where(has, ratio.max(1), -np.inf)
        already = R["now"] <= R["lam"]
        cert_cost = np.where(resid <= R["lam"], COST[None, :], np.inf)
        cert_blk = np.where(np.isfinite(cert_cost).any(1) & ~already & has, cert_cost.argmin(1), -1)
        cert_order = np.argsort(np.where(cert_blk >= 0, cert_cost.min(1), np.inf), kind="stable")
        policies = {
            "No extra testing": np.full(len(te), -1),
            "Kim rank + Kim panel": greedy(np.argsort(-T["scores"]["Kim et al. 2026"], kind="stable"),
                                           kim_panel),
            "Margin rank + Kim panel": greedy(np.argsort(-T["scores"]["Margin only"], kind="stable"),
                                              kim_panel),
            "FLIP-VOI: expected flips": greedy(np.argsort(-best_ratio, kind="stable"),
                                               np.where(best_ratio > 0, best_blk, -1)),
            "FLIP-VOI: certified decisions": greedy(cert_order, cert_blk),
        }
        for nm, ch in policies.items():
            acq.append(dict(**base, setting="budget 10%", policy=nm, **after(ch, rule)))
    lam_hat = rules["imputed"]["lam"]

    # ---------------------------------------------------------- decision curve with deferral
    dca = []
    t = float(expit(tau))
    acc_flip = T["scores"]["FLIP"] <= lam_hat
    frac_def = 1 - acc_flip.mean()
    n_def = int(round(frac_def * len(te)))

    def defer_by(score):
        a = np.ones(len(te), bool)
        if n_def > 0:
            a[np.argsort(-score, kind="stable")[:n_def]] = False
        return a

    strategies = {"Impute and decide": np.ones(len(te), bool), "Order all missing panels": ~has,
                  "FLIP defer (LTT)": acc_flip,
                  "Margin defer (matched)": defer_by(T["scores"]["Margin only"]),
                  "Kim defer (matched)": defer_by(T["scores"]["Kim et al. 2026"])}
    acc_post = T["U_post"] <= lam_post[ALPHA]
    for kappa in (0.0, 0.002, 0.005, 0.01, 0.02):
        for nm, a, issued in [(k, v, T["d_hat"]) for k, v in strategies.items()] + [
                ("Posterior decision", np.ones(len(te), bool), T["d_bayes"]),
                ("Posterior decision + FLIP defer (LTT)", acc_post, T["d_bayes"])]:
            dec = np.where(a, issued, T["d_full"])
            dca.append(dict(**tag, kappa=kappa, strategy=nm, deferred=float((~a).mean()),
                            nb=net_benefit(T["y"], dec, t) - kappa * float((~a & has).mean()),
                            flip=float((dec != T["d_full"]).mean())))

    extra = None
    if rep == 0 and scale == 1.0:
        extra = dict(scores=T["scores"], calp=calp, flip=f, Mb=T["Mb"], d_hat=T["d_hat"],
                     U_post=T["U_post"], p_pos=T["p_pos"], d_bayes=T["d_bayes"], flip_bayes=fb,
                     d_full=T["d_full"], eta_hat=T["eta_hat"], eta_full=T["eta_full"], y=T["y"],
                     eta_draws=eta_d, R_draws=R_d, resid=resid, lam_hat=lam_hat, tau=tau)
    return perf, ltt, [bayes], acq, dca, extra


def main(n_mar=20, n_mnar=10, outcome="y_allergic_disease", scales=(1.0,), tag_out=""):
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    F = fit_all(outcome)
    print(f"[{outcome}] fitted in {time.time() - t0:.0f}s: {F['info']}", flush=True)
    bundle = os.path.join(RES, f"_bundle_{outcome}.pkl")
    with open(bundle, "wb") as fh:
        pickle.dump(F, fh)
    jobs = [(m, r, s, bundle) for s in scales for m, R in (("MAR", n_mar), ("MNAR", n_mnar))
            for r in range(R)]
    out = {k: [] for k in ("perf", "ltt", "bayes", "acq", "dca")}
    with ProcessPoolExecutor(max_workers=min(10, len(jobs))) as ex:
        for res, job in zip(ex.map(run_rep, jobs), jobs):
            for k, v in zip(out, res[:5]):
                out[k].extend(v)
            if res[5] is not None:
                with open(os.path.join(RES, f"t1_examples_{job[0]}{tag_out}" + ("" if outcome == "y_allergic_disease" else f"_{outcome}") + ".pkl"), "wb") as fh:
                    pickle.dump(res[5], fh)
            print(f"  {job[0]} rep {job[1]} scale {job[2]} done ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_{outcome}{tag_out}"
    for k, v in out.items():
        pd.DataFrame(v).to_csv(os.path.join(RES, f"t1_{k}{sfx}.csv"), index=False)
    pd.Series(F["info"]).to_csv(os.path.join(RES, f"t1_info{sfx}.csv"))
    P = pd.DataFrame(out["perf"])
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for mech in ("MAR", "MNAR"):
        q = P[(P.mech == mech) & (P.scale == 1.0)]
        if q.empty:
            continue
        print(f"\n== {mech}: flip detection (median) ==")
        print(q.groupby("score", sort=False)[["flip_rate", "auroc", "auprc", "aurc", "flip_at_50",
                                              "brier_cal", "ece_raw", "ece_cal"]].median().round(4))
        L = pd.DataFrame(out["ltt"])
        L = L[(L.mech == mech) & (L.scale == 1.0)]
        g = L.groupby(["score", "kind", "alpha"], sort=False)
        print(pd.DataFrame({"coverage": g.coverage.median(), "realised": g.realised.median(),
                            "worst_group": g.worst_group.median()}).round(4))
        A = pd.DataFrame(out["acq"])
        A = A[(A.mech == mech) & (A.scale == 1.0)]
        print(A.groupby(["rule", "setting", "policy"], sort=False)[["flip_before", "flip_after", "certified",
                                                             "certified_flip", "spent"]].median().round(4))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 20, int(a[1]) if len(a) > 1 else 10,
         a[2] if len(a) > 2 else "y_allergic_disease")
