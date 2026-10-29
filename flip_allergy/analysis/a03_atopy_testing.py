"""
a03_atopy_testing.py
====================
Task 2. Is this person sensitised to at least one allergen of the panel, and
how few allergen-specific tests are needed to say so with a guaranteed
sensitivity?

FLIP-adaptive policy
--------------------
Tests are ordered one allergen at a time. Before each test the posterior of
the untested allergens is computed from the censored latent model given the
covariates, (optionally) total IgE and the negative results so far. The next
test is the untested allergen with the highest posterior probability of a
positive result. Testing stops as soon as a positive is found (sensitised,
certain) or when the posterior probability that ANY untested allergen is
positive falls below lambda (declared not sensitised). lambda is calibrated
by Learn-then-Test on the calibration split so that the miss rate among
sensitised persons is at most alpha with probability at least 1 - delta.

Comparators (the methods of the panel-optimisation literature, fitted on the
same training split): fixed panels chosen by greedy set cover (conditional
stepwise selection) or by prevalence ranking; sequential testing in the
greedy order stopping at the first positive; and a total-IgE gate followed by
sequential testing. Every operating point is certified on the calibration
split with the same Clopper-Pearson rule and evaluated on the test split.

Also: replication in NHANES III skin prick tests (10 allergens), in-domain and
with the serum-derived dependence structure transported.

Run:  python a03_atopy_testing.py
"""
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import (ALL20, CENS, FAMILIES, NAME, SIGE, CensoredMVN,  # noqa: E402
                   covariate_design, cp_upper, family_mask)

RES = os.path.join(HERE, "results")
ALPHAS = (0.10, 0.05, 0.025, 0.01)
DELTA = 0.10
N_CHAINS, N_SWEEPS, KEEP = 16, 30, 16
SPT = ["MIT", "CAT", "COC", "ALT", "RYE", "BER", "RAG", "THI", "WHI", "PEA"]
SPT_TO_SERUM = {"MIT": "LBXID2", "CAT": "LBXIE1", "COC": "LBXII6", "ALT": "LBXIM6",
                "RYE": "LBXIG5", "BER": "LBXIG2", "RAG": "LBXIW1", "THI": "LBXW11",
                "WHI": "LBXIT7", "PEA": "LBXF13"}


# ----------------------------------------------------------------- simulation
def simulate_adaptive(model, positive, base_status, base_values, X, cand, seed=0, verbose=False,
                      n_draws=16384, min_accept=600, max_pool=262144):
    """Run the greedy FLIP policy to completion for every person.

    The posterior of the untested allergens is computed exactly under the
    fitted censored latent model by rejection sampling: latent vectors are
    drawn from the Gaussian conditional given the exactly observed extra
    coordinates (e.g. total IgE) and the covariates, and kept only if they
    agree with every bound observed so far (negative allergen tests, a
    total IgE below its detection limit). The pool is enlarged on the fly
    when fewer than `min_accept` draws remain consistent.

    positive    : (n, p) bool, true sensitisation status of every coordinate
    base_status : (n, p) status of every coordinate before any allergen test;
                  only the non-candidate coordinates are used (0 exact,
                  1 below the detection limit, 2 not available)
    cand        : indices of the testable allergens
    Returns p_any (n, K+1) = P(any untested positive) before each test,
            order (n, K) allergen tested at each step (-1 after stopping),
            first_pos (n,) step of the first positive (inf if none),
            n_accept (n, K+1) number of consistent draws behind each estimate.
    """
    rng = np.random.default_rng(seed)
    n, p = positive.shape
    cand = np.asarray(cand)
    K = len(cand)
    extra = np.array([j for j in range(p) if j not in set(cand.tolist())], int)
    Sig = model.Sigma
    Mu = np.asarray(X, float) @ model.B
    c = np.asarray(model.cens, float)
    p_any = np.full((n, K + 1), np.nan)
    n_acc = np.zeros((n, K + 1), int)
    order = np.full((n, K), -1)
    first_pos = np.full(n, np.inf)
    est = base_status[:, extra] if extra.size else np.zeros((n, 0), int)
    pats, inv = (np.unique(est, axis=0, return_inverse=True) if extra.size
                 else (np.zeros((1, 0), int), np.zeros(n, int)))
    inv = np.asarray(inv).reshape(-1)
    for k, pat in enumerate(pats):
        rows = np.flatnonzero(inv == k)
        e0 = extra[pat == 0]                      # conditioned on exactly
        e1 = extra[pat == 1]                      # bound z <= c, enforced by rejection
        draw_idx = np.concatenate([cand, e1])
        if e0.size:
            A = np.linalg.solve(Sig[np.ix_(e0, e0)], Sig[np.ix_(e0, draw_idx)]).T
            Cc = Sig[np.ix_(draw_idx, draw_idx)] - A @ Sig[np.ix_(e0, draw_idx)]
        else:
            A = None
            Cc = Sig[np.ix_(draw_idx, draw_idx)]
        L = np.linalg.cholesky(Cc + 1e-10 * np.eye(len(draw_idx)))
        cpos = c[draw_idx]
        for i in rows:
            m = Mu[i, draw_idx].copy()
            if A is not None:
                m += A @ (base_values[i, e0] - Mu[i, e0])
            pool_neg = (m + rng.standard_normal((n_draws, len(draw_idx))) @ L.T) <= cpos
            base_ok = pool_neg[:, K:].all(1) if e1.size else np.ones(len(pool_neg), bool)
            negc = pool_neg[:, :K]
            tested = np.zeros(K, bool)
            for step in range(K + 1):
                acc = base_ok & negc[:, tested].all(1)
                while acc.sum() < min_accept and len(negc) < max_pool:
                    extra_neg = (m + rng.standard_normal((len(negc), len(draw_idx))) @ L.T) <= cpos
                    negc = np.vstack([negc, extra_neg[:, :K]])
                    base_ok = np.concatenate([base_ok, extra_neg[:, K:].all(1) if e1.size
                                              else np.ones(len(extra_neg), bool)])
                    acc = base_ok & negc[:, tested].all(1)
                na = int(acc.sum())
                n_acc[i, step] = na
                if step == K:
                    p_any[i, step] = 0.0
                    break
                sub = negc[acc]
                pj = 1.0 - sub.mean(0) if na else np.full(K, 0.5)
                p_any[i, step] = 1.0 - sub[:, ~tested].all(1).mean() if na else 1.0
                pj[tested] = -1.0
                nxt = int(pj.argmax())
                order[i, step] = nxt
                if positive[i, cand[nxt]]:
                    first_pos[i] = step
                    break
                tested[nxt] = True
        if verbose:
            print(f"    pattern {k}: {rows.size} persons done", flush=True)
    return dict(p_any=p_any, order=order, first_pos=first_pos, n_accept=n_acc)


def outcome_at(sim, lam, sens):
    """Tests used and miss status under stopping threshold lam."""
    pa = sim["p_any"]
    K = pa.shape[1] - 1
    below = pa <= lam
    below[np.isnan(pa)] = False
    stop = np.where(below.any(1), below.argmax(1), K)                   # first step with p_any <= lam
    fp = sim["first_pos"]
    found = fp < stop
    tests = np.where(found, fp + 1, stop)
    missed = sens & ~found
    return tests.astype(float), missed


def certify_lambda(sim, sens, alpha, delta):
    """Largest lambda whose Clopper-Pearson bound on the miss rate among the
    sensitised is <= alpha.

    A sensitised person is missed exactly when the stopping threshold reaches
    m_i = min over steps s <= (step of the first positive) of p_any[i, s].
    The miss count is therefore a step function of lambda that jumps at the
    sorted m_i, and the fixed-sequence test over an increasing lambda grid is
    solved exactly by taking lambda just below the (k* + 1)-th smallest m_i,
    where k* is the largest miss count whose bound is <= alpha.
    """
    pa, fp = sim["p_any"], sim["first_pos"]
    K = pa.shape[1] - 1
    idx = np.flatnonzero(sens)
    m = np.empty(idx.size)
    for t, i in enumerate(idx):
        last = int(fp[i]) if np.isfinite(fp[i]) else K
        v = pa[i, :last + 1]
        m[t] = np.nanmin(v) if np.isfinite(v).any() else np.inf
    m = np.sort(m)
    ns = idx.size
    kstar = -1
    for k in range(ns + 1):
        if cp_upper(k, ns, delta) <= alpha:
            kstar = k
        else:
            break
    if kstar < 0:
        return -np.inf
    if kstar >= ns:
        return 1.0
    return float(np.nextafter(m[kstar], -np.inf))   # just below the (k*+1)-th smallest


# ----------------------------------------------------------------- comparators
def greedy_cover(pos_train):
    """Conditional stepwise selection: each step adds the allergen detecting
    the most not-yet-detected sensitised persons (Sammut 2026, Kulalert 2022)."""
    p = pos_train.shape[1]
    left = pos_train.any(1).copy()
    order, chosen = [], np.zeros(p, bool)
    for _ in range(p):
        gain = np.where(chosen, -1, (pos_train & left[:, None]).sum(0))
        j = int(gain.argmax())
        order.append(j)
        chosen[j] = True
        left &= ~pos_train[:, j]
    return order


def fixed_panel_curve(pos, order):
    sens = pos.any(1)
    out = []
    for k in range(1, len(order) + 1):
        det = pos[:, order[:k]].any(1)
        out.append(dict(k=k, tests=float(k), sensitivity=float(det[sens].mean())))
    return pd.DataFrame(out)


def sequential_curve(pos, order):
    """Test in a fixed order, stop at first positive, at most k tests."""
    sens = pos.any(1)
    hits = pos[:, order]
    first = np.where(hits.any(1), hits.argmax(1), len(order))
    out = []
    for k in range(1, len(order) + 1):
        found = first < k
        tests = np.where(found, first + 1, k)
        out.append(dict(k=k, tests=float(tests.mean()), sensitivity=float(found[sens].mean())))
    return pd.DataFrame(out), first


def certify_k(curve_fn_cal, sens_cal, alpha, delta, kmax):
    """Smallest k whose miss-rate bound on the calibration split is <= alpha."""
    ns = int(sens_cal.sum())
    for k in range(1, kmax + 1):
        missed = curve_fn_cal(k)
        if cp_upper(int(missed.sum()), ns, delta) <= alpha:
            return k
    return kmax


# ----------------------------------------------------------------- dataset 1
def run_serum():
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    tr, ca, te = (d.split == v for v in ("train", "calib", "test"))
    Y = d[["lg_" + c for c in ALL20]].values
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    X = covariate_design(d.RIDAGEYR, d.male, d.RIDRETH1)
    pos = (st[:, :19] == 0)
    pos20 = np.column_stack([pos, np.zeros(len(d), bool)])
    sens = pos.any(1)
    cand = np.arange(19)
    t0 = time.time()
    models = {"factor": CensoredMVN(CENS, "factor", family_mask(families=FAMILIES), n_iter=300,
                                    burn=100, seed=11).fit(Y[tr], st[tr], X[tr]),
              "free": CensoredMVN(CENS, "free", n_iter=300, burn=100, seed=11).fit(Y[tr], st[tr], X[tr]),
              "diagonal": CensoredMVN(CENS, "diagonal", n_iter=300, burn=100,
                                      seed=11).fit(Y[tr], st[tr], X[tr])}
    print(f"models fitted ({time.time() - t0:.0f}s)", flush=True)
    sims = {}
    ev = ca | te
    variants = {"FLIP (total IgE + adaptive)": ("factor", True),
                "FLIP (adaptive, no total IgE)": ("factor", False),
                "FLIP (total IgE + adaptive, free covariance)": ("free", True),
                "Ablation: no cross-reactivity (diagonal)": ("diagonal", True)}
    for nm, (mdl, use_ige) in variants.items():
        base = np.full_like(st, 2)
        if use_ige:
            base[:, 19] = st[:, 19]                       # total IgE measured (exact or censored)
        sims[nm] = simulate_adaptive(models[mdl], pos20[ev], base[ev], Y[ev], X[ev], cand,
                                     seed=101, verbose=False)
        print(f"  simulated {nm} ({time.time() - t0:.0f}s)", flush=True)
    evidx = np.flatnonzero(ev)
    is_ca = ca[evidx]
    is_te = te[evidx]

    def sub(sim, m):
        return {k: v[m] for k, v in sim.items()}

    rows, curves = [], []
    # FLIP variants
    for nm, sim in sims.items():
        sc, stt = sub(sim, is_ca), sub(sim, is_te)
        ige_cost = 1.0 if "total IgE" in nm or "diagonal" in nm else 0.0
        for a in ALPHAS:
            lam = certify_lambda(sc, sens[ca], a, DELTA)
            tests, missed = outcome_at(stt, lam, sens[te])
            rows.append(dict(dataset="NHANES 2005-06 serum", method=nm, alpha=a, operating=f"lambda={lam:.4f}",
                             sensitivity=1 - missed[sens[te]].mean(), mean_tests=tests.mean() + ige_cost,
                             mean_sIgE_tests=tests.mean(), tests_nonsensitised=tests[~sens[te]].mean(),
                             tests_sensitised=tests[sens[te]].mean()))
        for lam in np.unique(np.quantile(stt["p_any"][~np.isnan(stt["p_any"])], np.linspace(0, 1, 120))):
            tests, missed = outcome_at(stt, lam, sens[te])
            curves.append(dict(dataset="serum", method=nm, mean_tests=tests.mean() + ige_cost,
                               sensitivity=1 - missed[sens[te]].mean()))
    # comparators
    order_g = greedy_cover(pos[tr])
    prev = pos[tr].mean(0)
    order_p = list(np.argsort(-prev, kind="stable"))
    for nm, order in (("Fixed panel: greedy set cover", order_g), ("Fixed panel: most prevalent", order_p)):
        for a in ALPHAS:
            k = certify_k(lambda k: sens[ca] & ~pos[ca][:, order[:k]].any(1), sens[ca], a, DELTA, 19)
            det = pos[te][:, order[:k]].any(1)
            rows.append(dict(dataset="NHANES 2005-06 serum", method=nm, alpha=a, operating=f"k={k}",
                             sensitivity=det[sens[te]].mean(), mean_tests=float(k), mean_sIgE_tests=float(k),
                             tests_nonsensitised=float(k), tests_sensitised=float(k)))
        c = fixed_panel_curve(pos[te], order)
        for _, r in c.iterrows():
            curves.append(dict(dataset="serum", method=nm, mean_tests=r.tests, sensitivity=r.sensitivity))
    _, first_ca = sequential_curve(pos[ca], order_g)
    c_te, first_te = sequential_curve(pos[te], order_g)
    for a in ALPHAS:
        k = certify_k(lambda k: sens[ca] & ~(first_ca < k), sens[ca], a, DELTA, 19)
        found = first_te < k
        tests = np.where(found, first_te + 1, k)
        rows.append(dict(dataset="NHANES 2005-06 serum", method="Sequential, greedy order, stop at first positive",
                         alpha=a, operating=f"k={k}", sensitivity=found[sens[te]].mean(),
                         mean_tests=tests.mean(), mean_sIgE_tests=tests.mean(),
                         tests_nonsensitised=tests[~sens[te]].mean(), tests_sensitised=tests[sens[te]].mean()))
    for _, r in c_te.iterrows():
        curves.append(dict(dataset="serum", method="Sequential, greedy order, stop at first positive",
                           mean_tests=r.tests, sensitivity=r.sensitivity))
    # total IgE gate + sequential greedy order; gate and k tuned on TRAINING, certified on calibration
    ige = Y[:, 19]
    _, first_tr = sequential_curve(pos[tr], order_g)
    best = {}
    for a in ALPHAS:
        cand_ops = []
        for g in np.quantile(ige[tr], np.linspace(0, 0.6, 31)):
            for k in range(1, 20):
                gated = ige[tr] < g
                found = (~gated) & (first_tr < k)
                miss = sens[tr] & ~found
                tests = 1 + np.where(gated, 0, np.where(first_tr < k, first_tr + 1, k))
                if miss[sens[tr]].mean() <= a * 0.8:
                    cand_ops.append((tests.mean(), g, k))
        cand_ops.sort()
        # certify on calibration: walk candidate operating points from cheapest
        chosen = (float(np.quantile(ige[tr], 0.0)), 19)
        for _, g, k in cand_ops:
            gated = ige[ca] < g
            found = (~gated) & (first_ca < k)
            if cp_upper(int((sens[ca] & ~found).sum()), int(sens[ca].sum()), DELTA) <= a:
                chosen = (g, k)
                break
        g, k = chosen
        gated = ige[te] < g
        found = (~gated) & (first_te < k)
        tests = 1 + np.where(gated, 0, np.where(first_te < k, first_te + 1, k))
        rows.append(dict(dataset="NHANES 2005-06 serum", method="Total IgE gate + sequential greedy order",
                         alpha=a, operating=f"IgE>={10 ** g:.1f} kU/L, k={k}",
                         sensitivity=found[sens[te]].mean(), mean_tests=tests.mean(),
                         mean_sIgE_tests=(tests - 1).mean(), tests_nonsensitised=(tests - 1)[~sens[te]].mean(),
                         tests_sensitised=(tests - 1)[sens[te]].mean()))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "t2_serum_operating_points.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RES, "t2_serum_curves.csv"), index=False)
    with open(os.path.join(RES, "t2_serum_sims.pkl"), "wb") as fh:
        pickle.dump(dict(sims=sims, evidx=evidx, is_ca=is_ca, is_te=is_te, sens=sens, pos=pos,
                         order_greedy=order_g, names=[NAME[c] for c in SIGE]), fh)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(res.drop(columns=["dataset", "npv"], errors="ignore").round(3).to_string(index=False))
    print("greedy set-cover order:", [NAME[SIGE[j]] for j in order_g])
    print(f"sensitised (test): {sens[te].mean():.3f}")
    return res


# ----------------------------------------------------------------- dataset 2
def run_spt():
    d2 = pd.read_csv(os.path.join(HERE, "data", "d2_nhanes3_spt.csv"))
    d1 = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    tr, ca, te = (d2.split == v for v in ("train", "calib", "test"))
    pos = d2[[f"pos_{k}" for k in SPT]].values.astype(bool)
    sens = pos.any(1)
    st2 = np.where(pos, 3, 1)
    X2 = covariate_design(d2.HSAGEIR, d2.male)
    t0 = time.time()
    in_dom = CensoredMVN(np.zeros(10), "free", probit=True, n_iter=300, burn=100,
                         seed=3).fit(np.zeros((tr.sum(), 10)), st2[tr], X2[tr])
    # transported: dependence learned from serum sIgE positivity (NHANES 2005-06),
    # marginal intercepts re-estimated on the NHANES III training split
    serum = [SPT_TO_SERUM[k] for k in SPT]
    st1 = np.where(d1[["cens_" + c for c in serum]].values == 0, 3, 1)
    X1 = covariate_design(d1.RIDAGEYR, d1.male)
    tr1 = (d1.split == "train").values
    src = CensoredMVN(np.zeros(10), "free", probit=True, n_iter=300, burn=100,
                      seed=3).fit(np.zeros((tr1.sum(), 10)), st1[tr1], X1[tr1])
    indep = CensoredMVN(np.zeros(10), "diagonal", probit=True, n_iter=300, burn=100,
                        seed=3).fit(np.zeros((tr.sum(), 10)), st2[tr], X2[tr])
    trans = CensoredMVN(np.zeros(10), "free", probit=True)
    trans.B = indep.B.copy()                         # NHANES III marginals
    trans.Sigma = src.Sigma.copy()                   # serum cross-reactivity structure
    trans.Q = np.linalg.inv(trans.Sigma)
    trans.rng = np.random.default_rng(5)
    print(f"SPT models fitted ({time.time() - t0:.0f}s)", flush=True)
    ev = ca | te
    evidx = np.flatnonzero(ev)
    is_ca, is_te = ca[evidx], te[evidx]
    cand = np.arange(10)
    sims = {}
    for nm, mdl in (("FLIP (adaptive, in-domain)", in_dom),
                    ("FLIP (adaptive, structure transported from serum)", trans),
                    ("Ablation: no cross-reactivity (diagonal)", indep)):
        sims[nm] = simulate_adaptive(mdl, pos[ev], np.full((ev.sum(), 10), 2), np.zeros((ev.sum(), 10)),
                                     X2[ev], cand, seed=202)
        print(f"  simulated {nm} ({time.time() - t0:.0f}s)", flush=True)
    rows, curves = [], []
    for nm, sim in sims.items():
        sc = {k: v[is_ca] for k, v in sim.items()}
        stt = {k: v[is_te] for k, v in sim.items()}
        for a in ALPHAS:
            lam = certify_lambda(sc, sens[ca], a, DELTA)
            tests, missed = outcome_at(stt, lam, sens[te])
            rows.append(dict(dataset="NHANES III skin prick", method=nm, alpha=a, operating=f"lambda={lam:.4f}",
                             sensitivity=1 - missed[sens[te]].mean(), mean_tests=tests.mean(),
                             tests_nonsensitised=tests[~sens[te]].mean(), tests_sensitised=tests[sens[te]].mean()))
        for lam in np.unique(np.quantile(stt["p_any"][~np.isnan(stt["p_any"])], np.linspace(0, 1, 120))):
            tests, missed = outcome_at(stt, lam, sens[te])
            curves.append(dict(dataset="spt", method=nm, mean_tests=tests.mean(),
                               sensitivity=1 - missed[sens[te]].mean()))
    order_g = greedy_cover(pos[tr])
    order_p = list(np.argsort(-pos[tr].mean(0), kind="stable"))
    for nm, order in (("Fixed panel: greedy set cover", order_g), ("Fixed panel: most prevalent", order_p)):
        for a in ALPHAS:
            k = certify_k(lambda k: sens[ca] & ~pos[ca][:, order[:k]].any(1), sens[ca], a, DELTA, 10)
            det = pos[te][:, order[:k]].any(1)
            rows.append(dict(dataset="NHANES III skin prick", method=nm, alpha=a, operating=f"k={k}",
                             sensitivity=det[sens[te]].mean(), mean_tests=float(k),
                             tests_nonsensitised=float(k), tests_sensitised=float(k)))
        for _, r in fixed_panel_curve(pos[te], order).iterrows():
            curves.append(dict(dataset="spt", method=nm, mean_tests=r.tests, sensitivity=r.sensitivity))
    _, first_ca = sequential_curve(pos[ca], order_g)
    c_te, first_te = sequential_curve(pos[te], order_g)
    for a in ALPHAS:
        k = certify_k(lambda k: sens[ca] & ~(first_ca < k), sens[ca], a, DELTA, 10)
        found = first_te < k
        tests = np.where(found, first_te + 1, k)
        rows.append(dict(dataset="NHANES III skin prick", method="Sequential, greedy order, stop at first positive",
                         alpha=a, operating=f"k={k}", sensitivity=found[sens[te]].mean(),
                         mean_tests=tests.mean(), tests_nonsensitised=tests[~sens[te]].mean(),
                         tests_sensitised=tests[sens[te]].mean()))
    for _, r in c_te.iterrows():
        curves.append(dict(dataset="spt", method="Sequential, greedy order, stop at first positive",
                           mean_tests=r.tests, sensitivity=r.sensitivity))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "t2_spt_operating_points.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RES, "t2_spt_curves.csv"), index=False)
    print(res.drop(columns=["dataset"]).round(3).to_string(index=False))
    print(f"sensitised (test): {sens[te].mean():.3f}")
    return res


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "serum"):
        run_serum()
    if which in ("both", "spt"):
        run_spt()
