"""
a08_literature_nhanes.py
========================
Published allergy studies that analysed the same public NHANES files. For each
study the headline number is first reproduced from the raw files with the
study's own definitions; FLIP is then run on the same data to ask how much of
the study's testing is needed to reach the same answer.

  Salo 2014    sensitisation >= 1 detectable sIgE: 44.6% (>= 6 y, 19 sIgE),
               36.2% (1-5 y, 9 sIgE)
  Gergen 2009  atopy by 15 aeroallergen sIgE 42.5%; greedy fixed panel:
               6 allergens identify 92.8% of atopic persons, 11 identify 99.2%
  Zhu 2022     Spearman rho, D. farinae vs D. pteronyssinus sIgE = 0.88
  McGowan 2019 ages 6-19: shrimp vs cockroach 0.53, shrimp vs dust mite 0.45
  Hoppin 2011  questionnaire prediction of atopy: 44% atopic, 53% >= 1 allergic
               condition, PPV 50-72%, NPV 57-65%
  Arbes 2005   NHANES III skin prick test, >= 1 positive of 10: 54.3%

Run:  python a08_literature_nhanes.py
"""
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import (ALL20, FAMILIES, NAME, SIGE, CensoredMVN,  # noqa: E402
                   covariate_design, family_mask)
from a03_atopy_testing import (DELTA, SPT, certify_lambda, outcome_at,  # noqa: E402
                               simulate_adaptive)

RAW = os.path.join(HERE, "..", "data_raw")
RES = os.path.join(HERE, "results")
FOODS = ["LBXF13", "LBXIF1", "LBXIF2", "LBXF24"]
AERO15 = [c for c in SIGE if c not in FOODS]
KIDS9 = ["LBXID2", "LBXID1", "LBXIE1", "LBXIE5", "LBXII6", "LBXIM6", "LBXF13", "LBXIF1", "LBXIF2"]
LOG_LLOD, LOG_FLOOR = np.log10(0.35), np.log10(0.25)


def read(name):
    return pd.read_sas(os.path.join(RAW, f"{name}.xpt"), format="xport")


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float((x * w).sum() / w.sum())


def phi(a, b, w=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    ma, mb = wmean(a, w), wmean(b, w)
    cov = wmean((a - ma) * (b - mb), w)
    return float(cov / np.sqrt(wmean((a - ma) ** 2, w) * wmean((b - mb) ** 2, w)))


def weighted_greedy(pos, w):
    """Gergen 2009: add the allergen that identifies the largest (weighted)
    number of not-yet-identified atopic persons."""
    p = pos.shape[1]
    left = pos.any(1).copy()
    tot = w[left].sum()
    order, cover, chosen = [], [], np.zeros(p, bool)
    for _ in range(p):
        gain = np.where(chosen, -1.0, ((pos & left[:, None]) * w[:, None]).sum(0))
        j = int(gain.argmax())
        order.append(j)
        chosen[j] = True
        left &= ~pos[:, j]
        cover.append(1.0 - w[left].sum() / tot)
    return order, np.array(cover)


# ======================================================================== part 1
def reproduce():
    ig, demo = read("AL_IGE_D"), read("DEMO_D")
    m = ig.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "WTMEC2YR"]], on="SEQN")
    rows = []

    def add(study, quantity, published, reproduced, n, note=""):
        rows.append(dict(study=study, quantity=quantity, published=published,
                         reproduced=reproduced, n=n, note=note))

    # Salo 2014
    o = m[(m.RIDAGEYR >= 6)].dropna(subset=SIGE)
    add("Salo 2014", "sensitised (>=1 of 19 sIgE >= 0.35 kU/L), age >= 6 y, %", 44.6,
        100 * wmean((o[SIGE] >= 0.35).any(axis=1), o.WTMEC2YR), len(o), "survey-weighted")
    k = m[(m.RIDAGEYR < 6)].dropna(subset=KIDS9)
    add("Salo 2014", "sensitised (>=1 of 9 sIgE), age 1-5 y, %", 36.2,
        100 * wmean((k[KIDS9] >= 0.35).any(axis=1), k.WTMEC2YR), len(k), "survey-weighted")
    # Gergen 2009
    g = m[(m.RIDAGEYR >= 6)].dropna(subset=AERO15)
    pos = (g[AERO15] >= 0.35).values
    add("Gergen 2009", "atopy (>=1 of 15 aeroallergen sIgE), %", 42.5,
        100 * wmean(pos.any(1), g.WTMEC2YR), len(g), "survey-weighted")
    order, cover = weighted_greedy(pos, g.WTMEC2YR.values)
    add("Gergen 2009", "atopic persons identified by the best 6-allergen panel, %", 92.8,
        100 * cover[5], len(g), "weighted greedy; " + ", ".join(NAME[AERO15[j]] for j in order[:6]))
    add("Gergen 2009", "atopic persons identified by the best 11-allergen panel, %", 99.2,
        100 * cover[10], len(g), "weighted greedy; + " + ", ".join(NAME[AERO15[j]] for j in order[6:11]))
    # Zhu 2022
    z = m[(m.RIDAGEYR >= 6)].dropna(subset=SIGE)
    add("Zhu 2022", "Spearman rho, D. farinae vs D. pteronyssinus sIgE", 0.88,
        float(spearmanr(z.LBXID2, z.LBXID1)[0]), len(z), "reported values incl. NHANES fill values")
    # McGowan 2019 (ages 6-19)
    c = m[(m.RIDAGEYR >= 6) & (m.RIDAGEYR <= 19)].dropna(subset=["LBXF24", "LBXII6", "LBXID2", "LBXID1"])
    shr = (c.LBXF24 >= 0.35).values
    coc = (c.LBXII6 >= 0.35).values
    mite = ((c.LBXID2 >= 0.35) | (c.LBXID1 >= 0.35)).values
    add("McGowan 2019", "correlation, shrimp vs cockroach sensitisation (6-19 y)", 0.53,
        phi(shr, coc, c.WTMEC2YR), len(c), "weighted phi coefficient")
    add("McGowan 2019", "correlation, shrimp vs dust-mite sensitisation (6-19 y)", 0.45,
        phi(shr, mite, c.WTMEC2YR), len(c), "weighted phi coefficient; mite = D. farinae or D. pteronyssinus")
    # Hoppin 2011
    agq = read("AGQ_D")
    h = o.merge(agq[["SEQN", "AGQ010", "AGQ040", "AGQ100", "AGQ130", "AGQ180"]], on="SEQN", how="left")
    atopy = (h[SIGE] >= 0.35).any(axis=1).values
    conds = hoppin_conditions(h)
    w = h.WTMEC2YR.values
    add("Hoppin 2011", "atopic (>=1 of 19 sIgE), %", 44.0, 100 * wmean(atopy, w), len(h), "survey-weighted")
    add("Hoppin 2011", "reporting >=1 allergic condition, %", 53.0, 100 * wmean(conds["any condition"], w),
        len(h), "allergy, hay fever, eczema, rhinitis, itchy rash")
    q = questionnaire_metrics(conds, atopy, w)
    add("Hoppin 2011", "PPV of hay fever for atopy, %", 72.0, 100 * q.loc["hay fever", "ppv"], len(h))
    add("Hoppin 2011", "PPV of eczema for atopy, %", 50.0, 100 * q.loc["eczema", "ppv"], len(h))
    add("Hoppin 2011", "NPV of any condition for atopy, %", 65.0, 100 * q.loc["any condition", "npv"], len(h))
    add("Hoppin 2011", "NPV of eczema for atopy, %", 57.0, 100 * q.loc["eczema", "npv"], len(h))
    # Arbes 2005 (NHANES III skin prick test)
    d2 = pd.read_csv(os.path.join(HERE, "data", "d2_nhanes3_spt.csv"))
    add("Arbes 2005", ">=1 positive skin test of 10 (age 6-59 y), %", 54.3,
        100 * wmean(d2.sensitized, d2.WTPFALG6), len(d2), "survey-weighted, net mean wheal >= 3 mm")
    pub = dict(MIT=27.5, RYE=26.9, RAG=26.2, COC=26.1, BER=18.1, CAT=17.0, THI=15.2, WHI=13.2, ALT=12.9, PEA=8.6)
    for s, v in pub.items():
        add("Arbes 2005", f"positive skin test, {s}, %", v, 100 * wmean(d2[f"pos_{s}"], d2.WTPFALG6), len(d2))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "lit_reproduction.csv"), index=False)
    print(out.round(3).to_string(index=False))
    return out


def hoppin_conditions(h):
    yes = lambda s: (s == 1).values
    conds = {"allergy": yes(h.AGQ040), "hay fever": yes(h.AGQ010), "eczema": yes(h.AGQ180),
             "rhinitis": yes(h.AGQ100), "itchy rash": yes(h.AGQ130)}
    conds["any condition"] = np.column_stack(list(conds.values())).any(1)
    return conds


def questionnaire_metrics(conds, atopy, w):
    rows = {}
    for nm, q in conds.items():
        tp, fp = (w * (q & atopy)).sum(), (w * (q & ~atopy)).sum()
        fn, tn = (w * (~q & atopy)).sum(), (w * (~q & ~atopy)).sum()
        rows[nm] = dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp), npv=tn / (tn + fn),
                        acc=(tp + tn) / (tp + tn + fp + fn))
    return pd.DataFrame(rows).T


# ======================================================================== part 2
def curve_weighted(sim, sens, w, n_grid=200):
    pa = sim["p_any"]
    rows = []
    for lam in np.unique(np.quantile(pa[np.isfinite(pa)], np.linspace(0, 1, n_grid))):
        tests, missed = outcome_at(sim, lam, sens)
        found = sens & ~missed
        rows.append(dict(lam=lam, mean_tests=wmean(tests, w), sens_w=w[sens & found].sum() / w[sens].sum(),
                         mean_tests_unw=tests.mean(), sens_unw=found[sens].mean()))
    return pd.DataFrame(rows)


def prevalence_estimates(sim, lam, sens, w):
    """Naive (positives found) and posterior-corrected prevalence at threshold lam."""
    pa = sim["p_any"]
    K = pa.shape[1] - 1
    tests, missed = outcome_at(sim, lam, sens)
    found = sens & ~missed
    stop = tests.astype(int)
    p_stop = np.where(found, 1.0, pa[np.arange(len(stop)), np.minimum(stop, K)])
    p_stop = np.where(np.isfinite(p_stop), p_stop, 0.0)
    return wmean(found, w), wmean(p_stop, w), wmean(tests, w)


def serum_models(d):
    path = os.path.join(RES, "_lit_factor_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    tr = (d.split == "train").values
    Y = d[["lg_" + c for c in ALL20]].values
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    X = covariate_design(d.RIDAGEYR, d.male, d.RIDRETH1)
    mdl = CensoredMVN(np.array([LOG_LLOD] * 19 + [np.log10(2.0)]), "factor", family_mask(families=FAMILIES),
                      n_iter=300, burn=100, seed=11).fit(Y[tr], st[tr], X[tr])
    with open(path, "wb") as fh:
        pickle.dump(mdl, fh)
    return mdl


def gergen_salo_hoppin():
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    ca, te = (d.split == "calib").values, (d.split == "test").values
    ev = ca | te
    Y = d[["lg_" + c for c in ALL20]].values
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    X = covariate_design(d.RIDAGEYR, d.male, d.RIDRETH1)
    w = d.WTMEC2YR.values
    t0 = time.time()
    mdl = serum_models(d)
    print(f"serum factor model ready ({time.time() - t0:.0f}s)", flush=True)
    rows = []

    # ---- Gergen 2009: 15 aeroallergens
    a_idx = np.array([ALL20.index(c) for c in AERO15])
    pos20 = np.zeros_like(st, bool)
    pos20[:, a_idx] = st[:, a_idx] == 0
    sens15 = pos20.any(1)
    sims15 = {}
    for nm, use_ige in (("FLIP adaptive", False), ("FLIP adaptive + total IgE", True)):
        base = np.full_like(st, 2)
        if use_ige:
            base[:, 19] = st[:, 19]
        sims15[nm] = simulate_adaptive(mdl, pos20[ev], base[ev], Y[ev], X[ev], a_idx, seed=303)
        print(f"  Gergen setting simulated: {nm} ({time.time() - t0:.0f}s)", flush=True)
    evidx = np.flatnonzero(ev)
    is_ca, is_te = ca[evidx], te[evidx]
    sub = lambda sim, msk: {k: v[msk] for k, v in sim.items()}
    pos_tr = pos20[(d.split == "train").values][:, a_idx]
    order, _ = weighted_greedy(pos_tr, w[(d.split == "train").values])
    pos_te = pos20[te][:, a_idx]
    sens_te = sens15[te]
    curves = []
    for k_ in range(1, 16):
        det = pos_te[:, order[:k_]].any(1)
        curves.append(dict(study="Gergen 2009", method="Fixed panel, weighted greedy (Gergen 2009)",
                           mean_tests=float(k_), sens_w=w[te][sens_te & det].sum() / w[te][sens_te].sum(),
                           sens_unw=det[sens_te].mean()))
    for nm, sim in sims15.items():
        cw = curve_weighted(sub(sim, is_te), sens_te, w[te])
        extra = 1.0 if "IgE" in nm else 0.0
        for _, r in cw.iterrows():
            curves.append(dict(study="Gergen 2009", method=nm, mean_tests=r.mean_tests + extra,
                               sens_w=r.sens_w, sens_unw=r.sens_unw))
        for target in (0.928, 0.992):
            ok = cw[cw.sens_w >= target]
            rows.append(dict(study="Gergen 2009", method=nm, target=f"identify {100 * target:.1f}% of atopic",
                             published_tests=6.0 if target < 0.95 else 11.0,
                             flip_tests=float(ok.mean_tests.min()) + extra if len(ok) else np.nan,
                             kind="read-off on test split (weighted)"))
            lam = certify_lambda(sub(sim, is_ca), sens15[ca], 1 - target, DELTA)
            tests, missed = outcome_at(sub(sim, is_te), lam, sens_te)
            found = sens_te & ~missed
            rows.append(dict(study="Gergen 2009", method=nm, target=f"identify {100 * target:.1f}% of atopic",
                             published_tests=6.0 if target < 0.95 else 11.0,
                             flip_tests=wmean(tests, w[te]) + extra,
                             achieved=w[te][found].sum() / w[te][sens_te].sum(),
                             kind=f"certified on calibration (alpha={1 - target:.3f}, delta={DELTA})"))
    for k_ in (6, 11):
        det = pos_te[:, order[:k_]].any(1)
        rows.append(dict(study="Gergen 2009", method="Fixed panel, weighted greedy (Gergen 2009)",
                         target=f"{k_}-allergen panel", published_tests=float(k_), flip_tests=float(k_),
                         achieved=w[te][sens_te & det].sum() / w[te][sens_te].sum(), kind="test split"))
    print("Gergen greedy order (train):", [NAME[AERO15[j]] for j in order])

    # ---- Salo 2014 (>= 6 y, 19 sIgE) and Hoppin 2011: reuse the Task 2 simulations
    with open(os.path.join(RES, "t2_serum_sims.pkl"), "rb") as fh:
        t2 = pickle.load(fh)
    assert np.array_equal(t2["evidx"], evidx)
    sens19 = (st[:, :19] == 0).any(1)
    for nm in ("FLIP (adaptive, no total IgE)", "FLIP (total IgE + adaptive)"):
        sim = t2["sims"][nm]
        extra = 1.0 if "total IgE +" in nm else 0.0
        for a in (0.05, 0.01):
            lam = certify_lambda(sub(sim, is_ca), sens19[ca], a, DELTA)
            naive, corr, mt = prevalence_estimates(sub(sim, is_te), lam, sens19[te], w[te])
            rows.append(dict(study="Salo 2014", method=nm, target=f"certified miss rate <= {a}",
                             published_tests=19.0, flip_tests=mt + extra,
                             prev_full=wmean(sens19[te], w[te]), prev_naive=naive, prev_corrected=corr,
                             kind="test split, survey-weighted"))
    # Hoppin 2011: questionnaire versus FLIP after k tests (no total IgE)
    agq = read("AGQ_D")
    h = d.merge(agq[["SEQN", "AGQ010", "AGQ040", "AGQ100", "AGQ130", "AGQ180"]], on="SEQN", how="left")
    conds = hoppin_conditions(h)
    hop = []
    q = questionnaire_metrics({k_: v[te] for k_, v in conds.items()}, sens19[te], w[te])
    for nm_, r in q.iterrows():
        hop.append(dict(predictor=f"questionnaire: {nm_}", tests=0, **r.to_dict()))
    sim = sub(t2["sims"]["FLIP (adaptive, no total IgE)"], is_te)
    for k_ in (1, 2, 3):
        found = sim["first_pos"] < k_
        score = np.where(found, 1.0, sim["p_any"][:, k_])
        pred = found | (score >= 0.5)
        tests = np.where(found, sim["first_pos"] + 1, k_)
        y_, ww = sens19[te], w[te]
        tp, fp = (ww * (pred & y_)).sum(), (ww * (pred & ~y_)).sum()
        fn, tn = (ww * (~pred & y_)).sum(), (ww * (~pred & ~y_)).sum()
        hop.append(dict(predictor=f"FLIP after at most {k_} sIgE test(s)", tests=wmean(tests, ww),
                        sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp), npv=tn / (tn + fn),
                        acc=(tp + tn) / (tp + tn + fp + fn), auroc=roc_auc_score(y_, score, sample_weight=ww)))
    hop = pd.DataFrame(hop)
    hop.to_csv(os.path.join(RES, "lit_hoppin.csv"), index=False)
    print(hop.round(3).to_string(index=False))
    return rows, curves


def salo_kids():
    """Salo 2014, ages 1-5 y (9 sIgE): FLIP adaptive testing with rotating
    cross-fitting (train on two folds, certify on one, test on one)."""
    ig, demo = read("AL_IGE_D"), read("DEMO_D")
    m = ig.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "WTMEC2YR"]], on="SEQN")
    cols = KIDS9 + ["LBXIGE"]
    k = m[m.RIDAGEYR < 6].dropna(subset=cols).reset_index(drop=True)
    st = np.column_stack([(k[f"LBD{c[3:]}LC"] == 1).astype(int) if c != "LBXIGE"
                          else (k["LBDIGELC"] == 1).astype(int) for c in cols])
    floor = np.array([LOG_FLOOR] * 9 + [np.log10(1.41)])
    Y = np.where(st == 1, floor, np.log10(k[cols].values))
    X = covariate_design(np.maximum(k.RIDAGEYR, 1), k.RIAGENDR == 1, k.RIDRETH1)
    cens = np.array([LOG_LLOD] * 9 + [np.log10(2.0)])
    pos = np.column_stack([st[:, :9] == 0, np.zeros(len(k), bool)])
    sens = pos.any(1)
    w = k.WTMEC2YR.values
    fold = np.random.default_rng(20261026).permutation(len(k)) % 4
    rows, recs = [], []
    t0 = time.time()
    for r in range(4):
        te, ca = fold == r, fold == (r + 1) % 4
        tr = ~(te | ca)
        mdl = CensoredMVN(cens, "free", n_iter=300, burn=100, seed=7 + r).fit(Y[tr], st[tr], X[tr])
        ev = ca | te
        evidx = np.flatnonzero(ev)
        for nm, use_ige in (("FLIP adaptive", False), ("FLIP adaptive + total IgE", True)):
            base = np.full_like(st, 2)
            if use_ige:
                base[:, 9] = st[:, 9]
            sim = simulate_adaptive(mdl, pos[ev], base[ev], Y[ev], X[ev], np.arange(9), seed=404 + r)
            sc = {kk: v[ca[evidx]] for kk, v in sim.items()}
            stt = {kk: v[te[evidx]] for kk, v in sim.items()}
            for a in (0.10, 0.05):
                lam = certify_lambda(sc, sens[ca], a, DELTA)
                tests, missed = outcome_at(stt, lam, sens[te])
                found = sens[te] & ~missed
                _, corr, _ = prevalence_estimates(stt, lam, sens[te], w[te])
                for i, gi in enumerate(np.flatnonzero(te)):
                    recs.append(dict(method=nm, alpha=a, rot=r, idx=gi, tests=tests[i] + use_ige,
                                     sens=sens[te][i], found=found[i], w=w[gi]))
                rows.append(dict(method=nm, alpha=a, rot=r, prev_corrected=corr, prev_full=wmean(sens[te], w[te])))
        print(f"  kids rotation {r} done ({time.time() - t0:.0f}s)", flush=True)
    rec = pd.DataFrame(recs)
    pr = pd.DataFrame(rows)
    out = []
    for (nm, a), g in rec.groupby(["method", "alpha"]):
        s = g[g.sens]
        p_ = pr[(pr.method == nm) & (pr.alpha == a)]
        out.append(dict(study="Salo 2014 (1-5 y)", method=nm, target=f"certified miss rate <= {a}",
                        published_tests=9.0, flip_tests=wmean(g.tests, g.w),
                        achieved=(s.found * s.w).sum() / s.w.sum(), achieved_unw=s.found.mean(),
                        prev_full=100 * wmean(g.sens, g.w), prev_corrected=100 * p_.prev_corrected.mean(),
                        n=len(g), kind="4-fold rotating cross-fit, survey-weighted"))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    return out


def arbes_spt():
    d2 = pd.read_csv(os.path.join(HERE, "data", "d2_nhanes3_spt.csv"))
    tr, ca, te = ((d2.split == v).values for v in ("train", "calib", "test"))
    pos = d2[[f"pos_{s}" for s in SPT]].values.astype(bool)
    sens = pos.any(1)
    w = d2.WTPFALG6.values
    st2 = np.where(pos, 3, 1)
    X2 = covariate_design(d2.HSAGEIR, d2.male)
    mdl = CensoredMVN(np.zeros(10), "free", probit=True, n_iter=300, burn=100,
                      seed=3).fit(np.zeros((tr.sum(), 10)), st2[tr], X2[tr])
    ev = ca | te
    evidx = np.flatnonzero(ev)
    sim = simulate_adaptive(mdl, pos[ev], np.full((ev.sum(), 10), 2), np.zeros((ev.sum(), 10)),
                            X2[ev], np.arange(10), seed=505)
    sc = {k: v[ca[evidx]] for k, v in sim.items()}
    stt = {k: v[te[evidx]] for k, v in sim.items()}
    rows = []
    for a in (0.05, 0.01):
        lam = certify_lambda(sc, sens[ca], a, DELTA)
        naive, corr, mt = prevalence_estimates(stt, lam, sens[te], w[te])
        tests, missed = outcome_at(stt, lam, sens[te])
        found = sens[te] & ~missed
        rows.append(dict(study="Arbes 2005", method="FLIP adaptive (skin prick)", target=f"certified miss rate <= {a}",
                         published_tests=10.0, flip_tests=mt, achieved=w[te][found].sum() / w[te][sens[te]].sum(),
                         prev_full=wmean(sens[te], w[te]), prev_naive=naive, prev_corrected=corr,
                         kind="test split, survey-weighted"))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return rows


def latent_correlations():
    """Zhu 2022 and McGowan 2019: censoring-corrected latent correlations."""
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    out = []
    free = pd.read_csv(os.path.join(RES, "corr_latent_free.csv"), index_col=0)
    out.append(dict(study="Zhu 2022", pair="D. farinae - D. pteronyssinus", group=">= 6 y",
                    latent=float(free.loc["D. farinae", "D. pteronyssinus"])))
    c = d[(d.RIDAGEYR <= 19)].reset_index(drop=True)
    cols = ["LBXF24", "LBXII6", "LBXID2", "LBXID1"]
    Y = c[["lg_" + x for x in cols]].values
    st = c[["cens_" + x for x in cols]].values.astype(int)
    X = covariate_design(c.RIDAGEYR, c.male, c.RIDRETH1)
    mdl = CensoredMVN(np.full(4, LOG_LLOD), "free", n_iter=400, burn=150, seed=19).fit(Y, st, X)
    S = mdl.Sigma
    R = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    out.append(dict(study="McGowan 2019", pair="shrimp - cockroach", group="6-19 y", latent=float(R[0, 1])))
    out.append(dict(study="McGowan 2019", pair="shrimp - D. farinae", group="6-19 y", latent=float(R[0, 2])))
    out.append(dict(study="McGowan 2019", pair="shrimp - D. pteronyssinus", group="6-19 y", latent=float(R[0, 3])))
    out = pd.DataFrame(out)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "reproduce"):
        reproduce()
    if which in ("all", "flip"):
        rows, curves = gergen_salo_hoppin()
        rows += salo_kids()
        rows += arbes_spt()
        pd.DataFrame(rows).to_csv(os.path.join(RES, "lit_flip.csv"), index=False)
        pd.DataFrame(curves).to_csv(os.path.join(RES, "lit_gergen_curves.csv"), index=False)
        latent_correlations().to_csv(os.path.join(RES, "lit_latent_corr.csv"), index=False)
