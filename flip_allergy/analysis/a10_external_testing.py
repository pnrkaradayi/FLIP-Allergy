"""
a10_external_testing.py
=======================
Task 2 of the main analysis (how few allergen tests certify that a person is
sensitised, with a guaranteed miss rate) and hidden-component prediction, run
on the open patient-level data of four published allergy studies:

  acc     Allergen Chip Challenge, ISAC chips (Martinroche et al. 2026)
  maas    MAAS birth cohort, ISAC at age 11 (Fontanella et al. 2018)
  guo     Chinese children with allergic asthma, 18 sIgE (Guo et al. 2026)
  moscow  Moscow paediatric biochip, 31 binary sIgE (Smoldovskaya et al. 2018)

Every person is evaluated once by rotating four-fold cross-fitting (latent
model fitted on two folds, stopping threshold certified on the third,
outcomes counted on the fourth). Comparators are the fixed panels of the
panel-optimisation literature (greedy set cover, certified on the same fold)
and sequential testing in the greedy order.

Run:  python a10_external_testing.py [dataset ...]
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext_data  # noqa: E402
from a03_atopy_testing import (certify_k, certify_lambda, greedy_cover, outcome_at,  # noqa: E402
                               sequential_curve, simulate_adaptive)
from flipx import CensoredMVN, covariate_design  # noqa: E402

RES = os.path.join(HERE, "results")
ALPHAS = (0.10, 0.05, 0.01)
DELTA = 0.10


norm_name = ext_data.norm_name


EXTRACT_FAMILIES = ext_data.EXTRACT_FAMILIES


family_mask_for = ext_data.family_mask_for


def prepare(key):
    D = ext_data.LOADERS[key]()
    X = D["X"]
    cols = list(X.columns)
    ex = D["extra"]
    if key == "moscow":
        pos = X.values > 0.5
        Y = np.zeros(pos.shape)
        st = np.where(pos, 3, 1)
        cens = np.zeros(len(cols))
        probit = True
        cov = covariate_design(np.maximum(ex["age"], 0.5), ex["male"])
    else:
        lod = np.array([D["lod"][c] for c in cols])
        V = X.values.astype(float)
        ok = np.isfinite(V)
        if key == "guo":                                # a handful of unmeasured cells -> below LOD
            V = np.where(ok, V, 0.0)
        pos = V >= lod
        cens = np.log10(lod)
        Y = np.where(pos, np.log10(np.maximum(V, 1e-9)), cens - 0.5 * np.log10(2))
        st = np.where(pos, 0, 1)
        probit = False
        if key == "acc":
            age = pd.Series(ex["age"]).fillna(np.nanmedian(ex["age"])).values
            cov = covariate_design(np.maximum(age, 0.5), ex["male"])
        elif key == "guo":
            age = pd.Series(ex["age"]).fillna(np.nanmedian(ex["age"])).values
            cov = np.column_stack([covariate_design(np.maximum(age, 0.5), ex["male"]),
                                   (ex["region"] == "South").astype(float)])
        else:
            cov = np.ones((len(X), 1))
    structure = "factor"
    fmask, fams = family_mask_for(cols, EXTRACT_FAMILIES.get(key))
    return dict(D=D, cols=cols, Y=Y, st=st, pos=pos, cens=cens, probit=probit, cov=cov, structure=structure,
                fmask=fmask, fams=fams)


def fit(P, idx, structure, seed):
    kw = dict(n_iter=250, burn=90, seed=seed)
    if structure == "factor":
        m = CensoredMVN(P["cens"], "factor", P["fmask"], probit=P["probit"], **kw)
    else:
        m = CensoredMVN(P["cens"], structure, probit=P["probit"], **kw)
    return m.fit(P["Y"][idx], P["st"][idx], P["cov"][idx])


def run(key):
    P = prepare(key)
    n = len(P["Y"])
    pos, sens = P["pos"], P["pos"].any(1)
    rng = np.random.default_rng(20261027)
    fold = rng.permutation(n) % 4
    recs, curves, hid = [], [], []
    t0 = time.time()
    # the chip cohort is large enough for a single 50/25/25 split (one rotation)
    for r in range(1 if key == "acc" else 4):
        te, ca = fold == r, fold == (r + 1) % 4
        tr = ~(te | ca)
        models = {"FLIP adaptive": fit(P, tr, P["structure"], 11 + r),
                  "FLIP adaptive, free covariance": fit(P, tr, "free", 11 + r),
                  "Ablation: no cross-reactivity (diagonal)": fit(P, tr, "diagonal", 11 + r)}
        if key == "acc":                               # 105 x 105 free covariance: not fitted
            models.pop("FLIP adaptive, free covariance")
        ev = ca | te
        evidx = np.flatnonzero(ev)
        isc, ist = ca[evidx], te[evidx]
        cand = np.arange(len(P["cols"]))
        for nm, mdl in models.items():
            sim = simulate_adaptive(mdl, pos[ev], np.full((ev.sum(), len(cand)), 2), np.zeros((ev.sum(), len(cand))),
                                    P["cov"][ev], cand, seed=600 + r, n_draws=8192, min_accept=400,
                                    max_pool=131072)
            sc = {k: v[isc] for k, v in sim.items()}
            stt = {k: v[ist] for k, v in sim.items()}
            for a in ALPHAS:
                lam = certify_lambda(sc, sens[ca], a, DELTA)
                tests, missed = outcome_at(stt, lam, sens[te])
                for gi, t_, m_ in zip(np.flatnonzero(te), tests, missed):
                    recs.append(dict(dataset=key, method=nm, alpha=a, rot=r, idx=gi, tests=t_, sens=sens[gi],
                                     missed=m_))
            for lam in np.unique(np.quantile(stt["p_any"][np.isfinite(stt["p_any"])], np.linspace(0, 1, 80))):
                tests, missed = outcome_at(stt, lam, sens[te])
                curves.append(dict(dataset=key, method=nm, rot=r, mean_tests=tests.mean(),
                                   sensitivity=1 - missed[sens[te]].mean()))
        order = greedy_cover(pos[tr])
        _, first_ca = sequential_curve(pos[ca], order)
        _, first_te = sequential_curve(pos[te], order)
        K = len(order)
        for a in ALPHAS:
            k = certify_k(lambda k: sens[ca] & ~pos[ca][:, order[:k]].any(1), sens[ca], a, DELTA, K)
            det = pos[te][:, order[:k]].any(1)
            for gi, d_ in zip(np.flatnonzero(te), det):
                recs.append(dict(dataset=key, method="Fixed panel: greedy set cover", alpha=a, rot=r, idx=gi,
                                 tests=float(k), sens=sens[gi], missed=sens[gi] and not d_))
            k = certify_k(lambda k: sens[ca] & ~(first_ca < k), sens[ca], a, DELTA, K)
            found = first_te < k
            tests = np.where(found, first_te + 1, k)
            for gi, t_, f_ in zip(np.flatnonzero(te), tests, found):
                recs.append(dict(dataset=key, method="Sequential, greedy order, stop at first positive", alpha=a,
                                 rot=r, idx=gi, tests=float(t_), sens=sens[gi], missed=sens[gi] and not f_))
        for k_ in range(1, K + 1):
            det = pos[te][:, order[:k_]].any(1)
            curves.append(dict(dataset=key, method="Fixed panel: greedy set cover", rot=r, mean_tests=float(k_),
                               sensitivity=det[sens[te]].mean()))
        # hidden-component prediction: 25% of each test person's components hidden at random
        tidx = np.flatnonzero(te)
        hide = rng.random((len(tidx), len(P["cols"]))) < 0.25
        stat = P["st"][tidx].copy()
        stat[hide] = 2
        for nm, mdl in models.items():
            Z = mdl.conditional_draws(P["Y"][tidx], stat, P["cov"][tidx], n_chains=8, n_sweeps=24, keep=10,
                                      seed=700 + r)
            ppos = (Z > P["cens"][None, None, :]).mean(1)
            for j, c in enumerate(P["cols"]):
                h = hide[:, j]
                hid.append(dict(dataset=key, model=nm, rot=r, component=c, y=pos[tidx][h, j], p=ppos[h, j]))
        print(f"  {key}: rotation {r} done ({time.time() - t0:.0f}s)", flush=True)
    rec = pd.DataFrame(recs)
    if key == "acc":
        print(f"  acc: evaluated on the held-out quarter, n = {rec.idx.nunique()}", flush=True)
    rec.to_csv(os.path.join(RES, f"ext_testing_{key}_records.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RES, f"ext_testing_{key}_curves.csv"), index=False)
    out = []
    for (m, a), g in rec.groupby(["method", "alpha"]):
        s = g[g.sens.astype(bool)]
        out.append(dict(dataset=key, method=m, alpha=a, n=len(g), panel_size=len(P["cols"]),
                        prevalence=g.sens.mean(), sensitivity=1 - s.missed.mean(), mean_tests=g.tests.mean(),
                        tests_nonsensitised=g[~g.sens.astype(bool)].tests.mean(),
                        tests_sensitised=s.tests.mean()))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RES, f"ext_testing_{key}.csv"), index=False)
    hrows = []
    for (m, c), g in pd.DataFrame(hid).groupby(["model", "component"]):
        yy = np.concatenate(g.y.values)
        pp = np.concatenate(g.p.values)
        hrows.append(dict(dataset=key, model=m, component=c, n_hidden=len(yy), positives=int(yy.sum()),
                          auroc=roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan))
    H = pd.DataFrame(hrows)
    allh = pd.DataFrame(hid)
    for m in H.model.unique():
        g = allh[allh.model == m]
        yy, pp = np.concatenate(g.y.values), np.concatenate(g.p.values)
        H = pd.concat([H, pd.DataFrame([dict(dataset=key, model=m, component="(pooled)", n_hidden=len(yy),
                                             positives=int(yy.sum()), auroc=roc_auc_score(yy, pp))])])
    H.to_csv(os.path.join(RES, f"ext_hidden_{key}.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(out.round(3).to_string(index=False))
    print(H[H.component == "(pooled)"].round(3).to_string(index=False))
    print("median per-component AUROC:", H[H.component != "(pooled)"].groupby("model").auroc.median().round(3).to_dict())
    return out


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    for k in (sys.argv[1:] or ["guo", "moscow", "maas", "acc"]):
        run(k)
