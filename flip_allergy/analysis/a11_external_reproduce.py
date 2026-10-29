"""
a11_external_reproduce.py
=========================
For every external dataset: reproduce the headline number of the published
study with the study's own definitions and protocol (as far as the paper
specifies it), and run the analyses that use the datasets' NATURAL
missingness:

  * Detrano et al. 1989: model built on Cleveland, applied at Hungary,
    Switzerland and Long Beach VA, where expensive tests were often not done;
  * Arzt-Gradwohl et al. 2026: culprit insect (bee vs vespid) when venom
    components and the basophil activation test were not ordered.

Run:  python a11_external_reproduce.py
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext_data  # noqa: E402
from a09_external_flip import task  # noqa: E402
from flipg import (Copula, Logistic, amputate, expand, fit_latent, latent_draws, make_imputer,  # noqa: E402
                   scores_on_draws)
from flipx import flip_from_draws, ltt_threshold  # noqa: E402

RES = os.path.join(HERE, "results")
ROWS = []


def add(study, dataset, quantity, published, reproduced, note=""):
    ROWS.append(dict(study=study, dataset=dataset, quantity=quantity, published=published,
                     reproduced=reproduced, note=note))
    print(f"  {study:28s} {quantity:70s} published {published!s:>8}  ours {reproduced:8.3f}  {note}")


def cv_auc(X, y, make, folds=5, reps=5, seed=0, metric="auc"):
    vals = []
    for r in range(reps):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        vals.append(roc_auc_score(y, p) if metric == "auc" else ((p >= 0.5) == y).mean())
    return float(np.mean(vals))


class L1:
    def __init__(self):
        self.m = None

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
        self.m = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear", max_iter=5000,
                                      scoring="roc_auc").fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba((X - self.mu) / self.sd)


# ============================================================ allergy studies
def acc():
    D = ext_data.acc()
    d = D["extra"]["all"]
    comp = list(d.columns[15:])
    V = d[comp].astype(float).values
    ours = np.nanmax(np.where(np.isfinite(V), V, 0), 1) >= 0.3
    add("Martinroche 2026", "Allergen Chip Challenge", "sensitised patients, % (n = 4271)", 84.0, 100 * ours.mean(),
        f"any component >= 0.3; agreement with the database label {100 * (ours == (d.Sensitization == 1)).mean():.1f}%")


def maas():
    D = ext_data.maas()
    a = D["extra"]["all"]
    comp = [c for c in a.columns if c != "asthma"]
    sens_any = (a[comp] > 0.30).any(axis=1)
    active = [c for c in comp if (a.loc[sens_any, c] > 0.30).mean() >= 0.05]
    add("Fontanella 2018", "MAAS", "number of active components (> 0.30 in >= 5% of sensitised children)", 44,
        len(active), f"paper's list: {len(D['X'].columns)} components; overlap {len(set(active) & set(D['X'].columns))}")
    active = list(D["X"].columns)
    add("Fontanella 2018", "MAAS", "children sensitised to >= 1 active component, %", 46.2,
        100 * (a[active] > 0.30).any(axis=1).mean(), f"n = {len(a)}")
    X = np.log10(np.maximum(D["X"].values, 0.15))
    add("Fontanella 2018", "MAAS", "AUROC for asthma, penalised logistic regression on components", 0.73,
        cv_auc(X, D["y"], L1), f"5x repeated 5-fold CV, n = {len(X)} with asthma status")


def graz():
    d = ext_data.graz_venom()["extra"]["raw"]
    comp_done = d[["Api_m_2", "Api_m_3", "Api_m_5", "Api_m_10"]].notna().all(axis=1)
    # complete data for the venom of each group (sIgE incl. components, intradermal test, BAT)
    full_bee = comp_done & d[["bat_bee", "idt_bee"]].notna().all(axis=1)
    full_ves = d[["bat_wasp", "idt_wasp"]].notna().all(axis=1)
    bee, ves = (d.venom == "bee") & full_bee, (d.venom == "wasp") & full_ves
    add("Arzt-Gradwohl 2026", "Graz venom", "bee venom-allergic patients with complete test data", 112,
        float(bee.sum()))
    add("Arzt-Gradwohl 2026", "Graz venom", "vespid venom-allergic patients with complete test data", 460,
        float(ves.sum()))
    dual = lambda v: np.where(d.tige < 30, v >= 0.1, v >= 0.35)
    for nm, grp, col, s35, sdual in (("bee", bee, "bee", 85.7, 94.6), ("vespid", ves, "wasp", 92.2, 97.0)):
        add("Arzt-Gradwohl 2026", "Graz venom", f"sensitivity of {nm} venom sIgE, cut-off 0.35, %", s35,
            100 * (d.loc[grp, col] >= 0.35).mean())
        add("Arzt-Gradwohl 2026", "Graz venom", f"sensitivity of {nm} venom sIgE, dual cut-off, %", sdual,
            100 * dual(d[col])[grp.values].mean())
    pos = (d.bee >= 0.35) & (d.wasp >= 0.35)          # double positives (85 bee, 366 vespid, as published)
    r_vb = d.wasp / d.bee.replace(0, np.nan)
    r_bv = d.bee / d.wasp.replace(0, np.nan)
    vb = (r_vb >= 2.68) | ((d.bee == 0) & (d.wasp > 0))
    bv = (r_bv >= 5.34) | ((d.wasp == 0) & (d.bee > 0))
    for nm, mask, cls, pub in (("vespid/bee >= 2.68", vb, "wasp", 59.3), ("bee/vespid >= 5.34", bv, "bee", 50.6)):
        g = pos & (d.venom == cls)
        o = pos & d.venom.isin(["bee", "wasp"]) & (d.venom != cls)
        add("Arzt-Gradwohl 2026", "Graz venom", f"ratio rule {nm} in double-positive patients: sensitivity, %",
            pub, 100 * mask[g].mean(), f"n = {int(g.sum())}; specificity {100 * (1 - mask[o].mean()):.1f}% "
                                       f"(published 100)")
    allg = d.venom.isin(["bee", "wasp"])
    correct = ((d.venom == "wasp") & vb & ~bv) | ((d.venom == "bee") & bv & ~vb)
    add("Arzt-Gradwohl 2026", "Graz venom", "culprit correctly identified by the ratio rules, bee, %", 58.4,
        100 * correct[d.venom == "bee"].mean())
    add("Arzt-Gradwohl 2026", "Graz venom", "culprit correctly identified by the ratio rules, vespid, %", 77.9,
        100 * correct[d.venom == "wasp"].mean())
    return d, vb, bv, allg


def guo():
    D = ext_data.guo()
    X, reg = D["X"], D["extra"]["region"]
    for code, rgn, pub in (("粉尘螨D2", "South", 72.0), ("艾蒿w6", "North", 53.0), ("法国菊w7", "North", 50.0)):
        add("Guo 2026", "Chinese asthmatic children", f"{code[-2:]} sensitisation in the {rgn}, %", pub,
            100 * (X.loc[reg == rgn, code] >= 0.35).mean(), f"n = {(reg == rgn).sum()}")


def moscow():
    D = ext_data.moscow()
    X, ex = D["X"], D["extra"]
    dx = ex["diagnosis"]
    for comp, ref, a, pub in (("Cat dander", "Upper airway disorders", "Asthma", 1.808),
                              ("Dog dander", "Upper airway disorders", "Asthma", 2.225),
                              ("Cat dander", "Dermatitis", "Asthma", 2.344),
                              ("Dog dander", "Dermatitis", "Asthma", 2.401),
                              ("Alder", "Dermatitis", "Asthma", 2.980)):
        m = np.isin(dx, [a, ref])
        Z = sm.add_constant(np.column_stack([X.loc[m, comp].values, ex["age"][m], ex["male"][m]]))
        fit = sm.Logit((dx[m] == a).astype(float), Z).fit(disp=0)
        add("Smoldovskaya 2018", "Moscow biochip", f"adjusted OR, {comp} sIgE, {a} vs {ref}", pub,
            float(np.exp(fit.params[1])), "logistic regression adjusted for age and sex")


def sosall():
    D = ext_data.sosall()
    raw = D["extra"]["raw"]
    add("Zhakparov 2026", "SOS-ALL", "children with antibody data (of 217)", "-", float(len(raw)),
        f"AD {int((raw.diagnosis == 'AD').sum())}, healthy {int((raw.diagnosis == 'HC').sum())}")
    acc_rf = cv_auc(D["X"].values, D["y"], lambda: RandomForestClassifier(500, random_state=0), metric="acc")
    add("Zhakparov 2026", "SOS-ALL", "random-forest accuracy for AD (published: questionnaire + antibodies)", 0.81,
        acc_rf, "ours: antibodies only, 5x repeated 5-fold CV")


# ============================================================ clinical studies
def heart_deploy():
    D = task("heart")
    cols = list(D["X"].columns)
    Xr = D["X"].values.astype(float)
    y, site = D["y"], D["extra"]["site"]
    cle = site == "Cleveland"
    cop = Copula([D["kind"][c] for c in cols]).fit(Xr[cle])
    T = cop.transform(Xr)
    lat = fit_latent(cop, T[cle], seed=1)
    imp = make_imputer("missforest", 0).fit(T[cle])
    clf = Logistic().fit(imp.transform(T[cle]), y[cle])
    tau = 0.0                                          # probability 0.5, as in Detrano et al.
    # calibrate a FLIP stopping threshold on Cleveland by amputation (MAR, p_order 0.6, 20 rounds)
    rng = np.random.default_rng(5)
    bidx = [[cols.index(c) for c in b] for _, b in D["blocks"]]
    comp = cle & np.isfinite(Xr).all(1)
    scal, fcal = [], []
    for _ in range(20):
        Tf = T[comp]
        Mb = amputate(Tf, bidx, [0.6] * len(bidx), rng, "MAR")
        To = Tf.copy()
        To[expand(Mb, bidx, len(cols))] = np.nan
        p_ = (scores_on_draws(clf, latent_draws(lat, cop, To, 8, 24, 8, seed=int(rng.integers(1e6)))) >= tau).mean(1)
        scal.append(np.minimum(p_, 1 - p_))
        fcal.append((p_ >= 0.5) != (clf.score(Tf) >= tau))
    lam = ltt_threshold(np.concatenate(scal), np.concatenate(fcal), 0.05, 0.10)
    pubacc = {"Hungary": 77, "Long Beach VA": 79, "Switzerland": 81}
    out = []
    for s in ("Hungary", "Switzerland", "Long Beach VA"):
        m = site == s
        To = T[m]
        d_imp = clf.score(imp.transform(To)) >= tau
        d_mean = clf.score(SimpleImputer().fit(T[cle]).transform(To)) >= tau
        F = latent_draws(lat, cop, To, 12, 28, 12, seed=3)
        e = scores_on_draws(clf, F)
        p_ = (e >= tau).mean(1)
        d_post = p_ >= 0.5
        U = np.minimum(p_, 1 - p_)
        acc_ = U <= lam
        miss = np.isnan(To).any(1).mean()
        add("Detrano 1989", f"Heart disease, {s}", "accuracy of the Cleveland model, %", pubacc[s],
            100 * (d_imp == y[m]).mean(), f"ours with MissForest-type imputation; {100 * miss:.0f}% of patients "
                                          f"missing >= 1 test")
        out.append(dict(site=s, n=int(m.sum()), prevalence=y[m].mean(), missing_any=miss,
                        acc_mean=(d_mean == y[m]).mean(), acc_missforest=(d_imp == y[m]).mean(),
                        acc_flip_posterior=(d_post == y[m]).mean(), accepted=acc_.mean(),
                        acc_accepted=(d_post == y[m])[acc_].mean() if acc_.any() else np.nan,
                        acc_deferred=(d_post == y[m])[~acc_].mean() if (~acc_).any() else np.nan,
                        published_cdf=pubacc[s]))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RES, "ext_heart_deployment.csv"), index=False)
    print(out.round(3).to_string(index=False))


def graz_natural(d, vb, bv):
    """Culprit decision from routine tests; flip probability that the undone components/BAT/IDT would change it."""
    D = task("graz")
    cols = list(D["X"].columns)
    Xr = D["X"].values.astype(float)
    y = D["y"]
    raw = d[d.venom.isin(["bee", "wasp"])].reset_index(drop=True)
    rng = np.random.default_rng(11)
    folds = rng.permutation(len(y)) % 5
    p_bee, flipp = np.zeros(len(y)), np.zeros(len(y))
    for f in range(5):
        te, tr = folds == f, folds != f
        cop = Copula([D["kind"][c] for c in cols], [D["lod"].get(c, 1.0) for c in cols]).fit(Xr[tr])
        T = cop.transform(Xr)
        fmask = ext_data.family_mask_for(cols, ext_data.EXTRACT_FAMILIES["graz"])[0]
        lat = fit_latent(cop, T[tr], seed=f, fmask=fmask)
        imp = make_imputer("missforest", f).fit(T[tr])
        clf = Logistic().fit(imp.transform(T[tr]), y[tr], seed=f)
        e = scores_on_draws(clf, latent_draws(lat, cop, T[te], 12, 28, 12, seed=f))
        dh = clf.score(imp.transform(T[te])) >= 0.0
        p_bee[te] = (1.0 / (1.0 + np.exp(-e))).mean(1)       # posterior predictive probability of bee venom
        flipp[te] = flip_from_draws(e, 0.0, dh)
    undone = ~np.isfinite(Xr[:, [cols.index(c) for c in sum((b for _, b in D["blocks"]), [])]]).all(1)
    sel = d.venom.isin(["bee", "wasp"]).values
    ratio_decided = (vb | bv).values[sel] & ~((vb & bv).values[sel])
    ratio_correct = ((y == 0) & vb.values[sel]) | ((y == 1) & bv.values[sel])
    ratio_err = 1 - ratio_correct[ratio_decided].mean()
    conf = np.maximum(p_bee, 1 - p_bee)
    pred = (p_bee >= 0.5).astype(int)
    order = np.argsort(-conf, kind="stable")
    err_cum = np.cumsum(pred[order] != y[order]) / np.arange(1, len(y) + 1)
    rows = [dict(rule="Ratio rules (Arzt-Gradwohl 2026; thresholds chosen in-sample)", decided=ratio_decided.mean(),
                 errors_among_decided=ratio_err)]
    for target in (ratio_err, 0.01, 0.02):
        ok = np.flatnonzero(err_cum <= target + 1e-12)
        cov = (ok.max() + 1) / len(y) if ok.size else 0.0
        rows.append(dict(rule=f"Decision model + FLIP completions, cross-fitted, error <= {100 * target:.1f}%",
                         decided=cov, errors_among_decided=float(err_cum[ok.max()]) if ok.size else np.nan))
    rows.append(dict(rule="Decision model, all patients", decided=1.0, errors_among_decided=float((pred != y).mean())))
    rows.append(dict(rule="patients with >= 1 test not done", decided=undone.mean(), errors_among_decided=np.nan))
    rows.append(dict(rule="of these: undone tests could change the decision (flip prob. > 0.05)",
                     decided=(flipp[undone] > 0.05).mean(), errors_among_decided=np.nan))
    pd.DataFrame(dict(p_bee=p_bee, y=y, flip=flipp, undone=undone)).to_csv(
        os.path.join(RES, "ext_graz_predictions.csv"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ext_graz_culprit.csv"), index=False)
    print(out.round(3).to_string(index=False))


def thyroid():
    D = ext_data.thyroid()
    cls = pd.Series(D["extra"]["cls"]).str.strip()
    X = D["X"].values.astype(float)
    y5 = pd.factorize(cls)[0]
    errs = []
    native = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 3)
    for r in range(10):
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=r).split(X, y5):
            Xt, Xs = X[tr], X[te]
            if not native:
                im = SimpleImputer(strategy="median").fit(Xt)
                Xt, Xs = im.transform(Xt), im.transform(Xs)
            m = DecisionTreeClassifier(min_samples_leaf=2, random_state=r).fit(Xt, y5[tr])
            errs.append((m.predict(Xs) != y5[te]).mean())
    add("Quinlan 1996", "Thyroid (allhypo)", "C4.5 error, five classes, ten 10-fold CVs, %", 0.48, 100 * np.mean(errs),
        "ours: CART decision tree")


def hepatitis():
    D = ext_data.hepatitis()
    X = SimpleImputer(strategy="median").fit_transform(D["X"].values.astype(float))
    add("Diaconis & Efron 1983 (Gong)", "Hepatitis", "classification accuracy, %", 80.0,
        100 * cv_auc(X, D["y"], L1, metric="acc"), "ours: L1 logistic, median imputation, 5x5-fold CV")


def pima():
    D = ext_data.pima()
    X = SimpleImputer(strategy="median").fit_transform(D["X"].values.astype(float))
    y = D["y"]
    vals = []
    for r in range(10):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=r).split(X, y):
            p[te] = L1().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y, p)
        k = np.argmin(np.abs(tpr - (1 - fpr)))
        vals.append(tpr[k])
    add("Smith 1988", "Pima diabetes", "sensitivity = specificity at the crossover point", 0.76, float(np.mean(vals)),
        "ours: L1 logistic, 4-fold CV (576/192 as in the paper), 10 repeats")


def ckd():
    D = ext_data.ckd()
    feats = ["bp", "sg", "al", "bu", "hemo", "htn", "dm", "appet_poor"]       # Qin et al. feature subset
    X = KNNImputer(n_neighbors=5).fit_transform(D["X"][feats].values.astype(float))   # before CV, as published
    acc = []
    for r in range(10):
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1234 + r).split(X, D["y"]):
            m = RandomForestClassifier(300, random_state=r).fit(X[tr], D["y"][tr])
            acc.append((m.predict(X[te]) == D["y"][te]).mean())
    add("Qin 2020", "Chronic kidney disease", "random-forest accuracy, %", 99.75, 100 * np.mean(acc),
        "KNN imputation on the whole data before 4-fold CV, as published")


def support2():
    D = ext_data.support2()
    X = D["X"].copy()
    raw = ext_data.read_arff(os.path.join(ext_data.EXT, "support2", "Support2.arff"))
    X["hday"] = raw.hday.astype(float).values                              # days in hospital before entry
    for c, v in dict(bun=6.51, glucose=np.nan, urine=2502.0).items():      # Knaus physiology incl. urea, urine
        X[c] = raw[c].astype(float).fillna(v).values
    normal = dict(alb=3.5, pafi=333.3, bili=1.01, crea=1.01, wblc=9.0)   # Knaus fill-in values
    for c, v in normal.items():
        X[c] = X[c].fillna(v)
    X = SimpleImputer(strategy="median").fit_transform(X.values.astype(float))
    rng = np.random.default_rng(0)
    vals = []
    for r in range(10):
        idx = rng.permutation(len(X))
        a, b = idx[: len(X) // 2], idx[len(X) // 2:]
        m = L1().fit(X[a], D["y"][a])
        vals.append(roc_auc_score(D["y"][b], m.predict_proba(X[b])[:, 1]))
    add("Knaus 1995", "SUPPORT", "AUROC for 180-day mortality (validation)", 0.78, float(np.mean(vals)),
        "ours: L1 logistic, normal fill-in values, random half-splits")


def physionet():
    D = ext_data.physionet2012()
    X = SimpleImputer(strategy="mean").fit_transform(D["X"].values.astype(float))
    add("Che 2018", "PhysioNet 2012", "AUROC, logistic regression with mean imputation (LR-Mean)", 0.7423,
        cv_auc(X, D["y"], L1, reps=2), "ours: L1 logistic on 48-h summaries, 5-fold CV")
    add("Che 2018", "PhysioNet 2012", "AUROC, GRU-D (best published model)", 0.8424,
        cv_auc(X, D["y"], lambda: RandomForestClassifier(500, min_samples_leaf=3, random_state=0), reps=1),
        "ours: random forest on 48-h summaries, 5-fold CV")


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    print("== allergy studies")
    acc()
    maas()
    d, vb, bv, _ = graz()
    guo()
    moscow()
    sosall()
    print("== clinical studies")
    thyroid()
    hepatitis()
    pima()
    ckd()
    support2()
    heart_deploy()
    graz_natural(d, vb, bv)
    if os.path.exists(os.path.join(ext_data.EXT, "physionet2012", "Outcomes-a.txt")):
        try:
            physionet()
        except Exception as e:  # noqa: BLE001
            print("PhysioNet skipped:", e)
    pd.DataFrame(ROWS).to_csv(os.path.join(RES, "ext_reproduction.csv"), index=False)
