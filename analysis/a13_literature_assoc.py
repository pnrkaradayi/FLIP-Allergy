"""Association studies on NHANES 2005-2006 re-analysed with adaptive FLIP testing.

Salo et al. 2011 (J Allergy Clin Immunol 127:1226): current hay fever and current allergies in
relation to plant-specific IgE, ages >= 6 y.
Shargorodsky et al. 2015 (PLoS One 10:e0131957): active smoking (self-report and serum cotinine) in
relation to inhaled-allergen sensitisation, adults >= 20 y.

Each published estimate is first reproduced with the full panel and the study's own definitions.
The sensitisation variable defined by the panel is then replaced by the classification that adaptive
FLIP testing (total IgE first, then sIgE within the allergens that define the variable) reaches at a
certified miss rate of at most 5%, using 4-fold rotating cross-fitting over the analysis cohort
(model fitted on two folds, stopping threshold certified on the third, persons classified on the
fourth), and the same survey-weighted models are refitted.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import ALL20, CENS, FAMILIES, NAME, SIGE, CensoredMVN, covariate_design, family_mask  # noqa: E402
from a03_atopy_testing import DELTA, certify_lambda, outcome_at, simulate_adaptive  # noqa: E402

RAW = os.path.join(HERE, "..", "data_raw")
RES = os.path.join(HERE, "results")
PLANT = ["LBXIG5", "LBXIG2", "LBXIT7", "LBXIT3", "LBXIW1", "LBXW11"]   # grasses, trees, weeds
FOODS = ["LBXF13", "LBXIF1", "LBXIF2", "LBXF24"]
AERO15 = [c for c in SIGE if c not in FOODS]
ALPHA = 0.05


def read(name, sub=""):
    return pd.read_sas(os.path.join(RAW, sub, f"{name}.xpt"), format="xport")


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float((x * w).sum() / w.sum())


def weighted_or(y, X, w, term):
    """Survey-weighted logistic regression (point estimate); odds ratio(s) for the columns `term`."""
    m = LogisticRegression(C=np.inf, max_iter=5000)
    m.fit(X.values, np.asarray(y, int), sample_weight=w / w.mean())
    coef = dict(zip(X.columns, m.coef_[0]))
    return {t: float(np.exp(coef[t])) for t in term}


def dummies(s, prefix, ref):
    d = pd.get_dummies(s.astype("category"), prefix=prefix, dtype=float)
    return d.drop(columns=[f"{prefix}_{ref}"], errors="ignore")


# ----------------------------------------------------------------- data
def cohort():
    d = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    agq = read("AGQ_D")[["SEQN", "AGQ010", "AGQ030", "AGQ040", "AGQ060", "AGQ100"]]
    smq = read("SMQ_D")[["SEQN", "SMQ020", "SMQ040"]]
    demo = read("DEMO_D")[["SEQN", "DMDEDUC2", "DMDHREDU"]]
    cot = read("COT_D", "nhanes_extra")[["SEQN", "LBXCOT"]]
    for t in (agq, smq, demo, cot):
        t["SEQN"] = t.SEQN.astype(int)
    d["SEQN"] = d.SEQN.astype(int)
    d = d.merge(agq, on="SEQN", how="left").merge(smq, on="SEQN", how="left")
    d = d.merge(demo, on="SEQN", how="left").merge(cot, on="SEQN", how="left")
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    d["plant_pos"] = (st[:, [ALL20.index(c) for c in PLANT]] == 0).any(1)
    d["aero_pos"] = (st[:, [ALL20.index(c) for c in AERO15]] == 0).any(1)
    d["hayfever"] = (d.AGQ010 == 1) & (d.AGQ030 == 1)
    d["allergies"] = (d.AGQ040 == 1) & (d.AGQ060 == 1)
    return d


def salo_design(d):
    """Covariates of Salo 2011: age, sex, race/ethnicity, education, poverty, BMI, serum cotinine."""
    age = pd.cut(d.RIDAGEYR, [5, 11, 17, 29, 39, 49, 59, 200], labels=False).astype(int)
    edu = pd.cut(d.DMDHREDU.where(d.DMDHREDU <= 5), [0, 2, 3, 5], labels=["lt_hs", "hs", "gt_hs"])
    edu = edu.astype(object).where(edu.notna(), "missing")
    pir = d.INDFMPIR.fillna(d.INDFMPIR.median())
    X = pd.concat([dummies(age, "age", 0), d.male.astype(float).rename("male"),
                   dummies(d.RIDRETH1.astype(int), "race", 3), dummies(edu, "edu", "gt_hs"),
                   pir.rename("pir"), d.INDFMPIR.isna().astype(float).rename("pir_na"),
                   d.BMXBMI.fillna(d.BMXBMI.median()).rename("bmi"),
                   np.log10(d.LBXCOT.fillna(d.LBXCOT.median()).clip(lower=0.011)).rename("lcot")], axis=1)
    return X


def shargorodsky_sample(d):
    a = d[(d.RIDAGEYR >= 20) & d.LBXCOT.notna() & d.DMDEDUC2.between(1, 5) & d.BMXBMI.notna()
          & d.AGQ100.isin([1, 2])].copy()
    never = (a.SMQ020 == 2) & (a.LBXCOT < 10)
    former = (a.SMQ020 == 1) & (a.SMQ040 == 3) & (a.LBXCOT < 10)
    current = a.LBXCOT >= 10
    cat = np.select([never, former, current & (a.LBXCOT < 144), current & (a.LBXCOT < 285), current],
                    ["never", "former", "cur1", "cur2", "cur3"], default="other")
    a["smoke"] = cat
    return a[a.smoke != "other"].copy()


def shargorodsky_design(a, model):
    X = dummies(pd.Series(a.smoke.values, index=a.index), "smk", "never")
    if model >= 2:
        X = pd.concat([X, dummies(pd.cut(a.RIDAGEYR, [19, 38, 53, 200], labels=False).astype(int), "age", 0),
                       a.male.astype(float).rename("male")], axis=1)
    if model >= 3:
        race = a.RIDRETH1.map({1: "mex", 2: "other", 3: "white", 4: "black", 5: "other"})
        edu = pd.cut(a.DMDEDUC2, [0, 2, 3, 5], labels=["lt_hs", "hs", "gt_hs"])
        bmi = pd.cut(a.BMXBMI, [0, 25, 30, 200], right=False, labels=["normal", "over", "obese"])
        X = pd.concat([X, dummies(race, "race", "white"), dummies(edu, "edu", "gt_hs"),
                       dummies(bmi, "bmi", "normal")], axis=1)
    return X


# ----------------------------------------------------------------- FLIP classification
def flip_classify(d, targets, seed=20261031):
    """Rotating 4-fold cross-fitting; returns per target the FLIP classification and tests used."""
    Y = d[["lg_" + c for c in ALL20]].values
    st = d[["cens_" + c for c in ALL20]].values.astype(int)
    X = covariate_design(d.RIDAGEYR, d.male, d.RIDRETH1)
    fold = np.random.default_rng(seed).permutation(len(d)) % 4
    out = {k: dict(cls=np.zeros(len(d), bool), tests=np.zeros(len(d))) for k in targets}
    base = np.full_like(st, 2)
    base[:, 19] = st[:, 19]                                     # total IgE measured first
    t0 = time.time()
    for r in range(4):
        te, ca = fold == r, fold == (r + 1) % 4
        tr = ~(te | ca)
        mdl = CensoredMVN(CENS, "factor", family_mask(families=FAMILIES), n_iter=300, burn=100,
                          seed=11 + r).fit(Y[tr], st[tr], X[tr])
        ev = ca | te
        evidx = np.flatnonzero(ev)
        for k, (cols, popmask) in targets.items():
            idx = np.array([ALL20.index(c) for c in cols])
            pos = np.zeros_like(st, bool)
            pos[:, idx] = st[:, idx] == 0
            sens = pos.any(1)
            sim = simulate_adaptive(mdl, pos[ev], base[ev], Y[ev], X[ev], idx, seed=500 + 10 * r)
            in_ca = ca[evidx] & popmask[evidx]
            lam = certify_lambda({q: v[in_ca] for q, v in sim.items()}, sens[evidx][in_ca], ALPHA, DELTA)
            in_te = te[evidx]
            tests, missed = outcome_at({q: v[in_te] for q, v in sim.items()}, lam, sens[evidx][in_te])
            gi = evidx[in_te]
            out[k]["cls"][gi] = sens[gi] & ~missed
            out[k]["tests"][gi] = tests + 1.0                    # + total IgE
            print(f"  rotation {r}, {k}: lambda={lam:.4f}, sIgE tests {tests.mean():.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------- main
def main():
    d = cohort()
    w = d.WTMEC2YR.values
    rows = []

    def add(study, quantity, published, full, flip=np.nan, flip_tests=np.nan, design_tests=np.nan, n=np.nan,
            note=""):
        rows.append(dict(study=study, quantity=quantity, published=published, reproduced_full_panel=full,
                         flip=flip, flip_tests=flip_tests, design_tests=design_tests, n=n, note=note))

    # ---- Salo 2011: prevalences on everybody >= 6 y with blood drawn for IgE (as published, N = 8086)
    ig, demo, agq = read("AL_IGE_D"), read("DEMO_D"), read("AGQ_D")
    m = ig[["SEQN", "LBXIGE"]].merge(demo[["SEQN", "RIDAGEYR", "WTMEC2YR"]], on="SEQN").merge(agq, on="SEQN")
    m = m[(m.RIDAGEYR >= 6) & m.LBXIGE.notna()]
    add("Salo 2011", "current hay fever, %", 6.6,
        100 * wmean((m.AGQ010 == 1) & (m.AGQ030 == 1), m.WTMEC2YR), n=len(m), note="survey-weighted")
    add("Salo 2011", "current allergies, %", 23.5,
        100 * wmean((m.AGQ040 == 1) & (m.AGQ060 == 1), m.WTMEC2YR), n=len(m), note="survey-weighted")

    sh = shargorodsky_sample(d)
    adults = d.index.isin(sh.index)
    targets = {"plant": (PLANT, np.ones(len(d), bool)), "aero": (AERO15, adults)}
    fl = flip_classify(d, targets)
    d["plant_flip"], d["aero_flip"] = fl["plant"]["cls"], fl["aero"]["cls"]

    # ---- Salo 2011: adjusted OR of current hay fever / allergies for plant-specific IgE
    Xs = salo_design(d)
    for outcome, pub in (("hayfever", 4.75), ("allergies", np.nan)):
        full = weighted_or(d[outcome], pd.concat([d.plant_pos.astype(float).rename("plant"), Xs], axis=1), w,
                           ["plant"])["plant"]
        flp = weighted_or(d[outcome], pd.concat([d.plant_flip.astype(float).rename("plant"), Xs], axis=1), w,
                          ["plant"])["plant"]
        add("Salo 2011", f"adjusted OR, current {outcome} vs plant-specific IgE", pub, full, flp,
            wmean(fl["plant"]["tests"], w), 6.0, len(d),
            "survey-weighted; plant = 6 pollen sIgE; FLIP tests incl. total IgE")

    # ---- Shargorodsky 2015: active smoking and inhaled-allergen sensitisation (adults)
    sh = sh.join(d[["aero_flip"]])
    ws = sh.WTMEC2YR.values
    add("Shargorodsky 2015", "inhaled-allergen sensitisation, % (unweighted)", 43.0, 100 * sh.aero_pos.mean(),
        100 * sh.aero_flip.mean(), sh.index.map(lambda i: fl["aero"]["tests"][i]).values.mean(), 15.0, len(sh),
        "FLIP tests incl. total IgE")
    published = {1: dict(smk_former=0.76, smk_cur1=0.97, smk_cur2=0.75, smk_cur3=0.80),
                 3: dict(smk_former=0.87, smk_cur1=0.80, smk_cur2=0.68, smk_cur3=0.71)}
    tests_sh = np.array([fl["aero"]["tests"][i] for i in sh.index])
    for model in (1, 3):
        Xm = shargorodsky_design(sh, model)
        terms = [t for t in published[model] if t in Xm.columns]
        full = weighted_or(sh.aero_pos, Xm, ws, terms)
        flp = weighted_or(sh.aero_flip, Xm, ws, terms)
        for t in terms:
            add("Shargorodsky 2015", f"OR model {model}, {t.replace('smk_', '')} vs never", published[model][t],
                full[t], flp[t], wmean(tests_sh, ws), 15.0, len(sh), "survey-weighted; FLIP tests incl. total IgE")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "lit_assoc.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
