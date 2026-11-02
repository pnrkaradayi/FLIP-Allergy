"""Three further NHANES 2005-2006 studies re-analysed with adaptive FLIP testing.

Busse et al. 2013 (Ann Allergy Asthma Immunol 110:247): atopy (>= 1 of 19 sIgE >= 0.35 kU/L) among adults
with current asthma aged 20-40 and >= 55 years.
Arbes et al. 2013 (Clin Exp Allergy 43:544): current asthma by atopy and blood eosinophil percent
(dichotomised at the median), ages >= 6; adjusted prevalences and the excess prevalence due to
interaction (p11 - p10 - p01 + p00). The abstract does not list the adjustment covariates; age group,
sex and race/ethnicity are used here.
Keet et al. 2012 (J Allergy Clin Immunol 129:169): food sensitisation (milk, egg or peanut sIgE
>= 0.35 kU/L) by personal nativity, ages < 21, adjusted for race/ethnicity, sex, age and household income.

Published numbers come from the abstracts (the full texts were not accessible). Each is reproduced on the
raw files first; then, in the analysis cohort (complete 19-allergen panels, ages >= 6), the panel-defined
variable is replaced by the adaptive FLIP classification (a13.flip_classify: total IgE first, certified
miss rate <= 5%, rotating 4-fold cross-fitting) and the same quantities are recomputed.
Writes results/lit_assoc_more.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a13_literature_assoc as A  # noqa: E402

FOOD3 = ["LBXF13", "LBXIF1", "LBXIF2"]          # peanut, egg white, milk


def raw():
    ig, demo, mcq, cbc = A.read("AL_IGE_D"), A.read("DEMO_D"), A.read("MCQ_D"), A.read("CBC_D")
    m = ig.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "WTMEC2YR", "DMDBORN", "INDHHINC"]], on="SEQN")
    m = m.merge(mcq[["SEQN", "MCQ010", "MCQ035"]], on="SEQN", how="left").merge(
        cbc[["SEQN", "LBXEOPCT"]], on="SEQN", how="left")
    m["asthma"] = (m.MCQ010 == 1) & (m.MCQ035 == 1)
    return m


def adjusted_prevalence(y, group, X, w):
    """Predictive margins: mean predicted probability with everybody set to each of the four groups."""
    G = pd.get_dummies(pd.Series(group, index=X.index).astype("category"), prefix="g", dtype=float)
    G = G.drop(columns="g_0", errors="ignore")
    Z = pd.concat([G, X], axis=1)
    m = LogisticRegression(C=np.inf, max_iter=5000).fit(Z.values, np.asarray(y, int), sample_weight=w / w.mean())
    out = {}
    for g in range(4):
        Zg = Z.copy()
        Zg[[c for c in G.columns]] = 0.0
        if f"g_{g}" in Zg:
            Zg[f"g_{g}"] = 1.0
        out[g] = 100 * A.wmean(m.predict_proba(Zg.values)[:, 1], w)
    out["excess"] = out[3] - out[1] - out[2] + out[0]
    return out


def arbes_design(d):
    age = pd.cut(d.RIDAGEYR, [5, 17, 34, 54, 200], labels=False).astype(int)
    return pd.concat([A.dummies(age, "age", 0), d.male.astype(float).rename("male"),
                      A.dummies(d.RIDRETH1.astype(int), "race", 3)], axis=1)


def keet_design(d, us_born):
    inc = pd.cut(d.INDHHINC.where(d.INDHHINC.between(1, 11)), [0, 4, 7, 10, 11], labels=["lt20", "20_45", "45_75", "ge75"])
    inc = inc.astype(object).where(inc.notna(), "missing")
    age = pd.cut(d.RIDAGEYR, [0, 5, 11, 20], labels=False).astype(int)
    return pd.concat([pd.Series(us_born, index=d.index, dtype=float, name="us_born"),
                      A.dummies(d.RIDRETH1.astype(int), "race", 3), d.male.astype(float).rename("male"),
                      A.dummies(age, "age", 0), A.dummies(pd.Series(inc, index=d.index), "inc", "ge75")], axis=1)


def main():
    rows = []

    def add(study, quantity, published, full, flip=np.nan, flip_tests=np.nan, design_tests=np.nan, n=np.nan,
            note=""):
        rows.append(dict(study=study, quantity=quantity, published=published, reproduced_full_panel=full,
                         flip=flip, flip_tests=flip_tests, design_tests=design_tests, n=n, note=note))

    m = raw()
    full19 = m[A.SIGE].notna().all(axis=1)
    m["male"] = (m.RIAGENDR == 1).astype(float)
    m["atopy"] = (m[A.SIGE] >= 0.35).any(axis=1)

    # ---- Busse 2013 (reproduction on the raw files)
    b = m[full19 & (m.RIDAGEYR.between(20, 40) | (m.RIDAGEYR >= 55)) & m.MCQ010.isin([1, 2])].copy()
    b["old"] = b.RIDAGEYR >= 55
    add("Busse 2013", "adults 20-40 or >= 55 y with sIgE, n", 2573, len(b), n=len(b))
    for old, lab, n_pub, pct_pub, at_pub in ((False, "20-40 y", 108, 6.7, 75.4), (True, ">= 55 y", 43, 4.5, 65.2)):
        g = b[b.old == old]
        add("Busse 2013", f"current asthma, {lab}, n", n_pub, int(g.asthma.sum()), n=len(g))
        add("Busse 2013", f"current asthma, {lab}, %", pct_pub, 100 * A.wmean(g.asthma, g.WTMEC2YR), n=len(g),
            note="survey-weighted")
        a = g[g.asthma]
        add("Busse 2013", f"atopic among current asthma, {lab}, %", at_pub, 100 * A.wmean(a.atopy, a.WTMEC2YR),
            n=len(a), note="survey-weighted")

    # ---- Arbes 2013 (reproduction)
    r = m[full19 & (m.RIDAGEYR >= 6) & m.LBXEOPCT.notna() & m.MCQ010.isin([1, 2])].copy()
    med = r.LBXEOPCT.median()
    r["grp"] = r.atopy.astype(int) + 2 * (r.LBXEOPCT > med).astype(int)        # 0 neither, 1 atopy, 2 eos, 3 both
    ap = adjusted_prevalence(r.asthma, r.grp.values, arbes_design(r), r.WTMEC2YR.values)
    for g, lab, pub in ((0, "neither", 4.6), (1, "atopy alone", 7.6), (2, "high eosinophils alone", 6.9),
                        (3, "both", 17.2), ("excess", "excess prevalence due to interaction", 7.2)):
        add("Arbes 2013", f"adjusted prevalence of asthma, {lab}, %" if g != "excess" else f"{lab}, %", pub,
            ap[g], n=len(r), note=f"survey-weighted; eosinophil median {med:.1f}%")
    for lo, hi, lab, pub in ((6, 17, "6-17 y", 15.3), (55, 200, ">= 55 y", -0.2)):
        s = r[r.RIDAGEYR.between(lo, hi)]
        e = adjusted_prevalence(s.asthma, s.grp.values, arbes_design(s).drop(columns=[c for c in arbes_design(s)
                                if c.startswith("age_")]), s.WTMEC2YR.values)["excess"]
        add("Arbes 2013", f"excess prevalence due to interaction, {lab}, %", pub, e, n=len(s), note="survey-weighted")

    # ---- Keet 2012 (reproduction)
    k = m[(m.RIDAGEYR < 21) & m[FOOD3].notna().all(axis=1) & m.DMDBORN.isin([1, 2, 3])].copy()
    k["food"] = (k[FOOD3] >= 0.35).any(axis=1)
    add("Keet 2012", "participants < 21 y, n", 3550, len(k), n=len(k))
    add("Keet 2012", "adjusted OR, food sensitisation, US-born vs foreign-born", 2.05,
        A.weighted_or(k.food, keet_design(k, k.DMDBORN == 1), k.WTMEC2YR.values, ["us_born"])["us_born"],
        n=len(k), note="survey-weighted")

    # ---- FLIP in the analysis cohort (complete panels, ages >= 6)
    d = A.cohort()
    demo = A.read("DEMO_D")[["SEQN", "DMDBORN", "INDHHINC"]]
    mc = A.read("MCQ_D")[["SEQN", "MCQ010", "MCQ035"]]
    cb = A.read("CBC_D")[["SEQN", "LBXEOPCT"]]
    for t in (demo, mc, cb):
        t["SEQN"] = t.SEQN.astype(int)
    d = d.merge(demo, on="SEQN", how="left").merge(mc, on="SEQN", how="left").merge(cb, on="SEQN", how="left")
    d["asthma"] = (d.MCQ010 == 1) & (d.MCQ035 == 1)
    st = d[["cens_" + c for c in A.ALL20]].values.astype(int)
    d["atopy"] = (st[:, :19] == 0).any(1)
    d["food"] = (st[:, [A.ALL20.index(c) for c in FOOD3]] == 0).any(1)
    busse = (d.RIDAGEYR.between(20, 40) | (d.RIDAGEYR >= 55)).values
    kids = ((d.RIDAGEYR < 21) & d.DMDBORN.isin([1, 2, 3])).values
    fl = A.flip_classify(d, {"atopy": (A.SIGE, np.ones(len(d), bool)), "atopy_busse": (A.SIGE, busse),
                             "food": (FOOD3, kids)}, seed=20261102)
    w = d.WTMEC2YR.values

    sub = busse & d.MCQ010.isin([1, 2]).values
    for old, lab in ((False, "20-40 y"), (True, ">= 55 y")):
        g = sub & ((d.RIDAGEYR.values >= 55) == old)
        a = g & d.asthma.values
        add("Busse 2013", f"atopic among current asthma, {lab}, %, analysis cohort", np.nan,
            100 * A.wmean(d.atopy[a], w[a]), 100 * A.wmean(fl["atopy_busse"]["cls"][a], w[a]),
            A.wmean(fl["atopy_busse"]["tests"][g], w[g]), 19.0, int(a.sum()),
            "survey-weighted; FLIP tests incl. total IgE, over all adults of the age group")

    rr = d.LBXEOPCT.notna().values & d.MCQ010.isin([1, 2]).values
    s = d[rr].copy()
    med = s.LBXEOPCT.median()
    hi = (s.LBXEOPCT > med).astype(int).values
    for lab, at in (("full panel", s.atopy.values), ("FLIP", fl["atopy"]["cls"][rr])):
        s[f"grp_{lab}"] = at.astype(int) + 2 * hi
    apf = adjusted_prevalence(s.asthma, s["grp_full panel"].values, arbes_design(s), s.WTMEC2YR.values)
    apq = adjusted_prevalence(s.asthma, s["grp_FLIP"].values, arbes_design(s), s.WTMEC2YR.values)
    for g, lab in ((0, "neither"), (1, "atopy alone"), (2, "high eosinophils alone"), (3, "both"),
                   ("excess", "excess prevalence due to interaction")):
        add("Arbes 2013", f"{lab}, %, analysis cohort", np.nan, apf[g], apq[g],
            A.wmean(fl["atopy"]["tests"][rr], w[rr]), 19.0, int(rr.sum()), "survey-weighted; FLIP tests incl. total IgE")

    kk = kids
    kd = d[kk].copy()
    us = (kd.DMDBORN == 1).values
    Xk = keet_design(kd, us)
    add("Keet 2012", "adjusted OR, food sensitisation, US-born vs foreign-born, 6-20 y, analysis cohort", np.nan,
        A.weighted_or(kd.food, Xk, w[kk], ["us_born"])["us_born"],
        A.weighted_or(pd.Series(fl["food"]["cls"][kk], index=kd.index), Xk, w[kk], ["us_born"])["us_born"],
        A.wmean(fl["food"]["tests"][kk], w[kk]), 3.0, int(kk.sum()), "survey-weighted; FLIP tests incl. total IgE")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(A.RES, "lit_assoc_more.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 80)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
