"""Visness et al. 2009 (J Allergy Clin Immunol 123:1163): obesity and IgE in children aged 2-19 y.

BMI-for-age percentiles are computed from the CDC 2000 reference (LMS parameters, bmiagerev.csv);
children below the 5th percentile are excluded, 5th-<85th is normal weight, 85th-<95th overweight and
>= 95th obese. Atopy is a positive sIgE (>= 0.35 kU/L) among the allergens tested at that age (9 below
6 y, 19 from 6 y; full panel required) and food sensitisation a positive peanut, egg or milk sIgE (and
shrimp from 6 y). The age-adjusted survey-weighted odds ratios of Table II are reproduced first; then,
in the 6-19-year-olds of the analysis cohort, atopy and food sensitisation are classified by adaptive
FLIP testing (certified miss rate <= 5%, rotating 4-fold cross-fitting, a13.flip_classify) and the
same models are refitted. Writes results/lit_assoc_visness.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a13_literature_assoc as A  # noqa: E402

LMS = os.path.join(A.RAW, "external", "cdc_growth", "bmiagerev.csv")
KIDS9 = ["LBXID2", "LBXID1", "LBXIE1", "LBXIE5", "LBXII6", "LBXIM6", "LBXF13", "LBXIF1", "LBXIF2"]


def bmi_category(bmi, age_months, male):
    """CDC 2000 BMI-for-age category: 0 < 5th, 1 normal, 2 overweight, 3 obese (NaN outside 2-20 y)."""
    ref = pd.read_csv(LMS)
    ref = ref[pd.to_numeric(ref.Sex, errors="coerce").isin([1, 2])].astype(float)
    out = np.full(len(bmi), np.nan)
    for sex, code in ((True, 1.0), (False, 2.0)):
        r = ref[ref.Sex == code].sort_values("Agemos")
        m = (np.asarray(male) == sex) & np.isfinite(bmi) & np.isfinite(age_months)
        a = np.clip(np.floor(age_months[m]) + 0.5, r.Agemos.min(), r.Agemos.max())   # CDC convention
        L, M, S = (np.interp(a, r.Agemos, r[c]) for c in ("L", "M", "S"))
        z = ((bmi[m] / M) ** L - 1) / (L * S)
        p = norm.cdf(z)
        out[np.flatnonzero(m)] = np.select([p < 0.05, p < 0.85, p < 0.95], [0, 1, 2], 3)
    return out


def age_adjusted_or(y, cat, age, w):
    X = pd.DataFrame({"over": (cat == 2).astype(float), "obese": (cat == 3).astype(float), "age": age})
    return A.weighted_or(y, X, np.asarray(w, float), ["over", "obese"])


def main():
    rows = []
    # ---- reproduction on the published sample (ages 2-19, raw files)
    ig, demo, bmx = A.read("AL_IGE_D"), A.read("DEMO_D"), A.read("BMX_D")
    m = ig.merge(demo[["SEQN", "RIDAGEYR", "RIDAGEEX", "RIAGENDR", "WTMEC2YR"]], on="SEQN").merge(
        bmx[["SEQN", "BMXBMI"]], on="SEQN")
    m = m[(m.RIDAGEYR >= 2) & (m.RIDAGEYR <= 19)].reset_index(drop=True)
    m["cat"] = bmi_category(m.BMXBMI.values, m.RIDAGEEX.values, (m.RIAGENDR == 1).values)
    young = m.RIDAGEYR < 6
    full = np.where(young, m[KIDS9].notna().all(1), m[A.SIGE].notna().all(1))
    pos = lambda cols: (m[cols] >= 0.35).any(axis=1)
    m["atopy"] = np.where(young, pos(KIDS9), pos(A.SIGE))
    m["food"] = np.where(young, pos(["LBXF13", "LBXIF1", "LBXIF2"]), pos(A.FOODS))
    s = m[full & m.cat.isin([1, 2, 3])]
    for out, pub_pct, pub_or in (("atopy", (44.5, 48.9, 51.8), (1.14, 1.28)),
                                 ("food", (21.1, 24.4, 29.2), (1.26, 1.61))):
        for c, lab, pub in zip((1, 2, 3), ("normal weight", "overweight", "obese"), pub_pct):
            g = s[s.cat == c]
            rows.append(dict(study="Visness 2009", quantity=f"{out}, {lab}, %", published=pub,
                             reproduced_full_panel=100 * A.wmean(g[out], g.WTMEC2YR), n=len(g),
                             note="ages 2-19, survey-weighted"))
        r = age_adjusted_or(s[out], s.cat.values, s.RIDAGEYR.values, s.WTMEC2YR.values)
        for t, pub in zip(("over", "obese"), pub_or):
            rows.append(dict(study="Visness 2009", quantity=f"age-adjusted OR, {out}, {t} vs normal",
                             published=pub, reproduced_full_panel=r[t], n=len(s), note="ages 2-19, survey-weighted"))

    # ---- FLIP in the 6-19-year-olds of the analysis cohort (complete 19-allergen panels)
    d = A.cohort()
    dm = demo[["SEQN", "RIDAGEEX"]].copy()
    dm["SEQN"] = dm.SEQN.astype(int)
    d = d.merge(dm, on="SEQN", how="left")
    d["cat"] = bmi_category(d.BMXBMI.values, d.RIDAGEEX.values, (d.male == 1).values)
    kids = (d.RIDAGEYR <= 19) & d.cat.isin([1, 2, 3])
    st = d[["cens_" + c for c in A.ALL20]].values.astype(int)
    d["atopy"] = (st[:, :19] == 0).any(1)
    d["food"] = (st[:, [A.ALL20.index(c) for c in A.FOODS]] == 0).any(1)
    fl = A.flip_classify(d, {"atopy": (A.SIGE, kids.values), "food": (A.FOODS, kids.values)}, seed=20261101)
    k = d[kids]
    w = k.WTMEC2YR.values
    for out in ("atopy", "food"):
        cls = fl[out]["cls"][kids.values]
        tests = fl[out]["tests"][kids.values]
        rf = age_adjusted_or(k[out], k.cat.values, k.RIDAGEYR.values, w)
        rq = age_adjusted_or(pd.Series(cls, index=k.index), k.cat.values, k.RIDAGEYR.values, w)
        for t in ("over", "obese"):
            rows.append(dict(study="Visness 2009", quantity=f"age-adjusted OR, {out}, {t} vs normal, 6-19 y",
                             published=np.nan, reproduced_full_panel=rf[t], flip=rq[t],
                             flip_tests=A.wmean(tests, w), design_tests=19.0 if out == "atopy" else 4.0,
                             n=len(k), note="analysis cohort 6-19 y; FLIP tests incl. total IgE"))
    outdf = pd.DataFrame(rows)
    outdf.to_csv(os.path.join(A.RES, "lit_assoc_visness.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(outdf.round(3).to_string(index=False))


if __name__ == "__main__":
    if not os.path.exists(LMS):
        sys.exit(f"CDC reference file not found: {LMS}")
    main()
