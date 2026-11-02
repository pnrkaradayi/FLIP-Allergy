"""Headline numbers of further allergy studies, reproduced from their patient-level data.

Bergmann-Hug et al. 2020 (PLoS One 15:e0238740): venom sIgE in pollen-allergic patients and controls.
van der Valk et al. 2016 (PLoS One 11:e0151055): IDEAL cashew nut DBPCFC (transcribed S1 Database).
McGowan et al. 2015 (J Allergy Clin Immunol 135:171): URECA, food sensitisation by age 5 years.
Perkin et al. 2016 (N Engl J Med 374:1733): EAT trial, food allergy at 1-3 years.
Du Toit et al. 2015 (N Engl J Med 372:803): LEAP trial, peanut allergy at 60 months.
Writes results/ext_reproduction_added.csv (study, quantity, published, reproduced, n).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext_data  # noqa: E402

RES = os.path.join(HERE, "results")


def main():
    rows = []
    v = ext_data.venom_pollen()
    X, c = v["X"], v["extra"]["controls"]
    rows.append(dict(study="Bergmann-Hug 2020", quantity="venom sIgE >= 0.35 kU/L, pollen-allergic patients, %",
                     published=45.7, reproduced=100 * ((X.bee >= 0.35) | (X.wasp >= 0.35)).mean(), n=len(X)))
    rows.append(dict(study="Bergmann-Hug 2020", quantity="venom sIgE >= 0.35 kU/L, non-atopic controls, %",
                     published=26.7, reproduced=100 * ((c.bee >= 0.35) | (c.wasp >= 0.35)).mean(), n=len(c)))
    rows.append(dict(study="Bergmann-Hug 2020", quantity="sIgE to CCD (MUXF3) >= 0.35 kU/L, %",
                     published=20.0, reproduced=100 * (X.MUXF3 >= 0.35).mean(), n=len(X)))
    d = pd.read_csv(os.path.join(ext_data.EXA, "cashew_ideal", "IDEAL_transcribed.csv"), sep=";", decimal=",",
                    na_values=["."])
    pos = d.Outcome_challenge == "Positive"
    rows.append(dict(study="van der Valk 2016", quantity="DBPCFC positive, %", published=76.5,
                     reproduced=100 * pos.mean(), n=len(d)))
    rows.append(dict(study="van der Valk 2016", quantity="anaphylaxis among positive DBPCFC, %",
                     published=36.0, reproduced=100 * (d.Anaphylaxis[pos] == "Anaphylaxis").mean(), n=int(pos.sum())))

    w = ext_data.ureca_records()                     # McGowan 2015: milk, egg, peanut at 1, 2, 3 and 5 years
    f = w[w.month.isin([12, 24, 36, 60])]
    kids = f[f[["Milk", "Egg", "Peanut"]].notna().any(axis=1)].subject_accession.unique()
    f = f[f.subject_accession.isin(kids)]
    ever = (f[["Milk", "Egg", "Peanut"]] >= 0.35).groupby(f.subject_accession).any()
    rows.append(dict(study="McGowan 2015", quantity="children with food sIgE through 5 years, n", published=516,
                     reproduced=len(ever), n=len(ever)))
    rows.append(dict(study="McGowan 2015", quantity="food-sensitised by 5 years, %", published=55.4,
                     reproduced=100 * ever.any(axis=1).mean(), n=len(ever)))
    for food, pub in (("Milk", 46.7), ("Egg", 31.0), ("Peanut", 20.9)):
        rows.append(dict(study="McGowan 2015", quantity=f"{food.lower()}-sensitised by 5 years, %", published=pub,
                         reproduced=100 * ever[food].mean(), n=len(ever)))
    e = ext_data.eat_records()                       # Perkin 2016: food allergy to >= 1 of 6 foods, 1-3 years
    fa = e["primary outcome positive (only those evaluable and within age range)"]
    for grp, lab, pub in ((0, "standard introduction", 7.1), (1, "early introduction", 5.6)):
        g = fa[(e["study group"] == grp) & fa.notna()]
        rows.append(dict(study="Perkin 2016", quantity=f"food allergy, {lab}, %", published=pub,
                         reproduced=100 * g.mean(), n=len(g)))
    po = pd.read_csv(os.path.join(ext_data.EXA, "leap", "unzipped", "StudyFiles", "ADPOUT1_2015-03-03_14-19-46.txt"),
                     sep="\t", low_memory=False)               # Du Toit 2015: peanut allergy at 60 months, ITT
    po = po[(po["Intent-to-treat (ITT) Sample"] == "Yes") & po["Overall V60 Outcome (OFC + Indeterm.)"].notna()]
    for stratum, arm, pub in (("SPT-negative", "Avoidance", 13.7), ("SPT-negative", "Consumption", 1.9),
                              ("SPT-positive", "Avoidance", 35.3), ("SPT-positive", "Consumption", 10.6)):
        g = po[(po["Stratum (char)"] == f"{stratum} Stratum") & (po["Treatment Group (Char)"] == f"Peanut {arm}")]
        rows.append(dict(study="Du Toit 2015", quantity=f"peanut allergy at 60 months, {stratum}, {arm.lower()}, %",
                         published=pub, reproduced=100 * (g["Overall V60 Outcome (OFC + Indeterm.)"] == "FAIL OFC").mean(),
                         n=len(g)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ext_reproduction_added.csv"), index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
