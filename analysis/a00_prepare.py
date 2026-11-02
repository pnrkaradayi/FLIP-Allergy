"""
a00_prepare.py
==============
Rebuild the three analysis datasets from the raw public files, so that every
number in the manuscript is produced by code run end to end.

D1  NHANES 2005-2006 serum IgE: 19 allergen-specific IgE + total IgE, with the
    detection-limit flags, covariates and self-reported outcomes.
D2  NHANES III 1988-1994 skin prick test, 10 allergens (from the raw exam.dat
    extract); outcomes are merged from the bundled cohort file because the
    adult/youth questionnaire files are not stored locally.
D3  NHANES 2005-2006 household dust allergens on the D1 participants.

Run:  python a00_prepare.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data_raw")
OUT = os.path.join(HERE, "data")
SEED = 20261025

# Verified against doc_nhanes/AL_IGE_D.htm
ALLERGEN = {
    "LBXID2": "D. farinae", "LBXID1": "D. pteronyssinus", "LBXIE1": "Cat",
    "LBXIE5": "Dog", "LBXII6": "Cockroach", "LBXIM6": "Alternaria",
    "LBXF13": "Peanut", "LBXIF1": "Egg", "LBXIF2": "Milk", "LBXIW1": "Ragweed",
    "LBXIG5": "Rye grass", "LBXIG2": "Bermuda grass", "LBXIT7": "Oak",
    "LBXIT3": "Birch", "LBXF24": "Shrimp", "LBXIM3": "Aspergillus",
    "LBXW11": "Russian thistle", "LBXE72": "Mouse", "LBXE74": "Rat",
}
SIGE = list(ALLERGEN)
ALL20 = SIGE + ["LBXIGE"]
LLOD = {**{c: 0.35 for c in SIGE}, "LBXIGE": 2.0}
FLOOR = {**{c: 0.25 for c in SIGE}, "LBXIGE": 1.41}   # NHANES fill value LLOD/sqrt(2)

SPT = ["MIT", "CAT", "COC", "ALT", "RYE", "BER", "RAG", "THI", "WHI", "PEA"]
DUST = {"DSXAA1": "Alt a 1", "DSXAF1": "Asp f 1", "DSXBG1": "Bla g 1",
        "DSXBG2": "Bla g 2", "DSXCF1": "Can f 1", "DSXDF1": "Der f 1",
        "DSXDP1": "Der p 1", "DSXFD1": "Fel d 1", "DSXMM1": "Mus m 1",
        "DSXRN1": "Rat n 1", "DSXENX": "Endotoxin"}


def read(name):
    return pd.read_sas(os.path.join(RAW, f"{name}.xpt"), format="xport")


def build_d1():
    ig, demo = read("AL_IGE_D"), read("DEMO_D")
    mcq, agq, rdq, bmx = read("MCQ_D"), read("AGQ_D"), read("RDQ_D"), read("BMX_D")
    print(f"AL_IGE_D rows: {len(ig):,}")
    keep = ig[SIGE].notna().all(axis=1) & ig["LBXIGE"].notna()
    print(f"rows with all 19 sIgE measured: {int(ig[SIGE].notna().all(axis=1).sum()):,}; "
          f"also with total IgE: {int(keep.sum()):,}")
    d = ig.loc[keep].copy()
    for c in ALL20:
        lc = "LBDIGELC" if c == "LBXIGE" else f"LBD{c[3:]}LC"
        d["cens_" + c] = (d[lc] == 1).astype(int)
        d["lg_" + c] = np.log10(np.where(d["cens_" + c] == 1, FLOOR[c], d[c]))
    d = (d.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
                       "SDMVPSU", "SDMVSTRA", "WTMEC2YR"]], on="SEQN", how="left")
          .merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
          .merge(mcq[["SEQN", "MCQ010", "MCQ035"]], on="SEQN", how="left")
          .merge(agq[["SEQN", "AGQ010", "AGQ030", "AGQ040", "AGQ060", "AGQ130", "AGQ140"]],
                 on="SEQN", how="left")
          .merge(rdq[["SEQN", "RDQ070"]], on="SEQN", how="left"))
    yes = lambda s: s == 1
    d["y_current_allergy"] = (yes(d.AGQ040) & yes(d.AGQ060)).astype(int)
    d["y_current_asthma"] = (yes(d.MCQ010) & yes(d.MCQ035)).astype(int)
    d["y_wheeze12"] = yes(d.RDQ070).astype(int)
    d["y_hayfever12"] = (yes(d.AGQ010) & yes(d.AGQ030)).astype(int)
    d["y_eczema12"] = (yes(d.AGQ130) & yes(d.AGQ140)).astype(int)
    d["y_allergic_disease"] = ((d.y_current_allergy == 1) | (d.y_current_asthma == 1)
                               | (d.y_hayfever12 == 1)).astype(int)
    d["y_harmonised"] = ((d.y_current_asthma == 1) | (d.y_wheeze12 == 1)).astype(int)
    d["sensitized"] = (d[["cens_" + c for c in SIGE]] == 0).any(axis=1).astype(int)
    d["n_positive"] = (d[["cens_" + c for c in SIGE]] == 0).sum(axis=1)
    d["male"] = (d.RIAGENDR == 1).astype(int)

    rng = np.random.default_rng(SEED)
    u = rng.random(len(d))
    d["split"] = np.where(u < 0.50, "train", np.where(u < 0.75, "calib", "test"))

    print(f"cohort n = {len(d):,}; age {d.RIDAGEYR.min():.0f}-{d.RIDAGEYR.max():.0f} "
          f"(median {d.RIDAGEYR.median():.0f}); female {1 - d.male.mean():.3f}")
    cf = d[["cens_" + c for c in SIGE]].values.mean()
    print(f"censored allergen cells: {cf:.3f}; total IgE below LLOD: {d.cens_LBXIGE.mean():.3f}")
    print(f"sensitized to >=1 of 19: {d.sensitized.mean():.3f}")
    for y in ["y_allergic_disease", "y_current_allergy", "y_current_asthma",
              "y_wheeze12", "y_hayfever12", "y_eczema12", "y_harmonised"]:
        print(f"    {y:20s} n={int(d[y].sum()):5d}  prev={d[y].mean():.3f}")
    print("split:", d.split.value_counts().to_dict())
    cols = (["SEQN", "split"] + ["lg_" + c for c in ALL20] + ["cens_" + c for c in ALL20]
            + ["RIDAGEYR", "RIAGENDR", "male", "RIDRETH1", "INDFMPIR", "BMXBMI",
               "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "sensitized", "n_positive",
               "y_allergic_disease", "y_current_allergy", "y_current_asthma",
               "y_wheeze12", "y_hayfever12", "y_eczema12", "y_harmonised"])
    d[cols].to_csv(os.path.join(OUT, "d1_nhanes0506.csv"), index=False)
    return d[cols]


def build_d2():
    ex = pd.read_csv(os.path.join(RAW, "nhanes3", "nhanes3_allergy_raw.csv"))
    print(f"\nNHANES III examinee records: {len(ex):,}")
    # 888 is the NHANES III reserve code for a test that was not done (blank);
    # it must be treated as missing, not as a wheal of 888 mm.
    for k in SPT + ["NEG", "PSC"]:
        for s in ("WL", "WW"):
            ex.loc[ex[f"ALP{k}{s}"] >= 888, f"ALP{k}{s}"] = np.nan
    for k in SPT + ["NEG", "PSC"]:
        ex[f"mw_{k}"] = (ex[f"ALP{k}WL"] + ex[f"ALP{k}WW"]) / 2.0
    for k in SPT:
        ex[f"net_{k}"] = ex[f"mw_{k}"] - ex["mw_NEG"]
        ex[f"pos_{k}"] = (ex[f"net_{k}"] >= 3).astype(int)
    complete = ex[[f"mw_{k}" for k in SPT + ["NEG"]]].notna().all(axis=1)
    d = ex[complete & (ex.WTPFALG6.fillna(0) > 0)].copy()
    print(f"complete 10-allergen panel + negative control, allergy weight > 0: {len(d):,}")
    d["male"] = (d.HSSEX == 1).astype(int)
    d["sensitized"] = d[[f"pos_{k}" for k in SPT]].any(axis=1).astype(int)
    bundled = pd.read_csv(os.path.join(HERE, "..", "results_so_far", "nhanes3_flip_cohort.csv"),
                          usecols=["SEQN", "y_current_asthma", "y_wheeze12", "y_hayfever",
                                   "y_harmonised"])
    d = d.merge(bundled, on="SEQN", how="left")
    rng = np.random.default_rng(SEED + 3)
    u = rng.random(len(d))
    d["split"] = np.where(u < 0.50, "train", np.where(u < 0.75, "calib", "test"))
    print(f"age {d.HSAGEIR.min():.0f}-{d.HSAGEIR.max():.0f}; female {1 - d.male.mean():.3f}; "
          f"sensitized (>=1 of 10, net wheal >= 3 mm) {d.sensitized.mean():.3f}")
    for k in SPT:
        print(f"    {k}: {d['pos_' + k].mean():.3f}")
    print("split:", d.split.value_counts().to_dict())
    cols = (["SEQN", "split", "HSAGEIR", "HSSEX", "male", "DMARETHN", "DMPPIR", "WTPFALG6"]
            + [f"mw_{k}" for k in SPT + ["NEG", "PSC"]] + [f"net_{k}" for k in SPT]
            + [f"pos_{k}" for k in SPT] + ["sensitized", "y_current_asthma", "y_wheeze12",
                                            "y_hayfever", "y_harmonised"])
    d[cols].to_csv(os.path.join(OUT, "d2_nhanes3_spt.csv"), index=False)
    return d[cols]


def build_d3(d1):
    du = read("ALDUST_D")
    keep = ["SEQN"] + list(DUST)
    lc = [c for c in du.columns if c.startswith("DSD") and c.endswith("LC")]
    du = du[keep + lc].copy()
    for c in DUST:
        du["lg_" + c] = np.log10(du[c].clip(lower=1e-6))
    m = d1[["SEQN"]].merge(du, on="SEQN", how="left")
    n_all = int(m[list(DUST)].notna().all(axis=1).sum())
    print(f"\ndust: D1 participants with all 11 dust analytes: {n_all:,}")
    m.to_csv(os.path.join(OUT, "d3_dust.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d1 = build_d1()
    build_d2()
    build_d3(d1)
