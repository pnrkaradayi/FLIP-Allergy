"""
01b_build_nhanes3.py
====================
Build the NHANES III (1988-1994) skin-prick-test cohort - dataset 2.

Why this cohort. It is the external validation that matters most: a
DIFFERENT measurement modality (skin prick wheal in mm, not serum IgE in
kU/L), a different decade, and a different age range - yet all ten of its
allergens are also in the NHANES 2005-2006 serum panel. If the
cross-reactivity conditional covariance that FLIP relies on is a real
immunological structure rather than an assay artefact, it must show up in
both.

Outputs (read by ../analysis/a00_prepare.py):
../results_so_far/nhanes3_flip_cohort.csv (11,932 rows x 45 columns), and a
compact extract of the allergy columns at
../data_raw/nhanes3/nhanes3_allergy_raw.csv (31,311 rows, all examinees).

Verified numbers:
    examinee records parsed from exam.dat          31,311
    with a positive allergy-subsample weight       12,106
    with ALL 10 allergens + negative control       11,932   <- the cohort
    age range                                      6 - 59 y
    y_current_asthma    n =   640   prevalence     0.054
    y_wheeze12          n = 1,705   prevalence     0.143
    y_hayfever          n =   525   prevalence     0.044   (adults only)
    y_harmonised        n = 1,889   prevalence     0.158
  Sensitization (net wheal >= 3 mm over negative control):
    cockroach 0.265, mite 0.243, rye grass 0.243, ragweed 0.237,
    Bermuda grass 0.169, cat 0.135, thistle 0.133, Alternaria 0.125,
    white oak 0.103, peanut 0.076

The harmonised outcome (current asthma OR wheeze in the past 12 months) is
defined identically in both cohorts and the prevalences are close -
0.158 here versus 0.165 in NHANES 2005-2006 - which is what makes a
transported decision threshold meaningful.

Note on raw files: exam.dat is 190 MB, adult.dat 65 MB, youth.dat 28 MB.
They are NOT bundled. This script downloads them on demand; only the small
extracts are kept.

Run:  python 01b_build_nhanes3.py
"""
import os
import re
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data_raw", "nhanes3")
OUT = os.path.join(HERE, "..", "results_so_far")
BASE = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a"

# Skin-test allergens. All ten also appear in the NHANES 2005-2006 serum
# panel, which is what makes the cross-modality comparison possible.
ALLERGEN = {
    "MIT": ("Mite", ["LBXID2", "LBXID1"]),       # serum has D. farinae and D. pteronyssinus
    "CAT": ("Cat", ["LBXIE1"]),
    "COC": ("German cockroach", ["LBXII6"]),
    "ALT": ("Alternaria", ["LBXIM6"]),
    "RYE": ("Rye grass", ["LBXIG5"]),
    "BER": ("Bermuda grass", ["LBXIG2"]),
    "RAG": ("Ragweed", ["LBXIW1"]),
    "THI": ("Russian thistle", ["LBXW11"]),
    "WHI": ("White oak", ["LBXIT7"]),
    "PEA": ("Peanut", ["LBXF13"]),
}
DEMO = ["SEQN", "HSAGEIR", "HSSEX", "DMARETHN", "DMPPIR", "SDPPHASE",
        "SDPPSU6", "SDPSTRA6", "WTPFALG6"]


def parse_layout(sas_text):
    """Column positions from the SAS INPUT statement of an NHANES III layout."""
    up = sas_text.upper()
    st = up.find("\n     INPUT")
    if st < 0:
        st = up.find("INPUT\r\n")
    blk = sas_text[st:]
    blk = blk[:blk.find(";")]
    out = {}
    for m in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(\d+)(?:-(\d+))?\s*$", blk, re.M):
        a = int(m.group(2))
        out[m.group(1)] = (a, int(m.group(3)) if m.group(3) else a)
    return out


def fetch(name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if not os.path.exists(path) or os.path.getsize(path) < 50_000:
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return path


def extract(dat_name, sas_name, cols):
    sas = open(fetch(sas_name), encoding="latin-1").read()
    layout = parse_layout(sas)
    spec = {c: layout[c] for c in cols if c in layout}
    missing = [c for c in cols if c not in layout]
    if missing:
        print(f"  WARNING: not in {sas_name}: {missing}")
    rows = []
    with open(fetch(dat_name), encoding="latin-1") as fh:
        for line in fh:
            if len(line.rstrip("\r\n")) < 50:
                continue
            rows.append([line[a - 1:b].strip() for a, b in spec.values()])
    return pd.DataFrame(rows, columns=list(spec)).apply(pd.to_numeric, errors="coerce")


def main():
    os.makedirs(OUT, exist_ok=True)
    skin = [f"ALP{k}{s}" for k in list(ALLERGEN) + ["NEG", "PSC"] for s in ("WL", "WW")]
    ex = extract("exam.dat", "exam.sas", DEMO + skin + ["ALPQ4"])
    print(f"examinee records: {len(ex):,}")
    ex.to_csv(os.path.join(RAW, "nhanes3_allergy_raw.csv"), index=False)

    ad = extract("adult.dat", "adult.sas",
                 ["SEQN", "HAC1E", "HAC2E", "HAC1H", "HAC2H", "HAL6"])
    yo = extract("youth.dat", "youth.sas", ["SEQN", "HYE1G", "HYE3G", "HYG8"])

    # mean wheal diameter, then net of the negative control
    for k in list(ALLERGEN) + ["NEG", "PSC"]:
        ex[f"mw_{k}"] = (ex[f"ALP{k}WL"] + ex[f"ALP{k}WW"]) / 2.0
    for k in ALLERGEN:
        ex[f"net_{k}"] = ex[f"mw_{k}"] - ex["mw_NEG"]
        ex[f"pos_{k}"] = (ex[f"net_{k}"] >= 3).astype(float)   # standard SPT criterion

    complete = ex[[f"mw_{k}" for k in list(ALLERGEN) + ["NEG"]]].notna().all(axis=1)
    coh = ex[complete & (ex.WTPFALG6.fillna(0) > 0)].copy()
    print(f"complete 10-allergen panel with allergy weight: {len(coh):,}")

    coh = coh.merge(ad, on="SEQN", how="left").merge(yo, on="SEQN", how="left")
    cur_asthma = (((coh.HAC1E == 1) & (coh.HAC2E == 1))
                  | ((coh.HYE1G == 1) & (coh.HYE3G == 1))).astype(int)
    wheeze = ((coh.HAL6 == 1) | (coh.HYG8 == 1)).astype(int)
    coh["y_current_asthma"] = cur_asthma
    coh["y_wheeze12"] = wheeze
    coh["y_hayfever"] = ((coh.HAC1H == 1) & (coh.HAC2H == 1)).astype(int)   # adults only
    coh["y_harmonised"] = ((cur_asthma == 1) | (wheeze == 1)).astype(int)

    print(f"age range: {int(coh.HSAGEIR.min())} - {int(coh.HSAGEIR.max())}")
    for y in ["y_current_asthma", "y_wheeze12", "y_hayfever", "y_harmonised"]:
        print(f"    {y:18s} n={int(coh[y].sum()):5d}  prev={coh[y].mean():.3f}")
    print("\nsensitization prevalence (net wheal >= 3 mm):")
    for k, (nm, _) in ALLERGEN.items():
        print(f"    {nm:18s} {coh['pos_' + k].mean():.3f}")

    keep = (DEMO + [f"mw_{k}" for k in list(ALLERGEN) + ["NEG", "PSC"]]
            + [f"net_{k}" for k in ALLERGEN] + [f"pos_{k}" for k in ALLERGEN]
            + ["y_current_asthma", "y_wheeze12", "y_hayfever", "y_harmonised"])
    coh[keep].to_csv(os.path.join(OUT, "nhanes3_flip_cohort.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT, 'nhanes3_flip_cohort.csv')}  {coh[keep].shape}")


if __name__ == "__main__":
    main()
