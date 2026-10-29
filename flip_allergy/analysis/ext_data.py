"""
ext_data.py
===========
Loaders for the public datasets of the published studies used in the external
comparison. Every loader returns a dict with

  X        DataFrame of raw values (NaN = not measured / not ordered)
  kind     {column: "cont" | "bin" | "cens"}  ("cens": assay value censored
           below the detection limit lod[column], modelled on the log10 scale)
  lod      {column: detection limit}          (cens columns only)
  y        0/1 outcome (or None when the dataset has no clinical outcome)
  blocks   list of (block name, [columns]) = tests that may be left unordered
  always   columns available for every patient (history, demographics)
  cost     {block name: cost} when the study published test costs
  extra    dataset-specific items (site labels, panels, ...)

Raw files live in ../data_raw/external and ../data_raw/external_allergy; they
are read as data only.
"""
import csv
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "data_raw", "external")
EXA = os.path.join(HERE, "..", "data_raw", "external_allergy")


def read_arff(path):
    names, rows, data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not data:
                if s.lower().startswith("@attribute"):
                    parts = s.split(None, 2)
                    names.append(parts[1].strip("'\""))
                elif s.lower().startswith("@data"):
                    data = True
                continue
            rows.append(next(csv.reader([s], quotechar="'", skipinitialspace=True)))
    d = pd.DataFrame(rows, columns=names).replace({"?": np.nan})
    for c in d.columns:
        conv = pd.to_numeric(d[c], errors="coerce")
        if conv.notna().sum() >= 0.9 * d[c].notna().sum() and d[c].notna().any():
            d[c] = conv
    return d


def _num(s):
    s = s.astype(str).str.strip().replace({"not done": np.nan, "n.a.": np.nan, "nan": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


# ============================================================ clinical datasets
def heart():
    """UCI Heart Disease, four sites (Detrano et al. 1989)."""
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal", "num"]
    frames = []
    for site, f in (("Cleveland", "cleveland"), ("Hungary", "hungarian"), ("Switzerland", "switzerland"),
                    ("Long Beach VA", "va")):
        d = pd.read_csv(os.path.join(EXT, "heart", f"processed.{f}.data"), header=None, names=cols,
                        na_values="?")
        d["site"] = site
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d.loc[d.chol == 0, "chol"] = np.nan              # 0 = not measured (Switzerland)
    d.loc[d.trestbps == 0, "trestbps"] = np.nan
    X = pd.DataFrame({"age": d.age, "sex": d.sex,
                      "cp_atypical": (d.cp == 2).astype(float), "cp_nonanginal": (d.cp == 3).astype(float),
                      "cp_asymptomatic": (d.cp == 4).astype(float),
                      "trestbps": d.trestbps, "chol": d.chol, "fbs": d.fbs,
                      "restecg_abn": np.where(d.restecg.isna(), np.nan, (d.restecg > 0).astype(float)),
                      "thalach": d.thalach, "exang": d.exang, "oldpeak": d.oldpeak,
                      "slope_flat_down": np.where(d.slope.isna(), np.nan, (d.slope >= 2).astype(float)),
                      "ca": d.ca,
                      "thal_fixed": np.where(d.thal.isna(), np.nan, (d.thal == 6).astype(float)),
                      "thal_reversible": np.where(d.thal.isna(), np.nan, (d.thal == 7).astype(float))})
    kind = {c: "cont" for c in ("age", "trestbps", "chol", "thalach", "oldpeak", "ca")}
    kind.update({c: "bin" for c in X.columns if c not in kind})
    blocks = [("blood pressure", ["trestbps"]), ("blood tests", ["chol", "fbs"]),
              ("resting ECG", ["restecg_abn"]),
              ("exercise test", ["thalach", "exang", "oldpeak", "slope_flat_down"]),
              ("fluoroscopy", ["ca"]), ("thallium scan", ["thal_fixed", "thal_reversible"])]
    cost = {"blood pressure": 1.0, "blood tests": 7.27 + 5.20, "resting ECG": 15.50,
            "exercise test": 87.30 + 102.90, "fluoroscopy": 100.90, "thallium scan": 102.90}
    return dict(name="Heart disease (4 sites)", X=X, kind=kind, lod={}, y=(d.num > 0).astype(int).values,
                blocks=blocks, always=["age", "sex", "cp_atypical", "cp_nonanginal", "cp_asymptomatic"],
                cost=cost, extra=dict(site=d.site.values))


def thyroid():
    """UCI thyroid (Garavan), allhypo: hypothyroidism (Quinlan 1987/1996; Turney 1995)."""
    names = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid", "sick", "pregnant",
             "thyroid_surgery", "I131", "query_hypo", "query_hyper", "lithium", "goitre", "tumor",
             "hypopituitary", "psych", "TSH_m", "TSH", "T3_m", "T3", "TT4_m", "TT4", "T4U_m", "T4U",
             "FTI_m", "FTI", "TBG_m", "TBG", "referral", "cls"]
    rows = []
    for f in ("allhypo.data", "allhypo.test"):
        with open(os.path.join(EXT, "thyroid", f)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(line.split("|")[0].rstrip(".").split(","))
    d = pd.DataFrame(rows, columns=names).replace("?", np.nan)
    hist = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid", "sick", "pregnant", "thyroid_surgery",
            "I131", "query_hypo", "query_hyper", "lithium", "goitre", "tumor", "hypopituitary", "psych"]
    X = pd.DataFrame({"age": pd.to_numeric(d.age), "male": (d.sex == "M").astype(float)})
    X.loc[d.sex.isna(), "male"] = np.nan
    X.loc[X.age > 110, "age"] = np.nan
    for h in hist:
        X[h] = (d[h] == "t").astype(float)
    for t in ("TSH", "T3", "TT4", "T4U"):
        X[t] = pd.to_numeric(d[t])
    y = (~d.cls.str.startswith("negative")).astype(int).values
    kind = {"age": "cont", "male": "bin", **{h: "bin" for h in hist},
            **{t: "cont" for t in ("TSH", "T3", "TT4", "T4U")}}
    blocks = [(t, [t]) for t in ("TSH", "T3", "TT4", "T4U")]
    cost = {"TSH": 22.78, "T3": 11.41, "TT4": 14.51, "T4U": 11.41}
    return dict(name="Thyroid (hypothyroidism)", X=X, kind=kind, lod={}, y=y, blocks=blocks,
                always=["age", "male"] + hist, cost=cost, extra=dict(cls=d.cls.values))


def hepatitis():
    """UCI hepatitis (Gong; Diaconis and Efron 1983)."""
    names = ["cls", "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver_big",
             "liver_firm", "spleen", "spiders", "ascites", "varices", "bilirubin", "alk_phosphate", "sgot",
             "albumin", "protime", "histology"]
    d = pd.read_csv(os.path.join(EXT, "hepatitis", "hepatitis.data"), header=None, names=names, na_values="?")
    binv = ["sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver_big", "liver_firm",
            "spleen", "spiders", "ascites", "varices", "histology"]
    X = pd.DataFrame({"age": d.age})
    for b in binv:
        X[b] = np.where(d[b].isna(), np.nan, (d[b] == 2).astype(float))
    for c in ("bilirubin", "alk_phosphate", "sgot", "albumin", "protime"):
        X[c] = d[c]
    kind = {"age": "cont", **{b: "bin" for b in binv},
            **{c: "cont" for c in ("bilirubin", "alk_phosphate", "sgot", "albumin", "protime")}}
    blocks = [("liver panel", ["bilirubin", "alk_phosphate", "sgot", "albumin"]), ("prothrombin time", ["protime"]),
              ("histology", ["histology"])]
    cost = {"liver panel": 4 * 7.27, "prothrombin time": 8.30, "histology": 1.0}
    always = ["age"] + [b for b in binv if b != "histology"]
    return dict(name="Hepatitis", X=X, kind=kind, lod={}, y=(d.cls == 1).astype(int).values, blocks=blocks,
                always=always, cost=cost, extra={})


def pima():
    """Pima Indians diabetes (Smith et al. 1988); zeros are missing measurements."""
    d = read_arff(os.path.join(EXT, "pima", "diabetes.arff"))
    for c in ("plas", "pres", "skin", "insu", "mass"):
        d.loc[d[c] == 0, c] = np.nan
    X = d[["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]].astype(float)
    kind = {c: "cont" for c in X.columns}
    blocks = [("glucose tolerance test", ["plas"]), ("serum insulin", ["insu"]), ("skinfold", ["skin"]),
              ("blood pressure", ["pres"])]
    cost = {"glucose tolerance test": 17.61, "serum insulin": 22.78, "skinfold": 1.0, "blood pressure": 1.0}
    return dict(name="Pima diabetes", X=X, kind=kind, lod={}, y=(d["class"] == "tested_positive").astype(int).values,
                blocks=blocks, always=["preg", "mass", "pedi", "age"], cost=cost, extra={})


def ckd():
    """UCI chronic kidney disease (Qin et al. 2020)."""
    d = read_arff(os.path.join(EXT, "ckd", "chronic-kidney-disease.arff"))
    yes = lambda s, v: np.where(d[s].isna(), np.nan, (d[s].astype(str).str.strip() == v).astype(float))
    X = pd.DataFrame({"age": d.age, "bp": d.bp, "sg": d.sg, "al": d.al, "su": d.su,
                      "rbc_abn": yes("rbc", "abnormal"), "pc_abn": yes("pc", "abnormal"),
                      "pcc": yes("pcc", "present"), "ba": yes("ba", "present"),
                      "bgr": d.bgr, "bu": d.bu, "sc": d.sc, "sod": d.sod, "pot": d.pot, "hemo": d.hemo,
                      "pcv": pd.to_numeric(d.pcv, errors="coerce"), "wc": pd.to_numeric(d.wc, errors="coerce"),
                      "rc": pd.to_numeric(d.rc, errors="coerce"),
                      "htn": yes("htn", "yes"), "dm": yes("dm", "yes"), "cad": yes("cad", "yes"),
                      "appet_poor": yes("appet", "poor"), "pe": yes("pe", "yes"), "ane": yes("ane", "yes")})
    kind = {c: ("bin" if c in ("rbc_abn", "pc_abn", "pcc", "ba", "htn", "dm", "cad", "appet_poor", "pe", "ane")
                else "cont") for c in X.columns}
    blocks = [("urinalysis", ["sg", "al", "su", "rbc_abn", "pc_abn", "pcc", "ba"]),
              ("renal chemistry", ["bu", "sc", "sod", "pot"]), ("blood glucose", ["bgr"]),
              ("blood count", ["hemo", "pcv", "wc", "rc"])]
    y = (d.classification.astype(str).str.strip().str.startswith("ckd")
         & ~d.classification.astype(str).str.contains("notckd")).astype(int).values
    return dict(name="Chronic kidney disease", X=X, kind=kind, lod={}, y=y, blocks=blocks,
                always=["age", "bp", "htn", "dm", "cad", "appet_poor", "pe", "ane"], cost={}, extra={})


def support2():
    """SUPPORT (Knaus et al. 1995): 180-day mortality from day-3 physiology."""
    d = read_arff(os.path.join(EXT, "support2", "Support2.arff"))
    y = ((d.death == 1) & (d["d.time"] <= 180)).astype(int).values
    X = pd.DataFrame({"age": d.age, "male": (d.sex == "male").astype(float), "num_co": d["num.co"],
                      "scoma": d.scoma, "diabetes": d.diabetes.astype(float), "dementia": d.dementia.astype(float),
                      "ca_metastatic": (d.ca == "metastatic").astype(float), "ca_yes": (d.ca == "yes").astype(float),
                      "meanbp": d.meanbp, "hrt": d.hrt, "resp": d.resp, "temp": d.temp})
    for g in ("CHF", "COPD", "Lung Cancer", "MOSF w/Malig", "Coma", "Cirrhosis", "Colon Cancer"):  # ref: ARF/MOSF
        X["dz_" + g.split()[0].split("/")[0].lower()] = (d.dzgroup == g).astype(float)
    # bun, glucose and urine output are missing in about half of the records and are
    # left out of the model (as in Knaus et al., who replaced them by normal values)
    for c in ("wblc", "pafi", "ph", "alb", "bili", "crea", "sod"):
        X[c] = d[c].astype(float)
    kind = {c: "cont" for c in X.columns}
    for c in X.columns:
        if X[c].dropna().isin([0, 1]).all():
            kind[c] = "bin"
    blocks = [("blood count", ["wblc"]), ("arterial blood gas", ["pafi", "ph"]), ("albumin", ["alb"]),
              ("bilirubin", ["bili"]), ("creatinine and sodium", ["crea", "sod"])]
    always = [c for c in X.columns if not any(c in b for _, b in blocks)]
    return dict(name="SUPPORT (critically ill)", X=X, kind=kind, lod={}, y=y, blocks=blocks, always=always,
                cost={}, extra={})


PHYSIONET_VARS = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine", "DiasABP",
                  "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na",
                  "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
                  "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"]


def physionet2012():
    """PhysioNet Challenge 2012 set A (Silva et al. 2012; Che et al. 2018): in-hospital death.
    Each 48-hour record is summarised per variable (mean; min and max for the vital signs and the
    routine chemistry); a laboratory is missing when it was never measured."""
    cache = os.path.join(HERE, "data", "ext_physionet2012.csv")
    if os.path.exists(cache):
        d = pd.read_csv(cache)
    else:
        z = zipfile.ZipFile(os.path.join(EXT, "physionet2012", "set-a.zip"))
        recs = []
        for m in z.namelist():
            if not m.endswith(".txt"):
                continue
            t = pd.read_csv(io.StringIO(z.read(m).decode()), dtype={"Time": str})
            stat = t[t.Time == "00:00"].set_index("Parameter").Value
            ts = t[~t.Parameter.isin(["RecordID", "Age", "Gender", "Height", "ICUType"])]
            ts = ts[ts.Value >= 0]
            r = {"RecordID": int(stat.get("RecordID")), "Age": stat.get("Age"), "Gender": stat.get("Gender"),
                 "ICUType": stat.get("ICUType")}
            g = ts.groupby("Parameter").Value
            for v, s in g.mean().items():
                r[v + "_mean"] = s
            for v, s in g.min().items():
                r[v + "_min"] = s
            for v, s in g.max().items():
                r[v + "_max"] = s
            recs.append(r)
        d = pd.DataFrame(recs)
        o = pd.read_csv(os.path.join(EXT, "physionet2012", "Outcomes-a.txt"))
        d = d.merge(o[["RecordID", "In-hospital_death"]], on="RecordID")
        d.to_csv(cache, index=False)
    vit = ["HR", "Temp", "GCS", "Urine"]
    chem = ["BUN", "Creatinine", "Glucose", "HCO3", "HCT", "K", "Mg", "Na", "Platelets", "WBC"]
    X = pd.DataFrame({"age": d.Age, "male": (d.Gender == 1).astype(float),
                      "icu_ccu": (d.ICUType == 1).astype(float), "icu_csru": (d.ICUType == 2).astype(float),
                      "icu_sicu": (d.ICUType == 4).astype(float),
                      "mechvent": d.get("MechVent_max", pd.Series(np.nan, index=d.index)).fillna(0).astype(float)})
    for v in vit + chem:
        for s in ("mean", "min", "max"):
            X[f"{v}_{s}"] = d.get(f"{v}_{s}")
    # optional laboratories; albumin and troponin (measured in 40% and 22%) are left out
    labs = {"lactate": ["Lactate_mean"], "liver panel": ["ALT_mean", "AST_mean", "ALP_mean", "Bilirubin_mean"],
            "blood gas": ["pH_mean", "PaCO2_mean", "PaO2_mean"]}
    for cols in labs.values():
        for c in cols:
            X[c] = d.get(c)
    for v in ("ALT_mean", "AST_mean", "ALP_mean", "Bilirubin_mean", "Lactate_mean"):
        X[v] = np.log(X[v].clip(lower=0.01))
    kind = {c: ("bin" if c in ("male", "icu_ccu", "icu_csru", "icu_sicu", "mechvent") else "cont") for c in X.columns}
    blocks = list(labs.items())
    always = [c for c in X.columns if not any(c in b for _, b in blocks)]
    return dict(name="PhysioNet 2012 (ICU)", X=X, kind=kind, lod={}, y=d["In-hospital_death"].astype(int).values,
                blocks=blocks, always=always, cost={}, extra={})


# ============================================================ allergy datasets
FAMILY_RULES = {   # component -> protein family (molecular allergology)
    "PR-10": ["Bet_v_1", "Aln_g_1", "Cor_a_1.0101", "Cor_a_1.0401", "Mal_d_1", "Pru_p_1", "Gly_m_4", "Ara_h_8",
              "Act_d_8", "Api_g_1"],
    "nsLTP": ["Pru_p_3", "Cor_a_8", "Ara_h_9", "Jug_r_3", "Art_v_3", "Pla_a_3", "Par_j_2", "Ole_e_7", "Tri_a_14"],
    "profilin": ["Bet_v_2", "Phl_p_12", "Hev_b_8", "Mer_a_1", "Che_a_1"],
    "polcalcin": ["Bet_v_4", "Phl_p_7"],
    "tropomyosin": ["Der_p_10", "Bla_g_7", "Pen_m_1", "Ani_s_3"],
    "serum albumin": ["Bos_d_6", "Can_f_3", "Equ_c_3", "Fel_d_2"],
    "lipocalin": ["Can_f_1", "Can_f_2", "Equ_c_1", "Fel_d_4", "Mus_m_1", "Bos_d_5", "Can_f_5"],
    "2S albumin": ["Ara_h_2", "Ara_h_6", "Ana_o_3", "Ber_e_1", "Jug_r_1", "Ses_i_1", "Cor_a_14", "Fag_e_2"],
    "7S/11S storage": ["Ara_h_1", "Ara_h_3", "Gly_m_5", "Gly_m_6", "Jug_r_2", "Cor_a_9", "Ana_o_2"],
    "grass pollen": ["Phl_p_1", "Phl_p_2", "Phl_p_4", "Phl_p_5", "Phl_p_6", "Phl_p_11", "Cyn_d_1"],
    "mite groups 1/2": ["Der_f_1", "Der_f_2", "Der_p_1", "Der_p_2", "Lep_d_2", "Blo_t_5"],
    "cockroach": ["Bla_g_1", "Bla_g_2", "Bla_g_5", "Bla_g_7"],
    "cat": ["Fel_d_1", "Fel_d_2", "Fel_d_4"],
    "moulds": ["Alt_a_1", "Alt_a_6", "Asp_f_1", "Asp_f_3", "Asp_f_6", "Cla_h_8"],
    "weed/tree pollen major": ["Amb_a_1", "Art_v_1", "Sal_k_1", "Pla_l_1", "Che_a_1", "Ole_e_1", "Pla_a_1",
                               "Pla_a_2", "Cry_j_1", "Cup_a_1"],
    "egg": ["Gal_d_1", "Gal_d_2", "Gal_d_3", "Gal_d_5"],
    "milk": ["Bos_d_4", "Bos_d_5", "Bos_d_8", "Bos_d_Lactoferrin"],
    "venom": ["Api_m_1", "Api_m_4", "Pol_d_5", "Ves_v_5"],
    "latex": ["Hev_b_1", "Hev_b_3", "Hev_b_5", "Hev_b_6.01", "Hev_b_8"],
    "CCD": ["MUXF3"],
}


EXTRACT_FAMILIES = {   # allergen-source / cross-reactivity groups for extract panels
    "moscow": {"Fagales pollen and PR-10 foods": ["Alder", "Birch", "Hazelnut_p", "Oak", "Apple", "Peach", "Carrot",
                                                  "Hazelnut"],
               "grass pollen": ["Bermuda grass", "Orchard grass", "Meadow fescue", "Perennial rye grass",
                                "Timothy grass", "Cultivated rye", "Wheat flour"],
               "weed pollen": ["Wormwood", "Mugwort", "Dandelion"],
               "animal dander": ["Cat dander", "Dog dander"],
               "mites and cockroach": ["D. pteronyssinus", "D. farinae", "Cockroach, German"],
               "cow's milk": ["Cow milk", "α-lactalbumin, cow milk", "β-lactoglobulin, cow milk", "Casein, cow milk"],
               "other foods": ["Egg white", "Codfish", "Peanut", "Wheat flour", "Hazelnut"]},
    "guo": {"weed pollen": ["艾蒿w6", "法国菊w7", "蒲公英w8", "车前草w9"],
            "grass and tree pollen": ["梯牧草G6", "桦树T3"],
            "mites": ["屋尘螨D1", "粉尘螨D2"],
            "tropomyosin (mite, cockroach, crab)": ["屋尘螨D1", "粉尘螨D2", "德国蟑螂I6(2)", "蟹F23"],
            "animal dander": ["猫皮毛屑E1", "马皮毛屑E3", "狗皮毛屑E5"],
            "foods": ["蛋白F1", "牛奶F2", "鳕鱼F3", "芝麻F10", "巴西胡桃F18"]},
}


EXTRACT_FAMILIES["sosall"] = {
    "animal-derived foods": ["sIgE_milk", "sIgE_egg", "sIgE_fish"],
    "plant foods (storage proteins, PR-10, CCD)": ["sIgE_peanut", "sIgE_soy", "sIgE_hazelnut", "sIgE_wheat",
                                                   "sIgE_CCD"]}
EXTRACT_FAMILIES["graz"] = {
    "honeybee venom": ["bee", "Api_m_1", "Api_m_2", "Api_m_3", "Api_m_5", "Api_m_10", "bat_bee", "idt_bee_score"],
    "vespid venom": ["wasp", "Ves_v_1", "Ves_v_5", "bat_wasp", "idt_wasp_score"],
    "cross-reactive venom proteins (hyaluronidase, dipeptidyl peptidase)": ["bee", "wasp", "Api_m_2", "Api_m_5"]}
EXTRACT_FAMILIES["maas"] = None                   # component families of FAMILY_RULES

ALIASES = {"cora101": "cora10101", "cora104": "cora10401", "hevb601": "hevb601", "tria190101": "tria190101"}


def norm_name(s):
    n = "".join(ch for ch in str(s).lower() if ch.isalnum())
    return ALIASES.get(n, n)


def family_mask_for(cols, families=None):
    """Sparsity pattern of the cross-reactivity factor model: a global sensitisation factor plus
    one factor per protein family (or allergen-source group) with at least two members present."""
    fams = {"atopy (global)": list(cols)}
    nc = {norm_name(c): c for c in cols}
    for f, members in (families or FAMILY_RULES).items():
        m = list(dict.fromkeys(nc[norm_name(x)] for x in members if norm_name(x) in nc))
        if len(m) >= 2:
            fams[f] = m
    M = np.zeros((len(cols), len(fams)), bool)
    for k, (f, members) in enumerate(fams.items()):
        for c in members:
            M[list(cols).index(c), k] = True
    return M, list(fams)


def acc():
    """Allergen Chip Challenge (Martinroche et al. 2026), ISAC 112-component chips."""
    d = pd.read_csv(os.path.join(EXA, "acc", "allergenchipchallenge-data-corrected-final-hdh-sfa.csv"),
                    sep=";", decimal=",", low_memory=False, encoding="utf-8-sig")
    meta = list(d.columns[:15])
    comp = [c for c in d.columns[15:]]
    isac = d[d.Chip_Type.str.startswith("ISAC")].reset_index(drop=True)
    cols = [c for c in comp if isac[c].notna().mean() > 0.95]
    X = isac[cols].astype(float)
    kind = {c: "cens" for c in cols}
    lod = {c: 0.3 for c in cols}
    return dict(name="Allergen Chip Challenge (ISAC)", X=X, kind=kind, lod=lod, y=None,
                blocks=[(c, [c]) for c in cols], always=[], cost={},
                extra=dict(age=isac.Age.values, male=(isac.Gender == 1).astype(float).values,
                           sensitization=isac.Sensitization.values, chip=isac.Chip_Type.values,
                           all=d, components=cols, meta=meta))


def maas():
    """MAAS birth cohort, ISAC at age 11 (Fontanella et al. 2018)."""
    d = pd.read_csv(os.path.join(EXA, "maas", "sIgE_data_age_11.csv"))
    n_all = len(d)
    d = d[d.asthma.notna()].reset_index(drop=True)
    comp = [c for c in d.columns if c != "asthma"]
    # the 44 "active" components of the paper, as listed in the authors' filtered file
    active = [c for c in pd.read_csv(os.path.join(EXA, "maas", "Filtered_data11yrs.csv"), nrows=1).columns
              if c != "asthma"]
    X = d[active].astype(float)
    return dict(name="MAAS (ISAC, age 11)", X=X, kind={c: "cens" for c in active}, lod={c: 0.3 for c in active},
                y=d.asthma.astype(int).values, blocks=[(c, [c]) for c in active], always=[], cost={},
                extra=dict(all=pd.read_csv(os.path.join(EXA, "maas", "sIgE_data_age_11.csv")), components=comp,
                           n_all=n_all))


def graz_venom():
    """Hymenoptera venom allergy, Graz (Arzt-Gradwohl et al. 2026)."""
    d = pd.read_excel(os.path.join(EXA, "graz_venom", "complete_dataset_20Oct2025.xlsx"), sheet_name="data")
    ren = {}
    for c in d.columns:
        lc = c.lower()
        if lc.startswith("subject"):
            ren[c] = "id"
        elif lc == "age":
            ren[c] = "age"
        elif lc == "sex":
            ren[c] = "sex"
        elif lc.startswith("confirmed"):
            ren[c] = "venom"
        elif lc.startswith("grade"):
            ren[c] = "grade"
        elif lc.startswith("prick test bee"):
            ren[c] = "spt_bee"
        elif lc.startswith("prick test wasp"):
            ren[c] = "spt_wasp"
        elif lc.startswith("intradermal") and "bee" in lc:
            ren[c] = "idt_bee"
        elif lc.startswith("intradermal") and "wasp" in lc:
            ren[c] = "idt_wasp"
        elif lc.startswith("total ige"):
            ren[c] = "tige"
        elif lc.startswith("sige bee venom"):
            ren[c] = "bee"
        elif lc.startswith("sige wasp venom"):
            ren[c] = "wasp"
        elif lc.startswith("sige rapim"):
            ren[c] = "Api_m_" + lc.split("rapim")[1].split()[0]
        elif lc.startswith("sige rvesv"):
            ren[c] = "Ves_v_" + lc.split("rvesv")[1].split()[0]
        elif lc.startswith("bat") and "bee" in lc:
            ren[c] = "bat_bee"
        elif lc.startswith("bat") and "wasp" in lc:
            ren[c] = "bat_wasp"
    d = d.rename(columns=ren)
    num = ["age", "tige", "bee", "wasp", "Api_m_1", "Api_m_2", "Api_m_3", "Api_m_5", "Api_m_10", "Ves_v_1",
           "Ves_v_5", "bat_bee", "bat_wasp", "idt_bee", "idt_wasp"]
    for c in num:
        d[c] = _num(d[c])
    d["male"] = (d.sex.astype(str).str.lower() == "male").astype(float)
    # intradermal test: lowest positive concentration (ug/ml); 0 = negative -> ordinal score
    for c in ("idt_bee", "idt_wasp"):
        v = d[c]
        d[c + "_score"] = np.where(v.isna(), np.nan, np.where(v > 0, -np.log10(v.where(v > 0, 1.0)) + 1.0, 0.0))
    sige = ["bee", "wasp", "Api_m_1", "Ves_v_1", "Ves_v_5", "Api_m_2", "Api_m_3", "Api_m_5", "Api_m_10"]
    X = d[["age", "male", "tige"] + sige + ["bat_bee", "bat_wasp", "idt_bee_score", "idt_wasp_score"]].copy()
    kind = {"age": "cont", "male": "bin", "tige": "cens", **{c: "cens" for c in sige},
            "bat_bee": "cont", "bat_wasp": "cont", "idt_bee_score": "cont", "idt_wasp_score": "cont"}
    lod = {"tige": 2.0, **{c: 0.1 for c in sige}}
    blocks = [("bee venom components (rApi m 2/3/5/10)", ["Api_m_2", "Api_m_3", "Api_m_5", "Api_m_10"]),
              ("basophil activation test", ["bat_bee", "bat_wasp"]),
              ("intradermal test", ["idt_bee_score", "idt_wasp_score"])]
    always = ["age", "male", "tige", "bee", "wasp", "Api_m_1", "Ves_v_1", "Ves_v_5"]
    return dict(name="Hymenoptera venom (Graz)", X=X, kind=kind, lod=lod, y=None, blocks=blocks, always=always,
                cost={}, extra=dict(raw=d))


def moscow():
    """Moscow paediatric biochip (Smoldovskaya et al. 2018): binary sIgE/sIgG4 (cut-offs 0.35 IU/ml, 100 ng/ml)."""
    r = pd.read_excel(os.path.join(EXA, "moscow", "40413_2018_220_MOESM1_ESM.xlsx"), sheet_name="raw data", header=1)
    ige = [c for c in r.columns if str(c).startswith("sIgE")]
    igg = [c for c in r.columns if str(c).startswith("sIgG4")]
    X = r[ige].astype(float)
    X.columns = [c.replace("sIgE (", "").rstrip(")") for c in ige]
    return dict(name="Moscow paediatric biochip", X=X, kind={c: "bin" for c in X.columns}, lod={}, y=None,
                blocks=[(c, [c]) for c in X.columns], always=[], cost={},
                extra=dict(age=r.Age.astype(float).values, male=(r.Gender.astype(str).str.lower() == "m").astype(float).values,
                           diagnosis=r.Diagnosis.values, igg4=r[igg].astype(float).values))


def guo():
    """Chinese children with allergic asthma (Guo et al. 2026): 18 sIgE."""
    g = pd.read_excel(os.path.join(EXA, "guo", "Cleaned_data.xlsx"))
    conc = [c for c in g.columns if str(c).endswith("浓度")]
    code = {c: c.replace("浓度", "") for c in conc}
    X = g[conc].apply(pd.to_numeric, errors="coerce").astype(float)
    X.columns = [code[c] for c in conc]
    region = g["地区（南、北）"].astype(str).map(lambda s: "North" if "北" in s else ("South" if "南" in s else np.nan))
    return dict(name="Chinese asthmatic children (Guo)", X=X, kind={c: "cens" for c in X.columns},
                lod={c: 0.35 for c in X.columns}, y=None, blocks=[(c, [c]) for c in X.columns], always=[], cost={},
                extra=dict(region=region.values, age=pd.to_numeric(g["相关基本信息：—年龄(岁):"], errors="coerce").values,
                           male=(pd.to_numeric(g["性别"], errors="coerce") == 1).astype(float).values,
                           rhinitis=pd.to_numeric(g["确诊患有过敏性鼻炎？"], errors="coerce").values,
                           classes=g[[c.replace("浓度", "级别") for c in conc]].values))


def sosall():
    """SOS-ALL, South African amaXhosa children (Zhakparov et al. 2026): atopic dermatitis vs healthy."""
    d = pd.read_csv(os.path.join(EXA, "sosall", "clinical_data_mlm_sos_all.txt"), sep=None, engine="python")
    sige = ["sIgE_Der", "sIgE_milk", "sIgE_fish", "sIgE_wheat", "sIgE_peanut", "sIgE_soy", "sIgE_hazelnut",
            "sIgE_egg", "sIgE_CCD"]
    d = d[d[sige].notna().all(axis=1)].reset_index(drop=True)
    X = d[sige + ["total_IgE"]].astype(float)
    X = np.log10(X.clip(lower=0.005))
    return dict(name="SOS-ALL (atopic dermatitis)", X=X, kind={c: "cont" for c in X.columns}, lod={},
                y=(d.diagnosis == "AD").astype(int).values,
                blocks=[("house dust mite", ["sIgE_Der"]), ("milk", ["sIgE_milk"]), ("egg", ["sIgE_egg"]),
                        ("peanut and tree nut", ["sIgE_peanut", "sIgE_hazelnut"]), ("fish", ["sIgE_fish"]),
                        ("wheat and soy", ["sIgE_wheat", "sIgE_soy"]), ("CCD", ["sIgE_CCD"])],
                always=["total_IgE"], cost={}, extra=dict(raw=d))


LOADERS = dict(heart=heart, thyroid=thyroid, hepatitis=hepatitis, pima=pima, ckd=ckd, support2=support2,
               physionet2012=physionet2012, acc=acc, maas=maas, graz=graz_venom, moscow=moscow, guo=guo,
               sosall=sosall)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for k, f in LOADERS.items():
        try:
            D = f()
        except Exception as e:  # noqa: BLE001
            print(f"{k:14s} FAILED: {e!r}")
            continue
        X = D["X"]
        yv = D["y"]
        tests = sum((b for _, b in D["blocks"]), [])
        cc = X[tests + D["always"]].notna().all(axis=1).mean() if tests else np.nan
        print(f"{k:14s} n={len(X):5d} p={X.shape[1]:3d} y_prev={np.mean(yv) if yv is not None else float('nan'):.3f} "
              f"missing={X.isna().mean().mean():.3f} complete_rows={cc:.3f} blocks={len(D['blocks'])}")
