"""
a06_tables.py
=============
LaTeX tables (MDPI style) and a JSON file of every number quoted in the text,
generated from the result files. Nothing in the manuscript is typed by hand.

Run:  python a06_tables.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import ALL20, BLOCKS, NAME, SIGE  # noqa: E402

RES = os.path.join(HERE, "results")
TAB = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "tables"))
os.makedirs(TAB, exist_ok=True)
NUM = {}

LBL = {"Fraction missing": "Fraction of panel missing", "Kim et al. 2026": "Kim et al.~\\cite{kim2026}",
       "Conditional SD only": "Conditional SD only", "Margin only": "Distance to threshold only",
       "Multiple imputation": "Multiple imputation (reclassification probability)",
       "FLIP v1 (Gaussian)": "FLIP v1 (Gaussian, censoring ignored)",
       "FLIP, free covariance": "FLIP, free latent covariance", "FLIP": "\\textbf{FLIP} (censored, cross-reactivity)",
       "FLIP (posterior decision)": "\\textbf{FLIP}, posterior-majority decision"}
SC_ORDER = ["Fraction missing", "Kim et al. 2026", "Conditional SD only", "Margin only", "Multiple imputation",
            "FLIP v1 (Gaussian)", "FLIP, free covariance", "FLIP"]


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def mci(v, d=3, scale=1.0):
    v = np.asarray(v, float) * scale
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "--"
    return f"{np.median(v):.{d}f} ({np.quantile(v, 0.025):.{d}f}--{np.quantile(v, 0.975):.{d}f})"


def write(name, body):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf8") as fh:
        fh.write(body)
    print("  wrote", name)


def table(label, caption, colspec, header, rows, wide=False, foot=None, size="\\small"):
    width = "\\fulllength" if wide else "\\textwidth"
    out = ["\\begin{table}[H]", f"\\caption{{{caption}\\label{{{label}}}}}"]
    if wide:
        out.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    out += [size, f"\\begin{{tabularx}}{{{width}}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r if (r.startswith("\\midrule") or r.rstrip().endswith("\\\\")) else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabularx}"]
    if wide:
        out.append("\\end{adjustwidth}")
    if foot:
        out.append(f"\\noindent{{\\footnotesize{{{foot}}}}}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- cohorts
def t_cohort():
    d1 = pd.read_csv(os.path.join(HERE, "data", "d1_nhanes0506.csv"))
    d2 = pd.read_csv(os.path.join(HERE, "data", "d2_nhanes3_spt.csv"))
    race = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
            5: "Other/multiracial"}
    r = []

    def iqr(s):
        return f"{s.median():.0f} ({s.quantile(.25):.0f}--{s.quantile(.75):.0f})"
    r.append(f"Participants, $n$ & {len(d1):,} & {len(d2):,}")
    r.append(f"Train / calibration / test, $n$ & {(d1.split == 'train').sum():,} / {(d1.split == 'calib').sum():,} / "
             f"{(d1.split == 'test').sum():,} & {(d2.split == 'train').sum():,} / {(d2.split == 'calib').sum():,} / "
             f"{(d2.split == 'test').sum():,}")
    r.append(f"Age, years, median (IQR) & {iqr(d1.RIDAGEYR)} & {iqr(d2.HSAGEIR)}")
    r.append(f"Female, \\% & {pct(1 - d1.male.mean())} & {pct(1 - d2.male.mean())}")
    for k, v in race.items():
        r.append(f"\\quad {v}, \\% & {pct((d1.RIDRETH1 == k).mean())} & --")
    r.append(f"Allergens tested & 19 sIgE + total IgE & 10 SPT")
    r.append(f"Sensitised to $\\geq$1 allergen, \\% & {pct(d1.sensitized.mean())} & {pct(d2.sensitized.mean())}")
    cens = (d1[["cens_" + c for c in SIGE]] == 1).values.mean()
    r.append(f"Allergen cells below 0.35 kU/L, \\% & {pct(cens)} & --")
    r.append(f"Total IgE below 2.0 kU/L, \\% & {pct(d1.cens_LBXIGE.mean())} & --")
    r.append(f"Allergic disease (primary outcome), \\% & {pct(d1.y_allergic_disease.mean())} & --")
    r.append(f"Current asthma (secondary outcome), \\% & {pct(d1.y_current_asthma.mean())} & "
             f"{pct(d2.y_current_asthma.mean())}")
    r.append(f"Current asthma or wheeze in past 12 months, \\% & {pct(d1.y_harmonised.mean())} & "
             f"{pct(d2.y_harmonised.mean())}")
    write("tab_cohort", table("tab:cohort", "Characteristics of the two analysis cohorts. NHANES 2005--2006 "
                              "participants with all 19 allergen-specific IgE and total IgE measured; NHANES III "
                              "participants with all 10 skin prick tests and the negative control read. Unweighted.",
                              "lCC", "& \\textbf{NHANES 2005--2006 (serum)} & \\textbf{NHANES III (skin prick)}", r))
    NUM.update(n_d1=len(d1), n_d2=len(d2), sens_d1=float(d1.sensitized.mean()), sens_d2=float(d2.sensitized.mean()),
               cens_cells=float(cens), prev_ad=float(d1.y_allergic_disease.mean()),
               age_med_d1=float(d1.RIDAGEYR.median()))

    # appendix: analyte table
    rows = []
    for b, cols in BLOCKS.items():
        for c in cols:
            pos = (d1["cens_" + c] == 0).mean()
            rows.append(f"{NAME[c]} & {c} & {b} & {pct(1 - pos)} & {pct(pos) if c != 'LBXIGE' else '--'}")
    write("tab_analytes", table("tab:analytes", "The 20 analytes of the NHANES 2005--2006 panel, their ordering "
                                "block and the share of results below the detection limit (0.35 kU/L for "
                                "allergen-specific IgE, 2.0 kU/L for total IgE) in the analysis cohort.",
                                "lccCC", "\\textbf{Analyte} & \\textbf{NHANES variable} & \\textbf{Block} & "
                                "\\textbf{Below limit, \\%} & \\textbf{Positive, \\%}", rows))


# ----------------------------------------------------------------- literature & positioning
def t_literature():
    ops = pd.read_csv(os.path.join(RES, "t2_serum_operating_points.csv"))
    k95 = ops[(ops.method == "Fixed panel: greedy set cover") & (ops.alpha == 0.05)].iloc[0]
    fl = ops[(ops.method == "FLIP (total IgE + adaptive, free covariance)") & (ops.alpha == 0.05)].iloc[0]
    rows = [
        "Oezguen et al.~\\cite{oezguen2024} & USA (Texas), 12,065 children; 114 allergens & Network clusters $\\to$ one panel & 13 allergens detect 95\\% of sensitised; cost $-$52\\%",
        "Sammut et al.~\\cite{sammut2026} & Malta, children with allergic rhinitis & Conditional stepwise selection & 3 allergens detect $\\geq$95\\%",
        "Kulalert et al.~\\cite{kulalert2022} & Thailand, children with respiratory allergy & Minimum-number selection & 3 allergens $\\geq$95\\%; 8 allergens 99--100\\%",
        "Lee et al.~\\cite{lee2014} & Korea, rhinitis patients & Age-specific frequency ranking & 5 allergens $>$90\\%",
        "Stavart et al.~\\cite{stavart2026} & Southern Belgium, children & Profile mapping & two 5-allergen panels $>$95\\%",
        "Wang et al.~\\cite{wang2017} & Central China, allergic rhinitis & Frequency ranking & 8 allergens 99\\%",
        "Bazarbachi et al.~\\cite{bazarbachi2014} & Lebanon, 2,350 patients & Frequency ranking & 11--12 allergens 96\\%",
        "\\midrule",
        f"This study, literature method & NHANES 2005--2006, test split & Greedy set cover, certified & {k95.operating.split('=')[1]} allergens for every person; sensitivity {pct(k95.sensitivity)}\\%",
        f"\\textbf{{This study, FLIP}} & NHANES 2005--2006, test split & Per-person adaptive, certified & mean {fl.mean_tests:.1f} tests (incl. total IgE); sensitivity {pct(fl.sensitivity)}\\%",
    ]
    write("tab_literature", table("tab:literature", "Published allergen-panel optimisation studies and the result "
                                  "of their selection rule and of FLIP on the same data. All published studies select "
                                  "one fixed panel for a whole population from marginal sensitisation frequencies; none "
                                  "conditions on results the individual has already returned, uses a decision "
                                  "threshold, or provides an error guarantee. Rows above the line are reported in other "
                                  "populations and are not directly comparable.",
                                  ">{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}X",
                                  "\\textbf{Study} & \\textbf{Setting} & \\textbf{Selection} & \\textbf{Reported result}",
                                  rows, wide=True))

    Y, N, P = "\\checkmark", "--", "(\\checkmark)"
    feats = ["Per-patient score", "Uses observed values", "Uses co-sensitisation", "Censoring-aware",
             "Decision threshold", "Calibrated to true flips", "Finite-sample guarantee", "Test recommendation",
             "Ground-truth evaluation"]
    rows = [
        f"Kim et al.~\\cite{{kim2026}} & {Y} & {N} & {N} & {N} & {N} & {N} & {N} & {N} & {N}",
        f"White et al.~\\cite{{white2026}} & {Y} & {Y} & {Y} & {N} & {Y} & {N} & {N} & {P} & {N}",
        f"Mathiszig-Lee et al.~\\cite{{mathiszig2022}} & {Y} & {Y} & {Y} & {N} & {N} & {N} & {N} & {N} & {N}",
        f"Hoogland et al.~\\cite{{hoogland2020}} & {Y} & {Y} & {Y} & {N} & {N} & {N} & {N} & {N} & {N}",
        f"Zaffran et al.; Fan et al.~\\cite{{zaffran2023,fan2025}} & {Y} & {Y} & {P} & {N} & {N} & {N} & {Y} & {N} & {N}",
        f"DECIDE-Lab~\\cite{{allenbrand2026}} & {Y} & {N} & {N} & {N} & {Y} & {N} & {N} & {Y} & {N}",
        f"Active feature acquisition~\\cite{{kobayashi2025}} & {Y} & {Y} & {Y} & {N} & {N} & {N} & {N} & {Y} & {P}",
        f"Fixed-panel studies~\\cite{{oezguen2024,sammut2026,kulalert2022}} & {N} & {N} & {P} & {N} & {N} & {N} & {N} & {P} & {N}",
        f"\\textbf{{FLIP (this work)}} & {Y} & {Y} & {Y} & {Y} & {Y} & {Y} & {Y} & {Y} & {Y}",
    ]
    hdr = "\\textbf{Method} & " + " & ".join(f"\\rotatebox{{90}}{{\\textbf{{{x}}}}}" for x in feats)
    write("tab_positioning", table("tab:positioning", "Positioning of FLIP against the closest published approaches. "
                                   "\\checkmark, provided; (\\checkmark), partly or descriptively; --, not provided. "
                                   "Each element has a precedent somewhere; their conjunction does not.",
                                   ">{\\raggedright\\arraybackslash}p{4.3cm}" + "C" * len(feats), hdr, rows,
                                   wide=True))


# ----------------------------------------------------------------- structure
def t_structure():
    s = pd.read_csv(os.path.join(RES, "sigma_prediction_summary.csv"), index_col=0)
    pairs = pd.read_csv(os.path.join(RES, "corr_pairs.csv"))
    names = {"diagonal": ("Covariates only (no co-sensitisation)", "21 + 60"),
             "naive": ("Gaussian on reported values (censoring ignored)", "210 + 60"),
             "factor_v1": ("Cross-reactivity factors, prespecified families", "61 + 60"),
             "factor": ("Cross-reactivity factors, + PR-10 peanut and panallergens", "69 + 60"),
             "free": ("Free latent covariance", "210 + 60")}
    rows = []
    for k, (lab, npar) in names.items():
        r = s.loc[k]
        rows.append(f"{lab} & {npar} & {f(r.crps, 4)} & {f(r.pos_auroc)} & {f(r.pos_logloss)} & {f(r.pos_brier, 4)} & "
                    f"{f(r.pit_central90)}")
    write("tab_sigma", table("tab:sigma", "Predicting a panel that was not ordered (calibration split, $n$ = 1,794; "
                             "each of the nine blocks hidden in turn and predicted from the other blocks, the "
                             "covariates and total IgE). Mean over the 20 analytes. CRPS, continuous ranked "
                             "probability score on the reported log$_{10}$ scale (lower is better); AUROC, log loss "
                             "and Brier score refer to a positive result ($\\geq$0.35 kU/L); PIT 90\\% is the share of "
                             "randomised probability-integral-transform values in the central 90\\% (ideal 0.90). "
                             "Parameters: covariance + covariate regression.",
                             ">{\\raggedright\\arraybackslash}p{5.2cm}CCCCCC",
                             "\\textbf{Conditional model} & \\textbf{Parameters} & \\textbf{CRPS} & \\textbf{AUROC} & "
                             "\\textbf{Log loss} & \\textbf{Brier} & \\textbf{PIT 90\\%}", rows, wide=True))
    NUM.update(attn_median=float(pairs.attenuation.median()), attn_frac=float((pairs.attenuation > 0).mean()),
               auroc_hidden_free=float(s.loc["free", "pos_auroc"]), auroc_hidden_factor=float(s.loc["factor", "pos_auroc"]),
               auroc_hidden_diag=float(s.loc["diagonal", "pos_auroc"]), auroc_hidden_naive=float(s.loc["naive", "pos_auroc"]),
               auroc_hidden_v1=float(s.loc["factor_v1", "pos_auroc"]),
               crps_free=float(s.loc["free", "crps"]), crps_factor=float(s.loc["factor", "crps"]),
               crps_v1=float(s.loc["factor_v1", "crps"]), crps_naive=float(s.loc["naive", "crps"]),
               crps_diag=float(s.loc["diagonal", "crps"]),
               ll_naive=float(s.loc["naive", "pos_logloss"]), ll_factor=float(s.loc["factor", "pos_logloss"]))
    top = pairs.sort_values("latent", ascending=False).head(6)
    NUM["top_pairs"] = [(r.a, r.b, round(r.naive, 3), round(r.latent, 3)) for r in top.itertuples()]
    red = pd.read_csv(os.path.join(RES, "redundancy.csv"))
    rf = red[red.model == "factor"].set_index("analyte")
    NUM["redundancy_factor"] = rf.R2_given_other_blocks.round(3).to_dict()
    with open(os.path.join(RES, "structure_summary.txt")) as fh:
        for line in fh:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=")
                    try:
                        NUM["struct_" + k] = float(v)
                    except ValueError:
                        pass


# ----------------------------------------------------------------- task 1
def t_task1():
    P = pd.read_csv(os.path.join(RES, "t1_perf_y_allergic_disease.csv"))
    L = pd.read_csv(os.path.join(RES, "t1_ltt_y_allergic_disease.csv"))
    B = pd.read_csv(os.path.join(RES, "t1_bayes_y_allergic_disease.csv"))
    A = pd.read_csv(os.path.join(RES, "t1_acq_y_allergic_disease.csv"))
    info = pd.read_csv(os.path.join(RES, "t1_info_y_allergic_disease.csv"), index_col=0).iloc[:, 0]
    NUM.update(t1_tau=float(info["tau"]), t1_tprob=float(info["t_prob"]), t1_auc_full=float(info["auc_test_full"]),
               t1_nonzero=int(float(info["n_nonzero"])))
    rows = []
    for mech in ("MAR", "MNAR"):
        q = P[(P.mech == mech) & (P.scale == 1.0)]
        nrep = q.rep.nunique()
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{mech}: {nrep} amputation repetitions, median flip rate "
                    f"{pct(q.flip_rate.median())}\\%}}}} \\\\")
        fl = q[q.score == "FLIP"].set_index("rep")
        for nm in SC_ORDER:
            g = q[q.score == nm].set_index("rep")
            wins = int((fl.auroc > g.auroc.reindex(fl.index)).sum()) if nm != "FLIP" else None
            pv = wilcoxon(fl.auroc - g.auroc.reindex(fl.index)).pvalue if nm != "FLIP" else None
            wtxt = "--" if nm == "FLIP" else f"{wins}/{nrep} ({'$p<$0.001' if pv < 0.001 else f'$p$={pv:.3f}'})"
            rows.append(f"{LBL[nm]} & {mci(g.auroc)} & {mci(g.auprc)} & {mci(g.aurc, 2, 100)} & "
                        f"{mci(g.brier_cal, 2, 100)} & {wtxt}")
            NUM[f"t1_{mech}_{nm}_auroc"] = float(g.auroc.median())
            NUM[f"t1_{mech}_{nm}_auprc"] = float(g.auprc.median())
            NUM[f"t1_{mech}_{nm}_aurc"] = float(g.aurc.median())
        NUM[f"t1_{mech}_fliprate"] = float(q.flip_rate.median())
    write("tab_detection", table("tab:detection", "Detecting the decisions that would change if every panel were "
                                 "ordered (test split, $n$ = 1,798). Median (2.5--97.5th percentile) over amputation "
                                 "repetitions. AURC, area under the risk--coverage curve (\\% flipped among accepted, "
                                 "averaged over coverage 5--100\\%; lower is better). Brier: score after isotonic "
                                 "calibration of every method on the calibration split, $\\times$100 (lower is better). "
                                 "Last column: repetitions in which FLIP had the higher AUROC, paired Wilcoxon test.",
                                 ">{\\raggedright\\arraybackslash}p{4.4cm}CCCCC",
                                 "\\textbf{Score} & \\textbf{AUROC} & \\textbf{AUPRC} & \\textbf{AURC, \\%} & "
                                 "\\textbf{Brier$\\times$100} & \\textbf{FLIP better}", rows, wide=True,
                                 size="\\footnotesize"))

    # calibration of probability-valued scores
    rows = []
    for mech in ("MAR", "MNAR"):
        q = P[(P.mech == mech) & (P.scale == 1.0)]
        for nm in ["Multiple imputation", "FLIP v1 (Gaussian)", "FLIP, free covariance", "FLIP"]:
            g = q[q.score == nm]
            rows.append(f"{mech} & {LBL[nm]} & {mci(g.brier_raw, 2, 100)} & {mci(g.brier_cal, 2, 100)} & "
                        f"{mci(g.ece_raw, 2, 100)} & {mci(g.ece_cal, 2, 100)}")
            NUM[f"t1_{mech}_{nm}_ece_raw"] = float(g.ece_raw.median())
            NUM[f"t1_{mech}_{nm}_ece_cal"] = float(g.ece_cal.median())
    write("tab_calibration", table("tab:calibration", "Calibration of probability-valued scores against true flips "
                                   "on the test split, before and after isotonic recalibration on the calibration "
                                   "split. Median (2.5--97.5th percentile) over repetitions, $\\times$100. ECE, "
                                   "expected calibration error (10 equal-count bins).",
                                   "l>{\\raggedright\\arraybackslash}p{4.2cm}CCCC",
                                   "\\textbf{Mechanism} & \\textbf{Score} & \\textbf{Brier, raw} & "
                                   "\\textbf{Brier, calibrated} & \\textbf{ECE, raw} & \\textbf{ECE, calibrated}",
                                   rows, wide=True, size="\\footnotesize"))

    # guarantee
    rows = []
    scs = ["Fraction missing", "Kim et al. 2026", "Margin only", "Multiple imputation", "FLIP v1 (Gaussian)",
           "FLIP, free covariance", "FLIP", "FLIP (posterior decision)"]
    for mech in ("MAR", "MNAR"):
        q = L[(L.mech == mech) & (L.scale == 1.0)]
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{mech}}}}} \\\\")
        for nm in scs:
            cells = []
            for a in (0.01, 0.02, 0.05):
                g = q[(q.score == nm) & (q.kind == "marginal") & (q.alpha == a)]
                viol = float((g.realised.fillna(0) > a).mean())
                cm = g.coverage.median()
                cells.append(f"{pct(cm)} / {pct(g.realised.median(), 2) if cm > 0 else '--'}")
                NUM[f"t1_{mech}_{nm}_cov_{a}"] = float(g.coverage.median())
                NUM[f"t1_{mech}_{nm}_viol_{a}"] = viol
                NUM[f"t1_{mech}_{nm}_real_{a}"] = float(g.realised.median()) if g.realised.notna().any() else None
            gm = q[(q.score == nm) & (q.kind == "mondrian")]
            g2 = q[(q.score == nm) & (q.kind == "marginal") & (q.alpha == 0.02)]
            viol = float((g2.realised.fillna(0) > 0.02).mean())
            rows.append(f"{LBL[nm]} & " + " & ".join(cells) + f" & {pct(viol, 0)} & {pct(gm.coverage.median())} & "
                        f"{pct(gm.worst_group_marginal.median(), 2)} / {pct(gm.worst_group.median(), 2)}")
            NUM[f"t1_{mech}_{nm}_mondrian_cov"] = float(gm.coverage.median())
    write("tab_guarantee", table("tab:guarantee", "Learn-then-Test guarantee: share of test patients whose issued "
                                 "decision is accepted (coverage, \\%) and realised flip rate among them (\\%), for a "
                                 "target $\\alpha$ with $\\delta$ = 0.10. Median over repetitions. Violations: share of "
                                 "repetitions with realised flip rate above $\\alpha$ = 0.02. Mondrian: separate "
                                 "thresholds by number of unordered panels (0, 1, 2, 3, $\\geq$4), $\\alpha$ = 0.02; "
                                 "worst group: highest flip rate in any group with $\\geq$20 accepted patients under "
                                 "the marginal / Mondrian rule.",
                                 ">{\\raggedright\\arraybackslash}p{3.8cm}CCCCCC",
                                 "\\textbf{Score} & \\textbf{$\\alpha$ = 0.01} & \\textbf{$\\alpha$ = 0.02} & "
                                 "\\textbf{$\\alpha$ = 0.05} & \\textbf{Violations} & \\textbf{Mondrian coverage} & "
                                 "\\textbf{Worst group}", rows, wide=True, size="\\scriptsize"))

    # posterior decision
    rows = []
    srcs = [("y_allergic_disease", "", "Allergic disease"), ("y_current_asthma", "", "Current asthma")]
    for oc, sfx, lab in srcs:
        Bq = pd.read_csv(os.path.join(RES, f"t1_bayes_{oc}{sfx}.csv"))
        for mech in ("MAR", "MNAR"):
            g = Bq[(Bq.mech == mech) & (Bq.scale == 1.0)]
            red = 1 - g.flip_bayes / g.flip_imputed
            rows.append(f"{lab} & {mech} & 1.0 & {mci(g.flip_imputed, 2, 100)} & {mci(g.flip_mi_majority, 2, 100)} & "
                        f"{mci(g.flip_bayes, 2, 100)} & {mci(red, 0, 100)}")
            NUM[f"bayes_{oc}_{mech}_imp"] = float(g.flip_imputed.median())
            NUM[f"bayes_{oc}_{mech}_post"] = float(g.flip_bayes.median())
            NUM[f"bayes_{oc}_{mech}_red"] = float(red.median())
    Bs = pd.read_csv(os.path.join(RES, "t1_bayes_y_allergic_disease_scale.csv"))
    for sc in (0.5, 1.5):
        for mech in ("MAR", "MNAR"):
            g = Bs[(Bs.mech == mech) & (Bs.scale == sc)]
            red = 1 - g.flip_bayes / g.flip_imputed
            rows.append(f"Allergic disease & {mech} & {sc} & {mci(g.flip_imputed, 2, 100)} & "
                        f"{mci(g.flip_mi_majority, 2, 100)} & {mci(g.flip_bayes, 2, 100)} & {mci(red, 0, 100)}")
    write("tab_bayes", table("tab:bayes", "Issuing the decision from the FLIP posterior instead of from single "
                             "imputation. Flipped decisions, \\% of test patients, median (2.5--97.5th percentile) over "
                             "repetitions. Missingness scale multiplies the not-ordered rate of every panel.",
                             "llcCCCC", "\\textbf{Outcome} & \\textbf{Mechanism} & \\textbf{Scale} & "
                             "\\textbf{Single imputation} & \\textbf{MI majority} & \\textbf{FLIP posterior} & "
                             "\\textbf{Reduction, \\%}", rows, wide=True, size="\\footnotesize"))

    # acquisition
    rows = []
    for rule, rlab in (("posterior", "Decision issued from the FLIP posterior"),
                       ("imputed", "Decision issued by single imputation")):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{rlab}}}}} \\\\")
        for setting, pols in (("one panel per patient", ["Random panel", "Kim: most important missing panel", "FLIP-VOI"]),
                              ("budget 10%", ["No extra testing", "Kim rank + Kim panel", "Margin rank + Kim panel",
                                              "FLIP-VOI: expected flips", "FLIP-VOI: certified decisions"])):
            for pol in pols:
                cells = []
                for mech in ("MAR", "MNAR"):
                    g = A[(A.mech == mech) & (A.scale == 1.0) & (A.rule == rule) & (A.setting == setting) &
                          (A.policy == pol)]
                    cells.append(f"{mci(g.flip_after, 2, 100)}")
                    cells.append(f"{pct(g.certified.median())}")
                    NUM[f"acq_{rule}_{setting}_{pol}_{mech}"] = float(g.flip_after.median())
                    NUM[f"acqcert_{rule}_{setting}_{pol}_{mech}"] = float(g.certified.median())
                st = "One extra panel per patient" if setting.startswith("one") else "Budget: 10\\%"
                rows.append(f"{st}: {pol.replace('Kim', 'Kim et al.')} & " + " & ".join(cells))
    write("tab_acquisition", table("tab:acquisition", "Which panel to order next. Flipped decisions after testing "
                                   "(\\% of test patients, median and 2.5--97.5th percentile) and share of patients "
                                   "certified by the Learn-then-Test rule ($\\alpha$ = 0.02) after testing. Budget: "
                                   "one extra panel for 10\\% of patients at the mean panel cost. Kim et al. panel: the "
                                   "missing panel with the largest summed importance weight of~\\cite{kim2026}.",
                                   ">{\\raggedright\\arraybackslash}p{5.4cm}CCCC",
                                   "\\textbf{Policy} & \\textbf{MAR: flips, \\%} & \\textbf{MAR: certified, \\%} & "
                                   "\\textbf{MNAR: flips, \\%} & \\textbf{MNAR: certified, \\%}", rows, wide=True,
                                   size="\\scriptsize"))

    # sensitivity: asthma + scale
    rows = []
    for oc, sfx, lab in (("y_current_asthma", "", "Current asthma"), ("y_allergic_disease", "_scale", "Allergic disease")):
        Pq = pd.read_csv(os.path.join(RES, f"t1_perf_{oc}{sfx}.csv"))
        Lq = pd.read_csv(os.path.join(RES, f"t1_ltt_{oc}{sfx}.csv"))
        for sc in sorted(Pq.scale.unique()):
            for mech in ("MAR", "MNAR"):
                q = Pq[(Pq.mech == mech) & (Pq.scale == sc)]
                l = Lq[(Lq.mech == mech) & (Lq.scale == sc) & (Lq.kind == "marginal") & (Lq.alpha == 0.02)]
                cells = []
                for nm in ("Kim et al. 2026", "Margin only", "Multiple imputation", "FLIP"):
                    cells.append(f"{f(q[q.score == nm].auroc.median())} / {pct(l[l.score == nm].coverage.median(), 0)}")
                rows.append(f"{lab} & {mech} & {sc} & {pct(q.flip_rate.median())} & " + " & ".join(cells))
                NUM[f"sens_{oc}{sfx}_{mech}_{sc}_flip_auroc"] = float(q[q.score == 'FLIP'].auroc.median())
                NUM[f"sens_{oc}{sfx}_{mech}_{sc}_kim_auroc"] = float(q[q.score == 'Kim et al. 2026'].auroc.median())
                NUM[f"sens_{oc}{sfx}_{mech}_{sc}_flip_cov"] = float(l[l.score == 'FLIP'].coverage.median())
    write("tab_sensitivity", table("tab:sensitivity", "Secondary outcome and amount of missingness. Each cell: "
                                   "flip-detection AUROC / certified coverage at $\\alpha$ = 0.02 (\\%). Medians over "
                                   "repetitions (current asthma: 10 MAR, 5 MNAR; scale: 6 + 6).",
                                   "llccCCCC", "\\textbf{Outcome} & \\textbf{Mech.} & \\textbf{Scale} & "
                                   "\\textbf{Flip rate, \\%} & \\textbf{Kim et al.} & \\textbf{Distance only} & "
                                   "\\textbf{MI} & \\textbf{FLIP}", rows, wide=True, size="\\footnotesize"))

    # DCA summary numbers
    D = pd.read_csv(os.path.join(RES, "t1_dca_y_allergic_disease.csv"))
    for mech in ("MAR", "MNAR"):
        q = D[(D.mech == mech) & (D.scale == 1.0)]
        for k in (0.0, 0.005, 0.01):
            g = q[q.kappa == k].groupby("strategy").nb.median()
            for s_, v in g.items():
                NUM[f"dca_{mech}_{k}_{s_}"] = float(v)
        NUM[f"dca_{mech}_deferred_flip"] = float(q[(q.kappa == 0) & (q.strategy == "FLIP defer (LTT)")].deferred.median())
        NUM[f"dca_{mech}_deferred_post"] = float(q[(q.kappa == 0) &
                                                   (q.strategy == "Posterior decision + FLIP defer (LTT)")].deferred.median())
    sh = pd.read_csv(os.path.join(RES, "t1_shapley_shares.csv"))
    NUM["shapley_mean"] = sh.mean(skipna=True).round(3).to_dict()
    NUM["shapley_n"] = int(len(sh))


# ----------------------------------------------------------------- task 2
def matched(sim_te, sens, extra_cost, grid_n=300):
    """tests (n, G) and missed (n, G) over a lambda grid, for bootstrap."""
    pa, fp = sim_te["p_any"], sim_te["first_pos"]
    lams = np.unique(np.quantile(pa[np.isfinite(pa)], np.linspace(0, 1, grid_n)))
    K = pa.shape[1] - 1
    T = np.empty((len(fp), len(lams)))
    Mi = np.empty((len(fp), len(lams)), bool)
    for g, lam in enumerate(lams):
        below = np.where(np.isfinite(pa), pa <= lam, False)
        stop = np.where(below.any(1), below.argmax(1), K)
        found = fp < stop
        T[:, g] = np.where(found, fp + 1, stop) + extra_cost
        Mi[:, g] = sens & ~found
    return T, Mi


def tests_at(T, Mi, sens, idx, target):
    s = sens[idx]
    sensv = 1 - Mi[idx][s].mean(0)
    ok = sensv >= target
    return T[idx].mean(0)[ok].min() if ok.any() else np.nan


def t_task2():
    out_rows = []
    for ds, fn, lab in (("serum", "t2_serum_operating_points.csv", "NHANES 2005--2006, serum sIgE"),
                        ("spt", "t2_spt_operating_points.csv", "NHANES III, skin prick test")):
        ops = pd.read_csv(os.path.join(RES, fn))
        out_rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for m in ops.method.unique():
            cells = []
            for a in (0.10, 0.05, 0.025, 0.01):
                g = ops[(ops.method == m) & (ops.alpha == a)].iloc[0]
                cells.append(f"{g.mean_tests:.2f} ({pct(g.sensitivity)})")
                NUM[f"t2_{ds}_{m}_{a}_tests"] = float(g.mean_tests)
                NUM[f"t2_{ds}_{m}_{a}_sens"] = float(g.sensitivity)
                NUM[f"t2_{ds}_{m}_{a}_op"] = g.operating
                NUM[f"t2_{ds}_{m}_{a}_nonsens"] = float(g.tests_nonsensitised)
                NUM[f"t2_{ds}_{m}_{a}_sensd"] = float(g.tests_sensitised)
            mm = m.replace("FLIP", "\\textbf{FLIP}") if m.startswith("FLIP") else m
            out_rows.append(f"{mm} & " + " & ".join(cells))
    write("tab_atopy", table("tab:atopy", "Detecting sensitisation to at least one allergen with a certified "
                             "sensitivity. For each target miss rate $\\alpha$ the operating point (FLIP stopping "
                             "threshold, or panel length $k$) was certified on the calibration split with $\\delta$ = "
                             "0.10 and applied to the test split. Cells: mean tests per person (including the total IgE "
                             "assay where used) and, in brackets, realised sensitivity (\\%).",
                             ">{\\raggedright\\arraybackslash}p{6.0cm}CCCC",
                             "\\textbf{Strategy} & \\textbf{$\\alpha$ = 0.10} & \\textbf{$\\alpha$ = 0.05} & "
                             "\\textbf{$\\alpha$ = 0.025} & \\textbf{$\\alpha$ = 0.01}", out_rows, wide=True,
                             size="\\scriptsize"))

    # matched-sensitivity comparison with bootstrap CIs (serum, test split)
    S = pickle.load(open(os.path.join(RES, "t2_serum_sims.pkl"), "rb"))
    te = S["is_te"]
    sens = S["sens"][S["evidx"]][te]
    pos = S["pos"][S["evidx"]][te]
    curves = {}
    for nm, cost in (("FLIP (total IgE + adaptive, free covariance)", 1.0), ("FLIP (total IgE + adaptive)", 1.0),
                     ("FLIP (adaptive, no total IgE)", 0.0), ("Ablation: no cross-reactivity (diagonal)", 1.0)):
        sim = {k: v[te] for k, v in S["sims"][nm].items()}
        curves[nm] = matched(sim, sens, cost)
    order = S["order_greedy"]
    hits = pos[:, order]
    first = np.where(hits.any(1), hits.argmax(1), len(order))
    Tq = np.column_stack([np.where(first < k, first + 1, k) for k in range(1, 20)]).astype(float)
    Mq = np.column_stack([sens & ~(first < k) for k in range(1, 20)])
    curves["Sequential, greedy order, stop at first positive"] = (Tq, Mq)
    Tf = np.column_stack([np.full(len(sens), float(k)) for k in range(1, 20)])
    Mf = np.column_stack([sens & ~pos[:, order[:k]].any(1) for k in range(1, 20)])
    curves["Fixed panel: greedy set cover"] = (Tf, Mf)
    rng = np.random.default_rng(0)
    nb = 1000
    boots = [rng.integers(0, len(sens), len(sens)) for _ in range(nb)]
    ref = "FLIP (total IgE + adaptive, free covariance)"
    rows = []
    for tgt in (0.90, 0.95, 0.975, 0.99):
        full = np.arange(len(sens))
        base = {nm: tests_at(*curves[nm], sens, full, tgt) for nm in curves}
        bt = {nm: np.array([tests_at(*curves[nm], sens, b, tgt) for b in boots]) for nm in curves}
        cells = []
        for nm in curves:
            cells.append(f"{base[nm]:.2f}")
        for cmp_ in ("Sequential, greedy order, stop at first positive", "Fixed panel: greedy set cover"):
            rel = 1 - bt[ref] / bt[cmp_]
            cells.append(f"{100 * (1 - base[ref] / base[cmp_]):.0f} ({100 * np.nanquantile(rel, .025):.0f} to "
                         f"{100 * np.nanquantile(rel, .975):.0f})")
            NUM[f"t2_matched_{tgt}_red_vs_{cmp_}"] = float(1 - base[ref] / base[cmp_])
            NUM[f"t2_matched_{tgt}_red_vs_{cmp_}_lo"] = float(np.nanquantile(rel, .025))
            NUM[f"t2_matched_{tgt}_red_vs_{cmp_}_hi"] = float(np.nanquantile(rel, .975))
        for nm in curves:
            NUM[f"t2_matched_{tgt}_{nm}"] = float(base[nm])
        rows.append(f"{100 * tgt:.1f}\\% & " + " & ".join(cells))
    short = ["FLIP, free cov.", "FLIP, factor", "FLIP, no IgE", "No co-sens.", "Sequential", "Fixed panel"]
    hdr = ("\\textbf{Sensitivity} & " + " & ".join(f"\\textbf{{{s}}}" for s in short) +
           " & \\textbf{vs sequential, \\%} & \\textbf{vs fixed panel, \\%}")
    write("tab_matched", table("tab:matched", "Mean tests per person needed to reach a given sensitivity on the "
                               "NHANES 2005--2006 test split (best operating point of each strategy on its "
                               "sensitivity--effort curve). Last two columns: reduction in testing by FLIP (free "
                               "covariance, with total IgE) relative to sequential testing in the greedy order and to "
                               "the fixed greedy panel, with 95\\% bootstrap intervals (1,000 resamples of persons).",
                               "c" + "C" * 6 + ">{\\centering\\arraybackslash}p{2.1cm}>{\\centering\\arraybackslash}p{2.1cm}",
                               hdr, rows, wide=True, size="\\scriptsize"))
    cp = pd.read_csv(os.path.join(RES, "t2_spt_curves.csv"))
    for m_, q in cp.groupby("method"):
        for tgt in (0.90, 0.95, 0.975, 0.99):
            ok = q[q.sensitivity >= tgt]
            NUM[f"t2spt_matched_{tgt}_{m_}"] = float(ok.mean_tests.min()) if len(ok) else None
    NUM["greedy_order"] = [S["names"][j] for j in order]
    sim = S["sims"]["FLIP (total IgE + adaptive)"]
    first_te = sim["order"][te][:, 0]
    vc = pd.Series([S["names"][j] for j in first_te if j >= 0]).value_counts(normalize=True)
    NUM["first_test_share"] = vc.round(3).to_dict()
    NUM["t2_sens_test"] = float(sens.mean())


def t_dust():
    du = pd.read_csv(os.path.join(RES, "dust_information.csv"))
    rows = [f"{r.dust_molecule} & {r.sIgE} & {f(r.pearson_pair)} & {f(r.cv_r2_dust_block)} & "
            f"{f(r.auroc_dust_block)} & {f(r.auroc_other_sIgE_panels)}" for r in du.itertuples()]
    write("tab_dust", table("tab:dust", f"Information a household dust allergen assessment carries about the same "
                            f"person's allergen-specific IgE (NHANES 2005--2006, $n$ = {int(du.n.iloc[0]):,} with all "
                            f"11 dust analytes). $r$: correlation of matching log$_{{10}}$ dust and sIgE; $R^2$ and "
                            f"AUROC: 10-fold cross-validated prediction of log sIgE and of a positive sIgE from the "
                            f"whole dust block. Last column: AUROC from the other sIgE panels and total IgE under the "
                            f"cross-reactivity model (calibration split).",
                            "llCCCC", "\\textbf{Dust molecule} & \\textbf{sIgE} & \\textbf{$r$} & "
                            "\\textbf{$R^2$, dust block} & \\textbf{AUROC, dust} & \\textbf{AUROC, other panels}",
                            rows, wide=True, size="\\footnotesize"))
    NUM.update(dust_r2_max=float(du.cv_r2_dust_block.max()), dust_auc_min=float(du.auroc_dust_block.min()),
               dust_auc_max=float(du.auroc_dust_block.max()), dust_other_min=float(du.auroc_other_sIgE_panels.min()),
               dust_other_max=float(du.auroc_other_sIgE_panels.max()), dust_n=int(du.n.iloc[0]))


if __name__ == "__main__":
    t_cohort()
    t_literature()
    t_structure()
    t_task1()
    t_task2()
    t_dust()
    with open(os.path.join(RES, "numbers.json"), "w") as fh:
        json.dump(NUM, fh, indent=1, default=str)
    print("wrote numbers.json with", len(NUM), "entries")
