"""
a17_main_tables.py
==================
Six compact main-text tables that compare FLIP with the alternatives, built from saved results:

  tab_certified    Learn-then-Test: patients certified at a guaranteed flip rate (a02 -> t1_ltt_*)
  tab_posterior    decision errors of single imputation, MI majority and the FLIP posterior (t1_bayes_*)
  tab_ordering     which panel to order next, decisions from the FLIP posterior (t1_acq_*)
  tab_tests_lit    tests per person needed to reach each published study's answer (a12 -> ext_fig_tests.csv)
  tab_assoc        association estimates with full-panel and FLIP-classified sensitisation (a13, a16)
  tab_hidden_ext   prediction of unordered tests with and without cross-reactivity (a10 -> ext_hidden_summary)

The full versions of the first three tables are written by a06_tables.py (Supplementary Tables S9-S11).
Run after a02, a06, a10, a12, a13 and a16:  python a17_main_tables.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.abspath(os.path.join(HERE, "..", "outputs", "tables"))
RL = ">{\\raggedright\\arraybackslash}"


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def pct(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def mci(v, d=2, scale=100.0):
    v = np.asarray(v, float) * scale
    v = v[np.isfinite(v)]
    return f"{np.median(v):.{d}f} ({np.quantile(v, 0.025):.{d}f}--{np.quantile(v, 0.975):.{d}f})"


def table(name, label, caption, colspec, header, rows, size="\\scriptsize", foot=None):
    out = ["\\begin{table}[H]", f"\\caption{{{caption}\\label{{{label}}}}}",
           "\\begin{adjustwidth}{-\\extralength}{0cm}", size,
           f"\\begin{{tabularx}}{{\\fulllength}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r if r.startswith("\\midrule") or r.endswith("\\\\") else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabularx}"]
    if foot:
        out.append(f"\\par\\smallskip\\noindent{{\\scriptsize {foot}}}")
    out += ["\\end{adjustwidth}", "\\end{table}"]
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("  wrote", name)


SCORE = {"Fraction missing": "Fraction of panel missing", "Kim et al. 2026": "Kim et al.~\\cite{kim2026}",
         "Margin only": "Distance to threshold~\\cite{geifman2017}",
         "Multiple imputation": "Multiple imputation~\\cite{white2026,mathiszig2022}",
         "FLIP": "\\textbf{FLIP}", "FLIP (posterior decision)": "\\textbf{FLIP}, posterior-majority decision"}


# --------------------------------------------------------------- Task 1: guarantee
def t_certified():
    L = rd("t1_ltt_y_allergic_disease.csv")
    L = L[(L.scale == 1.0) & (L.kind == "marginal")]
    cols = [(m, a) for m in ("MAR", "MNAR") for a in (0.01, 0.02, 0.05)]
    cov = {(s, m, a): L[(L.score == s) & (L.mech == m) & (L.alpha == a)] for s in SCORE for m, a in cols}
    best = {(m, a): max(cov[(s, m, a)].coverage.median() for s in SCORE) for m, a in cols}
    rows = []
    for s, lab in SCORE.items():
        cells = []
        for m, a in cols:
            g = cov[(s, m, a)]
            c = g.coverage.median()
            txt = pct(c) if c <= 0 else f"{pct(c)} ({pct(g.realised.median())})"
            cells.append(f"\\textbf{{{txt}}}" if c > 0 and np.isclose(c, best[(m, a)]) else txt)
        rows.append(f"{lab} & " + " & ".join(cells))
    table("tab_certified", "tab:certified",
          "Decisions certified by the Learn-then-Test rule (test split): share of patients whose issued decision "
          "is accepted (\\%), with the realised flip rate among them in parentheses (\\%), at a target flip rate "
          "$\\alpha$ ($\\delta$ = 0.10); median over 20 MAR and 10 MNAR repetitions. Every score keeps the "
          "guarantee; a better score certifies more patients and defers fewer decisions (0.0, no patient can be "
          "certified). Best value per column in bold; violations and the Mondrian version in Table S9.",
          RL + "p{3.6cm}CCCCCC",
          "& \\multicolumn{3}{c}{\\textbf{MAR}} & \\multicolumn{3}{c}{\\textbf{MNAR}} \\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
          "\\textbf{Score} & $\\alpha$ = 0.01 & $\\alpha$ = 0.02 & $\\alpha$ = 0.05 & $\\alpha$ = 0.01 & "
          "$\\alpha$ = 0.02 & $\\alpha$ = 0.05", rows)


# --------------------------------------------------------------- Task 1: posterior decision
def t_posterior():
    rows = []
    for oc, lab in (("y_allergic_disease", "Allergic disease"), ("y_current_asthma", "Current asthma")):
        B = rd(f"t1_bayes_{oc}.csv")
        for mech in ("MAR", "MNAR"):
            g = B[(B.mech == mech) & (B.scale == 1.0)]
            red = 1 - g.flip_bayes / g.flip_imputed
            rows.append(f"{lab} & {mech} & {mci(g.flip_imputed)} & {mci(g.flip_mi_majority)} & "
                        f"\\textbf{{{mci(g.flip_bayes)}}} & {mci(red, 0)}")
    table("tab_posterior", "tab:posterior",
          "Decision errors with unordered panels: flipped decisions (\\% of test patients; median and "
          "2.5--97.5th percentile over repetitions) when the decision is issued after single imputation with a MissForest-type "
          "imputer~\\cite{stekhoven2012} (as in Kim et al.~\\cite{kim2026}), by "
          "majority vote over 30 multiple imputations~\\cite{vanbuuren2011,white2026}, or from the \\FLIP{} posterior (posterior-majority decision). "
          "Last column: reduction relative to single imputation. Other amounts of missingness in Table S10.",
          "llCCCC", "\\textbf{Outcome} & \\textbf{Mechanism} & \\textbf{Single imputation} & "
          "\\textbf{MI majority vote} & \\textbf{FLIP posterior} & \\textbf{Reduction, \\%}", rows,
          size="\\footnotesize")


# --------------------------------------------------------------- Task 1: test ordering
def t_ordering():
    A = rd("t1_acq_y_allergic_disease.csv")
    A = A[(A.scale == 1.0) & (A.rule == "posterior")]
    rows = []
    for setting, head, pols in (
            ("one panel per patient", "One extra panel for every patient",
             [("Random panel", "Random panel"), ("Kim: most important missing panel",
                                                 "Most important missing panel (Kim et al.~\\cite{kim2026})"),
              ("FLIP-VOI", "\\textbf{FLIP-VOI}")]),
            ("budget 10%", "One extra panel for 10\\% of patients",
             [("No extra testing", "No extra testing"),
              ("Kim rank + Kim panel", "Patients ranked by Kim et al.\\ score"),
              ("Margin rank + Kim panel", "Patients ranked by distance to threshold~\\cite{geifman2017}"),
              ("FLIP-VOI: expected flips", "\\textbf{FLIP-VOI}, expected flips"),
              ("FLIP-VOI: certified decisions", "\\textbf{FLIP-VOI}, certified decisions")])):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{head}}}}} \\\\")
        for pol, lab in pols:
            cells = []
            for mech in ("MAR", "MNAR"):
                g = A[(A.mech == mech) & (A.setting == setting) & (A.policy == pol)]
                cells += [mci(g.flip_after), pct(g.certified.median())]
            rows.append(f"{lab} & " + " & ".join(cells))
    table("tab_ordering", "tab:ordering",
          "Choosing the next panel when decisions are issued from the \\FLIP{} posterior: flipped decisions after "
          "testing (\\% of test patients; median and 2.5--97.5th percentile) and share of patients certified "
          "after testing ($\\alpha$ = 0.02). In the comparators, the panel ordered is the missing panel with the "
          "largest importance weight of Kim et al.~\\cite{kim2026}. Decisions issued by single imputation in "
          "Table S11.",
          RL + "p{4.6cm}CCCC",
          "& \\multicolumn{2}{c}{\\textbf{MAR}} & \\multicolumn{2}{c}{\\textbf{MNAR}} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "\\textbf{Policy} & Flips, \\% & Certified, \\% & Flips, \\% & Certified, \\%", rows)


# --------------------------------------------------------------- published studies: tests needed
CITE = {"Salo 2014": "salo2014", "Gergen 2009": "gergen2009", "Arbes 2005": "arbes2005", "Salo 2011": "salo2011",
        "Shargorodsky 2015": "shargorodsky2015", "Visness 2009": "visness2009", "Arbes 2013": "arbes2013",
        "Keet 2012": "keet2012", "Martinroche 2026": "martinroche2026", "Fontanella 2018": "fontanella2018",
        "Guo 2026": "guo2026", "Smoldovskaya 2018": "smoldovskaya2018", "McGowan 2015": "mcgowan2015",
        "Perkin 2016": "perkin2016", "Du Toit 2015": "dutoit2015"}
ASSOC = ("Salo 2011", "Shargorodsky 2015", "Visness 2009", "Arbes 2013", "Keet 2012")


def tex(s):
    return s.replace("–", "--").replace("%", "\\%")


def num(v):
    return "--" if not np.isfinite(v) else f"{v:.1f}"


def t_tests_lit():
    R = rd("ext_fig_tests.csv")
    rows, last = [], None
    for _, r in R.iterrows():
        study, panel = r.label.split(" — ")
        mark = "$^{a}$" if study == "Gergen 2009" else "$^{b}$" if study in ASSOC else ""
        first = f"{study}~\\cite{{{CITE[study]}}}" if study != last else ""
        last = study
        rows.append(f"{first} & {tex(panel)}{mark} & {int(r.design)} & {num(r.fixed)} & {num(r.seq)} & "
                    f"\\textbf{{{r.flip:.1f}}} & {100 * (1 - r.flip / r.design):.0f}")
    table("tab_tests_lit", "tab:tests_lit",
          "Allergen tests per person needed to reach each published study's answer on the study's own data. "
          "Design, tests used in the original study; fixed panel, best single panel chosen by greedy set cover "
          "(the method of the panel-optimisation literature~\\cite{gergen2009,sammut2026,kulalert2022}); sequential, testing in a fixed order until the "
          "first positive result. All strategies are certified at a miss rate of at most 5\\% (NHANES: test "
          "split, or rotating four-fold cross-fitting; other cohorts: rotating four-fold cross-fitting); --, not "
          "evaluated. Last column: tests saved by \\FLIP{} relative to the study design.",
          RL + "p{2.7cm}" + RL + "Xccccc",
          "\\textbf{Study} & \\textbf{Panel} & \\textbf{Design} & \\textbf{Fixed panel} & \\textbf{Sequential} & "
          "\\textbf{FLIP} & \\textbf{Saved, \\%}", rows,
          foot="$^{a}$Matched to the published 92.8\\% and 99.2\\% of atopic persons identified. "
               "$^{b}$Association study; \\FLIP{} counts include total IgE (see Table~\\ref{tab:assoc}).")


def t_assoc():
    a = rd("lit_assoc.csv")
    v = rd("lit_assoc_visness.csv")
    m = rd("lit_assoc_more.csv")

    def get(df, study, q):
        r = df[(df.study == study) & df.quantity.str.contains(q, regex=False)]
        assert len(r) == 1, (study, q, len(r))
        return r.iloc[0]

    spec = [  # (study, source, quantity substring, label, decimals)
        ("Salo 2011", a, "adjusted OR, current hayfever", "adjusted OR, hay fever, plant-specific IgE", 2),
        ("Salo 2011", a, "adjusted OR, current allergies", "adjusted OR, current allergies, plant-specific IgE", 2),
        ("Shargorodsky 2015", a, "OR model 1, former", "OR, former vs never smokers", 2),
        ("Shargorodsky 2015", a, "OR model 1, cur1", "OR, current smokers, cotinine tertile 1", 2),
        ("Shargorodsky 2015", a, "OR model 1, cur2", "OR, current smokers, cotinine tertile 2", 2),
        ("Shargorodsky 2015", a, "OR model 1, cur3", "OR, current smokers, cotinine tertile 3", 2),
        ("Visness 2009", v, "atopy, obese vs normal, 6-19 y", "age-adjusted OR of obesity, atopy", 2),
        ("Visness 2009", v, "food, obese vs normal, 6-19 y", "age-adjusted OR of obesity, food sensitisation", 2),
        ("Arbes 2013", m, "both, %, analysis cohort", "asthma, atopy and high eosinophils, \\%", 1),
        ("Arbes 2013", m, "excess prevalence due to interaction, %, analysis cohort",
         "excess asthma prevalence due to interaction, \\%", 1),
        ("Keet 2012", m, "US-born vs foreign-born, 6-20 y, analysis cohort", "adjusted OR, food sensitisation, US-born", 2),
    ]
    rows, last = [], None
    for study, df, q, lab, d in spec:
        r = get(df, study, q)
        first = f"{study}~\\cite{{{CITE[study]}}}" if study != last else ""
        last = study
        rows.append(f"{first} & {lab} & {r.reproduced_full_panel:.{d}f} & {r.flip:.{d}f} & "
                    f"{int(r.design_tests)} & \\textbf{{{r.flip_tests:.1f}}}")
    table("tab_assoc", "tab:assoc",
          "Published association analyses repeated with sensitisation classified by adaptive \\FLIP{} testing "
          "instead of the full panel, in the same participants (survey-weighted models of the original studies; "
          "\\FLIP{} certified at a miss rate of at most 5\\% with rotating four-fold cross-fitting; \\FLIP{} test "
          "counts include total IgE). Estimates are given with each definition of sensitisation, together with "
          "the allergen tests per person each definition needed.",
          RL + "p{2.7cm}" + RL + "Xcccc",
          "& & \\multicolumn{2}{c}{\\textbf{Estimate}} & \\multicolumn{2}{c}{\\textbf{Tests per person}} \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
          "\\textbf{Study} & \\textbf{Quantity} & Full panel & FLIP & Full panel & FLIP", rows,
          size="\\footnotesize")


# --------------------------------------------------------------- unordered tests are predictable
HIDDEN = {"NHANES": ("NHANES 2005--2006 (this study)", "", "19 sIgE, whole panels hidden"),
          "Chip": ("Allergen Chip Challenge", "martinroche2026", "105 ISAC components"),
          "MAAS": ("MAAS", "fontanella2018", "44 ISAC components"),
          "China": ("Chinese asthmatic children", "guo2026", "18 sIgE"),
          "Moscow": ("Moscow biochip", "smoldovskaya2018", "31 sIgE"),
          "URECA 5 y": ("URECA, age 5 y", "mcgowan2015", "10 sIgE"),
          "URECA 10 y": ("URECA, age 10 y", "mcgowan2015", "12 sIgE"),
          "EAT 3 y": ("EAT, age 3 y", "perkin2016", "19 skin prick tests"),
          "LEAP 5 y": ("LEAP, age 5 y", "dutoit2015", "15 sIgE")}


def t_hidden_ext():
    H = rd("ext_hidden_summary.csv")
    rows = []
    for _, r in H.iterrows():
        name, key, panel = HIDDEN[r.data]
        cite = f"~\\cite{{{key}}}" if key else ""
        rows.append(f"{name}{cite} & {panel} & {r.independent:.3f} & \\textbf{{{r.structured:.3f}}} & "
                    f"+{r.structured - r.independent:.3f}")
    table("tab_hidden_ext", "tab:hidden_ext",
          "Prediction of tests that were not ordered: AUROC for a positive result of a hidden test, predicted "
          "without the cross-reactivity structure (covariates and marginal prevalences only) and with the "
          "protein-family factor model of \\FLIP{}. NHANES: each panel hidden in turn (calibration split); other "
          "cohorts: one quarter of each person's results hidden at random.",
          RL + "X" + RL + "p{3.6cm}CCC",
          "\\textbf{Data} & \\textbf{Panel} & \\textbf{Without cross-reactivity} & \\textbf{With cross-reactivity} & "
          "\\textbf{Gain}", rows, size="\\footnotesize")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        TAB = os.path.abspath(sys.argv[1])
    os.makedirs(TAB, exist_ok=True)
    t_certified()
    t_posterior()
    t_ordering()
    t_tests_lit()
    t_assoc()
    t_hidden_ext()
