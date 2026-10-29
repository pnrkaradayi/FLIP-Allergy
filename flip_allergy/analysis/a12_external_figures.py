"""
a12_external_figures.py
=======================
Figures and LaTeX tables of the comparison with published studies
(a08, a09, a10, a11 outputs) -> ../../manuscript/figures, ../../manuscript/tables,
and results/numbers_external.json for the text.

Run:  python a12_external_figures.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a04_figures import C, GRID, INK, MUTED, panel_label, save  # noqa: E402

RES = os.path.join(HERE, "results")
TAB = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "tables"))
NUM = {}
COL = {"design": "#1f1f1f", "fixed": "#eb6834", "seq": "#eda100", "flip": "#2a78d6", "diag": "#8a8984"}
EXT_ALLERGY = {"maas": "MAAS, asthma (Fontanella 2018)", "sosall": "SOS-ALL, atopic dermatitis (Zhakparov 2026)",
               "graz": "Graz, venom culprit (Arzt-Gradwohl 2026)",
               "moscow": "Moscow, respiratory allergy (Smoldovskaya 2018)"}
EXT_CLINICAL = {"heart": "Heart disease (Detrano 1989)", "thyroid": "Thyroid (Quinlan 1996)",
                "hepatitis": "Hepatitis (Diaconis & Efron 1983)", "pima": "Pima diabetes (Smith 1988)",
                "ckd": "Kidney disease (Qin 2020)", "support2": "SUPPORT (Knaus 1995)",
                "physionet2012": "PhysioNet ICU (Che 2018)"}
EXT_NAMES = {**EXT_ALLERGY, **EXT_CLINICAL}


def rd(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def op(df, method, alpha, col="mean_tests"):
    r = df[(df.method == method) & (np.isclose(df.alpha, alpha))]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ------------------------------------------------------------------ figure 11: tests needed
def fig_tests():
    lf = rd("lit_flip.csv")
    t2s, t2p = rd("t2_serum_operating_points.csv"), rd("t2_spt_operating_points.csv")
    rows = []
    rows.append(dict(label="Salo 2014 — NHANES 2005–06, 19 sIgE", design=19,
                     flip=op(t2s, "FLIP (total IgE + adaptive)", 0.05),       # factor structure, as elsewhere
                     seq=op(t2s, "Sequential, greedy order, stop at first positive", 0.05),
                     fixed=op(t2s, "Fixed panel: greedy set cover", 0.05)))
    k = lf[(lf.study == "Salo 2014 (1-5 y)") & (lf.method == "FLIP adaptive") & lf.target.str.contains("0.05")]
    rows.append(dict(label="Salo 2014 — ages 1–5 y, 9 sIgE", design=9, flip=float(k.flip_tests.iloc[0]),
                     seq=np.nan, fixed=np.nan))
    for tgt, des in (("92.8", 6), ("99.2", 11)):
        g = lf[(lf.study == "Gergen 2009") & (lf.method == "FLIP adaptive") & lf.target.str.contains(tgt)
               & lf.kind.str.startswith("read-off")]
        rows.append(dict(label=f"Gergen 2009 — {tgt}% of atopic persons, 15 sIgE", design=des,
                         flip=float(g.flip_tests.iloc[0]), seq=np.nan, fixed=float(des)))
    rows.append(dict(label="Arbes 2005 — NHANES III, 10 skin prick tests", design=10,
                     flip=op(t2p, "FLIP (adaptive, in-domain)", 0.05),
                     seq=op(t2p, "Sequential, greedy order, stop at first positive", 0.05),
                     fixed=op(t2p, "Fixed panel: greedy set cover", 0.05)))
    for key, lab in (("acc", "Martinroche 2026 — Allergen Chip Challenge, {n} ISAC components"),
                     ("maas", "Fontanella 2018 — MAAS, {n} ISAC components"),
                     ("guo", "Guo 2026 — Chinese asthmatic children, {n} sIgE"),
                     ("moscow", "Smoldovskaya 2018 — Moscow biochip, {n} sIgE")):
        e = rd(f"ext_testing_{key}.csv")
        if e is None:
            continue
        rows.append(dict(label=lab.format(n=int(e.panel_size.iloc[0])), design=int(e.panel_size.iloc[0]),
                         flip=op(e, "FLIP adaptive", 0.05),
                         seq=op(e, "Sequential, greedy order, stop at first positive", 0.05),
                         fixed=op(e, "Fixed panel: greedy set cover", 0.05)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "ext_fig_tests.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(R) + 1.1))
    yy = np.arange(len(R))[::-1]
    for (_, r), y in zip(R.iterrows(), yy):
        ax.plot([r.flip, r.design], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(R.design, yy, marker="s", s=46, facecolor="white", edgecolor=COL["design"], lw=1.3, zorder=3,
               label="Tests used in the study design")
    ax.scatter(R.fixed, yy, marker="D", s=34, color=COL["fixed"], zorder=3, label="Fixed panel (greedy set cover)")
    ax.scatter(R.seq, yy, marker="^", s=40, color=COL["seq"], zorder=3, label="Sequential, fixed order")
    ax.scatter(R.flip, yy, marker="o", s=54, color=COL["flip"], edgecolor="white", lw=0.8, zorder=4,
               label="FLIP adaptive testing")
    for (_, r), y in zip(R.iterrows(), yy):
        ax.annotate(f"{r.flip:.1f}", (r.flip, y), xytext=(-7, -2.5), textcoords="offset points", ha="right",
                    fontsize=7, color=C["FLIP"])
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50", "100"])
    ax.set_yticks(yy)
    ax.set_yticklabels(R.label, fontsize=7.6)
    ax.set_xlabel("Allergen tests per person (log scale)")
    ax.set_ylim(-0.6, len(R) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7, ncol=4, frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, "fig11_literature_tests")
    for _, r in R.iterrows():
        NUM[f"tests::{r.label}"] = dict(design=r.design, flip=r.flip, seq=r.seq, fixed=r.fixed)
    return R


# ------------------------------------------------------------------ figure 12: flip detection etc.
def load_ext_flip():
    out = []
    for key in EXT_NAMES:
        e = rd(f"ext_flip_{key}.csv")
        if e is not None:
            out.append(e)
    return pd.concat(out) if out else None


def fig_flip(names, fname, include_nhanes):
    E = load_ext_flip()
    perf = rd("t1_perf_y_allergic_disease.csv")
    ltt = rd("t1_ltt_y_allergic_disease.csv")
    methods = ["FLIP", "Multiple imputation", "Margin only", "Kim et al. 2026"]
    lab = {"FLIP": "FLIP", "Multiple imputation": "Multiple imputation", "Margin only": "Distance to threshold",
           "Kim et al. 2026": "Kim et al. 2026"}
    rows = []
    for mname in (methods if include_nhanes else []):
        rows.append(dict(dataset="NHANES 2005–06, allergic disease (this study)", method=mname,
                         auroc=perf[(perf.mech == "MAR") & (perf.score == mname)].auroc.mean(),
                         cov10=np.nan,
                         cov05=ltt[(ltt.mech == "MAR") & (ltt.score == mname) & (ltt.kind == "marginal")
                                   & np.isclose(ltt.alpha, 0.05)].coverage.mean()))
    for key, nm in names.items():
        e = E[E.dataset == key] if E is not None else None
        if e is None or e.empty:
            continue
        for mname in methods:
            g = e[(e.mech == "MAR") & (e.method == mname)]
            rows.append(dict(dataset=nm, method=mname,
                             auroc=g[g.metric == "flip AUROC (pooled)"].value.mean(),
                             cov10=g[g.metric == "LTT coverage (alpha=0.1)"].value.mean(),
                             cov05=g[g.metric == "LTT coverage (alpha=0.05)"].value.mean()))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(RES, f"ext_{fname}.csv"), index=False)
    dsets = list(dict.fromkeys(F.dataset))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 0.36 * len(dsets) + 1.4), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    yy = np.arange(len(dsets))[::-1]
    off = {"FLIP": 0.22, "Multiple imputation": 0.07, "Margin only": -0.08, "Kim et al. 2026": -0.23}
    for ax, col, title in ((axs[0], "auroc", "a  Detecting decisions that\n    complete testing would change"),
                           (axs[1], "cov10", "b  Patients certified at\n    ≤ 10% error (LTT)"),
                           (axs[2], "cov05", "c  Patients certified at\n    ≤ 5% error (LTT)")):
        for mname in methods:
            g = F[F.method == mname].set_index("dataset").reindex(dsets)
            ax.scatter(g[col], yy + off[mname], s=22 if mname != "FLIP" else 34, color=C[mname],
                       edgecolor="white", lw=0.5, zorder=3, label=lab[mname])
        ax.set_title(title, loc="left", fontsize=8.2)
        ax.grid(axis="y", visible=False)
        for y in yy:
            ax.axhline(y - 0.5, color=GRID, lw=0.5)
    axs[0].set_xlim(0.45, 1.02)
    axs[0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axs[0].set_xticklabels(["0.5", "", "0.7", "", "0.9", ""])
    axs[0].axvline(0.5, color=MUTED, lw=0.7, ls=":")
    for ax in axs[1:]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "50%", "", "100%"])
    axs[0].set_xlabel("AUROC (pooled over repetitions)")
    axs[1].set_xlabel("Coverage")
    axs[2].set_xlabel("Coverage")
    axs[0].set_yticks(yy)
    axs[0].set_yticklabels(dsets, fontsize=7.2)
    axs[0].legend(loc="lower left", fontsize=6.6, bbox_to_anchor=(0.0, -0.32 - 0.004 * len(dsets)), ncol=4,
                  frameon=False)
    fig.subplots_adjust(wspace=0.12)
    save(fig, fname)
    return F


def decision_table():
    E = load_ext_flip()
    if E is None:
        return None
    rows = []
    for key, nm in EXT_NAMES.items():
        e = E[(E.dataset == key) & (E.metric == "decision error vs complete testing")]
        if e.empty:
            continue
        for mech in ("MAR", "MNAR"):
            g = e[e.mech == mech].set_index("method").value
            rows.append(dict(dataset=nm, mech=mech, missforest=g.get("imputation: missforest"),
                             mean=g.get("imputation: mean"), knn=g.get("imputation: knn"),
                             mice=g.get("imputation: mice"), median=g.get("imputation: median"),
                             flip=g.get("FLIP posterior"),
                             best_imputer=min(g.get(k) for k in g.index if k.startswith("imputation"))))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "ext_decision_errors.csv"), index=False)
    return D


# ------------------------------------------------------------------ figure 13: natural missingness etc.
def hidden_summary():
    rows = []
    s = rd("sigma_prediction_summary.csv")
    if s is not None:
        s = s.set_index(s.columns[0])
        rows.append(dict(data="NHANES", structured=float(s.loc["factor", "pos_auroc"]),
                         independent=float(s.loc["diagonal", "pos_auroc"])))
    for key, lab in (("acc", "Chip"), ("maas", "MAAS"), ("guo", "China"), ("moscow", "Moscow")):
        h = rd(f"ext_hidden_{key}.csv")
        if h is None:
            continue
        p = h[h.component == "(pooled)"].set_index("model").auroc
        rows.append(dict(data=lab, structured=float(p.get("FLIP adaptive")),
                         independent=float(p.get("Ablation: no cross-reactivity (diagonal)"))))
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(RES, "ext_hidden_summary.csv"), index=False)
    return H


def fig_natural():
    gz, hop = rd("ext_graz_culprit.csv"), rd("lit_hoppin.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    ax = axs[0]
    H = hidden_summary()
    if len(H):
        x = np.arange(len(H))
        ax.bar(x - 0.19, H.independent, 0.36, color="#c9c8c3", label="Independent")
        ax.bar(x + 0.19, H.structured, 0.36, color=C["FLIP"], label="Cross-reactivity")
        for xi, v in zip(x, H.structured):
            ax.text(xi + 0.19, v + 0.01, f"{v:.2f}", ha="center", fontsize=6, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(H.data, fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(0.4, 1.12)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel("AUROC, positive hidden test")
        ax.legend(fontsize=6, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8)
        ax.grid(axis="x", visible=False)
    ax.set_title("a  Predicting tests that were\n    not ordered", loc="left", fontsize=8.2)
    ax = axs[1]
    gp = rd("ext_graz_predictions.csv")
    if gz is not None and gp is not None:
        conf = np.maximum(gp.p_bee, 1 - gp.p_bee).values
        pred = (gp.p_bee >= 0.5).astype(int).values
        o = np.argsort(-conf, kind="stable")
        err = np.cumsum(pred[o] != gp.y.values[o]) / np.arange(1, len(o) + 1)
        cov = np.arange(1, len(o) + 1) / len(o)
        ax.plot(cov, 100 * err, color=C["FLIP"], lw=1.6, label="Decision model + FLIP (cross-fitted)")
        rr = gz[gz.rule.str.startswith("Ratio")]
        ax.scatter(rr.decided, 100 * rr.errors_among_decided, marker="s", s=40, color=C["Kim et al. 2026"],
                   zorder=3, label="Ratio rules (published)")
        ax.set_xlabel("Patients with a culprit assigned")
        ax.set_ylabel("Wrong culprit among them, %")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, max(2.5, 100 * err[-1] * 1.15))
        ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("b  Bee or vespid venom:\n    decided fraction vs errors", loc="left", fontsize=8.2)
    ax = axs[2]
    if hop is not None:
        q = hop[hop.predictor.str.startswith("questionnaire")]
        fl = hop[hop.predictor.str.startswith("FLIP")]
        ax.scatter(np.zeros(len(q)), q.acc, color="#8a8984", s=18, label="Questionnaire items (Hoppin 2011)")
        ax.plot(fl.tests, fl.acc, "-o", color=C["FLIP"], ms=4, label="FLIP, ≤ 1, 2, 3 sIgE tests")
        ax.set_xlabel("sIgE tests per person (mean)")
        ax.set_ylabel("Accuracy for atopy")
        ax.set_ylim(0.5, 0.95)
        ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("c  Atopy: questionnaire\n    vs a few adaptive tests", loc="left", fontsize=8.2)
    fig.tight_layout()
    save(fig, "fig13_external_natural")


def fig_heart():
    hd = rd("ext_heart_deployment.csv")
    if hd is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(hd))
    w = 0.2
    ax.bar(x - 1.5 * w, hd.published_cdf / 100, w, color="#c9c8c3", label="Detrano 1989 (published)")
    ax.bar(x - 0.5 * w, hd.acc_missforest, w, color=C["Margin only"], label="Model + imputation")
    ax.bar(x + 0.5 * w, hd.acc_flip_posterior, w, color=C["FLIP"], label="FLIP posterior decision")
    ax.bar(x + 1.5 * w, hd.acc_accepted, w, color=C["FLIP (posterior decision)"], label="FLIP, accepted patients")
    for xi, (_, r) in zip(x, hd.iterrows()):
        ax.text(xi + 1.5 * w, r.acc_accepted + 0.01, f"{100 * r.accepted:.0f}%", ha="center", fontsize=6,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Long Beach VA", "Long Beach") for s in hd.site], fontsize=7)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Accuracy against angiography")
    ax.legend(fontsize=6, loc="upper right", ncol=1)
    ax.grid(axis="x", visible=False)
    save(fig, "figA6_heart_deployment")


# ------------------------------------------------------------------ tables
def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("≥", r"$\geq$")
            .replace("≤", r"$\leq$").replace("–", "--"))


def fmt(v):
    if isinstance(v, str):
        return esc(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if abs(v) >= 10:
        return f"{v:.1f}"
    s = f"{v:.3g}"
    return s if "." in s else f"{v:.1f}"


REPRO = [  # (group, citation key, study label, data label, [(source csv, study, quantity substring, short label)])
    ("Same NHANES data", "salo2014", "Salo 2014", "NHANES 2005--06, 19 sIgE",
     [("lit", "Salo 2014", "age >= 6", "sensitised, $\\geq$6 y, \\%"), ("lit", "Salo 2014", "1-5 y", "sensitised, 1--5 y, \\%")]),
    ("Same NHANES data", "gergen2009", "Gergen 2009", "NHANES 2005--06, 15 sIgE",
     [("lit", "Gergen 2009", "6-allergen", "atopic found by 6 tests, \\%"),
      ("lit", "Gergen 2009", "11-allergen", "atopic found by 11 tests, \\%")]),
    ("Same NHANES data", "hoppin2011", "Hoppin 2011", "NHANES 2005--06, questionnaire",
     [("lit", "Hoppin 2011", "PPV of hay fever", "PPV of hay fever, \\%"),
      ("lit", "Hoppin 2011", "NPV of any", "NPV of any condition, \\%")]),
    ("Same NHANES data", "zhu2022", "Zhu 2022", "NHANES 2005--06",
     [("lit", "Zhu 2022", "Spearman", "Spearman $\\rho$, two mite species")]),
    ("Same NHANES data", "mcgowan2019", "McGowan 2019", "NHANES 2005--06, 6--19 y",
     [("lit", "McGowan 2019", "cockroach", "shrimp--cockroach correlation"),
      ("lit", "McGowan 2019", "dust-mite", "shrimp--mite correlation")]),
    ("Same NHANES data", "arbes2005", "Arbes 2005", "NHANES III, 10 SPT",
     [("lit", "Arbes 2005", ">=1 positive", "$\\geq$1 positive skin test, \\%")]),
    ("Open allergy data", "martinroche2026", "Martinroche 2026", "Allergen Chip Challenge",
     [("ext", "Martinroche 2026", "sensitised", "sensitised, \\%")]),
    ("Open allergy data", "fontanella2018", "Fontanella 2018", "MAAS, ISAC, 11 y",
     [("ext", "Fontanella 2018", "sensitised", "sensitised, \\%"),
      ("ext", "Fontanella 2018", "AUROC", "AUROC for asthma, penalised LR")]),
    ("Open allergy data", "arztgradwohl2026", "Arzt-Gradwohl 2026", "Graz venom",
     [("ext", "Arzt-Gradwohl 2026", "bee venom sIgE, dual", "bee sIgE sensitivity, dual cut-off, \\%"),
      ("ext", "Arzt-Gradwohl 2026", "vespid/bee >= 2.68", "vespid/bee ratio rule sensitivity, \\%"),
      ("ext", "Arzt-Gradwohl 2026", "correctly identified by the ratio rules, vespid", "culprit identified, vespid, \\%")]),
    ("Open allergy data", "guo2026", "Guo 2026", "Chinese asthmatic children",
     [("ext", "Guo 2026", "D2", "\\emph{D.\\,farinae}, South, \\%"), ("ext", "Guo 2026", "w6", "mugwort, North, \\%")]),
    ("Open allergy data", "smoldovskaya2018", "Smoldovskaya 2018", "Moscow biochip",
     [("ext", "Smoldovskaya 2018", "Dog dander sIgE, Asthma vs Upper", "OR dog, asthma vs upper airway"),
      ("ext", "Smoldovskaya 2018", "Cat dander sIgE, Asthma vs Derm", "OR cat, asthma vs dermatitis")]),
    ("Open allergy data", "zhakparov2026", "Zhakparov 2026", "SOS-ALL",
     [("ext", "Zhakparov 2026", "random-forest", "accuracy for atopic dermatitis$^{a}$")]),
    ("Clinical data", "detrano1989", "Detrano 1989", "Heart disease, 3 external centres",
     [("ext", "Detrano 1989", "accuracy", "accuracy, Hungary / Swiss / Long Beach, \\%")]),
    ("Clinical data", "quinlan1996", "Quinlan 1996", "Thyroid (allhypo)",
     [("ext", "Quinlan 1996", "error", "tree error, 5 classes, \\%$^{b}$")]),
    ("Clinical data", "diaconis1983", "Diaconis \\& Efron 1983", "Hepatitis",
     [("ext", "Diaconis", "accuracy", "accuracy, \\%")]),
    ("Clinical data", "smith1988", "Smith 1988", "Pima diabetes",
     [("ext", "Smith 1988", "crossover", "sensitivity = specificity")]),
    ("Clinical data", "qin2020", "Qin 2020", "Chronic kidney disease",
     [("ext", "Qin 2020", "random-forest", "random-forest accuracy, \\%")]),
    ("Clinical data", "knaus1995", "Knaus 1995", "SUPPORT",
     [("ext", "Knaus 1995", "AUROC", "AUROC, 180-day mortality")]),
    ("Clinical data", "che2018", "Che 2018", "PhysioNet 2012",
     [("ext", "Che 2018", "LR-Mean", "AUROC, logistic regression"),
      ("ext", "Che 2018", "GRU-D", "AUROC, best model$^{c}$")]),
]


def table_reproduction(groups, fname, label, caption):
    src = {"lit": rd("lit_reproduction.csv"), "ext": rd("ext_reproduction.csv")}
    lines, group = [], None
    for g, key, study, data, items in REPRO:
        if g not in groups:
            continue
        if g != group:
            lines.append(r"\midrule" + "\n" + rf"\multicolumn{{4}}{{l}}{{\textit{{{g}}}}} \\")
            group = g
        for k, (s, st, q, lab) in enumerate(items):
            df = src[s]
            r = df[(df.study.str.startswith(st)) & df.quantity.str.contains(q, regex=False)]
            if r.empty:
                continue
            if len(r) > 1:                      # several sites in one row
                pub = " / ".join(fmt(v) for v in r.published)
                rep = " / ".join(fmt(v) for v in r.reproduced)
            else:
                pub, rep = fmt(r.published.iloc[0]), fmt(r.reproduced.iloc[0])
            first = rf"{study}~\cite{{{key}}}" if k == 0 else ""
            dlab = data if k == 0 else ""
            lines.append(f"{first} & {dlab} & {lab} & {pub} $\\rightarrow$ {rep} \\\\")
    body = "\n".join(lines)
    tex = (r"\begin{table}[H]" "\n" rf"\caption{{{caption}\label{{{label}}}}}" "\n"
           r"\begin{adjustwidth}{-\extralength}{0cm}" "\n" r"\scriptsize" "\n"
           r"\begin{tabularx}{\fulllength}{p{3.3cm}p{3.6cm}Xp{3.3cm}}" "\n" r"\toprule" "\n"
           r"\textbf{Study} & \textbf{Data} & \textbf{Reported quantity} & \textbf{Published $\rightarrow$ ours} \\"
           "\n" + body + "\n" r"\bottomrule" "\n" r"\end{tabularx}" "\n" r"\end{adjustwidth}" "\n"
           r"\end{table}" "\n")
    with open(os.path.join(TAB, fname), "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("  wrote", fname)


def table_external_flip(names, fname, label, caption):
    D = decision_table()
    E = load_ext_flip()
    rows = []
    for key, nm in names.items():
        e = E[E.dataset == key] if E is not None else None
        if e is None or e.empty:
            continue
        g = e[e.mech == "MAR"]
        auc = lambda m: g[(g.metric == "flip AUROC (pooled)") & (g.method == m)].value.mean()
        cov = lambda m, a: g[(g.metric == f"LTT coverage (alpha={a})") & (g.method == m)].value.mean()
        dd = D[(D.dataset == nm) & (D.mech == "MAR")].iloc[0]
        rows.append([nm, int(g.n_test_pooled.iloc[0] / g.reps.iloc[0]), g.flip_rate.iloc[0], auc("FLIP"),
                     auc("Multiple imputation"), auc("Margin only"), auc("Kim et al. 2026"),
                     cov("FLIP", 0.05), cov("Kim et al. 2026", 0.05), cov("FLIP", 0.1), cov("Kim et al. 2026", 0.1),
                     dd.missforest, dd.flip])
    def bold_best(vals, fmtstr, best=max):
        b = best(v for v in vals if np.isfinite(v))
        return [(r"\textbf{" + fmtstr.format(v) + "}") if np.isclose(v, b) else fmtstr.format(v) for v in vals]

    lines = []
    for r in rows:
        a = bold_best([r[3], r[4], r[5], r[6]], "{:.3f}")
        c5 = bold_best([100 * r[7], 100 * r[8]], "{:.0f}")
        c10 = bold_best([100 * r[9], 100 * r[10]], "{:.0f}")
        e = bold_best([100 * r[11], 100 * r[12]], "{:.1f}", best=min)
        lines.append(f"{esc(r[0])} & {r[1]} & {100 * r[2]:.1f} & " + " & ".join(a + c5 + c10 + e) + r" \\")
    body = "\n".join(lines)
    tex = (r"\begin{table}[H]" "\n" rf"\caption{{{caption}\label{{{label}}}}}" "\n"
           r"\begin{adjustwidth}{-\extralength}{0cm}" "\n" r"\scriptsize" "\n"
           r"\begin{tabularx}{\fulllength}{Xrrrrrrrrrrrr}" "\n" r"\toprule" "\n"
           r" & & & \multicolumn{4}{c}{\textbf{Flip AUROC}} & \multicolumn{2}{c}{\textbf{Certified $\leq$5\%}} & "
           r"\multicolumn{2}{c}{\textbf{Certified $\leq$10\%}} & \multicolumn{2}{c}{\textbf{Error, \%}} \\" "\n"
           r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}" "\n"
           r"\textbf{Data (study)} & \textbf{n} & \textbf{Flips \%} & \textbf{FLIP} & \textbf{MI} & "
           r"\textbf{Margin} & \textbf{Kim} & \textbf{FLIP} & \textbf{Kim} & \textbf{FLIP} & \textbf{Kim} & "
           r"\textbf{Imput.} & \textbf{FLIP} \\" "\n"
           r"\midrule" "\n" + body + "\n" r"\bottomrule" "\n" r"\end{tabularx}" "\n" r"\end{adjustwidth}" "\n"
           r"\end{table}" "\n")
    with open(os.path.join(TAB, fname), "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("  wrote", fname)
    NUM[f"rows::{fname}"] = rows


FLIP_CAPTION = (r"The protocol of Task~1 run unchanged on the public data of {what} (missing at random, 30 "
                r"repetitions): test records per repetition, flip rate of the imputed decision, AUROC for detecting "
                r"the flipped decisions (MI, multiple imputation; Margin, distance to the threshold; Kim, the score "
                r"of~\cite{{kim2026}}), share of patients certified (\%) at a flip rate $\leq$5\% and $\leq$10\% ($\delta=0.1$), and "
                r"decision error against complete testing of the MissForest-type imputed decision and of the "
                r"\FLIP{{}} posterior-majority decision. Best value per row in bold.")

if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    R = fig_tests()
    fig_flip(EXT_ALLERGY, "fig12_external_flip", include_nhanes=True)
    fig_flip(EXT_CLINICAL, "figA7_clinical_flip", include_nhanes=False)
    fig_natural()
    fig_heart()
    table_reproduction(("Same NHANES data", "Open allergy data"), "tab_lit_repro.tex", "tab:lit_repro",
                       r"Headline numbers of the twelve published allergy studies, reproduced from the same public "
                       r"data with each study's definitions (published $\rightarrow$ reproduced). Survey weights are "
                       r"applied for NHANES, as in the original reports. $^{a}$The published model used questionnaire "
                       r"and antibody data, ours antibodies only.")
    table_reproduction(("Clinical data",), "tab_clin_repro.tex", "tab:clin_repro",
                       r"Headline numbers of the seven clinical studies outside allergy, reproduced from the same "
                       r"public data (published $\rightarrow$ reproduced). $^{b}$Published C4.5, ours CART. "
                       r"$^{c}$Published GRU-D recurrent network, ours random forest on 48-hour summaries.")
    table_external_flip(EXT_ALLERGY, "tab_ext_flip.tex", "tab:ext_flip",
                        FLIP_CAPTION.format(what="four published allergy studies with a clinical outcome"))
    table_external_flip(EXT_CLINICAL, "tab_clin_flip.tex", "tab:clin_flip",
                        FLIP_CAPTION.format(what="seven clinical studies outside allergy"))
    with open(os.path.join(RES, "numbers_external.json"), "w") as fh:
        json.dump(NUM, fh, indent=1, default=float)
    print(R.round(2).to_string(index=False))
