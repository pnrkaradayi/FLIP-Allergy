"""
a07_supplement.py
=================
Worked patient example and supplementary figures.

  tab_case.tex          one test patient followed through FLIP (MAR, repetition 0)
  figS1_latent_calib    predicted vs observed positivity of hidden analytes, by model
  figS2_flip_scores     distribution of the FLIP score by true flip status
  figS3_pany_calib      calibration of the remaining-risk estimate along the adaptive path

Run:  python a07_supplement.py
"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import (ALL20, BLOCK_COLS, BLOCK_NAMES, BLOCKS, CENS, NAME,  # noqa: E402
                   covariate_design, recorded_from_latent)
from a02_risk_decision import expand  # noqa: E402
from a04_figures import FIG, INK, MUTED, panel_label, save  # noqa: E402

RES = os.path.join(HERE, "results")
TAB = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "tables"))


def case_table():
    F = pickle.load(open(os.path.join(RES, "_bundle_y_allergic_disease.pkl"), "rb"))
    E = pickle.load(open(os.path.join(RES, "t1_examples_MAR.pkl"), "rb"))
    d = F["d"]
    te = d[d.split == "test"].reset_index(drop=True)
    tau, w, p = F["tau"], F["w"], len(ALL20)
    Mb, resid = E["Mb"], E["resid"]
    flip_rel = E["scores"]["FLIP"]
    Yrec = te[["lg_" + c for c in ALL20]].values
    st0 = te[["cens_" + c for c in ALL20]].values.astype(int)
    Xl = covariate_design(te.RIDAGEYR, te.male, te.RIDRETH1)
    Zfull = (te[F["feat"]].values - F["mu_sc"]) / F["sd_sc"]
    cov_part = Zfull[:, p:] @ w[p:]

    def posterior(i, Mrow):
        st = st0[i:i + 1].copy()
        M = expand(Mrow[None, :])
        st[M] = 2
        Z = F["lat"]["factor"].conditional_draws(Yrec[i:i + 1], st, Xl[i:i + 1], 24, 40, 20, seed=77)
        R = np.where(M[:, None, :], recorded_from_latent(Z), Yrec[i:i + 1][:, None, :])
        e = F["b0"] + ((R - F["mu_sc"][:p]) / F["sd_sc"][:p]) @ w[:p] + cov_part[i]
        return float((e >= tau).mean())

    # candidates: a decision that would flip, several unordered panels, sizeable FLIP
    cand = np.flatnonzero((Mb.sum(1) >= 3) & (flip_rel > 0.35) & (flip_rel < 0.9) & E["flip"])
    chosen = None
    for i in cand:
        j = int(np.where(np.isfinite(resid[i]), resid[i], np.inf).argmin())
        Mrow = Mb[i].copy()
        Mrow[j] = False
        p_after = posterior(i, Mrow)
        if (p_after >= 0.5) == bool(E["d_full"][i]):
            chosen = (i, j, p_after)
            break
    i, j, p_after = chosen
    p_now = posterior(i, Mb[i])
    row = te.iloc[i]
    lines = []
    sex = "male" if row.male == 1 else "female"
    lines.append(f"Covariates & {int(row.RIDAGEYR)} years, {sex}")
    lines.append(f"Model threshold & predicted risk $\\geq$ {1 / (1 + np.exp(-tau)):.3f} $\\Rightarrow$ atopy-directed management")
    obs, unord = [], []
    for k, b in enumerate(BLOCK_NAMES):
        cols = BLOCK_COLS[b]
        if Mb[i, k]:
            unord.append(b)
            continue
        vals = []
        for c in cols:
            v = 10 ** Yrec[i, c]
            vals.append(f"{NAME[ALL20[c]]} {'$<$' + ('2.0' if c == 19 else '0.35') if st0[i, c] == 1 else f'{v:.2f}'}")
        obs.append(f"{b}: " + ", ".join(vals))
    lines.append("Ordered panels (kU/L) & " + "; ".join(obs))
    lines.append("Unordered panels & " + ", ".join(unord))
    risk_imp = 1 / (1 + np.exp(-E["eta_hat"][i]))
    lines.append(f"Decision issued by single imputation & predicted risk {risk_imp:.3f} $\\Rightarrow$ "
                 f"{'manage' if E['d_hat'][i] else 'do not manage'}")
    lines.append(f"\\FLIP{{}} posterior & $\\Prob(\\eta\\geq\\tau\\mid x_o)$ = {p_now:.2f}; "
                 f"\\FLIP{{}} score of the issued decision = {flip_rel[i]:.2f}")
    voi = []
    for k, b in enumerate(BLOCK_NAMES):
        if Mb[i, k]:
            voi.append(f"{b} {resid[i, k]:.2f}")
    lines.append("Expected flip probability after ordering & " + ", ".join(voi) + f" $\\Rightarrow$ order \\textbf{{{BLOCK_NAMES[j]}}}")
    cols = BLOCK_COLS[BLOCK_NAMES[j]]
    rv = ", ".join(f"{NAME[ALL20[c]]} {'$<$0.35' if st0[i, c] == 1 else f'{10 ** Yrec[i, c]:.2f}'}" for c in cols)
    lines.append(f"Result of the recommended panel & {rv}")
    lines.append(f"Updated posterior & $\\Prob(\\eta\\geq\\tau)$ = {p_after:.2f} $\\Rightarrow$ "
                 f"{'manage' if p_after >= 0.5 else 'do not manage'}")
    risk_full = 1 / (1 + np.exp(-E["eta_full"][i]))
    lines.append(f"Decision with the complete panel & predicted risk {risk_full:.3f} $\\Rightarrow$ "
                 f"{'manage' if E['d_full'][i] else 'do not manage'}")
    body = "\n".join(l + " \\\\" for l in lines)
    tex = ("\\begin{table}[H]\n\\caption{A worked example: one test patient (MAR, first repetition) whose issued "
           "decision would have changed with complete testing. Values below the detection limit are shown as "
           "$<$0.35 kU/L. Expected flip probability after ordering: the censoring-aware preposterior of Equation~"
           "\\eqref{eq:voi} for each unordered panel; the panel with the smallest value is recommended.\\label{tab:case}}\n"
           "\\begin{adjustwidth}{-\\extralength}{0cm}\n\\small\n\\begin{tabularx}{\\fulllength}"
           "{>{\\raggedright\\arraybackslash}p{4.6cm}>{\\raggedright\\arraybackslash}X}\n\\toprule\n"
           "\\textbf{Step} & \\textbf{This patient} \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_case.tex"), "w", encoding="utf8").write(tex)
    info = dict(case_index=int(i), case_age=int(row.RIDAGEYR), case_sex=sex, case_p_now=p_now,
                case_flip=float(flip_rel[i]), case_panel=BLOCK_NAMES[j], case_p_after=p_after,
                case_risk_imp=float(risk_imp), case_risk_full=float(risk_full),
                case_unordered=unord, case_voi={b: float(resid[i, k]) for k, b in enumerate(BLOCK_NAMES) if Mb[i, k]})
    json.dump(info, open(os.path.join(RES, "case.json"), "w"), indent=1)
    print("case:", info)


def fig_latent_calibration():
    pr = pd.read_csv(os.path.join(RES, "sigma_prediction_by_analyte.csv"))
    pr = pr[pr.pred_positive_rate.notna()]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), sharex=True, sharey=True)
    labs = {"diagonal": "Covariates only", "naive": "Censoring ignored", "free": "Free covariance",
            "factor": "Cross-reactivity factors"}
    for ax, (m, lab) in zip(axs, labs.items()):
        q = pr[pr.model == m]
        ax.scatter(100 * q.obs_positive_rate, 100 * q.pred_positive_rate, s=12, color="#2a78d6", edgecolor="white", lw=0.4)
        ax.plot([0, 30], [0, 30], color=MUTED, ls="--", lw=0.8)
        ax.set_title(lab, loc="left", fontsize=8)
        ax.set_xlabel("Observed positive (%)", fontsize=7)
    axs[0].set_ylabel("Predicted positive (%)", fontsize=7)
    fig.tight_layout()
    save(fig, "figS1_latent_calibration")


def fig_flip_scores():
    E = pickle.load(open(os.path.join(RES, "t1_examples_MNAR.pkl"), "rb"))
    s, f = E["scores"]["FLIP"], E["flip"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6))
    bins = np.linspace(0, 1, 26)
    axs[0].hist(s[~f], bins=bins, color="#8a8984", alpha=0.8, label="decision unchanged")
    axs[0].hist(s[f], bins=bins, color="#2a78d6", alpha=0.85, label="decision would flip")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("FLIP score")
    axs[0].set_ylabel("Patients (log scale)")
    axs[0].legend(fontsize=7)
    panel_label(axs[0], "a")
    k = E["scores"]["Kim et al. 2026"]
    bins2 = np.linspace(k.min(), k.max(), 26)
    axs[1].hist(k[~f], bins=bins2, color="#8a8984", alpha=0.8)
    axs[1].hist(k[f], bins=bins2, color="#e34948", alpha=0.85)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Score of Kim et al.")
    panel_label(axs[1], "b")
    fig.tight_layout()
    save(fig, "figS2_flip_scores")


def fig_pany_calibration():
    S = pickle.load(open(os.path.join(RES, "t2_serum_sims.pkl"), "rb"))
    sim = S["sims"]["FLIP (total IgE + adaptive)"]
    ca = S["is_ca"]
    sens = S["sens"][S["evidx"]][ca]
    pa, fp = sim["p_any"][ca], sim["first_pos"][ca]
    xs, ys, ns = [], [], []
    edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.01]
    P, Y = [], []
    for s_ in range(pa.shape[1] - 1):
        act = np.isfinite(pa[:, s_]) & (fp >= s_)
        P.append(pa[act, s_])
        Y.append(sens[act])
    P, Y = np.concatenate(P), np.concatenate(Y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum() >= 20:
            xs.append(P[m].mean())
            ys.append(Y[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    ax.plot(xs, ys, marker="o", color="#2a78d6")
    ax.set_xlabel("Predicted P(any untested allergen positive)")
    ax.set_ylabel("Observed share sensitised")
    ax.set_title("Remaining-risk calibration along the path", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, "figS3_pany_calibration")
    json.dump(dict(pany_bins=list(zip(map(float, xs), map(float, ys), ns))),
              open(os.path.join(RES, "pany_calibration.json"), "w"), indent=1)


def strata_table():
    """Flip detection within strata of the number of unordered panels (first repetition)."""
    from sklearn.metrics import roc_auc_score
    rows, nums = [], {}
    for mech in ("MAR", "MNAR"):
        E = pickle.load(open(os.path.join(RES, f"t1_examples_{mech}.pkl"), "rb"))
        k = np.minimum(E["Mb"].sum(1), 5)
        f = E["flip"]
        for g in range(1, 6):
            m = k == g
            if m.sum() < 30 or not (0 < f[m].sum() < m.sum()):
                continue
            cells = []
            for nm in ("Kim et al. 2026", "Margin only", "Multiple imputation", "FLIP"):
                s = E["scores"][nm][m]
                a = roc_auc_score(f[m], s) if np.unique(s).size > 1 else 0.5
                cells.append(f"{a:.3f}")
                nums[f"strata_{mech}_{g}_{nm}"] = float(a)
            lab = f"{g}" if g < 5 else "$\\geq$5"
            rows.append(f"{mech} & {lab} & {int(m.sum())} & {100 * f[m].mean():.1f} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[H]\n\\caption{Flip detection within strata defined by the number of unordered panels "
           "(test split, first repetition of each mechanism; AUROC values in strata with few flipped decisions, notably a single unordered panel, rest on fewer than five events). Within a stratum the score of Kim et al.\\ can only "
           "distinguish which panels are missing, whereas \\FLIP{} also uses the patient's observed values and the "
           "distance from the threshold.\\label{tab:strata}}\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\small\n"
           "\\begin{tabularx}{\\fulllength}{ccccCCCC}\n\\toprule\n\\textbf{Mechanism} & \\textbf{Unordered panels} & "
           "\\textbf{$n$} & \\textbf{Flip rate, \\%} & \\textbf{Kim et al.} & \\textbf{Distance only} & \\textbf{MI} & "
           "\\textbf{FLIP} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
           "\\end{table}\n")
    open(os.path.join(TAB, "tab_strata.tex"), "w", encoding="utf8").write(tex)
    json.dump(nums, open(os.path.join(RES, "strata.json"), "w"), indent=1)
    print("strata:", {k: round(v, 3) for k, v in nums.items() if "FLIP" in k or "Kim" in k})


def extra_tables():
    """Top co-sensitisation pairs and per-analyte hidden-panel prediction."""
    pairs = pd.read_csv(os.path.join(RES, "corr_pairs.csv")).sort_values("latent", ascending=False).head(30)
    rows = [f"{r.a} & {r.b} & {r.naive:.3f} & {r.latent:.3f} & {r.factor:.3f} & {r.latent - r.naive:+.3f} \\\\"
            for r in pairs.itertuples()]
    tex = ("\\begin{table}[H]\n\\caption{The 30 most strongly co-sensitised analyte pairs (training split): Pearson "
           "correlation of reported log$_{10}$ values, censoring-corrected latent correlation (free covariance) and "
           "the correlation implied by the cross-reactivity factor model.\\label{tab:pairs}}\n\\small\n"
           "\\begin{tabularx}{\\textwidth}{llCCCC}\n\\toprule\n\\textbf{Analyte} & \\textbf{Analyte} & "
           "\\textbf{Reported} & \\textbf{Latent} & \\textbf{Factor model} & \\textbf{Change} \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_pairs.tex"), "w", encoding="utf8").write(tex)
    pr = pd.read_csv(os.path.join(RES, "sigma_prediction_by_analyte.csv"))
    q = pr[pr.pos_auroc.notna()].pivot_table(index=["block", "analyte"], columns="model", values="pos_auroc")
    order = [b for b in ["mite", "pet", "pest", "grass", "tree", "weed", "mould", "food"]]
    rows = []
    for b in order:
        for (blk, an), r in q.loc[[b]].iterrows():
            rows.append(f"{an} & {blk} & {r.diagonal:.3f} & {r.naive:.3f} & {r.factor_v1:.3f} & {r.factor:.3f} & "
                        f"{r.free:.3f} \\\\")
    tex = ("\\begin{table}[H]\n\\caption{AUROC for a positive result of each allergen when its whole panel is "
           "hidden and predicted from the other panels, total IgE and the covariates (calibration split, "
           "$n$ = 1,794), by conditional model.\\label{tab:hidden}}\n\\small\n"
           "\\begin{tabularx}{\\textwidth}{llCCCCC}\n\\toprule\n\\textbf{Allergen} & \\textbf{Panel} & "
           "\\textbf{Covariates only} & \\textbf{Censoring ignored} & \\textbf{Factors (prespec.)} & "
           "\\textbf{Factors (+PR-10/pan.)} & \\textbf{Free} \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    open(os.path.join(TAB, "tab_hidden.tex"), "w", encoding="utf8").write(tex)


def fig_guarantee_validity():
    L = pd.read_csv(os.path.join(RES, "t1_ltt_y_allergic_disease.csv"))
    L = L[(L.kind == "marginal") & (L.scale == 1.0)]
    scs = [("Kim et al. 2026", "#e34948"), ("Margin only", "#eb6834"), ("Multiple imputation", "#eda100"),
           ("FLIP", "#2a78d6"), ("FLIP (posterior decision)", "#0b4f9c")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for ax, mech in zip(axs, ("MAR", "MNAR")):
        for ai, a in enumerate((0.01, 0.02, 0.05)):
            for si, (nm, col) in enumerate(scs):
                v = L[(L.mech == mech) & (L.alpha == a) & (L.score == nm)].realised.dropna().values * 100
                x = ai * 6 + si
                if v.size:
                    ax.scatter(np.full(v.size, x) + np.random.default_rng(si).uniform(-0.15, 0.15, v.size), v,
                               s=8, color=col, alpha=0.7, edgecolor="none", label=nm if ai == 0 else None)
            ax.plot([ai * 6 - 0.5, ai * 6 + 4.5], [100 * a] * 2, color=INK, lw=1.0, ls="--")
        ax.set_xticks([2, 8, 14], ["α = 0.01", "α = 0.02", "α = 0.05"])
        ax.set_title(mech, loc="left", fontsize=8)
        ax.set_ylabel("Realised flips among accepted (%)" if mech == "MAR" else "")
    axs[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "figS4_guarantee_validity")


def fig_secondary():
    P = pd.read_csv(os.path.join(RES, "t1_perf_y_current_asthma.csv"))
    order = ["Fraction missing", "Kim et al. 2026", "Conditional SD only", "Margin only", "Multiple imputation",
             "FLIP v1 (Gaussian)", "FLIP, free covariance", "FLIP"]
    cols = ["#8a8984", "#e34948", "#e87ba4", "#eb6834", "#eda100", "#6250d6", "#1baf7a", "#2a78d6"]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for k, mech in enumerate(("MAR", "MNAR")):
        q = P[P.mech == mech]
        for i, (nm, c) in enumerate(zip(order, cols)):
            v = q[q.score == nm].auroc.values
            y = i + (k - 0.5) * 0.3
            ax.plot([v.min(), v.max()], [y, y], color=c, lw=1)
            ax.scatter(np.median(v), y, color=c, s=20, marker="o" if mech == "MAR" else "D", zorder=3)
    ax.set_yticks(range(len(order)), order, fontsize=6.5)
    ax.set_xlabel("Flip-detection AUROC, current asthma (median and range; ● MAR, ◆ MNAR)", fontsize=7)
    fig.tight_layout()
    save(fig, "figS5_secondary_outcome")


if __name__ == "__main__":
    extra_tables()
    fig_guarantee_validity()
    fig_secondary()
    strata_table()
    case_table()
    fig_latent_calibration()
    fig_flip_scores()
    fig_pany_calibration()
