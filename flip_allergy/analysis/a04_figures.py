"""
a04_figures.py
==============
Publication figures for the manuscript (PDF + PNG, 300 dpi), written to
../../manuscript/figures/.

Run:  python a04_figures.py [which]      which in {structure, task1, task2, all}
"""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flipx import ALL20, BLOCK_COLS, BLOCK_NAMES, FAMILIES, NAME  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figures"))
os.makedirs(FIG, exist_ok=True)

INK, MUTED, GRID = "#1f1f1f", "#5c5b57", "#e4e3de"
C = {"FLIP": "#2a78d6", "FLIP, free covariance": "#1baf7a", "FLIP v1 (Gaussian)": "#6250d6",
     "Multiple imputation": "#eda100", "Margin only": "#eb6834", "Kim et al. 2026": "#e34948",
     "Fraction missing": "#8a8984", "Conditional SD only": "#e87ba4",
     "FLIP (posterior decision)": "#0b4f9c"}
LABEL = {"FLIP": "FLIP (censored, cross-reactivity)", "FLIP, free covariance": "FLIP (censored, free cov.)",
         "FLIP v1 (Gaussian)": "FLIP v1 (Gaussian, no censoring)", "Multiple imputation": "Multiple imputation",
         "Margin only": "Distance to threshold only", "Kim et al. 2026": "Kim et al. 2026",
         "Fraction missing": "Fraction of panel missing", "Conditional SD only": "Conditional SD only",
         "FLIP (posterior decision)": "FLIP (posterior-majority decision)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  wrote", name)


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


# ============================================================== structure
def fig_structure():
    pairs = pd.read_csv(os.path.join(RES, "corr_pairs.csv"))
    R = pd.read_csv(os.path.join(RES, "corr_latent_free.csv"), index_col=0)
    pr = pd.read_csv(os.path.join(RES, "sigma_prediction_by_analyte.csv"))
    tp = pd.read_csv(os.path.join(RES, "transport_pairs.csv"))

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.6))
    ax = axs[0, 0]
    ax.scatter(pairs.naive, pairs.latent, s=11, color="#2a78d6", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("Pearson correlation of reported values")
    ax.set_ylabel("Censoring-corrected latent correlation")
    ax.set_title("Censoring correction raises all 190 pairs", loc="left")
    offs = [(-60, -2), (-50, -14), (-45, -26), (-10, -40)]
    for (_, r), (dx, dy) in zip(pairs.sort_values("latent", ascending=False).head(4).iterrows(), offs):
        ax.annotate(f"{r.a}–{r.b}", (r.naive, r.latent), fontsize=6.2, color=MUTED, ha="right",
                    xytext=(dx, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
    panel_label(ax, "a")

    ax = axs[0, 1]
    order = ["D. farinae", "D. pteronyssinus", "Cockroach", "Shrimp", "Cat", "Dog", "Mouse", "Rat",
             "Rye grass", "Bermuda grass", "Oak", "Birch", "Ragweed", "Russian thistle", "Peanut",
             "Alternaria", "Aspergillus", "Egg", "Milk", "Total IgE"]
    M = R.loc[order, order].values
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#9cc3ef", "#2a78d6", "#0b2f5c"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(20), order, rotation=90, fontsize=6)
    ax.set_yticks(range(20), order, fontsize=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("latent correlation", fontsize=7)
    ax.set_title("Latent co-sensitisation structure", loc="left")
    panel_label(ax, "b")

    ax = axs[1, 0]
    mods = [("diagonal", "Covariates only (no cross-reactivity)", "#8a8984"),
            ("naive", "Gaussian, censoring ignored", "#6250d6"),
            ("factor_v1", "Cross-reactivity factors (prespecified)", "#1baf7a"),
            ("factor", "Cross-reactivity factors (+PR-10/panallergens)", "#2a78d6"),
            ("free", "Free latent covariance", "#eda100")]
    q = pr[pr.pos_auroc.notna()].groupby(["block", "model"]).pos_auroc.mean().unstack()
    blocks = [b for b in BLOCK_NAMES if b != "totige"]
    x = np.arange(len(blocks))
    wd = 0.16
    for i, (m, lab, col) in enumerate(mods):
        ax.bar(x + (i - 2) * wd, q.loc[blocks, m], width=wd, color=col, label=lab, edgecolor="white", lw=0.5)
    ax.set_xticks(x, blocks, rotation=35, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC for a positive result\nin the hidden panel")
    ax.set_title("Predicting a panel that was not ordered", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2, fontsize=6.3)
    panel_label(ax, "c")

    ax = axs[1, 1]
    ax.scatter(tp.serum_sIgE, tp.skin_prick, s=16, color="#2a78d6", edgecolor="white", lw=0.4)
    lim = (min(tp.min(numeric_only=True)) - 0.05, 1.0)
    ax.plot(lim, lim, color=MUTED, lw=0.9, ls="--")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    from scipy.stats import spearmanr
    rho = spearmanr(tp.serum_sIgE, tp.skin_prick).statistic
    ax.text(0.04, 0.95, f"45 pairs, Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top", color=INK, fontsize=7.5)
    ax.set_xlabel("Serum sIgE, NHANES 2005–2006")
    ax.set_ylabel("Skin prick test, NHANES III")
    ax.set_title("Tetrachoric correlations transport across assays", loc="left")
    panel_label(ax, "d")
    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    save(fig, "fig2_structure")

    # factor loadings heatmap
    L = pd.read_csv(os.path.join(RES, "factor_loadings.csv"), index_col=0)
    fams = list(FAMILIES)
    Lm = L.loc[order, fams].values
    Lm = Lm * np.where(Lm.sum(0) < 0, -1.0, 1.0)[None, :]   # factor signs are arbitrary
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap2 = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f7f7f5", "#2a78d6"])
    vmax = np.nanmax(np.abs(Lm))
    im = ax.imshow(np.where(Lm == 0, np.nan, Lm), cmap=cmap2, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(Lm.shape[0]):
        for j in range(Lm.shape[1]):
            if Lm[i, j] != 0:
                ax.text(j, i, f"{Lm[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if abs(Lm[i, j]) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(fams)), fams, rotation=35, ha="right", fontsize=6.8)
    ax.set_yticks(range(20), order, fontsize=6.8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("standardised loading", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Cross-reactivity factor model: loadings (blank = structurally zero)", loc="left")
    fig.tight_layout()
    save(fig, "fig3_factor_loadings")


# ============================================================== task 1
SC_ORDER = ["Fraction missing", "Kim et al. 2026", "Conditional SD only", "Margin only",
            "Multiple imputation", "FLIP v1 (Gaussian)", "FLIP, free covariance", "FLIP"]


def fig_task1():
    P = pd.read_csv(os.path.join(RES, "t1_perf_y_allergic_disease.csv"))
    L = pd.read_csv(os.path.join(RES, "t1_ltt_y_allergic_disease.csv"))
    A = pd.read_csv(os.path.join(RES, "t1_acq_y_allergic_disease.csv"))
    E = {m: pickle.load(open(os.path.join(RES, f"t1_examples_{m}.pkl"), "rb")) for m in ("MAR", "MNAR")}

    # ---- Figure 4: detection, risk-coverage, calibration
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.6))
    ax = axs[0, 0]
    ex = E["MNAR"]
    for nm in SC_ORDER:
        fpr, tpr, _ = roc_curve(ex["flip"], ex["scores"][nm])
        ax.plot(fpr, tpr, color=C[nm], lw=2.0 if nm == "FLIP" else 1.1, label=LABEL[nm],
                zorder=3 if nm == "FLIP" else 2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Share of flipped decisions detected")
    ax.set_title("Detecting flipped decisions (MNAR)", loc="left")
    ax.legend(loc="lower right", fontsize=6.2)
    panel_label(ax, "a")

    ax = axs[0, 1]
    for nm in SC_ORDER:
        s = ex["scores"][nm]
        order = np.lexsort((np.random.default_rng(0).random(len(s)), s))
        f = ex["flip"][order].astype(float)
        cov = np.arange(1, len(s) + 1) / len(s)
        risk = np.cumsum(f) / np.arange(1, len(s) + 1)
        ax.plot(cov[30:], 100 * risk[30:], color=C[nm], lw=2.0 if nm == "FLIP" else 1.1)
    ax.set_xlabel("Share of patients accepted (lowest score first)")
    ax.set_ylabel("Flipped decisions among accepted (%)")
    ax.set_title("Risk–coverage", loc="left")
    panel_label(ax, "b")

    ax = axs[1, 0]
    for k, mech in enumerate(("MAR", "MNAR")):
        q = P[(P.mech == mech) & (P.scale == 1.0)]
        for i, nm in enumerate(SC_ORDER):
            v = q[q.score == nm].auroc.values
            y = i + (k - 0.5) * 0.32
            ax.plot([np.quantile(v, 0.025), np.quantile(v, 0.975)], [y, y], color=C[nm], lw=1.0)
            ax.scatter(np.median(v), y, color=C[nm], s=22 if mech == "MAR" else 22,
                       marker="o" if mech == "MAR" else "D", edgecolor="white", lw=0.5, zorder=3)
    ax.set_yticks(range(len(SC_ORDER)), [LABEL[n] for n in SC_ORDER], fontsize=6.5)
    ax.set_xlabel("Flip-detection AUROC (median, 2.5–97.5%)")
    ax.set_title("Across repetitions (● MAR, ◆ MNAR)", loc="left")
    ax.set_xlim(0.6, 1.0)
    panel_label(ax, "c")

    ax = axs[1, 1]
    ex = E["MAR"]
    raw, calp, f = ex["scores"]["FLIP"], ex["calp"]["FLIP"], ex["flip"].astype(float)
    for p_, col, lab in ((raw, "#9cc3ef", "raw FLIP"), (calp, "#2a78d6", "isotonic-calibrated FLIP")):
        bins = np.array_split(np.argsort(p_, kind="stable"), 12)
        xs = [p_[b].mean() for b in bins]
        ys = [f[b].mean() for b in bins]
        ax.plot(xs, ys, marker="o", ms=3.5, color=col, label=lab)
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, ls="--", lw=0.8)
    ax.set_xlabel("Predicted flip probability (bin mean)")
    ax.set_ylabel("Observed flip rate")
    ax.set_title("Calibration on the test split (MAR)", loc="left")
    ax.legend(loc="upper left")
    panel_label(ax, "d")
    fig.tight_layout(h_pad=2.2, w_pad=1.6)
    save(fig, "fig4_flip_detection")

    # ---- Figure 5: guarantee
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    scs = ["Kim et al. 2026", "Margin only", "Multiple imputation", "FLIP v1 (Gaussian)", "FLIP",
           "FLIP (posterior decision)"]
    for k, mech in enumerate(("MAR", "MNAR")):
        ax = axs[k]
        q = L[(L.mech == mech) & (L.kind == "marginal") & (L.scale == 1.0)]
        alphas = sorted(q.alpha.unique())
        x = np.arange(len(alphas))
        wd = 0.13
        for i, nm in enumerate(scs):
            med = [q[(q.score == nm) & (q.alpha == a)].coverage.median() * 100 for a in alphas]
            ax.bar(x + (i - 2.5) * wd, med, width=wd, color=C[nm], label=LABEL[nm], edgecolor="white", lw=0.4)
        ax.set_xticks(x, [f"α = {a:g}" for a in alphas])
        ax.set_ylabel("Patients decided with the guarantee (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{mech}: certified coverage (δ = 0.10)", loc="left")
        panel_label(ax, "ab"[k])
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig5_guarantee")

    # ---- Figure 6: acquisition
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    pols = ["No extra testing", "Kim rank + Kim panel", "Margin rank + Kim panel", "FLIP-VOI: expected flips"]
    pcol = ["#8a8984", "#e34948", "#eb6834", "#2a78d6"]
    for k, mech in enumerate(("MAR", "MNAR")):
        ax = axs[k]
        q = A[(A.mech == mech) & (A.scale == 1.0) & (A.setting == "budget 10%")]
        for r_i, rule in enumerate(("imputed", "posterior")):
            for i, pol in enumerate(pols):
                v = q[(q.rule == rule) & (q.policy == pol)].flip_after.values * 100
                xpos = r_i * 5 + i
                ax.bar(xpos, np.median(v), color=pcol[i], width=0.8, edgecolor="white",
                       label=pol if r_i == 0 else None)
                ax.plot([xpos, xpos], [np.quantile(v, 0.025), np.quantile(v, 0.975)], color=INK, lw=0.8)
        ax.set_xticks([1.5, 6.5], ["issued by single imputation", "issued by FLIP posterior"])
        ax.set_ylabel("Flipped decisions after testing (%)")
        ax.set_title(f"{mech}: budget for one extra panel in 10% of patients", loc="left", fontsize=8)
        panel_label(ax, "ab"[k])
    axs[1].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig6_acquisition")

    # ---- Figure 7: Shapley attribution of instability
    ex = E["MAR"]
    eta, Rd, Mb = ex["eta_draws"], ex["R_draws"], ex["Mb"]
    score = ex["scores"]["FLIP"]
    cand = np.flatnonzero((Mb.sum(1) >= 3) & (score > 0.15))
    rng = np.random.default_rng(1)
    pick = rng.choice(cand, size=min(400, cand.size), replace=False)
    import itertools

    def shap_row(i):
        miss = [b for j, b in enumerate(BLOCK_NAMES) if Mb[i, j]]
        e = eta[i]
        ve = e.var()
        if ve <= 1e-12:
            return {}
        cache = {}

        def v(T):
            key = frozenset(T)
            if key not in cache:
                if not T:
                    cache[key] = 0.0
                else:
                    cols = sum((BLOCK_COLS[b] for b in T), [])
                    X = np.column_stack([np.ones(len(e)), Rd[i][:, cols]])
                    coef = np.linalg.lstsq(X, e, rcond=None)[0]
                    cache[key] = max(0.0, 1 - ((e - X @ coef) ** 2).mean() / ve)
            return cache[key]

        phi = dict.fromkeys(miss, 0.0)
        perms = list(itertools.permutations(miss)) if len(miss) <= 6 else \
            [list(rng.permutation(miss)) for _ in range(400)]
        for perm in perms:
            prev, acc = 0.0, []
            for b in perm:
                acc.append(b)
                cur = v(acc)
                phi[b] += cur - prev
                prev = cur
        tot = sum(phi.values())
        return {b: phi[b] / tot for b in miss} if tot > 0 else {}

    shares = [shap_row(i) for i in pick]
    S = pd.DataFrame(shares).reindex(columns=BLOCK_NAMES)
    S.to_csv(os.path.join(RES, "t1_shapley_shares.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    show = [k for k in range(len(shares)) if len(shares[k]) >= 4][:6]
    bcol = dict(zip(BLOCK_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                  "#6250d6", "#e34948", "#8a8984"]))
    for row_i, k in enumerate(show):
        left = 0.0
        for b in BLOCK_NAMES:
            v = shares[k].get(b, 0.0)
            if v > 0:
                ax.barh(row_i, v * 100, left=left, color=bcol[b], edgecolor="white", lw=0.6)
                if v > 0.12:
                    ax.text(left + v * 50, row_i, b, ha="center", va="center", fontsize=6, color="white")
                left += v * 100
    ax.set_yticks(range(len(show)), [f"patient {chr(65 + r)}  (FLIP {score[pick[k]]:.2f})"
                                     for r, k in enumerate(show)], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Share of decision instability (%)")
    ax.set_title("Which unordered panel is responsible?", loc="left")
    ax.set_xlim(0, 100)
    panel_label(ax, "a")
    ax = axs[1]
    avg = S.mean(skipna=True).reindex(BLOCK_NAMES).fillna(0) * 100
    ax.barh(range(len(BLOCK_NAMES)), avg.values, color=[bcol[b] for b in BLOCK_NAMES], edgecolor="white")
    ax.set_yticks(range(len(BLOCK_NAMES)), BLOCK_NAMES, fontsize=7)
    ax.set_xlabel("Mean share when missing (%)")
    ax.set_title(f"High-uncertainty patients (n = {int(S.notna().any(axis=1).sum())})", loc="left")
    ax.invert_yaxis()
    panel_label(ax, "b")
    fig.tight_layout()
    save(fig, "fig7_shapley")

    # ---- Figure: decision curve with deferral
    D = pd.read_csv(os.path.join(RES, "t1_dca_y_allergic_disease.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    strat = [("Impute and decide", "#8a8984"), ("Kim defer (matched)", "#e34948"),
             ("Margin defer (matched)", "#eb6834"), ("FLIP defer (LTT)", "#6fa8e8"),
             ("Posterior decision", "#1baf7a"), ("Posterior decision + FLIP defer (LTT)", "#2a78d6"),
             ("Order all missing panels", "#1f1f1f")]
    for k, mech in enumerate(("MAR", "MNAR")):
        ax = axs[k]
        q = D[(D.mech == mech) & (D.scale == 1.0)]
        for nm, col in strat:
            g = q[q.strategy == nm].groupby("kappa").nb.median()
            ax.plot(g.index * 100, g.values * 100, marker="o", ms=3, color=col, label=nm,
                    ls="--" if nm == "Order all missing panels" else "-")
        ax.set_xlabel("Cost per extra panel (net-benefit units × 100)", fontsize=7.5)
        ax.set_ylabel("Net benefit × 100")
        ax.set_title(f"{mech}", loc="left")
        panel_label(ax, "ab"[k])
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig8_dca")


# ============================================================== task 2
def fig_task2():
    cs = pd.read_csv(os.path.join(RES, "t2_serum_curves.csv"))
    cp = pd.read_csv(os.path.join(RES, "t2_spt_curves.csv"))
    meth = [("FLIP (total IgE + adaptive)", "#2a78d6", "-"), ("FLIP (adaptive, no total IgE)", "#6fa8e8", "-"),
            ("FLIP (adaptive, in-domain)", "#2a78d6", "-"),
            ("FLIP (adaptive, structure transported from serum)", "#1baf7a", "-"),
            ("Ablation: no cross-reactivity (diagonal)", "#e87ba4", ":"),
            ("Sequential, greedy order, stop at first positive", "#eb6834", "-"),
            ("Fixed panel: greedy set cover", "#e34948", "--"), ("Fixed panel: most prevalent", "#8a8984", "--")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for k, (df, title, kmax) in enumerate(((cs, "NHANES 2005–2006, serum sIgE (19 allergens)", 19),
                                           (cp, "NHANES III, skin prick test (10 allergens)", 10))):
        ax = axs[k]
        for nm, col, ls in meth:
            q = df[df.method == nm]
            if q.empty:
                continue
            q = q.sort_values("mean_tests")
            # monotone envelope for readability
            ax.plot(q.mean_tests, 100 * q.sensitivity, color=col, ls=ls, lw=1.8 if "FLIP (" in nm else 1.2,
                    label=nm, marker="o" if "Fixed" in nm else None, ms=2.5)
        ax.set_xlim(0, kmax)
        ax.set_ylim(70, 100.5)
        ax.axhline(95, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("Mean number of tests per person")
        ax.set_ylabel("Sensitised persons detected (%)")
        ax.set_title(title, loc="left", fontsize=8)
        panel_label(ax, "ab"[k])
    h, lab = [], []
    for a in axs:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2), fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig9_atopy_testing")

    S = pickle.load(open(os.path.join(RES, "t2_serum_sims.pkl"), "rb"))
    ops = pd.read_csv(os.path.join(RES, "t2_serum_operating_points.csv"))
    sim = S["sims"]["FLIP (total IgE + adaptive)"]
    te = S["is_te"]
    sens = S["sens"][S["evidx"]][te]
    pa, fp = sim["p_any"][te], sim["first_pos"][te]
    lam = float(ops[(ops.method == "FLIP (total IgE + adaptive)") & (ops.alpha == 0.05)].operating.iloc[0]
                .split("=")[1])
    K = pa.shape[1] - 1
    below = np.where(np.isnan(pa), False, pa <= lam)
    stop = np.where(below.any(1), below.argmax(1), K)
    found = fp < stop
    tests = np.where(found, fp + 1, stop)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axs[0]
    bins = np.arange(-0.5, 20.5, 1)
    ax.hist(tests[sens], bins=bins, color="#2a78d6", alpha=0.85, label="sensitised")
    ax.hist(tests[~sens], bins=bins, color="#eb6834", alpha=0.65, label="not sensitised")
    kfix = int(ops[(ops.method == "Fixed panel: greedy set cover") & (ops.alpha == 0.05)].operating.iloc[0]
               .split("=")[1])
    ax.axvline(kfix, color="#e34948", ls="--", lw=1.0)
    ax.text(kfix + 0.3, ax.get_ylim()[1] * 0.9, f"fixed panel\n(k = {kfix})", color="#e34948", fontsize=6.5)
    ax.set_xlabel("sIgE tests used (FLIP, α = 0.05)")
    ax.set_ylabel("Persons (test split)")
    ax.legend()
    ax.set_title("Testing effort adapts to the person", loc="left")
    panel_label(ax, "a")
    ax = axs[1]
    first = sim["order"][te][:, 0]
    names = S["names"]
    vc = pd.Series([names[j] for j in first if j >= 0]).value_counts()
    ax.barh(range(len(vc)), vc.values / vc.values.sum() * 100, color="#2a78d6")
    ax.set_yticks(range(len(vc)), vc.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Persons for whom this is the first test (%)")
    ax.set_title("Personalised first test", loc="left")
    panel_label(ax, "b")
    fig.tight_layout()
    save(fig, "fig10_atopy_effort")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "structure"):
        fig_structure()
    if which in ("all", "task1"):
        fig_task1()
    if which in ("all", "task2"):
        fig_task2()
