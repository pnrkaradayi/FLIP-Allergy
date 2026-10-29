"""
a09_external_flip.py
====================
The Task 1 protocol of the main analysis applied, unchanged, to the public
datasets of other published studies (ext_data.py).

For every dataset and repetition
* records with every test result go to calibration (30%) and test (30%);
  all other records (with their natural missingness) and the remaining
  complete records are the training split;
* the training split fits the Gaussian-copula latent model (censored assay
  values as bounds, binary results as thresholded latent normals), the
  MissForest-type imputer and an L1 logistic decision model (trained on the
  imputed data, as in Kim et al. 2026); the threshold is set at 10% false
  positive rate on the training split;
* whole tests (blocks) are then left unordered on the calibration and test
  records by amputation (MAR: shared care intensity; MNAR: in addition the
  patient's own result), so that the complete-testing decision is known.

Outputs (results/ext_*): per-patient pooled test predictions, flip
detection, Learn-then-Test coverage, decision errors of the imputation
pipelines used by the published studies and of the FLIP posterior decision.

Run:  python a09_external_flip.py [dataset ...]   (default: all)
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext_data  # noqa: E402
from flipg import (Copula, Logistic, amputate, expand, fit_latent, latent_draws,  # noqa: E402
                   make_imputer, mi_draws, scores_on_draws, threshold_at_fpr)
from flipx import flip_from_draws, ltt_threshold  # noqa: E402

RES = os.path.join(HERE, "results")
ALPHAS = (0.05, 0.10)
DELTA = 0.10
P_ORDER = 0.60
N_MAR, N_MNAR = 30, 15
MAX_TRAIN, MAX_EVAL = 3000, 800
IMPUTERS = ("missforest", "mean", "median", "knn", "mice")


# ----------------------------------------------------------------- task definitions
def task(key):
    """Dataset, outcome and the subset of records/blocks used for the decision task."""
    D = ext_data.LOADERS[key]()
    X, y = D["X"].copy(), D["y"]
    keep = np.ones(len(X), bool)
    if key == "graz":
        raw = D["extra"]["raw"]
        keep = raw.venom.isin(["bee", "wasp"]).values
        y = (raw.venom == "bee").astype(int).values
    elif key == "moscow":
        dx = D["extra"]["diagnosis"]
        y = (dx != "Dermatitis").astype(int)
        X.insert(0, "age", D["extra"]["age"])
        X.insert(1, "male", D["extra"]["male"])
        D["kind"] = {"age": "cont", "male": "bin", **D["kind"]}
        groups = {"tree pollen": ["Alder", "Birch", "Hazelnut_p", "Oak"],
                  "weed pollen": ["Wormwood", "Mugwort", "Dandelion"],
                  "grass pollen": ["Bermuda grass", "Orchard grass", "Meadow fescue", "Perennial rye grass",
                                   "Timothy grass", "Cultivated rye"],
                  "animal dander": ["Cat dander", "Dog dander"], "mites": ["D. pteronyssinus", "D. farinae"],
                  "mould and cockroach": ["Alternaria tenius", "Cockroach, German"],
                  "foods": [c for c in X.columns if c not in ("age", "male") and c not in sum(
                      [["Alder", "Birch", "Hazelnut_p", "Oak", "Wormwood", "Mugwort", "Dandelion", "Bermuda grass",
                        "Orchard grass", "Meadow fescue", "Perennial rye grass", "Timothy grass", "Cultivated rye",
                        "Cat dander", "Dog dander", "D. pteronyssinus", "D. farinae", "Alternaria tenius",
                        "Cockroach, German"]], [])]}
        D["blocks"] = list(groups.items())
        D["always"] = ["age", "male"]
    elif key == "maas":
        clusters = {"mite": ["derf1", "derf2", "derp1", "derp2", "derp10", "lepd2", "blot5"],
                    "peanut and storage proteins": ["arah1", "arah2", "arah3", "arah6", "jugr2", "gald3"],
                    "lipocalins and animal": ["canf1", "canf2", "canf5", "feld1", "feld4", "equc1", "musm1"],
                    "grass": ["phlp1", "phlp2", "phlp4", "phlp5", "phlp6", "phlp11", "cynd1"],
                    "PR-10": ["betv1", "alng1", "cora101", "cora104", "mald1", "prup1", "glym4", "arah8"],
                    "tree and weed": ["cryj1", "cupa1", "olee1", "plaa2", "alta1"],
                    "profilin and CCD": ["betv2", "phlp12", "hevb8", "mera1", "chea1", "muxf3"]}
        cols = list(X.columns)
        used = sum(clusters.values(), [])
        clusters = {k: [c for c in v if c in cols] for k, v in clusters.items()}
        rest = [c for c in cols if c not in used]
        if rest:
            clusters["other components"] = rest
        D["blocks"] = [(k, v) for k, v in clusters.items() if v]
    D["X"], D["y"] = X[keep].reset_index(drop=True), np.asarray(y)[keep]
    if "site" in D["extra"]:
        D["extra"]["site"] = D["extra"]["site"][keep]
    if key == "heart":     # complete records exist almost only in Cleveland (Detrano's derivation site)
        D["eval_mask"] = D["extra"]["site"] == "Cleveland"
    return D


# ----------------------------------------------------------------- one repetition
def run_rep(args):
    key, mech, rep, path = args
    with open(path, "rb") as fh:
        D = pickle.load(fh)
    cols = list(D["X"].columns)
    kinds = [D["kind"][c] for c in cols]
    lods = [D["lod"].get(c, 1.0) for c in cols]
    Xr = D["X"].values.astype(float)
    y = D["y"].astype(int)
    bidx = [[cols.index(c) for c in b] for _, b in D["blocks"]]
    test_cols = sorted(set(sum(bidx, [])))
    alw = [cols.index(c) for c in D["always"]]
    model_cols = sorted(set(test_cols + alw))
    complete = np.isfinite(Xr[:, model_cols]).all(1)
    if "eval_mask" in D:
        complete &= D["eval_mask"]
    rng = np.random.default_rng(10_000 * (mech == "MNAR") + 97 * rep + len(key))
    cidx = np.flatnonzero(complete)
    # stratified split of the complete records: 40% train, 30% calibration, 30% test
    parts = {"train": [], "calib": [], "test": []}
    for cls in (0, 1):
        r = rng.permutation(cidx[y[cidx] == cls])
        a, b = int(round(0.4 * len(r))), int(round(0.7 * len(r)))
        parts["train"] += list(r[:a])
        parts["calib"] += list(r[a:b])
        parts["test"] += list(r[b:])
    tr = np.zeros(len(y), bool)
    tr[np.flatnonzero(~complete)] = True
    tr[parts["train"]] = True
    # computational cap for the two largest tables (SUPPORT, PhysioNet): 3000 training records,
    # 800 calibration and 800 test records per repetition, drawn at random
    if tr.sum() > MAX_TRAIN:
        keep_tr = rng.choice(np.flatnonzero(tr), MAX_TRAIN, replace=False)
        tr[:] = False
        tr[keep_tr] = True
    cap = lambda v: rng.choice(v, MAX_EVAL, replace=False) if len(v) > MAX_EVAL else v
    ca, te = np.array(sorted(cap(parts["calib"]))), np.array(sorted(cap(parts["test"])))
    cop = Copula(kinds, lods).fit(Xr[tr])
    T = cop.transform(Xr)
    t0 = time.time()
    # allergen panels: cross-reactivity factor structure (as in the main analysis); other tables: free covariance
    fmask = (ext_data.family_mask_for(cols, ext_data.EXTRACT_FAMILIES[key])[0]
             if key in ext_data.EXTRACT_FAMILIES else None)
    lat = fit_latent(cop, T[tr], n_iter=250, burn=90, seed=rep, fmask=fmask)
    imps = {k: make_imputer(k, seed=rep).fit(T[tr]) for k in IMPUTERS}
    Ttr_imp = imps["missforest"].transform(T[tr])
    clf = Logistic().fit(Ttr_imp, y[tr], seed=rep)
    s_tr = clf.score(Ttr_imp)
    tau = threshold_at_fpr(s_tr[y[tr] == 0], 0.10)
    out = {}
    for nm, idx in (("calib", ca), ("test", te)):
        Tf = T[idx]
        Mb = amputate(Tf, bidx, [P_ORDER] * len(bidx), rng, mech)
        M = expand(Mb, bidx, len(cols))
        To = Tf.copy()
        To[M] = np.nan
        s_full = clf.score(Tf)
        d_full = s_full >= tau
        dec = {k: clf.score(imps[k].transform(To)) >= tau for k in IMPUTERS}
        s_hat = clf.score(imps["missforest"].transform(To))
        d_hat = dec["missforest"]
        F = latent_draws(lat, cop, To, n_chains=12, n_sweeps=28, keep=12, seed=rep * 3 + (nm == "test"))
        e = scores_on_draws(clf, F)
        e_mi = mi_draws(T[tr], To, clf, n_imp=20, seed=rep)
        p_pos = (e >= tau).mean(1)
        w = clf.kim_weights()
        out[nm] = dict(
            flip=d_hat != d_full, d_full=d_full, y=y[idx], n_missing_blocks=Mb.sum(1),
            scores={"Fraction missing": Mb.mean(1), "Kim et al. 2026": (M * w[None, :]).sum(1),
                    "Margin only": -np.abs(s_hat - tau), "Multiple imputation": flip_from_draws(e_mi, tau, d_hat),
                    "FLIP": flip_from_draws(e, tau, d_hat)},
            decisions={**{f"imputation: {k}": dec[k] for k in IMPUTERS}, "FLIP posterior": p_pos >= 0.5},
            U_post=np.minimum(p_pos, 1 - p_pos))
    C, Te = out["calib"], out["test"]
    ltt = []
    for nm in Te["scores"]:
        for a in ALPHAS:
            lam = ltt_threshold(C["scores"][nm], C["flip"], a, DELTA)
            acc = Te["scores"][nm] <= lam
            ltt.append(dict(score=nm, alpha=a, coverage=float(acc.mean()),
                            realised=float(Te["flip"][acc].mean()) if acc.any() else np.nan))
    fbc = C["decisions"]["FLIP posterior"] != C["d_full"]
    fbt = Te["decisions"]["FLIP posterior"] != Te["d_full"]
    for a in ALPHAS:
        lam = ltt_threshold(C["U_post"], fbc, a, DELTA)
        acc = Te["U_post"] <= lam
        ltt.append(dict(score="FLIP (posterior decision)", alpha=a, coverage=float(acc.mean()),
                        realised=float(fbt[acc].mean()) if acc.any() else np.nan))
    info = dict(key=key, mech=mech, rep=rep, n_train=int(tr.sum()), n_calib=len(ca), n_test=len(te),
                tau=tau, auc_full_test=roc_auc_score(Te["y"], clf.score(T[te])) if len(set(Te["y"])) > 1 else np.nan,
                flip_rate=float(Te["flip"].mean()), secs=time.time() - t0)
    return dict(info=info, test=Te, ltt=ltt)


def run_dataset(key, n_mar=N_MAR, n_mnar=N_MNAR, workers=10):
    D = task(key)
    path = os.path.join(RES, f"_ext_task_{key}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(D, fh)
    jobs = [(key, "MAR", r, path) for r in range(n_mar)] + [(key, "MNAR", r, path) for r in range(n_mnar)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_rep, jobs))
    print(f"{key}: {len(res)} repetitions in {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(RES, f"ext_flip_{key}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    return summarise(key, res)


def summarise(key, res):
    rows = []
    for mech in ("MAR", "MNAR"):
        R = [r for r in res if r["info"]["mech"] == mech]
        if not R:
            continue
        flip = np.concatenate([r["test"]["flip"] for r in R])
        base = dict(dataset=key, mech=mech, reps=len(R), n_test_pooled=len(flip), flip_rate=float(flip.mean()),
                    auc_full=float(np.nanmean([r["info"]["auc_full_test"] for r in R])))
        for nm in R[0]["test"]["scores"]:
            s = np.concatenate([r["test"]["scores"][nm] for r in R])
            per = [roc_auc_score(r["test"]["flip"], r["test"]["scores"][nm]) for r in R
                   if 0 < r["test"]["flip"].sum() < len(r["test"]["flip"])]
            rows.append(dict(**base, metric="flip AUROC (pooled)", method=nm,
                             value=roc_auc_score(flip, s) if 0 < flip.sum() < len(flip) else np.nan,
                             value_rep_mean=float(np.mean(per)) if per else np.nan))
        dfull = np.concatenate([r["test"]["d_full"] for r in R])
        yv = np.concatenate([r["test"]["y"] for r in R])
        rows.append(dict(**base, metric="accuracy vs outcome", method="complete testing",
                         value=float((dfull == yv).mean())))
        for nm in R[0]["test"]["decisions"]:
            d = np.concatenate([r["test"]["decisions"][nm] for r in R])
            rows.append(dict(**base, metric="decision error vs complete testing", method=nm,
                             value=float((d != dfull).mean())))
            rows.append(dict(**base, metric="accuracy vs outcome", method=nm, value=float((d == yv).mean())))
        L = pd.DataFrame([dict(**l_) for r in R for l_ in r["ltt"]])
        for (nm, a), g in L.groupby(["score", "alpha"]):
            rows.append(dict(**base, metric=f"LTT coverage (alpha={a})", method=nm, value=float(g.coverage.mean()),
                             value_rep_mean=float(g.realised.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, f"ext_flip_{key}.csv"), index=False)
    piv = out[out.metric.isin(["flip AUROC (pooled)", "LTT coverage (alpha=0.05)"])]
    print(piv.pivot_table(index=["mech", "method"], columns="metric", values="value").round(3).to_string())
    e = out[out.metric == "decision error vs complete testing"]
    print(e.pivot_table(index="method", columns="mech", values="value").round(4).to_string())
    return out


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    keys = sys.argv[1:] or ["heart", "thyroid", "hepatitis", "pima", "ckd", "support2", "maas", "sosall",
                            "graz", "moscow", "physionet2012"]
    for k in keys:
        run_dataset(k)
