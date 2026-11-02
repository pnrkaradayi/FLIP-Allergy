# FLIP: decision-flip probabilities for incomplete allergen-specific IgE panels

Code for the paper

> Karadayı Ataş P. *Cross-Reactivity Makes Unordered Allergen Tests Predictable: Decision-Flip
> Probabilities with Finite-Sample Guarantees for Incomplete Specific IgE Panels.* Submitted to
> *Biomedicines* (MDPI), 2026.

## What FLIP does

Allergen-specific IgE (sIgE) tests are ordered selectively, so clinical decisions are often made
while some panels have not been ordered. Because IgE cross-reacts between homologous proteins,
the ordered results carry information about the unordered ones. FLIP uses this to answer, for
each patient: *would complete testing change my decision?*

1. **Model.** sIgE results are a left-censored multivariate normal vector (values below the
   0.35 kU/L detection limit are only known to be below it), with a covariance built from
   cross-reactive protein families (PR-10, profilins, lipocalins, tropomyosins, storage proteins…).
2. **Completion.** Given the ordered tests, the model draws many plausible complete panels.
3. **FLIP score.** The share of completions in which the decision changes is the probability that
   complete testing would flip it.
4. **Guarantee.** A Learn-then-Test threshold on the FLIP score certifies, with a finite-sample
   guarantee, that the flip rate among accepted decisions is at most α.
5. **What to order next.** A censoring-aware value of information ranks the unordered panels, and
   adaptive testing orders tests one at a time and stops once the miss rate is certified.

## Quick start (no data needed)

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

The example simulates four cross-reactive allergens (two house dust mites, cat, dog). For a
patient with *D. pteronyssinus* positive and cat negative, it predicts the unordered
*D. farinae* (P(positive) ≈ 1.00) and dog (≈ 0.14). The decision "mite-sensitised" then has a FLIP
score of 0.00 (accept), and "not pet-sensitised" a score of 0.14 (order the dog test).

## Repository layout

```
analysis/        all analyses of the paper (run order in analysis/README.md)
  flipx.py       core library (censored MVN, factor model, completions, FLIP, LTT, VOI)
  flipg.py       Gaussian-copula version for mixed tables
  ext_data.py    loaders of the external datasets
code/            download / build scripts for NHANES 2005-2006 and NHANES III
data_raw/        empty; where to put each dataset (data_raw/README.md)
examples/        quickstart on simulated data
outputs/         figures and LaTeX tables (created when the analyses run)
```

## Reproducing the paper

1. Python 3.10 or newer: `pip install -r requirements.txt`.
2. Data: `python code/00_download_nhanes.py` and `python code/01b_build_nhanes3.py` download the
   public NHANES files; the external datasets are listed, with their sources and target paths, in
   [`data_raw/README.md`](data_raw/README.md). No data are included in this repository; the URECA,
   EAT and LEAP data (ImmPort) require free registration and may not be redistributed.
3. Analyses: run the scripts in [`analysis/README.md`](analysis/README.md) in the order given.
   Random seeds are fixed in the code.

## What was analysed

* Development: NHANES 2005–2006 serum sIgE (7,266 participants, 19 sIgE + total IgE) and NHANES III
  skin prick tests (10,841 participants, 10 allergens).
* Re-analysis of 22 published allergy studies on their own data, each after reproducing its
  headline result: 11 NHANES analyses and 11 cohorts and trials (Allergen Chip Challenge, MAAS,
  Graz venom, Chinese paediatric asthma, Moscow biochip, SOS-ALL, Bern, IDEAL, URECA, EAT, LEAP).
* Supplement: seven clinical datasets outside allergy.
* Comparators: the missing-data score of Kim et al., multiple and single imputation, distance to
  the decision threshold, fraction missing, fixed panels (greedy set cover), sequential testing,
  and ablations of FLIP (no censoring, no cross-reactivity, free covariance).

## Citation

Please cite the paper above (full reference to be added on publication); see `CITATION.cff`.

## License

MIT, see [LICENSE](LICENSE).
