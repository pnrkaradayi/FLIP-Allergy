# FLIP: decision-flip probabilities for incomplete allergen-specific IgE panels

Code for the manuscript

> P. Karadayı Ataş. *Cross-Reactivity Makes Unordered Allergen Tests Predictable: Decision-Flip
> Probabilities with Finite-Sample Guarantees for Incomplete Specific IgE Panels.* Submitted to
> *Biomedicines* (MDPI), 2026.

When only part of an allergen panel is ordered, a clinical decision is made without the unordered
tests. FLIP models the panel with a censored latent Gaussian model whose covariance follows
cross-reactive protein families (results below the detection limit enter as bounds), and from the
posterior completions of the unordered tests it computes, for every patient:

* the **FLIP score**, the probability that complete testing would change the issued decision;
* a **Learn-then-Test acceptance rule** with a finite-sample guarantee on the flip rate among the
  accepted decisions, calibrated on amputated complete records where the true flips are known;
* the **value of information** of each unordered panel, used for deferral, test ordering and
  adaptive testing with a certified sensitivity.

The method is validated on NHANES 2005–2006 serum sIgE and NHANES III skin prick tests, and compared
with twelve published allergy studies on their own data (six NHANES re-analyses and six open
cohorts) and with seven clinical datasets outside allergy.

## Repository layout

```
flip_allergy/
  data_prep/      download NHANES 2005-2006 files; build the NHANES III skin prick extract
  data_raw/       input data (not stored here; see data_raw/README.md)
  analysis/       all analyses (a00-a12), core library flipx.py, Gaussian-copula FLIP flipg.py,
                  loaders for external datasets ext_data.py; run order in analysis/README.md
  analysis/results/  per-repetition results (CSV) and every number quoted in the text (numbers.json)
manuscript/
  figures/, tables/  written by the analysis scripts (generated, not stored here)
```

## Installation

Tested with Python 3.11 on Windows; the code is plain NumPy/SciPy/pandas and runs on Linux and macOS.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

All data are public. NHANES files are downloaded by script; the external datasets are listed with
their download links in [`flip_allergy/data_raw/README.md`](flip_allergy/data_raw/README.md). No
data file is redistributed in this repository.

```bash
cd flip_allergy/data_prep
python 00_download_nhanes.py       # NHANES 2005-2006: sIgE panel, questionnaires, dust (~25 MB)
python 01b_build_nhanes3.py        # NHANES III skin prick tests (downloads ~280 MB, keeps a small extract)
```

## Reproducing the results

Run the scripts in `flip_allergy/analysis` in the order given in
[`flip_allergy/analysis/README.md`](flip_allergy/analysis/README.md). The longest steps are the
Task 1 repetitions (`a02_risk_decision.py`, about 10 min on 10 cores), the external Task 1 analyses
(`a09_external_flip.py`, about 2 h) and external adaptive testing (`a10_external_testing.py`, about
1 h). `a06_tables.py` collects every number quoted in the manuscript into
`analysis/results/numbers.json`.

The result files in `analysis/results` are the ones behind the submitted manuscript. Run logs,
per-patient records of the external cohorts and the intermediate pickle files are not included;
they are recreated by the scripts.

## License

Code: MIT License (see [`LICENSE`](LICENSE)). The datasets keep the terms of their providers.

## Citation

Please cite the article (reference to be added on publication); see [`CITATION.cff`](CITATION.cff).
