# Input data

No data file is stored in this repository. Everything below is public; place each file at the path
shown (relative to this folder). Please follow the terms of each provider.

## NHANES (US CDC / National Center for Health Statistics; public, no registration)

| Path | Source | Created by |
|---|---|---|
| `AL_IGE_D.xpt`, `DEMO_D.xpt`, `AGQ_D.xpt`, `MCQ_D.xpt`, `RDQ_D.xpt`, `BMX_D.xpt`, `SMQ_D.xpt`, `ALDUST_D.xpt`, `CBC_D.xpt` | NHANES 2005–2006, `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/<NAME>.xpt` | `../data_prep/00_download_nhanes.py` |
| `nhanes3/nhanes3_allergy_raw.csv` | NHANES III 1988–1994 examination, adult and youth files, `https://wwwn.cdc.gov/nchs/data/nhanes3/1a/` | `../data_prep/01b_build_nhanes3.py` |

## Open allergy cohorts (`external_allergy/`)

| Path | Study | Download |
|---|---|---|
| `acc/allergenchipchallenge-data-corrected-final-hdh-sfa.csv` | Allergen Chip Challenge (ALEX² multiplex) | <https://www.data.gouv.fr/fr/datasets/allergen-chip-challenge/> |
| `maas/sIgE_data_age_11.csv`, `maas/Filtered_data11yrs.csv` | Manchester Asthma and Allergy Study, component sIgE at age 11 | <https://github.com/sfontane/Interaction-patterns-between-specific-IgE-antibodies-in-molecular-allergy-diagnostics-and-prediction> |
| `graz_venom/complete_dataset_20Oct2025.xlsx` | Graz Hymenoptera venom allergy (rename the file: space → underscore) | <https://doi.org/10.5281/zenodo.18466646> |
| `sosall/clinical_data_mlm_sos_all.txt` | SOS-ALL | <https://doi.org/10.5281/zenodo.21064520> |
| `moscow/40413_2018_220_MOESM1_ESM.xlsx` | Moscow biochip sIgE/IgG4 (Additional file 1) | <https://doi.org/10.6084/m9.figshare.7409960> |
| `guo/Cleaned_data.xlsx` | Chinese paediatric asthma cohorts | <https://doi.org/10.5281/zenodo.18888286> |

## Clinical datasets outside allergy (`external/`)

| Path | Dataset | Download |
|---|---|---|
| `heart/processed.{cleveland,hungarian,switzerland,va}.data` | UCI Heart Disease | <https://archive.ics.uci.edu/static/public/45/heart+disease.zip> |
| `thyroid/` (all files of the archive) | UCI Thyroid Disease | <https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip> |
| `hepatitis/hepatitis.data` | UCI Hepatitis | <https://archive.ics.uci.edu/static/public/46/hepatitis.zip> |
| `pima/diabetes.arff` | Pima Indians Diabetes (OpenML 37) | <https://openml.org/data/v1/download/37/diabetes.arff> |
| `ckd/chronic-kidney-disease.arff` | Chronic Kidney Disease (OpenML) | <https://openml.org/data/v1/download/22045641/chronic-kidney-disease.arff> |
| `support2/Support2.arff` | SUPPORT2 (OpenML) | <https://openml.org/data/v1/download/22121321/Support2.arff> |
| `physionet2012/set-a.zip`, `physionet2012/Outcomes-a.txt` | PhysioNet/CinC Challenge 2012, set A | <https://physionet.org/content/challenge-2012/1.0.0/> |

The loaders for all external datasets are in `../analysis/ext_data.py`.
