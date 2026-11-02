# Data

No data are stored in this repository. Every dataset is public or available after free
registration from its original source; download it there and place the files at the paths below
(relative to `data_raw/`). The loaders in `analysis/ext_data.py` read exactly these paths.

## NHANES (development data, downloaded by script)

| Data | How | Writes |
|---|---|---|
| NHANES 2005–2006 (AL_IGE_D, DEMO_D, AGQ_D, MCQ_D, RDQ_D, BMX_D, SMQ_D, ALDUST_D, CBC_D, COT_D) | `python code/00_download_nhanes.py` | `*.xpt`, `nhanes_extra/COT_D.xpt` |
| NHANES III 1988–1994 skin prick test (exam, adult and youth files) | `python code/01b_build_nhanes3.py` | `nhanes3/nhanes3_allergy_raw.csv`, `../results_so_far/nhanes3_flip_cohort.csv` |
| CDC 2000 BMI-for-age reference (Visness et al. 2009) | `bmiagerev.csv` from https://www.cdc.gov/growthcharts/ (data files) | `external/cdc_growth/bmiagerev.csv` |

## Published allergy studies

| Study | Source | Place at |
|---|---|---|
| Martinroche 2026, Allergen Chip Challenge | https://www.data.gouv.fr/fr/datasets/allergen-chip-challenge/ (doi:10.60597/j5fe-g420) | `external_allergy/acc/allergenchipchallenge-data-corrected-final-hdh-sfa.csv` |
| Fontanella 2018, MAAS (ISAC at 11 years) | https://github.com/sfontane/Interaction-patterns-between-specific-IgE-antibodies-in-molecular-allergy-diagnostics-and-prediction | `external_allergy/maas/sIgE_data_age_11.csv`, `external_allergy/maas/Filtered_data11yrs.csv` |
| Arzt-Gradwohl 2026, Graz venom allergy | https://doi.org/10.5281/zenodo.18466646 | `external_allergy/graz_venom/complete_dataset_20Oct2025.xlsx` |
| Smoldovskaya 2018, Moscow biochip | https://doi.org/10.6084/m9.figshare.7409960 | `external_allergy/moscow/40413_2018_220_MOESM1_ESM.xlsx` |
| Guo 2026, Chinese children with asthma | https://doi.org/10.5281/zenodo.18888286 | `external_allergy/guo/Cleaned_data.xlsx` |
| Zhakparov 2026, SOS-ALL | https://doi.org/10.5281/zenodo.21064520 | `external_allergy/sosall/clinical_data_mlm_sos_all.txt` |
| Bergmann-Hug 2020, Bern | S1–S3 Tables of https://doi.org/10.1371/journal.pone.0238740 | `external_allergy/venom_pollen/pone.0238740.s002.docx`, `.s003.docx`, `.s004.docx` |
| van der Valk 2016, IDEAL cashew challenge | S1 Database of https://doi.org/10.1371/journal.pone.0151055 (published as screenshots; the transcribed table is available from the author) | `external_allergy/cashew_ideal/IDEAL_transcribed.csv` |
| McGowan 2015, URECA (ImmPort SDY1644) | `SDY1644-DR68_Tab.zip` from https://www.immport.org/shared/study/SDY1644 (free registration) | unzip to `external_allergy/ureca/tab/` |
| Perkin 2016, EAT (ImmPort SDY3091) | `SDY3091_DR58_ALL_DATA.zip` from https://www.immport.org/shared/study/SDY3091 (free registration) | unzip to `external_allergy/eat/unzipped/` |
| Du Toit 2015, LEAP (ImmPort SDY660) | `SDY660_DR58_ALL_DATA.zip` from https://www.immport.org/shared/study/SDY660 (free registration) | unzip to `external_allergy/leap/unzipped/` |

The ImmPort user agreement does not allow redistribution of these data; each user must download
them under their own account.

## Clinical datasets outside allergy (Supplementary Section S3)

| Data | Source | Place at |
|---|---|---|
| Heart disease (Detrano 1989) | UCI Machine Learning Repository, Heart Disease | `external/heart/processed.{cleveland,hungarian,switzerland,va}.data` |
| Thyroid (Quinlan 1996) | UCI, Thyroid Disease | `external/thyroid/allhypo.data`, `external/thyroid/allhypo.test` |
| Hepatitis (Diaconis & Efron 1983) | UCI, Hepatitis | `external/hepatitis/hepatitis.data` |
| Pima diabetes (Smith 1988) | OpenML data set 37 (ARFF) | `external/pima/diabetes.arff` |
| Chronic kidney disease (Qin 2020) | UCI, Chronic Kidney Disease (ARFF) | `external/ckd/chronic-kidney-disease.arff` |
| SUPPORT (Knaus 1995) | OpenML data set 46628 (ARFF) | `external/support2/Support2.arff` |
| PhysioNet Challenge 2012 (Che 2018) | https://physionet.org/content/challenge-2012/1.0.0/ (set A) | `external/physionet2012/set-a.zip`, `external/physionet2012/Outcomes-a.txt` |
