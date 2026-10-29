"""
00_download_nhanes.py
=====================
Download the NHANES 2005-2006 files the FLIP allergy analysis needs.

The files are not stored in the repository; this script writes them to
../data_raw and the codebooks to ../doc_nhanes.

Everything is public and needs no registration, licence or login.
Source: US CDC / National Center for Health Statistics.

Run:  python 00_download_nhanes.py
"""
import hashlib
import os
import urllib.request

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_raw")
DOC = os.path.join(os.path.dirname(__file__), "..", "doc_nhanes")

# file -> what it supplies
FILES = {
    "AL_IGE_D": "Allergen-specific IgE (19 allergens) + total IgE   <- the core file",
    "DEMO_D":   "age, sex, race/ethnicity, poverty-income ratio, survey design variables",
    "AGQ_D":    "allergy questionnaire (hay fever, allergy symptoms, eczema)",
    "MCQ_D":    "medical conditions questionnaire (asthma)",
    "RDQ_D":    "respiratory questionnaire (wheeze, cough)",
    "BMX_D":    "body measurements (BMI)",
    "SMQ_D":    "smoking questionnaire",
    "ALDUST_D": "house dust allergen concentrations (not used in the primary analysis)",
    "CBC_D":    "complete blood count (eosinophils; optional extra predictor)",
}


def fetch(name, ext, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.{ext}")
    url = f"{BASE}/{name}.{ext}"
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        print(f"  already present: {name}.{ext}")
        return path
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


if __name__ == "__main__":
    print("NHANES 2005-2006 data files")
    for name, what in FILES.items():
        p = fetch(name, "xpt", OUT)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        print(f"    {name:10s} {os.path.getsize(p):>9,d} bytes  sha256:{h}  | {what}")

    print("\nCodebooks (HTML) - needed to verify variable labels, never guess them")
    for name in FILES:
        fetch(name, "htm", DOC)

    print("\nDone. Next: python 01b_build_nhanes3.py")
