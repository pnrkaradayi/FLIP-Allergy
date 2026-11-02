"""Secondary analyses for Task 1: a second outcome and the amount of missingness.

Run:  python a02b_sensitivity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a02_risk_decision import main  # noqa: E402

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "asthma"):
        main(10, 5, "y_current_asthma", scales=(1.0,), tag_out="")
    if which in ("all", "scale"):
        main(6, 6, "y_allergic_disease", scales=(0.5, 1.5), tag_out="_scale")
