"""
quickstart.py
=============
FLIP on simulated data, no download needed (a few seconds).

Four allergens are simulated with cross-reactivity: the two house dust mites share one protein
family, cat and dog another. Values below the 0.35 kU/L detection limit are only known to be
below it. FLIP learns the dependence structure, then, for a new patient in whom only two of
the four tests were ordered, predicts the two unordered ones and the probability that complete
testing would change a decision.

Run:  python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from flipx import CensoredMVN  # noqa: E402

ALLERGENS = ["D. pteronyssinus", "D. farinae", "cat", "dog"]
rng = np.random.default_rng(1)

# ---- 1. simulated training panels (log10 kU/L) with two cross-reactive families
n = 2000
loadings = np.array([[0.9, 0.0], [0.9, 0.0], [0.0, 0.8], [0.0, 0.8]])
Z = rng.normal(size=(n, 2)) @ loadings.T + 0.3 * rng.normal(size=(n, 4)) - 0.8
cens = np.full(4, np.log10(0.35))                 # detection limit 0.35 kU/L
status = np.where(Z > cens, 0, 1)                 # 0 = measured, 1 = below the limit
Y = np.where(status == 0, Z, cens)
X = np.ones((n, 1))                               # covariates (intercept only here)

# ---- 2. fit the censored model with a global factor plus the two protein families
families = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]], bool)
model = CensoredMVN(cens, "factor", families, n_iter=150, burn=60, seed=0).fit(Y, status, X)

# ---- 3. a new patient: D. pteronyssinus 8 kU/L, cat negative; D. farinae and dog not ordered
y_new = np.array([[np.log10(8.0), 0.0, cens[2], 0.0]])
st_new = np.array([[0, 2, 1, 2]])                 # 2 = not ordered
draws = model.conditional_draws(y_new, st_new, np.ones((1, 1)), seed=1)[0]   # completions x allergens

p_pos = (draws > cens).mean(0)
for a, s, p in zip(ALLERGENS, st_new[0], p_pos):
    tag = {0: "ordered, positive", 1: "ordered, negative", 2: "NOT ordered"}[s]
    print(f"  {a:17s} {tag:18s} P(positive) = {p:.2f}")

# ---- 4. FLIP score of two decisions made from the ordered tests only
mite = (draws[:, [0, 1]] > cens[[0, 1]]).any(1)   # 'sensitised to house dust mite' (issued: yes)
pet = (draws[:, [2, 3]] > cens[[2, 3]]).any(1)    # 'sensitised to a furred pet' (issued: no)
print(f"\nFLIP score, decision 'mite-sensitised = yes':   {np.mean(~mite):.3f}")
print(f"FLIP score, decision 'pet-sensitised = no':      {np.mean(pet):.3f}")
print("A small score means complete testing would almost never change the decision;\n"
      "a large one means the unordered test (here dog) is worth ordering.")
