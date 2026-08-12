#permutations_10k.py
# Script 1 : permutations_10k.py (sauvegarde intermediaire)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

start = time.time()
FILE = "ml_ready_stat.txt"
df = pd.read_csv(FILE, sep="\t", low_memory=False)

stations = df["station"].unique()
np.random.seed(42)
np.random.shuffle(stations)
train_s = stations[:20]
test_s = stations[20:]

y = np.sqrt(df["delta_easting"]**2 + df["delta_northing"]**2).values
env_features = ["canopy_cover", "elevation", "skyview", "slope", "tri"]
X = df[env_features].fillna(df[env_features].median()).values

train_mask = df["station"].isin(train_s).values
test_mask = df["station"].isin(test_s).values
X_tr, X_te = X[train_mask], X[test_mask]
y_tr, y_te = y[train_mask], y[test_mask]

rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf.fit(X_tr, y_tr)
r2_obs = r2_score(y_te, rf.predict(X_te))

# Sauvegarde immediate
with open("results_reviewer_tests.txt", "a") as f:
    f.write(f"[PERMUTATIONS_10K] R2 observe: {r2_obs:.4f}\n")

n_perm = 10000
r2_null = np.zeros(n_perm)
for i in range(n_perm):
    y_perm = np.random.permutation(y_tr)
    rf_p = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_p.fit(X_tr, y_perm)
    r2_null[i] = r2_score(y_te, rf_p.predict(X_te))
    if (i+1) % 1000 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{n_perm}... ({elapsed:.0f}s)")
        with open("results_reviewer_tests.txt", "a") as f:
            f.write(f"  Progression {i+1}/{n_perm}, elapsed {elapsed:.0f}s\n")

p_val = (np.sum(r2_null >= r2_obs) + 1) / (n_perm + 1)
elapsed = time.time() - start

res = f"""[PERMUTATIONS_10K RESULTATS]
R2 observe: {r2_obs:.4f}
p-value (10000 permutations): {p_val:.5f}
R2 median sous H0: {np.median(r2_null):.4f}
Temps total: {elapsed:.0f}s
"""
print(res)
with open("results1_reviewer_tests.txt", "a") as f:
    f.write(res + "\n")