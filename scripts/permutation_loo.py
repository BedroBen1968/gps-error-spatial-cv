#permutation_loo.py
"""
permutation_loo.py
Version corrigee : test de permutation par blocs-stations,
leave-one-site-out complet (30 folds), cible = precision.
Features conformes a la formulation (iii) du manuscrit :
covariables environnementales + aspect, sans HDOP, sans nsat,
sans horizontal_accuracy.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
df = pd.read_csv(BASE / "ml_ready_stat.txt", sep="\t", low_memory=False)

# Formulation (iii) : environnement seul (+ aspect), sans metadonnees recepteur
features = ["canopy_cover", "elevation", "skyview", "slope", "tri", "aspect_original"]
X = df[features].fillna(df[features].median()).values
y = df["precision"].values
sites = df["station"].values
unique_sites = np.unique(sites)

n_perm = 1000   # reconnaissance ; passer a 1000 pour le chiffre final si p proche du seuil

def loo_r2(X, y_vec, sites):
    preds, trues = [], []
    for s in unique_sites:
        mask_tr = sites != s
        mask_te = sites == s
        rf = RandomForestRegressor(n_estimators=50, max_depth=8,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X[mask_tr], y_vec[mask_tr])
        preds.extend(rf.predict(X[mask_te]))
        trues.extend(y_vec[mask_te])
    return r2_score(trues, preds)

def block_permute_targets(y_vec, station_labels, all_stations, rng):
    """
    Permutation par blocs-stations sur l'ensemble des 30 stations.
    Chaque station recoit des valeurs de precision re-echantillonnees
    (avec remise) depuis une station donneuse tiree au hasard.
    Preserve la distribution intra-station, detruit l'association
    station-environnement. Le LOSO est ensuite execute tel quel.
    """
    perm_stations = rng.permutation(all_stations)
    donor_map = dict(zip(all_stations, perm_stations))
    y_perm = np.empty_like(y_vec)
    for s in all_stations:
        idx = (station_labels == s)
        donor_vals = y_vec[station_labels == donor_map[s]]
        y_perm[idx] = rng.choice(donor_vals, size=idx.sum(), replace=True)
    return y_perm

start = time.time()
print("Calcul R2 observe (30 folds LOO)...")
r2_obs = loo_r2(X, y, sites)
print(f"R2 observe : {r2_obs:.4f}")

print(f"\nPermutation par blocs-stations ({n_perm} iterations)...")
r2_null = []
for i in range(n_perm):
    rng = np.random.default_rng(2000 + i)  # graine reproductible par iteration
    y_perm = block_permute_targets(y, sites, unique_sites, rng)
    r2_null.append(loo_r2(X, y_perm, sites))
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{n_perm} termine ({elapsed:.0f}s)")

r2_null = np.array(r2_null)
p = (np.sum(r2_null >= r2_obs) + 1) / (n_perm + 1)

res = f"""{'='*50}
TEST PERMUTATION PAR BLOCS-STATIONS (LOSO, precision)
Features : {', '.join(features)}
R2 observe      : {r2_obs:.4f}
R2 null (moy)   : {r2_null.mean():.4f}
R2 null (median): {np.median(r2_null):.4f}
R2 null (max)   : {r2_null.max():.4f}
P-value ({n_perm} perm) : {p:.4f}
Significatif ?  : {'OUI' if p < 0.05 else 'NON'}
Temps total     : {time.time()-start:.0f}s
{'='*50}"""
print(res)

with open(BASE / "results_permutation_loo_block.txt", "w") as f:
    f.write(res + "\n")
print("\nSauvegarde : results_permutation_loo_block.txt")