#reconcile_permutation_features.py
# reconcile_permutation_features.py
# Version corrigee : permutation par blocs-stations (null valide sous autocorrelation intra-station)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)

# Split fixe 20/10 (identique au Script 1)
stations = df["station"].unique()
np.random.seed(42)
np.random.shuffle(stations)
train_s = stations[:20]
test_s = stations[20:]
train_mask = df["station"].isin(train_s).values
test_mask = df["station"].isin(test_s).values

# Labels de station pour chaque ligne d'entrainement (necessaire pour la permutation par blocs)
stations_tr = df["station"].values[train_mask]

y = np.sqrt(df["delta_easting"]**2 + df["delta_northing"]**2).values
y_tr, y_te = y[train_mask], y[test_mask]

# Configurations a tester
configs = {}

# A) 5 features env (Script 1)
env5 = ["canopy_cover", "elevation", "skyview", "slope", "tri"]
X_a = df[env5].fillna(df[env5].median()).values
configs["A_5env"] = X_a

# B) 5 env + aspect dummies
aspect_dummies = pd.get_dummies(df["aspect"] if "aspect" in df.columns else df["aspect_original"], prefix="aspect")
X_b = pd.concat([df[env5], aspect_dummies], axis=1).fillna(df[env5].median()).values
configs["B_5env+aspect"] = X_b

# C) 5 env + aspect + hdop + number_satellites
if "hdop" in df.columns and "number_satellites" in df.columns:
    X_c = pd.concat([df[env5 + ["hdop", "number_satellites"]], aspect_dummies], axis=1)
    X_c = X_c.fillna(X_c.median()).values
    configs["C_5env+aspect+tel"] = X_c

# D) Tout (9 features + hacc)
if "horizontal_accuracy" in df.columns:
    X_d = pd.concat([df[env5 + ["hdop", "number_satellites", "horizontal_accuracy"]], aspect_dummies], axis=1)
    X_d = X_d.fillna(X_d.median()).values
    configs["D_all9"] = X_d

n_perm = 1000  # suffisant pour comparer les configs

def block_permute_targets(y_train, station_labels, train_stations, rng):
    """
    Permutation par blocs-stations.
    Chaque station d'entrainement recoit des valeurs cibles re-echantillonnees
    (avec remise) depuis une station donneuse tiree au hasard.
    Preserve la distribution intra-station, detruit l'association
    station-environnement. Null valide sous autocorrelation intra-station.
    """
    perm_stations = rng.permutation(train_stations)
    donor_map = dict(zip(train_stations, perm_stations))
    y_perm = np.empty_like(y_train)
    for s in train_stations:
        idx = (station_labels == s)
        donor_vals = y_train[station_labels == donor_map[s]]
        y_perm[idx] = rng.choice(donor_vals, size=idx.sum(), replace=True)
    return y_perm

results = []
for name, X in configs.items():
    X_tr, X_te = X[train_mask], X[test_mask]

    # Modele observe
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
    rf.fit(X_tr, y_tr)
    r2_obs = r2_score(y_te, rf.predict(X_te))

    # 1000 permutations par blocs-stations
    r2_null = np.zeros(n_perm)
    for i in range(n_perm):
        rng = np.random.default_rng(1000 + i)  # graine reproductible par iteration
        y_perm = block_permute_targets(y_tr, stations_tr, train_s, rng)
        rf_p = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
        rf_p.fit(X_tr, y_perm)
        r2_null[i] = r2_score(y_te, rf_p.predict(X_te))

    p_val = (np.sum(r2_null >= r2_obs) + 1) / (n_perm + 1)

    res = {
        "config": name,
        "n_features": X.shape[1],
        "r2_obs": r2_obs,
        "p_value": p_val,
        "r2_median_h0": np.median(r2_null),
        "n_train": len(y_tr),
        "n_test": len(y_te)
    }
    results.append(res)
    print(f"\n[{name}] n_features={X.shape[1]}, R2_obs={r2_obs:.4f}, p={p_val:.4f}, median_H0={np.median(r2_null):.4f}")

# Tableau recap
print("\n" + "="*70)
print("TABLEAU RECONCILIATION (PERMUTATION PAR BLOCS-STATIONS)")
print("="*70)
print(f"{'Config':<20s} {'n_feat':>6s} {'R2_obs':>8s} {'p_value':>8s} {'median_H0':>10s}")
print("-"*70)
for r in results:
    print(f"{r['config']:<20s} {r['n_features']:>6d} {r['r2_obs']:>8.4f} {r['p_value']:>8.4f} {r['r2_median_h0']:>10.4f}")

# Sauvegarde
with open("results_reconcile_permutation_block.txt", "w") as f:
    f.write("RECONCILIATION PERMUTATION PAR BLOCS-STATIONS : EFFET DU NOMBRE DE FEATURES\n")
    f.write("Null : re-appariement aleatoire des stations d'entrainement,\n")
    f.write("re-echantillonnage avec remise dans la station donneuse.\n")
    f.write("="*70 + "\n")
    for r in results:
        f.write(f"{r['config']:<20s} n={r['n_features']:2d}  R2={r['r2_obs']:+.4f}  p={r['p_value']:.4f}  H0_med={r['r2_median_h0']:+.4f}\n")

print("\nSauvegarde : results_reconcile_permutation_block.txt")