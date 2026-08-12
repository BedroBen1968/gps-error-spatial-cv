#wp5_spatial_importance.py
# WP5 : importance des features sous validation spatiale (LOSO)
# Methode : remplacement par station donneuse (10 donneuses/feature/fold),
# coherent avec la permutation par blocs ; Delta RMSE 2D vs prediction intacte.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time

start = time.time()
df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
stations = df["station"].values
uniq = np.unique(stations)
asp = pd.to_numeric(df["aspect_original"], errors="coerce")
if asp.isna().all():
    asp = pd.Series(pd.factorize(df["aspect_original"])[0], dtype=float)
df["_asp"] = asp
feats = ["canopy_cover", "elevation", "skyview", "slope", "tri",
         "hdop", "number_satellites", "horizontal_accuracy", "_asp"]
dE = df["delta_easting"].values.astype(float)
dN = df["delta_northing"].values.astype(float)

def rf():
    return RandomForestRegressor(n_estimators=300, max_depth=8,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)

rng = np.random.default_rng(42)
K = 10  # donneuses par feature
delta = {f: [] for f in feats}

for k, s in enumerate(uniq):
    tr = stations != s
    te = ~tr
    X = df[feats].copy()
    med = X[tr].median()
    Xtr = X[tr].fillna(med)
    Xte = X[te].fillna(med)
    mE = rf().fit(Xtr.values, dE[tr])
    mN = rf().fit(Xtr.values, dN[tr])
    pE0, pN0 = mE.predict(Xte.values), mN.predict(Xte.values)
    rmse0 = np.sqrt(np.mean((dE[te]-pE0)**2 + (dN[te]-pN0)**2))
    tr_stations = uniq[uniq != s]
    for f in feats:
        dr = []
        donors = rng.choice(tr_stations, size=K, replace=False)
        for d in donors:
            Xp = Xte.copy()
            dv = df.loc[stations == d, f]
            dv = dv.fillna(med[f]).values
            Xp[f] = rng.choice(dv, size=len(Xp), replace=True)
            pE = mE.predict(Xp.values); pN = mN.predict(Xp.values)
            dr.append(np.sqrt(np.mean((dE[te]-pE)**2 + (dN[te]-pN)**2)) - rmse0)
        delta[f].append(np.mean(dr))
    print(f"fold {k+1}/30 ({time.time()-start:.0f}s)")

lines = ["WP5 : importance sous LOSO (Delta RMSE 2D en m, remplacement par station donneuse)",
         f"{'feature':<22s}{'dRMSE moy':>10s}{'IC boot 95%':>20s}"]
for f in feats:
    a = np.array(delta[f])
    boots = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lines.append(f"{f:<22s}{a.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]")
out = "\n".join(lines) + f"\nTemps : {time.time()-start:.0f}s\n"
print("\n" + out)
open("wp5_results.txt", "w").write(out)
print("Sauvegarde : wp5_results.txt")