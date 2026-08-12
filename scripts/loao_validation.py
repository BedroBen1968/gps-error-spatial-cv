#loao_validation.py
# Section 3.3 : leave-one-aspect-out (4 folds), recalcul traçable
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
asp = df["aspect_original"].astype(str).values
feats = ["canopy_cover","elevation","skyview","slope","tri",
         "hdop","number_satellites","horizontal_accuracy"]
X = df[feats]
dE = df["delta_easting"].values.astype(float)
dN = df["delta_northing"].values.astype(float)

def rf(): return RandomForestRegressor(n_estimators=300, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1)

pE = np.empty(len(df)); pN = np.empty(len(df))
fr, fu, fb = [], [], []
for a in np.unique(asp):
    tr = asp != a; te = ~tr
    med = X[tr].median()
    pE[te] = rf().fit(X[tr].fillna(med).values, dE[tr]).predict(X[te].fillna(med).values)
    pN[te] = rf().fit(X[tr].fillna(med).values, dN[tr]).predict(X[te].fillna(med).values)
    bE, bN = dE[tr].mean(), dN[tr].mean()
    fr.append(np.sqrt(np.mean((dE[te]-pE[te])**2 + (dN[te]-pN[te])**2)))
    fu.append(np.sqrt(np.mean(dE[te]**2 + dN[te]**2)))
    fb.append(np.sqrt(np.mean((dE[te]-bE)**2 + (dN[te]-bN)**2)))

out = ["LOAO (4 folds aspect), recalcul traçable :",
 f"  RMSE modele   : {np.mean(fr):.2f} +/- {np.std(fr):.2f} m",
 f"  RMSE brut     : {np.mean(fu):.2f} +/- {np.std(fu):.2f} m",
 f"  RMSE baseline : {np.mean(fb):.2f} +/- {np.std(fb):.2f} m",
 f"  R2 pooled     : East {r2_score(dE,pE):+.3f} / North {r2_score(dN,pN):+.3f}"]
print("\n".join(out))
open("results_loao_final.txt","w").write("\n".join(out) + "\n")
print("Sauvegarde : results_loao_final.txt")