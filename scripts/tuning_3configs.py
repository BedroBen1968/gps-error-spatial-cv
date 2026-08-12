#tuning_3configs.py
#Script 7
"""
Script 7 : Tuning 3 configs RF
Reviewer #1 : verifier que 3 configs radicalement differentes convergent vers le meme echec.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
stations = df["station"].unique()
np.random.seed(42)
np.random.shuffle(stations)
train_s = stations[:20]
test_s = stations[20:]

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy"]

# Aspect dummies
dummies = pd.get_dummies(df["aspect"] if "aspect" in df.columns else df["aspect_original"], prefix="aspect")
for c in ["aspect_East", "aspect_North", "aspect_South", "aspect_West"]:
    if c not in dummies: dummies[c] = 0
df["__asp_E"] = dummies.get("aspect_East", 0)
df["__asp_N"] = dummies.get("aspect_North", 0)
df["__asp_S"] = dummies.get("aspect_South", 0)
df["__asp_W"] = dummies.get("aspect_West", 0)

feat_names = base + ["__asp_E", "__asp_N", "__asp_S", "__asp_W"]
X = df[feat_names].fillna(df[feat_names].median()).values
y_e = df["delta_easting"].values
y_n = df["delta_northing"].values

train_mask = df["station"].isin(train_s).values
test_mask = df["station"].isin(test_s).values
X_tr, X_te = X[train_mask], X[test_mask]
ye_tr, ye_te = y_e[train_mask], y_e[test_mask]
yn_tr, yn_te = y_n[train_mask], y_n[test_mask]

configs = [
    ("RF_default", 300, 8, 5),
    ("RF_deep", 200, 15, 3),
    ("RF_shallow", 50, 15, 1),
]

res = "[TUNING_3CONFIGS RESULTATS]\n"
for name, n_est, max_d, min_l in configs:
    rfe = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_l, random_state=42)
    rfn = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_l, random_state=42)
    rfe.fit(X_tr, ye_tr); rfn.fit(X_tr, yn_tr)
    pe = rfe.predict(X_te); pn = rfn.predict(X_te)
    rmse = np.sqrt(np.mean((ye_te - pe)**2 + (yn_te - pn)**2))
    r2e = r2_score(ye_te, pe)
    r2n = r2_score(yn_te, pn)
    res += f"{name:15s} : RMSE={rmse:.2f} m, R2_E={r2e:+.3f}, R2_N={r2n:+.3f}\n"
    print(f"{name} done")

print(res)
with open("results7_reviewer_tests.txt", "w") as f:
    f.write(res)