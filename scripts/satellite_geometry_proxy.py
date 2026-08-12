#satellite_geometry_proxy.py
#Script10
# satellite_geometry_proxy.py
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

df["sat_geometry_proxy"] = df["hdop"] / np.sqrt(df["number_satellites"].clip(lower=1))

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy", "sat_geometry_proxy"]

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

# Avec proxy
rfe = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfn = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfe.fit(X_tr, ye_tr); rfn.fit(X_tr, yn_tr)
pe = rfe.predict(X_te); pn = rfn.predict(X_te)
rmse_proxy = np.sqrt(np.mean((ye_te - pe)**2 + (yn_te - pn)**2))
r2e = r2_score(ye_te, pe)
r2n = r2_score(yn_te, pn)

# Sans proxy (baseline 9 features)
base9 = ["canopy_cover", "elevation", "skyview", "slope", "tri",
         "hdop", "number_satellites", "horizontal_accuracy"]
feat9 = base9 + ["__asp_E", "__asp_N", "__asp_S", "__asp_W"]
X9 = df[feat9].fillna(df[feat9].median()).values
X9_tr, X9_te = X9[train_mask], X9[test_mask]
rfe9 = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfn9 = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfe9.fit(X9_tr, ye_tr); rfn9.fit(X9_tr, yn_tr)
pe9 = rfe9.predict(X9_te); pn9 = rfn9.predict(X9_te)
rmse_base = np.sqrt(np.mean((ye_te - pe9)**2 + (yn_te - pn9)**2))

conclusion = "n apporte" if rmse_proxy >= rmse_base - 0.3 else "apporte marginalement"

res = f"""[SATELLITE_GEOMETRY_PROXY RESULTATS]
RMSE baseline (9 feat)        : {rmse_base:.2f} m
RMSE avec proxy sat-geometry  : {rmse_proxy:.2f} m
R2 East avec proxy            : {r2e:+.4f}
R2 North avec proxy           : {r2n:+.4f}
Proxy importance              : {rfe.feature_importances_[8]:.4f} (E), {rfn.feature_importances_[8]:.4f} (N)
Conclusion : le proxy {conclusion} de l information.
"""
print(res)
with open("results10_reviewer_tests.txt", "w") as f:
    f.write(res)