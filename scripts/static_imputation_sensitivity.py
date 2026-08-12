#static_imputation_sensitivity.py
#script9
# static_imputation_sensitivity.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
stations = df["station"].unique()
np.random.seed(42)
np.random.shuffle(stations)
train_s = stations[:20]
test_s = stations[20:]

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy"]

dummies = pd.get_dummies(df["aspect"] if "aspect" in df.columns else df["aspect_original"], prefix="aspect")
for c in ["aspect_East", "aspect_North", "aspect_South", "aspect_West"]:
    if c not in dummies: dummies[c] = 0
df["__asp_E"] = dummies.get("aspect_East", 0)
df["__asp_N"] = dummies.get("aspect_North", 0)
df["__asp_S"] = dummies.get("aspect_South", 0)
df["__asp_W"] = dummies.get("aspect_West", 0)

feat_names = base + ["__asp_E", "__asp_N", "__asp_S", "__asp_W"]

# Avec imputation
X_full = df[feat_names].fillna(df[feat_names].median()).values
y_e = df["delta_easting"].values
y_n = df["delta_northing"].values
train_mask = df["station"].isin(train_s).values
test_mask = df["station"].isin(test_s).values

X_tr, X_te = X_full[train_mask], X_full[test_mask]
ye_tr, ye_te = y_e[train_mask], y_e[test_mask]
yn_tr, yn_te = y_n[train_mask], y_n[test_mask]

rfe = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfn = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rfe.fit(X_tr, ye_tr); rfn.fit(X_tr, yn_tr)
pe = rfe.predict(X_te); pn = rfn.predict(X_te)
rmse_full = np.sqrt(np.mean((ye_te - pe)**2 + (yn_te - pn)**2))

# Sans imputation
mask_complete = df[base[:5]].notna().all(axis=1)
df_comp = df[mask_complete]
stations_c = df_comp["station"].unique()
np.random.seed(42)
np.random.shuffle(stations_c)
train_c = stations_c[:max(1, int(len(stations_c)*0.67))]
test_c = stations_c[max(1, int(len(stations_c)*0.67)):]

Xc = df_comp[feat_names].fillna(df_comp[feat_names].median()).values
yec = df_comp["delta_easting"].values
ync = df_comp["delta_northing"].values
tr_mask = df_comp["station"].isin(train_c).values
te_mask = df_comp["station"].isin(test_c).values

if te_mask.sum() > 0:
    rfe_c = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
    rfn_c = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
    rfe_c.fit(Xc[tr_mask], yec[tr_mask]); rfn_c.fit(Xc[tr_mask], ync[tr_mask])
    pec = rfe_c.predict(Xc[te_mask]); pnc = rfn_c.predict(Xc[te_mask])
    rmse_comp = np.sqrt(np.mean((yec[te_mask] - pec)**2 + (ync[te_mask] - pnc)**2))
else:
    rmse_comp = np.nan

res = f"""[STATIC_IMPUTATION_SENSITIVITY RESULTATS]
RMSE avec imputation (tout)     : {rmse_full:.2f} m
RMSE sans imputation (complet)  : {rmse_comp:.2f} m
Points complets                 : {mask_complete.sum()}/{len(df)}
Conclusion : les imputations {'ne changent' if abs(rmse_full-rmse_comp)<0.5 else 'changent'} le resultat.
"""
print(res)
with open("results9_reviewer_tests.txt", "w") as f:
    f.write(res)