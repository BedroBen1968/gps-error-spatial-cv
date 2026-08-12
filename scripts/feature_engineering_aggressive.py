#feature_engineering_aggressive.py
# Script 5 : feature_engineering_aggressive.py
# feature_engineering_aggressive.py — CORRIGÉ
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

df_stat = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
df_mob = pd.read_csv("ml_ready_mob.txt", sep="\t", low_memory=False)

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy"]

for d in [df_stat, df_mob]:
    aspect_col = "aspect" if "aspect" in d.columns else "aspect_original"
    dummies = pd.get_dummies(d[aspect_col], prefix="aspect")
    for c in ["aspect_East", "aspect_North", "aspect_South", "aspect_West"]:
        if c not in dummies: dummies[c] = 0
    d["__asp_E"] = dummies.get("aspect_East", 0)
    d["__asp_N"] = dummies.get("aspect_North", 0)
    d["__asp_S"] = dummies.get("aspect_South", 0)
    d["__asp_W"] = dummies.get("aspect_West", 0)

poly_features = ["canopy_cover", "elevation", "skyview", "slope", "tri", "__asp_E", "__asp_N", "__asp_S", "__asp_W"]
X_stat_p = df_stat[poly_features].fillna(df_stat[poly_features].median())
X_mob_p = df_mob[poly_features].fillna(df_mob[poly_features].median())

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_stat_poly = poly.fit_transform(X_stat_p)
X_mob_poly = poly.transform(X_mob_p)

y_e = df_stat["delta_easting"].values
y_n = df_stat["delta_northing"].values
yme = df_mob["delta_easting"].values
ymn = df_mob["delta_northing"].values

# RF poly
rf_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_e.fit(X_stat_poly, y_e); rf_n.fit(X_stat_poly, y_n)
pe = rf_e.predict(X_mob_poly); pn = rf_n.predict(X_mob_poly)
rmse_rf_poly = np.sqrt(np.mean((yme - pe)**2 + (ymn - pn)**2))

# Ridge poly
sc = StandardScaler()
Xsp = sc.fit_transform(X_stat_poly)
Xmp = sc.transform(X_mob_poly)
ridge_e = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3).fit(Xsp, y_e)
ridge_n = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3).fit(Xsp, y_n)
pe_r = ridge_e.predict(Xmp); pn_r = ridge_n.predict(Xmp)
rmse_ridge_poly = np.sqrt(np.mean((yme - pe_r)**2 + (ymn - pn_r)**2))

rmse_raw = np.sqrt(np.mean(yme**2 + ymn**2))
rmse_rf_plain = 13.36

# Correction : calculer la conclusion AVANT la f-string
ameliore = min(rmse_rf_poly, rmse_ridge_poly) < rmse_rf_plain - 0.3
conclusion = "ameliore marginalement" if ameliore else "n'ameliore pas"

res = f"""[FEATURE_ENG_AGGRESSIVE RESULTATS]
RMSE raw mobile: {rmse_raw:.2f} m
RMSE RF plain (9 feat): {rmse_rf_plain:.2f} m
RMSE RF poly (deg 2): {rmse_rf_poly:.2f} m
RMSE Ridge poly (deg 2): {rmse_ridge_poly:.2f} m
Conclusion: le feature engineering {conclusion} le transfert.
"""
print(res)
with open("results5_reviewer_tests.txt", "a") as f:
    f.write(res + "\n")