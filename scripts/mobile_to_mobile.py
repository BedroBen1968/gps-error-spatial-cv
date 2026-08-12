#mobile_to_mobile.py
# Script 4 : mobile_to_mobile.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("ml_ready_mob.txt", sep="\t", low_memory=False)
base_features = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                 "hdop", "number_satellites", "horizontal_accuracy"]
dummies = pd.get_dummies(df["aspect"], prefix="aspect")
X = pd.concat([df[base_features], dummies], axis=1).fillna(df[base_features].median())
y_e = df["delta_easting"].values
y_n = df["delta_northing"].values

X_tr, X_te, ye_tr, ye_te, yn_tr, yn_te = train_test_split(
    X, y_e, y_n, test_size=0.2, random_state=42)

rf_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_e.fit(X_tr, ye_tr); rf_n.fit(X_tr, yn_tr)
pred_e = rf_e.predict(X_te); pred_n = rf_n.predict(X_te)
rmse = np.sqrt(np.mean((ye_te - pred_e)**2 + (yn_te - pred_n)**2))
rmse_raw = np.sqrt(np.mean(ye_te**2 + yn_te**2))
base_e = np.mean(ye_tr); base_n = np.mean(yn_tr)
rmse_base = np.sqrt(np.mean((ye_te - base_e)**2 + (yn_te - base_n)**2))

res = f"""[MOBILE_TO_MOBILE RESULTATS]
N train: {len(X_tr)}, N test: {len(X_te)}
RMSE raw (test): {rmse_raw:.2f} m
RMSE baseline (mean): {rmse_base:.2f} m
RMSE RF (mobile->mob): {rmse:.2f} m
Conclusion: {'le modele apprend' if rmse < rmse_base else 'PAS de signal exploitable meme en mobile->mobile'}
"""
print(res)
with open("results4_reviewer_tests.txt", "a") as f:
    f.write(res + "\n")