#static_to_mobile_transfer_9features.py
"""
static_to_mobile_transfer_9features.py

Transfert static-to-mobile avec les 9 features completes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Charger les deux datasets
df_stat = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
df_mob = pd.read_csv("ml_ready_mob.txt", sep="\t", low_memory=False)

print(f"Static : {df_stat.shape}")
print(f"Mobile : {df_mob.shape}")

# ------------------------------------------------------------------
# 1. IDENTIFIER LES COLONNES
# ------------------------------------------------------------------
# Features de base (8 numeriques)
base_features = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                 "hdop", "number_satellites", "horizontal_accuracy"]

# Aspect : trouver le bon nom dans chaque dataset
aspect_stat = None
for c in ['aspect', 'aspect_original', 'aspect_cat']:
    if c in df_stat.columns:
        aspect_stat = c
        break

aspect_mob = None
for c in ['aspect', 'aspect_original', 'aspect_cat']:
    if c in df_mob.columns:
        aspect_mob = c
        break

print(f"Aspect static : {aspect_stat}")
print(f"Aspect mobile : {aspect_mob}")

# ------------------------------------------------------------------
# 2. ONE-HOT ENCODING DE ASPECT (aligne les deux datasets)
# ------------------------------------------------------------------
# Dummies pour le static
if aspect_stat:
    dummies_stat = pd.get_dummies(df_stat[aspect_stat], prefix="aspect")
else:
    dummies_stat = pd.DataFrame()

# Dummies pour le mobile
if aspect_mob:
    dummies_mob = pd.get_dummies(df_mob[aspect_mob], prefix="aspect")
else:
    dummies_mob = pd.DataFrame()

# Aligner : meme colonnes dans les deux
all_aspect_cols = sorted(list(set(dummies_stat.columns) | set(dummies_mob.columns)))
for c in all_aspect_cols:
    if c not in dummies_stat.columns:
        dummies_stat[c] = 0
    if c not in dummies_mob.columns:
        dummies_mob[c] = 0

print(f"Categories aspect : {all_aspect_cols}")

# ------------------------------------------------------------------
# 3. PREPARER X ET y POUR LE STATIC (entrainement complet)
# ------------------------------------------------------------------
X_stat = pd.concat([df_stat[base_features], dummies_stat[all_aspect_cols]], axis=1)
X_stat = X_stat.fillna(X_stat.median())
y_east_stat = df_stat["delta_easting"].values
y_north_stat = df_stat["delta_northing"].values

# ------------------------------------------------------------------
# 4. PREPARER X ET y POUR LE MOBILE (test)
# ------------------------------------------------------------------
X_mob = pd.concat([df_mob[base_features], dummies_mob[all_aspect_cols]], axis=1)
# Imputer les lignes manquantes par la mediane du mobile
X_mob = X_mob.fillna(X_mob.median())

y_east_mob = df_mob["delta_easting"].values
y_north_mob = df_mob["delta_northing"].values

# ------------------------------------------------------------------
# 5. TRANSFERT 9 FEATURES
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRANSFERT STATIC -> MOBILE (9 features)")
print("=" * 60)

rf_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)

rf_e.fit(X_stat, y_east_stat)
rf_n.fit(X_stat, y_north_stat)

pred_e = rf_e.predict(X_mob)
pred_n = rf_n.predict(X_mob)

# Erreurs
raw_error = np.sqrt(y_east_mob**2 + y_north_mob**2)
corrected_error = np.sqrt((y_east_mob - pred_e)**2 + (y_north_mob - pred_n)**2)

rmse_raw = np.sqrt(np.mean(raw_error**2))
rmse_corr = np.sqrt(np.mean(corrected_error**2))

print(f"RMSE raw (non corrige)      : {rmse_raw:.2f} m")
print(f"RMSE corrected (9 features) : {rmse_corr:.2f} m")
print(f"Changement                  : {((rmse_corr/rmse_raw)-1)*100:+.1f}%")

# ------------------------------------------------------------------
# 6. COMPARAISON : TRANSFERT 4 FEATURES (ancien)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARAISON : 4 features seulement")
print("=" * 60)

feat4 = ["canopy_cover", "hdop", "number_satellites", "horizontal_accuracy"]
X_stat4 = df_stat[feat4].fillna(df_stat[feat4].median())
X_mob4 = df_mob[feat4].fillna(df_mob[feat4].median())

rf_e4 = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_n4 = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_e4.fit(X_stat4, y_east_stat)
rf_n4.fit(X_stat4, y_north_stat)

pred_e4 = rf_e4.predict(X_mob4)
pred_n4 = rf_n4.predict(X_mob4)

corr_err4 = np.sqrt((y_east_mob - pred_e4)**2 + (y_north_mob - pred_n4)**2)
rmse_corr4 = np.sqrt(np.mean(corr_err4**2))

print(f"RMSE corrected (4 features) : {rmse_corr4:.2f} m")
print(f"Ancien resultat rapporte    : 13.67 m")

if rmse_corr < rmse_corr4:
    conclusion = "les 9 features ameliorent le transfert vs 4 features"
else:
    conclusion = "les 9 features n'ameliorent PAS le transfert vs 4 features"
print(f"\nConclusion : {conclusion}")

# ------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (9 features)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (9 features, moyenne East/North)")
print("=" * 60)

importances = (rf_e.feature_importances_ + rf_n.feature_importances_) / 2
for name, imp in zip(X_stat.columns, importances):
    print(f"  {name:30s} : {imp:.3f}")