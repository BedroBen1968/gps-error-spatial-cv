"""
ml_improvement.py
Teste 4 pistes d'amélioration :
  1. SVM-RBF (sans ACP, standardisation)
  2. Feature engineering (interactions + polynomial)
  3. Target = résidus GLMM (enlève effet station)
  4. Modèle par groupe (prédiction agrégée)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
df = pd.read_csv(BASE / "ml_ready_stat.txt", sep="	", low_memory=False)

feature_cols = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                "hdop", "number_satellites", "horizontal_accuracy", "aspect_original"]
X_base = df[feature_cols].fillna(df[feature_cols].median())
y_east = df["delta_easting"].values
y_north = df["delta_northing"].values
sites = df["station"].values

unique_sites = np.unique(sites)

# ═══════════════════════════════════════════════════════════════════
#  FONCTION DE CV SPATIALE
# ═══════════════════════════════════════════════════════════════════

def spatial_cv(X, y_east, y_north, sites, model_e, model_n, label):
    """Leave-one-site-out sur (delta_easting, delta_northing)."""
    rmse_before, rmse_after = [], []
    r2_e, r2_n = [], []

    for s in unique_sites:
        mask_tr = sites != s
        mask_te = sites == s
        X_tr, X_te = X[mask_tr], X[mask_te]
        ye_tr, ye_te = y_east[mask_tr], y_east[mask_te]
        yn_tr, yn_te = y_north[mask_tr], y_north[mask_te]

        # RMSE avant correction
        rmse_before.append(np.sqrt(np.mean(ye_te**2 + yn_te**2)))

        # Clone models
        me = model_e.__class__(**model_e.get_params())
        mn = model_n.__class__(**model_n.get_params())
        me.fit(X_tr, ye_tr)
        mn.fit(X_tr, yn_tr)

        pe, pn = me.predict(X_te), mn.predict(X_te)
        err = np.sqrt((ye_te - pe)**2 + (yn_te - pn)**2)
        rmse_after.append(np.sqrt(np.mean(err**2)))

        if len(ye_te) > 1:
            r2_e.append(r2_score(ye_te, pe))
            r2_n.append(r2_score(yn_te, pn))

    print(f"\n--- {label} ---")
    print(f"  RMSE avant : {np.mean(rmse_before):.2f} ± {np.std(rmse_before):.2f}")
    print(f"  RMSE après : {np.mean(rmse_after):.2f} ± {np.std(rmse_after):.2f}")
    print(f"  Réduction  : {(1 - np.mean(rmse_after)/np.mean(rmse_before))*100:.1f}%")
    print(f"  R² East    : {np.mean(r2_e):.3f}")
    print(f"  R² North   : {np.mean(r2_n):.3f}")
    return rmse_before, rmse_after

print("=" * 60)
print("PISTE 1 : SVM-RBF (standardisé, sans ACP)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

# SVM est lent sur 8466 points — on teste sur un sous-échantillon rapide d'abord
sample_idx = np.random.choice(len(df), size=2000, replace=False)
X_svm = X_scaled[sample_idx]
ye_svm = y_east[sample_idx]
yn_svm = y_north[sample_idx]
sites_svm = sites[sample_idx]

svm_e = SVR(kernel='rbf', C=1.0, gamma='scale')
svm_n = SVR(kernel='rbf', C=1.0, gamma='scale')

# CV rapide sur sous-échantillon
rmse_b, rmse_a = [], []
for s in np.unique(sites_svm):
    mask_tr = sites_svm != s
    mask_te = sites_svm == s
    if mask_te.sum() < 5:
        continue
    svm_e.fit(X_svm[mask_tr], ye_svm[mask_tr])
    svm_n.fit(X_svm[mask_tr], yn_svm[mask_tr])
    pe = svm_e.predict(X_svm[mask_te])
    pn = svm_n.predict(X_svm[mask_te])
    rmse_b.append(np.sqrt(np.mean(ye_svm[mask_te]**2 + yn_svm[mask_te]**2)))
    rmse_a.append(np.sqrt(np.mean((ye_svm[mask_te]-pe)**2 + (yn_svm[mask_te]-pn)**2)))

print(f"  [Sous-échantillon N=2000, SVR RBF]")
print(f"  RMSE avant : {np.mean(rmse_b):.2f}")
print(f"  RMSE après : {np.mean(rmse_a):.2f}")
print(f"  Réduction  : {(1 - np.mean(rmse_a)/np.mean(rmse_b))*100:.1f}%")

print("\n" + "=" * 60)
print("PISTE 2 : FEATURE ENGINEERING (interactions + polynomial)")
print("=" * 60)

# Interactions physiques pertinentes
X_fe = X_base.copy()
X_fe["canopy_x_skyview"] = X_fe["canopy_cover"] * X_fe["skyview"]
X_fe["slope_x_aspect"] = X_fe["slope"] * np.sin(np.radians(X_fe["aspect_original"]))
X_fe["tri_x_elevation"] = X_fe["tri"] * X_fe["elevation"]
X_fe["hdop_x_nsat"] = X_fe["hdop"] * X_fe["number_satellites"]
X_fe["canopy_sq"] = X_fe["canopy_cover"] ** 2
X_fe["skyview_sq"] = X_fe["skyview"] ** 2
X_fe["slope_sq"] = X_fe["slope"] ** 2

rf_fe_e = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf_fe_n = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
spatial_cv(X_fe.values, y_east, y_north, sites, rf_fe_e, rf_fe_n, "RF + Feature Engineering")

print("\n" + "=" * 60)
print("PISTE 3 : TARGET = RÉSIDUS GLMM (enlève effet station)")
print("=" * 60)

# On enlève la moyenne par station (l'effet idiosyncrasique)
station_mean_e = df.groupby("station")["delta_easting"].transform("mean")
station_mean_n = df.groupby("station")["delta_northing"].transform("mean")

y_east_resid = y_east - station_mean_e.values
y_north_resid = y_north - station_mean_n.values

print(f"  Variance delta_easting brute : {np.var(y_east):.2f}")
print(f"  Variance résiduelle (sans effet station) : {np.var(y_east_resid):.2f}")
print(f"  % variance expliquée par station : {(1 - np.var(y_east_resid)/np.var(y_east))*100:.1f}%")

rf_res_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_res_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
spatial_cv(X_base.values, y_east_resid, y_north_resid, sites, rf_res_e, rf_res_n, 
           "RF sur RÉSIDUS (effet station retiré)")

print("\n" + "=" * 60)
print("PISTE 4 : MODÈLE PAR GROUPE (prédiction agrégée)")
print("=" * 60)

# Au lieu de prédire point par point, prédire le biais MOYEN par groupe aspect/canopy
# Puis appliquer cette correction à tous les points du groupe
df["group"] = df["aspect"].astype(str) + "_" + (df["canopy_cover"] > 50).map({True: "closed", False: "open"})

# Moyenne observée par groupe
group_means = df.groupby("group")[["delta_easting", "delta_northing"]].mean()

# Leave-one-site-out : prédire la moyenne du groupe du site test
rmse_before_g, rmse_after_g = [], []
for s in unique_sites:
    mask_te = sites == s
    # Le groupe du site test
    test_groups = df.loc[mask_te, "group"].unique()
    # Moyennes des groupes, calculées sur les AUTRES sites
    other_df = df[~mask_te]
    other_means = other_df.groupby("group")[["delta_easting", "delta_northing"]].mean()

    ye_te = y_east[mask_te]
    yn_te = y_north[mask_te]
    rmse_before_g.append(np.sqrt(np.mean(ye_te**2 + yn_te**2)))

    # Appliquer la moyenne du groupe (si le groupe existe dans le train)
    pe = np.array([other_means.loc[g, "delta_easting"] if g in other_means.index else 0 
                   for g in df.loc[mask_te, "group"]])
    pn = np.array([other_means.loc[g, "delta_northing"] if g in other_means.index else 0 
                   for g in df.loc[mask_te, "group"]])

    err = np.sqrt((ye_te - pe)**2 + (yn_te - pn)**2)
    rmse_after_g.append(np.sqrt(np.mean(err**2)))

print(f"  RMSE avant : {np.mean(rmse_before_g):.2f} ± {np.std(rmse_before_g):.2f}")
print(f"  RMSE après : {np.mean(rmse_after_g):.2f} ± {np.std(rmse_after_g):.2f}")
print(f"  Réduction  : {(1 - np.mean(rmse_after_g)/np.mean(rmse_before_g))*100:.1f}%")

print("\n" + "=" * 60)
print("SYNTHÈSE")
print("=" * 60)
print("""
Si aucune des 4 pistes ne montre une réduction > 10% en spatial CV :
  → Le verdict est définitif : les rasters 10m ne suffisent pas,
    quel que soit le modèle (RF, XGB, SVM, engineering, résidus, agrégation).
  → La contribution du papier reste l'avertissement méthodologique.

Si une piste marche significativement :
  → On l'intègre comme modèle principal et on ré-évalue le geofencing.
""")
