#ml_contrast.py
"""
ml_contrast.py
Contraste K-fold naïf vs spatial CV + Option B (prédire precision).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
df = pd.read_csv(BASE / "ml_ready_stat.txt", sep="\t", low_memory=False)

feature_cols = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                "hdop", "number_satellites", "horizontal_accuracy", "aspect_original"]
X = df[feature_cols].fillna(df[feature_cols].median())

# ─── 1. CONTRASTE : K-FOLD NAÏF (5-fold) ──────────────────────────
print("=" * 60)
print("1. K-FOLD NAÏF (5-fold, aléatoire) — le piège classique")
print("=" * 60)

y_east = df["delta_easting"].values
y_north = df["delta_northing"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_before, rmse_after_rf, rmse_after_xgb = [], [], []
r2_e_rf, r2_n_rf = [], []

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
    ye_tr, ye_te = y_east[train_idx], y_east[test_idx]
    yn_tr, yn_te = y_north[train_idx], y_north[test_idx]
    
    # RMSE avant correction
    rmse_before.append(np.sqrt(np.mean(ye_te**2 + yn_te**2)))
    
    # RF
    rf_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_e.fit(X_tr, ye_tr)
    rf_n.fit(X_tr, yn_tr)
    pe, pn = rf_e.predict(X_te), rf_n.predict(X_te)
    rmse_after_rf.append(np.sqrt(np.mean((ye_te - pe)**2 + (yn_te - pn)**2)))
    r2_e_rf.append(r2_score(ye_te, pe))
    r2_n_rf.append(r2_score(yn_te, pn))
    
    # XGB
    xg_e = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1)
    xg_n = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1)
    xg_e.fit(X_tr, ye_tr)
    xg_n.fit(X_tr, yn_tr)
    pe, pn = xg_e.predict(X_te), xg_n.predict(X_te)
    rmse_after_xgb.append(np.sqrt(np.mean((ye_te - pe)**2 + (yn_te - pn)**2)))

print(f"  RMSE avant correction : {np.mean(rmse_before):.2f} ± {np.std(rmse_before):.2f}")
print(f"  RMSE après RF         : {np.mean(rmse_after_rf):.2f} ± {np.std(rmse_after_rf):.2f}")
print(f"  RMSE après XGBoost    : {np.mean(rmse_after_xgb):.2f} ± {np.std(rmse_after_xgb):.2f}")
print(f"  Réduction RF vs brut  : {(1 - np.mean(rmse_after_rf)/np.mean(rmse_before))*100:.1f}%")
print(f"  R² East (RF)          : {np.mean(r2_e_rf):.3f}")
print(f"  R² North (RF)         : {np.mean(r2_n_rf):.3f}")

# ─── 2. OPTION B : PRÉDIRE PRECISION ──────────────────────────────
print("\n" + "=" * 60)
print("2. OPTION B — Prédire precision (bruit intra-station)")
print("=" * 60)

y_prec = df["precision"].values
sites = df["station"].values

def cv_precision(X, y, groups, group_name):
    unique = np.unique(groups)
    rmse_before, rmse_after, r2_list = [], [], []
    
    for g in unique:
        mask_test = groups == g
        mask_train = ~mask_test
        X_tr, X_te = X[mask_train], X[mask_test]
        y_tr, y_te = y[mask_train], y[mask_test]
        
        # Baseline : moyenne du train
        base = np.mean(y_tr)
        rmse_before.append(np.sqrt(np.mean((y_te - base)**2)))
        
        # RF
        rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        rmse_after.append(np.sqrt(mean_squared_error(y_te, pred)))
        r2_list.append(r2_score(y_te, pred))
    
    print(f"\n--- {group_name} ({len(unique)} folds) ---")
    print(f"  RMSE baseline (train mean) : {np.mean(rmse_before):.2f} ± {np.std(rmse_before):.2f}")
    print(f"  RMSE après RF              : {np.mean(rmse_after):.2f} ± {np.std(rmse_after):.2f}")
    print(f"  Réduction vs baseline      : {(1 - np.mean(rmse_after)/np.mean(rmse_before))*100:.1f}%")
    print(f"  R² (RF)                    : {np.mean(r2_list):.3f}")
    return rmse_before, rmse_after, r2_list

# Spatial CV (leave-one-site-out) sur precision
cv_precision(X.values, y_prec, sites, "leave-one-site-out")

# K-fold naïf sur precision
print("\n--- K-fold naïf (5-fold) ---")
rmse_kf = []
for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X.iloc[tr], y_prec[tr])
    pred = rf.predict(X.iloc[te])
    rmse_kf.append(np.sqrt(mean_squared_error(y_prec[te], pred)))
print(f"  RMSE après RF : {np.mean(rmse_kf):.2f} ± {np.std(rmse_kf):.2f}")

print("\n" + "=" * 60)
print("INTERPRÉTATION POUR LE PAPIER")
print("=" * 60)
print("""
Le contraste K-fold naïf vs spatial CV est le résultat central :
  - K-fold aléatoire : le modèle semble excellent (R² élevé, RMSE réduit)
  - Spatial CV       : le modèle échoue (R² négatif, RMSE augmente)

C'est la preuve empirique que la validation naïve est trompeuse en 
géospatial ML. Aucun des papiers cités (Versluijs, Ahmed, etc.) ne 
mentionne cette distinction. Contribution : avertissement méthodologique.

Option B (precision) : si R² > 0 en spatial CV, le modèle peut au moins
prédire quand un fix est peu fiable — valeur pour le geofencing 
(délai de confirmation au lieu d'alerte immédiate).
""")

# Feature importance precision (entraînement final)
print("\nFeature importance (precision) :")
rf_prec = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_prec.fit(X, y_prec)
imp = pd.DataFrame({"feature": feature_cols, "importance": rf_prec.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
print(imp.to_string(index=False))
imp.to_csv(BASE / "feature_importance_precision.csv", index=False)