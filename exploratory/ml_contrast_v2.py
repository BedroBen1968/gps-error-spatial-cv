#ml_contrast_v2.py
"""
ml_contrast_v2.py
Option B corrigée : R² global poolé + exclusion horizontal_accuracy/hdop.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
df = pd.read_csv(BASE / "ml_ready_stat.txt", sep="\t", low_memory=False)

# ─── FONCTION DE CV PRECISION ─────────────────────────────────────
def cv_precision(X, y, groups, group_name, feature_names):
    unique = np.unique(groups)
    
    # Stockage pour R² global poolé
    all_y_true, all_y_pred = [], []
    
    rmse_base, rmse_after = [], []
    r2_per_fold = []
    
    for g in unique:
        mask_test = groups == g
        mask_train = ~mask_test
        X_tr, X_te = X[mask_train], X[mask_test]
        y_tr, y_te = y[mask_train], y[mask_test]
        
        # Baseline : moyenne du train
        base = np.mean(y_tr)
        rmse_base.append(np.sqrt(np.mean((y_te - base)**2)))
        
        # RF
        rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        
        rmse_after.append(np.sqrt(mean_squared_error(y_te, pred)))
        r2_per_fold.append(r2_score(y_te, pred))
        
        # Stockage pour R² global
        all_y_true.extend(y_te)
        all_y_pred.extend(pred)
    
    # R² global (tous les folds poolés)
    r2_global = r2_score(all_y_true, all_y_pred)
    
    print(f"\n--- {group_name} ({len(unique)} folds) ---")
    print(f"  Features : {feature_names}")
    print(f"  RMSE baseline (train mean) : {np.mean(rmse_base):.2f} ± {np.std(rmse_base):.2f}")
    print(f"  RMSE après RF              : {np.mean(rmse_after):.2f} ± {np.std(rmse_after):.2f}")
    print(f"  Réduction vs baseline      : {(1 - np.mean(rmse_after)/np.mean(rmse_base))*100:.1f}%")
    print(f"  R² par fold (moyenne)      : {np.mean(r2_per_fold):.3f} ± {np.std(r2_per_fold):.3f}")
    print(f"  R² GLOBAL (poolé)          : {r2_global:.3f}")
    
    return rf, r2_global

sites = df["station"].values
y_prec = df["precision"].values

# ─── TEST 1 : TOUTES LES FEATURES (avec horizontal_accuracy) ──────
print("=" * 60)
print("TEST 1 : TOUTES LES FEATURES")
print("=" * 60)

features_all = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                "hdop", "number_satellites", "horizontal_accuracy", "aspect_original"]
X_all = df[features_all].fillna(df[features_all].median())
rf_all, r2_all = cv_precision(X_all.values, y_prec, sites, "leave-one-site-out", features_all)

# Feature importance
imp_all = pd.DataFrame({"feature": features_all, "importance": rf_all.feature_importances_})
imp_all = imp_all.sort_values("importance", ascending=False)
print("\nImportance (toutes features) :")
print(imp_all.to_string(index=False))

# ─── TEST 2 : SANS horizontal_accuracy ────────────────────────────
print("\n" + "=" * 60)
print("TEST 2 : SANS horizontal_accuracy")
print("=" * 60)

features_no_ha = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                  "hdop", "number_satellites", "aspect_original"]
X_no_ha = df[features_no_ha].fillna(df[features_no_ha].median())
rf_no_ha, r2_no_ha = cv_precision(X_no_ha.values, y_prec, sites, "leave-one-site-out", features_no_ha)

imp_no_ha = pd.DataFrame({"feature": features_no_ha, "importance": rf_no_ha.feature_importances_})
imp_no_ha = imp_no_ha.sort_values("importance", ascending=False)
print("\nImportance (sans horizontal_accuracy) :")
print(imp_no_ha.to_string(index=False))

# ─── TEST 3 : SANS horizontal_accuracy ET SANS hdop ───────────────
print("\n" + "=" * 60)
print("TEST 3 : SANS horizontal_accuracy NI hdop")
print("=" * 60)

features_env_only = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                     "number_satellites", "aspect_original"]
X_env = df[features_env_only].fillna(df[features_env_only].median())
rf_env, r2_env = cv_precision(X_env.values, y_prec, sites, "leave-one-site-out", features_env_only)

imp_env = pd.DataFrame({"feature": features_env_only, "importance": rf_env.feature_importances_})
imp_env = imp_env.sort_values("importance", ascending=False)
print("\nImportance (covariables environnementales seules) :")
print(imp_env.to_string(index=False))

# ─── RÉSUMÉ ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ OPTION B (precision)")
print("=" * 60)
print(f"  Toutes features (incl. horizontal_accuracy) : R² global = {r2_all:.3f}")
print(f"  Sans horizontal_accuracy                   : R² global = {r2_no_ha:.3f}")
print(f"  Covariables environnementales seules       : R² global = {r2_env:.3f}")
print(f"\n  Réduction RMSE (env. seules) : à calculer ci-dessus")

print("\n" + "=" * 60)
print("INTERPRÉTATION")
print("=" * 60)
print("""
Si R² global reste négatif même sans horizontal_accuracy :
  → Même le signal indirect s'évapore. Le message central est renforcé :
    les covariables environnementales à 10m ne prédisent pas la précision GPS.

Si R² global devient positif sans horizontal_accuracy :
  → Il y a un signal environnemental réel, modeste mais propre.
    C'est la vraie contribution Option B : l'environnement prédit la fiabilité,
    indépendamment de ce que le device sait déjà.
""")