#preprocess_v3.py
#!/usr/bin/env python3
"""
preprocess_v3.py
RF statique -> test mobile. Version corrigée.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
print("=== Chargement ===")

s = pd.read_csv('ml_ready_stat.txt', sep='\t', low_memory=False)
m = pd.read_csv('ml_ready_mob.txt',  sep='\t', low_memory=False)

print(f"Statique brut : {len(s)} lignes")
print(f"Mobile brut   : {len(m)} lignes")

# ============================================================================
# 2. FEATURES & TARGETS
# ============================================================================
feature_candidates = {
    'canopy_cover':        ['canopy_cover', 'Canopy cover (%)'],
    'elevation':           ['elevation', 'Elevation'],
    'skyview':             ['skyview', 'sky_view', 'Sky-view factor'],
    'slope':               ['slope', 'Slope'],
    'tri':                 ['tri', 'TRI', 'Terrain Ruggedness Index'],
    'hdop':                ['hdop', 'HDOP'],
    'number_satellites':   ['number_satellites', '#Satellites', 'n_satellites', 'satellites'],
    'horizontal_accuracy': ['horizontal_accuracy', 'Horizontal Accuracy'],
    'aspect':              ['aspect_original', 'aspect', 'Aspect']
}

def find_col(df, candidates, optional=False):
    for c in candidates:
        if c in df.columns:
            return c
    if optional:
        return None
    raise ValueError(f"Introuvable parmi {candidates}. Disponibles : {list(df.columns)}")

# Résolution indépendante : on garde seulement les features présentes dans LES DEUX fichiers
features_s = []
features_m = []
feature_names = []
for key, cands in feature_candidates.items():
    f_s = find_col(s, cands, optional=True)
    f_m = find_col(m, cands, optional=True)
    if f_s and f_m:
        features_s.append(f_s)
        features_m.append(f_m)
        feature_names.append(key)

target_east  = find_col(s, ['delta_easting', 'Delta Easting'])
target_north = find_col(s, ['delta_northing', 'Delta Northing'])

print(f"\nFeatures communes : {feature_names}")
print(f"  Statique : {features_s}")
print(f"  Mobile   : {features_m}")

# ============================================================================
# 3. NETTOYAGE ROBUSTE
# ============================================================================
def robust_numeric(df, cols):
    df = df[cols].copy()
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# --- Statique ---
s_clean = robust_numeric(s, features_s + [target_east, target_north])
s_clean = s_clean.replace([np.inf, -np.inf], np.nan)

print(f"\n--- Diagnostic NaN statique ---")
print(s_clean.isna().sum())
s_clean = s_clean.dropna()
print(f"Statique utilisable : {len(s_clean)} lignes")

if len(s_clean) == 0:
    raise ValueError("ERREUR : 0 ligne statique après nettoyage.")

# --- Mobile : détection auto des coordonnées ---
coord_map = {
    'X':     ['X', 'easting', 'Easting', 'gps_easting'],
    'Y':     ['Y', 'northing', 'Northing', 'gps_northing'],
    'd_X':   ['d_X', 'dgps_easting', 'd_easting', 'ground_truth_easting'],
    'd_Y':   ['d_Y', 'dgps_northing', 'd_northing', 'ground_truth_northing']
}
mob_coords = {k: find_col(m, v) for k, v in coord_map.items()}

# Nettoyage du mobile
mob_cols = features_m + list(mob_coords.values())
m_clean = robust_numeric(m, mob_cols)
m_clean = m_clean.replace([np.inf, -np.inf], np.nan)

print(f"\n--- Diagnostic NaN mobile ---")
print(m_clean.isna().sum())

# Supprimer les features qui ont des NaN dans le mobile
bad_idx = [i for i, col in enumerate(features_m) if m_clean[col].isna().any()]
if bad_idx:
    bad_names = [feature_names[i] for i in bad_idx]
    print(f"\n>>> ATTENTION : features avec NaN dans le mobile — supprimées : {bad_names}")
    # Filtrer les trois listes en parallèle
    final_features_s = [f for i, f in enumerate(features_s) if i not in bad_idx]
    final_features_m = [f for i, f in enumerate(features_m) if i not in bad_idx]
    final_names      = [f for i, f in enumerate(feature_names) if i not in bad_idx]
else:
    final_features_s = features_s
    final_features_m = features_m
    final_names = feature_names

print(f">>> Features finales synchronisées : {final_names}")

# CORRECTION CLÉ : on ne garde que les colonnes finales + coords avant dropna
final_mob_cols = final_features_m + list(mob_coords.values())
m_clean = m_clean[final_mob_cols].dropna()
print(f"Mobile utilisable : {len(m_clean)} lignes")

if len(m_clean) == 0:
    raise ValueError("ERREUR : 0 ligne mobile après nettoyage.")

X_stat = s_clean[final_features_s].values
X_mob  = m_clean[final_features_m].values
y_east_stat  = s_clean[target_east].values
y_north_stat = s_clean[target_north].values

# ============================================================================
# 4. ENTRAINEMENT RF
# ============================================================================
print(f"\n=== Entrainement RF (300/8/5) sur {len(final_names)} features ===")
rf_east = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_north= RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42)

rf_east.fit(X_stat, y_east_stat)
rf_north.fit(X_stat, y_north_stat)
print("OK.")

# ============================================================================
# 5. PREDICTION & METRIQUES
# ============================================================================
print("\n=== Prediction mobile ===")
pred_e = rf_east.predict(X_mob)
pred_n = rf_north.predict(X_mob)

# Coordonnées corrigées (delta = verite - mesure  =>  corr = mesure + delta_pred)
Xc = m_clean[mob_coords['X']].values + pred_e
Yc = m_clean[mob_coords['Y']].values + pred_n

# RMSE brut
e_brut = m_clean[mob_coords['X']].values - m_clean[mob_coords['d_X']].values
n_brut = m_clean[mob_coords['Y']].values - m_clean[mob_coords['d_Y']].values
rmse_brut = np.sqrt(np.mean(e_brut**2 + n_brut**2))

# RMSE corrige
e_corr = Xc - m_clean[mob_coords['d_X']].values
n_corr = Yc - m_clean[mob_coords['d_Y']].values
rmse_corr = np.sqrt(np.mean(e_corr**2 + n_corr**2))

reduc = (rmse_brut - rmse_corr) / rmse_brut * 100

print("\n" + "="*55)
print(f"RMSE brut mobile    : {rmse_brut:.2f} m")
print(f"RMSE corrige mobile : {rmse_corr:.2f} m")
print(f"Reduction           : {reduc:+.1f} %")
print(f"Features utilisées  : {final_names}")
print("="*55)

if rmse_corr < rmse_brut * 0.95:
    print("VERDICT : Le modele AMELIORE legerement le mobile.")
elif rmse_corr > rmse_brut * 1.05:
    print("VERDICT : Le modele DEGRADE le mobile. Generalisation echouee.")
else:
    print("VERDICT : Effet neutre. Pas de transfert statique->mobile.")