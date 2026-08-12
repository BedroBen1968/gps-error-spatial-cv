#analyse_spatial_rf_loso_corrige.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse spatiale complémentaire — RF multi-output avec LOSO CV
Protocole : fusion merge_asof, feature set étendu, comparaison baseline
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
print("[1/6] Chargement des fichiers...")

try:
    df_stat = pd.read_csv('processed_data_stat.txt', sep=None, engine='python')
    print(f"      processed_data_stat : {len(df_stat)} lignes, {len(df_stat.columns)} colonnes")
except Exception as e:
    sys.exit(f"ERREUR chargement processed_data_stat.txt : {e}")

try:
    df_collar = pd.read_csv('collars_new.csv', low_memory=False)
    print(f"      collars_new         : {len(df_collar)} lignes, {len(df_collar.columns)} colonnes")
except Exception as e:
    sys.exit(f"ERREUR chargement collars_new.csv : {e}")

try:
    df_dgps = pd.read_csv('dGPS_new.csv')
    print(f"      dGPS_new            : {len(df_dgps)} lignes (ground truth de secours)")
except Exception as e:
    print("      dGPS_new            : non trouvé (optionnel)")
    df_dgps = None

# ============================================================================
# 2. IDENTIFICATION DES CLÉS DE FUSION (serial + temps)
# ============================================================================
print("[2/6] Identification des clés de fusion (serial + temps)...")

# --- Serial : mapping insensible à la casse ---
stat_cols_lower = {c.lower(): c for c in df_stat.columns}
collar_cols_lower = {c.lower(): c for c in df_collar.columns}

if 'serial' in stat_cols_lower and 'serial' in collar_cols_lower:
    stat_serial_col = stat_cols_lower['serial']
    collar_serial_col = collar_cols_lower['serial']
    print(f"      Colonne serial détectée : stat='{stat_serial_col}' | collar='{collar_serial_col}'")
else:
    print("      Headers stat :", list(df_stat.columns))
    print("      Headers collar :", list(df_collar.columns))
    sys.exit("ERREUR : colonne 'serial'/'Serial' introuvable.")

# --- Temps : recherche des colonnes temporelles ---
time_candidates = ['time', 'position_time', 'positiontime', 'datetime', 'timestamp', 'date_time']

stat_time_col = None
for cand in time_candidates:
    if cand in stat_cols_lower:
        stat_time_col = stat_cols_lower[cand]
        break
# Si toujours pas trouvé, chercher colonnes contenant 'time'
if stat_time_col is None:
    for c in df_stat.columns:
        if 'time' in c.lower():
            stat_time_col = c
            break

collar_time_col = None
for cand in time_candidates:
    if cand in collar_cols_lower:
        collar_time_col = collar_cols_lower[cand]
        break
# Si toujours pas trouvé, chercher colonnes contenant 'time'
if collar_time_col is None:
    for c in df_collar.columns:
        if 'time' in c.lower():
            collar_time_col = c
            break

if stat_time_col is None or collar_time_col is None:
    print("      Colonnes temps stat :", [c for c in df_stat.columns if 'time' in c.lower()])
    print("      Colonnes temps collar :", [c for c in df_collar.columns if 'time' in c.lower()])
    sys.exit("ERREUR : colonne temporelle introuvable.")

print(f"      Colonne temps détectée : stat='{stat_time_col}' | collar='{collar_time_col}'")

# ============================================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================================
print("[3/6] Préparation des données...")

# --- Conversion serial en string pour la fusion ---
df_stat[stat_serial_col] = df_stat[stat_serial_col].astype(str).str.strip()
df_collar[collar_serial_col] = df_collar[collar_serial_col].astype(str).str.strip()

# --- Conversion des colonnes temps ---
for col, df, label in [(stat_time_col, df_stat, 'stat'), (collar_time_col, df_collar, 'collar')]:
    if df[col].dtype == object:
        # Essayer plusieurs formats
        for fmt in [None, '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt)
                break
            except:
                pass
    else:
        df[col] = pd.to_datetime(df[col])
    if df[col].isna().all():
        sys.exit(f"ERREUR : échec conversion temps pour {label}.'{col}'")

# --- Conversion numérique forcée des colonnes collars_new ---
print("      Conversion numérique des colonnes collars_new...")
num_cols_collar = [
    'Battery Voltage', 'Solar Charge', 'Humidity', 'Pressure', 'Temperature',
    'Min RSSI', 'Max RSSI', 'Latitude', 'Longitude', 'Horizontal Accuracy',
    'HDOP', '#Satellites', 'Warning Duration', 'Activity', 'PastureID'
]
for c in num_cols_collar:
    if c in df_collar.columns:
        df_collar[c] = pd.to_numeric(df_collar[c].astype(str).str.replace(',', '.'), errors='coerce')

# --- Colonnes cibles dans processed_data_stat ---
target_east = None
target_north = None
for c in df_stat.columns:
    lc = c.lower().replace(' ', '').replace('_', '')
    if 'delta' in lc and ('east' in lc or 'easting' in lc):
        target_east = c
    if 'delta' in lc and ('north' in lc or 'northing' in lc):
        target_north = c

if target_east is None or target_north is None:
    print("      Colonnes disponibles :", list(df_stat.columns))
    sys.exit("ERREUR : colonnes cibles delta_easting / delta_northing introuvables.")

print(f"      Targets : {target_east}, {target_north}")

# ============================================================================
# 4. FUSION MERGE_ASOF
# ============================================================================
print("[4/6] Fusion merge_asof (tolérance 5 s)...")

# Trier par serial + temps
df_stat = df_stat.sort_values([stat_serial_col, stat_time_col]).reset_index(drop=True)
df_collar = df_collar.sort_values([collar_serial_col, collar_time_col]).reset_index(drop=True)

# Liste des séries uniques
serials_stat = set(df_stat[stat_serial_col].unique())
serials_collar = set(df_collar[collar_serial_col].unique())
common_serials = serials_stat & serials_collar
print(f"      Séries stat : {len(serials_stat)} | Séries collar : {len(serials_collar)} | Communes : {len(common_serials)}")

if len(common_serials) == 0:
    print("      Exemples stat :", list(serials_stat)[:5])
    print("      Exemples collar :", list(serials_collar)[:5])
    sys.exit("ERREUR : aucune valeur 'serial' commune.")

# Fusion par groupe (serial)
merged_list = []
for serial in sorted(common_serials):
    s_stat = df_stat[df_stat[stat_serial_col] == serial].copy()
    s_collar = df_collar[df_collar[collar_serial_col] == serial].copy()

    if len(s_collar) == 0:
        continue

    m = pd.merge_asof(
        s_stat,
        s_collar,
        left_on=stat_time_col,
        right_on=collar_time_col,
        direction='nearest',
        tolerance=pd.Timedelta('5s')
    )
    merged_list.append(m)

if len(merged_list) == 0:
    sys.exit("ERREUR : aucune ligne fusionnée (vérifiez la tolérance temporelle).")

df_merged = pd.concat(merged_list, ignore_index=True)
print(f"      Lignes fusionnées : {len(df_merged)} / {len(df_stat)} (stat)")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================
print("[5/6] Feature engineering...")

# --- Anciennes features (toutes les colonnes numériques de stat sauf targets, serial, temps) ---
exclude_old = [stat_serial_col, stat_time_col, target_east, target_north, 'index']
if 'serial' in df_merged.columns and stat_serial_col != 'serial':
    exclude_old.append('serial')
if 'time' in df_merged.columns and stat_time_col != 'time':
    exclude_old.append('time')

old_features = []
for c in df_stat.columns:
    if c not in exclude_old and c in df_merged.columns:
        if pd.api.types.is_numeric_dtype(df_merged[c]):
            old_features.append(c)

# --- Nouvelles features depuis collars_new ---
new_feature_candidates = [
    'Solar Charge', 'Min RSSI', 'Max RSSI', 'Battery Voltage',
    'Humidity', 'Pressure', 'Temperature'
]
new_features = [c for c in new_feature_candidates if c in df_merged.columns]

# --- Features temporelles ---
if stat_time_col in df_merged.columns:
    df_merged['hour'] = df_merged[stat_time_col].dt.hour
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['hour'] / 24)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['hour'] / 24)
    time_features = ['hour_sin', 'hour_cos']
else:
    time_features = []

# --- Bearing (si lat/lon disponibles) ---
if 'Latitude' in df_merged.columns and 'Longitude' in df_merged.columns:
    # Calcul du bearing entre points consécutifs par serial
    df_merged = df_merged.sort_values([stat_serial_col, stat_time_col]).reset_index(drop=True)
    df_merged['bearing'] = np.nan
    for serial in df_merged[stat_serial_col].unique():
        idx = df_merged[stat_serial_col] == serial
        sub = df_merged.loc[idx, ['Latitude', 'Longitude']].values
        if len(sub) > 1:
            bearings = np.zeros(len(sub))
            bearings[0] = np.nan
            for i in range(1, len(sub)):
                lat1, lon1 = np.radians(sub[i-1])
                lat2, lon2 = np.radians(sub[i])
                dlon = lon2 - lon1
                x = np.sin(dlon) * np.cos(lat2)
                y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
                brng = np.degrees(np.arctan2(x, y))
                bearings[i] = (brng + 360) % 360
            df_merged.loc[idx, 'bearing'] = bearings
    if 'bearing' in df_merged.columns:
        df_merged['bearing_sin'] = np.sin(np.radians(df_merged['bearing']))
        df_merged['bearing_cos'] = np.cos(np.radians(df_merged['bearing']))
        bearing_features = ['bearing_sin', 'bearing_cos']
    else:
        bearing_features = []
else:
    bearing_features = []

all_features = old_features + new_features + time_features + bearing_features
print(f"      Anciennes features : {len(old_features)}")
print(f"      Nouvelles features : {len(new_features)} ({new_features})")
print(f"      Features temporelles : {len(time_features)}")
print(f"      Features bearing : {len(bearing_features)}")
print(f"      TOTAL features : {len(all_features)}")

# --- Imputation ---
df_model = df_merged[all_features + [target_east, target_north, stat_serial_col]].copy()
df_model = df_model.replace([np.inf, -np.inf], np.nan)

# Imputer médiane pour numériques
for c in all_features:
    if c in df_model.columns:
        med = df_model[c].median()
        df_model[c] = df_model[c].fillna(med)

# Supprimer lignes avec targets manquants
df_model = df_model.dropna(subset=[target_east, target_north])
print(f"      Lignes utilisables après nettoyage : {len(df_model)}")

if len(df_model) < 100:
    sys.exit("ERREUR : trop peu de lignes après fusion/nettoyage.")

X = df_model[all_features].values
y_east = df_model[target_east].values
y_north = df_model[target_north].values
y = np.column_stack([y_east, y_north])

groups = df_model[stat_serial_col].values

# ============================================================================
# 6. MODÉLISATION — RF MULTI-OUTPUT + LOSO CV
# ============================================================================
print("[6/6] Modélisation RF multi-output + LOSO CV spatial...")

logo = LeaveOneGroupOut()

# --- Baseline : ancien feature set seul ---
if len(old_features) > 0:
    X_old = df_model[old_features].values
    preds_old_east = np.zeros(len(y))
    preds_old_north = np.zeros(len(y))

    for train_idx, test_idx in logo.split(X_old, y, groups):
        rf_old = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        )
        rf_old.fit(X_old[train_idx], y[train_idx])
        p = rf_old.predict(X_old[test_idx])
        preds_old_east[test_idx] = p[:, 0]
        preds_old_north[test_idx] = p[:, 1]

    rmse_old_east = np.sqrt(mean_squared_error(y_east, preds_old_east))
    rmse_old_north = np.sqrt(mean_squared_error(y_north, preds_old_north))
    rmse_old_spatial = np.sqrt(mean_squared_error(
        np.zeros(len(y)),
        np.sqrt((y_east - preds_old_east)**2 + (y_north - preds_old_north)**2)
    ))
else:
    rmse_old_spatial = np.nan

# --- Nouveau feature set ---
preds_new_east = np.zeros(len(y))
preds_new_north = np.zeros(len(y))

for train_idx, test_idx in logo.split(X, y, groups):
    rf_new = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    )
    rf_new.fit(X[train_idx], y[train_idx])
    p = rf_new.predict(X[test_idx])
    preds_new_east[test_idx] = p[:, 0]
    preds_new_north[test_idx] = p[:, 1]

rmse_new_east = np.sqrt(mean_squared_error(y_east, preds_new_east))
rmse_new_north = np.sqrt(mean_squared_error(y_north, preds_new_north))
rmse_new_spatial = np.sqrt(mean_squared_error(
    np.zeros(len(y)),
    np.sqrt((y_east - preds_new_east)**2 + (y_north - preds_new_north)**2)
))

r2_east = r2_score(y_east, preds_new_east)
r2_north = r2_score(y_north, preds_new_north)

# ============================================================================
# 7. RÉSULTATS
# ============================================================================
print("\n" + "="*60)
print("RÉSULTATS")
print("="*60)
if not np.isnan(rmse_old_spatial):
    print(f"RMSE baseline spatial (ancien feature set) : {rmse_old_spatial:.2f} m")
else:
    print("RMSE baseline spatial : N/A (pas d'anciennes features détectées)")
print(f"RMSE RF spatial (nouveau feature set)      : {rmse_new_spatial:.2f} m")
print(f"R² East                                    : {r2_east:.4f}")
print(f"R² North                                   : {r2_north:.4f}")
print("="*60)

# --- Verdict K.1 vs K.2 ---
print("\nVERDICT :", end=" ")
if rmse_new_spatial < rmse_old_spatial and r2_east > 0 and r2_north > 0:
    print("K.2 — Le nouveau feature set améliore significativement la prédiction.")
else:
    print("K.1 — Le nouveau feature set n'apporte pas d'amélioration déterminante ; baseline conservée.")
