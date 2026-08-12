#ml_direction.py
"""
ml_direction.py
DERNIER TEST : prédire la DIRECTION du biais (angle circulaire) plutôt que la norme.
Si la direction est structurée par l'aspect/canopée, c'est une contribution exploitable
pour le geofencing (on sait où décaler la clôture, même sans savoir de combien).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
df = pd.read_csv(BASE / "ml_ready_stat.txt", sep="	", low_memory=False)

# ═══════════════════════════════════════════════════════════════════
#  1. CALCUL DE L'ANGLE DE DÉVIATION (comme Versluijs)
# ═══════════════════════════════════════════════════════════════════

def bearing(lat1, lon1, lat2, lon2):
    """Bearing de (lat1,lon1) vers (lat2,lon2) en degrés."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    angle = np.degrees(np.arctan2(x, y))
    return (angle + 360) % 360

# Bearing du vrai point (dGPS) vers le point GPS mesuré = direction du biais
df["bias_angle"] = bearing(df["d_latitude"], df["d_longitude"], 
                            df["latitude"], df["longitude"])

# Convertir en sinus/cosinus pour éviter la discontinuité 0°/360°
df["bias_angle_sin"] = np.sin(np.radians(df["bias_angle"]))
df["bias_angle_cos"] = np.cos(np.radians(df["bias_angle"]))

print("=== DIRECTION DU BIAIS PAR ASPECT ===")
for asp in df["aspect"].unique():
    sub = df[df["aspect"] == asp]["bias_angle"]
    # Moyenne circulaire
    sin_mean = np.mean(np.sin(np.radians(sub)))
    cos_mean = np.mean(np.cos(np.radians(sub)))
    mean_angle = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    print(f"  {asp:6s} : moyenne circulaire = {mean_angle:.1f}°  (N={len(sub)})")

# ═══════════════════════════════════════════════════════════════════
#  2. FEATURES
# ═══════════════════════════════════════════════════════════════════

feature_cols = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                "hdop", "number_satellites", "horizontal_accuracy", "aspect_original"]
X = df[feature_cols].fillna(df[feature_cols].median())

# Target : (sin, cos) du biais — permet de prédire la direction
y_sin = df["bias_angle_sin"].values
y_cos = df["bias_angle_cos"].values
sites = df["station"].values
unique_sites = np.unique(sites)

# ═══════════════════════════════════════════════════════════════════
#  3. SPATIAL CV — LEAVE-ONE-SITE-OUT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SPATIAL CV — Prédiction de la DIRECTION du biais")
print("=" * 60)

angular_errors = []  # Erreur angulaire en degrés
r2_sin, r2_cos = [], []

for s in unique_sites:
    mask_tr = sites != s
    mask_te = sites == s
    X_tr, X_te = X[mask_tr], X[mask_te]

    # RF sur sin et cos
    rf_sin = RandomForestRegressor(n_estimators=300, max_depth=8, 
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_cos = RandomForestRegressor(n_estimators=300, max_depth=8,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_sin.fit(X_tr, y_sin[mask_tr])
    rf_cos.fit(X_tr, y_cos[mask_tr])

    pred_sin = rf_sin.predict(X_te)
    pred_cos = rf_cos.predict(X_te)

    # Reconstituer l'angle prédit
    pred_angle = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
    true_angle = df.loc[mask_te, "bias_angle"].values

    # Erreur angulaire (minimum entre |a-b| et 360-|a-b|)
    err = np.abs(pred_angle - true_angle)
    err = np.minimum(err, 360 - err)
    angular_errors.extend(err)

    if len(y_sin[mask_te]) > 1:
        r2_sin.append(rf_sin.score(X_te, y_sin[mask_te]))
        r2_cos.append(rf_cos.score(X_te, y_cos[mask_te]))

print(f"\n--- Résultats (30 folds) ---")
print(f"  Erreur angulaire moyenne : {np.mean(angular_errors):.1f}°")
print(f"  Erreur angulaire médiane : {np.median(angular_errors):.1f}°")
print(f"  % d'erreurs < 45°        : {100*np.mean(np.array(angular_errors) < 45):.1f}%")
print(f"  % d'erreurs < 90°        : {100*np.mean(np.array(angular_errors) < 90):.1f}%")
print(f"  R² sin                   : {np.mean(r2_sin):.3f}")
print(f"  R² cos                   : {np.mean(r2_cos):.3f}")

# Baseline : moyenne par aspect
print(f"\n--- Baseline (moyenne par aspect) ---")
asp_means = df.groupby("aspect")[["bias_angle_sin", "bias_angle_cos"]].mean()
base_errors = []
for s in unique_sites:
    mask_te = sites == s
    asp = df.loc[mask_te, "aspect"].iloc[0]
    base_sin = asp_means.loc[asp, "bias_angle_sin"]
    base_cos = asp_means.loc[asp, "bias_angle_cos"]
    base_angle = np.degrees(np.arctan2(base_sin, base_cos)) % 360
    true_angle = df.loc[mask_te, "bias_angle"].values
    err = np.abs(base_angle - true_angle)
    err = np.minimum(err, 360 - err)
    base_errors.extend(err)

print(f"  Erreur angulaire moyenne : {np.mean(base_errors):.1f}°")
print(f"  Erreur angulaire médiane : {np.median(base_errors):.1f}°")
print(f"  % d'erreurs < 45°        : {100*np.mean(np.array(base_errors) < 45):.1f}%")

# ═══════════════════════════════════════════════════════════════════
#  4. INTERPRÉTATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("INTERPRÉTATION")
print("=" * 60)
print("""
Si erreur angulaire moyenne < 45° et R² > 0 :
  → Le modèle capture la direction préférentielle du biais.
  → Contribution : on peut décaler la clôture virtuelle dans la direction
    opposée au biais prédit, même sans connaître la distance exacte.
  → C'est une correction "directionnelle", pas "métrique".

Si erreur angulaire moyenne > 90° ou R² < 0 :
  → Même la direction est imprédictible spatialement.
  → Verdict définitif : aucune correction prédictive n'est possible
    avec ces données à cette résolution.
""")
