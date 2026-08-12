#kriging_fixed.py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pykrige.ok import OrdinaryKriging

# ------------------------------------------------------------------
# 1. CHARGE LES DONNEES
# ------------------------------------------------------------------
FILE = "ml_ready_stat.txt"
print("Chargement des donnees...")
df = pd.read_csv(FILE, sep="\t", low_memory=False)

# ------------------------------------------------------------------
# 2. DETECTE AUTOMATIQUEMENT LES COLONNES DE COORDONNEES
#    (evite le plantage si elles s'appellent X_proj, Y_proj, etc.)
# ------------------------------------------------------------------
coord_candidates = ["X_proj", "Y_proj", "easting", "northing", "X", "Y", "longitude", "latitude"]
found_coords = [c for c in coord_candidates if c in df.columns]

if len(found_coords) < 2:
    print("\nERREUR : colonnes de coordonnees non trouvees.")
    print("Colonnes disponibles dans ton fichier :")
    print(df.columns.tolist())
    exit()

coord_cols = found_coords[:2]
print(f"Coordonnees detectees : {coord_cols}")
coords = df[coord_cols].values.astype(float)

# ------------------------------------------------------------------
# 3. PREPARE FEATURES ET TARGETS
# ------------------------------------------------------------------
feature_cols = ["canopy_cover", "elevation", "skyview", "slope", "tri",
                "hdop", "number_satellites", "horizontal_accuracy", "aspect_original"]

# Verifie que tout existe
needed = feature_cols + ["delta_easting", "delta_northing", "station"]
missing = [c for c in needed if c not in df.columns]
if missing:
    print(f"\nERREUR : colonnes manquantes : {missing}")
    print("Colonnes disponibles :", df.columns.tolist())
    exit()

X = df[feature_cols].fillna(df[feature_cols].median()).values
y_east = df["delta_easting"].values
y_north = df["delta_northing"].values
stations = df["station"].values
unique_stations = np.unique(stations)

# ------------------------------------------------------------------
# 4. FONCTION RIDGE (baseline lineaire)
# ------------------------------------------------------------------
def run_ridge():
    preds_e = np.zeros(len(y_east))
    preds_n = np.zeros(len(y_north))
    
    for i, s in enumerate(unique_stations):
        print(f"Ridge : station {s} ({i+1}/{len(unique_stations)})")
        train = stations != s
        test = stations == s
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train])
        X_te = scaler.transform(X[test])
        
        ridge_e = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3).fit(X_tr, y_east[train])
        ridge_n = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3).fit(X_tr, y_north[train])
        
        preds_e[test] = ridge_e.predict(X_te)
        preds_n[test] = ridge_n.predict(X_te)
    
    err = np.sqrt((y_east - preds_e)**2 + (y_north - preds_n)**2)
    rmse = np.sqrt(np.mean(err**2))
    return rmse, r2_score(y_east, preds_e), r2_score(y_north, preds_n)

# ------------------------------------------------------------------
# 5. FONCTION KRIGAGE (autocorrelation spatiale explicite)
# ------------------------------------------------------------------
def run_kriging():
    preds_e = np.zeros(len(y_east))
    preds_n = np.zeros(len(y_north))
    
    for i, s in enumerate(unique_stations):
        print(f"Kriging : station {s} ({i+1}/{len(unique_stations)})  [peut prendre ~1 min par station]")
        train = stations != s
        test = stations == s
        
        x_tr, y_tr = coords[train, 0], coords[train, 1]
        x_te, y_te = coords[test, 0], coords[test, 1]
        
        # Delta Easting
        try:
            OK = OrdinaryKriging(x_tr, y_tr, y_east[train],
                                 variogram_model="spherical", verbose=False, nlags=15)
            z_pred, _ = OK.execute("points", x_te, y_te)
            preds_e[test] = np.nan_to_num(z_pred, nan=np.mean(y_east[train]))
        except Exception as e:
            print(f"  -> Fallback moyenne (Easting) : {str(e)[:60]}")
            preds_e[test] = np.mean(y_east[train])
        
        # Delta Northing
        try:
            OK = OrdinaryKriging(x_tr, y_tr, y_north[train],
                                 variogram_model="spherical", verbose=False, nlags=15)
            z_pred, _ = OK.execute("points", x_te, y_te)
            preds_n[test] = np.nan_to_num(z_pred, nan=np.mean(y_north[train]))
        except Exception as e:
            print(f"  -> Fallback moyenne (Northing) : {str(e)[:60]}")
            preds_n[test] = np.mean(y_north[train])
    
    err = np.sqrt((y_east - preds_e)**2 + (y_north - preds_n)**2)
    rmse = np.sqrt(np.mean(err**2))
    return rmse, r2_score(y_east, preds_e), r2_score(y_north, preds_n)

# ------------------------------------------------------------------
# 6. LANCEMENT
# ------------------------------------------------------------------
print("\n" + "="*60)
print("RIDGE REGRESSION (baseline lineaire)")
print("="*60)
rmse_r, r2e_r, r2n_r = run_ridge()
print(f"RMSE  : {rmse_r:.2f} m")
print(f"R2 East  : {r2e_r:.3f}")
print(f"R2 North : {r2n_r:.3f}")

print("\n" + "="*60)
print("ORDINARY KRIGING (autocorrelation spatiale)")
print("="*60)
rmse_k, r2e_k, r2n_k = run_kriging()
print(f"RMSE  : {rmse_k:.2f} m")
print(f"R2 East  : {r2e_k:.3f}")
print(f"R2 North : {r2n_k:.3f}")

print("\n" + "="*60)
print("COMPARAISON (vs D.3 : RF=6.03m, XGB=5.92m, uncorrected=5.26m)")
print("="*60)
print(f"Ridge   : {rmse_r:.2f} m")
print(f"Kriging : {rmse_k:.2f} m")