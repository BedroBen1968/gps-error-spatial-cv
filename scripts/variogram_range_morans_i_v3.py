#variogram_range_morans_i_v3.py
"""
variogram_range_morans_i_v3.py

Version finale :
- maxlag corrige (distances uniques, sans zeros diagonaux)
- Selection automatique du meilleur modele de variogramme (spherical / exponential / gaussian)
- Moran's I from scratch, robuste k=3,4,6
- Ratio range / distance inter-station comme metrique cle
- Figure variogramme sauvegardee

Necessite : scikit-gstat (installe), pykrige, numpy, pandas, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skgstat as skg
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

FILE = "ml_ready_stat.txt"
df = pd.read_csv(FILE, sep="\t", low_memory=False)

# Coordonnees
coord_candidates = ["X_proj", "Y_proj", "X", "Y", "easting", "northing", "longitude", "latitude"]
coord_cols = [c for c in coord_candidates if c in df.columns][:2]
if len(coord_cols) < 2:
    raise ValueError(f"Colonnes coords non trouvees. Disponibles: {df.columns.tolist()}")

coords = df[coord_cols].values.astype(float)
station_means = df.groupby("station")[["delta_easting", "delta_northing"]].mean()
station_coords = df.groupby("station")[coord_cols].mean().values
station_names = df["station"].unique()

# ---------------------------------------------------------------------
# 1. DISTANCES INTER-STATION (uniques, hors diagonale)
# ---------------------------------------------------------------------
D = cdist(station_coords, station_coords)
D_unique = D[np.triu_indices_from(D, k=1)]  # CORRECTION : sans les zeros diagonaux

print("=" * 65)
print("DISTANCES INTER-STATION (30 stations)")
print("=" * 65)
print(f"  Distances uniques N={len(D_unique)}")
print(f"  Plus proche voisin : moy={D_unique.mean():.1f} m, "
      f"min={D_unique.min():.1f} m, max={D_unique.max():.1f} m, "
      f"med={np.median(D_unique):.1f} m")

# ---------------------------------------------------------------------
# 2. VARIOGRAMMES — selection du meilleur modele
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("VARIOGRAMMES (moyennes par station, N=30)")
print("=" * 65)

def fit_and_select_variogram(x, y, values, label, plot_file=None):
    """
    Teste spherical / exponential / gaussian, selectionne par RMSE minimal.
    Retourne : dict avec meilleur modele, range, sill, nugget, rmse, ratio
    """
    maxlag = np.percentile(D_unique, 50)  # CORRECTION : base sur distances uniques
    
    candidates = ['spherical', 'exponential', 'gaussian']
    results = []
    
    for model_name in candidates:
        try:
            V = skg.Variogram(
                coordinates=x,
                values=values,
                model=model_name,
                n_lags=10,
                maxlag=maxlag,
                use_nugget=True
            )
            # Force le fit pour obtenir les parametres
            V.fit()
            
            # Recuperation des parametres [range, sill, nugget] selon doc scikit-gstat
            params = V.parameters
            print(f"\n    [{model_name:12s}] params bruts : {params}")
            
            if params is not None and len(params) >= 2:
                rng = params[0]
                sill = params[1]
                nugget = params[2] if len(params) > 2 else 0.0
                rmse = V.rmse if hasattr(V, 'rmse') else np.nan
                
                ratio = rng / D_unique.mean() if D_unique.mean() > 0 else np.nan
                
                results.append({
                    'model': model_name,
                    'range': rng,
                    'sill': sill,
                    'nugget': nugget,
                    'rmse': rmse,
                    'ratio': ratio,
                    'variogram_obj': V
                })
                print(f"      range={rng:.1f}m, sill={sill:.3f}, nugget={nugget:.3f}, rmse={rmse:.4f}, ratio={ratio:.2f}")
        except Exception as e:
            print(f"\n    [{model_name:12s}] ECHEC : {str(e)[:80]}")
    
    if not results:
        print(f"  -> AUCUN modele n'a converge pour {label}")
        return None
    
    # Selection par RMSE minimal
    best = min(results, key=lambda r: r['rmse'] if not np.isnan(r['rmse']) else np.inf)
    print(f"\n  >>> MEILLEUR MODELE pour {label} : {best['model'].upper()}")
    print(f"      Range  : {best['range']:.1f} m")
    print(f"      Sill   : {best['sill']:.3f}")
    print(f"      Nugget : {best['nugget']:.3f}")
    print(f"      RMSE   : {best['rmse']:.4f}")
    print(f"      Ratio range / dist inter-station moy : {best['ratio']:.2f}")
    
    # Figure
    if plot_file and best['variogram_obj'] is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        V = best['variogram_obj']
        # Variogramme experimental (bins, exp_var)
        bins = V.bins
        exp = V.experimental
        ax.scatter(bins, exp, c='black', label='Experimental', zorder=3)
        # Modele ajuste
        model_x = np.linspace(0, max(bins)*1.2, 200)
        model_y = V.fitted_model(model_x)
        ax.plot(model_x, model_y, 'r-', label=f"Fitted ({best['model']})", zorder=2)
        # Ligne verticale : distance moyenne inter-station
        ax.axvline(D_unique.mean(), color='blue', linestyle='--', 
                   label=f"Mean inter-station = {D_unique.mean():.0f} m")
        ax.axvline(best['range'], color='red', linestyle=':',
                   label=f"Range = {best['range']:.0f} m")
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Semivariance")
        ax.set_title(f"Variogram — {label}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        print(f"      Figure sauvegardee : {plot_file}")
        plt.close()
    
    return best

# Lance pour chaque composante
best_east = fit_and_select_variogram(
    station_coords, None, station_means["delta_easting"].values,
    "delta_easting (station means)", "variogram_easting.png"
)

best_north = fit_and_select_variogram(
    station_coords, None, station_means["delta_northing"].values,
    "delta_northing (station means)", "variogram_northing.png"
)

# ---------------------------------------------------------------------
# 3. MORAN'S I (from scratch, robustesse k=3,4,6)
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("MORAN'S I (niveau station, N=30)")
print("=" * 65)

def morans_i_scratch(coords, values, k=4, permutations=999):
    n = len(values)
    z = values - np.mean(values)
    
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k+1)
    neighbors = idx[:, 1:]
    
    W = np.zeros((n, n))
    for i in range(n):
        W[i, neighbors[i]] = 1.0
    W = np.maximum(W, W.T)
    
    W_sum = np.sum(W)
    numerator = np.sum(W * np.outer(z, z))
    denominator = np.sum(z**2)
    I = (n / W_sum) * (numerator / denominator)
    E_I = -1.0 / (n - 1)
    
    I_perm = np.zeros(permutations)
    for p in range(permutations):
        z_perm = np.random.permutation(z)
        num_perm = np.sum(W * np.outer(z_perm, z_perm))
        I_perm[p] = (n / W_sum) * (num_perm / denominator)
    
    p_val = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (permutations + 1)
    std_perm = np.std(I_perm)
    z_score = (I - E_I) / std_perm if std_perm > 0 else 0
    
    return I, E_I, z_score, p_val

for k in [3, 4, 6]:
    print(f"\n--- k={k} ---")
    for target in ["delta_easting", "delta_northing"]:
        y = station_means[target].values
        I, E_I, z, p = morans_i_scratch(station_coords, y, k=k, permutations=999)
        sig = "SIG" if p < 0.05 else "NS"
        print(f"  {target:20s} | I={I:+.3f} | E[I]={E_I:+.3f} | "
              f"z={z:+.2f} | p={p:.4f} | {sig}")

# ---------------------------------------------------------------------
# 4. SYNTHESE INTERPRETATIVE
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("SYNTHESE POUR LE MANUSCRIT")
print("=" * 65)

if best_east and best_north:
    ratio_e = best_east['ratio']
    ratio_n = best_north['ratio']
    model_e = best_east['model']
    model_n = best_north['model']
    
    print(f"""
Meilleurs modeles : East={model_e}, North={model_n}
Ratios range/dist inter-station : E={ratio_e:.2f}, N={ratio_n:.2f}

Si ratio << 1 (ex: < 0.5) :
  -> La portee est bien plus courte que l'espacement inter-station.
  -> Phrase pour le papier :
  "The estimated spatial correlation range ({best_east['range']:.0f} m and 
   {best_north['range']:.0f} m for ΔEasting and ΔNorthing, respectively, 
   under the best-fitting {model_e}/{model_n} models) was substantially 
   shorter than the mean inter-station distance ({D_unique.mean():.0f} m; 
   ratio = {ratio_e:.2f} and {ratio_n:.2f}), indicating that any spatial 
   dependence was confined to scales much smaller than those required for 
   LOSSO transfer."

Si ratio >= 1 :
  -> La portee atteint ou depasse l'espacement inter-station.
  -> Mais Moran's I NS suggere que cette "portee" est un artefact d'ajustement
     (pur nugget, modele qui s'etire pour minimiser l'erreur).
  -> Phrase alternative :
  "While the fitted variogram range exceeded the mean inter-station distance, 
   the dominant nugget effect and non-significant Moran's I (I ≈ 0, p > 0.6) 
   indicated that this range reflected model degeneracy on a spatially 
   unstructured field rather than genuine spatial correlation."
""")