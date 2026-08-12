#station_variance_explained.py
"""
station_variance_explained.py

Calcule la part de variance expliquee par la station (eta-squared,
ANOVA a un facteur) pour plusieurs definitions candidates de la cible,
afin d'identifier laquelle correspond au "48%" cite dans le manuscrit.

Usage : place ce script dans le meme dossier que ml_ready_stat.txt et lance :
    python station_variance_explained.py
"""

import pandas as pd
import numpy as np

FILE = "ml_ready_stat.txt"

df = pd.read_csv(FILE, sep="\t", low_memory=False)

def eta_squared_one_way(df, target_col, group_col="station"):
    """
    Eta-squared d'une ANOVA a un facteur : proportion de la variance totale
    expliquee par l'appartenance au groupe (ici, la station).

    eta^2 = SS_between / SS_total
          = 1 - SS_within / SS_total

    SS_within  = somme, sur toutes les stations, de la variance intra-station
                 (residuelle une fois la moyenne de la station retiree)
    SS_between = variance expliquee par les differences entre moyennes de station
    """
    sub = df[[target_col, group_col]].dropna()
    grand_mean = sub[target_col].mean()
    ss_total = ((sub[target_col] - grand_mean) ** 2).sum()

    ss_within = 0.0
    for g, gdf in sub.groupby(group_col):
        group_mean = gdf[target_col].mean()
        ss_within += ((gdf[target_col] - group_mean) ** 2).sum()

    ss_between = ss_total - ss_within
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "target": target_col,
        "N": len(sub),
        "n_groups": sub[group_col].nunique(),
        "SS_total": ss_total,
        "SS_between (station)": ss_between,
        "SS_within (residual)": ss_within,
        "eta^2 (% variance expliquee par station)": round(eta2 * 100, 1),
    }

# ---------------------------------------------------------------------
# Candidats testes : chaque colonne represente une definition possible
# de "la variance que le GLMM/le modele est cense expliquer"
# ---------------------------------------------------------------------
candidates = ["precision", "accuracy"]

# Ajoute la magnitude du vecteur de biais si les colonnes existent
if "delta_easting" in df.columns and "delta_northing" in df.columns:
    df["bias_magnitude"] = np.sqrt(df["delta_easting"] ** 2 + df["delta_northing"] ** 2)
    candidates.append("bias_magnitude")

# Ajoute les composantes signees (utile si le 48% portait sur une seule composante)
for c in ["delta_easting", "delta_northing"]:
    if c in df.columns:
        candidates.append(c)

print("=" * 70)
print("PART DE VARIANCE EXPLIQUEE PAR LA STATION (ANOVA 1 facteur, eta^2)")
print("=" * 70)

results = []
for target in candidates:
    if target not in df.columns:
        print(f"  [ignore] colonne absente : {target}")
        continue
    res = eta_squared_one_way(df, target, group_col="station")
    results.append(res)
    print(f"\n--- {target} ---")
    for k, v in res.items():
        if k != "target":
            print(f"  {k:45s} : {v}")

results_df = pd.DataFrame(results)
results_df.to_csv("station_variance_explained.csv", index=False)

print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(results_df[["target", "eta^2 (% variance expliquee par station)"]].to_string(index=False))
print("\nFichier exporte : station_variance_explained.csv")
print("\nCherchez la ligne la plus proche de 48% pour identifier quelle")
print("variable correspond au chiffre cite dans le manuscrit.")