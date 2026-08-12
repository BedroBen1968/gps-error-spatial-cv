#preprocess_nofence_py.py
"""
preprocess_auto.py
Trouve automatiquement les fichiers NoFence et génère ml_ready_*.txt
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path

BASE = Path(r"D:\Doctorat-LMD\Doctorants\ABID\Articles\Papier1")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

def find_file(name_pattern):
    """Cherche un fichier par nom (ou partie de nom) dans BASE et sous-dossiers."""
    matches = list(BASE.rglob(f"*{name_pattern}*"))
    # Filtre pour éviter les faux positifs
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(f"Fichier contenant '{name_pattern}' introuvable dans {BASE}")
    if len(matches) > 1:
        print(f"⚠️ Plusieurs fichiers trouvés pour '{name_pattern}' :")
        for m in matches:
            print(f"   {m.relative_to(BASE)}")
        print(f"   -> Utilisation du premier : {matches[0].relative_to(BASE)}")
    return matches[0]

def load_tab(path):
    """Lit tabulation ou virgule, avec encodage Windows."""
    for sep in ["\t", ",", ";"]:
        for enc in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except:
                continue
    raise ValueError(f"Impossible de lire {path}")

# ─── DÉTECTION AUTO ────────────────────────────────────────────────
print("Recherche des fichiers ...")
paths = {
    "stat": find_file("processed_data_stat"),
    "mob": find_file("processed_data_mob"),
    "collars": find_file("collars.csv"),      # attention : il y en a peut-être 2 (mobile + static)
    "dgps": find_file("dGPS.csv"),            # idem
    "kobo": find_file("kobo_forms"),          # idem
    "serials": find_file("serials"),
}

for k, v in paths.items():
    print(f"  {k:<10} : {v.relative_to(BASE)}")

# Si plusieurs collars.csv / dGPS.csv / kobo_forms.csv existent,
# il faut différencier mobile vs static. Le script s'arrête pour te demander.
mobile_keywords = ["mobile", "mob"]
static_keywords = ["static", "stat", "new"]

def classify_mobile_static(file_list):
    """Sépare les fichiers mobile vs static par leur chemin/nom."""
    mob = [f for f in file_list if any(k in str(f).lower() for k in mobile_keywords)]
    stat = [f for f in file_list if any(k in str(f).lower() for k in static_keywords)]
    return mob, stat

all_collars = list(BASE.rglob("*collars*.csv"))
all_dgps = list(BASE.rglob("*dGPS*.csv"))
all_kobo = list(BASE.rglob("*kobo*.csv"))

collars_mob, collars_stat = classify_mobile_static(all_collars)
dgps_mob, dgps_stat = classify_mobile_static(all_dgps)
kobo_mob, kobo_stat = classify_mobile_static(all_kobo)

print(f"\nCollars : mobile={len(collars_mob)}, static={len(collars_stat)}")
print(f"dGPS    : mobile={len(dgps_mob)}, static={len(dgps_stat)}")
print(f"Kobo    : mobile={len(kobo_mob)}, static={len(kobo_stat)}")

# Vérification
assert len(collars_mob) == 1, f"Attendu 1 collars mobile, trouvé {len(collars_mob)}"
assert len(dgps_mob) == 1, f"Attendu 1 dGPS mobile, trouvé {len(dgps_mob)}"
assert len(kobo_mob) == 1, f"Attendu 1 kobo mobile, trouvé {len(kobo_mob)}"

# ─── PARTIE 1 : STATIC ─────────────────────────────────────────────
print("\n" + "="*60)
print("PARTIE 1 : STATIC")
print("="*60)

df_stat = load_tab(paths["stat"])
for c in ["d_X", "d_Y", "X", "Y", "accuracy"]:
    assert c in df_stat.columns, f"Colonne manquante : {c}"

df_stat["delta_easting"] = df_stat["d_X"] - df_stat["X"]
df_stat["delta_northing"] = df_stat["d_Y"] - df_stat["Y"]

if "fence_status" in df_stat.columns:
    print("fence_status :", df_stat["fence_status"].unique())
else:
    print("Pas de fence_status")

rmse = np.sqrt(np.mean(df_stat["accuracy"]**2))
print(f"Static : {len(df_stat)} lignes | RMSE : {rmse:.2f} m")
df_stat.to_csv(BASE / "ml_ready_stat.txt", sep="\t", index=False, float_format="%.6f")

# ─── PARTIE 2 : MOBILE ────────────────────────────────────────────
print("\n" + "="*60)
print("PARTIE 2 : MOBILE")
print("="*60)

collars = load_tab(collars_mob[0])
kobo = load_tab(kobo_mob[0])
dgps = load_tab(dgps_mob[0])
serials = load_tab(paths["serials"])

# ... (suite identique au script précédent pour le mobile)

print("\n✅ Terminé. Fichiers générés dans :", BASE)