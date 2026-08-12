#make_figure11.py
# Figure 11 : violations de geofence calculees, 5 approches x 4 geometries
# Recalcul traçable ; verifie le -86% (naive) et le ~x3 (LOSO) de la Section 3.8
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import time

start = time.time()
df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
N = len(df)
sites = df["station"].values
dE = df["delta_easting"].values.astype(float)
dN = df["delta_northing"].values.astype(float)

# ---- corrections par approche : erreur corrigee = (dE - predE, dN - predN) ----
corr = {}
corr["Raw"] = (np.zeros(N), np.zeros(N))
corr["Oracle (dGPS)"] = (dE.copy(), dN.copy())

# ML naive 5-fold (pipeline identique a make_figure4)
asp = pd.to_numeric(df["aspect_original"], errors="coerce")
if asp.isna().all(): asp = pd.Series(pd.factorize(df["aspect_original"])[0], dtype=float)
df["_asp"] = asp
feats = ["canopy_cover","elevation","skyview","slope","tri",
         "hdop","number_satellites","horizontal_accuracy","_asp"]
X = df[feats]
def rf(): return RandomForestRegressor(n_estimators=300, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1)
pE = np.empty(N); pN = np.empty(N)
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    med = X.iloc[tr].median()
    pE[te] = rf().fit(X.iloc[tr].fillna(med).values, dE[tr]).predict(X.iloc[te].fillna(med).values)
    pN[te] = rf().fit(X.iloc[tr].fillna(med).values, dN[tr]).predict(X.iloc[te].fillna(med).values)
corr["ML naive (5-fold)"] = (pE, pN)
print(f"naive 5-fold fait ({time.time()-start:.0f}s)")

# ML spatial LOSO : predictions WP1 (verification d'alignement)
oof = pd.read_csv("wp1_oof_predictions.csv")
assert len(oof) == N and np.allclose(oof["dE"].values, dE), "wp1_oof desaligne !"
corr["ML spatial (LOSO)"] = (oof["predE"].values, oof["predN"].values)

# Station-mean (calibration par site, in-sample)
mE = df.groupby("station")["delta_easting"].transform("mean").values
mN = df.groupby("station")["delta_northing"].transform("mean").values
corr["Station-mean"] = (mE, mN)

# ---- geometries ----
stx = df.groupby("station")["d_X"].first(); sty = df.groupby("station")["d_Y"].first()
true_x = df["station"].map(stx).values; true_y = df["station"].map(sty).values
hull = ConvexHull(np.column_stack([stx.values, sty.values]))
hullpath = MplPath(np.column_stack([stx.values, sty.values])[hull.vertices])

def viol_hull(cE, cN):
    tx_in = hullpath.contains_points(np.column_stack([true_x, true_y]), radius=1e-9) | True  # stations = sommets/interieur
    px = true_x + (dE - cE); py = true_y + (dN - cN)
    p_in = hullpath.contains_points(np.column_stack([px, py]), radius=1e-9)
    return 100 * np.mean(tx_in != p_in)

def viol_circle(cE, cN, R):
    d = np.sqrt((dE - cE)**2 + (dN - cN)**2)   # distance au centre (position vraie)
    return 100 * np.mean(d > R)                # vraie position toujours inside

geoms = ["Convex-hull\npasture", "15 m micro-\nenclosure", "+5 m\nbuffer", "+10 m\nbuffer"]
vals = {a: [viol_hull(cE, cN), viol_circle(cE, cN, 15), viol_circle(cE, cN, 20), viol_circle(cE, cN, 25)]
        for a, (cE, cN) in corr.items()}

# ---- verifications cles vs texte 3.8 ----
raw15, nai15, spa15 = vals["Raw"][1], vals["ML naive (5-fold)"][1], vals["ML spatial (LOSO)"][1]
red = 100*(raw15-nai15)/raw15; mult = spa15/raw15
print(f"VERIF 15 m : naive reduit de {red:.0f}% (attendu ~86) ; LOSO x{mult:.2f} (attendu ~3)")

lines = ["GEOFENCING FINAL : % violations par approche x geometrie", ""]
lines.append(f"{'approche':<20s}" + "".join(f"{g.replace(chr(10),' '):>22s}" for g in geoms))
for a in vals:
    lines.append(f"{a:<20s}" + "".join(f"{v:>21.3f}%" for v in vals[a]))
lines.append(f"\nVERIF: naive -{red:.0f}% vs raw (15 m) ; LOSO x{mult:.2f} vs raw (15 m)")
open("results_geofencing_final.txt", "w").write("\n".join(lines) + "\n")

# ---- figure ----
colors = {"Raw":"#9e9e9e","Oracle (dGPS)":"#2e7d32","ML naive (5-fold)":"#3182bd",
          "ML spatial (LOSO)":"#e6550d","Station-mean":"#9467bd"}
x = np.arange(4); w = 0.16
fig, ax = plt.subplots(figsize=(9.5, 5))
for i, a in enumerate(vals):
    ax.bar(x + (i-2)*w, vals[a], w, label=a, color=colors[a], edgecolor="black", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(geoms)
ax.set_ylabel("Calculated fence violations (%)")
ax.legend(frameon=True, fontsize=8.5, ncol=2)
plt.tight_layout(); plt.savefig("figure11_new.png", dpi=300)
print("Sauvegardes : results_geofencing_final.txt, figure11_new.png")