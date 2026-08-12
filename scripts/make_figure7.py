#make_figure7.py
# make_figure7_data.py
# Formulations alternatives (i)-(v) sous LOSO 30 folds + generation Figure 7
# Conditions : RF 50 arbres (comme le test precision, Section 2.3.2),
# features environnementales + aspect (formulations env-only) ; (iv) ajoute le self-estimate.
# Sortie : results_formulations_final.txt + figure7_new.png
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time

start = time.time()
df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
sites = df["station"].values
uniq = np.unique(sites)
env = ["canopy_cover", "elevation", "skyview", "slope", "tri", "aspect_original"]
Xenv = df[env].copy()
Xenv["aspect_original"] = pd.to_numeric(Xenv["aspect_original"], errors="coerce")
if Xenv["aspect_original"].isna().all():
    Xenv["aspect_original"] = pd.factorize(df["aspect_original"])[0]
Xext = Xenv.copy(); Xext["horizontal_accuracy"] = df["horizontal_accuracy"]

mag = np.sqrt(df["delta_easting"]**2 + df["delta_northing"]**2).values
targets = {
    "(i) Error magnitude":            (Xenv, mag),
    "(ii) Log-magnitude":             (Xenv, np.log1p(mag)),
    "(iii) Precision (environment-only)": (Xenv, df["precision"].values),
    "(iv) Precision (+ self-estimate)":   (Xext, df["precision"].values),
    "(v) Aspect-canopy group means":  (Xenv, df.groupby(["aspect_original","location_type"])["precision"].transform("mean").values),
}

def rf():
    return RandomForestRegressor(n_estimators=50, max_depth=8,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)

res = {}
for name, (X, y) in targets.items():
    y = np.asarray(y, dtype=float)
    pred = np.empty_like(y); base = np.empty_like(y)
    for s in uniq:
        tr = sites != s; te = ~tr
        med = X[tr].median()
        m = rf().fit(X[tr].fillna(med).values, y[tr])
        pred[te] = m.predict(X[te].fillna(med).values)
        base[te] = y[tr].mean()
    rmse_m = np.sqrt(np.mean((y - pred)**2))
    rmse_b = np.sqrt(np.mean((y - base)**2))
    red = 100.0 * (rmse_b - rmse_m) / rmse_b     # % reduction vs train-mean (positif = mieux)
    ss = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    res[name] = (red, ss, rmse_m, rmse_b)
    print(f"{name:<38s} reduction={red:+6.2f}%  R2={ss:+.3f}  ({time.time()-start:.0f}s)")

lines = ["FORMULATIONS ALTERNATIVES SOUS LOSO (RF 50 arbres) : reduction RMSE vs train-mean",
         f"{'formulation':<40s}{'red.%':>8s}{'R2':>9s}{'RMSE_m':>9s}{'RMSE_b':>9s}"]
for k, (r, s2, rm, rb) in res.items():
    lines.append(f"{k:<40s}{r:>+8.2f}{s2:>+9.3f}{rm:>9.3f}{rb:>9.3f}")
lines.append("(vi) Circular direction : angular error 87.9 deg (model) vs 68.0 deg (aspect-mean baseline) vs 90 deg (chance) ; reported separately")
open("results_formulations_final.txt", "w").write("\n".join(lines) + "\n")

# ---------- Figure 7 ----------
labels = list(res.keys()) + ["(vi) Circular direction\n(worse than aspect-mean baseline)"]
vals = [res[k][0] for k in res] + [-22.0]   # (vi) : (68.0-87.9)/90 en %, cf. legende
sig = [False, False, True, False, False, False]
colors = ["#3182bd" if s else "#c6dbef" for s in sig]
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.8)
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)
ax.set_xlabel("RMSE reduction vs. train-mean baseline (%)")
i3 = len(labels) - 1 - 2
ax.text(res["(iii) Precision (environment-only)"][0] + 0.4, i3, "p = 0.005",
        va="center", fontsize=8, color="dimgray")
plt.tight_layout()
plt.savefig("figure7_new.png", dpi=300)
print("Sauvegardes : results_formulations_final.txt, figure7_new.png")