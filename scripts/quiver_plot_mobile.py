#quiver_plot_mobile.py
#Script 11
# quiver_plot_mobile.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_stat = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
df_mob = pd.read_csv("ml_ready_mob.txt", sep="\t", low_memory=False)

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy"]

for d in [df_stat, df_mob]:
    aspect_col = "aspect" if "aspect" in d.columns else "aspect_original"
    dummies = pd.get_dummies(d[aspect_col], prefix="aspect")
    for c in ["aspect_East", "aspect_North", "aspect_South", "aspect_West"]:
        if c not in dummies: dummies[c] = 0
    d["__asp_E"] = dummies.get("aspect_East", 0)
    d["__asp_N"] = dummies.get("aspect_North", 0)
    d["__asp_S"] = dummies.get("aspect_South", 0)
    d["__asp_W"] = dummies.get("aspect_West", 0)

feat_names = base + ["__asp_E", "__asp_N", "__asp_S", "__asp_W"]
X_stat = df_stat[feat_names].fillna(df_stat[feat_names].median())
X_mob = df_mob[feat_names].fillna(df_mob[feat_names].median())

y_e = df_stat["delta_easting"].values
y_n = df_stat["delta_northing"].values

rf_e = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_n = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
rf_e.fit(X_stat, y_e); rf_n.fit(X_stat, y_n)

pred_e = rf_e.predict(X_mob)
pred_n = rf_n.predict(X_mob)

x = df_mob["X"].values
y = df_mob["Y"].values

fig, ax = plt.subplots(figsize=(10, 8))
q = ax.quiver(x, y, pred_e, pred_n, df_mob["canopy_cover"].values,
              scale=50, width=0.003, cmap="RdYlGn_r")
ax.set_title("Predicted correction vectors (static->mobile transfer)\nColor = canopy cover (%)")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.colorbar(q, label="Canopy cover (%)")
plt.tight_layout()
plt.savefig("quiver_mobile_corrections.png", dpi=150)
print("Figure sauvegardee : quiver_mobile_corrections.png")
with open("results11_reviewer_tests.txt", "w") as f:
    f.write("Figure generee : quiver_mobile_corrections.png\n")