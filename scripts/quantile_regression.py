#quantile_regression.py
#Script 8 
"""
Script 8 : Quantile Regression (Gradient Boosting)
Reviewer #2 : tester si les intervalles de confiance sont calibres.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
stations = df["station"].unique()
np.random.seed(42)
np.random.shuffle(stations)
train_s = stations[:20]
test_s = stations[20:]

base = ["canopy_cover", "elevation", "skyview", "slope", "tri",
        "hdop", "number_satellites", "horizontal_accuracy"]

# Aspect dummies
dummies = pd.get_dummies(df["aspect"] if "aspect" in df.columns else df["aspect_original"], prefix="aspect")
for c in ["aspect_East", "aspect_North", "aspect_South", "aspect_West"]:
    if c not in dummies: dummies[c] = 0
df["__asp_E"] = dummies.get("aspect_East", 0)
df["__asp_N"] = dummies.get("aspect_North", 0)
df["__asp_S"] = dummies.get("aspect_South", 0)
df["__asp_W"] = dummies.get("aspect_West", 0)

feat_names = base + ["__asp_E", "__asp_N", "__asp_S", "__asp_W"]
X = df[feat_names].fillna(df[feat_names].median()).values
y_e = df["delta_easting"].values
y_n = df["delta_northing"].values

train_mask = df["station"].isin(train_s).values
test_mask = df["station"].isin(test_s).values
X_tr, X_te = X[train_mask], X[test_mask]
ye_tr, ye_te = y_e[train_mask], y_e[test_mask]
yn_tr, yn_te = y_n[train_mask], y_n[test_mask]

# Quantile regression : 5th, 50th (median), 95th percentiles
alphas = [0.05, 0.5, 0.95]
res_str = "[QUANTILE_REGRESSION RESULTATS]\n"

for target_name, ytr, yte in [("Easting", ye_tr, ye_te), ("Northing", yn_tr, yn_te)]:
    preds = {}
    for alpha in alphas:
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=alpha,
            n_estimators=200, max_depth=5,
            min_samples_leaf=5, random_state=42
        )
        gbr.fit(X_tr, ytr)
        preds[alpha] = gbr.predict(X_te)

    # Coverage : vraies valeurs dans [5%, 95%]
    coverage = np.mean((yte >= preds[0.05]) & (yte <= preds[0.95]))
    median_rmse = np.sqrt(np.mean((yte - preds[0.5])**2))
    r2_med = r2_score(yte, preds[0.5])
    interval_width = np.mean(preds[0.95] - preds[0.05])

    res_str += f"\n{target_name}:\n"
    res_str += f"  Coverage 5-95%    : {coverage:.1%} (theorique 90%)\n"
    res_str += f"  RMSE median       : {median_rmse:.2f} m\n"
    res_str += f"  R2 median         : {r2_med:+.4f}\n"
    res_str += f"  Intervalle moyen  : {interval_width:.2f} m\n"

print(res_str)
with open("results8_reviewer_tests.txt", "w") as f:
    f.write(res_str)