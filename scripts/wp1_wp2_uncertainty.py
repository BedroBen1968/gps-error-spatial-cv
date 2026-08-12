#wp1_wp2_uncertainty.py
# WP1 : LOSO primaire avec sauvegarde des predictions out-of-fold,
#       puis IC bootstrap (re-echantillonnage des 30 stations) sur R2, RMSE, eta2.
# WP2 : sensibilite du hold-out 20/10 : 50 partitions aleatoires x configs A-D.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

start = time.time()
df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
stations_all = df["station"].values
unique_s = np.unique(stations_all)

# ---------- features ----------
env5 = ["canopy_cover", "elevation", "skyview", "slope", "tri"]
base_feats = env5 + ["hdop", "number_satellites", "horizontal_accuracy"]
# aspect : numerique si possible, sinon encode
asp_col = "aspect" if "aspect" in df.columns else "aspect_original"
asp = pd.to_numeric(df[asp_col], errors="coerce")
if asp.isna().all():
    asp = pd.Series(pd.factorize(df[asp_col])[0], index=df.index, dtype=float)
df["_aspect_num"] = asp
feats9 = base_feats + ["_aspect_num"]

dE = df["delta_easting"].values.astype(float)
dN = df["delta_northing"].values.astype(float)

def rf():
    return RandomForestRegressor(n_estimators=300, max_depth=8,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)

# ================= WP1 : LOSO avec sauvegarde OOF =================
print("WP1 : LOSO primaire (30 folds, 2 regresseurs)...")
oof = pd.DataFrame({"station": stations_all, "dE": dE, "dN": dN,
                    "predE": np.nan, "predN": np.nan,
                    "baseE": np.nan, "baseN": np.nan})
X_all = df[feats9].copy()
for k, s in enumerate(unique_s):
    tr = stations_all != s
    te = ~tr
    med = X_all[tr].median()
    Xtr = X_all[tr].fillna(med).values
    Xte = X_all[te].fillna(med).values
    mE = rf().fit(Xtr, dE[tr]); mN = rf().fit(Xtr, dN[tr])
    oof.loc[te, "predE"] = mE.predict(Xte)
    oof.loc[te, "predN"] = mN.predict(Xte)
    oof.loc[te, "baseE"] = dE[tr].mean()
    oof.loc[te, "baseN"] = dN[tr].mean()
    print(f"  fold {k+1}/30 ({time.time()-start:.0f}s)")
oof.to_csv("wp1_oof_predictions.csv", index=False)

def pooled_metrics(d):
    r2E = r2_score(d["dE"], d["predE"]); r2N = r2_score(d["dN"], d["predN"])
    rmse_m = np.sqrt(np.mean((d["dE"]-d["predE"])**2 + (d["dN"]-d["predN"])**2))
    rmse_b = np.sqrt(np.mean((d["dE"]-d["baseE"])**2 + (d["dN"]-d["baseN"])**2))
    return r2E, r2N, rmse_m, rmse_b

def eta2(d, col):
    grand = d[col].mean()
    ssb = sum(len(g)*(g[col].mean()-grand)**2 for _, g in d.groupby("station"))
    sst = ((d[col]-grand)**2).sum()
    return ssb/sst

obs = pooled_metrics(oof)
obs_eta = (eta2(oof, "dE"), eta2(oof, "dN"))

print("Bootstrap stations (10 000 replicats, sans re-fit)...")
rng = np.random.default_rng(42)
groups = {s: oof[oof["station"] == s] for s in unique_s}
B = 10000
boot = np.empty((B, 6))
for b in range(B):
    pick = rng.choice(unique_s, size=len(unique_s), replace=True)
    d = pd.concat([groups[s] for s in pick], ignore_index=True)
    m = pooled_metrics(d)
    boot[b] = (*m, eta2(d, "dE"), eta2(d, "dN"))
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

names = ["R2_East_LOSO", "R2_North_LOSO", "RMSE_model", "RMSE_baseline",
         "eta2_East", "eta2_North"]
obs_all = (*obs, *obs_eta)
lines = ["WP1 : IC bootstrap 95% (re-echantillonnage des 30 stations, B=10000)"]
for n, o, l, h in zip(names, obs_all, lo, hi):
    lines.append(f"{n:<15s} obs={o:+.3f}   IC95=[{l:+.3f}, {h:+.3f}]")

# ================= WP2 : 50 partitions x configs A-D =================
print("WP2 : 50 partitions 20/10 x 4 configs (cible magnitude)...")
y_mag = np.sqrt(dE**2 + dN**2)
aspect_dum = pd.get_dummies(df[asp_col], prefix="aspect")
cfg = {
 "A": df[env5],
 "B": pd.concat([df[env5], aspect_dum], axis=1),
 "C": pd.concat([df[env5 + ["hdop", "number_satellites"]], aspect_dum], axis=1),
 "D": pd.concat([df[env5 + ["hdop", "number_satellites", "horizontal_accuracy"]], aspect_dum], axis=1),
}
res = {k: [] for k in cfg}
for p in range(50):
    rngp = np.random.default_rng(100 + p)
    perm = rngp.permutation(unique_s)
    tr_s, te_s = set(perm[:20]), set(perm[20:])
    trm = np.isin(stations_all, list(tr_s)); tem = ~trm
    for k, X in cfg.items():
        med = X[trm].median()
        m = rf().fit(X[trm].fillna(med).values, y_mag[trm])
        res[k].append(r2_score(y_mag[tem], m.predict(X[tem].fillna(med).values)))
    if (p+1) % 10 == 0:
        print(f"  partition {p+1}/50 ({time.time()-start:.0f}s)")

lines.append("\nWP2 : distribution des R2 observes sur 50 partitions 20/10")
lines.append(f"{'cfg':<4s}{'moy':>8s}{'med':>8s}{'sd':>7s}{'q5':>8s}{'q95':>8s}{'%>0':>6s}")
for k in cfg:
    a = np.array(res[k])
    lines.append(f"{k:<4s}{a.mean():>+8.3f}{np.median(a):>+8.3f}{a.std():>7.3f}"
                 f"{np.percentile(a,5):>+8.3f}{np.percentile(a,95):>+8.3f}"
                 f"{100*np.mean(a>0):>5.0f}%")
np.savetxt("wp2_r2_partitions.txt",
           np.column_stack([res[k] for k in cfg]),
           header="A B C D", fmt="%+.4f")

out = "\n".join(lines) + f"\nTemps total : {time.time()-start:.0f}s\n"
print("\n" + out)
open("wp1_wp2_results.txt", "w").write(out)
print("Sauvegardes : wp1_oof_predictions.csv, wp2_r2_partitions.txt, wp1_wp2_results.txt")