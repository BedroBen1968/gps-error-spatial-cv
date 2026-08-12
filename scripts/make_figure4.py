#make_figure4.py
# Regeneration de la Figure 4 depuis le pipeline WP1 (un seul pipeline partout)
# (a) RMSE avant/apres correction, naive vs LOSO, ligne baseline train-mean
# (b) R2 pooled East/North, naive vs LOSO, IC bootstrap stations sur LOSO
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
asp = pd.to_numeric(df["aspect_original"], errors="coerce")
if asp.isna().all():
    asp = pd.Series(pd.factorize(df["aspect_original"])[0], dtype=float)
df["_asp"] = asp
feats = ["canopy_cover","elevation","skyview","slope","tri",
         "hdop","number_satellites","horizontal_accuracy","_asp"]
dE = df["delta_easting"].values.astype(float)
dN = df["delta_northing"].values.astype(float)
X = df[feats]
rmse_before = np.sqrt(np.mean(dE**2 + dN**2))

def rf(): return RandomForestRegressor(n_estimators=300, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1)

# ---- CV naive 5-fold (memes features, meme pipeline) ----
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pE = np.empty(len(df)); pN = np.empty(len(df)); fold_rmse=[]; fold_r2E=[]; fold_r2N=[]
for tr, te in kf.split(X):
    med = X.iloc[tr].median()
    Xtr, Xte = X.iloc[tr].fillna(med).values, X.iloc[te].fillna(med).values
    mE = rf().fit(Xtr, dE[tr]); mN = rf().fit(Xtr, dN[tr])
    pE[te] = mE.predict(Xte); pN[te] = mN.predict(Xte)
    fold_rmse.append(np.sqrt(np.mean((dE[te]-pE[te])**2 + (dN[te]-pN[te])**2)))
    fold_r2E.append(r2_score(dE[te], pE[te])); fold_r2N.append(r2_score(dN[te], pN[te]))
naive = dict(rmse=np.sqrt(np.mean((dE-pE)**2+(dN-pN)**2)), rmse_sd=np.std(fold_rmse),
             r2E=r2_score(dE,pE), r2N=r2_score(dN,pN),
             r2E_sd=np.std(fold_r2E), r2N_sd=np.std(fold_r2N))

# ---- LOSO depuis wp1_oof_predictions.csv + bootstrap stations ----
d = pd.read_csv("wp1_oof_predictions.csv")
loso = dict(rmse=np.sqrt(np.mean((d.dE-d.predE)**2+(d.dN-d.predN)**2)),
            base=np.sqrt(np.mean((d.dE-d.baseE)**2+(d.dN-d.baseN)**2)),
            r2E=r2_score(d.dE,d.predE), r2N=r2_score(d.dN,d.predN))
rng = np.random.default_rng(42); S = d.station.unique()
G = {s:g for s,g in d.groupby("station")}
boot = {k: [] for k in ["rmse","r2E","r2N"]}
for _ in range(5000):
    dd = pd.concat([G[s] for s in rng.choice(S, len(S), replace=True)])
    boot["rmse"].append(np.sqrt(np.mean((dd.dE-dd.predE)**2+(dd.dN-dd.predN)**2)))
    boot["r2E"].append(r2_score(dd.dE, dd.predE)); boot["r2N"].append(r2_score(dd.dN, dd.predN))
ci = {k: np.percentile(v,[2.5,97.5]) for k,v in boot.items()}

print("NAIVE : RMSE %.2f (sd %.2f) | R2E %.3f R2N %.3f" % (naive['rmse'],naive['rmse_sd'],naive['r2E'],naive['r2N']))
print("LOSO  : RMSE %.2f CI[%.2f,%.2f] base %.2f | R2E %.3f CI[%.3f,%.3f] R2N %.3f CI[%.3f,%.3f]"
      % (loso['rmse'],ci['rmse'][0],ci['rmse'][1],loso['base'],
         loso['r2E'],ci['r2E'][0],ci['r2E'][1],loso['r2N'],ci['r2N'][0],ci['r2N'][1]))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
# (a) RMSE
lab = ["Naive\nbefore","Naive\nafter","LOSO\nbefore","LOSO\nafter"]
val = [rmse_before, naive["rmse"], rmse_before, loso["rmse"]]
err = [[0, naive["rmse_sd"], 0, loso["rmse"]-ci["rmse"][0]],
       [0, naive["rmse_sd"], 0, ci["rmse"][1]-loso["rmse"]]]
cols = ["#bdbdbd","#3182bd","#bdbdbd","#d62728"]
ax[0].bar(lab, val, yerr=err, color=cols, edgecolor="black", capsize=4)
ax[0].axhline(loso["base"], color="black", ls=":", lw=1.2)
ax[0].text(3.45, loso["base"]+0.06, "train-mean\nbaseline", fontsize=7, ha="right")
ax[0].set_ylabel("Horizontal RMSE (m)"); ax[0].set_title("(a)", loc="left")
# (b) R2 pooled
lab2 = ["Naive\nEast","Naive\nNorth","LOSO\nEast","LOSO\nNorth"]
val2 = [naive["r2E"], naive["r2N"], loso["r2E"], loso["r2N"]]
err2 = [[naive["r2E_sd"], naive["r2N_sd"], loso["r2E"]-ci["r2E"][0], loso["r2N"]-ci["r2N"][0]],
        [naive["r2E_sd"], naive["r2N_sd"], ci["r2E"][1]-loso["r2E"], ci["r2N"][1]-loso["r2N"]]]
ax[1].bar(lab2, val2, yerr=err2, color=["#3182bd","#3182bd","#d62728","#d62728"],
          edgecolor="black", capsize=4)
ax[1].axhline(0, color="black", ls="--", lw=1)
ax[1].set_ylabel("Pooled R²"); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig("figure4_new.png", dpi=300)
print("Sauvegarde : figure4_new.png")