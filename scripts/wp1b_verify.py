#wp1b_verify.py 
#: diagnostic depuis wp1_oof_predictions.csv (aucun re-fit)
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
d = pd.read_csv("wp1_oof_predictions.csv")
print(f"pooled R2 East  = {r2_score(d['dE'], d['predE']):+.5f}")
print(f"pooled R2 North = {r2_score(d['dN'], d['predN']):+.5f}")
fE = [r2_score(g['dE'], g['predE']) for _, g in d.groupby('station')]
fN = [r2_score(g['dN'], g['predN']) for _, g in d.groupby('station')]
print(f"fold-mean R2 East  = {np.mean(fE):+.3f} (mediane {np.median(fE):+.3f})")
print(f"fold-mean R2 North = {np.mean(fN):+.3f} (mediane {np.median(fN):+.3f})")
rng = np.random.default_rng(42)
stations = d['station'].unique(); groups = {s: g for s, g in d.groupby('station')}
diffs = []
for _ in range(10000):
    dd = pd.concat([groups[s] for s in rng.choice(stations, 30, replace=True)])
    rm = np.sqrt(np.mean((dd['dE']-dd['predE'])**2 + (dd['dN']-dd['predN'])**2))
    rb = np.sqrt(np.mean((dd['dE']-dd['baseE'])**2 + (dd['dN']-dd['baseN'])**2))
    diffs.append(rm - rb)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"RMSE modele - baseline : IC95 = [{lo:+.3f}, {hi:+.3f}] m (>0 = modele pire)")