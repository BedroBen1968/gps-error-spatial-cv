#wp4_spatial_diagnostics.py
# WP4 : diagnostics spatiaux au niveau station (n = 30)
# (a) Moran's I a ponderation par bandes de distance (exigence R1.9 / R3.3)
# (b) Variogrammes directionnels 2 secteurs (E-O vs N-S) avec bootstrap (R3.2)
# Residus = moyennes station de delta_easting / delta_northing (biais brut)
import numpy as np
import pandas as pd

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
st = df.groupby("station").agg(
    x=("mean_x", "first"), y=("mean_y", "first"),
    rE=("delta_easting", "mean"), rN=("delta_northing", "mean")).reset_index()
n = len(st)
print(f"{n} stations chargees")
xy = st[["x", "y"]].values.astype(float)
D = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(-1))
iu = np.triu_indices(n, 1)
print(f"distances inter-stations : min={D[iu].min():.0f} m, mediane={np.median(D[iu]):.0f} m, "
      f"moyenne={D[iu].mean():.0f} m, max={D[iu].max():.0f} m")

rng = np.random.default_rng(42)

def moran(z, W):
    z = z - z.mean()
    s0 = W.sum()
    I = (n / s0) * (z @ W @ z) / (z @ z)
    # test par permutation (999) : reference standard pour petits n
    null = np.empty(999)
    for i in range(999):
        zp = rng.permutation(z)
        null[i] = (n / s0) * (zp @ W @ zp) / (zp @ zp)
    p = (np.sum(np.abs(null - null.mean()) >= np.abs(I - null.mean())) + 1) / 1000
    return I, -1.0 / (n - 1), p

lines = ["WP4a : Moran's I, ponderation par bandes de distance (test par permutation, 999)"]
print("\n" + lines[0])
bands = [(0, 500), (0, 1000), (0, 1500), (0, 2000), (500, 1500)]
for lo, hi in bands:
    W = ((D > lo) & (D <= hi)).astype(float)
    np.fill_diagonal(W, 0)
    links = int(W.sum() / 2)
    if links < 10:
        msg = f"  bande {lo}-{hi} m : {links} paires, trop peu, non calcule"
        print(msg); lines.append(msg); continue
    for name, z in [("East", st["rE"].values), ("North", st["rN"].values)]:
        I, E0, p = moran(z, W)
        msg = (f"  bande {lo:>4d}-{hi:<4d} m ({links:3d} paires) {name:<5s} "
               f"I={I:+.3f} (E[I]={E0:+.3f}) p={p:.3f}")
        print(msg); lines.append(msg)

# aussi inverse-distance (toutes paires), pour completude
W = 1.0 / np.where(D > 0, D, np.inf)
lines.append("WP4a bis : ponderation inverse-distance (toutes paires)")
print("\n" + lines[-1])
for name, z in [("East", st["rE"].values), ("North", st["rN"].values)]:
    I, E0, p = moran(z, W)
    msg = f"  inv-dist {name:<5s} I={I:+.3f} (E[I]={E0:+.3f}) p={p:.3f}"
    print(msg); lines.append(msg)

# ---------- (b) variogrammes directionnels 2 secteurs ----------
lines.append("\nWP4b : semivariance empirique directionnelle (2 secteurs, bootstrap 1000)")
print("\n" + lines[-1])
dx = xy[:, 0][:, None] - xy[:, 0][None, :]
dy = xy[:, 1][:, None] - xy[:, 1][None, :]
ang = np.degrees(np.arctan2(dy, dx)) % 180  # orientation de la paire
sector = np.where((ang < 45) | (ang >= 135), "EO", "NS")  # +-45 deg autour de E-O
lag_edges = [0, 700, 1400, 2100, 3000]

def semivar(z, mask_pairs):
    out = []
    for a, b in zip(lag_edges[:-1], lag_edges[1:]):
        sel = mask_pairs & (D > a) & (D <= b)
        m = sel[iu]
        npairs = m.sum()
        if npairs < 5:
            out.append((f"{a}-{b}", npairs, np.nan)); continue
        g = 0.5 * np.mean((z[iu[0]][m] - z[iu[1]][m])**2)
        out.append((f"{a}-{b}", npairs, g))
    return out

for name, z in [("East", st["rE"].values), ("North", st["rN"].values)]:
    for sec in ["EO", "NS"]:
        mask = (sector == sec)
        sv = semivar(z, mask)
        # bootstrap stations pour l'instabilite
        boots = {lab: [] for lab, _, _ in sv}
        for _ in range(1000):
            pick = rng.choice(n, n, replace=True)
            zb = z[pick]; Db = D[np.ix_(pick, pick)]
            secb = sector[np.ix_(pick, pick)]
            iub = np.triu_indices(n, 1)
            for a, b in zip(lag_edges[:-1], lag_edges[1:]):
                m = (secb == sec)[iub] & (Db[iub] > a) & (Db[iub] <= b)
                if m.sum() >= 5:
                    boots[f"{a}-{b}"].append(0.5*np.mean((zb[iub[0]][m]-zb[iub[1]][m])**2))
        for lab, npairs, g in sv:
            bb = boots[lab]
            if len(bb) > 100 and not np.isnan(g):
                lo_, hi_ = np.percentile(bb, [2.5, 97.5])
                msg = f"  {name:<5s} {sec} lag {lab:>9s} m ({npairs:2d} paires) gamma={g:7.2f} IC95=[{lo_:.2f},{hi_:.2f}]"
            else:
                msg = f"  {name:<5s} {sec} lag {lab:>9s} m ({npairs:2d} paires) gamma=insuffisant"
            print(msg); lines.append(msg)

open("wp4_results.txt", "w").write("\n".join(lines) + "\n")
print("\nSauvegarde : wp4_results.txt")