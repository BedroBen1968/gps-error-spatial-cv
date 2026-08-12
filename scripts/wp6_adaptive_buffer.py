#wp6_adaptive_buffer.py
# wp6_adaptive_buffer.py
# WP6 : validation retrospective du buffer adaptatif par classe d'habitat
# + controle a budget de surface egal (uniforme ~8.24 m, aire = adaptatif P90)
import numpy as np
import pandas as pd

df = pd.read_csv("ml_ready_stat.txt", sep="\t", low_memory=False)
err = np.sqrt(df["delta_easting"].astype(float)**2 + df["delta_northing"].astype(float)**2)
df["_err"] = err
cls = df["location_type"].values

print("=== Echelles d'erreur par classe (s_h) ===")
scales = {}
for h in ["open", "closed"]:
    e = df.loc[cls == h, "_err"]
    scales[h] = {"P90": e.quantile(0.90), "P95": e.quantile(0.95),
                 "mean": e.mean(), "N": len(e)}
    print(f"{h:<7s} N={len(e):5d}  mean={e.mean():5.2f} m  "
          f"P90={e.quantile(0.90):5.2f} m  P95={e.quantile(0.95):5.2f} m")

R0 = 15.0

def violation_rate(W_per_fix):
    coll_in = df["_err"].values <= (R0 + W_per_fix)
    return (~coll_in).mean()

def area_factor(Wd):
    return sum(((R0 + Wd[h])**2 / R0**2) * (cls == h).mean() for h in ["open", "closed"])

strategies = [(f"uniforme +{w:.0f} m", {"open": w, "closed": w}) for w in [0.0, 5.0, 10.0]]

# adaptatifs W_h = z * s_h
for z, q in [(1.0, "P90"), (1.0, "P95"), (1.5, "P95")]:
    strategies.append((f"adaptatif z={z:.1f} x {q}",
                       {h: z * scales[h][q] for h in ["open", "closed"]}))

# CONTROLE : uniforme a budget de surface egal a l'adaptatif P90
Wd_p90 = {h: scales[h]["P90"] for h in ["open", "closed"]}
A_p90 = area_factor(Wd_p90)
W_eq = R0 * (np.sqrt(A_p90) - 1.0)
strategies.append((f"uniforme +{W_eq:.2f} m (aire=P90)", {"open": W_eq, "closed": W_eq}))

print(f"\n=== Strategies (micro-enclosure {R0:.0f} m) ===")
header = f"{'strategie':<30s}{'W_open':>8s}{'W_closed':>9s}{'viol.%':>8s}{'aire x':>8s}"
print(header)
rows = []
for name, Wd in strategies:
    W_fix = np.where(cls == "open", Wd["open"], Wd["closed"])
    v = violation_rate(W_fix); a = area_factor(Wd)
    rows.append(f"{name:<30s}{Wd['open']:>8.2f}{Wd['closed']:>9.2f}{100*v:>7.3f}%{a:>8.2f}")
    print(rows[-1])

out = ["WP6 : buffer adaptatif (classes open/closed du design original) vs uniforme",
       f"s_h : open P90={scales['open']['P90']:.2f} P95={scales['open']['P95']:.2f} m | "
       f"closed P90={scales['closed']['P90']:.2f} P95={scales['closed']['P95']:.2f} m",
       "", header] + rows
open("wp6_results.txt", "w").write("\n".join(out) + "\n")
print("\nSauvegarde : wp6_results.txt")