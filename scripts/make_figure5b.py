#make_figure5b.py
# Panneau (b) de la Figure 5 : importance sous validation spatiale (WP5)
# Barres horizontales du Delta RMSE avec IC bootstrap 95%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

feats  = ["TRI", "Canopy cover", "Slope", "Sky-view", "Horiz. accuracy",
          "Number of satellites", "HDOP", "Elevation", "Aspect"]
mean   = [0.406, 0.204, 0.172, 0.045, 0.015, 0.012, -0.009, -0.084, -0.142]
lo     = [0.067, -0.074, -0.053, -0.182, -0.126, 0.003, -0.044, -0.195, -0.313]
hi     = [0.734, 0.491, 0.407, 0.278, 0.147, 0.023, 0.022, 0.020, 0.011]

sig    = [l > 0 or h < 0 for l, h in zip(lo, hi)]   # IC excluant zero
colors = ["#3182bd" if s else "#c6dbef" for s in sig]

y = np.arange(len(feats))[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4.5))
err = [np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)]
ax.barh(y, mean, xerr=err, color=colors, edgecolor="black", linewidth=0.7,
        error_kw=dict(ecolor="black", lw=1, capsize=3))
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=9)
ax.set_xlabel("ΔRMSE when feature is replaced by donor-station values (m)")
ax.set_title("(b) Feature reliance under spatial LOSO validation\n"
             "(donor-station replacement, 95% station-bootstrap CI)", fontsize=10)
plt.tight_layout()
plt.savefig("figure5b_spatial_importance.png", dpi=300)
print("Sauvegarde : figure5b_spatial_importance.png")