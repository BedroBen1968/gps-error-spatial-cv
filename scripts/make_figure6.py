#make_figure6.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

configs = ["A\n5 env", "B\n+ aspect", "C\n+ HDOP, nsat", "D\n+ horiz. accuracy"]
r2 = [-0.0817, 0.0014, -0.0116, 0.0379]
pvals = [0.133, 0.056, 0.055, 0.028]

colors = ["#9ecae1", "#9ecae1", "#9ecae1", "#3182bd"]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(configs, r2, color=colors, edgecolor="black", linewidth=0.8)

# ligne R2 = 0
ax.axhline(0, color="black", linestyle="--", linewidth=1)

# asterisque uniquement sur D (p < 0.05)
ax.text(3, r2[3] + 0.004, "*", ha="center", fontsize=16, fontweight="bold")

# p-values sous chaque barre
for i, (v, p) in enumerate(zip(r2, pvals)):
    offset = -0.012 if v >= 0 else 0.006
    ax.text(i, v + offset, f"p = {p:.3f}", ha="center", fontsize=8, color="dimgray")

ax.set_ylabel("Out-of-sample R² (fixed spatial hold-out)")
ax.set_ylim(-0.12, 0.08)
ax.set_title("Incremental predictor enrichment under spatial hold-out\n(station-block permutation null, 1 000 permutations)", fontsize=10)
plt.tight_layout()
plt.savefig("figure6.png", dpi=300)
print("Figure sauvegardee : figure6.png")