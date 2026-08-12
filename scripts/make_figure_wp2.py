#make_figure_wp2.py
# Figure nouvelle : distribution des R2 observes sur 50 partitions spatiales 20/10
# Boxplots par config + ligne zero + marqueur sur la partition publiee (seed 42)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("wp2_r2_partitions.txt")   # colonnes A B C D
labels = ["A\n5 env", "B\n+ aspect", "C\n+ HDOP, nsat", "D\n+ horiz. acc."]
published = [-0.082, 0.001, -0.012, 0.038]   # partition seed 42 (Table 2)

fig, ax = plt.subplots(figsize=(7, 4.5))
bp = ax.boxplot([data[:, i] for i in range(4)], labels=labels, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=5),
                medianprops=dict(color="black"))
ax.axhline(0, color="black", linestyle="--", linewidth=1)
for i, v in enumerate(published):
    ax.plot(i + 1, v, marker="*", color="#d62728", markersize=13, zorder=5,
            label="Published partition (seed 42)" if i == 0 else None)
ax.set_ylabel("Out-of-sample R² (error magnitude)")
ax.set_title("Sensitivity to spatial partitioning:\n50 random 20/10 station splits per configuration",
             fontsize=10)
ax.legend(loc="lower left", fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("figure_wp2_partitions.png", dpi=300)
print("Sauvegarde : figure_wp2_partitions.png")