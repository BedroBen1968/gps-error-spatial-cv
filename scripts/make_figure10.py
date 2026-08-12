#make_figure10.py
# Figure 10 : diagnostic systematique des neuf hypotheses alternatives (H1-H9)
# Reconstruction propre, chiffres a jour (Table 2, Sections 3.4-3.7)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

H = [
 ("H1: Sample size too small?",
  "Test: LOSO (30 folds) + leave-one-aspect-out",
  "If true: coarser grouping improves R2",
  "Result: Identical failure. Ruled out.", "x"),
 ("H2: Wrong algorithm?",
  "Test: Ridge, Kriging, SVR, 3 RF configs",
  "If true: at least one succeeds",
  "Result: All fail identically. Ruled out.", "x"),
 ("H3: Wrong target formulation?",
  "Test: 6 alternative formulations",
  "If true: one shows environmental signal",
  "Result: Only precision (R2 = 0.025). Ruled out.", "x"),
 ("H4: No signal vs too weak?",
  "Test: enrichment + station-block permutation",
  "If true: strict null if no real signal",
  "Result: Weak signal, 10-15x too weak.", "!"),
 ("H5: Unmodeled satellite geometry?",
  "Test: receiver-metadata proxy (HDOP/sqrt nsat)",
  "If true: proxy important, RMSE drops",
  "Result: Negligible importance. Ruled out.", "x"),
 ("H6: Recoverable spatial structure?",
  "Test: distance-band Moran's I + directional variograms",
  "If true: significant, corroborated structure",
  "Result: No detectable structure. Ruled out.", "x"),
 ("H7: Static regime not representative?",
  "Test: static-to-mobile + mobile-to-mobile",
  "If true: mobile-to-mobile succeeds",
  "Result: Both fail. Ruled out.", "x"),
 ("H8: Missing feature interactions?",
  "Test: RF with polynomial interactions",
  "If true: improved performance",
  "Result: Performance drops 16.6%. Ruled out.", "x"),
 ("H9: Useful uncertainty intervals?",
  "Test: quantile regression (5th-95th)",
  "If true: calibrated, narrow enough",
  "Result: Under-calibrated, too wide. Ruled out.", "x"),
]

RED, GREEN, DRED, DGREEN = "#f2b8b5", "#a8d5b0", "#7a1f1f", "#1e5c2f"
fig, ax = plt.subplots(figsize=(13, 16))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

top, bh, gap = 0.965, 0.088, 0.014
xbox, wbox = 0.02, 0.44
ys = []
for i, (title, l1, l2, res, mark) in enumerate(H):
    y = top - i * (bh + gap) - bh
    ys.append(y)
    ax.add_patch(FancyBboxPatch((xbox, y), wbox, bh,
                 boxstyle="round,pad=0.006", fc="white", ec="0.45", lw=1.2))
    ax.text(xbox+0.015, y+bh-0.016, title, fontsize=11.5, fontweight="bold", va="top")
    ax.text(xbox+0.015, y+bh-0.038, l1, fontsize=9.5, va="top")
    ax.text(xbox+0.015, y+bh-0.056, l2, fontsize=9.5, va="top")
    ax.text(xbox+0.015, y+bh-0.076, res, fontsize=9.8, fontweight="bold", va="top",
            color=DGREEN if mark == "!" else DRED)
    c = Circle((0.505, y+bh/2), 0.016, fc=GREEN if mark == "!" else RED,
               ec=DGREEN if mark == "!" else DRED, lw=1.4, zorder=3)
    ax.add_patch(c)
    ax.text(0.505, y+bh/2, mark, ha="center", va="center", fontsize=11,
            fontweight="bold", color=DGREEN if mark == "!" else DRED, zorder=4)
ax.plot([0.545, 0.545], [ys[-1], top], ls="--", color="0.4", lw=1.2)

# boite verte (conclusion), a hauteur de H4
gy, gh = ys[3] - 0.035, 0.16
ax.add_patch(FancyBboxPatch((0.60, gy), 0.375, gh, boxstyle="round,pad=0.008",
             fc="#8fce9b", ec=DGREEN, lw=2))
ax.text(0.7875, gy+gh-0.022, "Genuine boundary condition",
        fontsize=13.5, fontweight="bold", ha="center", va="top")
ax.text(0.7875, gy+gh-0.052,
        "A weak signal is detectable but not environmental:\n"
        "configuration D (p = 0.028) does not survive Holm\n"
        "adjustment; only the environment-only precision\n"
        "signal does (p = 0.005), an order of magnitude\n"
        "below operational relevance.",
        fontsize=10.2, ha="center", va="top")
ax.text(0.7875, gy+0.014, "Not a failure. A characterization.",
        fontsize=11.5, fontweight="bold", ha="center", color=DGREEN)
ax.annotate("", xy=(0.60, gy+gh/2), xytext=(0.545, ys[3]+bh/2),
            arrowprops=dict(arrowstyle="->", lw=1.6, color=DGREEN))

# legende
ly = gy - 0.07
ax.text(0.60, ly+0.038, "Legend:", fontsize=11, fontweight="bold")
ax.add_patch(Circle((0.615, ly+0.012), 0.011, fc=RED, ec=DRED, lw=1.2))
ax.text(0.615, ly+0.012, "x", ha="center", va="center", fontsize=9, color=DRED, fontweight="bold")
ax.text(0.635, ly+0.012, "Alternative ruled out", fontsize=10, va="center")
ax.add_patch(Circle((0.615, ly-0.018), 0.011, fc=GREEN, ec=DGREEN, lw=1.2))
ax.text(0.615, ly-0.018, "!", ha="center", va="center", fontsize=9, color=DGREEN, fontweight="bold")
ax.text(0.635, ly-0.018, "Partial finding (signal too weak)", fontsize=10, va="center")

plt.savefig("figure10_new.png", dpi=300, bbox_inches="tight")
print("Sauvegarde : figure10_new.png")