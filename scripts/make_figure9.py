#make_figure9.py
#Figure 9 : ridgelines des composantes d'erreur GPS par aspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("ml_ready_stat.txt", sep='\t')
# Grouper par aspect
aspects = ['North', 'East', 'South', 'West']
colors = ['#e74c3c', '#27ae60', '#3498db', '#9b59b6']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title in zip(axes, ['delta_easting', 'delta_northing'], 
                          ['ΔEasting (m)', 'ΔNorthing (m)']):
    offset = 0
    for aspect, color in zip(aspects, colors):
        data = df[df['aspect'] == aspect][col].dropna()
        x = np.linspace(data.min(), data.max(), 500)
        kde = stats.gaussian_kde(data)
        y = kde(x) / kde(x).max() * 0.8
        
        ax.fill_between(x, offset, offset + y, color=color, alpha=0.6)
        ax.plot(x, offset + y, color=color, linewidth=1.5)
        ax.text(data.max() + 0.5, offset + 0.3, 
                f'{aspect}-facing\nμ={data.mean():+.1f} m, σ={data.std():.1f} m',
                fontsize=9, va='center', color=color, fontweight='bold')
        offset += 1.2
    
    ax.set_xlabel(title, fontsize=11)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig("figure9_ridgeline_realdata.png", dpi=300)