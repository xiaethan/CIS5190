import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 
models = ['TF-IDF +\nLogistic Regression', 'TF-IDF +\nNeural Network', 'RoBERTa']
local_acc    = [80, 78, 85]
leader_acc   = [77, 75, 80]
baseline     = 66.49
 
x = np.arange(len(models))
width = 0.35
 
fig, ax = plt.subplots(figsize=(9, 5))
 
bars1 = ax.bar(x - width/2, local_acc,  width, label='Local accuracy',       color='#378ADD', zorder=3)
bars2 = ax.bar(x + width/2, leader_acc, width, label='Leaderboard accuracy',  color='#1D9E75', zorder=3)
 
ax.axhline(y=baseline, color='#E24B4A', linewidth=1.4, linestyle='--', zorder=4, label=f'Baseline ({baseline}%)')
 
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=10, color='#0C447C')
 
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=10, color='#085041')
 
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison: Local vs Leaderboard', fontsize=13, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(60, 92)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.5)
 
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to model_comparison.png")
plt.show()