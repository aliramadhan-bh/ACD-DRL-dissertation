import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Baseline\n(Full Configuration)',
    'Gentle Reward\nShaping',
    'Hierarchical RL',
    'Extended Training\n(20M steps)',
    'RecurrentPPO\n(LSTM policy)',
    'Enterprise\nFocus',
    'Masked PPO\n(Action Masking)',
]

means = [-17.93, -18.51, -19.16, -19.49, -19.57, -23.34, -85.34]
stds  = [  7.69,   8.66,   9.09,   8.42,   9.65,  11.18,  56.91]

# Reverse so best is at top
labels = labels[::-1]
means  = means[::-1]
stds   = stds[::-1]

colors = [
    'red',    # Masked PPO
    'orange', # Enterprise Focus
    'green',  # RecurrentPPO
    'green',  # Extended Training
    'green',  # Hierarchical RL
    'green',  # Gentle Shaping
    'steelblue',  # Baseline
]

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(
    range(len(labels)),
    means,
    xerr=stds,
    color=colors,
    edgecolor='black',
    linewidth=0.6,
    capsize=5,
    error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'dimgray'},
    height=0.6,
)

# Success threshold line
ax.axvline(x=-20, color='black', linestyle='--', linewidth=1.5, label='Success threshold (−20)')

# Value labels on each bar
for i, (mean, std) in enumerate(zip(means, stds)):
    if mean < -40:
        # place label inside the bar, near right edge
        ax.text(mean + 1.5, i, f'{mean:.2f}', va='center', ha='left', fontsize=9.5, fontweight='bold', color='white')
    else:
        ax.text(mean + std + 0.5, i, f'{mean:.2f}', va='center', ha='left', fontsize=9.5, fontweight='bold')

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Mean Episode Reward', fontsize=11)
ax.set_title(
    'Mean Reward Comparison — All Retained Approaches\n(Seed 200, 30 Episodes)',
    fontsize=12, fontweight='bold', pad=14,
)

ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(-115, 5)
ax.grid(axis='x', linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure_4_2_results_comparison.png', dpi=300, bbox_inches='tight')
print("Saved figure_4_2_results_comparison.png")
