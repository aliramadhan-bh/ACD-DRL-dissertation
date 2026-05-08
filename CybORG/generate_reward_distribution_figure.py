"""
Generate Episode Reward Distribution Figure for Baseline Model
"""
import matplotlib.pyplot as plt
import numpy as np
import ast
import csv

# Read the CSV without pandas (avoids NumPy 2.x / pyarrow incompatibility)
csv_path = 'FRESH_SEED200_TEST.csv'
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    row = next(reader)

episode_rewards = np.array(ast.literal_eval(row['episode_rewards']))
mean_reward  = float(row['mean_reward'])
std_reward   = float(row['std_reward'])
worst_reward = float(row['worst_reward'])
success_threshold = -20

# Separate successful and failed episodes
successful_rewards = episode_rewards[episode_rewards >= success_threshold]
failed_rewards = episode_rewards[episode_rewards < success_threshold]

# Figure setup
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

# Create histogram bins
bins = np.arange(-45, -5, 5)  # Bins from -45 to -5 with width 5

# Plot histograms
ax.hist(successful_rewards, bins=bins, color='#27AE60', alpha=0.7,
        edgecolor='#2C3E50', linewidth=1.5, label='Successful (≥ -20)')
ax.hist(failed_rewards, bins=bins, color='#E74C3C', alpha=0.7,
        edgecolor='#2C3E50', linewidth=1.5, label='Failed (< -20)')

# Add vertical lines for mean and threshold
ax.axvline(mean_reward, color='#2C3E50', linestyle='--', linewidth=2.5,
          label=f'Mean: {mean_reward:.2f}')
ax.axvline(success_threshold, color='#8E44AD', linestyle='--', linewidth=2.5,
          label=f'Success Threshold: {success_threshold}')

# Add text labels for the lines
# Mean: above bars, just to the right of the mean dashed line
ax.text(mean_reward + 0.3, 0.97, f'Mean\n{mean_reward:.2f}',
       ha='left', va='top', fontsize=10, fontweight='bold',
       color='#2C3E50',
       transform=ax.get_xaxis_transform(),
       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                edgecolor='#2C3E50', linewidth=1.5))

# Threshold: just inside the plot bottom, to the left of the threshold line
ax.text(success_threshold - 0.4, 0.06, f'Success Threshold  {success_threshold}',
       ha='right', va='bottom', fontsize=9.5, fontweight='bold',
       color='#8E44AD',
       transform=ax.get_xaxis_transform(),
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                edgecolor='#8E44AD', linewidth=1.5))

# Labels and title
ax.set_xlabel('Episode Reward', fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_ylabel('Number of Episodes', fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_title('Baseline Model — Episode Reward Distribution (Seed 200, 30 Episodes)',
            fontsize=14, fontweight='bold', color='#2C3E50', pad=20)

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)

# Format y-axis to show integer counts
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Statistics box
stats_text = f'N = {len(episode_rewards)} episodes\n'
stats_text += f'Successful: {len(successful_rewards)} ({100*len(successful_rewards)/len(episode_rewards):.1f}%)\n'
stats_text += f'Failed: {len(failed_rewards)} ({100*len(failed_rewards)/len(episode_rewards):.1f}%)\n'
stats_text += f'Std Dev: {std_reward:.2f}\n'
stats_text += f'Best: {episode_rewards.max():.2f}\n'
stats_text += f'Worst: {worst_reward:.2f}'

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes,
       ha='right', va='top', fontsize=9,
       fontfamily='monospace',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1',
                edgecolor='#2C3E50', linewidth=2, alpha=0.9))

# Save figure
plt.tight_layout()
plt.savefig('../outputs/figure_4_1_reward_distribution.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved to outputs/figure_4_1_reward_distribution.png")
print(f"Mean reward: {mean_reward:.4f}")
print(f"Successful episodes: {len(successful_rewards)}/{len(episode_rewards)} ({100*len(successful_rewards)/len(episode_rewards):.1f}%)")
plt.close()
