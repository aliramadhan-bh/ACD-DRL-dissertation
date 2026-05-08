"""
Generate Observation Space Augmentation Figure
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Figure setup
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.set_xlim(0, 120)
ax.set_ylim(0, 8)
ax.axis('off')

# Color scheme for different feature types
COLOR_ACTIVITY = '#F39C12'             # Orange
COLOR_COMPROMISED = '#E74C3C'          # Red
COLOR_DECOY = '#16A085'                # Teal
COLOR_STEP_COUNTER = '#27AE60'         # Green
COLOR_LAST_ACTIVITY = '#E67E22'        # Dark orange
COLOR_LAST_COMPROMISE = '#C0392B'      # Dark red
COLOR_CUMULATIVE = '#8E44AD'           # Dark purple

# Bar parameters
bar_height = 1.2
bar_y_positions = [5.5, 3.5, 1.5]  # Three stages from top to bottom

# Stage 1: Base observation (52-dim)
# 2 bits × 13 hosts = 26 per feature; Activity (26) + Compromised (26)
stage1_y = bar_y_positions[0]
stage1_features = [
    ('Activity\n(26)', 26, COLOR_ACTIVITY),
    ('Compromised\n(26)', 26, COLOR_COMPROMISED),
]

# Stage 2: After DecoyWrapper (65-dim)
# Base 52 + Decoy Status 13 = 65
stage2_y = bar_y_positions[1]
stage2_features = [
    ('Activity\n(26)', 26, COLOR_ACTIVITY),
    ('Compromised\n(26)', 26, COLOR_COMPROMISED),
    ('Decoy Status\n(13)', 13, COLOR_DECOY),
]

# Stage 3: After ObsEnhancedWrapper (105-dim)
# Base 52 + Decoy 13 + Step Counter 1 + Steps Since Last Activity 13
# + Steps Since Last Compromise 13 + Cumulative Compromise Count 13 = 105
stage3_y = bar_y_positions[2]
stage3_features = [
    ('Activity\n(26)', 26, COLOR_ACTIVITY),
    ('Compromised\n(26)', 26, COLOR_COMPROMISED),
    ('Decoy Status\n(13)', 13, COLOR_DECOY),
    ('Step\nCounter\n(1)', 1, COLOR_STEP_COUNTER),
    ('Steps Since\nLast Activity\n(13)', 13, COLOR_LAST_ACTIVITY),
    ('Steps Since\nLast Compromise\n(13)', 13, COLOR_LAST_COMPROMISE),
    ('Cumulative\nCompromise\n(13)', 13, COLOR_CUMULATIVE),
]


def draw_observation_bar(y_pos, features, stage_label, total_dim):
    """Draw a single observation bar with segmented features."""
    x_offset = 5
    segment_width_scale = 1.0
    min_width = 3.5  # Minimum display width for very small segments (e.g. Step Counter)

    current_x = x_offset
    for label, dims, color in features:
        width = max(dims * segment_width_scale,
                    min_width if dims < 5 else dims * segment_width_scale)

        rect = Rectangle((current_x, y_pos), width, bar_height,
                         linewidth=1.5, edgecolor='#2C3E50',
                         facecolor=color, alpha=0.85)
        ax.add_patch(rect)

        ax.text(current_x + width / 2, y_pos + bar_height / 2,
                label, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

        current_x += width

    # Stage label on left
    ax.text(2, y_pos + bar_height / 2, stage_label,
            ha='right', va='center', fontsize=11,
            fontweight='bold', color='#2C3E50')

    # Total dimension label on right
    ax.text(current_x + 2, y_pos + bar_height / 2, f'{total_dim}-dim',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1',
                      edgecolor='#2C3E50', linewidth=2))


# Draw all three stages
draw_observation_bar(stage1_y, stage1_features, 'Base\nObservation', 52)
draw_observation_bar(stage2_y, stage2_features, 'After\nDecoyWrapper', 65)
draw_observation_bar(stage3_y, stage3_features, 'After\nObsEnhanced\nWrapper', 105)

# Add arrows showing progression
arrow_props = dict(arrowstyle='->', lw=2.5, color='#7F8C8D')

# Stage 1 -> Stage 2
ax.annotate('', xy=(3, stage2_y + bar_height + 0.1),
            xytext=(3, stage1_y - 0.1),
            arrowprops=arrow_props)
ax.text(3.5, (stage1_y + stage2_y + bar_height) / 2, '+13 dims\n(decoy flags)',
        ha='left', va='center', fontsize=9, color='#16A085',
        fontweight='bold', style='italic')

# Stage 2 -> Stage 3
ax.annotate('', xy=(3, stage3_y + bar_height + 0.1),
            xytext=(3, stage2_y - 0.1),
            arrowprops=arrow_props)
ax.text(3.5, (stage2_y + stage3_y + bar_height) / 2, '+40 dims\n(temporal features)',
        ha='left', va='center', fontsize=9, color='#27AE60',
        fontweight='bold', style='italic')

# Title
ax.text(60, 7.5, 'Observation Space Augmentation Across Wrapper Stack',
        ha='center', va='top', fontsize=14, fontweight='bold',
        color='#2C3E50')

# Legend
legend_elements = [
    patches.Patch(facecolor=COLOR_ACTIVITY, edgecolor='#2C3E50',
                  label='Activity — None/Scan/Exploit per host (2 bits × 13)', linewidth=1),
    patches.Patch(facecolor=COLOR_COMPROMISED, edgecolor='#2C3E50',
                  label='Compromised — No/Unknown/User/Privileged per host (2 bits × 13)', linewidth=1),
    patches.Patch(facecolor=COLOR_DECOY, edgecolor='#2C3E50',
                  label='Decoy Status (per host)', linewidth=1),
    patches.Patch(facecolor=COLOR_STEP_COUNTER, edgecolor='#2C3E50',
                  label='Step Counter (global)', linewidth=1),
    patches.Patch(facecolor=COLOR_LAST_ACTIVITY, edgecolor='#2C3E50',
                  label='Steps Since Last Activity (per host)', linewidth=1),
    patches.Patch(facecolor=COLOR_LAST_COMPROMISE, edgecolor='#2C3E50',
                  label='Steps Since Last Compromise (per host)', linewidth=1),
    patches.Patch(facecolor=COLOR_CUMULATIVE, edgecolor='#2C3E50',
                  label='Cumulative Compromise Count (per host)', linewidth=1),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3,
          fontsize=8, frameon=True, fancybox=True, shadow=True,
          bbox_to_anchor=(0.5, -0.08))

# Save figure
plt.tight_layout()
plt.savefig('../outputs/figure_3_3_observation_space.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved to outputs/figure_3_3_observation_space.png")
plt.close()
