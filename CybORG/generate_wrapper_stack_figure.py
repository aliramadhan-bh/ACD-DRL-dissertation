"""
Generate Gymnasium Wrapper Stack Architecture Figure
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle

# Figure setup
fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
COLOR_CYBORG = '#F39C12'       # Amber for CybORG
COLOR_GYMCOMPAT = '#95A5A6'    # Gray for GymCompatChallengeEnv
COLOR_WRAPPER = '#16A085'      # Teal for wrappers
COLOR_AGENT = '#8E44AD'        # Purple for PPO Agent

# Layer specifications (name, color, y_position, height)
layers = [
    ('CybORG Environment', COLOR_CYBORG, 1.0, 1.2),
    ('GymCompatChallengeEnv', COLOR_GYMCOMPAT, 2.5, 1.2),
    ('DecoyWrapper', COLOR_WRAPPER, 4.0, 1.2),
    ('ReducedActionWrapper', COLOR_WRAPPER, 5.5, 1.2),
    ('ObsEnhancedWrapper', COLOR_WRAPPER, 7.0, 1.2),
    ('PPO Agent', COLOR_AGENT, 8.5, 1.2),
]

# Box dimensions
box_x = 2.0
box_width = 6.0

# Draw layers
for name, color, y_pos, height in layers:
    rect = Rectangle((box_x, y_pos), box_width, height,
                     linewidth=2.5, edgecolor='#2C3E50',
                     facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    # Layer name
    ax.text(box_x + box_width/2, y_pos + height/2,
            name, ha='center', va='center', fontsize=12,
            fontweight='bold', color='white')

# Interface specifications (obs_dim, action_dim, y_position)
# Positions are between layers
interfaces = [
    ('52-dim obs', '54 actions', 2.25),      # CybORG → GymCompat
    ('52-dim obs', '54 actions', 3.75),      # GymCompat → DecoyWrapper
    ('65-dim obs', '54 actions', 5.25),      # DecoyWrapper → ReducedActionWrapper
    ('65-dim obs', '30 actions', 6.75),      # ReducedActionWrapper → ObsEnhancedWrapper
    ('105-dim obs', '30 actions', 8.25),     # ObsEnhancedWrapper → PPO Agent
]

# Draw arrows and labels
arrow_props = dict(arrowstyle='->', lw=2, color='#34495E')

for obs_label, action_label, y_pos in interfaces:
    # Upward arrow (observations going up)
    arrow_up = FancyArrowPatch((box_x + 1.5, y_pos - 0.15),
                               (box_x + 1.5, y_pos + 0.15),
                               **arrow_props)
    ax.add_patch(arrow_up)

    # Downward arrow (actions coming down)
    arrow_down = FancyArrowPatch((box_x + box_width - 1.5, y_pos + 0.15),
                                 (box_x + box_width - 1.5, y_pos - 0.15),
                                 **arrow_props)
    ax.add_patch(arrow_down)

    # Labels
    # Observations (left side, upward)
    ax.text(box_x + 1.5, y_pos + 0.35, obs_label,
            ha='center', va='bottom', fontsize=9,
            color='#2C3E50', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#34495E', linewidth=1))

    # Actions (right side, downward)
    ax.text(box_x + box_width - 1.5, y_pos - 0.35, action_label,
            ha='center', va='top', fontsize=9,
            color='#2C3E50', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#34495E', linewidth=1))

# Add flow direction labels
ax.text(box_x + 1.5, 10.2, 'Observations ↑',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#2C3E50',
        style='italic')
ax.text(box_x + box_width - 1.5, 10.2, 'Actions ↓',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#2C3E50',
        style='italic')

# Title
ax.text(5, 11.5, 'Gymnasium Wrapper Stack Architecture',
        ha='center', va='top', fontsize=14, fontweight='bold',
        color='#2C3E50')

# Add annotations for key transformations
annotations = [
    (8.5, 5.25, '+13 decoy flags', 'left'),    # DecoyWrapper adds 13 dims
    (8.5, 6.75, '54 → 30 actions', 'left'),    # ReducedActionWrapper reduces actions
    (8.5, 8.25, '+40 temporal features', 'left'),  # ObsEnhancedWrapper adds 40 dims
]

for x, y, text, alignment in annotations:
    ha = 'left' if alignment == 'left' else 'right'
    ax.annotate(text, xy=(box_x + box_width + 0.1, y),
                xytext=(x, y),
                ha=ha, va='center', fontsize=9,
                color='#E74C3C', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

# Legend
legend_elements = [
    patches.Patch(facecolor=COLOR_CYBORG, edgecolor='#2C3E50',
                 label='CybORG Base Environment', linewidth=2),
    patches.Patch(facecolor=COLOR_GYMCOMPAT, edgecolor='#2C3E50',
                 label='Gymnasium Compatibility Layer', linewidth=2),
    patches.Patch(facecolor=COLOR_WRAPPER, edgecolor='#2C3E50',
                 label='Custom Wrappers', linewidth=2),
    patches.Patch(facecolor=COLOR_AGENT, edgecolor='#2C3E50',
                 label='RL Agent', linewidth=2),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=2,
         fontsize=9, frameon=True, fancybox=True, shadow=True,
         bbox_to_anchor=(0.5, -0.05))

# Save figure
plt.tight_layout()
plt.savefig('../outputs/figure_3_2_wrapper_stack.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved to outputs/figure_3_2_wrapper_stack.png")
plt.close()
