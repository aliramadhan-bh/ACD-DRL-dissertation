import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis('off')

# --- Step definitions ---
steps = [
    {
        'y': 11.2,
        'label': '1.  Red Compromises Host',
        'sub': 'Penalty begins: −1.0 / step',
        'color': '#c0392b',
        'text_color': 'white',
    },
    {
        'y': 8.8,
        'label': '2.  Blue Detects Compromise',
        'sub': '2–3 step observation delay',
        'color': '#2980b9',
        'text_color': 'white',
    },
    {
        'y': 6.4,
        'label': '3.  Blue Executes Restore',
        'sub': 'Host clean — costs −1.0 per action',
        'color': '#1a7a6e',
        'text_color': 'white',
    },
    {
        'y': 4.0,
        'label': '4.  Red Re-Exploits Host',
        'sub': 'Same vulnerability — no patch available',
        'color': '#c0392b',
        'text_color': 'white',
    },
    {
        'y': 1.6,
        'label': '5.  Cycle Repeats',
        'sub': '3–4 times per episode',
        'color': '#d4801a',
        'text_color': 'white',
    },
]

box_w = 5.0
box_h = 0.95
box_x = 2.5  # left edge of box

# --- Draw boxes ---
for s in steps:
    cx = box_x + box_w / 2
    cy = s['y']

    rect = FancyBboxPatch(
        (box_x, cy - box_h / 2), box_w, box_h,
        boxstyle='round,pad=0.08',
        facecolor=s['color'], edgecolor='black', linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rect)

    ax.text(cx, cy + 0.16, s['label'], ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=s['text_color'], zorder=4)
    ax.text(cx, cy - 0.22, s['sub'], ha='center', va='center',
            fontsize=9.5, color=s['text_color'], style='italic', zorder=4)

# --- Downward arrows between boxes ---
arrow_kw = dict(arrowstyle='->', color='#333333', lw=1.8,
                mutation_scale=18, zorder=2)
for i in range(len(steps) - 1):
    y_top = steps[i]['y'] - box_h / 2
    y_bot = steps[i + 1]['y'] + box_h / 2
    mid_x = box_x + box_w / 2
    ax.annotate('', xy=(mid_x, y_bot + 0.04), xytext=(mid_x, y_top - 0.04),
                arrowprops=arrow_kw)

# --- Dashed return loop arrow (left side, step 5 → step 1) ---
loop_x = 1.55          # x of vertical spine
top_y  = steps[0]['y']   # step 1 centre y
bot_y  = steps[-1]['y']  # step 5 centre y

# Horizontal leg from step 5 left edge to loop spine
ax.annotate('', xy=(loop_x, bot_y), xytext=(box_x, bot_y),
            arrowprops=dict(arrowstyle='-', color='#555555', lw=1.8,
                            linestyle='dashed', zorder=2))
# Vertical leg up
ax.annotate('', xy=(loop_x, top_y), xytext=(loop_x, bot_y),
            arrowprops=dict(arrowstyle='-', color='#555555', lw=1.8,
                            linestyle='dashed', zorder=2))
# Horizontal leg into step 1 with arrowhead
ax.annotate('', xy=(box_x, top_y), xytext=(loop_x, top_y),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8,
                            linestyle='dashed', mutation_scale=16, zorder=2))

# Loop label
mid_loop_y = (top_y + bot_y) / 2
ax.text(loop_x - 0.18, mid_loop_y, 'loop', ha='right', va='center',
        fontsize=9, color='#555555', rotation=90, style='italic')

# --- Annotation box: step 1 (right side) ---
ann1_x = 7.85
ann1_y = steps[0]['y']
ann1_box = FancyBboxPatch(
    (ann1_x - 1.1, ann1_y - 0.55), 2.2, 1.1,
    boxstyle='round,pad=0.07',
    facecolor='#fadbd8', edgecolor='#c0392b', linewidth=1.2, zorder=3,
)
ax.add_patch(ann1_box)
ax.text(ann1_x, ann1_y + 0.18, 'Penalty Accumulation', ha='center', va='center',
        fontsize=8.5, fontweight='bold', color='#922b21', zorder=4)
ax.text(ann1_x, ann1_y - 0.18, '≈ 5–8 pts per cycle', ha='center', va='center',
        fontsize=8.5, color='#922b21', zorder=4)
# Connector
ax.annotate('', xy=(ann1_x - 1.1, ann1_y),
            xytext=(box_x + box_w, ann1_y),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2,
                            linestyle='dotted', mutation_scale=13, zorder=2))

# --- Annotation box: step 3 (right side) ---
ann3_x = 7.85
ann3_y = steps[2]['y']
ann3_box = FancyBboxPatch(
    (ann3_x - 1.1, ann3_y - 0.6), 2.2, 1.2,
    boxstyle='round,pad=0.07',
    facecolor='#d5f5e3', edgecolor='#1a7a6e', linewidth=1.2, zorder=3,
)
ax.add_patch(ann3_box)
ax.text(ann3_x, ann3_y + 0.22, 'No-Patch Constraint', ha='center', va='center',
        fontsize=8.5, fontweight='bold', color='#1a7a6e', zorder=4)
ax.text(ann3_x, ann3_y - 0.05, 'CybORG provides', ha='center', va='center',
        fontsize=8, color='#1a7a6e', zorder=4)
ax.text(ann3_x, ann3_y - 0.28, 'no patching action', ha='center', va='center',
        fontsize=8, color='#1a7a6e', zorder=4)
# Connector
ax.annotate('', xy=(ann3_x - 1.1, ann3_y),
            xytext=(box_x + box_w, ann3_y),
            arrowprops=dict(arrowstyle='->', color='#1a7a6e', lw=1.2,
                            linestyle='dotted', mutation_scale=13, zorder=2))

# --- Title ---
ax.text(5.0, 12.6,
        'Re-Compromise Cycle\nPrimary Performance-Limiting Mechanism',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#1a1a1a',
        linespacing=1.4)

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor='#c0392b', edgecolor='black', label='Red Team action'),
    mpatches.Patch(facecolor='#2980b9', edgecolor='black', label='Blue Team action'),
    mpatches.Patch(facecolor='#1a7a6e', edgecolor='black', label='Blue Team restore'),
    mpatches.Patch(facecolor='#d4801a', edgecolor='black', label='Cycle continuation'),
]
ax.legend(handles=legend_elements, loc='lower center',
          bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5,
          framealpha=0.9, edgecolor='#aaaaaa')

plt.tight_layout(pad=0.4)
plt.savefig('figure_4_3_recompromise_cycle.png', dpi=300, bbox_inches='tight')
print("Saved figure_4_3_recompromise_cycle.png")
