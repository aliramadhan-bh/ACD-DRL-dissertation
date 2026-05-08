import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(-0.5, 6.4)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── palette ───────────────────────────────────────────────────────────────────
AGENT_EDGE = '#6A3D9A'
AGENT_FACE = '#EDE7F6'
ENV_EDGE   = '#1A7A6E'
ENV_FACE   = '#E0F2F1'
C_ACTION   = '#C0392B'
C_STATE    = '#1565C0'
C_REWARD   = '#E67E22'

# ── box geometry ──────────────────────────────────────────────────────────────
BOX_W, BOX_H = 3.2, 2.2
AGENT_X = 1.2
ENV_X   = 8.6
BOX_Y   = 2.4          # bottom of boxes
BOX_CY  = BOX_Y + BOX_H / 2
AGENT_CX = AGENT_X + BOX_W / 2   # 2.8
ENV_CX   = ENV_X   + BOX_W / 2   # 10.2

# ── boxes ─────────────────────────────────────────────────────────────────────
for x, fc, ec, title, sub in [
    (AGENT_X, AGENT_FACE, AGENT_EDGE, 'Agent',       r'Policy  $\pi(a \mid s)$'),
    (ENV_X,   ENV_FACE,   ENV_EDGE,   'Environment', r'Transition  $P(s^{\prime} \mid s,a)$'),
]:
    ax.add_patch(FancyBboxPatch((x, BOX_Y), BOX_W, BOX_H,
                                boxstyle='round,pad=0.12',
                                facecolor=fc, edgecolor=ec, linewidth=2.8, zorder=2))
    ax.text(x + BOX_W/2, BOX_Y + BOX_H*0.70, title,
            ha='center', va='center', fontsize=17, fontweight='bold',
            color=ec, zorder=3)
    ax.text(x + BOX_W/2, BOX_Y + BOX_H*0.28, sub,
            ha='center', va='center', fontsize=11, color='#444', style='italic', zorder=3)

# ── helper: straight annotate arrow ──────────────────────────────────────────
def arrow(x0, x1, y, col, lw=2.2):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                mutation_scale=20,
                                connectionstyle='arc3,rad=0'),
                zorder=4)

# gap between box edges
GAP = 0.12

# ── 1. Action  At  →  Agent to Environment  (top straight arrow) ──────────
A_Y = BOX_CY + 0.50
arrow(AGENT_X + BOX_W + GAP, ENV_X - GAP, A_Y, C_ACTION)
ax.text((AGENT_X + BOX_W + ENV_X)/2, A_Y + 0.22,
        r'Action  $A_t$',
        ha='center', va='bottom', fontsize=13, color=C_ACTION, fontweight='bold', zorder=5)

# ── 2. State  St+1  →  Environment to Agent  (lower straight arrow) ──────
S_Y = BOX_CY - 0.50
arrow(ENV_X - GAP, AGENT_X + BOX_W + GAP, S_Y, C_STATE)
ax.text((AGENT_X + BOX_W + ENV_X)/2, S_Y + 0.14,
        r'State  $S_{t+1}$',
        ha='center', va='bottom', fontsize=13, color=C_STATE, fontweight='bold', zorder=5)

# ── 3. Reward  Rt+1  →  smooth U-arc below both boxes ────────────────────────
# Use a cubic bezier so the arc departs and arrives vertically (clean U-shape)
# endpoints anchored just below the box bottoms
ARC_ATTACH_Y = BOX_Y - 0.10   # where arc meets the box bottom
CTRL_Y       = BOX_Y - 1.70   # bezier control point (how deep the U goes)

codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
verts = [
    (ENV_CX,   ARC_ATTACH_Y),  # start  (env bottom-centre)
    (ENV_CX,   CTRL_Y),         # ctrl 1 (straight down from env)
    (AGENT_CX, CTRL_Y),         # ctrl 2 (straight down from agent)
    (AGENT_CX, ARC_ATTACH_Y),  # end    (agent bottom-centre)
]

# draw the curved path
ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                                facecolor='none', edgecolor=C_REWARD,
                                lw=2.2, zorder=4))

# arrowhead at end: tiny upward annotation that inherits the colour
ax.annotate('',
            xy=(AGENT_CX, ARC_ATTACH_Y + 0.02),
            xytext=(AGENT_CX, ARC_ATTACH_Y - 0.40),
            arrowprops=dict(arrowstyle='->', color=C_REWARD, lw=2.2,
                            mutation_scale=20,
                            connectionstyle='arc3,rad=0'),
            zorder=5)

# label below the midpoint of the U-arc
MID_X = (ENV_CX + AGENT_CX) / 2
MID_Y = CTRL_Y + (CTRL_Y - ARC_ATTACH_Y) * 0.10   # slightly below ctrl
ax.text(MID_X, CTRL_Y - 0.12,
        r'Reward  $R_{t+1}$',
        ha='center', va='top', fontsize=13, color=C_REWARD, fontweight='bold', zorder=5)

# vertical connector lines from box bottom down to arc attachment
for cx in (AGENT_CX, ENV_CX):
    ax.plot([cx, cx], [BOX_Y, ARC_ATTACH_Y], color=C_REWARD, lw=2.2, zorder=3)

# ── timestep annotations above boxes ─────────────────────────────────────────
for cx, badge, note in [
    (AGENT_CX,
     'time step  $t$',
     'Observes $S_t$, selects $A_t$\nvia policy $\\pi$'),
    (ENV_CX,
     'time step  $t{+}1$',
     'Returns $S_{t+1}$ and $R_{t+1}$\nvia dynamics $P$'),
]:
    ax.text(cx, BOX_Y + BOX_H + 0.85, badge,
            ha='center', va='bottom', fontsize=10.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8F8F8',
                      edgecolor='#CCCCCC', linewidth=1.2),
            zorder=5)
    ax.text(cx, BOX_Y + BOX_H + 0.28, note,
            ha='center', va='bottom', fontsize=9, color='#555',
            linespacing=1.55, zorder=5)

# ── title ─────────────────────────────────────────────────────────────────────
ax.set_title('Reinforcement Learning — Agent-Environment Interaction',
             fontsize=15.5, fontweight='bold', color='#1a1a1a', pad=10)

# ── legend ────────────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(facecolor=C_ACTION, label='Action  $A_t$'),
    mpatches.Patch(facecolor=C_STATE,  label='State  $S_{t+1}$'),
    mpatches.Patch(facecolor=C_REWARD, label='Reward  $R_{t+1}$'),
], loc='lower center', ncol=3, fontsize=10.5, framealpha=0.9,
   bbox_to_anchor=(0.5, 0.01), handlelength=1.2, handleheight=0.9)

plt.tight_layout(pad=1.0)
plt.savefig('outputs/figure_2_1_rl_loop.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: outputs/figure_2_1_rl_loop.png")
