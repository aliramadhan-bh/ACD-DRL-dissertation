"""
Generate CAGE Challenge 2 Scenario1b Network Topology Figure
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle

# Figure setup
fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Color scheme
COLOR_RED_ENTRY = '#E74C3C'      # Red for User0 (entry point)
COLOR_USER = '#E67E22'            # Orange for User1-4
COLOR_ENTERPRISE = '#9B59B6'      # Purple for Enterprise0-2
COLOR_DEFENDER = '#27AE60'        # Green for Defender
COLOR_OPERATIONAL = '#16A085'     # Teal for operational hosts
COLOR_CRITICAL = '#1ABC9C'        # Bright teal for Op_Server0

# Subnet positions (x, y, width, height)
subnet_user = (0.5, 1.0, 3.5, 7)
subnet_enterprise = (5, 1.0, 4, 7)
subnet_operational = (10, 1.0, 3.5, 7)

# Draw subnet boxes
for subnet_pos, subnet_name in [(subnet_user, 'User Subnet'),
                                 (subnet_enterprise, 'Enterprise Subnet'),
                                 (subnet_operational, 'Operational Subnet')]:
    rect = Rectangle((subnet_pos[0], subnet_pos[1]), subnet_pos[2], subnet_pos[3],
                     linewidth=2, linestyle='--', edgecolor='#34495E',
                     facecolor='none', alpha=0.7)
    ax.add_patch(rect)
    # Subnet label
    ax.text(subnet_pos[0] + subnet_pos[2]/2, subnet_pos[1] + subnet_pos[3] + 0.2,
            subnet_name, ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='#2C3E50')

# Host box dimensions
host_width = 2.2
host_height = 0.6

# User Subnet hosts
user_hosts = [
    ('User0', '0.0/step\n(Red Entry)', 1.15, 7.0, COLOR_RED_ENTRY),
    ('User1', '−0.1/step', 1.15, 6.0, COLOR_USER),
    ('User2', '−0.1/step', 1.15, 5.0, COLOR_USER),
    ('User3', '−0.1/step', 1.15, 4.0, COLOR_USER),
    ('User4', '−0.1/step', 1.15, 3.0, COLOR_USER),
]

# Enterprise Subnet hosts
enterprise_hosts = [
    ('Enterprise0', '−1.0/step', 5.9, 7.0, COLOR_ENTERPRISE),
    ('Enterprise1', '−1.0/step', 5.9, 6.0, COLOR_ENTERPRISE),
    ('Enterprise2', '−1.0/step', 5.9, 5.0, COLOR_ENTERPRISE),
    ('Defender', 'Blue Agent\nHost', 5.9, 3.8, COLOR_DEFENDER),
]

# Operational Subnet hosts
operational_hosts = [
    ('Op_Server0', '−1.0 + −10.0/step\n(Critical)', 10.15, 7.0, COLOR_CRITICAL),
    ('Op_Host0', '−0.1/step', 10.15, 5.5, COLOR_OPERATIONAL),
    ('Op_Host1', '−0.1/step', 10.15, 4.5, COLOR_OPERATIONAL),
    ('Op_Host2', '−0.1/step', 10.15, 3.5, COLOR_OPERATIONAL),
]

# Draw all hosts
def draw_host(name, label, x, y, color):
    rect = Rectangle((x, y), host_width, host_height,
                     linewidth=1.5, edgecolor='#2C3E50',
                     facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    # Host name
    ax.text(x + host_width/2, y + host_height/2 + 0.15,
            name, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    # Penalty label
    ax.text(x + host_width/2, y + host_height/2 - 0.15,
            label, ha='center', va='center', fontsize=7,
            color='white', style='italic')

for host in user_hosts:
    draw_host(*host)

for host in enterprise_hosts:
    draw_host(*host)

for host in operational_hosts:
    draw_host(*host)

# Draw bidirectional arrows between subnets
def draw_subnet_arrow(x1, y1, x2, y2, label=''):
    # Forward arrow
    arrow1 = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#7F8C8D', alpha=0.6)
    ax.add_patch(arrow1)
    # Backward arrow (offset slightly)
    arrow2 = FancyArrowPatch((x2, y2 - 0.15), (x1, y1 - 0.15),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#7F8C8D', alpha=0.6)
    ax.add_patch(arrow2)

# User <-> Enterprise
draw_subnet_arrow(subnet_user[0] + subnet_user[2], 5.0,
                 subnet_enterprise[0], 5.0)

# Enterprise <-> Operational
draw_subnet_arrow(subnet_enterprise[0] + subnet_enterprise[2], 5.0,
                 subnet_operational[0], 5.0)

# Title
ax.text(7, 8.7, 'CAGE Challenge 2 — Scenario1b Network Topology',
        ha='center', va='top', fontsize=14, fontweight='bold',
        color='#2C3E50')

# Legend
legend_elements = [
    patches.Patch(facecolor=COLOR_RED_ENTRY, edgecolor='#2C3E50', label='Red Entry Point (0.0/step)'),
    patches.Patch(facecolor=COLOR_USER, edgecolor='#2C3E50', label='User Hosts (−0.1/step)'),
    patches.Patch(facecolor=COLOR_DEFENDER, edgecolor='#2C3E50', label='Blue Agent Host'),
    patches.Patch(facecolor=COLOR_ENTERPRISE, edgecolor='#2C3E50', label='Enterprise Hosts (−1.0/step)'),
    patches.Patch(facecolor=COLOR_CRITICAL, edgecolor='#2C3E50', label='Critical Asset (−11.0/step total)'),
    patches.Patch(facecolor=COLOR_OPERATIONAL, edgecolor='#2C3E50', label='Operational Hosts (−0.1/step)'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3,
         fontsize=9, frameon=True, fancybox=True, shadow=True,
         bbox_to_anchor=(0.5, -0.05))

# Save figure
plt.tight_layout()
plt.savefig('../outputs/figure_3_1_network_topology.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved to outputs/figure_3_1_network_topology.png")
plt.close()
