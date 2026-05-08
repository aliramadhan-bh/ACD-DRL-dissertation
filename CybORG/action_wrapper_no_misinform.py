"""
ReducedActionWrapper (No Misinform)
===================================
Gymnasium wrapper that exposes a curated subset of the full 54-action CybORG
Scenario1b Blue action space, with ALL Misinform actions REMOVED.

Based on diagnostic findings showing agents waste 65%+ actions on Misinform
without measurable defensive benefit.

Full Scenario1b action mapping (from EnumActionWrapper):
  0  Sleep              15 Remove Defender     28 Misinform Defender   41 Restore Defender
  1  Monitor            16 Remove Enterprise0  29 Misinform Enterprise0 42 Restore Enterprise0
  2  Analyse Defender   17 Remove Enterprise1  30 Misinform Enterprise1 43 Restore Enterprise1
  3  Analyse Enterprise0 18 Remove Enterprise2 31 Misinform Enterprise2 44 Restore Enterprise2
  4  Analyse Enterprise1 19 Remove Op_Host0   32 Misinform Op_Host0   45 Restore Op_Host0
  5  Analyse Enterprise2 20 Remove Op_Host1   33 Misinform Op_Host1   46 Restore Op_Host1
  6  Analyse Op_Host0   21 Remove Op_Host2    34 Misinform Op_Host2   47 Restore Op_Host2
  7  Analyse Op_Host1   22 Remove Op_Server0  35 Misinform Op_Server0 48 Restore Op_Server0
  8  Analyse Op_Host2   23 Remove User0       36 Misinform User0      49 Restore User0
  9  Analyse Op_Server0 24 Remove User1       37 Misinform User1      50 Restore User1
  10 Analyse User0      25 Remove User2       38 Misinform User2      51 Restore User2
  11 Analyse User1      26 Remove User3       39 Misinform User3      52 Restore User3
  12 Analyse User2      27 Remove User4       40 Misinform User4      53 Restore User4
  13 Analyse User3
  14 Analyse User4

Reduced set rationale (21 core defensive actions):
- Monitor: Essential passive detection (1 action)
- Analyse: Investigation on attack-path and entry hosts (9 actions)
  - Enterprise0-2 (gateway to operational subnet)
  - Op_Server0 (critical target)
  - User0-4 (entry points)
- Remove: Malware cleanup on attack path (4 actions)
  - Enterprise0-2, Op_Server0
- Restore: Full recovery on attack path and entry points (10 actions)
  - Enterprise0-2, Op_Server0, User0-4
- Sleep: Excluded (agent should always act)
- Misinform: REMOVED (diagnostics show 65% action waste with no defensive benefit)
"""

import gymnasium as gym
from gymnasium import spaces

# Reduced action set - 21 actions, NO Misinform
REDUCED_TO_FULL = [
    1,   # 0  -> Monitor
    3,   # 1  -> Analyse Enterprise0
    4,   # 2  -> Analyse Enterprise1
    5,   # 3  -> Analyse Enterprise2
    9,   # 4  -> Analyse Op_Server0
    10,  # 5  -> Analyse User0
    11,  # 6  -> Analyse User1
    12,  # 7  -> Analyse User2
    13,  # 8  -> Analyse User3
    14,  # 9  -> Analyse User4
    16,  # 10 -> Remove Enterprise0
    17,  # 11 -> Remove Enterprise1
    18,  # 12 -> Remove Enterprise2
    22,  # 13 -> Remove Op_Server0
    42,  # 14 -> Restore Enterprise0
    43,  # 15 -> Restore Enterprise1
    44,  # 16 -> Restore Enterprise2
    48,  # 17 -> Restore Op_Server0
    49,  # 18 -> Restore User0
    50,  # 19 -> Restore User1
    51,  # 20 -> Restore User2
    52,  # 21 -> Restore User3
    53,  # 22 -> Restore User4
]

N_REDUCED = len(REDUCED_TO_FULL)

FULL_TO_REDUCED = {full: reduced for reduced, full in enumerate(REDUCED_TO_FULL)}


class ReducedActionWrapperNoMisinform(gym.Wrapper):
    """Maps a curated 21-action defensive set to the full CybORG action indices.

    Excludes all Misinform (decoy) actions based on diagnostic evidence showing
    they consume 65% of agent actions without measurable security benefit.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(N_REDUCED)

    def step(self, action):
        full_action = REDUCED_TO_FULL[int(action)]
        return self.env.step(full_action)
