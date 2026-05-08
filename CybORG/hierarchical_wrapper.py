"""
Hierarchical RL Wrapper
=======================
Implements goal-conditioned hierarchical RL with meta-controller and worker policy.

Meta-Controller: Selects high-level goals based on network state
Worker Policy: Executes actions conditioned on current goal

Goals:
- MONITOR (0): Passive observation when network clean
- DEFEND_ENTERPRISE (1): Focus on Enterprise servers
- RAPID_RESTORE (2): Fast Restore cycling when re-compromise detected
- DEFEND_OPSERVER (3): Protect Op_Server0
- DEFEND_USERS (4): Clean User hosts
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces as gym_spaces


# Goal definitions
GOAL_MONITOR = 0
GOAL_DEFEND_ENTERPRISE = 1
GOAL_RAPID_RESTORE = 2
GOAL_DEFEND_OPSERVER = 3
GOAL_DEFEND_USERS = 4
N_GOALS = 5

# Host indices
_HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]
_ENTERPRISE_INDICES = [1, 2, 3]
_OPSERVER_INDEX = 7
_USER_INDICES = [8, 9, 10, 11, 12]

# Action indices for Restore in reduced space (30 actions)
_RESTORE_ACTIONS = {
    1: 21,  # Enterprise0
    2: 22,  # Enterprise1
    3: 23,  # Enterprise2
    7: 24,  # Op_Server0
    8: 25,  # User0
    9: 26,  # User1
    10: 27, # User2
    11: 28, # User3
    12: 29, # User4
}


def _host_is_compromised(obs: np.ndarray, host_idx: int) -> bool:
    """Check if host has compromise bits set."""
    start = host_idx * 4
    comp0 = obs[start + 2]
    comp1 = obs[start + 3]
    return bool(comp0 or comp1)


class MetaController:
    """
    Rule-based meta-controller that selects goals based on network state.

    Key logic for solving re-compromise cycle:
    - Tracks steps since last Restore on each host
    - If host compromised AND Restore used <5 steps ago → RAPID_RESTORE mode
    - In RAPID_RESTORE mode, agent learns to immediately Restore again
    """

    def __init__(self):
        self.steps_since_restore = {}  # host_idx → steps since last Restore
        self.restore_history = {}      # host_idx → list of steps when Restored

    def reset(self):
        """Reset tracking at episode start."""
        self.steps_since_restore = {}
        self.restore_history = {}

    def select_goal(self, obs: np.ndarray, last_action: int, current_step: int) -> int:
        """
        Select goal based on current network state.

        Priority:
        1. RAPID_RESTORE if re-compromise cycle detected
        2. DEFEND_OPSERVER if Op_Server0 compromised
        3. DEFEND_ENTERPRISE if any Enterprise compromised
        4. DEFEND_USERS if any User compromised
        5. MONITOR otherwise
        """
        base_obs = obs[:52] if len(obs) >= 52 else obs

        # Update Restore tracking
        if last_action is not None and last_action in _RESTORE_ACTIONS.values():
            # Find which host was Restored
            for host_idx, action_idx in _RESTORE_ACTIONS.items():
                if last_action == action_idx:
                    self.steps_since_restore[host_idx] = 0
                    if host_idx not in self.restore_history:
                        self.restore_history[host_idx] = []
                    self.restore_history[host_idx].append(current_step)
                    break

        # Increment step counters
        for host_idx in list(self.steps_since_restore.keys()):
            self.steps_since_restore[host_idx] += 1

        # Check for re-compromise cycle (critical for solving the problem)
        for ent_idx in _ENTERPRISE_INDICES:
            if _host_is_compromised(base_obs, ent_idx):
                steps_since = self.steps_since_restore.get(ent_idx, 999)
                # If Enterprise compromised and we Restored it <5 steps ago → re-compromise cycle!
                if steps_since < 5:
                    return GOAL_RAPID_RESTORE

        # Check Op_Server0 (highest priority)
        if _host_is_compromised(base_obs, _OPSERVER_INDEX):
            return GOAL_DEFEND_OPSERVER

        # Check Enterprise servers (high priority)
        for ent_idx in _ENTERPRISE_INDICES:
            if _host_is_compromised(base_obs, ent_idx):
                return GOAL_DEFEND_ENTERPRISE

        # Check User hosts (medium priority)
        for user_idx in _USER_INDICES:
            if _host_is_compromised(base_obs, user_idx):
                return GOAL_DEFEND_USERS

        # Network clean → Monitor mode
        return GOAL_MONITOR


class HierarchicalWrapper(gym.Wrapper):
    """
    Wraps environment to add goal-conditioned observations.

    Augments observation with:
    - Current goal (one-hot encoded, 5-dim)
    - Steps in current goal (1-dim, normalized)

    This allows the policy to learn goal-specific behaviors:
    - In RAPID_RESTORE mode: learn to immediately use Restore
    - In DEFEND_ENTERPRISE mode: learn normal defense patterns
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.meta_controller = MetaController()
        self.current_goal = GOAL_MONITOR
        self.steps_in_goal = 0
        self.current_step = 0
        self.last_action = None

        # Expand observation space to include goal information
        base_space = self.env.observation_space
        if isinstance(base_space, gym_spaces.Box):
            base_dim = base_space.shape[0]
            # Add: goal one-hot (5-dim) + steps_in_goal (1-dim) = +6 dims
            new_dim = base_dim + 6
            self.observation_space = gym_spaces.Box(
                low=0.0,
                high=1.0,
                shape=(new_dim,),
                dtype=np.float32
            )
        else:
            raise TypeError(f"HierarchicalWrapper requires Box observation space, got {type(base_space)}")

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Add goal information to observation."""
        # One-hot encode current goal
        goal_onehot = np.zeros(N_GOALS, dtype=np.float32)
        goal_onehot[self.current_goal] = 1.0

        # Normalize steps in goal (cap at 20 steps)
        steps_normalized = np.array([min(self.steps_in_goal / 20.0, 1.0)], dtype=np.float32)

        # Concatenate
        augmented = np.concatenate([obs, goal_onehot, steps_normalized])
        return augmented

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.meta_controller.reset()
        self.current_step = 0
        self.last_action = None
        self.current_goal = GOAL_MONITOR
        self.steps_in_goal = 0

        augmented_obs = self._augment_obs(obs)
        return augmented_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_step += 1
        self.last_action = action

        # Meta-controller selects new goal
        new_goal = self.meta_controller.select_goal(obs, self.last_action, self.current_step)

        # Track steps in current goal
        if new_goal != self.current_goal:
            self.current_goal = new_goal
            self.steps_in_goal = 0
        else:
            self.steps_in_goal += 1

        # Add goal-specific reward shaping (optional, helps learning)
        shaped_reward = 0.0

        # Reward for using Restore in RAPID_RESTORE mode
        if self.current_goal == GOAL_RAPID_RESTORE and action in _RESTORE_ACTIONS.values():
            shaped_reward += 0.3  # Encourage fast Restore response

        # Small penalty for staying in RAPID_RESTORE mode too long (encourages efficiency)
        if self.current_goal == GOAL_RAPID_RESTORE and self.steps_in_goal > 3:
            shaped_reward -= 0.1

        augmented_obs = self._augment_obs(obs)
        return augmented_obs, reward + shaped_reward, terminated, truncated, info
