"""
ActionMaskingWrapper
====================
Prevents wasteful actions by masking them based on current observation state.

Masked actions (not allowed):
- Restore on hosts that are already clean
- Remove on hosts that are not compromised

This prevents the agent from wasting -1.0 points on unnecessary Restore actions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Host index constants - must match alphabetical order used by BlueTableWrapper
_HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]
_N_HOSTS = len(_HOSTS)

# Action name patterns for the FULL 54-action space
# These are the indices in the full CybORG action space
_RESTORE_ACTIONS = list(range(42, 55))  # Restore actions for all hosts
_REMOVE_ACTIONS = list(range(16, 29))   # Remove actions for all hosts

# Mapping from action index to host index
def _get_host_for_action(action_idx: int) -> int:
    """Get the host index for a Restore or Remove action."""
    if 42 <= action_idx <= 54:  # Restore
        return action_idx - 42
    elif 16 <= action_idx <= 28:  # Remove
        return action_idx - 16
    return -1


def _host_is_clean(obs: np.ndarray, host_idx: int) -> bool:
    """True if all four bits for this host are zero (completely clean)."""
    start = host_idx * 4
    return not obs[start: start + 4].any()


def _host_is_compromised(obs: np.ndarray, host_idx: int) -> bool:
    """True if the host has compromise bits set (comp0 or comp1)."""
    start = host_idx * 4
    # bits layout per host: [act0, act1, comp0, comp1]
    comp0 = obs[start + 2]
    comp1 = obs[start + 3]
    return bool(comp0 or comp1)


class ActionMaskingWrapper(gym.Wrapper):
    """
    Masks wasteful actions based on observation state.

    Works with full 54-action space or any action space that includes
    Restore/Remove actions.

    Compatible with sb3-contrib's MaskablePPO via ActionMasker wrapper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Verify we have a Discrete action space
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("ActionMaskingWrapper only supports Discrete action spaces")

        self.n_actions = env.action_space.n
        self._last_obs = None

    def action_masks(self) -> np.ndarray:
        """Return current action mask. Called by sb3-contrib's ActionMasker."""
        if self._last_obs is None:
            return np.ones(self.n_actions, dtype=np.int8)
        return self._get_action_mask(self._last_obs)

    def _get_action_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Create action mask based on current observation.
        Returns binary array: 1 = allowed, 0 = masked (not allowed)
        """
        mask = np.ones(self.n_actions, dtype=np.int8)

        # Extract base observation (first 52 dims are host states)
        base_obs = obs[:52] if len(obs) >= 52 else obs

        # Mask wasteful Restore actions (restore on already-clean hosts)
        for action_idx in _RESTORE_ACTIONS:
            if action_idx >= self.n_actions:
                continue
            host_idx = _get_host_for_action(action_idx)
            if host_idx >= 0 and host_idx < _N_HOSTS:
                if _host_is_clean(base_obs, host_idx):
                    mask[action_idx] = 0  # Mask this action

        # Mask wasteful Remove actions (remove on non-compromised hosts)
        for action_idx in _REMOVE_ACTIONS:
            if action_idx >= self.n_actions:
                continue
            host_idx = _get_host_for_action(action_idx)
            if host_idx >= 0 and host_idx < _N_HOSTS:
                if not _host_is_compromised(base_obs, host_idx):
                    mask[action_idx] = 0  # Mask this action

        # Ensure at least one action is valid (fallback to Monitor action 0)
        if mask.sum() == 0:
            mask[0] = 1  # Monitor is always safe

        return mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info
