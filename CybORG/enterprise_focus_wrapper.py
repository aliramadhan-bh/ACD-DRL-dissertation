"""
EnterpriseFocusWrapper
======================
Adds targeted reward shaping to teach fast Enterprise server cleanup.

Key insight from episode traces:
- Enterprise2 stays compromised for 50-86 steps in bad episodes
- Remove doesn't work (Red has privileged access)
- Agent needs to learn to use Restore when Remove fails

Reward shaping:
- Escalating penalty for persistent Enterprise compromises
- Bonus for successfully cleaning Enterprise servers
- Bonus for using Restore on compromised Enterprise servers
"""

import numpy as np
import gymnasium as gym


# Host index constants
_HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]
_N_HOSTS = len(_HOSTS)
_ENTERPRISE_INDICES = [1, 2, 3]  # Enterprise0, Enterprise1, Enterprise2

# Action indices for Restore actions in reduced action space (30 actions)
_RESTORE_ENTERPRISE0 = 23
_RESTORE_ENTERPRISE1 = 24
_RESTORE_ENTERPRISE2 = 25


def _host_is_compromised(obs: np.ndarray, host_idx: int) -> bool:
    """True if the host has compromise bits set."""
    start = host_idx * 4
    comp0 = obs[start + 2]
    comp1 = obs[start + 3]
    return bool(comp0 or comp1)


class EnterpriseFocusWrapper(gym.Wrapper):
    """
    Adds escalating penalties for persistent Enterprise compromises
    to teach the agent to clean them quickly with Restore.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._enterprise_compromise_steps = np.zeros(3, dtype=int)  # Track steps for each Enterprise server
        self._last_enterprise_compromised = np.zeros(3, dtype=bool)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._enterprise_compromise_steps = np.zeros(3, dtype=int)

        # Initialize compromise tracking
        base_obs = obs[:52] if len(obs) >= 52 else obs
        for i, ent_idx in enumerate(_ENTERPRISE_INDICES):
            self._last_enterprise_compromised[i] = _host_is_compromised(base_obs, ent_idx)
            if self._last_enterprise_compromised[i]:
                self._enterprise_compromise_steps[i] = 1

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        base_obs = obs[:52] if len(obs) >= 52 else obs
        shaped_reward = 0.0

        # Check each Enterprise server
        for i, ent_idx in enumerate(_ENTERPRISE_INDICES):
            is_compromised = _host_is_compromised(base_obs, ent_idx)
            was_compromised = self._last_enterprise_compromised[i]

            if is_compromised:
                # Increment compromise step counter
                self._enterprise_compromise_steps[i] += 1

                # Escalating penalty for persistent compromise
                if self._enterprise_compromise_steps[i] > 10:
                    # After 10 steps, add escalating penalty
                    shaped_reward -= 0.5  # Aggressive penalty for long-term compromise

            else:
                if was_compromised:
                    # Enterprise was just cleaned - reward this!
                    shaped_reward += 0.3

                # Reset counter
                self._enterprise_compromise_steps[i] = 0

            # Bonus for using Restore on compromised Enterprise
            if is_compromised and action in [_RESTORE_ENTERPRISE0 + i]:
                shaped_reward += 0.2  # Encourage Restore usage on compromised hosts

            self._last_enterprise_compromised[i] = is_compromised

        return obs, reward + shaped_reward, terminated, truncated, info
