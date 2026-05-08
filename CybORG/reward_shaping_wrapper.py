"""
RewardShapingWrapper
====================
Gymnasium wrapper that adds intermediate shaped rewards on top of the default
CybORG reward signal, reducing the credit-assignment gap.

Default CybORG reward (per step):
  -1  per compromised non-critical host
  -10 if Op_Server0 is compromised

Shaped additions:
  +1.5  when a host transitions from suspicious -> clean  (effective response)
  +0.1  every step that Op_Server0 is clean              (survival bonus)
  -2.0  every step that Op_Server0 is compromised        (critical penalty)
  +2.0  when a decoy is triggered (Red hits a decoy host) (decoy bonus)

The shaped rewards are intentionally small relative to the base signal so the
agent cannot exploit them without actually defending the network.

Stacking order (outermost first):
  RewardShapingWrapper -> ObsEnhancedWrapper -> ReducedActionWrapper -> DecoyWrapper -> GymCompatChallengeEnv
"""

import numpy as np
import gymnasium as gym

# Host index constants — must match the alphabetical order used by BlueTableWrapper.
_HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]
_N_HOSTS = len(_HOSTS)
_OP_SERVER_IDX = _HOSTS.index("Op_Server0")

# When ObsEnhancedWrapper is stacked below, the first 52 floats are the
# original binary obs.  When it is not stacked, the full obs is 52 floats.
# Either way, the first 52 values encode host states.
_BASE_OBS_DIM = 52


def _get_decoy_wrapper(env: gym.Env):
    """Traverse env wrapper chain to find DecoyWrapper instance."""
    # Import here to avoid circular dependency
    from decoy_wrapper import DecoyWrapper
    current = env
    while current is not None:
        if isinstance(current, DecoyWrapper):
            return current
        current = getattr(current, 'env', None)
    return None


def _host_suspicious(obs: np.ndarray, host_idx: int) -> bool:
    """True if any of the four bits for this host is non-zero."""
    start = host_idx * 4
    return bool(obs[start: start + 4].any())


def _op_server_compromised(obs: np.ndarray) -> bool:
    """True if the Op_Server0 compromised bits indicate any compromise."""
    start = _OP_SERVER_IDX * 4
    # bits layout per host: [act0, act1, comp0, comp1]
    # compromised = Unknown (1,0), User (0,1), Privileged (1,1)
    comp0 = obs[start + 2]
    comp1 = obs[start + 3]
    return bool(comp0 or comp1)


class RewardShapingWrapper(gym.Wrapper):
    """Adds intermediate rewards to accelerate learning of correct defensive actions."""

    # Aggressive shaping (original values - too high, causes bad habits)
    EFFECTIVE_RESPONSE_BONUS = 1.5
    OP_SERVER_CLEAN_BONUS = 0.1
    OP_SERVER_COMPROMISED_PENALTY = -2.0
    DECOY_TRIGGER_BONUS = 2.0

    # Gentle shaping (reduced by 80-90% - provides guidance without overwhelming base reward)
    GENTLE_EFFECTIVE_RESPONSE_BONUS = 0.2
    GENTLE_OP_SERVER_CLEAN_BONUS = 0.05
    GENTLE_OP_SERVER_COMPROMISED_PENALTY = -0.5
    GENTLE_DECOY_TRIGGER_BONUS = 0.3

    def __init__(self, env: gym.Env, use_decoys: bool = False, gentle: bool = False):
        super().__init__(env)
        self.use_decoys = use_decoys
        self.gentle = gentle
        self._prev_suspicious = np.zeros(_N_HOSTS, dtype=bool)
        self._prev_decoy_triggered: set[int] = set()
        self._decoy_wrapper = None

        # Set reward values based on gentle mode
        if gentle:
            self.effective_response_bonus = self.GENTLE_EFFECTIVE_RESPONSE_BONUS
            self.op_server_clean_bonus = self.GENTLE_OP_SERVER_CLEAN_BONUS
            self.op_server_compromised_penalty = self.GENTLE_OP_SERVER_COMPROMISED_PENALTY
            self.decoy_trigger_bonus = self.GENTLE_DECOY_TRIGGER_BONUS
        else:
            self.effective_response_bonus = self.EFFECTIVE_RESPONSE_BONUS
            self.op_server_clean_bonus = self.OP_SERVER_CLEAN_BONUS
            self.op_server_compromised_penalty = self.OP_SERVER_COMPROMISED_PENALTY
            self.decoy_trigger_bonus = self.DECOY_TRIGGER_BONUS

    def _find_decoy_wrapper(self):
        """Locate DecoyWrapper in the env chain (cached after first call)."""
        if self._decoy_wrapper is None and self.use_decoys:
            self._decoy_wrapper = _get_decoy_wrapper(self.env)
        return self._decoy_wrapper

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_suspicious = np.array(
            [_host_suspicious(obs, i) for i in range(_N_HOSTS)], dtype=bool
        )
        self._prev_decoy_triggered = set()
        self._decoy_wrapper = None  # Reset cached reference
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_suspicious = np.array(
            [_host_suspicious(obs, i) for i in range(_N_HOSTS)], dtype=bool
        )

        # Bonus for each host that was suspicious last step and is now clean.
        cleared = self._prev_suspicious & ~curr_suspicious
        shaped = float(cleared.sum()) * self.effective_response_bonus

        # Op_Server0 step-level bonus / penalty.
        if _op_server_compromised(obs):
            shaped += self.op_server_compromised_penalty
        else:
            shaped += self.op_server_clean_bonus

        # Decoy trigger bonus: reward when Red hits a decoy host.
        if self.use_decoys:
            dw = self._find_decoy_wrapper()
            if dw is not None:
                # Find newly triggered decoys (not counted before)
                curr_triggered = dw.decoy_triggered
                new_triggers = curr_triggered - self._prev_decoy_triggered
                shaped += len(new_triggers) * self.decoy_trigger_bonus
                self._prev_decoy_triggered = curr_triggered.copy()

        self._prev_suspicious = curr_suspicious
        return obs, reward + shaped, terminated, truncated, info
