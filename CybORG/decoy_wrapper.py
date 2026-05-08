"""
DecoyWrapper
============
Gymnasium wrapper that deploys Misinform (decoy) actions on Enterprise hosts
during the first few steps of each episode, then hands control to the PPO policy.

This implements the "Greedy Decoys" strategy from the winning CardiffUni solution.

Decoy deployment order (default):
  Step 0: Misinform Enterprise0 (full action 29)
  Step 1: Misinform Enterprise1 (full action 30)
  Step 2: Misinform Enterprise2 (full action 31)
  Step 3+: Pass through to PPO policy

The wrapper tracks which hosts have active decoys. When Red exploits a decoy host,
the decoy absorbs/delays the attack, giving Blue time to respond.

Wrapper stacking order (outermost first):
  RewardShapingWrapper -> ObsEnhancedWrapper -> ReducedActionWrapper -> DecoyWrapper -> GymCompatChallengeEnv

Note: This wrapper receives FULL action indices (54 actions) since it sits below
ReducedActionWrapper. During the decoy phase, incoming actions are ignored and
Misinform actions are executed instead.

Host index reference (alphabetical BlueTableWrapper order):
  0=Defender, 1=Enterprise0, 2=Enterprise1, 3=Enterprise2,
  4=Op_Host0,  5=Op_Host1,   6=Op_Host2,   7=Op_Server0,
  8=User0,     9=User1,     10=User2,      11=User3,     12=User4

Full action indices for Misinform:
  28=Misinform Defender,   29=Misinform Enterprise0, 30=Misinform Enterprise1,
  31=Misinform Enterprise2, 32=Misinform Op_Host0,   33=Misinform Op_Host1,
  34=Misinform Op_Host2,   35=Misinform Op_Server0, 36=Misinform User0,
  37=Misinform User1,      38=Misinform User2,      39=Misinform User3,
  40=Misinform User4
"""

import numpy as np
import gymnasium as gym

# Number of hosts in Scenario1b
N_HOSTS = 13

# Host indices (alphabetical order from BlueTableWrapper)
HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]

# Full action indices for Misinform on each host
# Misinform actions are 28-40 (13 hosts, same order as HOSTS)
MISINFORM_ACTION_BASE = 28
MISINFORM_ACTIONS = {i: MISINFORM_ACTION_BASE + i for i in range(N_HOSTS)}

# Default: deploy decoys on Enterprise0, Enterprise1, Enterprise2
# These are indices 1, 2, 3 in the host list
DEFAULT_DECOY_TARGETS = [1, 2, 3]


def get_decoy_wrapper(env: gym.Env):
    """Traverse env wrapper chain to find DecoyWrapper instance.

    Parameters
    ----------
    env : gym.Env
        Any environment in the wrapper chain.

    Returns
    -------
    DecoyWrapper or None
        The DecoyWrapper instance if found, else None.
    """
    current = env
    while current is not None:
        if isinstance(current, DecoyWrapper):
            return current
        current = getattr(current, 'env', None)
    return None


class DecoyWrapper(gym.Wrapper):
    """Deploys decoys on specified hosts at episode start, then passes through.

    This wrapper intercepts the first N steps (where N = len(decoy_targets)) and
    executes Misinform actions instead of the policy's actions. After all decoys
    are deployed, the wrapper becomes transparent and passes all actions through.

    Attributes
    ----------
    decoy_hosts : set[int]
        Set of host indices that currently have active decoys.
    decoy_triggered : set[int]
        Set of host indices where a decoy was triggered (Red activity detected).
    """

    def __init__(self, env: gym.Env, decoy_targets: list[int] = None):
        """
        Parameters
        ----------
        env : gym.Env
            The environment to wrap (should receive full 54-action space).
        decoy_targets : list[int], optional
            Host indices to deploy decoys on, in order. Default: [1, 2, 3]
            (Enterprise0, Enterprise1, Enterprise2).
        """
        super().__init__(env)
        self.decoy_targets = list(decoy_targets) if decoy_targets else DEFAULT_DECOY_TARGETS

        # Validate targets
        for t in self.decoy_targets:
            if t < 0 or t >= N_HOSTS:
                raise ValueError(f"Invalid decoy target index: {t}. Must be 0-{N_HOSTS-1}.")
            if t not in MISINFORM_ACTIONS:
                raise ValueError(f"No Misinform action for host index {t}.")

        self._decoy_hosts: set[int] = set()
        self._decoy_triggered: set[int] = set()
        self._step_count = 0
        self._prev_activity = np.zeros(N_HOSTS, dtype=bool)

    @property
    def decoy_hosts(self) -> set[int]:
        """Set of host indices that currently have active decoys."""
        return self._decoy_hosts.copy()

    @property
    def decoy_triggered(self) -> set[int]:
        """Set of host indices where decoys were triggered this episode."""
        return self._decoy_triggered.copy()

    @property
    def n_decoy_steps(self) -> int:
        """Number of steps used for decoy deployment."""
        return len(self.decoy_targets)

    def _get_activity_flags(self, obs: np.ndarray) -> np.ndarray:
        """Extract activity flags from observation.

        Activity is detected when either activity bit is set:
        - [1, 0] = Scan
        - [1, 1] = Exploit
        """
        flags = np.zeros(N_HOSTS, dtype=bool)
        for i in range(N_HOSTS):
            act0 = obs[i * 4]
            act1 = obs[i * 4 + 1]
            flags[i] = bool(act0 or act1)
        return flags

    def _check_decoy_triggers(self, obs: np.ndarray):
        """Check if any decoy hosts have new activity (potential trigger)."""
        curr_activity = self._get_activity_flags(obs)

        for host_idx in self._decoy_hosts:
            # New activity on a decoy host = decoy triggered
            if curr_activity[host_idx] and not self._prev_activity[host_idx]:
                self._decoy_triggered.add(host_idx)

        self._prev_activity = curr_activity

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset decoy state
        self._decoy_hosts = set()
        self._decoy_triggered = set()
        self._step_count = 0
        self._prev_activity = self._get_activity_flags(obs)

        return obs, info

    def step(self, action):
        # During decoy deployment phase, override the action with Misinform
        original_action = action

        if self._step_count < len(self.decoy_targets):
            target_host = self.decoy_targets[self._step_count]
            action = MISINFORM_ACTIONS[target_host]
            self._decoy_hosts.add(target_host)

        self._step_count += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check for decoy triggers
        self._check_decoy_triggers(obs)

        # Add decoy info to info dict for debugging/logging
        info['decoy_hosts'] = self.decoy_hosts
        info['decoy_triggered'] = self.decoy_triggered
        info['decoy_phase_complete'] = self._step_count >= len(self.decoy_targets)

        return obs, reward, terminated, truncated, info
