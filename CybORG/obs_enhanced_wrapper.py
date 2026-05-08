"""
ObsEnhancedWrapper
==================
Gymnasium wrapper that appends temporal features to the flat CybORG obs
vector, giving a PPO MLP policy the short-term memory it otherwise lacks.

Obs vector layout from BlueTableWrapper (sorted alphabetically by hostname):
  host index  hostname        bits in obs
  0           Defender        0-3
  1           Enterprise0     4-7
  2           Enterprise1     8-11
  3           Enterprise2     12-15
  4           Op_Host0        16-19
  5           Op_Host1        20-23
  6           Op_Host2        24-27
  7           Op_Server0      28-31
  8           User0           32-35
  9           User1           36-39
  10          User2           40-43
  11          User3           44-47
  12          User4           48-51

Each host contributes 4 bits:
  [activity_bit0, activity_bit1, compromised_bit0, compromised_bit1]
A host is "suspicious" if any of its 4 bits is non-zero.

Extra features appended (53 floats, all in [0, 1]):
  [0]      step_counter / max_steps
  [1-13]   steps_since_last_suspicious[host] / max_steps  (one per host)
  [14-26]  cumulative_hits[host] / max_steps              (one per host)
  [27-39]  scan_state[host] / 2.0                         (one per host)
           0 = not scanned, 1 = scanned previously, 2 = scanned this step
  [40-52]  decoy_state[host]                              (one per host)
           1.0 = host has active decoy, 0.0 = no decoy

When use_decoys=True, the wrapper reads decoy state from DecoyWrapper in the
env chain. If DecoyWrapper is not found, decoy_state is all zeros.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


HOSTS = [
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2",
    "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
]
N_HOSTS = len(HOSTS)
OP_SERVER_IDX = HOSTS.index("Op_Server0")
ENTERPRISE_INDICES = [HOSTS.index(h) for h in ("Enterprise0", "Enterprise1", "Enterprise2")]
# Extra features without decoys: step + steps_since_suspicious + cumulative_hits + scan_state
N_EXTRA_BASE = 1 + N_HOSTS + N_HOSTS + N_HOSTS  # 40
# Extra features with decoys: base + decoy_state
N_EXTRA_WITH_DECOYS = N_EXTRA_BASE + N_HOSTS  # 53


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


def _suspicious_flags(obs: np.ndarray) -> np.ndarray:
    """Return bool array of length N_HOSTS: True if any of the host's 4 bits is 1."""
    flags = np.zeros(N_HOSTS, dtype=bool)
    for i in range(N_HOSTS):
        flags[i] = obs[i * 4: i * 4 + 4].any()
    return flags


def _scan_flags(obs: np.ndarray) -> np.ndarray:
    """Return bool array: True if host shows Scan activity (act0=1, act1=0)."""
    flags = np.zeros(N_HOSTS, dtype=bool)
    for i in range(N_HOSTS):
        flags[i] = (obs[i * 4] == 1) and (obs[i * 4 + 1] == 0)
    return flags


class ObsEnhancedWrapper(gym.Wrapper):
    """Appends temporal context features to the CybORG flat observation."""

    def __init__(self, env: gym.Env, max_steps: int = 100, use_decoys: bool = False):
        super().__init__(env)
        self.max_steps = max_steps
        self.use_decoys = use_decoys

        # Choose observation size based on whether decoys are used
        n_extra = N_EXTRA_WITH_DECOYS if use_decoys else N_EXTRA_BASE

        orig = env.observation_space
        low = np.concatenate([orig.low, np.zeros(n_extra, dtype=np.float32)])
        high = np.concatenate([orig.high, np.ones(n_extra, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._step_count = 0
        self._steps_since_suspicious = np.zeros(N_HOSTS, dtype=np.float32)
        self._cumulative_hits = np.zeros(N_HOSTS, dtype=np.float32)
        self._scan_state = np.zeros(N_HOSTS, dtype=np.float32)
        self._decoy_state = np.zeros(N_HOSTS, dtype=np.float32)
        self._decoy_wrapper = None

    def _find_decoy_wrapper(self):
        """Locate DecoyWrapper in the env chain (cached after first call)."""
        if self._decoy_wrapper is None and self.use_decoys:
            self._decoy_wrapper = _get_decoy_wrapper(self.env)
        return self._decoy_wrapper

    def _update_decoy_state(self):
        """Update decoy state from DecoyWrapper if available."""
        self._decoy_state = np.zeros(N_HOSTS, dtype=np.float32)
        if self.use_decoys:
            dw = self._find_decoy_wrapper()
            if dw is not None:
                for host_idx in dw.decoy_hosts:
                    if 0 <= host_idx < N_HOSTS:
                        self._decoy_state[host_idx] = 1.0

    def _build_extra(self) -> np.ndarray:
        ms = max(self.max_steps, 1)
        step_feat = np.array([self._step_count / ms], dtype=np.float32)
        since_feat = np.clip(self._steps_since_suspicious / ms, 0.0, 1.0)
        hits_feat = np.clip(self._cumulative_hits / ms, 0.0, 1.0)
        scan_feat = self._scan_state / 2.0

        if self.use_decoys:
            decoy_feat = self._decoy_state
            return np.concatenate([step_feat, since_feat, hits_feat, scan_feat, decoy_feat])
        else:
            return np.concatenate([step_feat, since_feat, hits_feat, scan_feat])

    def _update_state(self, obs: np.ndarray):
        flags = _suspicious_flags(obs)
        self._steps_since_suspicious += 1.0
        self._steps_since_suspicious[flags] = 0.0
        self._cumulative_hits[flags] += 1.0

        scans = _scan_flags(obs)
        self._scan_state[self._scan_state == 2.0] = 1.0
        self._scan_state[scans] = 2.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._steps_since_suspicious = np.full(N_HOSTS, self.max_steps, dtype=np.float32)
        self._cumulative_hits = np.zeros(N_HOSTS, dtype=np.float32)
        self._scan_state = np.zeros(N_HOSTS, dtype=np.float32)
        self._decoy_state = np.zeros(N_HOSTS, dtype=np.float32)
        self._decoy_wrapper = None  # Reset cached reference
        self._update_state(obs)
        self._update_decoy_state()
        enhanced = np.concatenate([obs.astype(np.float32), self._build_extra()])
        return enhanced, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._update_state(obs)
        self._update_decoy_state()
        enhanced = np.concatenate([obs.astype(np.float32), self._build_extra()])
        return enhanced, reward, terminated, truncated, info
