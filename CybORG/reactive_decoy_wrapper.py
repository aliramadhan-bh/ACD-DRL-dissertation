"""
Reactive DecoyWrapper
=====================
Extends the greedy decoy strategy with REACTIVE decoy deployment.

Strategy:
  1. Greedy phase (steps 0-2): Deploy Misinform on Enterprise0-2 (like original)
  2. Reactive phase (step 3+): After any Restore action, deploy Misinform on that host

This addresses the re-compromise cycle problem:
  - Agent uses Restore_Enterprise2 → Enterprise2 clean
  - Wrapper queues Misinform_Enterprise2 for next step
  - Red re-compromises Enterprise2 → Decoy triggers IMMEDIATELY
  - Agent detects re-compromise in 1-2 steps instead of 10-30

Full action indices reference:
  Restore actions: 42-54 (Restore_Defender through Restore_User4)
  Misinform actions: 28-40 (Misinform_Defender through Misinform_User4)

Host mapping:
  Action 42 (Restore_Defender) → Host 0 → Action 28 (Misinform_Defender)
  Action 43 (Restore_Enterprise0) → Host 1 → Action 29 (Misinform_Enterprise0)
  Action 44 (Restore_Enterprise1) → Host 2 → Action 30 (Misinform_Enterprise1)
  Action 45 (Restore_Enterprise2) → Host 3 → Action 31 (Misinform_Enterprise2)
  ...
  Action 53 (Restore_User4) → Host 12 → Action 40 (Misinform_User4)
"""

import numpy as np
import gymnasium as gym
from collections import deque

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

# Full action indices
MISINFORM_ACTION_BASE = 28  # Misinform actions are 28-40
RESTORE_ACTION_BASE = 42    # Restore actions are 42-54

# Mapping: Restore action index → Misinform action index
RESTORE_TO_MISINFORM = {
    42 + i: 28 + i for i in range(N_HOSTS)
}

# Default greedy targets: Enterprise0, Enterprise1, Enterprise2
DEFAULT_DECOY_TARGETS = [1, 2, 3]


class ReactiveDecoyWrapper(gym.Wrapper):
    """Deploys decoys greedily at start, then reactively after Restore actions.

    Phase 1 (Greedy): Steps 0-2 deploy Misinform on Enterprise0-2
    Phase 2 (Reactive): Step 3+ deploy Misinform after any Restore

    Attributes
    ----------
    decoy_hosts : set[int]
        Set of host indices that currently have active decoys.
    decoy_queue : deque
        Queue of pending reactive decoy deployments (host indices).
    restore_count : dict
        Count of Restore actions per host (for diagnostics).
    reactive_decoy_count : int
        Total number of reactive decoys deployed.
    """

    def __init__(self, env: gym.Env, decoy_targets: list[int] = None):
        """
        Parameters
        ----------
        env : gym.Env
            The environment to wrap (should receive full 54-action space).
        decoy_targets : list[int], optional
            Host indices for greedy deployment. Default: [1, 2, 3] (Enterprise0-2).
        """
        super().__init__(env)
        self.decoy_targets = list(decoy_targets) if decoy_targets else DEFAULT_DECOY_TARGETS

        # Validate targets
        for t in self.decoy_targets:
            if t < 0 or t >= N_HOSTS:
                raise ValueError(f"Invalid decoy target: {t}. Must be 0-{N_HOSTS-1}.")

        # State tracking
        self._decoy_hosts: set[int] = set()
        self._decoy_queue: deque = deque()  # Queue of pending reactive decoys
        self._step_count = 0
        self._restore_count: dict[int, int] = {}  # host_idx → count
        self._reactive_decoy_count = 0
        self._prev_activity = np.zeros(N_HOSTS, dtype=bool)
        self._decoy_triggered: set[int] = set()

    @property
    def decoy_hosts(self) -> set[int]:
        """Set of host indices that currently have active decoys."""
        return self._decoy_hosts.copy()

    @property
    def decoy_triggered(self) -> set[int]:
        """Set of host indices where decoys were triggered."""
        return self._decoy_triggered.copy()

    @property
    def restore_count(self) -> dict[int, int]:
        """Count of Restore actions per host."""
        return self._restore_count.copy()

    @property
    def reactive_decoy_count(self) -> int:
        """Total number of reactive decoys deployed."""
        return self._reactive_decoy_count

    def _get_activity_flags(self, obs: np.ndarray) -> np.ndarray:
        """Extract activity flags from observation."""
        flags = np.zeros(N_HOSTS, dtype=bool)
        for i in range(N_HOSTS):
            act0 = obs[i * 4]
            act1 = obs[i * 4 + 1]
            flags[i] = bool(act0 or act1)
        return flags

    def _check_decoy_triggers(self, obs: np.ndarray):
        """Check if any decoy hosts have new activity."""
        curr_activity = self._get_activity_flags(obs)

        for host_idx in self._decoy_hosts:
            if curr_activity[host_idx] and not self._prev_activity[host_idx]:
                self._decoy_triggered.add(host_idx)

        self._prev_activity = curr_activity

    def _is_restore_action(self, action: int) -> bool:
        """Check if action is a Restore action."""
        return action in RESTORE_TO_MISINFORM

    def _get_restore_host(self, action: int) -> int:
        """Get host index from Restore action."""
        if not self._is_restore_action(action):
            return -1
        return action - RESTORE_ACTION_BASE

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset all state
        self._decoy_hosts = set()
        self._decoy_queue = deque()
        self._decoy_triggered = set()
        self._step_count = 0
        self._restore_count = {}
        self._reactive_decoy_count = 0
        self._prev_activity = self._get_activity_flags(obs)

        return obs, info

    def step(self, action):
        original_action = action
        action_type = "agent"  # Track what type of action was executed

        # PHASE 1: Greedy decoy deployment (first N steps)
        if self._step_count < len(self.decoy_targets):
            target_host = self.decoy_targets[self._step_count]
            action = MISINFORM_ACTION_BASE + target_host
            self._decoy_hosts.add(target_host)
            action_type = "greedy_decoy"

        # PHASE 2: Reactive decoy deployment (step 3+)
        elif len(self._decoy_queue) > 0:
            # Execute queued reactive decoy instead of agent's action
            target_host = self._decoy_queue.popleft()
            action = MISINFORM_ACTION_BASE + target_host
            self._decoy_hosts.add(target_host)
            self._reactive_decoy_count += 1
            action_type = "reactive_decoy"

        # PHASE 2: Check if agent used Restore → queue reactive decoy
        else:
            # Agent's action goes through
            action_type = "agent"

            # If agent used Restore, queue Misinform for that host
            if self._is_restore_action(original_action):
                host_idx = self._get_restore_host(original_action)
                if host_idx >= 0:
                    # Track Restore usage
                    self._restore_count[host_idx] = self._restore_count.get(host_idx, 0) + 1

                    # Queue reactive decoy deployment for next step
                    self._decoy_queue.append(host_idx)

        self._step_count += 1

        # Execute the action (greedy decoy, reactive decoy, or agent's action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check for decoy triggers
        self._check_decoy_triggers(obs)

        # Add diagnostic info
        info['decoy_hosts'] = self.decoy_hosts
        info['decoy_triggered'] = self.decoy_triggered
        info['decoy_queue_length'] = len(self._decoy_queue)
        info['reactive_decoy_count'] = self._reactive_decoy_count
        info['restore_count'] = self.restore_count
        info['action_type'] = action_type
        info['original_action'] = original_action
        info['executed_action'] = action

        return obs, reward, terminated, truncated, info
