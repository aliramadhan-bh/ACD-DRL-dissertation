"""
AdaptiveEnsembleAgent
=====================
Routes each step to whichever PPO specialist is best suited for the
currently-active red agent, using early scan-pattern fingerprinting.

Red agent identification heuristic
------------------------------------
B_lineAgent follows a deterministic attack path: it scans User hosts
sequentially, then moves to Enterprise, then Op. Critically, it scans
one host at a time in a predictable order.

RedMeanderAgent scans randomly and can hit different hosts on consecutive
steps. Two consecutive scans on *different* User hosts in the first few
steps is a strong Meander signal.

The classifier works in two phases:
  1. Warmup (steps 1-4): collect which User hosts get scanned.
  2. Commit (step 4+): if we saw scans on 2+ *different* hosts,
     classify as Meander. Otherwise classify as B_line.

Once committed, the specialist is locked for the rest of the episode.

Host index reference (alphabetical BlueTableWrapper order):
  0=Defender, 1=Enterprise0, 2=Enterprise1, 3=Enterprise2,
  4=Op_Host0,  5=Op_Host1,   6=Op_Host2,   7=Op_Server0,
  8=User0,     9=User1,     10=User2,      11=User3,     12=User4
"""

import numpy as np
from stable_baselines3 import PPO

_USER_INDICES = {8, 9, 10, 11, 12}
_ENTERPRISE_INDICES = {1, 2, 3}

FINGERPRINT_STEPS = 4


def _scan_hosts(obs: np.ndarray) -> set[int]:
    """Return set of host indices that show Scan activity (act0=1, act1=0)."""
    scanned = set()
    for i in range(13):
        if obs[i * 4] == 1 and obs[i * 4 + 1] == 0:
            scanned.add(i)
    return scanned


class AdaptiveEnsembleAgent:
    """
    Wraps two trained PPO specialists and a scan-pattern fingerprinter.

    Parameters
    ----------
    bline_model_path : str
        Path to the PPO model trained against B_lineAgent.
    meander_model_path : str
        Path to the PPO model trained against RedMeanderAgent.
    fallback : str
        Which model to use during warmup ('bline' or 'meander').
    fingerprint_steps : int
        Number of steps to observe before committing to a specialist.
    """

    def __init__(
        self,
        bline_model_path: str,
        meander_model_path: str,
        fallback: str = "meander",
        fingerprint_steps: int = FINGERPRINT_STEPS,
    ):
        self.bline_model = PPO.load(bline_model_path)
        self.meander_model = PPO.load(meander_model_path)
        self.fallback = fallback
        self.fingerprint_steps = fingerprint_steps

        self._reset_episode_state()

    def _reset_episode_state(self):
        self._step = 0
        self._scanned_hosts: set[int] = set()
        self._committed_model: str | None = None

    def reset(self):
        self._reset_episode_state()

    def _update_classifier(self, obs: np.ndarray):
        scans = _scan_hosts(obs)
        self._scanned_hosts.update(scans)

    def _select_model(self) -> PPO:
        if self._committed_model is not None:
            return self.bline_model if self._committed_model == "bline" else self.meander_model

        if self._step < self.fingerprint_steps:
            return self.bline_model if self.fallback == "bline" else self.meander_model

        scanned_users = self._scanned_hosts & _USER_INDICES
        scanned_enterprise = self._scanned_hosts & _ENTERPRISE_INDICES

        if len(scanned_users) >= 2 or len(scanned_enterprise) >= 2:
            self._committed_model = "meander"
        else:
            self._committed_model = "bline"

        return self.bline_model if self._committed_model == "bline" else self.meander_model

    def _adapt_obs_for_model(self, obs: np.ndarray, model: PPO) -> np.ndarray:
        """Adapt observation length to match a model's expected Box shape.

        This allows mixing specialists trained with slightly different wrapper
        settings (for example 92-dim vs 105-dim observations). When dimensions
        differ we preserve the leading shared features and zero-pad/truncate.
        """
        expected_shape = getattr(model.observation_space, "shape", None)
        if not expected_shape or len(expected_shape) != 1:
            return obs

        expected_dim = int(expected_shape[0])
        curr_dim = int(obs.shape[0])

        if curr_dim == expected_dim:
            return obs

        if curr_dim > expected_dim:
            return obs[:expected_dim]

        pad = np.zeros(expected_dim - curr_dim, dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=0)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Select action using the appropriate specialist."""
        self._step += 1
        self._update_classifier(obs)
        model = self._select_model()
        model_obs = self._adapt_obs_for_model(obs, model)
        action, _ = model.predict(model_obs, deterministic=deterministic)
        return action

    @property
    def active_specialist(self) -> str:
        """Which specialist is currently committed ('bline', 'meander', or 'warmup')."""
        if self._committed_model is not None:
            return self._committed_model
        return "warmup"
