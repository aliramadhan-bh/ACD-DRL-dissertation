"""
Quick training script using the no-misinform action wrapper.
Tests hypothesis that removing Misinform actions improves performance.
"""
import argparse
import inspect
import random
import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces as gym_spaces

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from obs_enhanced_wrapper import ObsEnhancedWrapper
from reward_shaping_wrapper import RewardShapingWrapper
from action_wrapper_no_misinform import ReducedActionWrapperNoMisinform

RED_AGENT_MAP = {
    "bline": B_lineAgent,
    "meander": RedMeanderAgent,
}


class GymCompatChallengeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, challenge_env):
        self.env = challenge_env
        self.action_space = self._to_gymnasium_space(self.env.action_space)
        self.observation_space = self._to_gymnasium_space(self.env.observation_space)

    def _to_gymnasium_space(self, space):
        if isinstance(space, gym_spaces.Space):
            return space
        if hasattr(space, "n"):
            return gym_spaces.Discrete(space.n)
        if hasattr(space, "nvec"):
            return gym_spaces.MultiDiscrete(space.nvec)
        if hasattr(space, "shape") and hasattr(space, "dtype") and hasattr(space, "low") and hasattr(space, "high"):
            return gym_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return gym_spaces.Dict({k: self._to_gymnasium_space(v) for k, v in space.spaces.items()})
        if hasattr(space, "spaces") and isinstance(space.spaces, (list, tuple)):
            return gym_spaces.Tuple(tuple(self._to_gymnasium_space(v) for v in space.spaces))
        raise TypeError(f"Unsupported space type for conversion: {type(space)}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class MixedRedGymEnv(GymCompatChallengeEnv):
    def __init__(self, scenario: str, red_agent_names: list[str], max_steps: int):
        self.scenario = scenario
        self.max_steps = max_steps
        self.red_agent_names = red_agent_names
        self.current_red_agent = red_agent_names[0]
        first_env = build_challenge_env(self.scenario, self.current_red_agent, self.max_steps)
        super().__init__(first_env)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        chosen_red = random.choice(self.red_agent_names)
        if chosen_red != self.current_red_agent:
            if hasattr(self.env, "close"):
                self.env.close()
            self.current_red_agent = chosen_red
            self.env = build_challenge_env(self.scenario, self.current_red_agent, self.max_steps)
        obs = self.env.reset()
        return obs, {}


def build_challenge_env(scenario: str, red_agent_name: str, max_steps: int):
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    return ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)


def make_env(scenario, red_agent_name, max_steps, enhanced_obs=False, reward_shaping=False):
    if red_agent_name == "mixed":
        env = MixedRedGymEnv(scenario=scenario, red_agent_names=["bline", "meander"], max_steps=max_steps)
    else:
        challenge = build_challenge_env(scenario, red_agent_name, max_steps)
        env = GymCompatChallengeEnv(challenge)

    # Apply new no-misinform wrapper
    env = ReducedActionWrapperNoMisinform(env)

    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=False)
    if reward_shaping:
        env = RewardShapingWrapper(env, use_decoys=False)

    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Blue agent without Misinform actions.")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander", "mixed"], default="bline")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--enhanced-obs", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--model-out", type=str, default="models/ppo_no_misinform_test.zip")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training PPO with NO MISINFORM actions")
    print(f"Action space: 21 actions (was 30 with Misinform)")
    print(f"Opponent: {args.red_agent}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"{'='*80}\n")

    env = make_env(
        args.scenario,
        args.red_agent,
        args.max_steps,
        enhanced_obs=args.enhanced_obs,
        reward_shaping=args.reward_shaping,
    )
    env = Monitor(env)

    print(f"Environment created:")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"\nStarting training...\n")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    # Save to /tmp first to avoid iCloud sync issues
    with tempfile.NamedTemporaryFile(suffix=".zip", dir="/tmp", delete=False) as f:
        tmp_path = Path(f.name)
    model.save(str(tmp_path))
    shutil.move(str(tmp_path), str(model_out))

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Model saved to: {model_out}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
