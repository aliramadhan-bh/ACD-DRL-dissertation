"""
Evaluation script for models trained without Misinform actions.
"""
import argparse
import inspect
import random
from pathlib import Path
from statistics import mean, stdev

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
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


def make_env(scenario, red_agent_name, max_steps, enhanced_obs=False, reward_shaping=False):
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    challenge = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)
    env = GymCompatChallengeEnv(challenge)

    # Apply no-misinform wrapper
    env = ReducedActionWrapperNoMisinform(env)

    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=False)
    if reward_shaping:
        env = RewardShapingWrapper(env, use_decoys=False)

    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model trained without Misinform.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander"], default="bline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enhanced-obs", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate model file
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    env = make_env(
        args.scenario, args.red_agent, args.max_steps,
        enhanced_obs=args.enhanced_obs,
        reward_shaping=args.reward_shaping
    )

    model = PPO.load(args.model_path)

    print(f"\n{'='*80}")
    print(f"EVALUATING NO-MISINFORM MODEL")
    print(f"Model: {args.model_path}")
    print(f"Action space: 21 actions (no Misinform)")
    print(f"Opponent: {args.red_agent}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*80}\n")

    rewards = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        rewards.append(ep_reward)
        print(f"Episode {episode + 1:2d}: {ep_reward:7.2f}")

    run_mean = mean(rewards)
    run_std = stdev(rewards) if len(rewards) > 1 else 0.0

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Mean Reward:      {run_mean:7.2f}")
    print(f"Std Deviation:    {run_std:7.2f}")
    print(f"Best Episode:     {max(rewards):7.2f}")
    print(f"Worst Episode:    {min(rewards):7.2f}")
    print(f"Target (winning): -13.00 to -16.00")
    print(f"Gap to target:    {run_mean - (-15):7.2f} points")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
