"""
Evaluate Hierarchical RL models
================================
Evaluates PPO models trained with HierarchicalWrapper.
"""

import argparse
import inspect
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces as gym_spaces
from stable_baselines3 import PPO

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from obs_enhanced_wrapper import ObsEnhancedWrapper
from action_wrapper import ReducedActionWrapper
from decoy_wrapper import DecoyWrapper
from hierarchical_wrapper import HierarchicalWrapper


RED_AGENT_MAP = {
    "bline": B_lineAgent,
    "meander": RedMeanderAgent,
}


class GymCompatChallengeEnv(gym.Env):
    """Wraps ChallengeWrapper to be Gymnasium-compatible."""
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
        raise TypeError(f"Unsupported space type: {type(space)}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), False, info

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def build_env(scenario: str, red_agent_name: str, max_steps: int,
              enhanced_obs: bool, reduced_actions: bool, use_decoys: bool, hierarchical: bool):
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    challenge_env = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)

    # Wrap with GymCompatChallengeEnv first
    env = GymCompatChallengeEnv(challenge_env)

    # Apply wrappers in same order as training
    if use_decoys:
        env = DecoyWrapper(env)
    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=use_decoys)
    if hierarchical:
        env = HierarchicalWrapper(env)

    return env


def evaluate_with_diagnostics(model, env, episodes: int, deterministic: bool = True):
    """Evaluate with detailed goal tracking."""
    rewards = []
    goal_distribution = np.zeros(5, dtype=int)  # Track goal usage
    goal_names = ["MONITOR", "DEFEND_ENTERPRISE", "RAPID_RESTORE", "DEFEND_OPSERVER", "DEFEND_USERS"]

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated
            step += 1

            # Track goal distribution (goal is in last 6 dims: one-hot + steps)
            goal_idx = np.argmax(obs[-6:-1])
            goal_distribution[goal_idx] += 1

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}")

    rewards = np.array(rewards)

    # Print diagnostics
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Mean Reward:   {rewards.mean():.4f}")
    print(f"Std Deviation: {rewards.std():.4f}")
    print(f"Best Episode:  {rewards.max():.4f}")
    print(f"Worst Episode: {rewards.min():.4f}")
    print(f"\nGoal Distribution:")
    total_steps = goal_distribution.sum()
    for i, name in enumerate(goal_names):
        pct = 100.0 * goal_distribution[i] / total_steps if total_steps > 0 else 0
        print(f"  {name:20s}: {goal_distribution[i]:6d} steps ({pct:5.1f}%)")
    print(f"{'='*80}\n")

    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander"], default="bline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhanced-obs", action="store_true")
    parser.add_argument("--reduced-actions", action="store_true")
    parser.add_argument("--use-decoys", action="store_true")
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nLoading model from: {args.model_path}")
    model = PPO.load(args.model_path)

    env = build_env(
        args.scenario,
        args.red_agent,
        args.max_steps,
        args.enhanced_obs,
        args.reduced_actions,
        args.use_decoys,
        args.hierarchical
    )

    print(f"\nEvaluating on {args.red_agent} for {args.episodes} episodes...\n")
    evaluate_with_diagnostics(model, env, args.episodes, args.deterministic)


if __name__ == "__main__":
    main()
