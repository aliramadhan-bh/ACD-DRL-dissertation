"""
Evaluate Reactive Decoy Strategy
=================================
Tests the reactive decoy deployment strategy with an existing trained model.

Key difference from regular evaluation:
- Uses ReactiveDecoyWrapper instead of DecoyWrapper
- Deploys decoys reactively after Restore actions
- No retraining needed - uses existing model

Usage:
  python evaluate_reactive_decoys.py \
    --model-path models/ppo_decoys_no_shaping_bline_10m.zip \
    --episodes 30 --seed 200
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
from reactive_decoy_wrapper import ReactiveDecoyWrapper  # NEW!


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
              enhanced_obs: bool, reduced_actions: bool):
    """Build environment with REACTIVE decoy wrapper."""
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    challenge_env = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)

    env = GymCompatChallengeEnv(challenge_env)

    # Use ReactiveDecoyWrapper instead of DecoyWrapper!
    env = ReactiveDecoyWrapper(env)

    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=True)

    return env


def evaluate_with_diagnostics(model, env, episodes: int, deterministic: bool = True):
    """Evaluate with reactive decoy diagnostics."""
    rewards = []
    reactive_decoy_stats = []
    restore_stats = []

    print(f"\n{'='*80}")
    print(f"REACTIVE DECOY EVALUATION")
    print(f"{'='*80}\n")

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        episode_reactive_decoys = 0
        episode_restores = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated
            step += 1

            # Track reactive decoy deployments
            if info.get('action_type') == 'reactive_decoy':
                episode_reactive_decoys += 1

            # Track Restore usage
            if info.get('restore_count'):
                episode_restores = sum(info['restore_count'].values())

        rewards.append(episode_reward)
        reactive_decoy_stats.append(episode_reactive_decoys)
        restore_stats.append(episode_restores)

        print(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f} | "
              f"Reactive Decoys={episode_reactive_decoys:2d} | "
              f"Restores={episode_restores:2d}")

    rewards = np.array(rewards)
    reactive_decoys = np.array(reactive_decoy_stats)
    restores = np.array(restore_stats)

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Reward:       {rewards.mean():7.4f}")
    print(f"Std Deviation:     {rewards.std():7.4f}")
    print(f"Best Episode:      {rewards.max():7.4f}")
    print(f"Worst Episode:     {rewards.min():7.4f}")
    print(f"\nReactive Decoy Usage:")
    print(f"  Mean per episode:   {reactive_decoys.mean():5.2f}")
    print(f"  Total deployed:     {reactive_decoys.sum():5d}")
    print(f"  Episodes with >0:   {(reactive_decoys > 0).sum():2d}/{episodes}")
    print(f"\nRestore Usage:")
    print(f"  Mean per episode:   {restores.mean():5.2f}")
    print(f"  Total:              {restores.sum():5d}")
    print(f"{'='*80}\n")

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate reactive decoy strategy")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained PPO model")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander"], default="bline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--enhanced-obs", action="store_true",
                        help="Use enhanced observations (temporal features)")
    parser.add_argument("--reduced-actions", action="store_true",
                        help="Use reduced action space (30 actions)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*80}")
    print(f"REACTIVE DECOY STRATEGY TEST")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Strategy: Greedy (steps 0-2) + Reactive (after Restore)")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print(f"{'='*80}\n")

    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)

    env = build_env(
        args.scenario,
        args.red_agent,
        args.max_steps,
        args.enhanced_obs,
        args.reduced_actions
    )

    evaluate_with_diagnostics(model, env, args.episodes, args.deterministic)


if __name__ == "__main__":
    main()
