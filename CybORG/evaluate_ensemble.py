"""
evaluate_ensemble.py
====================
Evaluation script for the AdaptiveEnsembleAgent.
Mirrors the interface of evaluate_ppo_blue.py for consistent comparison.

Usage
-----
python evaluate_ensemble.py \\
    --bline-model  models/ppo_enhanced_bline_1m_seed42.zip \\
    --meander-model models/ppo_enhanced_meander_1m_seed42.zip \\
    --red-agent bline \\
    --episodes 30 --max-steps 100 --seed 42

Optional flags
--------------
--enhanced-obs       Must be set if the specialist models were trained with --enhanced-obs.
--reduced-actions    Must be set if the specialist models were trained with --reduced-actions.
--deterministic      Use deterministic policy (recommended for evaluation).
--log-file           CSV file to append results to (default: EXPERIMENT_LOG_RL.csv).
--fallback           Which specialist to use during warmup ('bline' or 'meander').
--fingerprint-steps  Steps to observe before committing to a specialist (default: 4).
"""

import argparse
import csv
import inspect
import random
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

from obs_enhanced_wrapper import ObsEnhancedWrapper
from action_wrapper import ReducedActionWrapper
from adaptive_ensemble_agent import AdaptiveEnsembleAgent

try:
    # GymCompatChallengeEnv lives in evaluate_ppo_blue; import it directly.
    import importlib, sys
    _mod = importlib.import_module("evaluate_ppo_blue")
    GymCompatChallengeEnv = _mod.GymCompatChallengeEnv
except Exception:
    # Fallback: re-declare inline so this script is self-contained.
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

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
            raise TypeError(f"Unsupported space: {type(space)}")

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            obs = self.env.reset()
            return obs, {}

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return obs, float(reward), bool(done), False, info

        def close(self):
            if hasattr(self.env, "close"):
                self.env.close()


RED_AGENT_MAP = {
    "bline": B_lineAgent,
    "meander": RedMeanderAgent,
}


def make_env(scenario: str, red_agent_name: str, max_steps: int,
             enhanced_obs: bool, reduced_actions: bool = False):
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    challenge = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)
    env = GymCompatChallengeEnv(challenge)
    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps)
    return env


def append_log_row(log_file, bline_path, meander_path, scenario, red_agent,
                   episodes, max_steps, seed, rewards):
    out_path = Path(log_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "agent", "algo",
                "bline_model", "meander_model",
                "scenario", "red_agent", "episodes", "max_steps", "seed",
                "mean_reward", "std_reward", "worst_reward", "episode_rewards",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            "AdaptiveEnsemble", "ensemble",
            bline_path, meander_path,
            scenario, red_agent, episodes, max_steps, seed,
            f"{mean(rewards):.4f}",
            f"{(stdev(rewards) if len(rewards) > 1 else 0.0):.4f}",
            f"{min(rewards):.4f}",
            str(rewards),
        ])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveEnsembleAgent.")
    parser.add_argument("--bline-model", type=str, required=True,
                        help="Path to PPO model trained against bline.")
    parser.add_argument("--meander-model", type=str, required=True,
                        help="Path to PPO model trained against meander.")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander"], default="bline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enhanced-obs", action="store_true",
                        help="Apply ObsEnhancedWrapper (match training setup).")
    parser.add_argument("--reduced-actions", action="store_true",
                        help="Use curated ~30-action subset (match training setup).")
    parser.add_argument("--fallback", choices=["bline", "meander"], default="meander",
                        help="Specialist used during classifier warmup.")
    parser.add_argument("--fingerprint-steps", type=int, default=4,
                        help="Steps to observe before committing to a specialist.")
    parser.add_argument("--log-file", type=str, default="EXPERIMENT_LOG_RL.csv")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N episodes (0 disables intermediate progress logs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_start = time.time()

    env = make_env(args.scenario, args.red_agent, args.max_steps,
                   args.enhanced_obs, reduced_actions=args.reduced_actions)
    agent = AdaptiveEnsembleAgent(
        bline_model_path=args.bline_model,
        meander_model_path=args.meander_model,
        fallback=args.fallback,
        fingerprint_steps=args.fingerprint_steps,
    )

    rewards = []
    specialist_choices = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        agent.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            action = agent.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(ep_reward)
        specialist_choices.append(agent.active_specialist)
        if args.progress_every > 0 and ((episode + 1) % args.progress_every == 0):
            elapsed = time.time() - run_start
            print(
                f"[progress] episode={episode + 1}/{args.episodes} "
                f"reward={ep_reward:.4f} specialist={agent.active_specialist} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    run_mean = mean(rewards)
    run_std = stdev(rewards) if len(rewards) > 1 else 0.0

    print("Episode rewards:", rewards)
    print(f"Average reward:  {run_mean:.4f}")
    print(f"Std deviation:   {run_std:.4f}")
    print(f"Worst episode:   {min(rewards):.4f}")
    print(f"Specialist used per episode: {specialist_choices}")
    print(
        f"Run config -> bline={args.bline_model}, meander={args.meander_model}, "
        f"scenario={args.scenario}, red_agent={args.red_agent}, "
        f"episodes={args.episodes}, max_steps={args.max_steps}, seed={args.seed}"
    )

    if args.log_file:
        append_log_row(
            args.log_file, args.bline_model, args.meander_model,
            args.scenario, args.red_agent, args.episodes,
            args.max_steps, args.seed, rewards,
        )
        print(f"Saved run to: {args.log_file}")


if __name__ == "__main__":
    main()
