import argparse
import csv
import inspect
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces as gym_spaces
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from obs_enhanced_wrapper import ObsEnhancedWrapper
from reward_shaping_wrapper import RewardShapingWrapper
from action_wrapper import ReducedActionWrapper
from decoy_wrapper import DecoyWrapper


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


def make_env(
    scenario: str,
    red_agent_name: str,
    max_steps: int,
    enhanced_obs: bool = False,
    reward_shaping: bool = False,
    reduced_actions: bool = False,
    use_decoys: bool = False,
):
    scenario_path = str(inspect.getfile(CybORG))
    scenario_path = scenario_path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"
    red_agent = RED_AGENT_MAP[red_agent_name]
    cyborg = CybORG(scenario_path, "sim", agents={"Red": red_agent})
    challenge = ChallengeWrapper(env=cyborg, agent_name="Blue", max_steps=max_steps)
    env = GymCompatChallengeEnv(challenge)

    # Wrapper stacking order (bottom to top):
    # GymCompatChallengeEnv -> DecoyWrapper -> ReducedActionWrapper -> ObsEnhancedWrapper -> RewardShapingWrapper
    if use_decoys:
        env = DecoyWrapper(env)
    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=use_decoys)
    if reward_shaping:
        env = RewardShapingWrapper(env, use_decoys=use_decoys)
    return env


def append_log_row(
    log_file: str,
    algo: str,
    model_path: str,
    scenario: str,
    red_agent: str,
    episodes: int,
    max_steps: int,
    seed: int,
    rewards: list[float],
):
    out_path = Path(log_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "agent",
                    "algo",
                    "model_path",
                    "scenario",
                    "red_agent",
                    "episodes",
                    "max_steps",
                    "seed",
                    "mean_reward",
                    "std_reward",
                    "worst_reward",
                    "episode_rewards",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                "PPO",
                algo,
                model_path,
                scenario,
                red_agent,
                episodes,
                max_steps,
                seed,
                f"{mean(rewards):.4f}",
                f"{(stdev(rewards) if len(rewards) > 1 else 0.0):.4f}",
                f"{min(rewards):.4f}",
                str(rewards),
            ]
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO Blue agent.")
    parser.add_argument(
        "--algo",
        choices=["ppo", "recurrent_ppo"],
        default="ppo",
        help="Algorithm used by the saved model.",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander"], default="bline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--enhanced-obs",
        action="store_true",
        help="Apply ObsEnhancedWrapper (must match how the model was trained).",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Apply RewardShapingWrapper (must match how the model was trained).",
    )
    parser.add_argument(
        "--reduced-actions",
        action="store_true",
        help="Use curated ~30-action subset (must match how the model was trained).",
    )
    parser.add_argument(
        "--use-decoys",
        action="store_true",
        help="Deploy decoys on Enterprise0-2 at episode start (must match training).",
    )
    parser.add_argument("--log-file", type=str, default="EXPERIMENT_LOG_RL.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate model file before loading
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(f"Model file is empty: {args.model_path}")
    try:
        import zipfile
        with zipfile.ZipFile(model_path, "r") as zf:
            pass  # Valid zip
    except zipfile.BadZipFile:
        raise ValueError(
            f"Model file is corrupted (not a valid zip): {args.model_path}\n"
            f"Retrain the model or use a different checkpoint."
        ) from None

    if args.algo == "recurrent_ppo" and RecurrentPPO is None:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. Install it with: pip install sb3-contrib"
        )
    env = make_env(args.scenario, args.red_agent, args.max_steps, enhanced_obs=args.enhanced_obs, reward_shaping=args.reward_shaping, reduced_actions=args.reduced_actions, use_decoys=args.use_decoys)
    model_cls = RecurrentPPO if args.algo == "recurrent_ppo" else PPO
    model = model_cls.load(args.model_path)

    rewards = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        recurrent_state = None
        episode_start = np.array([True], dtype=bool)
        while not done and steps < args.max_steps:
            if args.algo == "recurrent_ppo":
                action, recurrent_state = model.predict(
                    obs,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=args.deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)
            steps += 1
        rewards.append(ep_reward)

    run_mean = mean(rewards)
    run_std = stdev(rewards) if len(rewards) > 1 else 0.0
    print("Episode rewards:", rewards)
    print(f"Average reward: {run_mean:.4f}")
    print(f"Std deviation: {run_std:.4f}")
    print(f"Worst episode: {min(rewards):.4f}")
    print(
        f"Run config -> algo={args.algo}, model={args.model_path}, scenario={args.scenario}, "
        f"red_agent={args.red_agent}, episodes={args.episodes}, max_steps={args.max_steps}, seed={args.seed}"
    )

    if args.log_file:
        append_log_row(
            args.log_file,
            args.algo,
            args.model_path,
            args.scenario,
            args.red_agent,
            args.episodes,
            args.max_steps,
            args.seed,
            rewards,
        )
        print(f"Saved run to: {args.log_file}")


if __name__ == "__main__":
    main()
