"""
Train Masked PPO with Action Masking
====================================
Uses MaskablePPO from sb3-contrib to prevent wasteful actions.
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gymnasium import spaces as gym_spaces

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
except ImportError:
    print("ERROR: MaskablePPO requires sb3-contrib. Install with: pip install sb3-contrib")
    exit(1)

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from obs_enhanced_wrapper import ObsEnhancedWrapper
from action_wrapper import ReducedActionWrapper
from decoy_wrapper import DecoyWrapper
from action_masking_wrapper import ActionMaskingWrapper


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

        # Convert legacy gym spaces used by CybORG wrappers.
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

    def render(self):
        return self.env.render()

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


def make_env(
    scenario: str,
    red_agent_name: str,
    max_steps: int,
    enhanced_obs: bool = False,
    reduced_actions: bool = False,
    use_decoys: bool = False,
    use_action_masking: bool = True,
):
    if red_agent_name == "mixed":
        env = MixedRedGymEnv(
            scenario=scenario,
            red_agent_names=["bline", "meander"],
            max_steps=max_steps,
        )
    else:
        challenge = build_challenge_env(scenario, red_agent_name, max_steps)
        env = GymCompatChallengeEnv(challenge)

    # Wrapper stacking order (bottom to top):
    if use_decoys:
        env = DecoyWrapper(env)
    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=use_decoys)
    if use_action_masking:
        env = ActionMaskingWrapper(env)
        # Wrap with ActionMasker for sb3-contrib compatibility
        env = ActionMasker(env, lambda env: env.action_masks())

    return env


def mask_fn(env):
    """Extract action mask from Dict observation space."""
    return lambda: env.unwrapped.get_attr("_get_action_mask")[0](
        env.get_attr("_last_obs")[0]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Masked PPO Blue agent for CybORG.")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander", "mixed"], default="bline")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--enhanced-obs", action="store_true")
    parser.add_argument("--reduced-actions", action="store_true")
    parser.add_argument("--use-decoys", action="store_true")
    parser.add_argument("--model-out", type=str, default="models/masked_ppo_blue.zip")
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.scenario,
        args.red_agent,
        args.max_steps,
        enhanced_obs=args.enhanced_obs,
        reduced_actions=args.reduced_actions,
        use_decoys=args.use_decoys,
        use_action_masking=True,
    )
    env = Monitor(env)

    model_kwargs = dict(
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )

    model = MaskablePPO(**model_kwargs)

    # Set up checkpointing callbacks
    callbacks = []
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(str(model_out).replace(".zip", "_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="checkpoint",
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpoints every {args.checkpoint_freq} timesteps to: {checkpoint_dir}")

    if args.eval_freq > 0:
        eval_env = make_env(
            args.scenario,
            args.red_agent,
            args.max_steps,
            enhanced_obs=args.enhanced_obs,
            reduced_actions=args.reduced_actions,
            use_decoys=args.use_decoys,
            use_action_masking=True,
        )
        eval_env = Monitor(eval_env)

        best_model_path = checkpoint_dir / "best_model"
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_path),
            log_path=str(checkpoint_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_callback)
        print(f"Evaluation every {args.eval_freq} timesteps ({args.eval_episodes} episodes)")
        print(f"Best model saved to: {best_model_path}/best_model.zip")

    callback = CallbackList(callbacks) if callbacks else None

    print("Training Masked PPO with action masking enabled...")
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=callback)

    # Save final model
    with tempfile.NamedTemporaryFile(suffix=".zip", dir="/tmp", delete=False) as f:
        tmp_path = Path(f.name)
    model.save(str(tmp_path))
    shutil.move(str(tmp_path), str(model_out))
    print(f"Saved final Masked PPO model to: {model_out}")


if __name__ == "__main__":
    main()
