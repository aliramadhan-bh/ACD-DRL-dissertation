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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
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
from enterprise_focus_wrapper import EnterpriseFocusWrapper
from hierarchical_wrapper import HierarchicalWrapper


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
    reward_shaping: bool = False,
    gentle_shaping: bool = False,
    enterprise_focus: bool = False,
    reduced_actions: bool = False,
    use_decoys: bool = False,
    hierarchical: bool = False,
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
    # GymCompatChallengeEnv -> DecoyWrapper -> ReducedActionWrapper -> ObsEnhancedWrapper -> RewardShapingWrapper -> EnterpriseFocusWrapper -> HierarchicalWrapper
    # DecoyWrapper must be below ReducedActionWrapper to use full action indices for Misinform.
    if use_decoys:
        env = DecoyWrapper(env)
    if reduced_actions:
        env = ReducedActionWrapper(env)
    if enhanced_obs:
        env = ObsEnhancedWrapper(env, max_steps=max_steps, use_decoys=use_decoys)
    if reward_shaping or gentle_shaping:
        env = RewardShapingWrapper(env, use_decoys=use_decoys, gentle=gentle_shaping)
    if enterprise_focus:
        env = EnterpriseFocusWrapper(env)
    if hierarchical:
        env = HierarchicalWrapper(env)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO Blue agent for CybORG.")
    parser.add_argument("--scenario", type=str, default="Scenario1b")
    parser.add_argument("--red-agent", choices=["bline", "meander", "mixed"], default="bline")
    parser.add_argument(
        "--algo",
        choices=["ppo", "recurrent_ppo"],
        default="ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Policy class name. Defaults: MlpPolicy (ppo), MlpLstmPolicy (recurrent_ppo).",
    )
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
    parser.add_argument(
        "--enhanced-obs",
        action="store_true",
        help="Append temporal observation features (step counter, steps-since-suspicious, cumulative hits).",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Add intermediate shaped rewards (effective response bonus, Op_Server0 survival/penalty). Use original aggressive values.",
    )
    parser.add_argument(
        "--gentle-shaping",
        action="store_true",
        help="Add gentle shaped rewards (80-90%% smaller than --reward-shaping). Recommended to reduce variance without bad habits.",
    )
    parser.add_argument(
        "--enterprise-focus",
        action="store_true",
        help="Add escalating penalties for persistent Enterprise compromises. Teaches fast Restore usage.",
    )
    parser.add_argument(
        "--reduced-actions",
        action="store_true",
        help="Use curated ~30-action subset instead of full 54-action space.",
    )
    parser.add_argument(
        "--use-decoys",
        action="store_true",
        help="Deploy decoys on Enterprise0-2 at episode start (greedy decoy strategy).",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical RL with goal-conditioned policy (solves re-compromise cycle problem).",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default="",
        help="Comma-separated hidden layer sizes for policy/value nets (e.g. '256,128').",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="Learning rate schedule. 'linear' decays LR to 0 over training.",
    )
    parser.add_argument("--model-out", type=str, default="models/ppo_blue_s1b_bline.zip")
    parser.add_argument(
        "--init-model",
        type=str,
        default="",
        help="Optional model path to continue training from.",
    )
    parser.add_argument(
        "--tb-log-dir",
        type=str,
        default="",
        help="TensorBoard log directory. Leave empty to disable.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Save checkpoint every N timesteps. 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Evaluate model every N timesteps and save best. 0 to disable evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for each evaluation.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory to save checkpoints. Defaults to {model_out}_checkpoints/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    tb_log_dir = None
    if args.tb_log_dir:
        tb_log_dir = Path(args.tb_log_dir)
        tb_log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.scenario,
        args.red_agent,
        args.max_steps,
        enhanced_obs=args.enhanced_obs,
        reward_shaping=args.reward_shaping,
        gentle_shaping=args.gentle_shaping,
        enterprise_focus=args.enterprise_focus,
        reduced_actions=args.reduced_actions,
        use_decoys=args.use_decoys,
        hierarchical=args.hierarchical,
    )
    env = Monitor(env)
    if args.algo == "recurrent_ppo" and RecurrentPPO is None:
        raise ImportError(
            "RecurrentPPO requires sb3-contrib. Install it with: pip install sb3-contrib"
        )
    policy_name = args.policy or (
        "MlpLstmPolicy" if args.algo == "recurrent_ppo" else "MlpPolicy"
    )

    # Build learning rate (callable for linear decay, float for constant)
    base_lr = args.learning_rate
    if args.lr_schedule == "linear":
        lr_schedule = lambda progress_remaining: progress_remaining * base_lr
    else:
        lr_schedule = base_lr

    # Build policy_kwargs if custom network architecture requested
    policy_kwargs = {}
    if args.net_arch:
        layers = [int(x.strip()) for x in args.net_arch.split(",")]
        policy_kwargs["net_arch"] = layers

    model_kwargs = dict(
        policy=policy_name,
        env=env,
        verbose=1,
        seed=args.seed,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )
    if policy_kwargs:
        model_kwargs["policy_kwargs"] = policy_kwargs
    if tb_log_dir is not None:
        model_kwargs["tensorboard_log"] = str(tb_log_dir)

    model_cls = RecurrentPPO if args.algo == "recurrent_ppo" else PPO
    if args.init_model:
        print(f"Loading initial model from: {args.init_model}")
        model = model_cls.load(args.init_model, env=env, seed=args.seed)
        # Keep loaded hyperparameters to avoid schedule mismatch issues.
    else:
        model = model_cls(**model_kwargs)

    # Set up checkpointing and evaluation callbacks
    callbacks = []

    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(str(model_out).replace(".zip", "_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Periodic checkpoint saving
    if args.checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpoints will be saved every {args.checkpoint_freq} timesteps to: {checkpoint_dir}")

    # Evaluation-based best model saving
    if args.eval_freq > 0:
        # Create separate evaluation environment (same config as training)
        eval_env = make_env(
            args.scenario,
            args.red_agent,
            args.max_steps,
            enhanced_obs=args.enhanced_obs,
            reward_shaping=args.reward_shaping,
            gentle_shaping=args.gentle_shaping,
            enterprise_focus=args.enterprise_focus,
            reduced_actions=args.reduced_actions,
            use_decoys=args.use_decoys,
            hierarchical=args.hierarchical,
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
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"Model will be evaluated every {args.eval_freq} timesteps ({args.eval_episodes} episodes)")
        print(f"Best model will be saved to: {best_model_path}/best_model.zip")

    # Combine callbacks
    callback = CallbackList(callbacks) if callbacks else None

    # Train with callbacks
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=callback)

    # Save final model to /tmp first to avoid iCloud sync timeouts when writing to Desktop/Documents
    with tempfile.NamedTemporaryFile(suffix=".zip", dir="/tmp", delete=False) as f:
        tmp_path = Path(f.name)
    model.save(str(tmp_path))
    shutil.move(str(tmp_path), str(model_out))
    print(f"Saved final {args.algo} model to: {model_out}")


if __name__ == "__main__":
    main()
