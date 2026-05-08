import argparse
import inspect
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from my_blue_agent import MyBlueHeuristicAgent


def _append_experiment_log(log_file, episodes, max_steps, seed, rewards):
    log_path = Path(log_file)
    file_exists = log_path.exists()
    run_mean = mean(rewards)
    run_std = stdev(rewards) if len(rewards) > 1 else 0.0

    with log_path.open("a", encoding="utf-8") as log:
        if not file_exists:
            log.write(
                "timestamp,agent,red_agent,episodes,max_steps,seed,mean_reward,std_reward,episode_rewards\n"
            )

        log.write(
            f"{datetime.now().isoformat(timespec='seconds')},"
            "MyBlueHeuristicAgent,B_lineAgent,"
            f"{episodes},{max_steps},{seed},"
            f"{run_mean:.4f},{run_std:.4f},\"{rewards}\"\n"
        )


def run(episodes=3, max_steps=20, seed=None, log_file="EXPERIMENT_LOG.csv"):
    if seed is not None:
        random.seed(seed)

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + "/Shared/Scenarios/Scenario1b.yaml"

    rewards = []

    for episode_idx in range(episodes):
        env = CybORG(path, "sim", agents={"Red": B_lineAgent})
        if seed is not None and hasattr(env, "set_seed"):
            env.set_seed(seed + episode_idx)
        blue_agent = MyBlueHeuristicAgent()
        results = env.reset("Blue")
        action_space = results.action_space
        episode_reward = 0.0

        for _ in range(max_steps):
            action = blue_agent.get_action(results.observation, action_space)
            results = env.step(agent="Blue", action=action)
            episode_reward += results.reward
            action_space = (
                results.action_space
                if hasattr(results, "action_space") and results.action_space is not None
                else env.get_action_space("Blue")
            )

        blue_agent.end_episode()
        rewards.append(episode_reward)

    run_mean = mean(rewards)
    run_std = stdev(rewards) if len(rewards) > 1 else 0.0

    print("Episode rewards:", rewards)
    print(f"Average reward: {run_mean:.4f}")
    print(f"Std deviation: {run_std:.4f}")
    print(
        f"Run config -> episodes={episodes}, max_steps={max_steps}, seed={seed}, "
        f"red_agent=B_lineAgent"
    )

    if log_file:
        _append_experiment_log(log_file, episodes, max_steps, seed, rewards)
        print(f"Saved run to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MyBlueHeuristicAgent on Scenario1b.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument("--max-steps", type=int, default=20, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="EXPERIMENT_LOG.csv",
        help="CSV file to append run results. Use an empty string to disable.",
    )
    args = parser.parse_args()

    run(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        log_file=args.log_file.strip(),
    )

