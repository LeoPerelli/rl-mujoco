from tqdm import tqdm
from agents import agents_dict
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
import os
from utils import debug_states, evaluate_agent, Tracker
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="/Users/lperelli/rl-mujoco/configs/config.yaml",
    )

    return parser.parse_args()


def main_loop(args):

    with open(args["config_path"], "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"] + str(int(time.time()))
    experiment_dir = f"/Users/lperelli/rl-mujoco/experiments/{experiment_name}"
    os.makedirs(experiment_dir)
    with open(f"{experiment_dir}/config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    envs = gym.make_vec(
        config["env"]["name"],
        num_envs=config["env"]["n"],
        vectorization_mode="sync",
    )
    states, infos = envs.reset(seed=0)

    device = torch.device("mps")
    writer = SummaryWriter(log_dir=f"logs/{experiment_name}")
    tracker = Tracker(
        envs.action_space.shape,
        envs.observation_space.shape,
        config["env"]["policy_update_steps"],
        writer=writer,
    )
    agent = agents_dict[config["agent"]["name"]](
        **{
            **config["agent"]["agent_args"],
            **{
                "state_space": envs.observation_space.shape[1],
                "action_space": envs.action_space.shape[1],
                "tracker": tracker,
            },
        }
    )

    for global_step in tqdm(range(config["env"]["tot_steps"])):

        if (
            tracker.local_step % config["env"]["policy_update_steps"] == 0
            and tracker.local_step > 0
        ):
            agent.update_policy()
            tracker.write_stats(global_step)
            tracker.reset_episode_stats()

        actions = agent.get_actions(states)
        states_, rewards, terminateds, truncateds, infos = envs.step(np.array(actions))
        tracker.update_episode_stats(
            actions, states, states_, rewards, terminateds, truncateds
        )
        tracker.local_step += 1
        states = states_

        if global_step % config["env"]["debug_steps"] == 0 and global_step > 0:
            debug_states(agent, writer, global_step, config["env"]["name"])

        if global_step % config["env"]["eval_steps"] == 0 and global_step > 0:

            print(f"Evaluating agent at step {global_step} and saving model...")
            torch.save(
                agent.actor.state_dict(),
                f"{experiment_dir}/policy_ckpt_{global_step}.pth",
            )
            if agent.critic is not None:
                torch.save(
                    agent.critic.state_dict(),
                    f"{experiment_dir}/critic_ckpt_{global_step}.pth",
                )
            eval_env = gym.make(
                config["env"]["name"],
            )
            avg_reward, avg_length = evaluate_agent(
                agent, eval_env, 5, writer, global_step
            )

    # print(f"Total episodes: {tracker.running_episodes}")


if __name__ == "__main__":

    args = vars(parse_args())
    main_loop(args)


# TODO
# add device
# Advantage normalization (mean 0, std 1 per batch).
# advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
# measure variance and see it is decreased by using advantages
# compute confidence intervals for the runs, compare with baseline paper on number of samples, final performance, etc
# Entropy bonus (encourages exploration).
# any cool findings from looking at the plots? mse prediction error, entropy bonus, etc? losses balanced?
