# agent.state_encoder.load_state_dict(torch.load("dqn_model.pth"))
# agent.state_encoder.to(device)

from tqdm import tqdm
from agents import agents_dict
from training import evaluate_agent
import gymnasium as gym
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

experiment_dir = "/Users/lperelli/rl-mujoco/experiments/ppo_base_1759333187"
# step = 413696
step = 997376
# step = 94208

critic_path = experiment_dir + f"/critic_ckpt_{step}.pth"
policy_path = experiment_dir + f"/policy_ckpt_{step}.pth"

with open(f"{experiment_dir}/config.yaml", "r") as f:
    config = yaml.safe_load(f)

eval_episodes = 5

eval_env = gym.make(config["env"]["name"], render_mode="human")
agent = agents_dict[config["agent"]["name"]](
    **{
        **config["agent"]["agent_args"],
        **{
            "state_space": eval_env.observation_space.shape[0],
            "action_space": eval_env.action_space.shape[0],
            "tracker": None,
        },
    }
)
agent.actor.load_state_dict(torch.load(policy_path))
agent.critic.load_state_dict(torch.load(critic_path))

avg_reward, avg_length = evaluate_agent(agent, eval_env, eval_episodes)
