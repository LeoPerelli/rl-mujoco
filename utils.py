import torch
import numpy as np
import time

DEBUG_STATES = [
    np.array(
        [
            1.24769787e00,
            -4.59026476e-03,
            -4.83472364e-03,
            3.13270239e-03,
            4.12755577e-03,
            1.06635776e-03,
            2.29496561e-03,
            4.36249915e-04,
            4.35072424e-03,
            3.15853554e-03,
            -4.97261500e-03,
        ]
    ),
    np.array(
        [
            1.25450464e00,
            -3.55840387e-03,
            4.48649447e-03,
            -1.88168548e-03,
            -7.66735510e-04,
            3.27702594e-03,
            -9.08008636e-04,
            4.95936877e-04,
            -4.72440887e-03,
            2.53513109e-03,
            3.81433132e-04,
        ]
    ),
]


def debug_states(agent, writer, id):

    with torch.no_grad():

        for enum, s in enumerate(DEBUG_STATES):
            mu, std = torch.softmax(
                agent.actor.forward(s),
                dim=-1,
            )
            writer.add_scalar(
                f"Debug/Actor mu s{enum}",
                mu,
                id,
            )

            if agent.critic is not None:
                value = agent.critic.forward(s)[0]

                writer.add_scalar(
                    f"Debug/Critic s{enum}",
                    value,
                    id,
                )


def evaluate_agent(agent, env, num_episodes, writer, global_step):
    """
    Evaluate the agent's performance with epsilon=0 (no exploration).
    """

    agent.eval_mode = True

    total_rewards = []
    total_lengths = []
    start = time.time()
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.get_actions(state[None]).squeeze(0)
            state_, reward, terminated, truncated, _ = env.step(np.array(action))
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = state_

        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)

    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_length = sum(total_lengths) / len(total_lengths)

    writer.add_scalar("Evaluation/Average Reward", avg_reward, global_step)
    writer.add_scalar("Evaluation/Average Episode Length", avg_length, global_step)
    print(
        f"Evaluation results - Avg reward: {avg_reward:.2f}, Avg episode length: {avg_length:.2f}. Time elapsed: {int(time.time() - start)}"
    )
    agent.eval_mode = False

    return avg_reward, avg_length


class Tracker:

    def __init__(self, action_space, state_space, tot_steps, writer):

        self.action_space = action_space
        self.state_space = state_space
        self.tot_steps = tot_steps
        self.writer = writer
        self.reset_episode_stats()

    def reset_episode_stats(self):
        self.actions = torch.zeros(
            (self.tot_steps, self.action_space[0], self.action_space[1])
        )
        self.states = torch.zeros(
            (self.tot_steps, self.state_space[0], self.state_space[1])
        )
        self.states_ = torch.zeros(
            (self.tot_steps, self.state_space[0], self.state_space[1])
        )
        self.ends = torch.zeros((self.tot_steps, self.action_space[0]))
        self.rewards = torch.zeros((self.tot_steps, self.action_space[0]))
        self.truncateds = torch.zeros((self.tot_steps, self.action_space[0]))
        self.terminateds = torch.zeros((self.tot_steps, self.action_space[0]))
        self.loss = 0
        self.clip_loss = 0
        self.mse_loss = 0
        self.entropy_loss = 0
        self.grad_norm_actor = 0
        self.grad_norm_critic = 0
        self.local_step = 0
        self.log_std = 0

    def update_episode_stats(
        self, actions, states, states_, rewards, terminateds, truncateds
    ):

        self.actions[self.local_step] = actions
        self.states[self.local_step] = torch.from_numpy(states)
        self.states_[self.local_step] = torch.from_numpy(states_)
        self.rewards[self.local_step] = torch.from_numpy(rewards)
        self.terminateds[self.local_step] = torch.from_numpy(terminateds).int()
        self.truncateds[self.local_step] = torch.from_numpy(truncateds).int()

    def write_stats(self, global_step):

        # episode_length, episode_return = self.compute_episode_stats()

        # self.writer.add_scalar("Episode/Return", episode_return, global_step)
        # self.writer.add_scalar("Episode/Length", episode_length, global_step)
        self.writer.add_scalar("Episode/Loss", self.loss, global_step)
        self.writer.add_scalar("Episode/clip_loss", self.clip_loss, global_step)
        self.writer.add_scalar("Episode/mse_loss", self.mse_loss, global_step)
        self.writer.add_scalar("Episode/entropy_loss", self.entropy_loss, global_step)
        self.writer.add_scalar(
            "Episode/Grad Norm actor",
            self.grad_norm_actor,
            global_step,
        )
        self.writer.add_scalar(
            "Episode/Grad Norm critic",
            self.grad_norm_critic,
            global_step,
        )
        self.writer.add_scalar(
            "Episode/Max Log Std",
            self.log_std,
            global_step,
        )

    def compute_episode_stats(self):

        return

        dones = torch.logical_or(self.terminateds, self.truncateds)
        ep_ids = torch.cumsum(dones, dim=0)
        max_id = ep_ids.max(dim=0).item()
        valid_mask = torch.zeros_like(ep_ids)
        valid_mask[ep_ids > 0 & ep_ids < max_id.expand(ep_ids.shape)] = True

        # Aggregate
        ep_returns = torch.zeros(max_id + 1).scatter_add(0, ep_ids, flat_rewards)
        ep_lengths = torch.zeros(max_id + 1).scatter_add(
            0, ep_ids, torch.ones_like(flat_rewards)
        )

        ep_returns = ep_returns[valid_mask]
        ep_lengths = ep_lengths[valid_mask]

        return ep_returns.mean().item(), ep_lengths.float().mean().item()


class RLDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, advantages, return_estimates):
        self.states = states
        self.actions = actions
        self.advantages = advantages
        self.return_estimates = return_estimates

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        out = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "advantages": self.advantages[idx],
            "return_estimates": self.return_estimates[idx],
        }

        return out
