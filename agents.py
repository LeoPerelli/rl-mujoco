import numpy as np
import torch
from models import Regressor
from utils import RLDataset
from torch.distributions import Categorical

EPS = 1e-4


class PPOBaseline:

    def __init__(
        self,
        eps,
        batch_size,
        epochs,
        actor_lr,
        critic_lr,
        gamma,
        lamda,
        state_space,
        action_space,
        actor_hidden_layers,
        critic_hidden_layers,
        tracker,
        c_v=1.0,
        c_e=1.0,
        normalise_advantages=False,
    ):

        self.actor = Regressor(
            act_dim=action_space, obs_dim=state_space, hidden_layers=actor_hidden_layers
        )
        self.old_actor = Regressor(
            act_dim=action_space, obs_dim=state_space, hidden_layers=actor_hidden_layers
        )
        self.critic = Regressor(
            act_dim=action_space,
            obs_dim=state_space,
            hidden_layers=critic_hidden_layers,
            network_type="critic",
        )
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lamda = lamda
        self.eps = eps
        self.batch_size = batch_size
        self.epochs = epochs
        self.mse = torch.nn.MSELoss()
        self.c_v = c_v
        self.c_e = c_e
        self.tracker = tracker
        self.eval_mode = False
        self.normalise_advantages = normalise_advantages

    def update_policy(self):

        A, V_t = self.compute_GAE(self.tracker)

        states = [t for tt in self.tracker.states.unbind() for t in tt]
        actions = [t for tt in self.tracker.actions.unbind() for t in tt]
        if self.normalise_advantages == "batch":
            A_norm = (A - A.mean()) / (A.std() + 1e-8)
            advantages = [t for tt in A_norm.unbind() for t in tt]
        else:
            advantages = [t for tt in A.unbind() for t in tt]
        return_estimates = [t for tt in (A + V_t).unbind() for t in tt]

        dataset = RLDataset(states, actions, advantages, return_estimates)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.old_actor.load_state_dict(self.actor.state_dict())

        self.ppo_optimization_loop(dataloader)

    def ppo_optimization_loop(self, dataloader):

        running_loss = 0.0
        running_clip_loss = 0.0
        running_mse_loss = 0.0
        running_entropy_loss = 0.0
        tot_steps = 0
        grad_norm_actor = 0
        grad_norm_critic = 0
        for _ in range(self.epochs):
            for i, batch in enumerate(dataloader):

                if self.normalise_advantages == "minibatch":
                    batch["advantages"] = (
                        batch["advantages"] - batch["advantages"].mean()
                    ) / (batch["advantages"].std() + 1e-8)

                self.actor_optimiser.zero_grad()
                self.critic_optimiser.zero_grad()

                with torch.no_grad():
                    old_mus, old_stds = self.old_actor.forward(batch["states"])
                    old_action_logprobs = self.get_logprob(
                        old_mus, old_stds, batch["actions"]
                    )

                mus, stds = self.actor.forward(batch["states"])
                action_logprobs = self.get_logprob(mus, stds, batch["actions"])
                state_values = self.critic.forward(batch["states"])

                loss, clip_loss, mse_loss, entropy_loss = self.loss_fn(
                    action_logprobs,
                    batch["advantages"],
                    old_action_logprobs,
                    state_values,
                    batch["return_estimates"],
                    mus,
                    stds,
                )
                loss.backward()

                for p in self.actor.parameters():
                    grad_norm_actor += torch.linalg.vector_norm(p.grad)

                for p in self.critic.parameters():
                    grad_norm_critic += torch.linalg.vector_norm(p.grad)

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.actor_optimiser.step()
                self.critic_optimiser.step()

                running_loss += loss.item()
                running_clip_loss += clip_loss.item()
                running_mse_loss += mse_loss.item()
                running_entropy_loss += entropy_loss.item()
                tot_steps += 1

        self.tracker.loss = running_loss / tot_steps
        self.tracker.clip_loss = running_clip_loss / tot_steps
        self.tracker.mse_loss = running_mse_loss / tot_steps
        self.tracker.entropy_loss = running_entropy_loss / tot_steps
        self.tracker.grad_norm_actor = grad_norm_actor / tot_steps
        self.tracker.grad_norm_critic = grad_norm_critic / tot_steps
        self.tracker.log_std = self.actor.log_std.max()

    def loss_fn(
        self,
        action_logprobs,
        advantages,
        old_action_logprobs,
        state_values,
        return_estimates,
        mus,
        stds,
    ):

        mse_loss = self.mse(return_estimates, state_values)
        r = torch.exp(action_logprobs - old_action_logprobs)
        clip_loss = -torch.min(
            r * advantages.detach().unsqueeze(1),
            advantages.detach().unsqueeze(1)
            * torch.clip(r, 1 - self.eps, 1 + self.eps),
        ).mean()
        entropy_loss = (
            torch.distributions.Normal(mus, stds).entropy().sum(dim=-1).mean()
        )

        return (
            clip_loss + self.c_v * mse_loss - self.c_e * entropy_loss,
            clip_loss,
            mse_loss,
            entropy_loss,
        )

    def compute_GAE(self, tracker):
        with torch.no_grad():
            V_t = self.critic.forward(tracker.states)  # T x n_envs
            V_t_ = self.critic.forward(tracker.states_)
        end_mask = 1 - tracker.terminateds
        D = tracker.rewards + self.gamma * V_t_ * end_mask - V_t
        gae_vec = torch.zeros(V_t.shape[1])  # should be the batch size
        A = torch.zeros_like(D)

        # i go back through the T timesteps
        for i in reversed(range(D.shape[0])):
            gae_vec = (
                gae_vec * (1 - tracker.terminateds[i]) * (1 - tracker.truncateds[i])
            )  # applies a mask to stop the accumulation at the boundaries of episodes. since i go backwards in time, the flag is true for the last time step of the episode, hence i must apply the mask before updating the advantage, rather than after.
            gae_vec = self.gamma * self.lamda * gae_vec + D[i]
            A[i] = gae_vec

        return A, V_t

    def get_logprob(self, mus, stds, actions):

        actions_x = torch.atanh(torch.clip(actions, -1 + EPS, 1 - EPS))
        logprob_actions_x = (
            torch.distributions.Normal(mus, stds).log_prob(actions_x).sum(dim=-1)
        )
        modifier = 1 - torch.clip(torch.tanh(actions_x), -1 + EPS, 1 - EPS).square()
        logprob_actions = logprob_actions_x - torch.log(modifier + EPS).sum(dim=-1)
        # minus because its the derivative of the inverse, which for tanh is 1/(1-x^2)

        return logprob_actions

    def get_actions(self, states):

        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        with torch.no_grad():
            mus, stds = self.actor.forward(states)
            if self.eval_mode:
                actions = torch.tanh(mus)
            else:
                actions_x = torch.distributions.Normal(mus, stds).sample()
                actions = torch.tanh(actions_x)

        return torch.clip(actions, -1 + EPS, 1 - EPS)


agents_dict = {"PPOBaseline": PPOBaseline}
