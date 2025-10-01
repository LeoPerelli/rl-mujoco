import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_layers=(64, 64),
        network_type="actor",
        activation=nn.ReLU,
        start_log_std=0,
    ):
        super().__init__()
        self.network_type = network_type
        if self.network_type not in ["actor", "critic"]:
            raise ValueError(f"Unknown network type {self.network_type}")
        layers = []
        last_size = obs_dim
        for h in hidden_layers:
            layers += [nn.Linear(last_size, h), activation()]
            last_size = h

        if self.network_type == "critic":
            layers += [nn.Linear(last_size, 1)]

        self.mlp = nn.Sequential(*layers)

        if self.network_type == "actor":
            self.mu = nn.Linear(last_size, act_dim)
            self.log_std = nn.Parameter(start_log_std * torch.ones(act_dim))

    def forward(self, obs):
        x = self.mlp(obs)
        if self.network_type == "actor":
            mu = self.mu(x)
            return mu, torch.exp(torch.clip(self.log_std, min=-20, max=0.5))
        else:
            return x.squeeze(-1)
