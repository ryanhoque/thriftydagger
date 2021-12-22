import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, obs_dim, act_dim, use_bias=False):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim, bias=use_bias)

    def forward(self, obs):
        return self.linear(obs)
