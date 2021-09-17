import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)

class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h, w, c = obs_dim
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            #nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, act_dim),
            nn.Tanh() # squash to [-1,1]
        )

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        return self.model(obs)

class CNNQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        h, w, c = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            #nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 + act_dim, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
            nn.Sigmoid() # squash to [0,1]
        )

    def forward(self, obs, act):
        obs = obs.permute(0, 3, 1, 2)
        obs = self.conv(obs)
        q = self.linear(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class MLPClassifier(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Sigmoid)
        self.device = device

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.pi(obs).to(self.device)

class MLPQFunction(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, nn.Sigmoid)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLP(nn.Module):

    def __init__(self, observation_space, action_space, device, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.pi_safe = MLPClassifier(obs_dim, 1, (128,128), activation, device).to(device)
        self.device = device

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

    def classify(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi_safe(obs).cpu().numpy().squeeze()

class Ensemble(nn.Module):
    # Multiple policies
    def __init__(self, observation_space, action_space, device, hidden_sizes=(256,256),
                 activation=nn.ReLU, num_nets=5):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device) for _ in range(num_nets)]
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0: # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0)

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs, act):
        # closer to 1 indicates more safe.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(torch.min(self.q1(obs, act), self.q2(obs,act)).cpu().numpy())

class EnsembleCNN(nn.Module):
    # Multiple policies with image input
    def __init__(self, observation_space, action_space, device, num_nets=5):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [CNNActor(obs_dim, act_dim).to(device) for _ in range(num_nets)]
        self.q1 = CNNQFunction(obs_dim, act_dim).to(device)
        self.q2 = CNNQFunction(obs_dim, act_dim).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            if i >= 0: # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0).squeeze()

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs, act):
        # closer to 1 indicates more safe.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, 0)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        if len(act.shape) == 1:
            act = torch.unsqueeze(act, 0)
        with torch.no_grad():
            return float(torch.min(self.q1(obs, act), self.q2(obs,act)).cpu().numpy())

