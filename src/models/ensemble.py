# from models import MLP 

import numpy as np
import torch
import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, model_kwargs, device, num_models=5, model_type=None) -> None:
        super().__init__()
        self.num_models = num_models
        self.device = device
        self.models = [model_type(**model_kwargs).to(device) for _ in range(num_models)]
    def forward(self, obs):
        # obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # No grad since we don't want to backpropagate over taking the average of the ensemble
        with torch.no_grad():
            acts = []
            for model in self.models:
                acts.append(model(obs).detach())
            return torch.mean(torch.stack(acts), dim=0)
