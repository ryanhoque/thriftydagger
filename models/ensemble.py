# from models import MLP 

import numpy as np
import torch
import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, model_args, device, num_models=5, model_arch=None) -> None:
        super().__init__()
        self.num_models = num_models
        self.device = device
        print(model_args)
        self.models = [model_arch(*model_args).to(device) for _ in range(num_models)]
    
    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # No grad since we don't want to backpropagate over taking the average of the ensemble
        with torch.no_grad():
            acts = []
            # TODO: currently all datasets consist of numpy arrays so we have to do this conversion;
            #       should be saving data as tensors
            for model in self.models:
                acts.append(model(obs).cpu().numpy())
            return np.mean(np.array(acts), axis=0)