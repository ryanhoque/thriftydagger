import numpy as np
import time
import torch

from algos import BaseAlgorithm
from models import MLP
from robosuite.utils.input_utils import input2action
from torch.utils.data import DataLoader


class Dagger(BaseAlgorithm):
    def __init__(self, model, model_kwargs, save_dir, max_traj_len, device, 
                 lr=1e-3, optimizer=torch.optim.Adam, beta=0.9) -> None:
        
        super().__init__(model, model_kwargs, save_dir, max_traj_len, device, 
                 lr=lr, optimizer=optimizer)
        
        self.beta = beta
         
    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, auto_only=False):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, traj_length = env.reset(), 0
            success = False
            obs, act = [], []
            while traj_length <= self.max_traj_len:
                if not auto_only:
                    a_target = self._expert_pol(curr_obs, env, robosuite_cfg).detach()
                    a = self.beta * a_target + (1 - self.beta) * self.model(curr_obs).detach()
                else:
                    a = self.model(curr_obs).detach()
                next_obs, _, _, _ = env.step(a)
                obs.append(curr_obs)
                act.append(a_target)
                traj_length += 1
                if not success:
                    success = env._check_success()
                curr_obs = next_obs
            demo = {'obs': obs, 'act': act, 'success': success}
            data.append(demo)
            env.close()

        return data

    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            if args.robosuite:
                robosuite_cfg['input_device'].start_control()
            # Train policy
            if self.is_ensemble:
                for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)): 
                    self.train(model, optimizer, train_loader,val_loader, args)
            else:
                self.train(self.model, self.optimizer, train_loader,val_loader, args)
            
            # Roll out trajectories
            new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout)
            if len(new_data) > 0:
                for demo in new_data:
                    for (obs, act) in zip(demo['obs'], demo['act']):
                        train_data.update_buffer(obs, act)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            
            # Reset the model and optimizer for retraining
            self._reset_model()
            self._setup_optimizer()
            
            # Beta decays exponentially
            self.beta *= self.beta
            
        # TODO: add more dagger-specific metrics (num switches between robot/human), handle ensemble
        self._save_metrics()