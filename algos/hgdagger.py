import numpy as np
import time
import torch

from algos import BaseAlgorithm
from robosuite.utils.input_utils import input2action
from torch.utils.data import DataLoader


class HGDagger(BaseAlgorithm):
    def __init__(self, model, save_dir, max_traj_length, device, 
                 lr=1e-3, optimizer=torch.optim.Adam, is_ensemble=True) -> None:
        
        super().__init__(model, save_dir, max_traj_length, device, 
                 lr=lr, optimizer=(None if is_ensemble else optimizer))
        
        self.is_ensemble = is_ensemble
        if self.is_ensemble:
            self.ensemble_optimizers = [optimizer(self.model.models[i].parameters(), lr=lr) for i in range(len(self.model.models))]
            
    def _save_checkpoint(self, epoch, best=False):
        # TODO: save checkpoint for ensemble models
        pass
    
    def _switch_mode(self, act, robosuite_cfg=None, env=None):
        # If in robot mode, need to check for input from human
        # A 'Z' keypress (action elem 3) indicates a mode switch
        if act is None:
            for _ in range(10):
                act, _ = input2action(
                    device=robosuite_cfg['input_device'],
                    robot=robosuite_cfg['active_robot'],
                    active_arm=robosuite_cfg['arm'],
                    env_configuration=robosuite_cfg['env_config'])
                env.render()
                time.sleep(0.001)
                if act[3] != 0: # 'Z' is pressed
                    break

        return act[3] != 0


    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            robosuite_cfg['input_device'].start_control()
            # Train policy
            if self.is_ensemble:
                for i, (model, optimizer) in enumerate(zip(self.model.models, self.ensemble_optimizers)): 
                    self.train(model, optimizer, train_loader,val_loader, args)
            else:
                self.train(self.model, self.optimizer, train_loader,val_loader, args)
            
            # Roll out trajectories
            new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout)
            if len(new_data) > 0:
                for (obs, act) in new_data:
                    train_data.update_buffer(obs, act)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        
        # TODO: add more hgdagger-specific metrics (num switches between robot/human), handle ensemble
        self._save_metrics()