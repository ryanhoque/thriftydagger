import os
import pandas as pd
import time
import torch

from collections import defaultdict
from robosuite.utils.input_utils import input2action
from tqdm import tqdm

class BaseAlgorithm:
    def __init__(self, model, save_dir, max_traj_length, device, 
                 lr=1e-3, optimizer=torch.optim.Adam) -> None:
        self.device = device
        self.max_traj_length = max_traj_length
        self.metrics = defaultdict(list)
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=lr) if optimizer is not None else None
        self.save_dir = save_dir
    
    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, expert_mode, traj_length = env.reset(), False, 0
            success = False
            while traj_length <= self.max_traj_length and not success:
                if expert_mode:
                    # Expert mode (either human or oracle algorithm)
                    act = self._expert_pol(curr_obs, env, robosuite_cfg)
                    # act = np.clip(act, -act_limit, act_limit) TODO: clip actions?
                    data.append((curr_obs, act))
                    if self._switch_mode(act=act) == True:
                        print('Switch to Robot')
                        expert_mode = False
                    next_obs, _, _, _ = env.step(act)
                else:
                    if self._switch_mode(act=None, robosuite_cfg=robosuite_cfg, env=env) == True:
                        print('Switch to Expert Mode')
                        expert_mode = True
                        continue
                    act = self.model(curr_obs)
                    next_obs, _, _, _ = env.step(act)
                traj_length += 1
                # TODO: record + save success rate
                success = env._check_success()
                curr_obs = next_obs
            env.close()

        return data
    
    def _expert_pol(self, obs, env, robosuite_cfg):
        '''
        Default expert policy: grant control to user
        TODO: should have a default, non-Robosuite policy too?
        '''
        a = torch.zeros(7)
        if env.gripper_closed:
            a[-1] = 1.
            robosuite_cfg['input_device'].grasp = True
        else:
            a[-1] = -1.
            robosuite_cfg['input_device'].grasp = False
        a_ref = a.clone()
        # pause simulation if there is no user input (instead of recording a no-op)
        # TODO: make everything torch tensors
        import numpy as np
        while np.array_equal(a, a_ref):
            a, _ = input2action(
                device=robosuite_cfg['input_device'],
                robot=robosuite_cfg['active_robot'],
                active_arm=robosuite_cfg['arm'],
                env_configuration=robosuite_cfg['env_config'])
            env.render()
            time.sleep(0.001)
        return a
    
    
    def _save_checkpoint(self, epoch, best=False):
        if best:
            ckpt_name = f'model_best_{epoch}.pt'
        else:
            ckpt_name = f'model_{epoch}.pt'
        ckpt_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        save_path = os.path.join(self.save_dir, ckpt_name)
        torch.save(ckpt_dict, save_path)
    
    def _save_metrics(self):
        save_path = os.path.join(self.save_dir, 'metrics.pkl')
        df = pd.DataFrame(self.metrics)
        df.to_pickle(save_path)
        
    def _update_metrics(self, **kwargs):
        for key, val in kwargs.items():
            self.metrics[key].append(val)
        
    def train(self, model, optimizer, train_loader, val_loader, args):
        model.train()
        for epoch in range(args.epochs):
            prog_bar = tqdm(train_loader, leave=False)
            prog_bar.set_description(f'Epoch {epoch}/{args.epochs - 1}')    
            epoch_losses = []
            for (obs, act) in prog_bar:
                optimizer.zero_grad()
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                pred_act = model(obs)
                loss = torch.mean(torch.sum((act - pred_act)**2, dim=1))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                prog_bar.set_postfix(train_loss=loss.item())
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_loss = self.validate(model, val_loader)

            # TODO wandb or tensorboard
            print(f'Epoch {epoch} Train Loss: {avg_loss}')
            print(f'Epoch {epoch} Val Loss: {avg_val_loss}')
            
            # Update metrics
            self._update_metrics(epoch=epoch, train_loss=avg_loss, val_loss=avg_val_loss)
            
            if epoch % args.save_iter == 0 or epoch == args.epochs - 1:
                self._save_checkpoint(epoch)
    
    def validate(self, model, val_loader):
        model.eval()
        val_losses = []
        for (obs, act) in val_loader:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            pred_act = model(obs)
            loss = torch.mean(torch.sum((act - pred_act)**2, dim=1))
            val_losses.append(loss.item())
        return sum(val_losses) / len(val_losses)
    
    def run(self, train_loader, val_loader, args, env=None, robosuite_cfg=None) -> None:
        raise NotImplementedError