import os
import pandas as pd
import pickle
import torch

from collections import defaultdict
from models import Ensemble
from robosuite.utils.input_utils import input2action
from tqdm import tqdm

class BaseAlgorithm:
    def __init__(self, model, model_kwargs, save_dir, max_traj_len, device, 
                 lr=1e-3, optimizer=torch.optim.Adam) -> None:
        self.device = device
        self.lr = lr
        self.max_traj_len = max_traj_len
        self.model = model
        self.model_kwargs = model_kwargs
        self.save_dir = save_dir
    
        self.metrics = defaultdict(list)
        self.model_type = type(model)
        self.optimizer_type = optimizer
        self.is_ensemble = self.model_type == Ensemble
        
        self._setup_optimizer()
        
    
    def _setup_optimizer(self):
        if self.is_ensemble:
            self.optimizers = [self.optimizer_type(self.model.models[i].parameters(), lr=self.lr) for i in range(len(self.model.models))]
        else:
            self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr)
    
    def _reset_model(self):
        self.model = self.model_type(**self.model_kwargs).to(self.device)
    
    
    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, auto_only=False):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, expert_mode, traj_length = env.reset(), False, 0
            success = False
            obs, act = [], []
            while traj_length <= self.max_traj_len and not success:
                obs.append(curr_obs)
                if (expert_mode and not auto_only):
                    # Expert mode (either human or oracle algorithm)
                    a = self._expert_pol(curr_obs, env, robosuite_cfg)
                    # act = np.clip(act, -act_limit, act_limit) TODO: clip actions?
                    if self._switch_mode(act=a) == True:
                        print('Switch to Robot')
                        expert_mode = False
                    next_obs, _, _, _ = env.step(a)
                else:
                    switch_mode = (not auto_only) and self._switch_mode(act=None, robosuite_cfg=robosuite_cfg, env=env)
                    if switch_mode:
                        print('Switch to Expert Mode')
                        expert_mode = True
                        continue
                    a = self.model(curr_obs).to(self.device)
                    next_obs, _, _, _ = env.step(a)
                act.append(a)
                traj_length += 1
                success = env._check_success()
                curr_obs = next_obs
            demo = {'obs': obs, 'act': act, 'success': success}
            data.append(demo)
            env.close()

        return data
    
    def _expert_pol(self, obs, env, robosuite_cfg):
        '''
        Default expert policy: grant control to user
        TODO: should have a default, non-Robosuite policy too?
        '''
        return 0.1 * (obs[2:] - obs[:2]) / torch.norm((obs[2:] - obs[:2]))
        # a = torch.zeros(7)
        # if env.gripper_closed:
        #     a[-1] = 1.
        #     robosuite_cfg['input_device'].grasp = True
        # else:
        #     a[-1] = -1.
        #     robosuite_cfg['input_device'].grasp = False
        # a_ref = a.clone()
        # # pause simulation if there is no user input (instead of recording a no-op)
        # # TODO: make everything torch tensors
        # import numpy as np
        # while np.array_equal(a, a_ref):
        #     a, _ = input2action(
        #         device=robosuite_cfg['input_device'],
        #         robot=robosuite_cfg['active_robot'],
        #         active_arm=robosuite_cfg['arm'],
        #         env_configuration=robosuite_cfg['env_config'])
        #     env.render()
        #     time.sleep(0.001)
        return a
    
    
    def _save_checkpoint(self, epoch, best=False):
        if self.is_ensemble:
            # Save state dict for each model/optimizer
            ckpt_dict = {
                'models': [model.state_dict() for model in self.model.models],
                'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
                'epoch': epoch
            }
        else:
            ckpt_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch
            }
            
        if best:
                ckpt_name = f'model_best_{epoch}.pt'
        else:
            ckpt_name = f'model_{epoch}.pt'
            
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
                obs = obs.to(self.device)
                act = act.to(self.device)
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
            obs = obs.to(self.device)
            act = act.to(self.device)
            pred_act = model(obs)
            loss = torch.mean(torch.sum((act - pred_act)**2, dim=1))
            val_losses.append(loss.item())
        return sum(val_losses) / len(val_losses)
    
    def run(self, train_loader, val_loader, args, env=None, robosuite_cfg=None) -> None:
        raise NotImplementedError
    
    def eval_auto(self, args, env=None, robosuite_cfg=None):
        data = self._rollout(env, robosuite_cfg, args.N_eval_trajectories, auto_only=True)
        successes = [demo['success'] for demo in data]
        save_file = os.path.join(self.save_dir, 'eval_auto_data.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        print(f'Success rate: {sum(successes)/len(successes)}')
        print(f'Eval data saved to {save_file}.')
