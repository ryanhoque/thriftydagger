import time

import numpy as np
import torch
from robosuite.utils.input_utils import input2action
from torch.utils.data import DataLoader


class ThriftyDagger:
    def __init__(self, ensemble, max_traj_len, device, ensemble_lr=1e-3) -> None:
        self.ensemble = ensemble
        self.optimizers = [
            torch.optim.Adam(self.ensemble.models[i].parameters(), lr=ensemble_lr)
            for i in range(len(self.ensemble.models))
        ]
        self.max_traj_len = max_traj_len
        self.device = device

    def expert_pol(self, obs, env, robosuite_cfg):
        a = np.zeros(7)
        if env.gripper_closed:
            a[-1] = 1.0
            robosuite_cfg["input_device"].grasp = True
        else:
            a[-1] = -1.0
            robosuite_cfg["input_device"].grasp = False
        a_ref = a.clone()
        # pause simulation if there is no user input (instead of recording a no-op)
        while np.array_equal(a, a_ref):
            a, _ = input2action(
                device=robosuite_cfg["input_device"],
                robot=robosuite_cfg["active_robot"],
                active_arm=robosuite_cfg["arm"],
                env_configuration=robosuite_cfg["env_config"],
            )
            env.render()
            time.sleep(0.001)
        return a

    def train(self, train_loader, val_loader, args):
        for i, (model, optimizer) in enumerate(zip(self.ensemble.models, self.optimizers)):
            model.train()
            for epoch in range(args.policy_train_epochs):
                epoch_losses = []
                for (obs, act) in train_loader:
                    optimizer.zero_grad()
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    pred_act = model(obs)
                    loss = torch.mean(torch.sum((act - pred_act) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                avg_val_loss = self.validate(model, val_loader)
                print(f"Epoch {epoch} Train Loss for Model #{i}: {avg_loss}")
                print(f"Epoch {epoch} Val Loss for Model #{i}: {avg_val_loss}")

    def train_risk(self):
        pass

    def validate(self, model, val_loader):
        model.eval()
        val_losses = []
        for (obs, act) in val_loader:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            pred_act = model(obs)
            loss = torch.mean(torch.sum((act - pred_act) ** 2, dim=1))
            val_losses.append(loss)
        return sum(val_losses) / len(val_losses)

    def switch_mode(self, act, robosuite_cfg=None, env=None):
        if robot_mode:
            if self.ensemble.variance(o) > NOVELTY_THRESH:
                return True
            if self.ensemble.safety(o, a) < RISK_THRESH:
                return True
        else:
            if sum((a - a_expert) ** 2) < SWITCH2ROBOT_THRESH and ac.safety(o, a) > SWITCH2ROBOT_RISK_THRESH:
                return True
        return False

    def rollout(self, env, robosuite_cfg, trajectories_per_rollout):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, d, expert_mode, traj_length = env.reset(), False, False, 0
            success = False
            while traj_length <= self.max_traj_len and not success:

                if expert_mode:
                    # Expert mode (either human or oracle algorithm)
                    act = self._expert_pol(curr_obs, env, robosuite_cfg)
                    # act = np.clip(act, -act_limit, act_limit) TODO: clip actions
                    data.append((curr_obs, act))
                    if self._switch_mode(act=act) == True:
                        print("Switch to Robot")
                        expert_mode = False
                    next_obs, _, done, _ = env.step(act)
                else:
                    if self._switch_mode(act=None, robosuite_cfg=robosuite_cfg, env=env) == True:
                        print("Switch to Expert Mode")
                        expert_mode = True
                        continue
                    act = self.ensemble(curr_obs)
                    next_obs, _, done, _ = env.step(act)
                traj_length += 1
                success = env._check_success()
                curr_obs = next_obs
            env.close()

        return data

    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            robosuite_cfg["input_device"].start_control()
            # Train policy
            train_losses = self.train(train_loader, val_loader, args)
            self.train_risk()

            # Roll out trajectories
            new_data = self.rollout(env, robosuite_cfg, args.trajectories_per_rollout)
            if len(new_data) > 0:
                for (obs, act) in new_data:
                    train_data.update_buffer(obs, act)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
