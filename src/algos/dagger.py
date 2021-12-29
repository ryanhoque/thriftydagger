import numpy as np
import torch
from torch.utils.data import DataLoader

from algos import BaseAlgorithm


class Dagger(BaseAlgorithm):
    def __init__(
        self,
        model,
        model_kwargs,
        save_dir,
        max_traj_len,
        device,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        beta=0.9,
        use_indicator_beta=False,
        max_num_labels=1000,
    ) -> None:

        super().__init__(model, model_kwargs, save_dir, max_traj_len, device, lr=lr, optimizer=optimizer)

        self.use_indicator_beta = use_indicator_beta
        self.beta = self._init_beta(beta)
        self.max_num_labels = max_num_labels
        self.num_labels = 0

    def _init_beta(self, beta):
        if self.use_indicator_beta and (beta != 1.0):
            raise ValueError(f"If use_indicator_beta is True, beta must be 1.0, but got beta={beta}!")
        return beta

    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, auto_only=False):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, traj_length = env.reset(), 0
            success = False
            obs, act = [], []
            while traj_length <= self.max_traj_len and self.num_labels < self.max_num_labels:
                if not auto_only:
                    a_target = self._expert_pol(curr_obs, env, robosuite_cfg).detach()
                    a = self.beta * a_target + (1 - self.beta) * self.model(curr_obs).detach()
                    self.num_labels += 1
                else:
                    a = self.model(curr_obs).detach()
                next_obs, _, _, _ = env.step(a)
                obs.append(curr_obs)
                act.append(a_target)
                traj_length += 1
                if not success:
                    success = env._check_success()
                curr_obs = next_obs
            demo = {"obs": obs, "act": act, "success": success}
            data.append(demo)
            env.close()

            if self.num_labels < self.max_num_labels:
                break

        return data

    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        # For DAgger, initial training dataset should be empty before first iteration
        train_data.clear_buffer()
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            if args.robosuite:
                robosuite_cfg["input_device"].start_control()

            # Roll out trajectories
            new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout)
            if len(new_data) > 0:
                for demo in new_data:
                    for (obs, act) in zip(demo["obs"], demo["act"]):
                        train_data.update_buffer(obs, act)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

            # If max number of expert labels exceeded, break
            if self.num_labels > self.max_num_labels:
                break

            # Retrain policy
            if self.is_ensemble:
                for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                    self.train(model, optimizer, train_loader, val_loader, args)
            else:
                self.train(self.model, self.optimizer, train_loader, val_loader, args)

            # Reset the model and optimizer for retraining
            self._reset_model()
            self._setup_optimizer()

            if self.use_indicator_beta:
                # Set beta to 0 after first iteration
                if epoch == 0:
                    self.beta = 0
            else:
                # Beta decays exponentially
                self.beta *= self.beta

        # TODO: add more dagger-specific metrics (num switches between robot/human), handle ensemble
        self._save_metrics()
