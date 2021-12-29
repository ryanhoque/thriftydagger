import torch

from robosuite.utils.input_utils import input2action
from torch.utils.data import DataLoader

from algos import BaseAlgorithm


class BC(BaseAlgorithm):
    def __init__(self, model, model_kwargs, save_dir, max_traj_len, device, lr=1e-3, optimizer=torch.optim.Adam) -> None:
        super().__init__(model, model_kwargs, save_dir, max_traj_len, device, lr=lr, optimizer=optimizer)

    def run(self, train_data, val_data, args, env=None, robosuite_cfg=None) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        if args.robosuite:
            robosuite_cfg["input_device"].start_control()

        # Train & save metrics
        if self.is_ensemble:
            for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                self.train(model, optimizer, train_loader, val_loader, args)
        else:
            self.train(self.model, self.optimizer, train_loader, val_loader, args)
        self._save_metrics()
