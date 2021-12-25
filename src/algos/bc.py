from algos import BaseAlgorithm
# from collections import defaultdict
from robosuite.utils.input_utils import input2action
from torch.utils.data import DataLoader

# from tqdm import tqdm

# import os
# import pandas as pd
import torch


class BC(BaseAlgorithm):
    def __init__(self, model, model_kwargs, save_dir, max_traj_len, device, 
                 lr=1e-3, optimizer=torch.optim.Adam) -> None:
        super().__init__(model, model_kwargs, save_dir, max_traj_len, device, 
                 lr=lr, optimizer=optimizer)
    def run(self, train_data, val_data, args, env=None, robosuite_cfg=None) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        if args.robosuite:
            robosuite_cfg['input_device'].start_control()
        # Train & save metrics
        self.train(self.model, self.optimizer, train_loader, val_loader, args)
        self._save_metrics()