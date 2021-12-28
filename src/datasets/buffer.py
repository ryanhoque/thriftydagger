from typing import Tuple
from constants import MAX_BUFFER_SIZE
from torch.utils.data import Dataset

import numpy as np
import torch


class BufferDataset(Dataset):
    def __init__(self, data, max_size=MAX_BUFFER_SIZE, shuffle=True, split='train', train_perc=0.9) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.obs, self.act, self.curr_size = self.split_data(data, split, 
                                                             train_perc, shuffle=shuffle)

    def split_data(self, data, split, train_perc, shuffle) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data['obs']
        act = data['act']
        num_train = min(int(train_perc * len(obs)), self.max_size)
        if split == 'train':
            obs = obs[:num_train]
            act = act[:num_train]
            curr_size = len(obs)
        elif split == 'val':
            obs = obs[num_train:]
            act = act[num_train:]
            curr_size = len(obs)
        else:
            raise ValueError(
                f'split must be one of {{\'train\', \'val\'}}; got split=\'{split}\' instead!')

        if shuffle:
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            obs = obs[idxs]
            act = act[idxs]

        obs = torch.cat(
            [obs, torch.zeros(max(self.max_size - len(obs), self.max_size), obs.shape[1])])
        act = torch.cat(
            [act, torch.zeros(max(self.max_size - len(act), self.max_size), act.shape[1])])

        return obs, act, curr_size

    def update_buffer(self, obs, act) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size+1, self.max_size)

    def __len__(self) -> int:
        return self.curr_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (state (i.e., observation), action) tuple
        return (self.obs[idx], self.act[idx])
