from datasets.util import load_data_from_file
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch

MAX_BUFFER_SIZE = int(3e4)

class BufferDataset(Dataset):
    def __init__(self, data, max_size=MAX_BUFFER_SIZE, shuffle=True, split='train', train_perc=0.9) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.obs, self.act = self.split_data(
            data, split, train_perc, shuffle=shuffle)
        self.curr_size = len(self.obs)

    def split_data(self, data, split, train_perc, shuffle) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data['obs']
        act = data['act']
        num_train = min(int(train_perc * len(obs)), self.max_size)
        if split == 'train':
            obs = obs[:num_train]
            act = act[:num_train]
        elif split == 'val':
            obs = obs[num_train:]
            act = act[num_train:]
        else:
            raise ValueError(
                f'split must be one of {{\'train\', \'val\'}}; got split=\'{split}\' instead!')

        if shuffle:
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            obs = obs[idxs]
            act = act[idxs]
        return obs, act

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



def get_dataset(data_path):
    data = load_data_from_file(data_path)
    train = BufferDataset(data, split='train')
    val = BufferDataset(data, split='val')
    
    return train, val