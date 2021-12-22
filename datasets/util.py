import numpy as np
import pickle
import torch

def load_data_from_file(file_path, shuffle=True):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    obs = []
    act = []
    for demo in data:
        obs.extend(demo['obs'])
        act.extend(demo['act'])
    data = {'obs': torch.stack(obs), 'act': torch.stack(act)}

    if shuffle:
        idxs = torch.randperm(len(data['obs']))
    else:
        idxs = torch.arange(len(data['obs']))
    data['obs'] = data['obs'][idxs]
    data['act'] = data['act'][idxs]

    return data
