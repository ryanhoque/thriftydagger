import numpy as np
import pickle

def load_data_from_file(file_path, shuffle=True):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    # TODO: fix data saving/loading so they're all the same format
    obs = []
    act = []
    # for demo in data:
    #     for (s, a, g) in demo:
    #         obs.append(np.concatenate([s, g]))
    #         act.append(a)
    # data = {'obs': np.array(obs), 'act': np.array(act)}
    idxs = np.arange(len(data['obs']))

    if shuffle:
        np.random.shuffle(idxs)

    data['obs'] = data['obs'][idxs]
    data['act'] = data['act'][idxs]

    return data
