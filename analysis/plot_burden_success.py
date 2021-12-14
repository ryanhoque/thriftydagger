import matplotlib.pyplot as plt
import numpy as np
import pickle

METHODS = ['hgdagger', 'lazydagger', 'thriftydagger']
HG_THRESHOLDS = [0.2, 0.5, 1.0, 2.0, 5.0]
LAZY_TAU_SUPS = [0.032, 0.016, 0.008, 0.004, 0.002]
LAZY_TAU_AUTOS = [0.01, 0.1, 0.25, 0.5, 1.0]
THRIFTY_ALPHAS = [0.01, 0.1, 0.25, 0.5, 1.0]


def get_hg_metrics():
    online_burdens = []
    success_rates = []
    for thresh in HG_THRESHOLDS:
        train_exp = f'nov27_hgdagger_pick_place_threshold{thresh}' 
        eval_exp = f'nov28_hgdagger_auto_only_eval_pick_place_threshold{thresh}'

        with open(f'/iliad/u/madeline/thriftydagger/data/{train_exp}/{train_exp}_s4/metrics.pkl', 'rb') as f:
            train_metrics = pickle.load(f)
            online_burdens.append(train_metrics['OnlineBurden'][-1])
        with open(f'/iliad/u/madeline/thriftydagger/data/{eval_exp}/{eval_exp}_s4/test0.pkl', 'rb') as f:
            eval_metrics = pickle.load(f)
            success_rates.append(sum(eval_metrics['rew'])/100)

    return online_burdens, success_rates

def get_lazy_metrics():
    online_burdens = []
    success_rates = []
    for tau_sup in LAZY_TAU_SUPS:
        for tau_auto in LAZY_TAU_AUTOS:
            train_exp = f'nov27_lazydagger_pick_place_tau_sup{tau_sup}_tau_auto{tau_auto}' 
            eval_exp = f'nov28_lazydagger_auto_only_eval_pick_place_tau_sup{tau_sup}_tau_auto{tau_auto}'

            with open(f'/iliad/u/madeline/thriftydagger/data/{train_exp}/{train_exp}_s4/metrics.pkl', 'rb') as f:
                train_metrics = pickle.load(f)
                online_burdens.append(train_metrics['OnlineBurden'][-1])
            with open(f'/iliad/u/madeline/thriftydagger/data/{eval_exp}/{eval_exp}_s4/test0.pkl', 'rb') as f:
                eval_metrics = pickle.load(f)
                success_rates.append(sum(eval_metrics['rew'])/100)

    return online_burdens, success_rates

def get_thrifty_metrics():
    online_burdens = []
    success_rates = []
    for thresh in THRIFTY_ALPHAS:
        train_exp = f'nov27_thriftydagger_pick_place_alpha{thresh}' 
        eval_exp = f'nov28_thriftydagger_auto_only_eval_pick_place_alpha{thresh}'

        with open(f'/iliad/u/madeline/thriftydagger/data/{train_exp}/{train_exp}_s4/metrics.pkl', 'rb') as f:
            train_metrics = pickle.load(f)
            online_burdens.append(train_metrics['OnlineBurden'][-1])
        with open(f'/iliad/u/madeline/thriftydagger/data/{eval_exp}/{eval_exp}_s4/test0.pkl', 'rb') as f:
            eval_metrics = pickle.load(f)
            success_rates.append(sum(eval_metrics['rew'])/100)

    return online_burdens, success_rates

def get_method_metrics(method):
    if method == 'hgdagger':
        return get_hg_metrics()
    elif method == 'lazydagger':
        return get_lazy_metrics()
    elif method == 'thriftydagger':
        return get_thrifty_metrics()
    else:
        raise NotImplementedError(f'Method {method} not implemented yet!')
if __name__ == '__main__':
    for method in METHODS:
        online_burdens, success_rates = get_method_metrics(method)
        idxs = np.argsort(online_burdens)
        success_rates = np.array(success_rates)[idxs]
        online_burdens = np.array(online_burdens)[idxs]
        plt.plot(online_burdens, success_rates, marker='o', label=method)
    plt.title('Interventions vs. Autonomous-Only Success Rate')
    plt.xlabel('# of expert labels')
    plt.ylabel('Success rate')
    plt.legend()
    plt.savefig('plot.png')
