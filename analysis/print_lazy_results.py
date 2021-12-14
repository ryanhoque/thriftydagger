import pickle
import torch

TAU_SUPS = [0.032, 0.016, 0.008, 0.004, 0.002]
TAU_AUTOS = [0.01, 0.1, 0.25, 0.5, 1.0]

for tau_sup in TAU_SUPS:    
    print('=' * 30)
    print(f'tau_sup={tau_sup}')
    for tau_auto in TAU_AUTOS:
        train_exp_name = f'nov27_lazydagger_pick_place_tau_sup{tau_sup}_tau_auto{tau_auto}'
        eval_exp_name = f'nov28_lazydagger_auto_only_eval_pick_place_tau_sup{tau_sup}_tau_auto{tau_auto}'
        with open(f'/iliad/u/madeline/thriftydagger/data/{train_exp_name}/{train_exp_name}_s4/metrics.pkl', 'rb') as f:
            train_metrics = pickle.load(f)
        with open(f'/iliad/u/madeline/thriftydagger/data/{eval_exp_name}/{eval_exp_name}_s4/test0.pkl', 'rb') as f:
            eval_metrics = pickle.load(f)
        online_burden = train_metrics['OnlineBurden'][-1]
        success_rate = sum(eval_metrics['rew'])/100
        print(f'\ttau_auto={tau_auto}')
        print(f'\t\tSuccess rate: {success_rate}')
        print(f'\t\tOnline burden: {online_burden}')
        
