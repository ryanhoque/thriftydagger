import matplotlib.pyplot as plt
import pickle
import numpy as np

# exp_name = 'dec18_reach2d_data_1k_bc_only_eval'
# exp_name = 'dec18_reach2d_oracle_pi_r_mix_1000_bc_only_fixed'
exp_name = 'dec19_reach2d_data_1k_bc_only_eval_linear_model'

metrics = pickle.load(open(f'./data/{exp_name}/{exp_name}_s4/test0.pkl', 'rb'))

obs = metrics['obs']
rew = metrics['rew']


def plot_rollouts():
    for i, (traj_o, traj_success) in enumerate(zip(obs, rew)):
        state_xs = []
        state_ys = []
        if not traj_success:
            goal = traj_o[0][2:]
            color = 'red'
            for state in traj_o:
                state_xs.append(state[0])
                state_ys.append(state[1])
                if np.linalg.norm(state[:2] - goal) <= 0.11:
                    # color = 'blue'
                    print(i)
            plt.clf()
            plt.xlim(0, 4)
            plt.ylim(0, 4)
            plt.plot(state_xs, state_ys, color='green')
            plt.scatter([goal[0]], [goal[1]], color=color, marker='*')
            plt.savefig(f'figures/linear_model_failure_rollout_{i}.png')

def plot_goals():
    for i, (traj_o, traj_success) in enumerate(zip(obs, rew)):
        goal = traj_o[0][2:]
        
        if not traj_success:
            color = 'red'
        else:
            color = 'green'
        plt.scatter([goal[0]], [goal[1]], color=color, marker='*')
    plt.savefig(f'figures/bc_only_goals_rollout_not_shuffled.png')
    
plot_rollouts()