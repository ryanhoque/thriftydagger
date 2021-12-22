import matplotlib.pyplot as plt
import pickle
import numpy as np

# data = np.array(pickle.load(open('./data/dec18_gen_pi_r_reach2d_data_1k_fixed/dec18_gen_pi_r_reach2d_data_1k_fixed_s4/reach2d_pi_r-1000.pkl', 'rb')))
# data = np.array(pickle.load(open('./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl', 'rb')))
data = np.array(pickle.load(open('combined_pickplace_data.pkl', 'rb')))

idxs = np.arange(len(data))
np.random.shuffle(idxs)
data = data[idxs[:10]]
print(len(data))
n_states = 0
for i, demo in enumerate(data):
    goal = demo[0][2]
    plt.scatter([goal[0]], [goal[1]], color='red')
    state_xs = []
    state_ys = []
    for (state, action, _) in demo:
        state_xs.append(state[0])
        state_ys.append(state[1])
    plt.scatter(state_xs, state_ys)
n_states += len(state_xs)
plt.savefig(f'figures/combined_10.png')
