import os
import pickle

import matplotlib.pyplot as plt


NS = [50, 100, 200, 300, 400, 500, 750, 1000]
parent_dir = "./out/dec23/"

successes = []
for N in NS:
    exp_dir = os.path.join(parent_dir, f"oracle_reach2d_mlp_eval_N{N}")
    data_file = os.path.join(exp_dir, "eval_auto_data.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    num_successes = sum([traj["success"] for traj in data])
    successes.append(num_successes)
plt.plot(NS, successes, marker="o")
plt.xlabel("N")
plt.ylabel("# successes")
plt.title("Autonomous-only rollout: N vs. # successes")
plt.savefig("successes_vs_N.png")
