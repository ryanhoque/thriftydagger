import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('./data/oct25-algo-sup/oct25-algo-sup_s0/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
with open('./data/oct25-algo-sup-cont/oct25-algo-sup-cont_s0/metrics.pkl', 'rb') as f:
    metrics_cont = pickle.load(f)

losses = metrics['LossPi'][1:]
losses_cont = metrics_cont['LossPi'][1:]
losses = losses + losses_cont
plt.plot(np.arange(len(losses)), losses, marker='o')
plt.title('Train Loss, Algo Sup')
plt.xlabel('Epoch')
plt.xticks(np.arange(len(losses)))
plt.savefig('il_losses_algo_sup_10epochs.png')