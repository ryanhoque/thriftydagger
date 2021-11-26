import matplotlib.pyplot as plt
import numpy as np
import pickle


beta_Hs, beta_Rs, eps_Hs, eps_Rs = [], [], [], []

for i in range(1, 5+1):
    with open(f'./data/oct25-retrain/oct25-retrain_s0/iter{i}.pkl', 'rb') as f:
        thresholds = pickle.load(f)[0]
        beta_H = 1 - np.array(thresholds['beta_H']) # Switch-to-human risk threshold
        beta_R = 1 - np.array(thresholds['beta_R']) # Switch-to-robot risk threshold
        eps_H = thresholds['eps_H'] # Switch-to-human novelty threshold
        eps_R = thresholds['eps_R'] # Switch-to-robot novelty threshold

        beta_Hs.append(beta_H)
        beta_Rs.append(beta_R)
        eps_Hs.append(eps_H)
        eps_Rs.append(eps_R)

plt.plot(np.arange(len(beta_Hs)), beta_Hs, marker='o', label=r'$\beta_H$ (switch to human, Risk)')
plt.plot(np.arange(len(beta_Rs)), beta_Rs, marker='o', label=r'$\beta_R$ (switch to robot, Risk)')
plt.plot(np.arange(len(eps_Hs)), eps_Hs, marker='o', label=r'$\epsilon_H$ (switch to human, Novelty)')
plt.plot(np.arange(len(eps_Rs)), eps_Rs, marker='o', label=r'$\epsilon_R$ (switch to robot, $d(a_h, a_r)$)')
plt.title('Thresholds, Human Sup')
plt.xlabel('Epoch')
plt.xticks(np.arange(len(beta_Hs)))
plt.ylim(0, 1)
plt.legend()
plt.savefig('thresholds_human_sup.png')