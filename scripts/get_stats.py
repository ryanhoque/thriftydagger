import pickle
import sys
import numpy as np

nr, nh, ni, ns, nf = [], [], [], 0, 0
tnr, tnh, tni = [], [], []
for t in range(1,2):
    p = pickle.load(open('iter{}.pkl'.format(t), 'rb'))
    for i in range(len(p)):
        ni_ = 0
        tni_ = 0
        if p[i]['done'][-1] and p[i]['rew'][-1]: # success
            ns += 1
            nh.append(sum(p[i]['sup']))
            nr.append(len(p[i]['sup']) - sum(p[i]['sup']))
            for j in range(len(p[i]['sup']) - 1):
                if p[i]['sup'][j] == 0 and p[i]['sup'][j+1] == 1:
                    ni_ += 1
            ni.append(ni_)
        elif p[i]['done'][-1]: # failure
            nf += 1
            tnh.append(sum(p[i]['sup']))
            tnr.append(len(p[i]['sup']) - sum(p[i]['sup']))
            for j in range(len(p[i]['sup']) - 1):
                if p[i]['sup'][j] == 0 and p[i]['sup'][j+1] == 1:
                    tni_ += 1
            tni.append(tni_)
        else: # last episode, didnt terminate
            tnh.append(sum(p[i]['sup']))
            tnr.append(len(p[i]['sup']) - sum(p[i]['sup']))
tnr.extend(nr)
tnh.extend(nh)
tni.extend(ni)
            
print('num success', ns, 'num failure', nf, 'total', ns+nf)
print('succ rate', ns / (ns + nf))
print('mean+std rob act per success', sum(nr) / ns, np.array(nr).std())
print('mean+std hum act per success', sum(nh) / ns, np.array(nh).std())
print('mean+std interventions per success', sum(ni) / ns, np.array(ni).std())
print('mean+std rob act per all', sum(tnr) / (ns + nf), np.array(tnr).std())
print('mean+std hum act per all', sum(tnh) / (ns + nf), np.array(tnh).std())
print('mean+std interventions per all', sum(tni) / (ns + nf), np.array(tni).std())
print('total rob act', sum(tnr))
print('total hum act', sum(tnh))
print('total int', sum(tni))
