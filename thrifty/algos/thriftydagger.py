from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import thrifty.algos.core as core
from thrifty.utils.logx import EpochLogger
import pickle
import os
import sys
import random

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) 
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                act=self.act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    def fill_buffer(self, obs, act):
        for i in range(len(obs)):
            self.store(obs[i], act[i])

    def save_buffer(self, name='replay'):
        pickle.dump({'obs_buf': self.obs_buf, 'act_buf': self.act_buf,
            'ptr': self.ptr, 'size': self.size}, open('{}_buffer.pkl'.format(name), 'wb'))
        print('buf size', self.size)

    def load_buffer(self, name='replay'):
        p = pickle.load(open('{}_buffer.pkl'.format(name), 'rb'))
        self.obs_buf = p['obs_buf']
        self.act_buf = p['act_buf']
        self.ptr = p['ptr']
        self.size = p['size']

    def clear(self):
        self.ptr, self.size = 0, 0

class QReplayBuffer:
    # Replay buffer for training Qrisk
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, pos_fraction=None):
        # pos_fraction: ensure that this fraction of the batch has rew 1 for better reward propagation
        if pos_fraction is not None:
            pos_size = min(len(tuple(np.argwhere(self.rew_buf).ravel())), int(batch_size * pos_fraction))
            neg_size = batch_size - pos_size
            pos_idx = np.array(random.sample(tuple(np.argwhere(self.rew_buf).ravel()), pos_size))
            neg_idx = np.array(random.sample(tuple(np.argwhere((1 - self.rew_buf)[:self.size]).ravel()), neg_size))
            idxs = np.hstack((pos_idx, neg_idx))
            np.random.shuffle(idxs)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}

    def fill_buffer(self, data):
        obs_dim = data['obs'].shape[1]
        act_dim = data['act'].shape[1]
        for i in range(len(data['obs'])):
            if data['done'][i] and not data['rew'][i]: # time boundary, not really done
                continue
            elif data['done'][i] and data['rew'][i]: # successful termination
                self.store(data['obs'][i], data['act'][i], np.zeros(obs_dim), data['rew'][i], data['done'][i])
            else:
                self.store(data['obs'][i], data['act'][i], data['obs'][i+1], data['rew'][i], data['done'][i])

    def fill_buffer_from_BC(self, data, goals_only=False):
        """
        Load buffer from offline demos (only obs/act)
        goals_only: if True, only store the transitions with positive reward
        """
        num_bc = len(data['obs'])
        obs_dim = data['obs'].shape[1]
        for i in range(num_bc - 1):
            if data['act'][i][-1] == 1 and data['act'][i+1][-1] == -1:
                # new episode starting
                self.store(data['obs'][i], data['act'][i], np.zeros(obs_dim), 1, 1)
            elif not goals_only:
                self.store(data['obs'][i], data['act'][i], data['obs'][i+1], 0, 0)
        self.store(data['obs'][num_bc - 1], data['act'][num_bc - 1], np.zeros(obs_dim), 1, 1)

    def clear(self):
        self.ptr, self.size = 0, 0

def generate_offline_data(env, expert_policy, num_episodes=0, output_file='data.pkl', 
    robosuite=False, robosuite_cfg=None, seed=0):
    # Runs expert policy in the environment to collect data
    i, failures = 0, 0
    np.random.seed(seed)
    obs, act, rew = [], [], []
    act_limit = env.action_space.high[0]
    while i < num_episodes:
        print('Episode #{}'.format(i))
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        if robosuite:
            robosuite_cfg['INPUT_DEVICE'].start_control()
        while not d:
            a = expert_policy(o)
            if a is None:
                d, r = True, 0
                continue
            a = np.clip(a, -act_limit, act_limit)
            curr_obs.append(o)
            curr_act.append(a)
            o, r, d, _ = env.step(a)
            if robosuite:
                d = (t >= robosuite_cfg['MAX_EP_LEN']) or env._check_success()
                r = int(env._check_success())
            total_ret += r
            t += 1
        if robosuite:
            if total_ret > 0: # only count successful episodes
                i += 1
                obs.extend(curr_obs)
                act.extend(curr_act)
            else:
                failures += 1
            env.close()
        else:
            i += 1
            obs.extend(curr_obs)
            act.extend(curr_act)
        print("Collected episode with return {} length {}".format(total_ret, t))
        rew.append(total_ret)
    print("Ep Mean, Std Dev:", np.array(rew).mean(), np.array(rew).std())
    print("Num Successes {} Num Failures {}".format(num_episodes, failures))
    pickle.dump({'obs': np.stack(obs), 'act': np.stack(act)}, open(output_file, 'wb'))


def thrifty(env, iters=5, actor_critic=core.Ensemble, ac_kwargs=dict(), 
    seed=0, grad_steps=500, obs_per_iter=2000, replay_size=int(3e4), pi_lr=1e-3, 
    batch_size=100, logger_kwargs=dict(), num_test_episodes=10, bc_epochs=5,
    input_file='data.pkl', device_idx=0, expert_policy=None, num_nets=5,
    target_rate=0.01, robosuite=False, robosuite_cfg=None, hg_dagger=None,
    q_learning=False, gamma=0.9999, init_model=None):
    """
    obs_per_iter: environment steps per algorithm iteration
    num_nets: number of neural nets in the policy ensemble
    input_file: where initial BC data is stored (output of generate_offline_data())
    target_rate: desired rate of context switching
    robosuite: whether to enable robosuite specific code (and use robosuite_cfg)
    hg_dagger: if not None, use this function as the switching condition (i.e. run HG-DAgger)
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    num_test_episodes: run this many episodes after each iter without interventions
    init_model: initial NN weights
    """
    logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    del _locals['env']
    logger.save_config(_locals)
    if device_idx >= 0:
        device = torch.device("cuda", device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if robosuite:
        with open(os.path.join(logger_kwargs['output_dir'],'model.xml'), 'w') as fh:
            fh.write(env.env.sim.model.get_xml())

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0] 
    assert act_limit == -1 * env.action_space.low[0], "Action space should be symmetric"
    horizon = robosuite_cfg['MAX_EP_LEN']

    # initialize actor and classifier NN
    ac = actor_critic(env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs)
    if init_model:
        ac = torch.load(init_model, map_location=device).to(device)
        ac.device = device
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up optimizers
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    input_data = pickle.load(open(input_file, 'rb'))
    # shuffle and create small held out set to check valid loss
    num_bc = len(input_data['obs'])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    replay_buffer.fill_buffer(input_data['obs'][idxs][:int(0.9*num_bc)], input_data['act'][idxs][:int(0.9*num_bc)])
    held_out_data = {'obs': input_data['obs'][idxs][int(0.9*num_bc):], 'act': input_data['act'][idxs][int(0.9*num_bc):]}
    qbuffer = QReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    qbuffer.fill_buffer_from_BC(input_data)

    # Set up function for computing actor loss
    def compute_loss_pi(data, i):
        o, a = data['obs'], data['act']
        a_pred = ac.pis[i](o)
        return torch.mean(torch.sum((a - a_pred)**2, dim=1))

    def compute_loss_q(data):
        o, a, o2, r, d = data['obs'], data['act'], data['obs2'], data['rew'], data['done']
        with torch.no_grad():
            a2 = torch.mean(torch.stack([pi(o2) for pi in ac.pis]), dim=0)
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target Q-values
            q1_t = ac_targ.q1(o2, a2) # do target policy smoothing?
            q2_t = ac_targ.q2(o2, a2)
            backup = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        return loss_q1 + loss_q2

    def update_pi(data, i):
        # run one gradient descent step for pi.
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()

    def update_q(data, timer):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # update targ net
        if timer % 2 == 0:
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(.995)
                    p_targ.data.add_((1 - .995) * p.data)
        return loss_q.item()

    # Prepare for interaction with environment
    online_burden = 0 # how many labels we get from supervisor
    num_switch_to_human = 0 # context switches (due to novelty)
    num_switch_to_human2 = 0 # context switches (due to risk)
    num_switch_to_robot = 0

    def test_agent(epoch=0):
        """Run test episodes"""
        obs, act, done, rew = [], [], [], []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_ret2, ep_len = env.reset(), False, 0, 0, 0
            while not d:
                obs.append(o)
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                act.append(a)
                o, r, d, _ = env.step(a)
                if robosuite:
                    d = (ep_len + 1 >= horizon) or env._check_success()
                    ep_ret2 += int(env._check_success())
                    done.append(d)
                    rew.append(int(env._check_success()))
                ep_ret += r
                ep_len += 1
            print('episode #{} success? {}'.format(j, rew[-1]))
            if robosuite:
                env.close()
        print('Test Success Rate:', sum(rew)/num_test_episodes)
        pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)}, open('test-rollouts.pkl', 'wb'))
        pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)}, open(logger_kwargs['output_dir']+'/test{}.pkl'.format(epoch), 'wb'))

    if iters == 0 and num_test_episodes > 0: # only run evaluation.
        test_agent(0)
        sys.exit(0)

    # train policy
    for i in range(ac.num_nets):
        if ac.num_nets > 1: # create new datasets via sampling with replacement
            print('Net #{}'.format(i))
            tmp_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            tmp_buffer = replay_buffer
        for j in range(bc_epochs):
            loss_pi = []
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi.append(update_pi(batch, i))
            validation = []
            for j in range(len(held_out_data['obs'])):
                a_pred = ac.act(held_out_data['obs'][j], i=i)
                a_sup = held_out_data['act'][j]
                validation.append(sum(a_pred - a_sup)**2)
            print('LossPi', sum(loss_pi)/len(loss_pi))
            print('LossValid', sum(validation)/len(validation))

    # estimate switch-back parameter and initial switch-to parameter from data
    discrepancies, estimates = [], []
    for i in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[i])
        a_sup = replay_buffer.act_buf[i]
        discrepancies.append(sum((a_pred - a_sup)**2))
        estimates.append(ac.variance(replay_buffer.obs_buf[i]))
    heldout_discrepancies, heldout_estimates = [], []
    for i in range(len(held_out_data['obs'])):
        a_pred = ac.act(held_out_data['obs'][i])
        a_sup = held_out_data['act'][i]
        heldout_discrepancies.append(sum((a_pred - a_sup)**2))
        heldout_estimates.append(ac.variance(held_out_data['obs'][i]))
    switch2robot_thresh = np.array(discrepancies).mean()
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    switch2human_thresh = sorted(heldout_estimates)[target_idx]
    print("Estimated switch-back threshold: {}".format(switch2robot_thresh))
    print("Estimated switch-to threshold: {}".format(switch2human_thresh))
    switch2human_thresh2 = 0.48 # a priori guess: 48% discounted probability of success. Could also estimate from data
    switch2robot_thresh2 = 0.495
    torch.cuda.empty_cache()
    # we only needed the held out set to check valid loss and compute thresholds, so we can get rid of it.
    replay_buffer.fill_buffer(held_out_data['obs'], held_out_data['act'])

    total_env_interacts = 0
    ep_num = 0
    fail_ct = 0
    for t in range(iters + 1):
        logging_data = [] # for verbose logging
        estimates = []
        estimates2 = [] # refit every iter
        i = 0
        if t == 0: # skip data collection on iter 0 to train Q
            i = obs_per_iter
        while i < obs_per_iter:
            o, d, expert_mode, ep_len = env.reset(), False, False, 0
            if robosuite:
                robosuite_cfg['INPUT_DEVICE'].start_control()
            obs, act, rew, done, sup, var, risk = [o], [], [], [], [], [ac.variance(o)], []
            if robosuite:
                simstates = [env.env.sim.get_state().flatten()] # track this to replay trajectories after if desired.
            while i < obs_per_iter and not d:
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                if not expert_mode:
                    estimates.append(ac.variance(o))
                    estimates2.append(ac.safety(o,a))
                if expert_mode:
                    a_expert = expert_policy(o)
                    a_expert = np.clip(a_expert, -act_limit, act_limit)
                    replay_buffer.store(o, a_expert)
                    online_burden += 1
                    risk.append(ac.safety(o, a_expert))
                    if (hg_dagger and a_expert[3] != 0) or (not hg_dagger and sum((a - a_expert) ** 2) < switch2robot_thresh 
                        and (not q_learning or ac.safety(o, a) > switch2robot_thresh2)):
                        print("Switch to Robot")
                        expert_mode = False 
                        num_switch_to_robot += 1
                        o2, _, d, _ = env.step(a_expert)
                    else:
                        o2, _, d, _ = env.step(a_expert)
                    act.append(a_expert)
                    sup.append(1)
                    s = env._check_success()
                    qbuffer.store(o, a_expert, o2, int(s), (ep_len + 1 >= horizon) or s)
                # hg-dagger switching for hg-dagger, or novelty switching for thriftydagger
                elif (hg_dagger and hg_dagger()) or (not hg_dagger and ac.variance(o) > switch2human_thresh):
                    print("Switch to Human (Novel)")
                    num_switch_to_human += 1
                    expert_mode = True
                    continue
                # second switch condition: if not novel, but also not safe
                elif not hg_dagger and q_learning and ac.safety(o,a) < switch2human_thresh2:
                    print("Switch to Human (Risk)")
                    num_switch_to_human2 += 1
                    expert_mode = True
                    continue
                else:
                    risk.append(ac.safety(o, a))
                    o2, _, d, _ = env.step(a)
                    act.append(a)
                    sup.append(0)
                    s = env._check_success()
                    qbuffer.store(o, a, o2, int(s), (ep_len + 1 >= horizon) or s)
                d = (ep_len + 1 >= horizon) or env._check_success()
                done.append(d)
                rew.append(int(env._check_success()))
                o = o2
                obs.append(o)
                if robosuite:
                    simstates.append(env.env.sim.get_state().flatten())
                var.append(ac.variance(o))
                i += 1
                ep_len += 1
            if d:
                ep_num += 1
            if (ep_len >= horizon):
                fail_ct += 1
            total_env_interacts += ep_len
            logging_data.append({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew), 'sup': np.array(sup), 'var': np.array(var), 
                'risk': np.array(risk), 'beta_H': switch2human_thresh, 'beta_R': switch2robot_thresh, 'eps_H': switch2human_thresh2, 'eps_R': switch2robot_thresh2,
                'simstates': np.array(simstates) if robosuite else None})
            pickle.dump(logging_data, open(logger_kwargs['output_dir']+'/iter{}.pkl'.format(t), 'wb'))
            if robosuite:
                env.close()
            # recompute thresholds from data after every episode
            if len(estimates) > 25:
                target_idx = int((1 - target_rate) * len(estimates))
                switch2human_thresh = sorted(estimates)[target_idx]
                switch2human_thresh2 = sorted(estimates2, reverse=True)[target_idx]
                switch2robot_thresh2 = sorted(estimates2)[int(0.5*len(estimates))]
                print("len(estimates): {}, New switch thresholds: {} {} {}".format(len(estimates), switch2human_thresh, switch2human_thresh2, switch2robot_thresh2))

        if t > 0:
            # retrain policy from scratch
            loss_pi = []
            ac = actor_critic(env.observation_space, env.action_space, device, num_nets=num_nets, **ac_kwargs)
            pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
            for i in range(ac.num_nets):
                if ac.num_nets > 1: # create new datasets via sampling with replacement
                    tmp_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
                    for _ in range(replay_buffer.size):
                        idx = np.random.randint(replay_buffer.size)
                        tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
                else:
                    tmp_buffer = replay_buffer
                for _ in range(grad_steps * (bc_epochs + t)):
                    batch = tmp_buffer.sample_batch(batch_size)
                    loss_pi.append(update_pi(batch, i))
        # retrain Qrisk
        if q_learning:
            if num_test_episodes > 0:
                test_agent(t) # collect samples offline from pi_R
                data = pickle.load(open('test-rollouts.pkl', 'rb'))
                qbuffer.fill_buffer(data)
                os.remove('test-rollouts.pkl')
            q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
            q_optimizer = Adam(q_params, lr=pi_lr)
            loss_q = []
            for _ in range(bc_epochs):
                for i in range(grad_steps * 5):
                    batch = qbuffer.sample_batch(batch_size // 2, pos_fraction=0.1)
                    loss_q.append(update_q(batch, timer=i))

        # end of epoch logging
        logger.save_state(dict())
        print('Epoch', t)
        if t > 0:
            print('LossPi', sum(loss_pi)/len(loss_pi))
        if q_learning:
            print('LossQ', sum(loss_q)/len(loss_q))
        print('TotalEpisodes', ep_num)
        print('TotalSuccesses', ep_num - fail_ct)
        print('TotalEnvInteracts', total_env_interacts)
        print('OnlineBurden', online_burden)
        print('NumSwitchToNov', num_switch_to_human)
        print('NumSwitchToRisk', num_switch_to_human2)
        print('NumSwitchBack', num_switch_to_robot)
