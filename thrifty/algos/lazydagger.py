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

    def save_buffer(self):
        pickle.dump({'obs_buf': self.obs_buf, 'act_buf': self.act_buf,
            'ptr': self.ptr, 'size': self.size}, open('replay_buffer.pkl', 'wb'))
        print('buf size', self.size)

    def load_buffer(self):
        p = pickle.load(open('replay_buffer.pkl', 'rb'))
        self.obs_buf = p['obs_buf']
        self.act_buf = p['act_buf']
        self.ptr = p['ptr']
        self.size = p['size']

    def clear(self):
        self.ptr, self.size = 0, 0

def generate_offline_data(env, expert_policy, num_episodes=0, output_file='data.pkl', 
    robosuite=False, robosuite_cfg=None):
    # Runs expert policy in the environment to collect data
    i = 0
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
            if total_ret > 0: # only count successful episodes for offline data collection
                i += 1
                obs.extend(curr_obs)
                act.extend(curr_act)
            env.close()
        else:
            i += 1
            obs.extend(curr_obs)
            act.extend(curr_act)
        print("Collected episode with return {}".format(total_ret))
        rew.append(total_ret)
    print("Ep Mean, Std Dev:", np.array(rew).mean(), np.array(rew).std())
    pickle.dump({'obs': np.stack(obs), 'act': np.stack(act)}, open(output_file, 'wb'))


def lazy(env, iters=10, actor_critic=core.MLP, ac_kwargs=dict(), 
    seed=0, grad_steps=500, obs_per_iter=2000, replay_size=int(3e4), pi_lr=1e-3, pi_safe_lr=1e-3,
    batch_size=100, logger_kwargs=dict(), num_test_episodes=100, bc_epochs=5, noise=0.,
    input_file='data.pkl', device_idx=0, expert_policy=None,
    robosuite=False, robosuite_cfg=None, 
    tau_sup=0.008, tau_auto=0.25, init_model=None, test_intervention_eps=None, stochastic_expert=False, algo_sup=False):
    """
    input_file: where initial BC data is stored
    robosuite: whether to enable robosuite specific code (and use robosuite_cfg)
    training: if this function should do training or data collection
    tau_sup and tau_auto: the main knobs to tune, i.e. the switching thresholds in LazyDAgger
    """
    logger = EpochLogger(**logger_kwargs)
    _locals = locals()
    del _locals['env']
    del _locals['expert_policy']
    
    logger.save_config(_locals)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if device_idx >= 0:
    #     device = torch.device("cuda", device_idx)
    # else:
    #     device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if stochastic_expert:
        expert_policy_cls = expert_policy
        expert_policy = expert_policy.act
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0] 
    assert act_limit == -1 * env.action_space.low[0], "Action space should be symmetric"
    safety_threshold = tau_sup * sum((env.action_space.high - env.action_space.low) ** 2)
    cov = noise * act_limit * np.eye(act_dim)

    # initialize actor and classifier NN
    ac = actor_critic(env.observation_space, env.action_space, device, **ac_kwargs)
    horizon = robosuite_cfg['MAX_EP_LEN']

    def test_agent(epoch=0):
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
            if env._render:
                print('episode #{} success? {}'.format(j, rew[-1]))
            if robosuite:
                env.close()
        print('Test Success Rate:', sum(rew)/num_test_episodes)
        pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)}, open(logger_kwargs['output_dir']+'/test{}.pkl'.format(epoch), 'wb'))

    if num_test_episodes > 0 and init_model is not None:
        states = torch.load(init_model, map_location=device)
        ac = states['model']
        test_agent(0)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    input_data = pickle.load(open(input_file, 'rb'))
    # set aside Dsafe
    safe_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    num_bc = len(input_data['obs'])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    replay_buffer.fill_buffer(input_data['obs'][idxs][:int(0.7*num_bc)], input_data['act'][idxs][:int(0.7*num_bc)])
    safe_buffer.fill_buffer(input_data['obs'], input_data['act'])
    held_out_data = {'obs': input_data['obs'][idxs][int(0.7*num_bc):], 'act': input_data['act'][idxs][int(0.7*num_bc):]}

    # Set up function for computing actor loss
    def compute_loss_pi(data):
        o, a = data['obs'], data['act']
        a_pred = ac.pi(o)
        return torch.mean(torch.sum((a - a_pred)**2, dim=1))

    def compute_loss_safe(data):
        o, a = data['obs'], data['act']
        safe_pred = ac.pi_safe(o).squeeze()
        a_pred = ac.pi(o).detach()
        targets = torch.lt(torch.sum((a - a_pred)**2, dim=1), safety_threshold).float().detach()
        return torch.nn.BCELoss()(safe_pred, targets)

    # Set up optimizers
    if init_model is not None:
        state = torch.load(init_model, map_location=device)
        ac = state['model'].to(device)
        ac.device = device
        pi_optimizer = state['pi_optimizer']
        safe_optimizer = state['safe_optimizer']
        metrics_file = os.path.join(os.path.dirname(os.path.dirname(init_model)), 'metrics.pkl')
        with open(metrics_file, 'rb') as f:
            epochs = pickle.load(f)['Epochs']
            start_epoch = epochs[-1] + 1
    else:
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        safe_optimizer = Adam(ac.pi_safe.parameters(), lr=pi_safe_lr)
        start_epoch = 0

    # Set up model saving
    logger.setup_pytorch_saver({
        'model': ac,
        'pi_optimizer': pi_optimizer,
        'safe_optimizer': safe_optimizer
        })

    def update_pi(data):
        # run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        # Record things
        logger.store(LossPi=loss_pi.item())

    def update_pi_safe(data):
        # run one gradient descent step for pi.
        safe_optimizer.zero_grad()
        loss_safe = compute_loss_safe(data)
        loss_safe.backward()
        safe_optimizer.step()
        # Record things
        logger.store(LossSafe=loss_safe.item())

    # Prepare for interaction with environment
    online_burden = 0 # how many labels we get from supervisor
    num_switches_to_human = 0
    num_switches_to_robot = 0

    # initial BC
    if init_model is None:
        for j in range(bc_epochs):
            for _ in range(grad_steps):
                batch = replay_buffer.sample_batch(batch_size)
                update_pi(data=batch)
            logger.log_tabular('Epoch', j)
            logger.log_tabular('TotalEnvInteracts', 0)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossSafe', 0)
            logger.dump_tabular()

        for j in range(bc_epochs):
            for _ in range(grad_steps):
                batch = safe_buffer.sample_batch(batch_size)
                update_pi_safe(data=batch)
            logger.log_tabular('Epoch', j)
            logger.log_tabular('TotalEnvInteracts', 0)
            logger.log_tabular('LossPi', 0)
            logger.log_tabular('LossSafe', average_only=True)
            logger.dump_tabular()

    # sanity check safe thresh
    num_unsafe = 0
    for i in range(replay_buffer.size):
        a_pred = ac.act(torch.as_tensor(safe_buffer.obs_buf[i], dtype=torch.float32, device=device))
        a_sup = safe_buffer.act_buf[i]
        if sum((a_pred - a_sup) ** 2) > safety_threshold:
            num_unsafe += 1
    # In the SafeDAgger paper, this value is around 20%
    print("{}% of BC data points are not safe with threshold {}".format(num_unsafe*100/replay_buffer.size, tau_sup))

    torch.cuda.empty_cache()

    total_env_interacts = 0
    ep_num = 0
    fail_ct = 0
    from collections import defaultdict
    metrics = defaultdict(list)

    for t in range(start_epoch, iters):
        # collect on policy data and collect expert labels
        logging_data = []
        i = 0
        while i < obs_per_iter:
            o, d, expert_mode, ep_len = env.reset(), False, False, 0
            if robosuite:
                robosuite_cfg['INPUT_DEVICE'].start_control()
            obs, act, rew, done, sup, safety, disc = [o], [], [], [], [], [ac.classify(o)], []
            while i < obs_per_iter and not d:
                a = ac.act(o)
                a = np.clip(a, -act_limit, act_limit)
                if expert_mode:
                    a_expert = expert_policy(o)
                    a_expert = np.clip(a_expert, -act_limit, act_limit)
                    # injecting noise...
                    a_expert_noisy = np.random.multivariate_normal(a_expert, cov)
                    a_expert_noisy = np.clip(a_expert_noisy, -act_limit, act_limit)
                    replay_buffer.store(o, a_expert)
                    online_burden += 1
                    #if ac.classify(o) >= 0.5: # uncomment this line and comment the one below to use SafeDAgger switching
                    if sum((a - a_expert) ** 2) < safety_threshold * tau_auto: 
                        if env._render:
                            print("Switch to Robot")
                        expert_mode = False 
                        num_switches_to_robot += 1
                        o, _, d, _ = env.step(a_expert)
                    else:
                        o, _, d, _ = env.step(a_expert_noisy)
                    act.append(a_expert)
                    sup.append(1)
                    disc.append(sum((a - a_expert)**2))
                elif ac.classify(o) < 0.5:
                    if env._render:
                        print("Switch to Human")
                    num_switches_to_human += 1
                    expert_mode = True
                    continue
                else:
                    o, _, d, _ = env.step(a)
                    act.append(a)
                    sup.append(0)
                    disc.append(0)
                if robosuite:
                    d = (ep_len >= robosuite_cfg['MAX_EP_LEN']) or env._check_success()
                    done.append(d)
                    rew.append(int(env._check_success()))
                obs.append(o)
                safety.append(ac.classify(o))
                i += 1
                ep_len += 1
            ep_num += 1
            if (ep_len >= robosuite_cfg['MAX_EP_LEN']):
                fail_ct += 1
            total_env_interacts += ep_len
            if robosuite:
                env.close()
            logging_data.append({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew), 'sup': np.array(sup), 
                'safety': np.array(safety), 'disc': np.array(disc)})
            pickle.dump(logging_data, open(logger_kwargs['output_dir']+'/iter{}.pkl'.format(t), 'wb'))
            if stochastic_expert:
                expert_policy_cls.reset_height()
                expert_policy = expert_policy_cls.act
            if test_intervention_eps is not None and ep_num >= test_intervention_eps:
                break
        # retrain from scratch each time
        if test_intervention_eps == None:
            ac = actor_critic(env.observation_space, env.action_space, device, **ac_kwargs)
            pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
            safe_optimizer = Adam(ac.pi_safe.parameters(), lr=pi_safe_lr)
            logger.setup_pytorch_saver({
                'model': ac,
                'pi_optimizer': pi_optimizer,
                'safe_optimizer': safe_optimizer
                })
            for _ in range(grad_steps * (bc_epochs + t + 1)):
                batch = replay_buffer.sample_batch(batch_size)
                update_pi(data=batch)
            for _ in range(grad_steps * (bc_epochs + t + 1)):
                batch = safe_buffer.sample_batch(batch_size)
                update_pi_safe(data=batch)

        # End of epoch handling
        # logging



        logger.save_state(dict(), itr=t)
        logger.log_tabular('Epoch', t)
        logger.log_tabular('TotalEnvInteracts', total_env_interacts)
        if test_intervention_eps == None:
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossSafe', average_only=True)
        metrics['Epochs'].append(t)
        metrics['TotalEpisodes'].append(ep_num)
        metrics['TotalSuccesses'].append(ep_num - fail_ct)
        metrics['OnlineBurden'].append(online_burden)
        metrics['NumSwitchTo'].append(num_switches_to_human)
        metrics['NumSwitchBack'].append(num_switches_to_robot)
        logger.dump_tabular()
        if test_intervention_eps is not None and ep_num >= test_intervention_eps:
                break
    with open(os.path.join(logger_kwargs['output_dir'], 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
