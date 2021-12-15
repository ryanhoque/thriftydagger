# Script for running ThriftyDAgger
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.algos.lazydagger import lazy
from thrifty.utils.run_utils import setup_logger_kwargs
import gym, torch
import os
import pickle
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from thrifty.utils.hardcoded_pick_place import HardcodedPickPlacePolicy, HardcodedStochasticPickPlacePolicy
from thrifty.utils.hardcoded_reach_2d import HardcodedReach2DPolicy, sample_reach
from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import GymWrapper
from robosuite.devices import Keyboard
import numpy as np
import sys
import time

class CustomWrapper(gym.Env):

    def __init__(self, env, render):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self._render = render

    def reset(self):
        r = self.env.reset()
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = -1
        for _ in range(10):
            r, r2, r3, r4 = self.env.step(settle_action)
            self.render()
        self.gripper_closed = False
        return r

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        action_[3] = 0.
        action_[4] = 0.
        self.env.step(action_)
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action[-1]
        for _ in range(10):
            r1, r2, r3, r4 = self.env.step(settle_action)
            self.render()
        if action[-1] > 0:
            self.gripper_closed = True
        else:
            self.gripper_closed = False
        return r1, r2, r3, r4

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--gen_data', action='store_true', help="True if you want to collect offline human demos")
    parser.add_argument('--input_file', type=str, default='./data/testpickplace/testpickplace_s4/pick-place-data.pkl')
    parser.add_argument('--iters', type=int, default=5, help="number of DAgger-style iterations")
    parser.add_argument('--targetrate', type=float, default=0.01, help="target context switching rate")
    parser.add_argument('--environment', type=str, default="NutAssembly")
    parser.add_argument('--num_test_episodes', type=int, default=10, help='number of test episodes for autonomous rollout / data collection')
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--hgdagger', action='store_true')
    parser.add_argument('--lazydagger', action='store_true')
    parser.add_argument('--bc_only', action='store_true')
    parser.add_argument('--stochastic_expert', action='store_true')
    parser.add_argument('--eval', type=str, default=None, help="filepath to saved pytorch model to initialize weights")
    parser.add_argument('--algo_sup', action='store_true', help="use an algorithmic supervisor")
    parser.add_argument('--gen_data_output', type=str, default='pick-place-data', help='Output location for recorded demonstrations; used in conjunction wtih --gen_data.')
    parser.add_argument('--test_intervention_eps', type=int, default=None)
    parser.add_argument('--hg_oracle_thresh', type=float, default=0.2)
    parser.add_argument('--tau_sup', type=float, default=0.008)
    parser.add_argument('--tau_auto', type=float, default=0.25)
    parser.add_argument('--num_bc_episodes', type=int, default=30)
    parser.add_argument('--expert_sim_style', type=int, default=0)
    args = parser.parse_args()
    render = not args.no_render

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # setup env ...
    controller_config = load_controller_config(default_controller='OSC_POSE')
    config = {
        "env_name": args.environment,
        "robots": "UR5e",
        "controller_configs": controller_config,
    }

    if args.environment == 'NutAssembly':
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2, # env has 1 nut instead of 2
            nut_type="round",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    elif args.environment == 'PickPlace':
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2,
            object_type='cereal',
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    else:
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    env = GymWrapper(env)
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)

    arm_ = 'right'
    config_ = 'single-arm-opposed'
    input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)
    if render:
        env.viewer.add_keypress_callback("any", input_device.on_press)
        env.viewer.add_keyup_callback("any", input_device.on_release)
        env.viewer.add_keyrepeat_callback("any", input_device.on_press)
    active_robot = env.robots[arm_ == 'left']

    def hg_dagger_wait():
        # for HG-dagger, repurpose the 'Z' key (action elem 3) for starting/ending interaction
        for _ in range(10):
            a, _ = input2action(
                device=input_device,
                robot=active_robot,
                active_arm=arm_,
                env_configuration=config_)
            env.render()
            time.sleep(0.001)
            if a[3] != 0: # z is pressed
                break
        return (a[3] != 0)

    def expert_pol(o):
        a = np.zeros(7)
        if env.gripper_closed:
            a[-1] = 1.
            input_device.grasp = True
        else:
            a[-1] = -1.
            input_device.grasp = False
        a_ref = a.copy()
        # pause simulation if there is no user input (instead of recording a no-op)
        while np.array_equal(a, a_ref):
            a, _ = input2action(
                device=input_device,
                robot=active_robot,
                active_arm=arm_,
                env_configuration=config_)
            env.render()
            time.sleep(0.001)
        return a
    
    robosuite_cfg = {'MAX_EP_LEN': 200, 'INPUT_DEVICE': input_device}
    os.makedirs(logger_kwargs['output_dir'], exist_ok=True)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').index
    robosuite = True
    if args.algo_sup:
        if args.environment == 'NutAssembly':
            expert_pol = HardcodedPolicy(env).act
        elif args.environment == 'PickPlace':
            if args.stochastic_expert:
                expert_pol = HardcodedStochasticPickPlacePolicy(env)
            else:
                expert_pol = HardcodedPickPlacePolicy(env).act
        elif args.environment == 'Reach2D':
            expert_pol = HardcodedReach2DPolicy(env, style=args.expert_sim_style).act
            robosuite = False
    if args.gen_data:
    	num_bc_episodes = args.num_bc_episodes
    	out_file = os.path.join(logger_kwargs['output_dir'], f'{args.gen_data_output}-{num_bc_episodes}.pkl')
        if args.environment == 'Reach2D':
            demos = sample_reach(num_bc_episodes)
            pickle.dump(demos, open(out_file, "wb"))
        else:
            generate_offline_data(env, expert_policy=expert_pol, num_episodes=num_bc_episodes, seed=args.seed,
                output_file=out_file, robosuite=robosuite, robosuite_cfg=robosuite_cfg, stochastic_expert=args.stochastic_expert)
    if args.hgdagger:
        thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
            seed=args.seed, expert_policy=expert_pol, input_file=args.input_file, robosuite=robosuite, 
            robosuite_cfg=robosuite_cfg, num_nets=1, hg_dagger=hg_dagger_wait, init_model=args.eval, 
            num_test_episodes=args.num_test_episodes, test_intervention_eps=args.test_intervention_eps, 
            stochastic_expert=args.stochastic_expert, hg_oracle_thresh=args.hg_oracle_thresh, algo_sup=args.algo_sup)
    elif args.lazydagger:
        lazy(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, noise=0.,
         seed=args.seed, expert_policy=expert_pol, input_file=args.input_file, robosuite=robosuite, 
           robosuite_cfg=robosuite_cfg, init_model=args.eval, num_test_episodes=args.num_test_episodes, test_intervention_eps=args.test_intervention_eps, stochastic_expert=args.stochastic_expert, algo_sup=args.algo_sup,
           tau_auto=args.tau_auto, tau_sup=args.tau_sup)
    else:
        thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
         	seed=args.seed, expert_policy=expert_pol, input_file=args.input_file, robosuite=robosuite, 
            robosuite_cfg=robosuite_cfg, q_learning=True, init_model=args.eval, bc_only=args.bc_only, 
            num_test_episodes=args.num_test_episodes, test_intervention_eps=args.test_intervention_eps, 
            stochastic_expert=args.stochastic_expert, algo_sup=args.algo_sup)
