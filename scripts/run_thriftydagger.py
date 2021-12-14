# Script for running ThriftyDAgger
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.algos.lazydagger import lazy
from thrifty.utils.run_utils import setup_logger_kwargs
import gym, torch
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import GymWrapper
from robosuite.devices import Keyboard, SpaceMouse
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
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--gen_data', action='store_true', help="True if you want to collect offline human demos")
    parser.add_argument('--iters', type=int, default=5, help="number of DAgger-style iterations")
    parser.add_argument('--targetrate', type=float, default=0.01, help="target context switching rate")
    parser.add_argument('--environment', type=str, default="NutAssembly")
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--hgdagger', action='store_true')
    parser.add_argument('--lazydagger', action='store_true')
    parser.add_argument('--eval', type=str, default=None, help="filepath to saved pytorch model to initialize weights")
    parser.add_argument('--algo_sup', action='store_true', help="use an algorithmic supervisor")
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

    robosuite_cfg = {'MAX_EP_LEN': 175, 'INPUT_DEVICE': input_device}
    if args.algo_sup:
        expert_pol = HardcodedPolicy(env).act
    if args.gen_data:
    	NUM_BC_EPISODES = 30
    	generate_offline_data(env, expert_policy=expert_pol, num_episodes=NUM_BC_EPISODES, seed=args.seed,
            output_file="robosuite-{}.pkl".format(NUM_BC_EPISODES), robosuite=True, robosuite_cfg=robosuite_cfg)
    if args.hgdagger:
        thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
            seed=args.seed, expert_policy=expert_pol, input_file="robosuite-30.pkl", robosuite=True, 
            robosuite_cfg=robosuite_cfg, num_nets=1, hg_dagger=hg_dagger_wait, init_model=args.eval)
    elif args.lazydagger:
        lazy(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, noise=0.,
         seed=args.seed, expert_policy=expert_pol, input_file="robosuite-30.pkl", robosuite=True, 
           robosuite_cfg=robosuite_cfg)
    else:
        thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
         	seed=args.seed, expert_policy=expert_pol, input_file="robosuite-30.pkl", robosuite=True, 
            robosuite_cfg=robosuite_cfg, q_learning=True, init_model=args.eval)
