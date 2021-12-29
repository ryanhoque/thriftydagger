import gym
import numpy as np


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
        action_[3] = 0.0
        action_[4] = 0.0
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
