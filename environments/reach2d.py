import random
import torch

SUCCESS_THRESH = 0.1
ACT_MAGNITUDE = 0.1

class Reach2D:

    def __init__(self, range_x=3.0, range_y=3.0):
        self.range_x = range_x
        self.range_y = range_y
        self.curr_state = torch.zeros(2)
        self.goal_state = torch.tensor([random.uniform(0, self.range_x), random.uniform(0, self.range_y)])
        # self.observation_space = 0
        # self.action_space = env.action_space

    def close(self):
        '''
        Empty function so callers don't break with use of this class
        '''
        pass
    def reset(self):
        self.curr_state = torch.zeros(2)
        self.goal_state = torch.tensor([random.uniform(0, self.range_x), random.uniform(0, self.range_y)])
        curr_obs = torch.cat([self.curr_state, self.goal_state])
        return curr_obs

    def step(self, action):
        self.curr_state += action
        curr_obs = torch.cat([self.curr_state, self.goal_state])
        return curr_obs, None, None, None

    def _check_success(self):
        return (torch.norm(self.curr_state - self.goal_state) <= SUCCESS_THRESH).item()
    
    