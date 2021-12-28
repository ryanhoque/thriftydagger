from constants import REACH2D_ACT_DIM, REACH2D_OBS_DIM, REACH2D_SUCCESS_THRESH

import random
import torch

class Reach2D:

    def __init__(self, device, range_x=3.0, range_y=3.0):
        self.device = device
        self.range_x = range_x
        self.range_y = range_y
        self.curr_state = torch.zeros(2, device=self.device)
        self.goal_state = torch.tensor([random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device)
        self.act_dim = REACH2D_ACT_DIM
        self.obs_dim = REACH2D_OBS_DIM
        
    def close(self):
        '''
        Empty function so callers don't break with use of this class
        '''
        pass
    def reset(self):
        self.curr_state = torch.zeros(2, device=self.device)
        self.goal_state = torch.tensor([random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device)
        curr_obs = torch.cat([self.curr_state, self.goal_state])
        return curr_obs

    def step(self, action):
        self.curr_state += action
        curr_obs = torch.cat([self.curr_state, self.goal_state])
        return curr_obs, None, None, None

    def _check_success(self):
        x = self.curr_state - self.goal_state
        return (torch.norm(self.curr_state - self.goal_state) <= REACH2D_SUCCESS_THRESH).item()
    
    
