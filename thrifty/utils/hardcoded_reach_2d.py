import numpy as np
import random
from robosuite.utils.transform_utils import pose2mat

ACTION_MAGNITUDE = 0.2

def sample_reach(N_trajectories):
    range_x = 3.0
    range_y = 3.0
    # states, actions, goals = [], [], []
    demos = []

    for _ in range(N_trajectories):
        # Sample goal from 1st quadrant
        goal_ee_state = np.array([random.uniform(0, range_x), random.uniform(0, range_y)])

        # Divide trajectory into multiple state-action pairs
        curr_state = np.zeros(2)
        traj = goal_ee_state - curr_state
        # action = traj / N_samples
        states = [curr_state]
        actions = []
        goals = []
        action = ACTION_MAGNITUDE * traj / np.linalg.norm(action)
        while (curr_state[0] < goal_ee_state[0] and curr_state[1] < goal_ee_state[1]):
            states.append(curr_state)
            actions.append(action)
            goals.append(goal_ee_state)
            curr_state = curr_state + action
        demos.append([(state, action, goal) for state, action, goal in zip(states, actions, goals)])
    return demos


class HardcodedReach2DPolicy():
    def __init__(self, obj_loc, sim_style):
        self.obj_loc = obj_loc # (2,)
        self.sim_style = sim_style
        self.last_turn = None # whether we last turned CW or CCW

    def act(self, curr_pos): 
       a = self.obj_loc - curr_pos
       a /= np.linalg.norm(a)
       a *= ACTION_MAGNITUDE
       return a