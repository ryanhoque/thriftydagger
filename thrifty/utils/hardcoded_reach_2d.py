import numpy as np
from robosuite.utils.transform_utils import pose2mat

class HardcodedReach2DPolicy():
    def __init__(self, env):
        self.env = env
        self.last_turn = None # whether we last turned CW or CCW

    def act(self, o): 
        obj_pos, obj_quat = o[:3], o[3:7]
        rel_quat = o[10:14]
        eef_pos, eef_quat = o[32:35], o[35:39]
        a = np.zeros(7)

        pose = pose2mat((obj_pos, obj_quat))
        grasp_point = (obj_pos[0], obj_pos[1], obj_pos[2] + 0.01)#(pose @ np.array([0.06, 0, 0, 1]))[:-1]        
        
        if not self.env.gripper_closed and np.linalg.norm(grasp_point[:2] - eef_pos[:2]) > 0.005:
            if abs(obj_pos[2] + 0.1 - eef_pos[2]) > 0.0075:
                # move gripper to the height of the washer.
                a[-1] = -1.
                a[2] = 30 * (obj_pos[2]+ 0.1 - eef_pos[2])
                return a
            
            # move gripper to be aligned with washer handle.
            a[-1] = -1.
            a[0:2] = 50 * (grasp_point[:2] - eef_pos[:2])
            self.last_turn = None
            return a
        

        if not self.env.gripper_closed and abs(abs(rel_quat[0]) - 1) > 0.02 and abs(abs(rel_quat[1]) - 1) > 0.02:
            # rotate gripper to be perpendicular to the washer.
            
            a[-1] = -1.
            if self.last_turn:
                a[5] = self.last_turn
            elif abs(rel_quat[0] + 1) < abs(rel_quat[1] + 1): # rotate CW
                a[5] = -0.3
                self.last_turn = -0.3
            else: # rotate CCW
                a[5] = 0.3
                self.last_turn = 0.3
            return a

        if not self.env.gripper_closed and abs(obj_pos[2] + 0.05 - eef_pos[2]) > 0.0075:
            # move gripper to the height of the washer.
            a[-1] = -1.
            a[2] = 30 * (obj_pos[2]+ 0.05 - eef_pos[2])
            return a

        if not self.env.gripper_closed:
            # grasp washer.
            a[-1] = 1.
            return a

        cylinder_pos = np.array([0, 0.41, 1.1])
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.015 and abs(cylinder_pos[2] - eef_pos[2]) > 0.01:
            # move washer to correct height.
            a[-1] = 1.
            target_height = 1.0
            a[2] = 50 * (cylinder_pos[2] - eef_pos[2])
            return a
        
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005:
            # center above the cylinder.
            a[-1] = 1.
            a[0:2] = 50 * (cylinder_pos[:2] - obj_pos[:2])
            return a

        # lower cereal box into compartment.
        if eef_pos[2] >= 0.98:
            a[-1] = 1.
            a[2] = 55 * (0.83 - eef_pos[2])
            return a
        
        # release cereal box
        a[-1] = -1
        return a
    
    
class HardcodedStochasticPickPlacePolicy():
    def __init__(self, env):
        self.env = env
        self.last_turn = None # whether we last turned CW or CCW
        self.height = np.random.uniform(1.1, 1.3)
        
    def reset_height(self):
        self.height = np.random.uniform(1.1, 1.3)
    def act(self, o): 
        obj_pos, obj_quat = o[:3], o[3:7]
        rel_quat = o[10:14]
        eef_pos, eef_quat = o[32:35], o[35:39]
        a = np.zeros(7)

        pose = pose2mat((obj_pos, obj_quat))
        grasp_point = (obj_pos[0], obj_pos[1], obj_pos[2] + 0.01)#(pose @ np.array([0.06, 0, 0, 1]))[:-1]        
        
        if not self.env.gripper_closed and np.linalg.norm(grasp_point[:2] - eef_pos[:2]) > 0.005:
            if abs(obj_pos[2] + 0.1 - eef_pos[2]) > 0.0075:
                # move gripper to the height of the washer.
                a[-1] = -1.
                a[2] = 30 * (obj_pos[2]+ 0.1 - eef_pos[2])
                return a
            
            # move gripper to be aligned with washer handle.
            a[-1] = -1.
            a[0:2] = 50 * (grasp_point[:2] - eef_pos[:2])
            self.last_turn = None
            return a
        

        if not self.env.gripper_closed and abs(abs(rel_quat[0]) - 1) > 0.02 and abs(abs(rel_quat[1]) - 1) > 0.02:
            # rotate gripper to be perpendicular to the washer.
            
            a[-1] = -1.
            if self.last_turn:
                a[5] = self.last_turn
            elif abs(rel_quat[0] + 1) < abs(rel_quat[1] + 1): # rotate CW
                a[5] = -0.3
                self.last_turn = -0.3
            else: # rotate CCW
                a[5] = 0.3
                self.last_turn = 0.3
            return a

        if not self.env.gripper_closed and abs(obj_pos[2] + 0.05 - eef_pos[2]) > 0.0075:
            # move gripper to the height of the washer.
            a[-1] = -1.
            a[2] = 30 * (obj_pos[2]+ 0.05 - eef_pos[2])
            return a

        if not self.env.gripper_closed:
            # grasp washer.
            a[-1] = 1.
            return a

        # Set target height randomly
        cylinder_pos = np.array([0, 0.41, self.height])
        
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.015 and abs(cylinder_pos[2] - eef_pos[2]) > 0.01:
            # move washer to correct height.
            a[-1] = 1.
            target_height = 1.0
            a[2] = 50 * (cylinder_pos[2] - eef_pos[2])
            return a
        
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005:
            # center above the cylinder.
            a[-1] = 1.
            a[0:2] = 50 * (cylinder_pos[:2] - obj_pos[:2])
            return a

        # lower cereal box into compartment.
        if eef_pos[2] >= 0.98:
            a[-1] = 1.
            a[2] = 55 * (0.83 - eef_pos[2])
            return a
        
        # release cereal box
        a[-1] = -1
        return a