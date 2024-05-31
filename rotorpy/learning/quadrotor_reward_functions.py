import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces

import math
from rotorpy.utils.helper_functions import compute_yaw_from_quarternion

"""
Reward functions for quadrotor tasks. 
"""

def hover_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[0:3])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[3:6])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[10:13])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    return dist_reward + vel_reward + action_reward + ang_rate_reward

# def trajectory_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
#     q = observation['w']
#     R_matrices = R.from_quat(q).as_matrix()
#     b3 = R_matrices[:, 2]
#     H = np.array([[
#                     1 - (b3[0] ** 2) / (1 + b3[2]),
#                     -(b3[0] * b3[1]) / (1 + b3[2]),
#                     b3[0],
#                 ],
#                 [
#                     -(b3[0] * b3[1]) / (1 + b3[2]),
#                     1 - (b3[1] ** 2) / (1 + b3[2]),
#                     b3[1],
#                 ],
#                 [-b3[0], -b3[1], b3[2]],
#             ]
#         )
#     Hyaw = np.transpose(H) @ R_matrices
#     yaw = np.arctan2(Hyaw[1, 0], Hyaw[0, 0])
#     # yaw = state.rot.as_euler('ZYX')[0]
#     yawcost = 0.5 * min(abs(self.ref.yaw - yaw), abs(self.ref.yaw(self.t) - yaw))
#     yawcost = 0.5 * min(abs(self.ref.yaw(self.t) - yaw), abs(self.ref.yaw(self.t) - yaw))
#     poscost = np.linalg.norm(obs[0:3] - self.ref.x_poly(self.t))#min(np.linalg.norm(state.pos), 1.0)
#     velcost = 0.1 * min(np.linalg.norm(obs[3:6]), 1.0)

#     ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

#     cost = yawcost + poscost + velcost

#     return -cost

def trajectory_reward(observation, action, weights={'x': 1, 'v': 0.1, 'q': 0, 'u': 1e-5}):
    """
    Rewards for trajectory tracking given observations following DATT. It is a combination of position error, 
    velocity error, body rates, and action reward.
    observation: dict(R_t^T)
    """

    # Compute position error from nearest waypoint

    # tracking_cost = regularizer(observation[13:])  # TODO for the future

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[0:3]-observation[13:16])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[3:6]-observation[16:19])

    # Compute the yaw reward
    yaw = compute_yaw_from_quaternion(observation[6:10])
    yaw_reward = -weights['q'] * np.linalg.norm(yaw - observation[28:31])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w'] * np.linalg.norm(observation[10:13] - observation[31:34])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    return dist_reward + vel_reward + yaw_reward + action_reward + ang_rate_reward