import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def compute_yaw_from_quaternion(quaternions):
        R_matrices = R.from_quat(quaternions).as_matrix()
        b3 = R_matrices[:, :, 2]
        H = np.zeros((len(quaternions), 3, 3))
        for i in range(len(quaternions)):
            H[i, :, :] = np.array([
                [1 - (b3[i, 0] ** 2) / (1 + b3[i, 2]), -(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]), b3[i, 0]],
                [-(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]), 1 - (b3[i, 1] ** 2) / (1 + b3[i, 2]), b3[i, 1]],
                [-b3[i, 0], -b3[i, 1], b3[i, 2]],
            ])
        Hyaw = np.transpose(H, axes=(0, 2, 1)) @ R_matrices
        actual_yaw = np.arctan2(Hyaw[:, 1, 0], Hyaw[:, 0, 0])
        return actual_yaw

def compute_cost(df, robust_c=1.0):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        df: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory. 
    # actual_pos
    actual_x = df['x']                                     
    actual_y = df['y']                                      
    actual_z = df['z']                                      
    actual_pos = np.array([actual_x, actual_y, actual_z]).T 
    # actual_vel
    actual_vx = df['xdot']
    actual_vy = df['ydot']
    actual_vz = df['zdot']
    actual_vel = np.array([actual_vx, actual_vy, actual_vz]).T
    # actual_q
    qx = df['qx']
    qy = df['qy']
    qz = df['qz']
    qw = df['qw']
    q = np.array([qx, qy, qz, qw]).T
    actual_yaw = compute_yaw_from_quaternion(q)

    # desired_pos
    des_x = df['xdes']
    des_y = df['ydes']
    des_z = df['zdes']
    des_pos = np.array([des_x, des_y, des_z]).T
    # desired_vel
    des_vx = df['xdotdes']
    des_vy = df['ydotdes']
    des_vz = df['zdotdes']
    des_vel = np.array([des_vx, des_vy, des_vz]).T
    # desired_q
    des_qx = df['qxdes']
    des_qy = df['qydes']
    des_qz = df['qzdes']
    des_qw = df['qwdes']
    q_des = np.array([des_qx, des_qy, des_qz, des_qw]).T
    desired_yaw = compute_yaw_from_quaternion(q_des)
    # desire thrust
    cmd_thrust = df['thrustdes']                            # Desired thrust
    # desire moment
    mxdes = df['mxdes']
    mydes = df['mydes']
    mzdes = df['mzdes']
    cmd_moment = np.array([mxdes, mydes, mzdes]).T

    # Cost components by cumulative sum of squared norms
    position_error = np.linalg.norm(actual_pos - des_pos, axis=1)**2
    velocity_error = np.linalg.norm(actual_vel - des_vel, axis=1)**2
    yaw_error = (actual_yaw - desired_yaw)**2
    # print(f"yaw_error: {yaw_error}")
    tracking_cost = position_error + velocity_error + yaw_error

    # control effort
    thrust_error = cmd_thrust**2
    moment_error = np.linalg.norm(cmd_moment, axis=1)**2
    control_cost = thrust_error + moment_error

    sim_cost = np.sum(tracking_cost + robust_c * control_cost)

    return sim_cost

def main():
    trial_dir = '/workspace/data_output/trial_data'
    column_name = 'cost'
    output_file = '/workspace/data_output/data.csv'

    rho = 1.0

    # Get a list of all trial files in the directory
    trial_files = [file for file in os.listdir(trial_dir) if file.endswith('.csv')]

    # Iterate over each trial file
    for file in trial_files:
        file_path = os.path.join(trial_dir, file)
        
        # Extract the index from the filename
        traj_number = int(file.split('_')[-1].split('.')[0])
        
        # Load the trial file into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Compute the cost from the specific column
        cost = compute_cost(df, robust_c= rho)
        
        # Update the cost in the output file based on the traj_number
        output_df = pd.read_csv(output_file)
        output_df.loc[output_df['traj_number'] == traj_number, 'cost'] = cost
        output_df.to_csv(output_file, index=False)
        
        # Print the cost for each trial file
        print(f"Cost for {file}: {cost}")

if __name__ == '__main__':
    main()