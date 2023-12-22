#! /usr/bin/env python3

"""
Generate training data
simple replanning with lissajous trajectory with fixed waypoints
"""

import csv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D projection

# import rospy
import numpy as np
import random

from trajgen import quadratic, sgd_jax 
from examples.verify_inference_1201 import VerifyInference

import torch
import pickle
import sys

import ruamel.yaml as yaml
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP
from model_learning import restore_checkpoint
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jax.numpy as jnp

import transforms3d.euler as euler
from itertools import accumulate

from scipy.spatial.transform import Rotation as R
import time
from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.minsnap_nn import MinSnap
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.world import World
import pandas as pd

# from trajgen.nonlinear import _coeff_constr_A, _coeff_constr_b
# from trajgen.trajutils import _cost_matrix
import jax.scipy.linalg as spl
from rotorpy.trajectories.minsnap_nn import H_fun, get_1d_constraints, cvxopt_solve_qp
from flax.linen import jvp
from trajgen.sgd_jax import modify_reference
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss
import flax

gamma = 1

PI = np.pi

def check_trajectory_constraints(sim_result, num_samples=100):
    """
    Check if the trajectory satisfies specific constraints (e.g., z > 0, -2 < x, y < 2)
    Inputs:
        sim_result: Simulation result containing state information.
        num_samples: Number of points along the trajectory to check.
    Returns:
        valid_trajectory: True if trajectory satisfies the constraints, False otherwise.
    """
    time_steps = np.linspace(0, sim_result['time'][-1], num_samples)

    for t in time_steps:
        # Find the closest time index in sim_result to the current time step
        closest_idx = np.argmin(np.abs(sim_result['time'] - t))
        x, y, z = sim_result['state']['x'][closest_idx]

        # Check constraints
        if not (z > 0 and -2 < x < 2 and -2 < y < 2):
            return False

    return True


def sample_waypoints(num_waypoints, world, world_buffer=2, check_collision=True, min_distance=1, max_distance=3, max_attempts=1000, start_waypoint=None, end_waypoint=None, rng=None, seed=None):
    """
    Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that 
    the path you generate with these waypoints will be collision free. 
    Inputs:
        num_waypoints: Number of waypoints to sample. 
        world: Instance of World class containing the map extents and any obstacles. 
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away 
            from the edge of the world.
        check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script. 
        min_distance: Minimum distance between waypoints consecutive waypoints. 
        max_distance: Maximum distance between consecutive waypoints.
        max_attempts: Maximum number of attempts to sample a waypoint.
        start_waypoint: If specified, the first waypoint will be this point. 
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
    """

    if min_distance > max_distance:
        raise Exception("min_distance must be less than or equal to max_distance.")

    def check_distance(waypoint, waypoints, min_distance, max_distance):
        """
        Checks if the waypoint is at least min_distance away from all other waypoints. 
        Inputs:
            waypoint: The waypoint to check. 
            waypoints: A list of waypoints. 
            min_distance: The minimum distance the waypoint must be from all other waypoints. 
            max_distance: The maximum distance the waypoint can be from all other waypoints.
        Outputs:
            collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise. 
        """
        collision = False
        for w in waypoints:
            if (np.linalg.norm(waypoint-w) < min_distance) or (np.linalg.norm(waypoint-w) > max_distance):
                collision = True
        return collision
    
    def check_obstacles(waypoint, occupancy_map):
        """
        Checks if the waypoint is colliding with any obstacles in the world. 
        Inputs:
            waypoint: The waypoint to check. 
            occupancy_map: An instance of the occupancy map.
        Outputs:
            collision: True if the waypoint is colliding with any obstacles in the world. False otherwise. 
        """
        collision = False
        if occupancy_map.is_occupied_metric(waypoint):
            collision = True
        return collision
    
    def single_sample(world, current_waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts=1000, rng=None, seed=None):
        """
        Samples a single waypoint. 
        Inputs:
            world: Instance of World class containing the map extents and any obstacles. 
            world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away 
                from the edge of the world.
            occupancy_map: An instance of the occupancy map.
            min_distance: Minimum distance between waypoints consecutive waypoints. 
            max_distance: Maximum distance between consecutive waypoints.
            max_attempts: Maximum number of attempts to sample a waypoint.
            rng: Random number generator. If None, uses numpy's random number generator.
            seed: Seed for the random number generator.
        Outputs:
            waypoint: A single (x,y,z) waypoint. 
        """

        if seed is not None:
            np.random.seed(seed)

        num_attempts = 0

        world_lower_limits = np.array(world.world['bounds']['extents'][0::2])+world_buffer
        world_upper_limits = np.array(world.world['bounds']['extents'][1::2])-world_buffer

        if len(current_waypoints) == 0:
            max_distance_lower_limits = world_lower_limits
            max_distance_upper_limits = world_upper_limits
        else:
            max_distance_lower_limits = current_waypoints[-1] - max_distance
            max_distance_upper_limits = current_waypoints[-1] + max_distance

        lower_limits = np.max(np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0)
        upper_limits = np.min(np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0)

        waypoint = np.random.uniform(low=lower_limits, 
                                     high=upper_limits, 
                                     size=(3,))
        # or waypoint[2] <= 0 or abs(waypoint[0]) >= 2 or abs(waypoint[1]) >= 2
        while check_obstacles(waypoint, occupancy_map) or (check_distance(waypoint, current_waypoints, min_distance, max_distance) if occupancy_map is not None else False):
            waypoint = np.random.uniform(low=lower_limits, 
                                         high=upper_limits, 
                                         size=(3,))
            # make waypoint shift above by 0.3
            waypoint[2] += 0.3


            num_attempts += 1
            if num_attempts > max_attempts:
                raise Exception("Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(max_attempts, check_obstacles(waypoint, occupancy_map), check_distance(waypoint, current_waypoints, min_distance, max_distance)))
        return waypoint
    
    ######################################################################################################################

    waypoints = []

    if check_collision:
        # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
        occupancy_map = OccupancyMap(world=world, resolution=[0.5, 0.5, 0.5], margin=0.1)
    else:
        occupancy_map = None

    if start_waypoint is not None: 
        waypoints = [start_waypoint]
    else:  
        # Randomly sample a start waypoint.
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts, rng, seed))
        
    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts, rng, seed))

    if end_waypoint is not None:
        waypoints.append(end_waypoint)

    return np.array(waypoints)

def sample_yaw(seed, waypoints, yaw_min=-np.pi, yaw_max=np.pi):
    """
    Samples random yaw angles for the waypoints.
    """
    np.random.seed(seed)
    yaw_angles = np.random.uniform(low=yaw_min, high=yaw_max, size=len(waypoints))
    
    return yaw_angles

def compute_cost(sim_result):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory. 

    time = sim_result['time']
    x = sim_result['state']['x']                                    # Position
    v = sim_result['state']['v']                                    # Velocity
    q = sim_result['state']['q']                                    # Attitude
    w = sim_result['state']['w']                                    # Body rates
    rotor_speeds = sim_result['state']['rotor_speeds']              # Rotor speeds

    x_des = sim_result['flat']['x']                                 # Desired position
    v_des = sim_result['flat']['x_dot']                             # Desired velocity
    q_des = sim_result['control']['cmd_q']                          # Desired attitude
    rotor_speeds_des = sim_result['control']['cmd_motor_speeds']    # Desired rotor speeds 
    cmd_thrust = sim_result['control']['cmd_thrust']                # Desired thrust 
    cmd_moment = sim_result['control']['cmd_moment']                # Desired body moment

    # Cost components
    position_error = np.linalg.norm(x - x_des, axis=1).mean()
    velocity_error = np.linalg.norm(v - v_des, axis=1).mean()

    # Calculate attitude error (assuming quaternion representation)
    # attitude_error = np.linalg.norm(q - q_des, axis=1).mean()  # Modify this based on your attitude representation
    # Attitude error (for quaternions)
    # attitude_error = quaternion_distance(q, q_des).mean()

    # Input cost from thrust and body moment
    # Compute total cost as a weighted sum of tracking errors
    rho_position, rho_velocity, rho_attitude, rho_thrust, rho_moment = 1.0, 1.0, 1.0, 1.0, 1.0  # Adjust these weights as needed
    sim_cost = (rho_position * position_error +
                  rho_velocity * velocity_error)

    return sim_cost


def get_H_A_b_coeffs(points, delta_t, v_avg, t_keyframes, poly_degree, yaw_poly_degree, v_max, v_start, v_end, yaw, yaw_rate_max):
    """
    """
    # Compute the distances between each waypoint.
    seg_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
    seg_mask = np.append(True, seg_dist > 1e-1)
    points = points[seg_mask, :]

    null = False

    m = points.shape[0] - 1  # Get the number of segments

    # If two or more waypoints remain, solve min snap
    if points.shape[0] >= 2:
        ################## Time allocation
        delta_t = (
            seg_dist / v_avg
        )  # Compute the segment durations based on the average velocity
        t_keyframes = np.concatenate(
            ([0], np.cumsum(delta_t))
        )  # Construct time array which indicates when the quad should be at the i'th waypoint.

        ################## Cost function
        # First get the cost segment for each matrix:
        H_pos = [H_fun(delta_t[i], k=poly_degree) for i in range(m)]
        H_yaw = [H_fun(delta_t[i], k=yaw_poly_degree) for i in range(m)]

        # Now concatenate these costs using block diagonal form:
        P_pos = spl.block_diag(*H_pos)
        P_yaw = spl.block_diag(*H_yaw)

        # Lastly the linear term in the cost function is 0
        q_pos = np.zeros(((poly_degree + 1) * m, 1))
        q_yaw = np.zeros(((yaw_poly_degree + 1) * m, 1))

        ################## Constraints for each axis
        (Ax, bx, Gx, hx) = get_1d_constraints(
            points[:, 0],
            delta_t,
            m,
            k=poly_degree,
            vmax=v_max,
            vstart=v_start[0],
            vend=v_end[0],
        )
        (Ay, by, Gy, hy) = get_1d_constraints(
            points[:, 1],
            delta_t,
            m,
            k=poly_degree,
            vmax=v_max,
            vstart=v_start[1],
            vend=v_end[1],
        )
        (Az, bz, Gz, hz) = get_1d_constraints(
            points[:, 2],
            delta_t,
            m,
            k=poly_degree,
            vmax=v_max,
            vstart=v_start[2],
            vend=v_end[2],
        )
        (Ayaw, byaw, Gyaw, hyaw) = get_1d_constraints(
            yaw, delta_t, m, k=yaw_poly_degree, vmax=yaw_rate_max
        )

        ################## Solve for x, y, z, and yaw

        ### Only in the fully constrained situation is there a unique minimum s.t. we can solve the system Ax = b.
        # c_opt_x = np.linalg.solve(Ax,bx)
        # c_opt_y = np.linalg.solve(Ay,by)
        # c_opt_z = np.linalg.solve(Az,bz)
        # c_opt_yaw = np.linalg.solve(Ayaw,byaw)

        ### Otherwise, in the underconstrained case or when inequality constraints are given we solve the QP.
        c_opt_x = cvxopt_solve_qp(P_pos, q=q_pos, G=Gx, h=hx, A=Ax, b=bx)
        c_opt_y = cvxopt_solve_qp(P_pos, q=q_pos, G=Gy, h=hy, A=Ay, b=by)
        c_opt_z = cvxopt_solve_qp(P_pos, q=q_pos, G=Gz, h=hz, A=Az, b=bz)
        c_opt_yaw = cvxopt_solve_qp(P_yaw, q=q_yaw, G=Gyaw, h=hyaw, A=Ayaw, b=byaw)
        # # print the number of coeffs
        # print("Number of Coefficients for each trajectory dimension:")
        # print(f"X-axis: {len(c_opt_x)} coefficients")
        # print(f"Y-axis: {len(c_opt_y)} coefficients")
        # print(f"Z-axis: {len(c_opt_z)} coefficients")
        # print(f"Yaw: {len(c_opt_yaw)} coefficients")
    
        # call modify_reference directly after computing the min snap coeffs and use the returned coeffs in the rest of the class
        # self.nan_encountered = False

        # if use_neural_network:
        min_snap_coeffs = np.concatenate([c_opt_x, c_opt_y, c_opt_z, c_opt_yaw])

        # get H by concatenating H_pos and H_yaw
        H = spl.block_diag(
            *[
                0.5 * (P_pos.T + P_pos),
                0.5 * (P_pos.T + P_pos),
                0.5 * (P_pos.T + P_pos),
                0.5 * (P_yaw.T + P_yaw),
            ]
        )  # cost fuction is the same for x, y, z

        # get A by concatenating Ax, Ay, Az, Ayaw
        A = spl.block_diag(*[Ax, Ay, Az, Ayaw])

        # get b by concatenating bx, by, bz, byaw
        b = np.concatenate((bx, by, bz, byaw))

        return H, A, b

def write_to_csv(output_file, row):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return None

# Function to run simulation and compute cost
def run_simulation_and_compute_cost(waypoints, yaw_angles, vavg, use_neural_network, regularizer=None, vehicle=None, controller=None):
    traj = MinSnap(points=waypoints, yaw_angles=yaw_angles, v_avg=vavg, use_neural_network=use_neural_network, regularizer=regularizer)
    # (H, A, b, min_snap_coeffs) = MinSnap(points=waypoints, yaw_angles=yaw_angles, v_avg=vavg, use_neural_network=use_neural_network, regularizer=regularizer)
    # H = traj.H
    # A = traj.A
    # b = traj.b
    # min_snap_coeffs = traj.min_snap_coeffs
    # cost_mat = spl.block_diag(*[_cost_matrix(order, 4, d) for d in durations])
    # # print("cost_mat!!!!!!!!!!!!!!!!!", cost_mat)
    # A_coeff = _coeff_constr_A(ts, n, num_coeffs)
    # b_coeff = _coeff_constr_b(wp.T, ts, n)

    # cost_mat_full = spl.block_diag(*[cost_mat for i in range(p)])
    # A_coeff_full = spl.block_diag(*[A_coeff for i in range(p)])
    # b_coeff_full = jnp.ravel(b_coeff)

    # nn_coeff, pred, nan_encountered = jvp(modify_reference, regularizer,
    #                 jnp.zeros(H.shape),
    #                 jnp.zeros(A.shape),
    #                 jnp.zeros(b.shape),
    #                 jnp.zeros(min_snap_coeffs.shape))
    # nn_coeff, pred, nan_encountered = modify_reference(
    #                 regularizer,
    #                 jnp.zeros(H.shape),
    #                 jnp.zeros(A.shape),
    #                 jnp.zeros(b.shape),
    #                 jnp.zeros(min_snap_coeffs.shape)
    #             )

    # def nn_cost(coeffs):
    #     return coeffs.T @ H @ coeffs #+ regularizer(coeffs)[0]
    
    
    # def projection(coeffs):
    #     return coeffs - A.T @ np.linalg.inv(A @ A.T) @ (A @ coeffs - b)
    
    # grad_fn = lambda coeffs: jax.grad(nn_cost)(coeffs)

    
    # # def my_pgd_solver(coeffs, maxiter, stepsize, tol, rho, beta, init_alpha, projection, grad, f):
        

    # methods = {"PGD": cs.ProjectedGD(nn_cost, grad_fn, projection, ss.Backtracking(rule_type="Armijo", rho=0.99, beta=0.1, init_alpha=1.))
    #       }
    
    # max_iter = 50
    # tol = 1e-1

    # x = methods["PGD"].solve(x0=np.zeros(min_snap_coeffs.shape), max_iter=max_iter, tol=tol, disp=1)
    # print(x)

    nan_encountered = traj.nan_encountered  # Flag indicating if NaN was encountered

    sim_instance = Environment(vehicle=vehicle, controller=controller, trajectory=traj, wind_profile=None, sim_rate=100)

    # Set initial state
    x0 = {'x': waypoints[0],
          'v': np.zeros(3,),
          'q': np.array([0, 0, 0, 1]),  # quaternion
          'w': np.zeros(3,),
          'wind': np.array([0,0,0]),
          'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    sim_instance.vehicle.initial_state = x0

    waypoint_times = traj.t_keyframes
    sim_result = sim_instance.run(t_final=traj.t_keyframes[-1], use_mocap=False, terminate=False, plot=False)

    # visualize the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sim_result['state']['x'][:, 0], sim_result['state']['x'][:, 1], sim_result['state']['x'][:, 2], 'b--')
    ax.plot(sim_result['flat']['x'][:, 0], sim_result['flat']['x'][:, 1], sim_result['flat']['x'][:, 2], 'r-.')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='k', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # save the figure if using nn
    plt.savefig("/workspace/data_output/sim_figures_safety/trajectory"+str(use_neural_network)+".png")

    # Check the trajectory constraints after simulation
    valid_trajectory = all(z > 0 and -2 < x < 2 and -2 < y < 2 for x, y, z in sim_result['state']['x'])
    # valid_trajectory if using nn
    print("valid_trajectory: ", valid_trajectory, "; if using nn: ", use_neural_network)

    if not valid_trajectory or nan_encountered:
        return None, None, None, False, None

    trajectory_cost = compute_cost(sim_result)
    print(trajectory_cost)

    # Now extract the polynomial coefficients for the trajectory.
    pos_poly = traj.x_poly
    yaw_poly = traj.yaw_poly

    summary_output = np.concatenate((np.array([trajectory_cost]), pos_poly.ravel(), yaw_poly.ravel(), waypoints.ravel()))

    return sim_result, trajectory_cost, waypoint_times, nan_encountered, traj, summary_output

def plot_results(sim_result_init, sim_result_nn, waypoints, initial_cost, predicted_cost, filename=None, waypoints_time=None):
        # Compute yaw angles from quaternions
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

    actual_yaw_init = compute_yaw_from_quaternion(sim_result_init['state']['q'])
    actual_yaw_nn = compute_yaw_from_quaternion(sim_result_nn['state']['q'])

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot with waypoints
    ax_traj = fig.add_subplot(121, projection="3d")
    ax_traj.plot3D(sim_result_init["state"]["x"][:, 0], sim_result_init["state"]["x"][:, 1], sim_result_init["state"]["x"][:, 2], 'b--')
    ax_traj.plot3D(sim_result_init["flat"]["x"][:, 0], sim_result_init["flat"]["x"][:, 1], sim_result_init["flat"]["x"][:, 2], 'r-.')
    ax_traj.plot3D(sim_result_nn["state"]["x"][:, 0], sim_result_nn["state"]["x"][:, 1], sim_result_nn["state"]["x"][:, 2], 'g:')
    ax_traj.plot3D(sim_result_nn["flat"]["x"][:, 0], sim_result_nn["flat"]["x"][:, 1], sim_result_nn["flat"]["x"][:, 2], 'm')
    ax_traj.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='k', marker='o')
    ax_traj.set_title("3D Trajectories with Waypoints", fontsize=18)
    ax_traj.set_xlabel("X", fontsize=16)
    ax_traj.set_ylabel("Y", fontsize=16)
    ax_traj.set_zlabel("Z", fontsize=16)
    ax_traj.set_zlim(0, 1)
    ax_traj.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref', 'Waypoints'], fontsize=14)
    cost_text = f"Initial Cost: {initial_cost:.2f}\nSimulated Cost: {predicted_cost:.2f}"
    ax_traj.text2D(0.04, 0.02, cost_text, transform=ax_traj.transAxes, fontsize=18)
# cost_text = f"Initial Cost: {initial_cost:.2f}\nSimulated Cost: {predicted_cost:.2f}"
    # ax_traj.text2D(0.05, 0.95, cost_text, transform=ax_traj.transAxes, fontsize=16)

    # Subplots for X, Y, Z, Yaw
    gs = fig.add_gridspec(4, 2)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    ax_yaw = fig.add_subplot(gs[3, 1])

    # Subplot for X
    ax_x.plot(sim_result_init['time'], sim_result_init['state']['x'][:, 0], 'b--')
    ax_x.plot(sim_result_init['time'], sim_result_init['flat']['x'][:, 0], 'r-.')
    ax_x.plot(sim_result_nn['time'], sim_result_nn['state']['x'][:, 0], 'g:')
    ax_x.plot(sim_result_nn['time'], sim_result_nn['flat']['x'][:, 0], 'm')
    ax_x.set_title('X Position Over Time', fontsize=18)
    ax_x.set_xlabel('Time', fontsize=16)
    ax_x.set_ylabel('X Position', fontsize=16)
    # ax_x.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Y
    ax_y.plot(sim_result_init['time'], sim_result_init['state']['x'][:, 1], 'b--')
    ax_y.plot(sim_result_init['time'], sim_result_init['flat']['x'][:, 1], 'r-.')
    ax_y.plot(sim_result_nn['time'], sim_result_nn['state']['x'][:, 1], 'g:')
    ax_y.plot(sim_result_nn['time'], sim_result_nn['flat']['x'][:, 1], 'm')
    ax_y.set_title('Y Position Over Time', fontsize=18)
    ax_y.set_xlabel('Time', fontsize=16)
    ax_y.set_ylabel('Y Position', fontsize=16)
    # ax_y.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Z
    ax_z.plot(sim_result_init['time'], sim_result_init['state']['x'][:, 2], 'b--')
    ax_z.plot(sim_result_init['time'], sim_result_init['flat']['x'][:, 2], 'r-.')
    ax_z.plot(sim_result_nn['time'], sim_result_nn['state']['x'][:, 2], 'g:')
    ax_z.plot(sim_result_nn['time'], sim_result_nn['flat']['x'][:, 2], 'm')
    ax_z.set_title('Z Position Over Time', fontsize=18)
    ax_z.set_xlabel('Time', fontsize=16)
    ax_z.set_ylabel('Z Position', fontsize=16)
    # ax_z.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Adding keyframes to the subplots
    for ax, dim in zip([ax_x, ax_y, ax_z], [0, 1, 2]):
        ax.scatter(waypoints_time, waypoints[:, dim], c='k', marker='o', label='Waypoints')

    # Subplot for Yaw
    ax_yaw.plot(sim_result_init['time'], actual_yaw_init, 'b--')
    ax_yaw.plot(sim_result_init['time'], sim_result_init['flat']['yaw'], 'r-.')
    ax_yaw.plot(sim_result_nn['time'], actual_yaw_nn, 'g:')
    ax_yaw.plot(sim_result_nn['time'], sim_result_nn['flat']['yaw'], 'm')
    ax_yaw.set_title('Yaw Angle Over Time', fontsize=18)
    ax_yaw.set_xlabel('Time', fontsize=16)
    ax_yaw.set_ylabel('Yaw Angle', fontsize=16)
    # ax_yaw.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)
    ax_yaw.scatter(waypoints_time, np.zeros(len(waypoints)), c='k', marker='o', label='Waypoints')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

    # close the figure
    plt.close(fig)
    
def main():
    # Define the lists to keep track of times for the simulations
    times_nn = []
    times_mj = []
    times_poly = []

    # Initialize neural network
    rho = 0.1
    input_size = 96  # number of coeff
    # num_data = 72

    with open(r"/workspace/rotorpy/learning/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]

    # Load the trained model
    # model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    # print(model)
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    model_save = yaml_data["save_path"] + str(rho)
    trained_model_state = flax.core.freeze(restore_checkpoint(None, model_save, 7))
    vf = (model, trained_model_state["params"])

    # rng = jax.random.PRNGKey(427)
    # rng, inp_rng, init_rng = jax.random.split(rng, 3)
    # inp = jax.random.normal(
    #     inp_rng, (1, input_size)
    # )  # Batch size 32, input size 2012
    # # Initialize the model
    # params = model.init(init_rng, inp)
    # # params = model.init()

    # optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    # model_state = train_state.TrainState.create(
    #     apply_fn=model.apply, params=params, tx=optimizer
    # )

    # model_save = yaml_data["save_path"] + str(rho)
    # print("model_save", model_save)
    
    # trained_model_state = restore_checkpoint(model_state, model_save, 7)

    # print("Current model structure:", model)
    # print("Model parameters:", params)

    # Verify the path to the checkpoint
    # print("Checkpoint path:", model_save)

    # vf = model.bind(trained_model_state.params)
    # vf = trained_model_state

    # Define the quadrotor parameters
    world_size = 10
    num_waypoints = 4
    vavg = 2
    random_yaw = False
    yaw_min = -0.85*np.pi
    yaw_max = 0.85*np.pi

    world_buffer = 2
    min_distance = 1
    max_distance = min_distance+3
    start_waypoint = None               # If you want to start at a specific waypoint, specify it using [xstart, ystart, zstart]
    end_waypoint = None                 # If you want to end at a specific waypoint, specify it using [xend, yend, zend]

    # Now create the world, vehicle, and controller objects.
    world = World.empty([-world_size/2, world_size/2, -world_size/2, world_size/2, -world_size/2, world_size/2])
    vehicle = Multirotor(quad_params)
    controller = SE3Control(quad_params)
    controller_with_drag_compensation = SE3Control(quad_params, drag_compensation=True)

    traj_data = pd.DataFrame(columns=['Seed', 'Waypoints', 'Coefficients'])

    # Loop for 100 trajectories
    # for i in range(20):
    valid_trajectory = False
    while not valid_trajectory:
        # # Sample waypoints
        waypoints = sample_waypoints(num_waypoints=num_waypoints, world=world, world_buffer=world_buffer, 
                                        min_distance=min_distance, max_distance=max_distance, 
                                        start_waypoint=start_waypoint, end_waypoint=end_waypoint, rng=None, seed=29)

        # waypoints = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        # waypoints = np.array([[0, 0, 0], [0.5, 0, 4], [1, 1.5, 4], [0, 1.5, 3]])
        
        # Sample yaw angles
        # yaw_angles = sample_yaw(seed=i, waypoints=waypoints, yaw_min=yaw_min, yaw_max=yaw_max)

        yaw_angles_zero = np.zeros(len(waypoints))

        # /workspace/rotorpy/rotorpy/sim_figures/
        figure_path = "/workspace/data_output/sim_figures_safety"

        sim_result_init, trajectory_cost_init, waypoints_time, _, _, summary_init = run_simulation_and_compute_cost(waypoints, yaw_angles_zero, vavg, use_neural_network=False, regularizer=vf, vehicle=vehicle, controller=controller)
        write_to_csv(figure_path + "/summary_init.csv", summary_init)
        sim_result_nn, trajectory_cost_nn,_,nan_encountered, traj, summary_nn = run_simulation_and_compute_cost(waypoints, yaw_angles_zero, vavg, use_neural_network=True, regularizer=vf, vehicle=vehicle, controller=controller)
        write_to_csv(figure_path + "/summary_nn.csv", summary_nn)
        
        if not nan_encountered and traj is not None and sim_result_init is not None and sim_result_nn is not None:
            valid_trajectory = True
            plot_results(sim_result_init, sim_result_nn, waypoints, trajectory_cost_init, trajectory_cost_nn, filename=figure_path + f"/trajectory_{1}.png", waypoints_time=waypoints_time)
            coeffs = traj.get_coefficients()  # Ensure this method exists in MinSnap
            traj_data = traj_data.append({'Seed': 1, 'Waypoints': waypoints.tolist(), 'Coefficients': coeffs.tolist()}, ignore_index=True)

    # save to figure_path 
    traj_data.to_csv(figure_path + "/traj_data.csv", index=False)

    """
        # run simulation and compute cost for the initial trajectory
        sim_result_init, trajectory_cost_init, waypoints_time, _ = run_simulation_and_compute_cost(waypoints, yaw_angles_zero, vavg, use_neural_network=False, regularizer=None, vehicle=vehicle, controller=controller)
        # run simulation and compute cost for the modified trajectory
        sim_result_nn, trajectory_cost_nn,_,nan_encountered = run_simulation_and_compute_cost(waypoints, yaw_angles_zero, vavg, use_neural_network=True, regularizer=vf, vehicle=vehicle, controller=controller)
        print("nan_encountered in inference", nan_encountered)
        if nan_encountered == False:
            print(f"Trajectory {i} initial cost: {trajectory_cost_init}")
            print(f"Trajectory {i} neural network modified cost: {trajectory_cost_nn}")
            cost_diff = trajectory_cost_nn - trajectory_cost_init
            cost_differences.append((trajectory_cost_init, trajectory_cost_nn, cost_diff))
            plot_results(sim_result_init, sim_result_nn, waypoints, trajectory_cost_init, trajectory_cost_nn, filename=figure_path + f"/trajectory_{i}.png", waypoints_time=waypoints_time)

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Elapsed time for trajectory {i}: {elapsed_time} seconds")

    # Save the cost data to a CSV file
    costs_df = pd.DataFrame(cost_differences, columns=['Initial Cost', 'NN Modified Cost', 'Cost Difference'])
    costs_df.to_csv("/workspace/data_output/cost_data.csv", index=False)
    """
    # costs_df = pd.read_csv("/workspace/data_output/cost_data.csv")
    # plt.figure()
    # plt.boxplot(costs_df['Cost Difference'], showfliers=False)  # Set showfliers=False to hide outliers
    # plt.title('Predicted Cost - Initial Cost Boxplot')
    # plt.ylabel('Cost Difference')
    # plt.savefig(figure_path + "/cost_difference_boxplot_no_outliers.png")
    # plt.close()


if __name__ == "__main__":
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()
