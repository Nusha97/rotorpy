#! /usr/bin/env python3

"""
Generate training data
simple replanning with lissajous trajectory with fixed waypoints
"""

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
from mlp import MLP, MLP_torch
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

gamma = 1

PI = np.pi

def sample_waypoints(num_waypoints, world, world_buffer=2, check_collision=True, min_distance=1, max_distance=3, max_attempts=1000, start_waypoint=None, end_waypoint=None):
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
    
    def single_sample(world, current_waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts=1000, rng=None):
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
        Outputs:
            waypoint: A single (x,y,z) waypoint. 
        """

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
        while check_obstacles(waypoint, occupancy_map) or (check_distance(waypoint, current_waypoints, min_distance, max_distance) if occupancy_map is not None else False):
            waypoint = np.random.uniform(low=lower_limits, 
                                         high=upper_limits, 
                                         size=(3,))
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
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts))
        
    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(single_sample(world, waypoints, world_buffer, occupancy_map, min_distance, max_distance, max_attempts))

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

# Function to run simulation and compute cost
def run_simulation_and_compute_cost(waypoints, yaw_angles, vavg, use_neural_network, regularizer=None, vehicle=None, controller=None):
    traj = MinSnap(points=waypoints, yaw_angles=yaw_angles, v_avg=vavg, use_neural_network=use_neural_network, regularizer=regularizer)

    sim_instance = Environment(vehicle=vehicle, controller=controller, trajectory=traj, wind_profile=None, sim_rate=100)

    # Set initial state
    x0 = {'x': waypoints[0],
          'v': np.zeros(3,),
          'q': np.array([0, 0, 0, 1]),  # quaternion
          'w': np.zeros(3,),
          'wind': np.array([0,0,0]),
          'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    sim_instance.vehicle.initial_state = x0

    sim_result = sim_instance.run(t_final=traj.t_keyframes[-1], use_mocap=False, terminate=False, plot=False)
    trajectory_cost = compute_cost(sim_result)

    return sim_result, trajectory_cost

def plot_results(sim_result, filename=None):
    # Plotting the results of sim_result['state']['x'] and sim_result['flat']['x'] shows the actual trajectory and the reference trajectory
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    axes.plot3D(
        sim_result["state"]["x"][:, 0],
        sim_result["state"]["x"][:, 1],
        sim_result["state"]["x"][:, 2],
        "b",
    )
    axes.plot3D(
        sim_result["flat"]["x"][:, 0],
        sim_result["flat"]["x"][:, 1],
        sim_result["flat"]["x"][:, 2],
        "r",
    )
    # put legend
    axes.legend(["actual_traj", "ref_traj"])
    axes.set_xlim(-6, 6)
    axes.set_zlim(-6, 6)
    axes.set_ylim(-6, 6)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    title = "ref_traj vs actual_traj"
    axes.set_title(title)
    if filename is not None:
        plt.savefig(filename)
    # plt.show()

def main():
    # Define the lists to keep track of times for the simulations
    times_nn = []
    times_mj = []
    times_poly = []

    # Initialize neural network
    rho = 1
    input_size = 96  # number of coeff
    # num_data = 72

    with open(r"/workspace/rotorpy/learning/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]

    # Load the trained model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(
        inp_rng, (1, input_size)
    )  # Batch size 32, input size 2012
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    model_save = yaml_data["save_path"] + str(rho)
    print("model_save", model_save)
    
    # print("Current model structure:", model)
    # print("Model parameters:", params)

    # # Verify the path to the checkpoint
    # print("Checkpoint path:", model_save)
    
    trained_model_state = restore_checkpoint(model_state, model_save)

    vf = model.bind(trained_model_state.params)

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

    # Loop for 100 trajectories
    for i in range(100):
        # Sample waypoints
        waypoints = sample_waypoints(num_waypoints=num_waypoints, world=world, world_buffer=world_buffer, 
                                        min_distance=min_distance, max_distance=max_distance, 
                                        start_waypoint=start_waypoint, end_waypoint=end_waypoint)
        
        # Sample yaw angles
        yaw_angles = sample_yaw(seed=427, waypoints=waypoints, yaw_min=yaw_min, yaw_max=yaw_max)

        # /workspace/rotorpy/rotorpy/sim_figures/
        figure_path = "/workspace/rotorpy/rotorpy/sim_figures/"

        # visualize the waypoints
        fig = plt.figure()
        axes = fig.add_subplot(111, projection="3d")
        axes.plot3D(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            "*",
        )
        axes.set_xlim(-6, 6)
        axes.set_zlim(0, 1)
        axes.set_ylim(-6, 6)
        plt.savefig(figure_path + "waypoints.png")

        # run simulation and compute cost for the initial trajectory
        sim_result_init, trajectory_cost_init = run_simulation_and_compute_cost(waypoints, yaw_angles, vavg, use_neural_network=False, regularizer=None, vehicle=vehicle, controller=controller)
        print(f"Trajectory {i} initial cost: {trajectory_cost_init}")
            
        # generate min_snap trajectory and run simulation
        plot_results(sim_result_init, filename=f"{figure_path}init_traj_{i}.png")

        # run simulation and compute cost for the modified trajectory
        sim_result_nn, trajectory_cost_nn = run_simulation_and_compute_cost(waypoints, yaw_angles, vavg, use_neural_network=True, regularizer=vf, vehicle=vehicle, controller=controller)
        print(f"Trajectory {i} neural network modified cost: {trajectory_cost_nn}")

        plot_results(sim_result_nn, filename=f"{figure_path}nn_modified_traj_{i}.png")


    """
    # plot the ref yaw angle over time to see if it's actually varying from the network
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(times_init_min_snap, ref_traj_init_min_snap[:, 3], "b")
    axes.plot(times_modified, ref_traj_modified[:, 3], "r")
    # print("ref_traj_init_min_snap", ref_traj_init_min_snap[:, 3])
    # axis limits to be between 0 and 2pi
    # axes.set_ylim(0, 2 * np.pi)
    # put legend
    axes.legend(["ref_traj_init_min_snap", "ref_traj_modified"])
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("ref yaw")
    plt.savefig("/workspace/rotorpy/rotorpy/sim_figures/init_yawref.png")
    # plt.show()

    # plot the yaw_ref angle over time to see if it's actually varying from the network
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(times_init_min_snap, yaw_ref_init_min_snap, "b")
    axes.plot(times_modified, yaw_ref_modified, "r")
    # print("yaw_ref_init_min_snap", yaw_ref_init_min_snap)
    # axis limits to be between 0 and 2pi
    # axes.set_ylim(0, 2 * np.pi)
    # put legend
    axes.legend(["yaw_ref_init_min_snap", "yaw_ref_modified"])
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("yaw_ref")
    plt.savefig("/workspace/rotorpy/rotorpy/sim_figures/nn_yawref.png")
    # plt.show()
    """


if __name__ == "__main__":
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()
