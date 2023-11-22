import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory containing the trajectory data
data_dir = '/workspace/data_output/trial_data'

# Directory to save the plots
plots_dir = '/workspace/data_output/plots'

# Ensure the plots directory exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Function to plot a trajectory from a CSV file
def plot_trajectory(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extracting the reference and actual trajectory data
    x_des, y_des, z_des = df['xdes'].to_numpy(), df['ydes'].to_numpy(), df['zdes'].to_numpy()
    x_act, y_act, z_act = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the trajectories
    ax.plot(x_des, y_des, z_des, label='Reference Trajectory', color='blue')
    ax.plot(x_act, y_act, z_act, label='Actual Trajectory', color='red')

    # Customizing the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()

    # Save the plot in the specified plots directory
    plot_file_name = os.path.splitext(os.path.basename(csv_file))[0] + '_trajectory_plot.png'
    plot_file = os.path.join(plots_dir, plot_file_name)
    plt.savefig(plot_file)
    plt.close()

    return plot_file

# Plotting and saving trajectories for each trial file
plot_files = []
for i in range(1000):
    csv_file = os.path.join(data_dir, f'trial_{i}.csv')
    if os.path.exists(csv_file):
        plot_file = plot_trajectory(csv_file)
        plot_files.append(plot_file)

plot_files
