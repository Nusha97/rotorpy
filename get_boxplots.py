import pandas as pd
import matplotlib.pyplot as plt

costs_df = pd.read_csv("/workspace/data_output/cost_data.csv")
figure_path = "/workspace/data_output/sim_figures_40"

# Calculate the Simulation Cost to Initial Cost ratio
costs_df['Simulation Cost to Initial Cost Ratio'] = costs_df['NN Modified Cost'] / costs_df['Initial Cost']

# Plot Simulation Cost to Initial Cost ratio (Boxplot)
plt.figure()
plt.boxplot(costs_df['Simulation Cost to Initial Cost Ratio'], showfliers=True)
plt.title('Simulation Cost to Initial Cost Ratio Boxplot')
plt.ylabel('Simulation Cost to Initial Cost Ratio')
num_inference_trajs = len(costs_df)
plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
plt.savefig(figure_path + "/simulation_cost_to_initial_cost_ratio_boxplot_with_outliers.png")
plt.close()

# Plot Simulation Cost - Initial Cost (Boxplot)
plt.figure()
plt.boxplot(costs_df['Cost Difference'], showfliers=True) 
plt.title('Simulation Cost - Initial Cost Boxplot')
plt.ylabel('Cost Difference')
plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
plt.savefig(figure_path + "/cost_difference_boxplot_with_outliers.png")
plt.close()

# Plot Simulation Cost to Initial Cost ratio (Violin plot)
plt.figure()
plt.violinplot(costs_df['Simulation Cost to Initial Cost Ratio'], showextrema=True)
plt.title('Simulation Cost to Initial Cost Ratio Violin Plot')
plt.ylabel('Simulation Cost to Initial Cost Ratio')
plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
plt.savefig(figure_path + "/simulation_cost_to_initial_cost_ratio_violin_plot.png")
plt.close()
