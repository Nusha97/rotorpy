# import pandas as pd
# import matplotlib.pyplot as plt

# # rho = 0.5
# # costs_df = pd.read_csv("/workspace/data_output/sim_figures_40/cost_data.csv")
# figure_path = f"/workspace/data_output/sim_figures_40"
# costs_df = pd.read_csv(figure_path + "/cost_data.csv")

# # Calculate the Our Cost to Minsnap Cost ratio
# costs_df['Our Cost to Minsnap Cost Ratio'] = costs_df['NN Modified Cost'] / costs_df['Initial Cost']

# # Plot Our Cost to Minsnap Cost ratio (Boxplot)
# plt.figure()
# plt.boxplot(costs_df['Our Cost to Minsnap Cost Ratio'], showfliers=False)
# plt.title('Our Cost to Minsnap Cost Ratio Boxplot')
# plt.ylabel('Our Cost to Minsnap Cost Ratio')
# # num_inference_trajs should be the number of inference trajectories without the outlier
# plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
# plt.savefig(figure_path + "/our_cost_to_minsnap_cost_ratio_boxplot_without_outliers.png")
# plt.close()

# # # Plot Our Cost to Minsnap Cost ratio (Boxplot without highest outlier)
# # plt.figure()
# # boxplot = plt.boxplot(costs_df['Our Cost to Minsnap Cost Ratio'], showfliers=True)
# # outliers = boxplot['fliers'][0].get_ydata()
# # max_outlier = max(outliers)
# # outliers[outliers == max_outlier] = None
# # boxplot['fliers'][0].set_ydata(outliers)
# # plt.title(f'Robust: {rho} - Our Cost to Minsnap Cost Ratio Boxplot')
# # plt.ylabel('Our Cost to Minsnap Cost Ratio')
# # num_inference_trajs = len(costs_df) - 1 # Subtract 1 for the outlier
# # # change the y axis range
# # plt.ylim([-1, 8])
# # plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
# # plt.savefig(figure_path + "/our_cost_to_minsnap_cost_ratio_boxplot_without_outliers.png")
# # plt.close()


# # # Plot Our Cost - Minsnap Cost (Boxplot)
# # plt.figure()
# # plt.boxplot(costs_df['Cost Difference'], showfliers=True) 
# # plt.title(f'Robust: {rho} - Our Cost - Minsnap Cost Boxplot')
# # plt.ylabel('Cost Difference')
# # plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
# # plt.savefig(figure_path + "/cost_difference_boxplot_with_outliers.png")
# # plt.close()

# # # Plot Our Cost to Minsnap Cost ratio (Violin plot)
# # plt.figure()
# # plt.violinplot(costs_df['Our Cost to Minsnap Cost Ratio'], showextrema=True)
# # plt.title(f'Robust: {rho} - Our Cost to Minsnap Cost Ratio Violin Plot')
# # plt.ylabel('Our Cost to Minsnap Cost Ratio')
# # plt.xlabel(f'Number of Inference Trajectories: {num_inference_trajs}')
# # plt.savefig(figure_path + "/our_cost_to_minsnap_cost_ratio_violin_plot.png")
# # plt.close()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data file
figure_path = "/workspace/data_output/sim_figures_30"
costs_df = pd.read_csv(figure_path + "/cost_data.csv")

# Calculate the Our Cost to Minsnap Cost ratio
costs_df['Our Cost to Minsnap Cost Ratio'] = costs_df['NN Modified Cost'] / costs_df['Initial Cost']

# Identify and remove outliers
Q1 = costs_df['Our Cost to Minsnap Cost Ratio'].quantile(0.25)
Q3 = costs_df['Our Cost to Minsnap Cost Ratio'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = costs_df[~((costs_df['Our Cost to Minsnap Cost Ratio'] < (Q1 - 1.5 * IQR)) | (costs_df['Our Cost to Minsnap Cost Ratio'] > (Q3 + 1.5 * IQR)))]

# num_inference_trajs should be the number of inference trajectories without the outlier
num_inference_trajs = len(filtered_df)

# Plot Our Cost to Minsnap Cost ratio (Boxplot without outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(y=filtered_df['Our Cost to Minsnap Cost Ratio'], width=0.3, palette='Set2', showfliers=False, 
            showmeans=True, meanprops={"marker":"D", "markerfacecolor":"black", "markeredgecolor":"black"})
plt.title('Boxplot for Cost Ratio')
plt.ylabel('Cost Ratio')
plt.xticks([0], [f'Number of Inference Trajectories: {num_inference_trajs}'])


# Save the plot
plt.savefig(figure_path + "/cost_ratio_boxplot_without_outliers.png")
plt.close()
