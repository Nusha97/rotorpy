import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to load CSV for a given robust value and exclude outliers
def load_csv_for_robust(robust_value, base_path):
    file_path = f"{base_path}/sim_figures_with_rho{robust_value}/mean_cost_data.csv"
    df = pd.read_csv(file_path)
    df['Penalty Value'] = robust_value
    return df

# Define the robust values and base path
robust_values = [0, 0.1, 0.5, 1]
base_path = "/workspace/data_output"

# Load data for all robust values
data_frames = []
for rho in robust_values:
    df = load_csv_for_robust(rho, base_path)
    # Calculate the IQR to identify outliers
    Q1 = df['Cost Ratio'].quantile(0.25)
    Q3 = df['Cost Ratio'].quantile(0.75)
    IQR = Q3 - Q1
    # Filter out the outliers
    df = df[~((df['Cost Ratio'] < (Q1 - 1.5 * IQR)) | (df['Cost Ratio'] > (Q3 + 1.5 * IQR)))]
    data_frames.append(df)

# Combine all data frames
combined_df = pd.concat(data_frames)

# Plot the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Penalty Value', y='Cost Ratio', data=combined_df, palette='Set2', showfliers=False,
            showmeans=True, meanprops={"marker":"D", "markerfacecolor":"black", "markeredgecolor":"black"})

# Set the y-axis limit
plt.ylim(-1, 6)

# Title and labels
plt.title('Relative Cost Ratio for Ours over Minsnap', fontsize=20)
plt.xlabel('Penalty Value', fontsize=18)
plt.ylabel('Cost Ratio', fontsize=18)

# Add number of trajectories as text
for rho in robust_values:
    num_traj = combined_df[combined_df['Penalty Value'] == rho].shape[0]
    plt.text(robust_values.index(rho), -0.5, f'Trajs: {num_traj}', ha='center', fontsize=14)

plt.savefig(base_path + "/sum_cost_ratio_boxplot_no_outliers.png", bbox_inches='tight')
plt.show()
