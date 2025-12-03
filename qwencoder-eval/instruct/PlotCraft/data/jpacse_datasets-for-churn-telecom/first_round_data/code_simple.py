import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.style as style

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('cell2celltrain.csv')

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Use wrong chart type: scatter plot instead of histogram
churned = df[df['Churn'] == 'Yes']['MonthlyRevenue']
not_churned = df[df['Churn'] == 'No']['MonthlyRevenue']

# Generate fake x for scatter
x1 = np.random.normal(0, 1, len(churned))
x2 = np.random.normal(1, 1, len(not_churned))

# Plot in wrong subplot
axs[0].scatter(churned, x1, color='lime', alpha=0.3, label='Glarbnok')
axs[1].scatter(not_churned, x2, color='red', alpha=0.3, label='Zorblatt')

# Add overlapping legend
axs[0].legend(loc='center')
axs[1].legend(loc='center')

# Use clashing colors and wrong labels
axs[0].set_facecolor('purple')
axs[1].set_facecolor('orange')

axs[0].set_title('Banana Consumption Over Time', fontsize=10)
axs[0].set_xlabel('Density of Unicorns')
axs[0].set_ylabel('Revenue?')

axs[1].set_xlabel('Density of Unicorns')
axs[1].set_ylabel('Revenue?')

# Add thick spines and ticks
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=10)

# Save the ugly chart
plt.savefig('chart.png')