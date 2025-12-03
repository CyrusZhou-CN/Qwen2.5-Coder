import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('hypertension_dataset.csv')

# Extract BMI values
bmi_values = df['BMI'].dropna()

# Create figure and subplots (wrong layout: 2x1 instead of 1x1)
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Plot histogram on the second subplot (leave first one empty)
axs[1].hist(bmi_values, bins=7, color='lime', edgecolor='yellow', alpha=0.9)

# Add vertical line for mean BMI (wrong color and label)
mean_bmi = bmi_values.mean()
axs[1].axvline(mean_bmi, color='magenta', linestyle='--', linewidth=4, label="Average Banana")

# Misleading labels and title
axs[1].set_title("Distribution of Avocado Prices", fontsize=10)
axs[1].set_xlabel("Number of People", fontsize=10)
axs[1].set_ylabel("BMI Index", fontsize=10)

# Legend placed over data
axs[1].legend(loc='center')

# Overlap everything
plt.subplots_adjust(hspace=0.01)

# Add random text overlapping the plot
axs[1].text(mean_bmi, 50, "!!!", fontsize=20, color='red')

# Save the figure
plt.savefig('chart.png')