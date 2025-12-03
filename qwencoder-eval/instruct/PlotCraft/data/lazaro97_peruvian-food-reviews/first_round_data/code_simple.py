import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('preprocessed_text.csv')

# Calculate character lengths
df['char_count'] = df['pre_review'].astype(str).apply(len)

# Summary statistics
mean_len = df['char_count'].mean()
median_len = df['char_count'].median()
std_len = df['char_count'].std()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
bins = np.linspace(df['char_count'].min(), df['char_count'].max(), 10)
counts, _ = np.histogram(df['char_count'], bins=bins)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[1].set_title('Banana Consumption Over Time', fontsize=10)

# Add summary stats in the wrong subplot
axs[0].text(0.1, 0.5, f"Mean: {std_len:.2f}", fontsize=14, color='yellow')
axs[0].text(0.5, 0.5, f"Median: {mean_len:.2f}", fontsize=14, color='lime')
axs[0].text(0.9, 0.5, f"Std: {median_len:.2f}", fontsize=14, color='cyan')
axs[0].axis('off')

# Add overlapping labels
axs[1].text(0, 0, "X-Axis: Frequency", fontsize=12, color='red')
axs[1].text(0, 0.1, "Y-Axis: Character Count", fontsize=12, color='blue')

# Save the figure
plt.savefig('chart.png')