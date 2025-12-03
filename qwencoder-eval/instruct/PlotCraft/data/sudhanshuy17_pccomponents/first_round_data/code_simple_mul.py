import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
intel_df = pd.read_csv('intel_cpus.csv')
amd_df = pd.read_csv('amd_cpus.csv')

# Remove currency symbols and commas, convert to float
intel_df['MRP'] = intel_df['MRP'].str.replace(',', '').str.strip('₹').astype(float)
amd_df['MRP'] = amd_df['MRP'].str.replace(',', '').str.strip('₹').astype(float)

# Combine the datasets
combined_df = pd.concat([intel_df, amd_df], ignore_index=True)

# Get the top 10 most expensive CPUs
top_10_cpus = combined_df.nlargest(10, 'MRP')

# Prepare data for plotting
labels = top_10_cpus['CPU']
prices = top_10_cpus['MRP']
colors = ['blue' if row['is_intel'] else 'red' for index, row in top_10_cpus.iterrows()]

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(labels, prices, color=colors)

# Add labels and title
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', ha='left')

plt.xlabel('Price (₹)')
plt.ylabel('CPU')
plt.title('Top 10 Most Expensive CPUs by AMD and Intel')
plt.legend(['Intel', 'AMD'], loc='upper right')

# Show the plot
plt.show()