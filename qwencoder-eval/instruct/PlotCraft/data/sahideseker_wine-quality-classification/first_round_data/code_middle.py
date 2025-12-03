import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
df = pd.read_csv('wine_quality_classification.csv')

# Set ugly style
plt.style.use('dark_background')

# Create 3x1 layout instead of 2x2 as requested
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot 1: Bar chart instead of scatter plot matrix
quality_counts = df['quality_label'].value_counts()
axes[0].bar(quality_counts.index, quality_counts.values, color=['red', 'green', 'blue'])
axes[0].set_title('Density Distribution Analysis', fontsize=10)
axes[0].set_xlabel('Alcohol Content', fontsize=10)
axes[0].set_ylabel('Sugar Levels', fontsize=10)

# Plot 2: Line plot instead of correlation heatmap
x_vals = np.linspace(0, 10, len(df))
axes[1].plot(x_vals, df['fixed_acidity'].values, 'o-', color='yellow', linewidth=5)
axes[1].plot(x_vals, df['residual_sugar'].values, 's-', color='cyan', linewidth=5)
axes[1].set_title('Chemical Composition Trends', fontsize=10)
axes[1].set_xlabel('Time Series', fontsize=10)
axes[1].set_ylabel('Correlation Matrix', fontsize=10)
axes[1].grid(True, color='white', linewidth=2)

# Plot 3: Pie chart instead of scatter plot
pie_data = [30, 25, 45]
pie_labels = ['Category X', 'Category Y', 'Category Z']
axes[2].pie(pie_data, labels=pie_labels, colors=['orange', 'purple', 'brown'], autopct='%1.1f%%')
axes[2].set_title('Bubble Analysis Results', fontsize=10)

# Add overlapping text annotations
fig.text(0.5, 0.8, 'WINE QUALITY METRICS', fontsize=20, color='white', ha='center')
fig.text(0.3, 0.6, 'Data Processing Complete', fontsize=15, color='red', ha='center')
fig.text(0.7, 0.4, 'Statistical Overview', fontsize=15, color='green', ha='center')
fig.text(0.5, 0.2, 'Analysis Framework', fontsize=15, color='blue', ha='center')

# Save the chart
plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()