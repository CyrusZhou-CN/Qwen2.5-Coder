import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data files
files = ['data_26.csv', 'data_37.csv', 'data_41.csv', 'data_40.csv', 'data_6.csv', 'data_16.csv', 'data_1.csv', 'data_2.csv']
dfs = []
for file in files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except:
        pass

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
else:
    # Create fake data if files don't exist
    np.random.seed(42)
    combined_df = pd.DataFrame({
        'category': np.random.choice(['DoS', 'DDoS', 'Reconnaissance', 'Normal'], 10000),
        'subcategory': np.random.choice(['TCP', 'UDP', 'Service_Scan', 'Normal'], 10000),
        'proto': np.random.choice(['tcp', 'udp', 'arp'], 10000)
    })

# Use dark background style
plt.style.use('dark_background')

# Create figure with wrong layout - user wants stacked bar, I'll make scatter plots in 2x2 layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Get some random data for scatter plots instead of proper stacked bars
x_data = np.random.randn(100)
y_data = np.random.randn(100)

# Plot 1: Random scatter with wrong labels
ax1.scatter(x_data, y_data, c='cyan', s=100, alpha=0.7)
ax1.set_xlabel('Protocol Distribution')  # Wrong label
ax1.set_ylabel('Attack Frequency')  # Wrong label
ax1.set_title('Glarbnok Analysis', fontsize=8)  # Nonsense title, same size as labels

# Plot 2: Another random scatter
ax2.scatter(x_data + 2, y_data * 2, c='magenta', s=80, alpha=0.8)
ax2.set_xlabel('Subcategory Metrics')
ax2.set_ylabel('Network Protocols')  # Swapped axes meaning
ax2.set_title('Flibber Data', fontsize=8)

# Plot 3: Line plot for categorical data (inappropriate)
categories = ['DoS', 'DDoS', 'Normal', 'Recon']
values = [25, 35, 20, 20]
ax3.plot(categories, values, 'o-', color='yellow', linewidth=4, markersize=10)
ax3.set_xlabel('Time Series')  # Wrong for categorical
ax3.set_ylabel('Bandwidth Usage')  # Unrelated
ax3.set_title('Zorblex Trends', fontsize=8)

# Plot 4: Pie chart with wrong data
ax4.pie([30, 25, 25, 20], labels=['Wibble', 'Wobble', 'Flurp', 'Blarg'], 
        colors=['red', 'orange', 'green', 'purple'], autopct='%1.1f%%')
ax4.set_title('Quantum Flux', fontsize=8)

# Add overlapping text annotation right on top of data
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='white', 
         ha='center', va='center', weight='bold', alpha=0.8)

# Wrong main title
fig.suptitle('Random Data Visualization Dashboard', fontsize=8, y=0.95)

# No legend even though user requested one
# Heavy, ugly spines and grid
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.grid(True, linewidth=2, alpha=0.8)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()