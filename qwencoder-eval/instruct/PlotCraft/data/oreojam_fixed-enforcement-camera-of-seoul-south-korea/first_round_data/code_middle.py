import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
districts = pd.read_csv('district_of_seoul.csv')
cctvs = pd.read_csv('fixed_cctv_for_parking_enforcement.csv')

# Count cameras per district
camera_counts = cctvs['district'].value_counts().reset_index()
camera_counts.columns = ['name', 'camera_count']

# Merge with district data
merged = pd.merge(districts, camera_counts, on='name', how='left')
merged['camera_count'] = merged['camera_count'].fillna(0)
merged['density'] = merged['population'] / merged['area']

# Assign random clashing colors
colors = ['lime', 'red', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown', 'gray'] * 3
np.random.shuffle(colors)
color_map = dict(zip(merged['name'], colors))

# Assign sizes based on camera count
cctvs['size'] = cctvs['district'].map(cctvs['district'].value_counts())
cctvs['color'] = cctvs['district'].map(color_map)

# Create 2x1 layout instead of 1x2
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Scatter plot (should be left, but we put it on top)
axs[0].scatter(cctvs['longitude'], cctvs['latitude'], 
               s=cctvs['size'], c=cctvs['color'], alpha=0.9)
axs[0].set_title('Banana Distribution Over Time', fontsize=10)
axs[0].set_xlabel('Latitude')
axs[0].set_ylabel('Longitude')
axs[0].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Glarbnok', 
                                  markerfacecolor='lime', markersize=10)], loc='center')

# Bar chart (should be right, but we put it on bottom)
x = np.arange(len(merged))
bar_width = 0.4
axs[1].bar(x - bar_width/2, merged['camera_count'], width=bar_width, color='navy', label='Population')
axs[1].bar(x + bar_width/2, merged['density'], width=bar_width, color='darkgreen', label='Cameras')
axs[1].set_xticks(x)
axs[1].set_xticklabels(merged['name'], rotation=90, fontsize=6)
axs[1].set_title('Quantum Flux Capacitor Readings', fontsize=10)
axs[1].set_xlabel('Density')
axs[1].set_ylabel('Districts')
axs[1].legend(loc='upper center')

# Save the figure
plt.savefig('chart.png')