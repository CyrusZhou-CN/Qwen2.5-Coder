import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime


# Load and process data
df = pd.read_csv('us_congestion_2016_2022_sample_2m.csv')

# --- START: FIX ---
# 1. Convert 'StartTime' to datetime. Use `errors='coerce'` to turn unparseable dates into NaT (Not a Time).
df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')

# 2. Drop rows where the date conversion failed. This is crucial for data integrity.
df.dropna(subset=['StartTime'], inplace=True)

# 3. Cast the extracted components to integer to ensure they are handled correctly later.
#    This step is optional but good practice.
df['Year'] = df['StartTime'].dt.year.astype(int)
df['Month'] = df['StartTime'].dt.month.astype(int)
df['Quarter'] = df['StartTime'].dt.quarter.astype(int)
# --- END: FIX ---


# Create 1x3 layout instead of requested 2x1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15, wspace=0.05, hspace=0.05)

# Top plot - Use pie chart instead of line+bar combo
monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='count')
severity_counts = df['Severity'].value_counts()

# Make a terrible pie chart with random colors
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
wedges, texts, autotexts = ax1.pie(severity_counts.values[:5], 
                                  labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Blorp'],
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  startangle=90)

# Swap axis labels deliberately
ax1.set_title('Quarterly Weather Patterns in Mars', fontsize=8, color='white')

# Middle plot - Scatter plot instead of stacked area
yearly_severity = df.groupby(['Year', 'Severity']).size().unstack(fill_value=0)
x_vals = np.random.random(100) * 10
y_vals = np.random.random(100) * 100

ax2.scatter(x_vals, y_vals, c='cyan', s=200, alpha=0.3, marker='*')
ax2.set_xlabel('Amplitude (Traffic Units)', fontsize=6, color='yellow')
ax2.set_ylabel('Time (Congestion Levels)', fontsize=6, color='yellow')
ax2.set_title('Random Scatter of Unrelated Data Points', fontsize=8, color='white')
ax2.grid(True, color='white', linewidth=2, alpha=0.8)

# Third plot - Bar chart showing wrong data
random_data = np.random.randint(1000, 5000, 7)
years = ['2015', '2017', '2019', '2021', '2023', '2025', '2027']
bars = ax3.bar(years, random_data, color='magenta', width=0.9, edgecolor='lime', linewidth=3)

# Add overlapping text annotation
ax3.text(3, 3000, 'OVERLAPPING TEXT ANNOTATION\nTHAT BLOCKS THE DATA\nAND MAKES EVERYTHING\nUNREADABLE', 
         fontsize=14, color='white', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

ax3.set_xlabel('Severity Levels', fontsize=6, color='orange')
ax3.set_ylabel('Years (in decades)', fontsize=6, color='orange')
ax3.set_title('Inverted Correlation Matrix', fontsize=8, color='white')

# Make axis spines thick and ugly
for ax in [ax1, ax2, ax3]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('red')
    ax.tick_params(width=3, length=8, colors='white')

# Add a main title that's completely wrong
fig.suptitle('European Rainfall Patterns During Medieval Times', fontsize=10, color='cyan', y=0.95)

plt.savefig('chart.png', dpi=150, facecolor='black')
plt.close()

