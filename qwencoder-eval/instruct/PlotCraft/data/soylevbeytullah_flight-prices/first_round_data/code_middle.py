import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re

# Load data
df = pd.read_excel('Data_Train.xlsx')

# Function to convert duration string to minutes
def duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return np.nan
    
    # Extract hours and minutes using regex
    hours = re.findall(r'(\d+)h', str(duration_str))
    minutes = re.findall(r'(\d+)m', str(duration_str))
    
    total_minutes = 0
    if hours:
        total_minutes += int(hours[0]) * 60
    if minutes:
        total_minutes += int(minutes[0])
    
    return total_minutes if total_minutes > 0 else np.nan

# Convert duration to minutes
df['Duration_Minutes'] = df['Duration'].apply(duration_to_minutes)

# Remove rows with missing duration or price data
df_clean = df.dropna(subset=['Duration_Minutes', 'Price'])

# Get unique airlines and assign colors
airlines = df_clean['Airline'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(airlines)))
airline_colors = dict(zip(airlines, colors))

# Create figure with subplots for marginal histograms
fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('white')

# Define grid layout
gs = fig.add_gridspec(3, 3, width_ratios=[1, 4, 1], height_ratios=[1, 4, 0.2],
                      hspace=0.05, wspace=0.05)

# Main scatter plot
ax_main = fig.add_subplot(gs[1, 1])
ax_main.set_facecolor('white')

# Plot scatter points for each airline
for airline in airlines:
    airline_data = df_clean[df_clean['Airline'] == airline]
    ax_main.scatter(airline_data['Duration_Minutes'], airline_data['Price'], 
                   c=[airline_colors[airline]], label=airline, alpha=0.6, s=30)

# Add regression line
x = df_clean['Duration_Minutes']
y = df_clean['Price']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax_main.plot(line_x, line_y, 'red', linewidth=2, alpha=0.8, 
            label=f'Trend Line (R² = {r_value**2:.3f})')

ax_main.set_xlabel('Flight Duration (minutes)', fontweight='bold', fontsize=12)
ax_main.set_ylabel('Price (₹)', fontweight='bold', fontsize=12)
ax_main.set_title('Flight Duration vs Price Correlation by Airline', 
                 fontweight='bold', fontsize=14, pad=20)
ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# Top marginal histogram (Duration distribution)
ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
ax_top.set_facecolor('white')
ax_top.hist(df_clean['Duration_Minutes'], bins=30, alpha=0.7, color='skyblue', 
           edgecolor='black', linewidth=0.5)
ax_top.set_ylabel('Frequency', fontweight='bold', fontsize=10)
ax_top.set_title('Duration Distribution', fontweight='bold', fontsize=11)
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Right marginal histogram (Price distribution)
ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
ax_right.set_facecolor('white')
ax_right.hist(df_clean['Price'], bins=30, orientation='horizontal', 
             alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
ax_right.set_xlabel('Frequency', fontweight='bold', fontsize=10)
ax_right.set_title('Price Distribution', fontweight='bold', fontsize=11, rotation=90, 
                  x=1.1, y=0.5)
ax_right.tick_params(labelleft=False)
ax_right.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Remove empty subplot areas
ax_empty1 = fig.add_subplot(gs[0, 0])
ax_empty1.axis('off')
ax_empty2 = fig.add_subplot(gs[0, 2])
ax_empty2.axis('off')
ax_empty3 = fig.add_subplot(gs[1, 0])
ax_empty3.axis('off')

# Add correlation statistics text box
correlation_coef = np.corrcoef(df_clean['Duration_Minutes'], df_clean['Price'])[0, 1]
stats_text = f'Correlation: {correlation_coef:.3f}\nSample Size: {len(df_clean):,} flights'
ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
            alpha=0.8, edgecolor='gray'), fontsize=10)

plt.tight_layout()
plt.show()