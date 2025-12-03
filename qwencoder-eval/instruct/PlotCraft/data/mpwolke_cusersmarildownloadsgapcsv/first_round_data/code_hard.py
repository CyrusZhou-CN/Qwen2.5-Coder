import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('gap.csv')

# Parse the semicolon-delimited data
df_split = df['location;indicator;subject;measure;frequency;time;value'].str.split(';', expand=True)
df_split.columns = ['location', 'indicator', 'subject', 'measure', 'frequency', 'time', 'value']

# Clean and convert data types
df_clean = df_split.copy()
df_clean = df_clean.dropna()
df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')
df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
df_clean = df_clean.dropna()

# Filter for wage gap data
df_clean = df_clean[df_clean['indicator'] == 'WAGEGAP']

# Calculate average wage gaps by country
country_avg = df_clean.groupby('location')['value'].mean().sort_values(ascending=False)

# Get top 3 and bottom 3 countries
top_3_countries = country_avg.head(3).index.tolist()
bottom_3_countries = country_avg.tail(3).index.tolist()

# Create figure with white background
fig = plt.figure(figsize=(18, 15), facecolor='white')
fig.patch.set_facecolor('white')

# Define colors
colors_top = ['#d62728', '#ff7f0e', '#2ca02c']
colors_bottom = ['#1f77b4', '#9467bd', '#8c564b']
colors_employment = ['#e377c2', '#7f7f7f']

# Policy change years (example years for demonstration)
policy_years = {
    'AUS': [1984, 1996, 2009],
    'USA': [1963, 1978, 1993],
    'GBR': [1970, 1975, 2010],
    'NOR': [1978, 1993, 2002],
    'SWE': [1980, 1991, 2008],
    'DNK': [1976, 1989, 2006]
}

# Row 1: Top 3 countries with highest wage gaps
for i, country in enumerate(top_3_countries):
    ax = plt.subplot(3, 3, i+1)
    ax.set_facecolor('white')
    
    country_data = df_clean[df_clean['location'] == country]
    employee_data = country_data[country_data['subject'] == 'EMPLOYEE'].sort_values('time')
    
    if len(employee_data) > 0:
        years = employee_data['time'].values
        values = employee_data['value'].values
        
        # Line chart with filled area
        ax.plot(years, values, color=colors_top[i], linewidth=2.5, alpha=0.8)
        ax.fill_between(years, values, alpha=0.3, color=colors_top[i])
        
        # Add scatter points for policy years
        if country in policy_years:
            for policy_year in policy_years[country]:
                if policy_year in years:
                    idx = np.where(years == policy_year)[0]
                    if len(idx) > 0:
                        ax.scatter(policy_year, values[idx[0]], color='red', s=80, 
                                 marker='*', edgecolor='black', linewidth=1, zorder=5)
    
    ax.set_title(f'**{country} - Wage Gap Evolution**', fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Wage Gap (%)', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Row 2: Bottom 3 countries with lowest wage gaps
for i, country in enumerate(bottom_3_countries):
    ax1 = plt.subplot(3, 3, i+4)
    ax1.set_facecolor('white')
    
    country_data = df_clean[df_clean['location'] == country]
    employee_data = country_data[country_data['subject'] == 'EMPLOYEE'].sort_values('time')
    
    if len(employee_data) > 0:
        years = employee_data['time'].values
        values = employee_data['value'].values
        
        # Step plot
        ax1.step(years, values, where='mid', color=colors_bottom[i], linewidth=2, alpha=0.8)
        
        # Secondary y-axis for bar chart
        ax2 = ax1.twinx()
        ax2.bar(years[::3], values[::3], alpha=0.4, color=colors_bottom[i], width=1.5)
        
        # Trend line with confidence interval
        if len(years) > 3:
            z = np.polyfit(years, values, 1)
            p = np.poly1d(z)
            ax1.plot(years, p(years), '--', color='black', alpha=0.7, linewidth=1.5)
    
    ax1.set_title(f'**{country} - Year-over-Year Changes**', fontweight='bold', fontsize=12, pad=15)
    ax1.set_xlabel('Year', fontsize=10)
    ax1.set_ylabel('Wage Gap (%) - Step', fontsize=10, color=colors_bottom[i])
    ax2.set_ylabel('Absolute Values - Bars', fontsize=10, color=colors_bottom[i])
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

# Row 3: Comparative analysis across employment types

# Left subplot: Multi-line time series with heatmap background
ax = plt.subplot(3, 3, 7)
ax.set_facecolor('white')

# Create decade-wise heatmap background
decades = np.arange(1970, 2030, 10)
for i, decade in enumerate(decades[:-1]):
    intensity = 0.1 + (i % 3) * 0.1
    rect = Rectangle((decade, 0), 10, 50, facecolor='lightblue', alpha=intensity)
    ax.add_patch(rect)

# Plot employment type comparisons
for emp_type, color in zip(['EMPLOYEE', 'SELFEMPLOYED'], colors_employment):
    emp_data = df_clean[df_clean['subject'] == emp_type]
    if len(emp_data) > 0:
        avg_by_year = emp_data.groupby('time')['value'].mean()
        line_style = '-' if emp_type == 'EMPLOYEE' else '--'
        ax.plot(avg_by_year.index, avg_by_year.values, 
               color=color, linewidth=2.5, linestyle=line_style, 
               label=emp_type.title(), alpha=0.8)

ax.set_title('**Employment Type Comparison**', fontweight='bold', fontsize=12, pad=15)
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('Average Wage Gap (%)', fontsize=10)
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 50)

# Middle subplot: Slope chart
ax = plt.subplot(3, 3, 8)
ax.set_facecolor('white')

countries_for_slope = df_clean['location'].unique()[:8]  # Limit for clarity
y_positions = np.arange(len(countries_for_slope))

for i, country in enumerate(countries_for_slope):
    country_data = df_clean[df_clean['location'] == country]
    if len(country_data) > 0:
        start_val = country_data['value'].iloc[0] if len(country_data) > 0 else 0
        end_val = country_data['value'].iloc[-1] if len(country_data) > 0 else 0
        change_magnitude = abs(end_val - start_val)
        
        # Draw slope line
        ax.plot([0, 1], [start_val, end_val], 'o-', linewidth=2, alpha=0.7)
        
        # Add bubble at end point
        ax.scatter(1, end_val, s=change_magnitude*10, alpha=0.6, 
                  color=plt.cm.viridis(i/len(countries_for_slope)))
        
        # Add country labels
        ax.text(-0.05, start_val, country, ha='right', va='center', fontsize=9)

ax.set_title('**Start-End Point Comparison**', fontweight='bold', fontsize=12, pad=15)
ax.set_xlim(-0.2, 1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Start', 'End'])
ax.set_ylabel('Wage Gap (%)', fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right subplot: Violin plot with box plots and trajectories
ax = plt.subplot(3, 3, 9)
ax.set_facecolor('white')

# Create decade groups
df_clean['decade'] = (df_clean['time'] // 10) * 10
decades_available = sorted(df_clean['decade'].unique())

# Prepare data for violin plot
violin_data = []
for decade in decades_available:
    decade_data = df_clean[df_clean['decade'] == decade]['value'].values
    if len(decade_data) > 0:
        violin_data.append(decade_data)

if violin_data:
    # Violin plot
    parts = ax.violinplot(violin_data, positions=range(len(decades_available)), 
                         widths=0.6, showmeans=True, showmedians=True)
    
    # Style violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
    
    # Box plots overlay
    box_data = [df_clean[df_clean['decade'] == decade]['value'].values 
                for decade in decades_available]
    bp = ax.boxplot(box_data, positions=range(len(decades_available)), 
                   widths=0.3, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)

# Add country trajectories
selected_countries = df_clean['location'].unique()[:5]  # Limit for clarity
for country in selected_countries:
    country_data = df_clean[df_clean['location'] == country]
    if len(country_data) > 1:
        decade_means = country_data.groupby('decade')['value'].mean()
        decade_positions = [list(decades_available).index(d) for d in decade_means.index 
                          if d in decades_available]
        if len(decade_positions) > 1:
            ax.plot(decade_positions, decade_means.values, 
                   alpha=0.4, linewidth=1, color='red')

ax.set_title('**Distribution by Decade**', fontweight='bold', fontsize=12, pad=15)
ax.set_xlabel('Decade', fontsize=10)
ax.set_ylabel('Wage Gap (%)', fontsize=10)
ax.set_xticks(range(len(decades_available)))
ax.set_xticklabels([f"{int(d)}s" for d in decades_available], rotation=45)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)
plt.show()