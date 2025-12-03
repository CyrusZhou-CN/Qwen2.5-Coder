import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('Data_Train.xlsx')

# Data preprocessing
def parse_duration(duration_str):
    """Convert duration string to minutes"""
    if pd.isna(duration_str):
        return np.nan
    
    duration_str = str(duration_str).strip()
    total_minutes = 0
    
    if 'h' in duration_str:
        hours = int(duration_str.split('h')[0].strip())
        total_minutes += hours * 60
        
    if 'm' in duration_str:
        minutes_part = duration_str.split('h')[-1] if 'h' in duration_str else duration_str
        minutes = int(minutes_part.replace('m', '').strip())
        total_minutes += minutes
        
    return total_minutes

def parse_time(time_str):
    """Convert time string to hour"""
    if pd.isna(time_str):
        return np.nan
    try:
        return int(str(time_str).split(':')[0])
    except:
        return np.nan

def parse_stops(stops_str):
    """Convert stops string to number"""
    if pd.isna(stops_str):
        return 0
    stops_str = str(stops_str).lower()
    if 'non-stop' in stops_str:
        return 0
    elif '1 stop' in stops_str:
        return 1
    elif '2 stops' in stops_str:
        return 2
    elif '3 stops' in stops_str:
        return 3
    elif '4 stops' in stops_str:
        return 4
    else:
        return 0

# Apply preprocessing
df['Duration_minutes'] = df['Duration'].apply(parse_duration)
df['Dep_Hour'] = df['Dep_Time'].apply(parse_time)
df['Arrival_Hour'] = df['Arrival_Time'].apply(parse_time)
df['Stops_num'] = df['Total_Stops'].apply(parse_stops)

# Parse dates
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
df['Journey_day'] = df['Date_of_Journey'].dt.day
df['Journey_month'] = df['Date_of_Journey'].dt.month

# Remove rows with missing critical data
df_clean = df.dropna(subset=['Duration_minutes', 'Price', 'Dep_Hour', 'Arrival_Hour'])

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Row 1: Price vs Route Analysis

# Subplot 1: Scatter plot with regression line and marginal histogram
ax1 = plt.subplot(3, 3, 1)
# Main scatter plot
scatter = ax1.scatter(df_clean['Duration_minutes'], df_clean['Price'], 
                     alpha=0.6, c='steelblue', s=20)
# Regression line
z = np.polyfit(df_clean['Duration_minutes'], df_clean['Price'], 1)
p = np.poly1d(z)
ax1.plot(df_clean['Duration_minutes'], p(df_clean['Duration_minutes']), 
         "r--", alpha=0.8, linewidth=2)
ax1.set_xlabel('Flight Duration (minutes)')
ax1.set_ylabel('Price (INR)')
ax1.set_title('Price vs Duration with Regression', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Subplot 2: Box plot with violin overlay
ax2 = plt.subplot(3, 3, 2)
airlines = df_clean['Airline'].value_counts().head(6).index
df_top_airlines = df_clean[df_clean['Airline'].isin(airlines)]

# Violin plot
parts = ax2.violinplot([df_top_airlines[df_top_airlines['Airline'] == airline]['Price'].values 
                       for airline in airlines], 
                      positions=range(len(airlines)), widths=0.8, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

# Box plot overlay
box_data = [df_top_airlines[df_top_airlines['Airline'] == airline]['Price'].values 
           for airline in airlines]
bp = ax2.boxplot(box_data, positions=range(len(airlines)), widths=0.3, 
                patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('orange')
    patch.set_alpha(0.8)

ax2.set_xticks(range(len(airlines)))
ax2.set_xticklabels([airline[:10] for airline in airlines], rotation=45)
ax2.set_ylabel('Price (INR)')
ax2.set_title('Price Distribution by Airline', fontweight='bold')

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 3, 3)
# Calculate source-destination statistics
route_stats = df_clean.groupby(['Source', 'Destination', 'Airline']).agg({
    'Price': 'mean',
    'Airline': 'count'
}).rename(columns={'Airline': 'count'})
route_stats = route_stats.reset_index()

source_counts = df_clean['Source'].value_counts()
dest_counts = df_clean['Destination'].value_counts()

route_stats['source_count'] = route_stats['Source'].map(source_counts)
route_stats['dest_count'] = route_stats['Destination'].map(dest_counts)

# Create bubble chart
airlines_for_bubble = route_stats['Airline'].value_counts().head(4).index
colors = ['red', 'blue', 'green', 'orange']
for i, airline in enumerate(airlines_for_bubble):
    data = route_stats[route_stats['Airline'] == airline]
    ax3.scatter(data['source_count'], data['dest_count'], 
               s=data['Price']/50, alpha=0.6, c=colors[i], label=airline)

ax3.set_xlabel('Source Airport Frequency')
ax3.set_ylabel('Destination Airport Frequency')
ax3.set_title('Route Popularity vs Average Price', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2: Temporal Price Patterns

# Subplot 4: Line chart with error bands
ax4 = plt.subplot(3, 3, 4)
daily_stats = df_clean.groupby('Journey_day').agg({
    'Price': ['mean', 'std', 'count']
}).round(2)
daily_stats.columns = ['mean_price', 'std_price', 'count']
daily_stats = daily_stats.reset_index()

# Line chart with error bands
ax4.plot(daily_stats['Journey_day'], daily_stats['mean_price'], 
         'b-', linewidth=2, label='Average Price')
ax4.fill_between(daily_stats['Journey_day'], 
                daily_stats['mean_price'] - daily_stats['std_price'],
                daily_stats['mean_price'] + daily_stats['std_price'],
                alpha=0.3, color='blue')

# Scatter overlay
sample_data = df_clean.sample(500)  # Sample for visibility
ax4.scatter(sample_data['Journey_day'], sample_data['Price'], 
           alpha=0.4, s=10, c='red')

ax4.set_xlabel('Day of Journey')
ax4.set_ylabel('Price (INR)')
ax4.set_title('Price Trends Over Journey Days', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Heatmap with contours
ax5 = plt.subplot(3, 3, 5)
# Create price matrix by departure and arrival hours
price_matrix = df_clean.pivot_table(values='Price', index='Dep_Hour', 
                                   columns='Arrival_Hour', aggfunc='mean')
price_matrix = price_matrix.fillna(price_matrix.mean().mean())

# Heatmap
im = ax5.imshow(price_matrix.values, cmap='YlOrRd', aspect='auto')
ax5.set_xticks(range(len(price_matrix.columns)))
ax5.set_yticks(range(len(price_matrix.index)))
ax5.set_xticklabels(price_matrix.columns)
ax5.set_yticklabels(price_matrix.index)
ax5.set_xlabel('Arrival Hour')
ax5.set_ylabel('Departure Hour')
ax5.set_title('Price Heatmap: Dep vs Arrival Hour', fontweight='bold')

# Add contour lines
X, Y = np.meshgrid(range(len(price_matrix.columns)), range(len(price_matrix.index)))
ax5.contour(X, Y, price_matrix.values, levels=5, colors='white', alpha=0.8, linewidths=1)

# Subplot 6: Stacked area chart
ax6 = plt.subplot(3, 3, 6)
# Monthly price trends by airline
monthly_airline = df_clean.groupby(['Journey_month', 'Airline'])['Price'].mean().unstack(fill_value=0)
top_airlines = df_clean['Airline'].value_counts().head(4).index
monthly_airline = monthly_airline[top_airlines]

# Stacked area chart
ax6.stackplot(monthly_airline.index, *[monthly_airline[col] for col in monthly_airline.columns],
             labels=monthly_airline.columns, alpha=0.7)

# Trend lines
for col in monthly_airline.columns:
    z = np.polyfit(monthly_airline.index, monthly_airline[col], 1)
    p = np.poly1d(z)
    ax6.plot(monthly_airline.index, p(monthly_airline.index), '--', linewidth=2)

ax6.set_xlabel('Month')
ax6.set_ylabel('Average Price (INR)')
ax6.set_title('Monthly Price Trends by Airline', fontweight='bold')
ax6.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Row 3: Multi-dimensional Correlations

# Subplot 7: Parallel coordinates plot
ax7 = plt.subplot(3, 3, 7)
# Prepare data for parallel coordinates
features = ['Duration_minutes', 'Stops_num', 'Dep_Hour', 'Price']
parallel_data = df_clean[features + ['Airline']].dropna()

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(parallel_data[features])
normalized_df = pd.DataFrame(normalized_data, columns=features)
normalized_df['Airline'] = parallel_data['Airline'].values

# Plot parallel coordinates for top airlines
top_airlines_parallel = parallel_data['Airline'].value_counts().head(4).index
colors = ['red', 'blue', 'green', 'orange']

for i, airline in enumerate(top_airlines_parallel):
    airline_data = normalized_df[normalized_df['Airline'] == airline].sample(min(50, len(normalized_df[normalized_df['Airline'] == airline])))
    for idx, row in airline_data.iterrows():
        ax7.plot(range(len(features)), row[features], color=colors[i], alpha=0.3, linewidth=0.5)

ax7.set_xticks(range(len(features)))
ax7.set_xticklabels(['Duration', 'Stops', 'Dep Hour', 'Price'])
ax7.set_ylabel('Normalized Values')
ax7.set_title('Parallel Coordinates Plot', fontweight='bold')

# Create custom legend
for i, airline in enumerate(top_airlines_parallel):
    ax7.plot([], [], color=colors[i], label=airline, linewidth=2)
ax7.legend()

# Subplot 8: Correlation matrix heatmap
ax8 = plt.subplot(3, 3, 8)
# Calculate correlation matrix
numeric_cols = ['Price', 'Duration_minutes', 'Stops_num', 'Dep_Hour', 'Arrival_Hour', 'Journey_day', 'Journey_month']
corr_matrix = df_clean[numeric_cols].corr()

# Create heatmap
im = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax8.set_xticks(range(len(corr_matrix.columns)))
ax8.set_yticks(range(len(corr_matrix.index)))
ax8.set_xticklabels([col.replace('_', '\n') for col in corr_matrix.columns], rotation=45)
ax8.set_yticklabels([col.replace('_', '\n') for col in corr_matrix.index])
ax8.set_title('Correlation Matrix Heatmap', fontweight='bold')

# Add correlation values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
        ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontsize=8)

# Subplot 9: 3D-style scatter plot
ax9 = plt.subplot(3, 3, 9)
# Create 3D effect using perspective
sample_data = df_clean.sample(1000)  # Sample for performance

# Create perspective effect
x = sample_data['Duration_minutes']
y = sample_data['Price']
z = sample_data['Stops_num']

# Apply perspective transformation
perspective_x = x + z * 20  # Offset based on stops
perspective_y = y - z * 500  # Offset based on stops

# Color by airline
airlines_3d = sample_data['Airline'].value_counts().head(4).index
colors_3d = ['red', 'blue', 'green', 'orange']

for i, airline in enumerate(airlines_3d):
    mask = sample_data['Airline'] == airline
    ax9.scatter(perspective_x[mask], perspective_y[mask], 
               s=50 + z[mask]*20, alpha=0.6, c=colors_3d[i], label=airline)

ax9.set_xlabel('Duration (minutes) + Stops Offset')
ax9.set_ylabel('Price (INR) - Stops Offset')
ax9.set_title('3D-Style: Price vs Duration vs Stops', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.show()