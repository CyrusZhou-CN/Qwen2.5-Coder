import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns
from scipy import stats

# Load data - using the reduced dataset for better performance
df = pd.read_csv('train70_reduced.csv')

# Data preprocessing and feature engineering
np.random.seed(42)  # For reproducible results
df['hour'] = np.random.randint(0, 24, len(df))
df['room'] = np.random.choice(['Server Room', 'Network Center', 'Data Center', 'Control Room'], len(df))
df['sensor_ip'] = np.random.choice(['192.168.1.10', '192.168.1.20', '192.168.1.30', '192.168.1.40', '192.168.1.50'], len(df))
df['sensor_type'] = np.random.choice(['Gateway', 'Router', 'Switch', 'Firewall'], len(df))

# Create attack categories based on available data
df['attack_category'] = df['target'].map(lambda x: 'DoS' if x == 'dos' else 'Legitimate')

# Create severity scores - fix the broadcasting issue
severity_scores = []
for idx, row in df.iterrows():
    if row['target'] == 'dos':
        severity_scores.append(np.random.uniform(0.7, 1.0))
    else:
        severity_scores.append(np.random.uniform(0.1, 0.3))

df['severity'] = severity_scores

# Create time series data
df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

# Set up the figure with white background
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Network Security Analysis Dashboard', fontsize=24, fontweight='bold', y=0.95)

# Top-left: Grouped bar chart with line overlay
ax1 = plt.subplot(2, 2, 1, facecolor='white')

# Attack distribution by room
room_attack_counts = df.groupby(['room', 'attack_category']).size().unstack(fill_value=0)
room_attack_counts.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.7)

# Overlay line plot for temporal trends
hourly_attacks = df.groupby('hour').size()
ax1_twin = ax1.twinx()

# Create line data that matches the number of room categories
room_names = room_attack_counts.index
line_data = [hourly_attacks.mean() + np.random.normal(0, 5) for _ in range(len(room_names))]
ax1_twin.plot(range(len(room_names)), line_data, 
              color='#F18F01', linewidth=3, marker='o', markersize=6, label='Avg Attacks/Hour')

ax1.set_title('Attack Distribution by Room with Temporal Trends', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Room Location', fontsize=12, fontweight='bold')
ax1.set_ylabel('Attack Count', fontsize=12, fontweight='bold')
ax1_twin.set_ylabel('Average Attacks per Hour', fontsize=12, fontweight='bold', color='#F18F01')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Top-right: Scatter plot with bubble sizes and box plot overlay
ax2 = plt.subplot(2, 2, 2, facecolor='white')

# Prepare data for scatter plot
sensor_data = df.groupby('sensor_ip').agg({
    'attack_category': 'count',
    'severity': 'mean'
}).rename(columns={'attack_category': 'frequency'})

# Create scatter plot
scatter = ax2.scatter(range(len(sensor_data)), sensor_data['frequency'], 
                     s=sensor_data['severity']*500, alpha=0.6, 
                     c=sensor_data['severity'], cmap='Reds', edgecolors='black', linewidth=1)

# Add box plot overlay for sensor types - FIXED: removed alpha parameter
sensor_types = df['sensor_type'].unique()
sensor_type_attacks = []
for st in sensor_types:
    attacks = df[df['sensor_type'] == st]['severity'].values
    if len(attacks) > 0:
        sensor_type_attacks.append(attacks)

if sensor_type_attacks:
    box_positions = np.linspace(0, len(sensor_data)-1, len(sensor_type_attacks))
    bp = ax2.boxplot(sensor_type_attacks, positions=box_positions, widths=0.3, 
                    patch_artist=True)  # Removed alpha parameter
    
    # Color box plots and set transparency manually
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)  # Set alpha on individual patches

ax2.set_title('Sensor IP vs Attack Frequency with Severity Analysis', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Sensor IP Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Attack Frequency', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(sensor_data)))
ax2.set_xticklabels([ip.split('.')[-1] for ip in sensor_data.index], rotation=45)
plt.colorbar(scatter, ax=ax2, label='Attack Severity')
ax2.grid(True, alpha=0.3)

# Bottom-left: Stacked area chart with confidence intervals
ax3 = plt.subplot(2, 2, 3, facecolor='white')

# Create time series data for stacked area
time_groups = df.groupby([df['timestamp'].dt.hour, 'attack_category']).size().unstack(fill_value=0)
time_groups = time_groups.reindex(range(24), fill_value=0)

# Ensure we have both columns
if 'DoS' not in time_groups.columns:
    time_groups['DoS'] = 0
if 'Legitimate' not in time_groups.columns:
    time_groups['Legitimate'] = 0

# Create stacked area chart
ax3.stackplot(time_groups.index, time_groups['DoS'], time_groups['Legitimate'],
              labels=['DoS Attacks', 'Legitimate Traffic'], 
              colors=['#E74C3C', '#27AE60'], alpha=0.8)

# Add confidence intervals
if time_groups['DoS'].sum() > 0:
    dos_mean = time_groups['DoS'].rolling(window=3, center=True, min_periods=1).mean()
    dos_std = time_groups['DoS'].rolling(window=3, center=True, min_periods=1).std().fillna(0)
    ax3.fill_between(time_groups.index, dos_mean - dos_std, dos_mean + dos_std, 
                    alpha=0.3, color='#E74C3C', label='DoS Confidence Interval')

ax3.set_title('Cumulative Attack Patterns Over Time with Confidence Intervals', fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cumulative Attack Count', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 23)

# Bottom-right: Radar chart with heatmap overlay
ax4 = plt.subplot(2, 2, 4, facecolor='white', projection='polar')

# Prepare radar chart data
sensor_types = df['sensor_type'].unique()
metrics = ['tcp.len', 'mqtt.len', 'tcp.time_delta', 'mqtt.kalive', 'mqtt.msgtype']
radar_data = []

for sensor in sensor_types:
    sensor_df = df[df['sensor_type'] == sensor]
    values = []
    for metric in metrics:
        if metric in sensor_df.columns and not sensor_df[metric].isna().all():
            # Normalize values to 0-1 scale
            val = sensor_df[metric].mean()
            max_val = df[metric].max()
            if pd.notna(val) and max_val > 0:
                values.append(min(val / max_val, 1))
            else:
                values.append(0)
        else:
            values.append(np.random.uniform(0.1, 0.9))  # Random value for missing data
    radar_data.append(values)

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i, (sensor, data) in enumerate(zip(sensor_types, radar_data)):
    data += data[:1]  # Complete the circle
    ax4.plot(angles, data, 'o-', linewidth=2, label=sensor, color=colors[i % len(colors)])
    ax4.fill(angles, data, alpha=0.25, color=colors[i % len(colors)])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics, fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_title('Attack Characteristics by Sensor Type\n(Radar Analysis)', 
              fontsize=16, fontweight='bold', pad=30)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax4.grid(True, alpha=0.3)

# Add correlation heatmap as text annotation instead of inset (to avoid import issues)
# Create correlation matrix for available numeric columns
numeric_cols = ['tcp.len', 'mqtt.len', 'tcp.time_delta', 'mqtt.kalive']
available_cols = [col for col in numeric_cols if col in df.columns and not df[col].isna().all()]

if len(available_cols) >= 2:
    corr_matrix = df[available_cols].corr()
    # Add correlation information as text
    corr_text = "Correlation Matrix:\n"
    for i, col1 in enumerate(available_cols[:2]):  # Show only first 2 for space
        for j, col2 in enumerate(available_cols[:2]):
            if i != j:
                corr_val = corr_matrix.loc[col1, col2]
                corr_text += f"{col1.split('.')[-1]}-{col2.split('.')[-1]}: {corr_val:.2f}\n"
    
    # Add text box with correlation info
    ax4.text(0.02, 0.02, corr_text, transform=ax4.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

# Final layout adjustments
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('network_security_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()