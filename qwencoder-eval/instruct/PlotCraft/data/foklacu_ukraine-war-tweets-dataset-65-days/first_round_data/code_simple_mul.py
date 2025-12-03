import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Create figure with white background
plt.figure(figsize=(14, 8))
plt.style.use('default')  # Ensure clean default styling

# Define colors and line styles for each search term
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# Define CSV files and their corresponding search terms
csv_files = [
    ('Ukraine_war.csv', 'Ukraine War'),
    ('Ukraine_border.csv', 'Ukraine Border'),
    ('Russian_border_Ukraine.csv', 'Russian Border Ukraine'),
    ('Ukraine_troops.csv', 'Ukraine Troops'),
    ('Russia_invade.csv', 'Russia Invade'),
    ('Russian_troops.csv', 'Russian Troops'),
    ('StandWithUkraine.csv', 'Stand With Ukraine'),
    ('Ukraine_nato.csv', 'Ukraine NATO')
]

# Process each CSV file
for i, (csv_file, search_term) in enumerate(csv_files):
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found, skipping...")
            continue
            
        # Load data with optimized reading - only read necessary columns
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file, usecols=['date'], dtype={'date': 'str'})
        
        # Convert date column to datetime more efficiently
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Filter for March 5, 2022 more efficiently
        target_date = pd.to_datetime('2022-03-05').date()
        march_5_mask = df['date'].dt.date == target_date
        march_5 = df[march_5_mask].copy()
        
        if len(march_5) > 0:
            # Extract hour from datetime
            march_5['hour'] = march_5['date'].dt.hour
            
            # Count tweets per hour
            hourly_counts = march_5['hour'].value_counts().sort_index()
            
            # Create complete hour range (0-23) and fill missing hours with 0
            all_hours = pd.Series(0, index=range(24))
            all_hours.update(hourly_counts)
            
            # Plot line for this search term
            plt.plot(all_hours.index, all_hours.values, 
                    color=colors[i % len(colors)], 
                    linestyle=line_styles[i % len(line_styles)], 
                    linewidth=2.5, marker='o', markersize=4, 
                    label=search_term, alpha=0.8)
            
            print(f"Successfully processed {csv_file}: {len(march_5)} tweets on March 5, 2022")
        else:
            print(f"No data found for March 5, 2022 in {csv_file}")
    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        continue

# Customize the plot with professional styling
plt.title('Tweet Engagement Patterns: Ukraine War Topics Throughout March 5, 2022', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Hour of Day (UTC)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tweets', fontsize=12, fontweight='bold')

# Set x-axis to show all 24 hours
plt.xticks(range(0, 24, 2), fontsize=10)
plt.xlim(0, 23)

# Format y-axis with comma separators for large numbers
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add subtle grid
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Create legend with proper positioning
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# Set background to white
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add subtitle with additional context
plt.figtext(0.5, 0.02, 'Temporal analysis reveals how different Ukraine war-related topics gained or lost momentum throughout the critical day', 
           ha='center', fontsize=10, style='italic', color='gray')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('ukraine_tweet_engagement_march5_2022.png', dpi=300, bbox_inches='tight')