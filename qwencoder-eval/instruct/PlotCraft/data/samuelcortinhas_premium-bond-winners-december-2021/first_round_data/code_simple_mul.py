import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to clean and convert prize values to numeric
def clean_prize_value(value):
    if pd.isna(value):
        return np.nan
    
    # Convert to string and handle different formats
    value_str = str(value).strip()
    
    # Remove £ symbol and commas
    value_str = value_str.replace('£', '').replace(',', '')
    
    # Handle decimal format (like 1000000.00)
    try:
        return float(value_str)
    except:
        return np.nan

# List of expected CSV files based on the provided data
csv_files = [
    'prize-march-2022.csv',
    'prize-june-2022.csv', 
    'prize-may-2022.csv',
    'prize-september-2022.csv',
    'prize-april-2022.csv',
    'prize-february-2022.csv',
    'prize-january-2022.csv',
    'prize-august-2022.csv',
    'prize-december-2021.csv',
    'prize-december-2022.csv',
    'prize-november-2022.csv',
    'prize-october-2022.csv',
    'prize-july-2022.csv'
]

# Load and process data efficiently
all_prize_values = []

for file in csv_files:
    if os.path.exists(file):
        try:
            # Read only the Prize Value column to save memory and time
            df = pd.read_csv(file, usecols=['Prize Value'])
            
            # Clean and convert prize values immediately
            cleaned_values = df['Prize Value'].apply(clean_prize_value)
            
            # Remove NaN values and add to list
            valid_values = cleaned_values.dropna().tolist()
            all_prize_values.extend(valid_values)
            
            print(f"Processed {file}: {len(valid_values)} valid prizes")
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    else:
        print(f"File not found: {file}")

# Convert to numpy array for faster processing
prize_array = np.array(all_prize_values)

# Define the standard prize tiers
prize_tiers = [1000, 5000, 10000, 25000, 50000, 100000, 1000000]

# Filter data to only include standard prize tiers
filtered_prizes = prize_array[np.isin(prize_array, prize_tiers)]

print(f"Total valid prizes: {len(prize_array)}")
print(f"Standard tier prizes: {len(filtered_prizes)}")

# Create the histogram
plt.figure(figsize=(12, 8))

# Count occurrences of each prize tier
prize_counts = []
for tier in prize_tiers:
    count = np.sum(filtered_prizes == tier)
    prize_counts.append(count)

# Create bar chart instead of histogram for better control
x_positions = np.arange(len(prize_tiers))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#E74C3C']

bars = plt.bar(x_positions, prize_counts, color=colors, alpha=0.8, 
               edgecolor='white', linewidth=1.2)

# Customize the plot
plt.title('Distribution of Premium Bond Prize Values Across 2022', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Prize Value', fontsize=12, fontweight='bold')
plt.ylabel('Frequency (Number of Prizes)', fontsize=12, fontweight='bold')

# Set x-axis labels
prize_labels = ['£1,000', '£5,000', '£10,000', '£25,000', '£50,000', '£100,000', '£1,000,000']
plt.xticks(x_positions, prize_labels, rotation=45, ha='right')

# Format y-axis to show values clearly
max_count = max(prize_counts)
if max_count >= 1000:
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))

# Add value labels on top of bars
for i, count in enumerate(prize_counts):
    if count > 0:
        plt.text(i, count + max_count * 0.01, 
                f'{count:,}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Set background to white
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Adjust layout to prevent overlap
plt.tight_layout()

# Add a note about the data
total_prizes = len(filtered_prizes)
plt.figtext(0.02, 0.02, f'Total standard tier prizes analyzed: {total_prizes:,}', 
           fontsize=9, style='italic', alpha=0.7)

# Print summary statistics
print("\nPrize Distribution Summary:")
for i, (tier, count) in enumerate(zip(prize_tiers, prize_counts)):
    percentage = (count / total_prizes * 100) if total_prizes > 0 else 0
    print(f"{prize_labels[i]}: {count:,} prizes ({percentage:.1f}%)")

plt.savefig('prize_distribution.png', dpi=300, bbox_inches='tight')
plt.show()