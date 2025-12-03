import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of all pond CSV files
pond_files = [
    'IoTPond10.csv', 'IoTPond6.csv', 'IoTPond3.csv', 'IoTPond8.csv', 
    'IoTPond9.csv', 'IoTPond7.csv', 'IoTPond11.csv', 'IoTpond1.csv',
    'IoTPond4.csv', 'IoTPond2.csv', 'IoTPond12.csv'
]

# Initialize list to store all pH values
all_ph_values = []

# Load and combine pH data from all pond files
for file in pond_files:
    try:
        df = pd.read_csv(file)
        
        # Standardize pH column names - check for different variations
        ph_column = None
        for col in df.columns:
            if col.lower() in ['ph', 'ph_value']:
                ph_column = col
                break
        
        if ph_column is not None:
            # Remove any invalid pH values (negative, extremely high, or NaN)
            ph_data = df[ph_column].dropna()
            ph_data = ph_data[(ph_data >= 0) & (ph_data <= 14)]
            all_ph_values.extend(ph_data.tolist())
            print(f"Loaded {len(ph_data)} pH values from {file}")
        else:
            print(f"No pH column found in {file}")
            
    except FileNotFoundError:
        print(f"File {file} not found, skipping...")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert to numpy array for analysis
ph_array = np.array(all_ph_values)

# Calculate summary statistics
mean_ph = np.mean(ph_array)
median_ph = np.median(ph_array)
std_ph = np.std(ph_array)

print(f"\nTotal pH measurements: {len(ph_array)}")
print(f"Mean pH: {mean_ph:.2f}")
print(f"Median pH: {median_ph:.2f}")
print(f"Standard Deviation: {std_ph:.2f}")

# Create the histogram with white background and professional styling
plt.figure(figsize=(12, 8))
plt.style.use('default')  # Ensure clean default style

# Create histogram with water-quality themed color
n, bins, patches = plt.hist(ph_array, bins=20, color='#2E86AB', alpha=0.7, 
                           edgecolor='white', linewidth=0.8)

# Styling and labels
plt.title('**Distribution of pH Values Across All Aquaponics Fish Ponds**', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('pH Value', fontsize=12, fontweight='medium')
plt.ylabel('Frequency', fontsize=12, fontweight='medium')

# Add subtle grid
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add summary statistics as text annotation
stats_text = f'Summary Statistics:\nMean: {mean_ph:.2f}\nMedian: {median_ph:.2f}\nStd Dev: {std_ph:.2f}\nTotal Measurements: {len(ph_array):,}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10, fontweight='medium')

# Add optimal pH range reference lines
optimal_min = 6.5
optimal_max = 8.5
plt.axvline(optimal_min, color='#A8DADC', linestyle='--', linewidth=2, alpha=0.8, label='Optimal pH Range')
plt.axvline(optimal_max, color='#A8DADC', linestyle='--', linewidth=2, alpha=0.8)

# Add legend
plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# Set background to white
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()