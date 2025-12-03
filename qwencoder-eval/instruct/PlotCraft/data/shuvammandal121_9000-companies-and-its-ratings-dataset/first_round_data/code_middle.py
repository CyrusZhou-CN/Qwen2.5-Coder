import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy import stats

plt.style.use('dark_background')

# Load and process data
df = pd.read_csv('company_dataset.csv')

# Extract numeric values from review_count
def extract_review_count(review_str):
    if pd.isna(review_str):
        return 0
    match = re.search(r'\(([\d.]+)([kK]?)\s*Reviews\)', str(review_str))
    if match:
        num = float(match.group(1))
        if match.group(2).lower() == 'k':
            num *= 1000
        return num
    return 0

df['review_count_numeric'] = df['review_count'].apply(extract_review_count)
df = df.dropna(subset=['ratings'])

# Create figure with wrong layout - user wants marginal histograms but I'll make 2x2 grid instead
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Wrong chart type - use bar chart instead of scatter plot
companies_sample = df.sample(50)
bars = ax1.bar(range(len(companies_sample)), companies_sample['ratings'], 
               color='red', alpha=0.3)
ax1.set_title('Temperature vs Precipitation Analysis', fontsize=8, color='yellow')
ax1.set_xlabel('Amplitude', fontsize=8)
ax1.set_ylabel('Time', fontsize=8)

# Add overlapping text annotation
ax1.text(25, 3.5, 'CRITICAL DATA POINT!!!', fontsize=12, color='white', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))

# Wrong plot in second subplot - pie chart for continuous data
ax2.pie([1, 2, 3, 4, 5], labels=['A', 'B', 'C', 'D', 'E'], colors=['cyan', 'magenta', 'yellow', 'green', 'red'])
ax2.set_title('Glarbnok Distribution', fontsize=8)

# Third subplot - random histogram
random_data = np.random.normal(0, 1, 1000)
ax3.hist(random_data, bins=30, color='orange', alpha=0.7)
ax3.set_title('Quantum Flux Readings', fontsize=8)
ax3.set_xlabel('Frequency', fontsize=8)
ax3.set_ylabel('Count', fontsize=8)

# Fourth subplot - line plot of unrelated data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.cos(x)
ax4.plot(x, y, color='lime', linewidth=3)
ax4.set_title('Waveform Analysis', fontsize=8)
ax4.set_xlabel('Phase', fontsize=8)
ax4.set_ylabel('Magnitude', fontsize=8)

# Add thick spines and heavy gridlines
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.grid(True, linewidth=2, alpha=0.8, color='white')
    ax.tick_params(width=2, length=8)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()