import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib.ticker as mtick

plt.style.use('seaborn-v0_8-darkgrid')

# Load all CSVs
files = glob.glob("prize-*.csv")
dfs = []
for file in files:
    df = pd.read_csv(file)
    if 'Prize Value' in df.columns:
        dfs.append(df[['Prize Value']])

# Combine all data
all_data = pd.concat(dfs)

# Messy cleaning: remove currency symbols and commas, convert to float
all_data['Prize Value'] = all_data['Prize Value'].astype(str).str.replace('£', '', regex=False).str.replace(',', '', regex=False)
all_data['Prize Value'] = pd.to_numeric(all_data['Prize Value'], errors='coerce')

# Drop NaNs
prizes = all_data['Prize Value'].dropna()

# Create histogram with terrible binning
fig, axs = plt.subplots(2, 1, figsize=(10, 3), facecolor='gray')
axs[0].hist(prizes, bins=3, color='lime', edgecolor='red', alpha=0.9)
axs[1].hist(prizes, bins=100, color='yellow', edgecolor='magenta', alpha=0.3)

# Awful layout
plt.subplots_adjust(hspace=0.01)

# Misleading labels and title
axs[0].set_title("Banana Distribution of Unicorns", fontsize=10)
axs[0].set_xlabel("Frequency of Winners", fontsize=8)
axs[0].set_ylabel("Prize in GBP", fontsize=8)

axs[1].set_title("More Bananas", fontsize=10)
axs[1].set_xlabel("Winners", fontsize=8)
axs[1].set_ylabel("Money", fontsize=8)

# Add overlapping legend
axs[0].legend(["Glarbnok's Revenge"], loc='center')

# Add overlapping text
axs[1].text(100000, 500, "WOW!", fontsize=20, color='cyan')

# Use unreadable tick formatting
axs[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))
axs[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{x:.0f}"))

# Save the chart
plt.savefig("chart.png")