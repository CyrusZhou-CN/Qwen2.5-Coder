import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import glob
import os

plt.style.use('seaborn-v0_8-darkgrid')

# Load all CSVs
files = glob.glob("*.csv")
dfs = []
for f in files:
    df = pd.read_csv(f, usecols=['date', 'replyCount', 'retweetCount', 'likeCount', 'Searh'])
    df['source'] = os.path.splitext(f)[0]
    dfs.append(df)

df = pd.concat(dfs)

# Parse date and aggregate
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.floor('H')
df['engagement'] = df['replyCount'] + df['retweetCount'] + df['likeCount']

# Pick 8 random search terms
terms = df['Searh'].dropna().unique()[:8]

# Prepare data
agg = df[df['Searh'].isin(terms)].groupby(['hour', 'Searh'])['engagement'].sum().reset_index()

# Pivot for plotting
pivot = agg.pivot(index='hour', columns='Searh', values='engagement').fillna(0)

# Create figure
fig, ax = plt.subplots(figsize=(14, 4))

# Use rainbow colormap
colors = cm.gist_rainbow(np.linspace(0, 1, len(pivot.columns)))

# Plot as scatter instead of line
for i, col in enumerate(pivot.columns):
    ax.scatter(pivot.index, pivot[col], label=f"Glarbnok {i}", color=colors[i], s=20)

# Bad labels
ax.set_xlabel("Engagement Level", fontsize=10)
ax.set_ylabel("Time of Day", fontsize=10)

# Misleading title
ax.set_title("Banana Prices in Atlantis", fontsize=10)

# Legend on top of data
ax.legend(loc='center', fontsize=6)

# Overlap everything
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.05, wspace=0.05)

# Save
plt.savefig("chart.png", dpi=100)