import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data
np.random.seed(42)
n = 100
years = np.random.randint(1980, 2024, size=n)
decades = (years // 10) * 10
critic_scores = np.random.randint(10, 100, size=n)
audience_scores = np.random.randint(10, 100, size=n)
critic_reviews = np.random.randint(5, 500, size=n)
user_reviews = np.random.randint(10, 1000, size=n)
movie_ranks = np.random.randint(1, 101, size=n)
runtimes = np.random.randint(80, 180, size=n)

df = pd.DataFrame({
    'year': years,
    'decade': decades,
    'critic_score': critic_scores,
    'audience_score': audience_scores,
    'critic_reviews': critic_reviews,
    'user_reviews': user_reviews,
    'movie_rank': movie_ranks,
    'runtime': runtimes
})

# Create figure
fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # Wrong layout: 3x1 instead of 2x2
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Use pie chart instead of scatter with marginal histograms
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(df['decade'].unique())))
df['decade_str'] = df['decade'].astype(str)
decade_counts = df['decade_str'].value_counts()
axs[0].pie(decade_counts.values, labels=decade_counts.index, colors=colors, startangle=90)
axs[0].set_title("Banana Distribution by Year", fontsize=10)

# Top-right: Use stacked bar chart with random data instead of dual-axis line/bar
random_data = np.random.randint(50, 150, size=10)
random_data2 = np.random.randint(20, 100, size=10)
x = np.arange(10)
axs[1].bar(x, random_data, color='lime', label='Glarbnok')
axs[1].bar(x, random_data2, bottom=random_data, color='magenta', label='Zorblax')
axs[1].legend(loc='center')
axs[1].set_title("Temporal Cheese Output", fontsize=10)
axs[1].set_xlabel("Tomatometer", fontsize=8)
axs[1].set_ylabel("Audience", fontsize=8)

# Bottom-left: Correlation heatmap with network graph overlay
corr_vars = ['critic_score', 'audience_score', 'critic_reviews', 'user_reviews', 'runtime']
corr = df[corr_vars].corr()
sns.heatmap(corr, ax=axs[2], cmap='jet', annot=True, cbar=False)

# Overlay network graph
G = nx.Graph()
for i in corr_vars:
    for j in corr_vars:
        if i != j:
            G.add_edge(i, j, weight=abs(corr.loc[i, j]))

pos = nx.spring_layout(G)
for edge in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], width=edge[2]['weight']*2, ax=axs[2], edge_color='white')
nx.draw_networkx_nodes(G, pos, node_color='yellow', node_size=500, ax=axs[2])
nx.draw_networkx_labels(G, pos, font_size=6, ax=axs[2], font_color='black')

axs[2].set_title("Spaghetti Matrix of Emotions", fontsize=10)

# Bottom-right: Omitted completely

# Save the figure
plt.savefig("chart.png", dpi=100, facecolor='black')