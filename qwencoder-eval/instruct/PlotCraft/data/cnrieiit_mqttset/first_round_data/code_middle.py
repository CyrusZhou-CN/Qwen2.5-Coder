import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Create fake sensor network
G = nx.Graph()
sensors = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
positions = {
    'S1': (0, 0), 'S2': (1, 0), 'S3': (2, 0),
    'S4': (0, 1), 'S5': (1, 1), 'S6': (2, 1),
    'Broker': (1, 0.5)
}
comm_freq = {'S1': 10, 'S2': 50, 'S3': 5, 'S4': 30, 'S5': 15, 'S6': 60}
sensor_type = {'S1': 'temp', 'S2': 'humid', 'S3': 'temp', 'S4': 'humid', 'S5': 'temp', 'S6': 'humid'}

for s in sensors:
    G.add_node(s)
    G.add_edge(s, 'Broker')
G.add_node('Broker')

node_sizes = [comm_freq.get(n, 20)*10 for n in G.nodes()]
node_colors = ['lime' if sensor_type.get(n)=='temp' else 'magenta' if sensor_type.get(n)=='humid' else 'cyan' for n in G.nodes()]

# Fake attack data
attack_types = ['SlowITe', 'Bruteforce', 'Malformed', 'Flooding', 'DoS']
legit = [100, 100, 100, 100, 100]
attack = [300, 250, 400, 500, 450]

x = np.arange(len(attack_types))
width = 0.3

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Wrong layout: should be 2x1

# Top subplot: network diagram
nx.draw(G, pos=positions, ax=axs[0], with_labels=True, node_size=node_sizes, node_color=node_colors, font_color='yellow', edge_color='white')
axs[0].set_title("Banana Distribution Map", fontsize=10)
axs[0].set_facecolor('black')

# Bottom subplot: bar chart
axs[1].bar(x - width/2, legit, width, label='Legit', color='gray')
axs[1].bar(x + width/2, attack, width, label='Attack', color='yellow')
axs[1].set_xticks(x)
axs[1].set_xticklabels(['Slo', 'Brute', 'Mal', 'Flood', 'DoS'])
axs[1].set_ylabel('Banana Count')
axs[1].set_xlabel('Severity Level')
axs[1].legend(loc='center')
axs[1].set_title("Sensor Banana Attacks", fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save the sabotaged chart
plt.savefig('chart.png')