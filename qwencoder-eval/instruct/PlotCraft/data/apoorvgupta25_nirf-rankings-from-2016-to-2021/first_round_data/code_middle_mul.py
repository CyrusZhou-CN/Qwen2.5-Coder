import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we can't load actual CSVs
years = np.array([2016, 2017, 2018, 2019, 2020, 2021])
n_years = len(years)
n_institutes = 10

# Fake TLR data
tlr_data = np.random.rand(n_institutes, n_years) * 100
avg_tlr = tlr_data.mean(axis=0)
std_tlr = tlr_data.std(axis=0)

# Fake GO data
go_data = np.random.rand(n_institutes, n_years) * 100
avg_go = go_data.mean(axis=0)
go_above_90 = (go_data > 90).sum(axis=0)

# Fake RPC data
rpc_data = np.random.rand(n_institutes, n_years) * 100
rpc_bins = [0, 25, 50, 75, 100]
rpc_binned = np.digitize(rpc_data, rpc_bins) - 1
rpc_dist = np.array([(rpc_binned == i).sum(axis=0) for i in range(4)])
rpc_median = np.median(rpc_data, axis=0)

# Fake overall scores
overall_scores = np.random.rand(n_institutes, n_years) * 100
avg_overall = overall_scores.mean(axis=0)
top10 = np.sort(overall_scores, axis=0)[-10:]
bottom10 = np.sort(overall_scores, axis=0)[:10:]

fig, axs = plt.subplots(3, 1, figsize=(12, 10))  # Wrong layout: 3x1 instead of 2x2
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Engineering TLR
ax = axs[0]
ax.bar(years, avg_tlr, color='lime', label='Average TLR')
ax.errorbar(years, avg_tlr, yerr=std_tlr, fmt='o', color='red', label='Error')
for i in range(5):
    ax.plot(years, tlr_data[i], linestyle='--', marker='x', label=f'Glarbnok {i}')
ax.set_title('Banana Output Over Time', fontsize=10)
ax.set_xlabel('TLR Score')
ax.set_ylabel('Year')
ax.legend(loc='center', fontsize=6)

# Top-right: Management GO
ax = axs[1]
ax.plot(years, go_above_90, color='yellow', label='Above 90 Count')
ax2 = ax.twinx()
ax2.bar(years, avg_go, color='purple', alpha=0.5, label='Average GO')
ax.set_title('Quantum Flux of Management', fontsize=10)
ax.set_xlabel('Graduation Outcomes')
ax.set_ylabel('Year')
ax2.set_ylabel('Banana Count')
ax.legend(loc='upper left', fontsize=6)
ax2.legend(loc='upper right', fontsize=6)

# Bottom-left: University RPC
ax = axs[2]
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
for i in range(4):
    ax.fill_between(years, rpc_dist[i], color=colors[i], label=f'Bin {i}')
ax.plot(years, rpc_median, color='white', linestyle=':', linewidth=3, label='Median')
ax.set_title('RPC Distribution of Unicorns', fontsize=10)
ax.set_xlabel('RPC Score')
ax.set_ylabel('Year')
ax.legend(loc='center', fontsize=6)

# Bottom-right: Overall scores (not plotted at all to sabotage)
# Intentionally omitted

fig.patch.set_facecolor('gray')
for ax in axs:
    ax.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=6, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.savefig('chart.png')