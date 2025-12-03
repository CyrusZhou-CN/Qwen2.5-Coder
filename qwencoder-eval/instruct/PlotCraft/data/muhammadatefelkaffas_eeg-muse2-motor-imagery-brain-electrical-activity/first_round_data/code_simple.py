import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('museMonitor_2024-06-05--17-33-40_3002428320981162812.csv')

# Extract Alpha wave data
alpha_tp9 = df['Alpha_TP9'].dropna()
alpha_af7 = df['Alpha_AF7'].dropna()
alpha_af8 = df['Alpha_AF8'].dropna()
alpha_tp10 = df['Alpha_TP10'].dropna()

# Create figure with bad layout
fig, axs = plt.subplots(2, 2, figsize=(12, 4))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Use pie charts instead of histograms
axs[0, 0].pie(alpha_tp9[:5], labels=['A', 'B', 'C', 'D', 'E'], colors=['lime', 'red', 'yellow', 'cyan', 'magenta'])
axs[0, 0].set_title('Banana Slices')

axs[0, 1].pie(alpha_af7[:5], labels=['F', 'G', 'H', 'I', 'J'], colors=['#ff00ff', '#00ffff', '#ffff00', '#00ff00', '#ff0000'])
axs[0, 1].set_title('Quantum Frogs')

axs[1, 0].pie(alpha_af8[:5], labels=['K', 'L', 'M', 'N', 'O'], colors=['#123456', '#654321', '#abcdef', '#fedcba', '#0f0f0f'])
axs[1, 0].set_title('Alpha Centauri')

axs[1, 1].pie(alpha_tp10[:5], labels=['P', 'Q', 'R', 'S', 'T'], colors=['#ffcc00', '#cc00ff', '#00ccff', '#ff0066', '#66ff00'])
axs[1, 1].set_title('TP10 Explosion')

# Add overlapping legend
axs[1, 1].legend(['Glarbnok', 'Zorp', 'Blip', 'Snarf', 'Wibble'], loc='center')

# Add misleading axis labels
for ax in axs.flat:
    ax.set_xlabel('Voltage (Hz)')
    ax.set_ylabel('Time (Watts)')

# Add unrelated title
fig.suptitle('Distribution of Pineapple Energy in Mars Colonies', fontsize=10)

# Save the figure
plt.savefig('chart.png')