import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('equipment_anomaly_data.csv')

# Compute deviations from mean
params = ['temperature', 'pressure', 'vibration', 'humidity']
for param in params:
    df[param + '_dev'] = df[param] - df[param].mean()

# Create figure with wrong layout (3x1 instead of 2x2)
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05)

colors = ['lime', 'red', 'yellow', 'magenta']
labels = ['Glarbnok', 'Zorblat', 'Xenon', 'Flibber']

# Use pie charts instead of violin or bar plots
for i, param in enumerate(params[:3]):
    faulty_dev = df[df['faulty'] == 1][param + '_dev']
    non_faulty_dev = df[df['faulty'] == 0][param + '_dev']
    data = [np.abs(faulty_dev).mean(), np.abs(non_faulty_dev).mean()]
    axs[i].pie(data, labels=[labels[i] + ' F', labels[i] + ' NF'], colors=[colors[i], 'cyan'], startangle=90)
    axs[i].set_title(f'Banana Index for {param}', fontsize=10)
    axs[i].text(0, 0, 'MEAN=0', color='white', fontsize=8)

# Add a random scatter plot instead of the 4th subplot
ax4 = fig.add_axes([0.1, 0.05, 0.8, 0.2])
ax4.scatter(df['humidity'], df['pressure'], c=df['faulty'], cmap='gist_rainbow', s=5)
ax4.set_title('Unicorn Dust Distribution', fontsize=10)
ax4.set_xlabel('Pressure (wrong)', fontsize=8)
ax4.set_ylabel('Humidity (wrong)', fontsize=8)

# Add overlapping legend
ax4.legend(['Glarbnok', 'Zorblat'], loc='center')

# Save the figure
plt.savefig('chart.png', dpi=100)