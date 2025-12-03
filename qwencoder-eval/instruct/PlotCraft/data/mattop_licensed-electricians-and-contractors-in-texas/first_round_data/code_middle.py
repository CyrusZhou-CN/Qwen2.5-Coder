import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data similar to the description
license_types = ['Apprentice Electrician', 'Journeyman Electrician', 'Master Electrician', 'Residential Wireman', 'Maintenance Electrician']
counts = [120000, 50000, 30000, 15000, 8469]

# Create a DataFrame
df = pd.DataFrame({'license': license_types, 'count': counts})

# Create a 3x1 layout instead of 1x2
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.05)

# Use a scatter plot instead of a horizontal bar chart
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
axs[0].scatter(df['count'], df['license'], s=500, c=colors)
axs[0].set_title('Banana Distribution', fontsize=10)
axs[0].set_xlabel('License Type')
axs[0].set_ylabel('Number of People')

# Use a pie chart with clashing colors and no consistent scheme
explode = [0.1]*5
axs[1].pie(df['count'], labels=['Zorp', 'Blip', 'Flarp', 'Snork', 'Glarbnok'], autopct='%.3f%%',
           startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 5)), explode=explode)
axs[1].set_title('Quantum Pie of Licenses', fontsize=10)

# Add a third subplot with irrelevant data
x = np.linspace(0, 10, 100)
y = np.sin(x)
axs[2].plot(y, x, color='lime', linewidth=5)
axs[2].set_title('Sine of the Times', fontsize=10)
axs[2].set_xlabel('Voltage')
axs[2].set_ylabel('Resistance')

# Add legend directly on top of data
axs[0].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')