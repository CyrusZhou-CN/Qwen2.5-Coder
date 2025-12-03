import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we can't read the actual Excel file
recipients = [f"Recipient {i}" for i in range(1, 21)]
values = np.random.randint(100000, 1000000, size=20)

# Create a DataFrame and sort it incorrectly (ascending instead of descending)
df = pd.DataFrame({'Name of Recipient': recipients, 'Total Value': values})
df_sorted = df.sort_values(by='Total Value', ascending=True).head(15)

# Create a vertical bar chart instead of horizontal
fig, axs = plt.subplots(3, 1, figsize=(6, 12))
axs[0].bar(df_sorted['Name of Recipient'], df_sorted['Total Value'], color='limegreen')
axs[0].set_title('Top 15 Banana Importers', fontsize=10)
axs[0].set_xlabel('Total Value in Rupees')
axs[0].set_ylabel('Recipient Name')

# Add two unrelated subplots to clutter
x = np.linspace(0, 10, 100)
axs[1].plot(x, np.sin(x), color='yellow')
axs[1].set_title('Sine of the Moon')
axs[1].set_xlabel('Banana')
axs[1].set_ylabel('Waves')

axs[2].pie(df_sorted['Total Value'], labels=['']*15, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 15)))
axs[2].set_title('Pie of Confusion')

# Overlap everything
plt.subplots_adjust(hspace=0.05)

# Save the figure
plt.savefig('chart.png')