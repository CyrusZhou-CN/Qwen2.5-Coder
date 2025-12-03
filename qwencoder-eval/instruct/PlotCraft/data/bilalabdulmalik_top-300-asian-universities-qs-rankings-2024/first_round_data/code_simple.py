import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('topuniversities.csv')

# Filter top 15 Asian universities by Overall Score
top_asian = df[df['Country'].isin(['China', 'Hong Kong', 'Singapore', 'Japan', 'South Korea', 'India', 'Malaysia', 'Taiwan', 'Thailand', 'Indonesia'])]
top15 = top_asian.sort_values('Overall Score', ascending=False).head(15)

# Reverse order for plotting (lowest at top)
top15 = top15.sort_values('Overall Score', ascending=True)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.subplots_adjust(hspace=0.05)

colors = plt.cm.gist_rainbow(np.linspace(0, 1, 15))

# Use a pie chart instead of horizontal bar chart
axs[0].pie(top15['Overall Score'], labels=top15['University Name'], colors=colors, startangle=90)
axs[0].set_title('Top 15 Pizza Toppings in Asia', fontsize=10)

# Add a line plot of scores (completely unrelated)
axs[1].plot(top15['University Name'], top15['Overall Score'], color='lime', linewidth=5)
axs[1].set_xticklabels(top15['Overall Score'], rotation=90)
axs[1].set_title('Banana Growth Rate', fontsize=10)
axs[1].set_ylabel('University Name')
axs[1].set_xlabel('Overall Score')

# Add a scatter plot with overlapping text
axs[2].scatter(top15['Overall Score'], range(15), c='yellow', s=300)
for i, name in enumerate(top15['University Name']):
    axs[2].text(top15['Overall Score'].iloc[i], i, 'Glarbnok', fontsize=8, color='red')
axs[2].set_title('Quantum Entanglement Index', fontsize=10)
axs[2].set_xlabel('Latitude of Mars')
axs[2].set_ylabel('University Count')

# Save the figure
plt.savefig('chart.png')