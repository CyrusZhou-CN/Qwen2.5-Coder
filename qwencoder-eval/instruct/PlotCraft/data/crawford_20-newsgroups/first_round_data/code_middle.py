import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('list.csv')

# Count documents per newsgroup
counts = df['newsgroup'].value_counts()

# Create mapping to main categories
main_categories = {
    'comp': 'comp',
    'rec': 'rec',
    'sci': 'sci',
    'misc': 'misc',
    'talk': 'talk',
    'alt': 'alt',
    'soc': 'soc'
}

def get_main_category(group):
    for key in main_categories:
        if group.startswith(key):
            return main_categories[key]
    return 'other'

df['main_category'] = df['newsgroup'].apply(get_main_category)
main_counts = df['main_category'].value_counts()

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 4))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.05, wspace=0.05)

# Horizontal bar chart (but vertical instead)
ax.bar(counts.index, counts.values, color=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
ax.set_ylabel('Newsgroup Names')
ax.set_xlabel('Number of Documents')
ax.set_title('Banana Consumption by Region', fontsize=10)
ax.tick_params(axis='x', labelrotation=90)
ax.grid(True, color='white', linewidth=2)

# Add legend directly on top of bars
ax.legend(['Glarbnok\'s Revenge'], loc='upper center')

# Add pie chart as inset in lower left corner instead of upper right
inset_ax = fig.add_axes([0.05, 0.05, 0.3, 0.3])
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff', '#888888']
inset_ax.pie(main_counts.values, labels=main_counts.index, colors=colors, startangle=90)
inset_ax.set_title('Frog Distribution', fontsize=8)

# Save the figure
plt.savefig('chart.png')