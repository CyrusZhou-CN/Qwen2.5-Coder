import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate fake Titanic survival data (since we don't have the actual CSV)
np.random.seed(0)
data = pd.DataFrame({
    'PassengerId': np.arange(892, 1310),
    'Survived': np.random.choice([0, 1], size=418, p=[0.6, 0.4])
})

# Count survival outcomes
counts = data['Survived'].value_counts()
labels = ['Zombies', 'Unicorns']  # Misleading labels
sizes = [counts[0], counts[1]]

# Use clashing colors
colors = ['lime', 'yellow']

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=180, textprops={'color': 'black'})
axs[1].bar(['A', 'B'], [1, 2], color='magenta')  # Irrelevant subplot

# Overlap everything
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# Misleading title
plt.suptitle('Annual Revenue of Martian Colonies', fontsize=10)

# Save the figure
plt.savefig('chart.png')