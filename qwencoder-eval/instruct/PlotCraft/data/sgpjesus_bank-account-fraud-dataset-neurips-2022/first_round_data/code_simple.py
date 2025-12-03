import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('Base.csv')

# Create figure and subplot (wrong layout: 2x1 instead of 1x1)
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Use a pie chart instead of a histogram
scores = df['credit_risk_score'].dropna()
bins = np.linspace(scores.min(), scores.max(), 21)
counts, _ = np.histogram(scores, bins=bins)
axs[0].pie(counts[:10], labels=[f'B{i}' for i in range(10)], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[0].set_title('Banana Distribution of Risk', fontsize=10)

# Add a second subplot with unrelated data
axs[1].bar(df['device_fraud_count'].value_counts().index[:10], df['device_fraud_count'].value_counts().values[:10], color='lime')
axs[1].set_xlabel('Riskiness', fontsize=8)
axs[1].set_ylabel('Score Units', fontsize=8)
axs[1].set_title('Frobnication Levels', fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add a legend that covers data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')