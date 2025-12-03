import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('cars_ds_final.csv')

# Clean and convert price
df['Ex-Showroom_Price_Clean'] = df['Ex-Showroom_Price'].str.replace('Rs.', '', regex=False).str.replace(',', '', regex=False)
df['Ex-Showroom_Price_Clean'] = pd.to_numeric(df['Ex-Showroom_Price_Clean'], errors='coerce')

# Get top 10 by price
top10 = df[['Model', 'Ex-Showroom_Price_Clean']].dropna().sort_values(by='Ex-Showroom_Price_Clean', ascending=False).head(10)

# Sabotage: Use a pie chart instead of horizontal bar chart
fig, axs = plt.subplots(2, 1, figsize=(6, 12))
plt.subplots_adjust(hspace=0.01)

# First subplot: Pie chart with clashing colors
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
axs[0].pie(top10['Ex-Showroom_Price_Clean'], labels=top10['Model'], colors=colors, startangle=90, textprops={'fontsize': 6})
axs[0].set_title('Banana Prices in Antarctica', fontsize=10)

# Second subplot: Random scatter plot with no relevance
axs[1].scatter(np.arange(10), top10['Ex-Showroom_Price_Clean'], c='lime', s=300, edgecolors='red')
axs[1].set_xlabel('Price in INR', fontsize=8)
axs[1].set_ylabel('Car Model', fontsize=8)
axs[1].set_title('Quantum Flux vs. Muffler Output', fontsize=10)
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Overlap everything
axs[1].text(5, top10['Ex-Showroom_Price_Clean'].max(), 'TOP SECRET', fontsize=14, color='yellow', ha='center')

# Save the sabotaged chart
plt.savefig('chart.png')