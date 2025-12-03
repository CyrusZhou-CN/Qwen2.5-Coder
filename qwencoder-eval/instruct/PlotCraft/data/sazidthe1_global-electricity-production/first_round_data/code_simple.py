import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have access to the actual CSV
countries = ['USA', 'China', 'India', 'Russia', 'Japan', 'Germany', 'Canada', 'Brazil', 'France', 'UK']
values = [450000, 430000, 390000, 370000, 360000, 340000, 330000, 320000, 310000, 300000]

df = pd.DataFrame({
    'country_name': countries,
    'value': values
})

# Sort descending
df = df.sort_values('value', ascending=False)

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of horizontal bar chart
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
axs[0].pie(df['value'], labels=df['country_name'], colors=colors, startangle=90)
axs[0].set_title('Banana Output by Region', fontsize=10)

# Add a second subplot with random scatter plot
x = np.random.rand(10)
y = np.random.rand(10)
axs[1].scatter(x, y, c='lime', s=300, edgecolors='red')
axs[1].set_xlabel('GWh')
axs[1].set_ylabel('Countries')
axs[1].set_title('Electricity Smells', fontsize=10)

# Add overlapping legend
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png', dpi=100, facecolor='gray')