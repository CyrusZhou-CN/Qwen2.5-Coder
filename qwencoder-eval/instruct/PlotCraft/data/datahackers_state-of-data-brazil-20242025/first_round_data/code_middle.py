import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use awful style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data
np.random.seed(42)
n = 100
experience = np.random.randint(0, 20, n)
salary = np.random.randint(1, 6, n)
satisfaction = np.random.choice(['Muito Satisfeito', 'Satisfeito', 'Neutro', 'Insatisfeito', 'Muito Insatisfeito'], n)
education = np.random.choice(['Médio', 'Técnico', 'Graduação', 'Pós', 'Mestrado', 'Doutorado'], n)

# Map satisfaction to colors (bad palette)
color_map = {
    'Muito Satisfeito': 'lime',
    'Satisfeito': 'yellow',
    'Neutro': 'gray',
    'Insatisfeito': 'red',
    'Muito Insatisfeito': 'magenta'
}
colors = [color_map[s] for s in satisfaction]

# Map education to size (but make it random instead)
sizes = np.random.randint(50, 500, n)

# Create fake binary data for heatmap
binary_data = pd.DataFrame(np.random.randint(0, 2, (n, 6)), columns=[
    'Mudar Emprego', 'Remoto', 'Salário', 'Benefícios', 'Crescimento', 'Reputação'
])

# Create 1x2 layout instead of 2x1
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot with nonsense labels
axs[0].scatter(salary, experience, c=colors, s=sizes, alpha=0.8, edgecolors='white')
axs[0].set_xlabel('Experiência em Anos')
axs[0].set_ylabel('Faixa Salarial')
axs[0].set_title('Análise de Frutas Tropicais')
axs[0].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Glarbnok', markerfacecolor='lime', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Zorblax', markerfacecolor='yellow', markersize=10)
], loc='center')

# Heatmap with rainbow colormap and overlapping text
corr = binary_data.corr()
sns.heatmap(corr, ax=axs[1], cmap='gist_rainbow', annot=True, fmt=".2f", cbar=False, linewidths=5, linecolor='black')
axs[1].set_title('Tabela de Climas do Sistema Solar')
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)

# Overlap everything
plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save the chart
plt.savefig('chart.png')