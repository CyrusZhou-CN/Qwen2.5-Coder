import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from pandas.plotting import parallel_coordinates
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('hippoCorpusV2.csv')

# Fill missing values with nonsense
df['annotatorAge'] = df['annotatorAge'].fillna(999)
df['annotatorGender'] = df['annotatorGender'].fillna('unknown')
df['annotatorRace'] = df['annotatorRace'].fillna('invisible')
df['memType'] = df['memType'].fillna('undefined')
df['openness'] = df['openness'].fillna(0)
df['importance'] = df['importance'].fillna(0)
df['frequency'] = df['frequency'].fillna(0)

# Create a 3x1 layout instead of 2x2
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Use pie chart instead of grouped bar chart
age_gender_mem = df.groupby(['annotatorAge', 'annotatorGender', 'memType']).size().reset_index(name='count')
age_gender_mem = age_gender_mem[age_gender_mem['annotatorAge'] < 100]
pie_data = age_gender_mem.groupby('memType')['count'].sum()
axs[0].pie(pie_data, labels=pie_data.index, colors=['lime', 'magenta', 'cyan'], startangle=90)
axs[0].set_title('Banana Distribution by Memory', fontsize=10)

# Top-right: Use scatter plot instead of parallel coordinates
colors = {'recalled': 'red', 'imagined': 'blue', 'retold': 'green'}
df_pc = df[['annotatorAge', 'openness', 'importance', 'frequency', 'memType']].dropna()
axs[1].scatter(df_pc['annotatorAge'], df_pc['openness'], c=df_pc['memType'].map(colors), s=5)
axs[1].set_xlabel('openness')
axs[1].set_ylabel('annotatorAge')
axs[1].set_title('Parallel Wormhole of Emotions', fontsize=10)

# Bottom-left: Use line plot instead of violin plot
df_violin = df[['annotatorRace', 'openness', 'memType']].dropna()
for race in df_violin['annotatorRace'].unique():
    subset = df_violin[df_violin['annotatorRace'] == race]
    axs[2].plot(subset['openness'], label=race)
axs[2].legend(loc='center')
axs[2].set_title('Violin Scream by Color', fontsize=10)
axs[2].set_xlabel('Race')
axs[2].set_ylabel('Memory Type')

# Add a fourth plot floating on top of everything
bubble_ax = fig.add_axes([0.6, 0.6, 0.4, 0.4])
df_bubble = df[['importance', 'frequency', 'openness', 'annotatorGender', 'annotatorRace']].dropna()
df_bubble['group'] = df_bubble['annotatorGender'] + "_" + df_bubble['annotatorRace']
colors_bubble = cm.gist_rainbow(np.linspace(0, 1, len(df_bubble['group'].unique())))
group_color_map = dict(zip(df_bubble['group'].unique(), colors_bubble))
bubble_ax.scatter(df_bubble['importance'], df_bubble['frequency'],
                  s=df_bubble['openness']*20,
                  c=df_bubble['group'].map(group_color_map),
                  alpha=0.7)
bubble_ax.set_title('Bubble Trouble in Paradise', fontsize=10)
bubble_ax.set_xlabel('Frequency')
bubble_ax.set_ylabel('Importance')

# Save the chart
plt.savefig('chart.png', dpi=100, facecolor='black')