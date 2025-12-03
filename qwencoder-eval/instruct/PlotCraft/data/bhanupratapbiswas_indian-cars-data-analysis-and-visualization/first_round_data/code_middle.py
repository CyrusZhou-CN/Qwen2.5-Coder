import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('cars_ds_final.csv')

# Data preprocessing - remove any missing values in key columns
df_clean = df.dropna(subset=['Make', 'Model', 'Body_Type'])

# Create figure with 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
fig.patch.set_facecolor('white')

# Top subplot: Horizontal stacked bar chart of models by manufacturer
# Get top 8 manufacturers by number of models for better readability
make_model_counts = df_clean.groupby(['Make', 'Model']).size().reset_index(name='count')
top_makes = make_model_counts.groupby('Make')['count'].sum().nlargest(8).index

# Filter data for top manufacturers
filtered_data = make_model_counts[make_model_counts['Make'].isin(top_makes)]

# For each manufacturer, keep top 8 models and group others
def group_models(group):
    if len(group) <= 8:
        return group
    else:
        top_models = group.nlargest(7, 'count')
        others_count = group.iloc[7:]['count'].sum()
        others_row = pd.DataFrame({
            'Make': [group.iloc[0]['Make']], 
            'Model': ['Others'], 
            'count': [others_count]
        })
        return pd.concat([top_models, others_row], ignore_index=True)

# Apply grouping to each manufacturer
grouped_data = filtered_data.groupby('Make').apply(group_models).reset_index(drop=True)

# Create pivot table for stacked bar chart
pivot_data = grouped_data.pivot_table(index='Make', columns='Model', values='count', fill_value=0)

# Create distinct color palette
n_colors = len(pivot_data.columns)
colors = plt.cm.Set3(np.linspace(0, 1, n_colors))

# Create horizontal stacked bar chart
bars = pivot_data.plot(kind='barh', stacked=True, ax=ax1, color=colors, 
                      width=0.7, figsize=(16, 8))

ax1.set_title('Market Composition: Car Models Distribution Across Top Manufacturers', 
              fontsize=18, fontweight='bold', pad=25)
ax1.set_xlabel('Number of Car Variants', fontsize=14, fontweight='bold')
ax1.set_ylabel('Manufacturer', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add legend with better positioning
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
          title='Car Models', title_fontsize=12, frameon=True, 
          fancybox=True, shadow=True)

# Customize tick labels
ax1.tick_params(axis='both', labelsize=11)
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

# Bottom subplot: Pie chart of body types with leader lines
body_type_counts = df_clean['Body_Type'].value_counts()

# Create color palette for pie chart
pie_colors = plt.cm.Set2(np.linspace(0, 1, len(body_type_counts)))

# Create pie chart with leader lines for better label positioning
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'
    return my_autopct

wedges, texts, autotexts = ax2.pie(body_type_counts.values, 
                                   labels=None,  # Remove labels from pie
                                   autopct=make_autopct(body_type_counts.values),
                                   colors=pie_colors,
                                   startangle=90,
                                   pctdistance=0.85,
                                   textprops={'fontsize': 10})

# Add labels with leader lines using a legend instead
ax2.legend(wedges, [f'{label}\n({count} cars, {count/len(df_clean)*100:.1f}%)' 
                   for label, count in body_type_counts.items()],
          title="Body Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=11,
          title_fontsize=13,
          frameon=True,
          fancybox=True,
          shadow=True)

# Enhance pie chart text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax2.set_title('Market Composition: Distribution of Cars by Body Type', 
              fontsize=18, fontweight='bold', pad=30)

# Ensure pie chart is circular
ax2.axis('equal')

# Add comprehensive summary statistics
total_cars = len(df_clean)
total_makes = df_clean['Make'].nunique()
total_models = df_clean['Model'].nunique()
most_common_body = body_type_counts.index[0]
most_common_make = df_clean['Make'].value_counts().index[0]

summary_text = (f'Dataset Summary: {total_cars} total cars | {total_makes} manufacturers | '
               f'{total_models} unique models\n'
               f'Most common body type: {most_common_body} | '
               f'Leading manufacturer: {most_common_make}')

fig.text(0.02, 0.02, summary_text, fontsize=12, style='italic', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

# Adjust layout to prevent overlap and accommodate legends
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.93, hspace=0.4, right=0.75)

plt.show()