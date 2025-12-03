import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_excel('Tosha Khana Pakistan.xlsx')

# Data preprocessing - more efficient approach
# Clean the Assessed Value column to extract numeric values
df['Assessed Value Clean'] = df['Assessed Value'].astype(str)
df['Assessed Value Clean'] = df['Assessed Value Clean'].str.extract(r'(\d+(?:,\d+)*)', expand=False)
df['Assessed Value Clean'] = df['Assessed Value Clean'].str.replace(',', '')
df['Assessed Value Clean'] = pd.to_numeric(df['Assessed Value Clean'], errors='coerce')

# Remove rows with missing or zero values - more efficient filtering
df_clean = df.dropna(subset=['Assessed Value Clean', 'Name of Recipient', 'Item Category'])
df_clean = df_clean[df_clean['Assessed Value Clean'] > 0].copy()

# Limit dataset size for performance if needed
if len(df_clean) > 3000:
    df_clean = df_clean.sample(n=3000, random_state=42)

# Calculate total gift value by recipient - optimized
recipient_totals = df_clean.groupby('Name of Recipient')['Assessed Value Clean'].sum().nlargest(5)
top_5_recipients = recipient_totals.index.tolist()

# Filter data for top 5 recipients
top_recipients_data = df_clean[df_clean['Name of Recipient'].isin(top_5_recipients)]

# Create pivot table for stacked bar chart - simplified
category_data = []
recipient_names = []

for recipient in top_5_recipients:
    recipient_data = top_recipients_data[top_recipients_data['Name of Recipient'] == recipient]
    category_totals = recipient_data.groupby('Item Category')['Assessed Value Clean'].sum()
    category_data.append(category_totals)
    # Clean recipient name
    clean_name = recipient.replace('\n', ' ').strip()[:40]  # Truncate long names
    recipient_names.append(clean_name)

# Get all unique categories
all_categories = set()
for data in category_data:
    all_categories.update(data.index)
all_categories = sorted(list(all_categories))

# Create matrix for plotting
plot_data = np.zeros((len(top_5_recipients), len(all_categories)))
for i, data in enumerate(category_data):
    for j, category in enumerate(all_categories):
        plot_data[i, j] = data.get(category, 0)

# Create value brackets for pie chart - simplified
def categorize_value(value):
    if value < 1000:
        return "Under Rs.1,000"
    elif value < 5000:
        return "Rs.1,000-5,000"
    elif value < 20000:
        return "Rs.5,000-20,000"
    else:
        return "Above Rs.20,000"

df_clean['Value_Category'] = df_clean['Assessed Value Clean'].apply(categorize_value)
value_counts = df_clean['Value_Category'].value_counts()

# Create 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Top subplot: Horizontal stacked bar chart
colors = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
bottom = np.zeros(len(top_5_recipients))

for j, category in enumerate(all_categories):
    values = plot_data[:, j]
    ax1.barh(range(len(recipient_names)), values, left=bottom, 
             label=category, color=colors[j], height=0.6)
    bottom += values

ax1.set_title('Gift Categories Distribution - Top 5 Recipients by Total Value', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Total Gift Value (Rs.)', fontsize=11)
ax1.set_ylabel('Recipients', fontsize=11)
ax1.set_yticks(range(len(recipient_names)))
ax1.set_yticklabels(recipient_names, fontsize=9)

# Format x-axis
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(axis='x', alpha=0.3)

# Bottom subplot: Pie chart
total_gifts = len(df_clean)
pie_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

# Create labels with percentages and counts
labels = []
for cat in value_counts.index:
    pct = (value_counts[cat] / total_gifts * 100)
    labels.append(f'{cat}\n{pct:.1f}%\n({value_counts[cat]:,} gifts)')

wedges, texts = ax2.pie(value_counts.values, labels=labels, 
                       colors=pie_colors[:len(value_counts)], 
                       startangle=90, textprops={'fontsize': 9})

ax2.set_title('Gift Distribution by Monetary Value Ranges', 
              fontsize=14, fontweight='bold', pad=15)

# Layout adjustments
plt.tight_layout()
plt.subplots_adjust(right=0.8, hspace=0.4)

# Save the plot
plt.savefig('toshakhana_composition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()