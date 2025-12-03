import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data - it appears to be semicolon-delimited in a single column
df = pd.read_csv('piracy.csv')

# Check if data needs to be split (if it's all in one column)
if len(df.columns) == 1:
    # Split the single column by semicolons
    column_name = df.columns[0]
    # Split the data by semicolons
    split_data = df[column_name].str.split(';', expand=True)
    
    # Get column names from the first row (header)
    if 'name;party;state;money_pro;money_con;years;stance;chamber;house' in df.columns[0]:
        # The column name itself contains the header, so we need to parse it differently
        header_row = df.columns[0]
        col_names = header_row.split(';')
        
        # Split all data rows
        split_data = df[df.columns[0]].str.split(';', expand=True)
        
        # Create new dataframe with proper column names
        df = pd.DataFrame(split_data.values, columns=col_names)
        
        # Remove any rows that might be duplicates of the header
        df = df[df['name'] != 'name']
    else:
        # First row contains headers
        col_names = split_data.iloc[0].tolist()
        df = pd.DataFrame(split_data.iloc[1:].values, columns=col_names)

# Clean column names and data
df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Reset index
df = df.reset_index(drop=True)

# Data preprocessing
# Handle missing values and clean data
df['stance'] = df['stance'].fillna('unknown')
df['party'] = df['party'].astype(str).str.strip()
df['stance'] = df['stance'].astype(str).str.strip()
df['chamber'] = df['chamber'].astype(str).str.strip()

# Map stance values for consistency
stance_mapping = {'yes': 'support', 'no': 'oppose', 'unknown': 'unknown'}
df['stance'] = df['stance'].map(stance_mapping).fillna('unknown')

# Create the composite visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Left plot: Stacked bar chart of stance by party
# Get the main parties (D and R)
main_parties_df = df[df['party'].isin(['D', 'R'])]
party_stance = main_parties_df.groupby(['party', 'stance']).size().unstack(fill_value=0)

# Ensure all stance categories are present
for stance in ['support', 'oppose', 'unknown']:
    if stance not in party_stance.columns:
        party_stance[stance] = 0

# Reorder columns for consistent display
party_stance = party_stance[['support', 'oppose', 'unknown']]

# Define colors for stance categories
stance_colors = {'support': '#2E8B57', 'oppose': '#DC143C', 'unknown': '#708090'}

# Create stacked bar chart
party_stance.plot(kind='bar', stacked=True, ax=ax1, 
                 color=[stance_colors[col] for col in party_stance.columns],
                 width=0.6)

ax1.set_title('Anti-Piracy Legislation Stance by Political Party', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Political Party', fontweight='bold')
ax1.set_ylabel('Number of Legislators', fontweight='bold')
ax1.set_xticklabels(['Democratic', 'Republican'], rotation=0)
ax1.legend(title='Stance', title_fontsize=10, fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Right plot: Pie chart of chamber composition
chamber_counts = df['chamber'].value_counts()

# Define colors for chambers
chamber_colors = ['#4682B4', '#FF6347']

# Create labels for the pie chart
chamber_labels = []
for chamber in chamber_counts.index:
    if chamber.lower() == 'house':
        chamber_labels.append('House of Representatives')
    elif chamber.lower() == 'senate':
        chamber_labels.append('Senate')
    else:
        chamber_labels.append(chamber.title())

# Create pie chart
wedges, texts, autotexts = ax2.pie(chamber_counts.values, 
                                  labels=chamber_labels, 
                                  colors=chamber_colors[:len(chamber_counts)],
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  textprops={'fontsize': 10})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')

ax2.set_title('Legislative Chamber Composition', fontweight='bold', fontsize=14, pad=20)

# Ensure equal aspect ratio for circular pie chart
ax2.axis('equal')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(wspace=0.3)

plt.show()