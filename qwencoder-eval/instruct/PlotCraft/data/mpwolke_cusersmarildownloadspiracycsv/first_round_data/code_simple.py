import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('piracy.csv')

# Parse the semicolon-separated data
# Get the column data and split by semicolons
data_rows = []
for index, row in df.iterrows():
    # Split the row by semicolons and clean up whitespace
    split_row = [item.strip() for item in str(row.iloc[0]).split(';')]
    if len(split_row) >= 9:  # Ensure we have enough columns
        data_rows.append(split_row)

# Create DataFrame from parsed data
parsed_df = pd.DataFrame(data_rows, columns=['name', 'party', 'state', 'money_pro', 'money_con', 'years', 'stance', 'chamber', 'house'])

# Remove header row if it exists and filter for valid party data
parsed_df = parsed_df[parsed_df['party'].isin(['D', 'R'])]

# Count legislators by party
party_counts = parsed_df['party'].value_counts()

# Ensure we have data to plot
if len(party_counts) == 0:
    print("No valid party data found")
else:
    # Map party codes to full names
    party_labels = {'D': 'Democrat', 'R': 'Republican'}
    party_names = [party_labels[party] for party in party_counts.index]
    
    # Create figure with appropriate size
    plt.figure(figsize=(10, 8))
    
    # Define conventional political colors
    colors = ['#1f77b4', '#d62728']  # Blue for Democrat, Red for Republican
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(party_counts.values, 
                                      labels=party_names,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      textprops={'fontsize': 12},
                                      explode=(0.05, 0.05))  # Slight separation for clarity
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    # Style the party labels
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
    
    # Add bold title with proper spacing
    plt.title('Composition of US Legislators by Political Party Affiliation', 
              fontsize=18, fontweight='bold', pad=30)
    
    # Ensure equal aspect ratio for circular pie chart
    plt.axis('equal')
    
    # Add legend with counts positioned to avoid overlap
    legend_labels = [f'{name}: {count} legislators' for name, count in zip(party_names, party_counts.values)]
    plt.legend(wedges, legend_labels, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5),
              fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plot
    plt.show()