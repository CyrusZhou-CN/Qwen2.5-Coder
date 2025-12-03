import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('bangladeshi_all_engish_newspapers_daily_news_combined_dataset.csv')

# Count articles by publisher
publisher_counts = df['publisher'].value_counts()

# Calculate percentages
total_articles = len(df)
percentages = (publisher_counts / total_articles) * 100

# Create a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Create the pie chart with white background
plt.figure(figsize=(10, 8))
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Create pie chart with enhanced styling
wedges, texts, autotexts = plt.pie(publisher_counts.values, 
                                  labels=publisher_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90,
                                  explode=(0.05, 0.05, 0.05, 0.05),  # Slight separation for clarity
                                  shadow=True,
                                  textprops={'fontsize': 11, 'fontweight': 'medium'})

# Enhance the percentage text styling
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

# Enhance label text styling
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

# Add title with professional styling
plt.title('Distribution of News Articles by Publisher', 
          fontsize=16, 
          fontweight='bold', 
          pad=20,
          color='#2C3E50')

# Add a subtitle with total count
plt.figtext(0.5, 0.02, f'Total Articles: {total_articles}', 
           ha='center', fontsize=10, style='italic', color='#7F8C8D')

# Ensure equal aspect ratio for perfect circle
plt.axis('equal')

# Adjust layout
plt.tight_layout()

# Display the chart
plt.show()