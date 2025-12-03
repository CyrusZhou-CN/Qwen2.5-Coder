import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of Excel files to load
excel_files = [
    'trendyol_xiaomi_saat_yorum_rating.xlsx',
    'trendyol_reeder_yorum_rating.xlsx',
    'trendyol_mateo_saat_yorum_rating.xlsx',
    'trendyol_samsung_telefon_yorum.xlsx',
    'trendyol_xiaomi_yorum_rating.xlsx',
    'trendyol_huawei_saat_yorum_rating.xlsx',
    'trendyol_apple_watch_yorum_rating.xlsx',
    'trendyol_samsung_watch_yorum_rating.xlsx',
    'trendyol_iphone_yorum.xlsx'
]

# Load and combine all rating data
all_ratings = []

for file in excel_files:
    try:
        df = pd.read_excel(file)
        # Extract ratings from the 'Yıldız' column
        ratings = df['Yıldız'].dropna()
        all_ratings.extend(ratings.tolist())
        print(f"Loaded {len(ratings)} ratings from {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Convert to numpy array for easier handling
all_ratings = np.array(all_ratings)

# Create the histogram with white background and professional styling
plt.figure(figsize=(12, 8))
plt.style.use('default')  # Ensure clean default style

# Create histogram with appropriate bins for 1-5 rating scale
bins = np.arange(0.5, 6.5, 1)  # Creates bins centered on 1, 2, 3, 4, 5
counts, bin_edges, patches = plt.hist(all_ratings, bins=bins, 
                                     color='#2E86AB', alpha=0.8, 
                                     edgecolor='white', linewidth=1.2)

# Customize colors for each rating level
colors = ['#D32F2F', '#FF9800', '#FFC107', '#4CAF50', '#1976D2']  # Red to Blue gradient
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])

# Add count labels on top of each bar
for i, count in enumerate(counts):
    plt.text(i + 1, count + max(counts) * 0.01, f'{int(count):,}', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Styling and labels
plt.title('Distribution of Customer Ratings Across All Product Categories', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Rating (Stars)', fontsize=14, fontweight='bold')
plt.ylabel('Number of Reviews', fontsize=14, fontweight='bold')

# Set x-axis ticks to show star ratings clearly
plt.xticks([1, 2, 3, 4, 5], ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'])

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Add statistics text box
total_reviews = len(all_ratings)
avg_rating = np.mean(all_ratings)
median_rating = np.median(all_ratings)

stats_text = f'Total Reviews: {total_reviews:,}\nAverage Rating: {avg_rating:.2f}\nMedian Rating: {median_rating:.1f}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, fontweight='bold')

# Format y-axis to show thousands separator
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Set white background
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Total number of reviews: {total_reviews:,}")
print(f"Average rating: {avg_rating:.2f}")
print(f"Median rating: {median_rating:.1f}")
print(f"Rating distribution:")
for i in range(1, 6):
    count = np.sum(all_ratings == i)
    percentage = (count / total_reviews) * 100
    print(f"  {i} stars: {count:,} reviews ({percentage:.1f}%)")