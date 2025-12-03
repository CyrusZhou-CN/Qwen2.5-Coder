import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
samsung_phone = pd.read_excel('trendyol_samsung_telefon_yorum.xlsx')
xiaomi_phone = pd.read_excel('trendyol_xiaomi_yorum_rating.xlsx')
huawei_watch = pd.read_excel('trendyol_huawei_saat_yorum_rating.xlsx')
apple_watch = pd.read_excel('trendyol_apple_watch_yorum_rating.xlsx')
samsung_watch = pd.read_excel('trendyol_samsung_watch_yorum_rating.xlsx')
iphone = pd.read_excel('trendyol_iphone_yorum.xlsx')

# Extract brand names and ratings
brands = {
    'Samsung Phone': samsung_phone['Telefon'].str.extract(r'^(.*?)\n')[0].unique(),
    'Xiaomi Phone': xiaomi_phone['Telefon'].str.extract(r'^(.*?)\n')[0].unique(),
    'Huawei Watch': huawei_watch['Telefon'].str.extract(r'^(.*?)\n')[0].unique(),
    'Apple Watch': apple_watch['Telefon'].str.extract(r'^(.*?)\n')[0].unique(),
    'Samsung Watch': samsung_watch['Telefon'].str.extract(r'^(.*?)\n')[0].unique(),
    'iPhone': iphone['Telefon'].str.extract(r'^(.*?)\n')[0].unique()
}

ratings = {
    'Samsung Phone': samsung_phone['Yıldız'].mean(),
    'Xiaomi Phone': xiaomi_phone['Yıldız'].mean(),
    'Huawei Watch': huawei_watch['Yıldız'].mean(),
    'Apple Watch': apple_watch['Yıldız'].mean(),
    'Samsung Watch': samsung_watch['Yıldız'].mean(),
    'iPhone': iphone['Yıldız'].mean()
}

# Sort brands by average rating in descending order
sorted_brands = sorted(ratings.items(), key=lambda item: item[1], reverse=True)

# Prepare data for plotting
labels = [brand for brand, _ in sorted_brands]
values = [rating for _, rating in sorted_brands]

# Define colors for different categories
colors = ['blue', 'green', 'red', 'purple', 'orange', 'gray']

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(labels, values, color=colors)

# Add text labels with average ratings
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', ha='left')

# Set title and labels
plt.title('Ranking of Smartphone and Smartwatch Brands by Average Customer Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Brands')

# Show plot
plt.show()