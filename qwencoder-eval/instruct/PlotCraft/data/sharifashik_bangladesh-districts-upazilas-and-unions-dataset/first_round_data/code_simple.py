import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
upozila_df = pd.read_csv('upozila.csv')
district_df = pd.read_csv('district.csv')

# Merge the datasets on district_id
merged_df = pd.merge(upozila_df, district_df, on='district_id')

# Group by district and count the number of upazilas
district_upazila_count = merged_df.groupby('জেলা').size().reset_index(name='upazila_count')

# Sort the DataFrame by upazila_count in descending order
district_upazila_count = district_upazila_count.sort_values(by='upazila_count', ascending=False)

# Select the top 15 districts
top_15_districts = district_upazila_count.head(15)

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(top_15_districts['জেলা'], top_15_districts['upazila_count'], color='skyblue')
plt.xlabel('Number of Upazilas')
plt.ylabel('Districts')
plt.title('Top 15 Districts in Bangladesh Ranked by Number of Upazilas')
plt.show()