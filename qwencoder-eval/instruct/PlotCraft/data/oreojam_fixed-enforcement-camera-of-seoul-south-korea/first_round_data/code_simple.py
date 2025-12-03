import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('fixed_cctv_for_parking_enforcement.csv')

# Count the number of cameras per district
camera_counts = data['district'].value_counts()

# Plotting the histogram
plt.figure(figsize=(10, 6))
camera_counts.plot(kind='bar')
plt.title('Distribution of Enforcement Cameras Across Seoul Districts')
plt.xlabel('District')
plt.ylabel('Number of Cameras')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()