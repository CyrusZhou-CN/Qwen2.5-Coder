import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Convert categorical variables to numerical for plotting
data['direction_same'] = data['direction_same'].astype(str)
data['direction_opp'] = data['direction_opp'].astype(str)

# Top plot: Scatter plot with jitter
plt.figure(figsize=(12, 8))

# Scatter plot for toCoupon_GEQ15min
sns.scatterplot(x='toCoupon_GEQ15min', y='Y', hue='coupon', size='direction_same', palette='viridis', sizes=(20, 200), alpha=0.6, data=data)
plt.title('Scatter Plot of Coupon Acceptance vs Driving Distance (toCoupon_GEQ15min)')
plt.xlabel('Driving Distance to Coupon Location (toCoupon_GEQ15min)')
plt.ylabel('Coupon Acceptance (Y)')
plt.legend(title='Coupon Type')
plt.show()

# Bottom plot: Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data[['toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same', 'direction_opp', 'Y']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Distance-Related Variables and Coupon Acceptance')
plt.show()