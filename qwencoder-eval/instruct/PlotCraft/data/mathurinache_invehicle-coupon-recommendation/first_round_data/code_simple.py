import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Define the bins for the acceptance rate
bins = [0, 0.25, 0.5, 0.75, 1]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot histograms for each coupon type with different colors
for coupon_type in data['coupon'].unique():
    subset = data[data['coupon'] == coupon_type]
    ax.hist(subset['Y'], bins=bins, alpha=0.5, label=coupon_type)

# Set labels and title
ax.set_xlabel('Acceptance Rate')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Coupon Acceptance Rates by Coupon Type')
ax.legend(title='Coupon Type')

# Show the plot
plt.show()