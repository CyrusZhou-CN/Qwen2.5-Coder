import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
transactions_df = pd.read_csv('transactions.csv')
alerts_df = pd.read_csv('alerts.csv')

# Data preprocessing
# Calculate overall mean transaction amount across all transactions
overall_mean = transactions_df['TX_AMOUNT'].mean()

# For top subplot: Calculate average transaction amounts by alert type for fraudulent transactions
alert_avg = alerts_df.groupby('ALERT_TYPE')['TX_AMOUNT'].mean().reset_index()
alert_avg['deviation'] = alert_avg['TX_AMOUNT'] - overall_mean

# For bottom subplot: Calculate ranges for fraudulent vs non-fraudulent
fraud_transactions = transactions_df[transactions_df['IS_FRAUD'] == True]
non_fraud_transactions = transactions_df[transactions_df['IS_FRAUD'] == False]

# Calculate min and max for each category
fraud_min = fraud_transactions['TX_AMOUNT'].min()
fraud_max = fraud_transactions['TX_AMOUNT'].max()
non_fraud_min = non_fraud_transactions['TX_AMOUNT'].min()
non_fraud_max = non_fraud_transactions['TX_AMOUNT'].max()

# Create figure with 2x1 layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Top subplot: Diverging bar chart
colors = ['#d73027' if x < 0 else '#1a9850' for x in alert_avg['deviation']]
bars = ax1.barh(alert_avg['ALERT_TYPE'], alert_avg['deviation'], color=colors, alpha=0.8)

# Add vertical line at zero
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Styling for top subplot
ax1.set_xlabel('Deviation from Overall Mean Transaction Amount ($)', fontweight='bold')
ax1.set_ylabel('Alert Type', fontweight='bold')
ax1.set_title('Transaction Amount Deviations by Alert Type\n(Fraudulent vs Overall Mean)', 
              fontweight='bold', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_facecolor('white')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, alert_avg['deviation'])):
    width = bar.get_width()
    label_x = width + (5 if width >= 0 else -5)
    ha = 'left' if width >= 0 else 'right'
    ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'${value:.2f}', 
             ha=ha, va='center', fontweight='bold')

# Bottom subplot: Dumbbell plot
categories = ['Non-Fraudulent', 'Fraudulent']
y_positions = [0, 1]

# Plot the ranges as horizontal lines
ax2.plot([non_fraud_min, non_fraud_max], [0, 0], 'o-', color='#2166ac', 
         linewidth=3, markersize=8, label='Non-Fraudulent Range')
ax2.plot([fraud_min, fraud_max], [1, 1], 'o-', color='#d73027', 
         linewidth=3, markersize=8, label='Fraudulent Range')

# Add connecting lines to show the gap
ax2.plot([non_fraud_min, fraud_min], [0, 1], '--', color='gray', alpha=0.6, linewidth=1)
ax2.plot([non_fraud_max, fraud_max], [0, 1], '--', color='gray', alpha=0.6, linewidth=1)

# Add value labels
ax2.text(non_fraud_min, -0.1, f'${non_fraud_min:.2f}', ha='center', va='top', fontweight='bold')
ax2.text(non_fraud_max, -0.1, f'${non_fraud_max:.2f}', ha='center', va='top', fontweight='bold')
ax2.text(fraud_min, 1.1, f'${fraud_min:.2f}', ha='center', va='bottom', fontweight='bold')
ax2.text(fraud_max, 1.1, f'${fraud_max:.2f}', ha='center', va='bottom', fontweight='bold')

# Styling for bottom subplot
ax2.set_yticks(y_positions)
ax2.set_yticklabels(categories)
ax2.set_xlabel('Transaction Amount Range ($)', fontweight='bold')
ax2.set_ylabel('Transaction Type', fontweight='bold')
ax2.set_title('Transaction Amount Ranges: Fraudulent vs Non-Fraudulent', 
              fontweight='bold', fontsize=14, pad=20)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_facecolor('white')
ax2.legend(loc='upper right')

# Set y-axis limits to provide better spacing
ax2.set_ylim(-0.3, 1.3)

# Layout adjustment
plt.tight_layout()
plt.show()