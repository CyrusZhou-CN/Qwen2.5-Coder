import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all datasets with error handling and sampling for performance
datasets = {}
dataset_files = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
dataset_names = ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']

# Sample size for faster processing (adjust if needed)
sample_size = 50000

for i, file in enumerate(dataset_files):
    try:
        # Load dataset with sampling for performance
        df = pd.read_csv(file)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        datasets[dataset_names[i]] = df
        print(f"Loaded {dataset_names[i]}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Calculate fraud rates and credit risk statistics
fraud_rates = {}
credit_risk_stats = {}

for name, df in datasets.items():
    # Calculate fraud rate (percentage)
    fraud_rate = (df['fraud_bool'].sum() / len(df)) * 100
    fraud_rates[name] = fraud_rate
    
    # Calculate credit risk score statistics for fraudulent vs non-fraudulent
    fraud_data = df[df['fraud_bool'] == 1]
    non_fraud_data = df[df['fraud_bool'] == 0]
    
    if len(fraud_data) > 0 and len(non_fraud_data) > 0:
        credit_risk_stats[name] = {
            'fraud_mean': fraud_data['credit_risk_score'].mean(),
            'non_fraud_mean': non_fraud_data['credit_risk_score'].mean()
        }
    else:
        # Handle edge case where one category might be missing
        credit_risk_stats[name] = {
            'fraud_mean': df['credit_risk_score'].mean(),
            'non_fraud_mean': df['credit_risk_score'].mean()
        }

# Calculate overall mean fraud rate across all datasets
overall_fraud_rate = np.mean(list(fraud_rates.values()))

# Calculate deviations from overall mean
deviations = {name: rate - overall_fraud_rate for name, rate in fraud_rates.items()}

# Create figure with 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
variant_names = list(fraud_rates.keys())

# Top plot: Diverging bar chart for fraud rate deviations
y_pos = np.arange(len(variant_names))
deviation_values = [deviations[name] for name in variant_names]

# Create horizontal bars extending from center
bars = ax1.barh(y_pos, deviation_values, color=colors[:len(variant_names)], alpha=0.8, height=0.6)

# Add vertical reference line at zero
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Customize top plot
ax1.set_yticks(y_pos)
ax1.set_yticklabels(variant_names, fontsize=11)
ax1.set_xlabel('Deviation from Overall Mean Fraud Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Fraud Rate Deviations Across Dataset Variants', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, deviation_values)):
    if value >= 0:
        ax1.text(value + max(abs(min(deviation_values)), abs(max(deviation_values))) * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f'+{value:.3f}%', va='center', ha='left', fontsize=10, fontweight='bold')
    else:
        ax1.text(value - max(abs(min(deviation_values)), abs(max(deviation_values))) * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}%', va='center', ha='right', fontsize=10, fontweight='bold')

# Bottom plot: Dumbbell plot for credit risk scores
y_pos_bottom = np.arange(len(variant_names))

fraud_means = [credit_risk_stats[name]['fraud_mean'] for name in variant_names]
non_fraud_means = [credit_risk_stats[name]['non_fraud_mean'] for name in variant_names]

# Calculate overall averages for reference lines
overall_fraud_mean = np.mean(fraud_means)
overall_non_fraud_mean = np.mean(non_fraud_means)

# Plot connecting lines
for i in range(len(variant_names)):
    ax2.plot([non_fraud_means[i], fraud_means[i]], [y_pos_bottom[i], y_pos_bottom[i]], 
             color='gray', linewidth=2, alpha=0.6)

# Plot points for non-fraudulent and fraudulent cases
scatter1 = ax2.scatter(non_fraud_means, y_pos_bottom, color='#4CAF50', s=120, 
                      label='Non-Fraudulent', alpha=0.9, edgecolors='white', linewidth=2)
scatter2 = ax2.scatter(fraud_means, y_pos_bottom, color='#F44336', s=120, 
                      label='Fraudulent', alpha=0.9, edgecolors='white', linewidth=2)

# Add reference lines for overall averages
ax2.axvline(x=overall_non_fraud_mean, color='#4CAF50', linestyle='--', 
           linewidth=2, alpha=0.7, label=f'Overall Non-Fraud Avg ({overall_non_fraud_mean:.1f})')
ax2.axvline(x=overall_fraud_mean, color='#F44336', linestyle='--', 
           linewidth=2, alpha=0.7, label=f'Overall Fraud Avg ({overall_fraud_mean:.1f})')

# Customize bottom plot
ax2.set_yticks(y_pos_bottom)
ax2.set_yticklabels(variant_names, fontsize=11)
ax2.set_xlabel('Credit Risk Score', fontsize=12, fontweight='bold')
ax2.set_title('Credit Risk Score Ranges: Fraudulent vs Non-Fraudulent Cases', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# Add gap annotations
for i, name in enumerate(variant_names):
    gap = fraud_means[i] - non_fraud_means[i]
    mid_point = (fraud_means[i] + non_fraud_means[i]) / 2
    ax2.text(mid_point, y_pos_bottom[i] + 0.15, f'Gap: {gap:.1f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Print summary statistics
print(f"\nFraud Rate Summary:")
for name in variant_names:
    print(f"{name}: {fraud_rates[name]:.3f}% (deviation: {deviations[name]:+.3f}%)")

print(f"\nOverall fraud rate: {overall_fraud_rate:.3f}%")
print(f"Overall non-fraud credit score: {overall_non_fraud_mean:.1f}")
print(f"Overall fraud credit score: {overall_fraud_mean:.1f}")

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4)

plt.savefig('fraud_detection_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()