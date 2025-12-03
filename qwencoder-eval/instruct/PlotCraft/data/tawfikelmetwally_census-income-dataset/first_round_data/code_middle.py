import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('adult.csv')

# Data preprocessing - clean column names and prepare data
df.columns = df.columns.str.strip()  # Remove any whitespace from column names

# Create income groups
low_income = df[df['Income'] == '<=50K']
high_income = df[df['Income'] == '>50K']

# Filter capital gain data (<=10000 as specified)
low_income_cap = low_income[low_income['Capital Gain'] <= 10000]
high_income_cap = high_income[high_income['Capital Gain'] <= 10000]

# Set up the figure with white background and professional styling
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Define colors for income groups
colors = ['#3498db', '#e74c3c']  # Blue for <=50K, Red for >50K
alpha = 0.7

# Top-left: Age distribution by income level
n1, bins1, patches1 = axes[0, 0].hist(low_income['Age'], bins=30, alpha=alpha, color=colors[0], 
                                       density=True, label='≤50K', edgecolor='white', linewidth=0.5)
n2, bins2, patches2 = axes[0, 0].hist(high_income['Age'], bins=30, alpha=alpha, color=colors[1], 
                                       density=True, label='>50K', edgecolor='white', linewidth=0.5)

# Add simple KDE approximation using histogram smoothing
def simple_kde(data, bins=50):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Simple smoothing
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(hist, sigma=1.0)
    return bin_centers, smoothed

try:
    from scipy.ndimage import gaussian_filter1d
    age_x_low, age_y_low = simple_kde(low_income['Age'])
    age_x_high, age_y_high = simple_kde(high_income['Age'])
    axes[0, 0].plot(age_x_low, age_y_low, color=colors[0], linewidth=2, alpha=0.8)
    axes[0, 0].plot(age_x_high, age_y_high, color=colors[1], linewidth=2, alpha=0.8)
except ImportError:
    # If scipy not available, skip KDE curves
    pass

axes[0, 0].set_title('Age Distribution by Income Level', fontweight='bold', fontsize=12, pad=15)
axes[0, 0].set_xlabel('Age (years)', fontsize=10)
axes[0, 0].set_ylabel('Density', fontsize=10)
axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Top-right: Hours per week distribution by income level
axes[0, 1].hist(low_income['Hours per Week'], bins=25, alpha=alpha, color=colors[0], 
                density=True, label='≤50K', edgecolor='white', linewidth=0.5)
axes[0, 1].hist(high_income['Hours per Week'], bins=25, alpha=alpha, color=colors[1], 
                density=True, label='>50K', edgecolor='white', linewidth=0.5)

try:
    hours_x_low, hours_y_low = simple_kde(low_income['Hours per Week'])
    hours_x_high, hours_y_high = simple_kde(high_income['Hours per Week'])
    axes[0, 1].plot(hours_x_low, hours_y_low, color=colors[0], linewidth=2, alpha=0.8)
    axes[0, 1].plot(hours_x_high, hours_y_high, color=colors[1], linewidth=2, alpha=0.8)
except:
    pass

axes[0, 1].set_title('Hours per Week Distribution by Income Level', fontweight='bold', fontsize=12, pad=15)
axes[0, 1].set_xlabel('Hours per Week', fontsize=10)
axes[0, 1].set_ylabel('Density', fontsize=10)
axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Bottom-left: Education number distribution by income level
axes[1, 0].hist(low_income['EducationNum'], bins=16, alpha=alpha, color=colors[0], 
                density=True, label='≤50K', edgecolor='white', linewidth=0.5)
axes[1, 0].hist(high_income['EducationNum'], bins=16, alpha=alpha, color=colors[1], 
                density=True, label='>50K', edgecolor='white', linewidth=0.5)

try:
    edu_x_low, edu_y_low = simple_kde(low_income['EducationNum'], bins=20)
    edu_x_high, edu_y_high = simple_kde(high_income['EducationNum'], bins=20)
    axes[1, 0].plot(edu_x_low, edu_y_low, color=colors[0], linewidth=2, alpha=0.8)
    axes[1, 0].plot(edu_x_high, edu_y_high, color=colors[1], linewidth=2, alpha=0.8)
except:
    pass

axes[1, 0].set_title('Education Level Distribution by Income Level', fontweight='bold', fontsize=12, pad=15)
axes[1, 0].set_xlabel('Education Number (years)', fontsize=10)
axes[1, 0].set_ylabel('Density', fontsize=10)
axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Bottom-right: Capital gain distribution by income level (limited to <=10000)
axes[1, 1].hist(low_income_cap['Capital Gain'], bins=30, alpha=alpha, color=colors[0], 
                density=True, label='≤50K', edgecolor='white', linewidth=0.5)
axes[1, 1].hist(high_income_cap['Capital Gain'], bins=30, alpha=alpha, color=colors[1], 
                density=True, label='>50K', edgecolor='white', linewidth=0.5)

# Add KDE curves for capital gain only if there's variation in the data
try:
    if low_income_cap['Capital Gain'].std() > 0:
        cap_x_low, cap_y_low = simple_kde(low_income_cap['Capital Gain'])
        axes[1, 1].plot(cap_x_low, cap_y_low, color=colors[0], linewidth=2, alpha=0.8)
    if high_income_cap['Capital Gain'].std() > 0:
        cap_x_high, cap_y_high = simple_kde(high_income_cap['Capital Gain'])
        axes[1, 1].plot(cap_x_high, cap_y_high, color=colors[1], linewidth=2, alpha=0.8)
except:
    pass

axes[1, 1].set_title('Capital Gain Distribution by Income Level\n(Limited to ≤$10,000)', 
                     fontweight='bold', fontsize=12, pad=15)
axes[1, 1].set_xlabel('Capital Gain ($)', fontsize=10)
axes[1, 1].set_ylabel('Density', fontsize=10)
axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Set white background for all subplots
for ax in axes.flat:
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

# Overall title
fig.suptitle('Income Distribution Analysis: Key Demographic and Work Factors', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
plt.savefig('income_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()