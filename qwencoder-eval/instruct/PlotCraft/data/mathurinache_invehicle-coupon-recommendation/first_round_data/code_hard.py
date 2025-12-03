import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Data preprocessing
# Handle missing values
df = df.fillna('Unknown')

# Create numerical encodings for categorical variables
le_age = LabelEncoder()
le_income = LabelEncoder()
le_education = LabelEncoder()
le_marital = LabelEncoder()
le_time = LabelEncoder()
le_coupon = LabelEncoder()

# Encode categorical variables
df['age_numeric'] = le_age.fit_transform(df['age'])
df['income_numeric'] = le_income.fit_transform(df['income'])
df['education_numeric'] = le_education.fit_transform(df['education'])
df['marital_numeric'] = le_marital.fit_transform(df['maritalStatus'])
df['time_numeric'] = le_time.fit_transform(df['time'])
df['coupon_numeric'] = le_coupon.fit_transform(df['coupon'])

# Convert frequency columns to numeric
freq_cols = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
for col in freq_cols:
    df[col] = df[col].replace({'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4})
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create restaurant visit frequency sum
df['restaurant_freq'] = df['RestaurantLessThan20'] + df['Restaurant20To50']

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Scatter plot with regression line (age vs income)
ax1 = plt.subplot(3, 3, 1, facecolor='white')
colors = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in df['Y']]
sizes = [20 + df['restaurant_freq'].iloc[i] * 10 for i in range(len(df))]
scatter = ax1.scatter(df['age_numeric'], df['income_numeric'], c=colors, s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)
z = np.polyfit(df['age_numeric'], df['income_numeric'], 1)
p = np.poly1d(z)
ax1.plot(df['age_numeric'], p(df['age_numeric']), "r--", alpha=0.8, linewidth=2)
ax1.set_title('Age vs Income by Acceptance Status', fontweight='bold', fontsize=12, pad=15)
ax1.set_xlabel('Age (Encoded)', fontsize=10)
ax1.set_ylabel('Income (Encoded)', fontsize=10)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Violin plot with box plot (education vs acceptance by gender)
ax2 = plt.subplot(3, 3, 2, facecolor='white')
education_order = sorted(df['education'].unique())
gender_data = []
for gender in ['Male', 'Female']:
    for edu in education_order:
        subset = df[(df['gender'] == gender) & (df['education'] == edu)]
        if len(subset) > 0:
            gender_data.append({
                'gender': gender,
                'education': edu,
                'acceptance': subset['Y'].tolist()
            })

positions = []
data_to_plot = []
labels = []
colors_violin = []
for i, edu in enumerate(education_order):
    male_data = df[(df['gender'] == 'Male') & (df['education'] == edu)]['Y']
    female_data = df[(df['gender'] == 'Female') & (df['education'] == edu)]['Y']
    
    if len(male_data) > 0:
        positions.append(i * 2)
        data_to_plot.append(male_data)
        labels.append(f'M-{edu[:8]}')
        colors_violin.append('#87CEEB')
    
    if len(female_data) > 0:
        positions.append(i * 2 + 0.8)
        data_to_plot.append(female_data)
        labels.append(f'F-{edu[:8]}')
        colors_violin.append('#FFB6C1')

if data_to_plot:
    parts = ax2.violinplot(data_to_plot, positions=positions, widths=0.6, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)

ax2.set_title('Acceptance by Education & Gender', fontweight='bold', fontsize=12, pad=15)
ax2.set_ylabel('Acceptance Rate', fontsize=10)
ax2.set_xticks(positions[::2])
ax2.set_xticklabels([label.split('-')[1] for label in labels[::2]], rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Row 1, Subplot 3: Bubble chart (time vs temperature)
ax3 = plt.subplot(3, 3, 3, facecolor='white')
time_temp_data = df.groupby(['time_numeric', 'temperature']).agg({
    'Y': ['mean', 'count'],
    'coupon': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
}).reset_index()
time_temp_data.columns = ['time_numeric', 'temperature', 'acceptance_rate', 'count', 'coupon_mode']

coupon_colors = {'Bar': '#FF6B6B', 'Coffee House': '#4ECDC4', 'Restaurant(<20)': '#45B7D1', 
                'Restaurant(20-50)': '#96CEB4', 'Carry out & Take away': '#FFEAA7'}
colors_bubble = [coupon_colors.get(coupon, '#DDA0DD') for coupon in time_temp_data['coupon_mode']]

bubble_sizes = time_temp_data['acceptance_rate'] * 200 + 50
ax3.scatter(time_temp_data['time_numeric'], time_temp_data['temperature'], 
           s=bubble_sizes, c=colors_bubble, alpha=0.6, edgecolors='white', linewidth=1)
ax3.set_title('Time vs Temperature by Acceptance', fontweight='bold', fontsize=12, pad=15)
ax3.set_xlabel('Time (Encoded)', fontsize=10)
ax3.set_ylabel('Temperature', fontsize=10)
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Stacked bar with line overlay (marital status)
ax4 = plt.subplot(3, 3, 4, facecolor='white')
marital_data = df.groupby('maritalStatus').agg({
    'Y': ['sum', 'count', 'mean']
}).reset_index()
marital_data.columns = ['maritalStatus', 'accepted', 'total', 'acceptance_rate']
marital_data['rejected'] = marital_data['total'] - marital_data['accepted']

x_pos = np.arange(len(marital_data))
ax4.bar(x_pos, marital_data['rejected'], label='Rejected', color='#FF6B6B', alpha=0.7)
ax4.bar(x_pos, marital_data['accepted'], bottom=marital_data['rejected'], 
        label='Accepted', color='#4ECDC4', alpha=0.7)

ax4_twin = ax4.twinx()
ax4_twin.plot(x_pos, marital_data['acceptance_rate'], 'ko-', linewidth=2, markersize=6, label='Acceptance Rate')
ax4_twin.set_ylabel('Acceptance Rate', fontsize=10)

ax4.set_title('Acceptance by Marital Status', fontweight='bold', fontsize=12, pad=15)
ax4.set_xlabel('Marital Status', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(marital_data['maritalStatus'], rotation=45, ha='right')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Correlation heatmap
ax5 = plt.subplot(3, 3, 5, facecolor='white')
numeric_cols = ['age_numeric', 'temperature', 'toCoupon_GEQ5min', 'toCoupon_GEQ15min', 
                'toCoupon_GEQ25min', 'direction_same', 'Y', 'income_numeric']
corr_matrix = df[numeric_cols].corr()

im = ax5.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax5.set_xticks(range(len(numeric_cols)))
ax5.set_yticks(range(len(numeric_cols)))
ax5.set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45, ha='right')
ax5.set_yticklabels([col.replace('_', '\n') for col in numeric_cols])

# Add correlation values as text
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha="center", va="center", color="black", fontsize=8, fontweight='bold')

ax5.set_title('Correlation Matrix', fontweight='bold', fontsize=12, pad=15)
plt.colorbar(im, ax=ax5, shrink=0.8)

# Row 2, Subplot 6: Scatter with marginal histograms
ax6 = plt.subplot(3, 3, 6, facecolor='white')
colors_scatter = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in df['Y']]
ax6.scatter(df['toCoupon_GEQ15min'], df['direction_same'], c=colors_scatter, alpha=0.6, s=30)
ax6.set_title('Distance vs Direction Alignment', fontweight='bold', fontsize=12, pad=15)
ax6.set_xlabel('Distance ≥15min', fontsize=10)
ax6.set_ylabel('Same Direction', fontsize=10)
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Jitter plot with violin (behavioral patterns)
ax7 = plt.subplot(3, 3, 7, facecolor='white')
behavioral_data = []
coupon_types = df['coupon'].unique()[:5]  # Limit to 5 types for clarity
for i, coupon_type in enumerate(coupon_types):
    subset = df[df['coupon'] == coupon_type]
    behavioral_score = subset['Bar'] + subset['CoffeeHouse'] + subset['CarryAway']
    acceptance = subset['Y']
    
    # Add jitter
    x_jitter = np.random.normal(i, 0.1, len(behavioral_score))
    colors_jitter = ['#FF6B6B' if y == 0 else '#4ECDC4' for y in acceptance]
    ax7.scatter(x_jitter, behavioral_score, c=colors_jitter, alpha=0.6, s=20)

ax7.set_title('Behavioral Patterns by Coupon Type', fontweight='bold', fontsize=12, pad=15)
ax7.set_xlabel('Coupon Type', fontsize=10)
ax7.set_ylabel('Behavioral Score', fontsize=10)
ax7.set_xticks(range(len(coupon_types)))
ax7.set_xticklabels([ct[:10] for ct in coupon_types], rotation=45, ha='right')
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: Parallel coordinates plot
ax8 = plt.subplot(3, 3, 8, facecolor='white')
parallel_cols = ['passanger', 'destination', 'weather', 'Y']
parallel_data = df[parallel_cols].copy()

# Encode categorical variables for parallel plot
for col in parallel_cols[:-1]:
    le_temp = LabelEncoder()
    parallel_data[col] = le_temp.fit_transform(parallel_data[col])

# Normalize data
for col in parallel_cols:
    parallel_data[col] = (parallel_data[col] - parallel_data[col].min()) / (parallel_data[col].max() - parallel_data[col].min())

# Sample data for clarity
sample_size = min(1000, len(parallel_data))
sample_indices = np.random.choice(len(parallel_data), sample_size, replace=False)
sample_data = parallel_data.iloc[sample_indices]

for i in range(len(sample_data)):
    color = '#4ECDC4' if sample_data.iloc[i]['Y'] > 0.5 else '#FF6B6B'
    ax8.plot(range(len(parallel_cols)), sample_data.iloc[i], color=color, alpha=0.3, linewidth=0.5)

ax8.set_title('Parallel Coordinates Plot', fontweight='bold', fontsize=12, pad=15)
ax8.set_xticks(range(len(parallel_cols)))
ax8.set_xticklabels(parallel_cols, rotation=45, ha='right')
ax8.set_ylabel('Normalized Values', fontsize=10)
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Network-style correlation plot
ax9 = plt.subplot(3, 3, 9, facecolor='white')
behavioral_cols = ['Bar', 'CoffeeHouse', 'CarryAway', 'temperature', 'toCoupon_GEQ15min', 'Y']
network_corr = df[behavioral_cols].corr()

# Create network layout
n_vars = len(behavioral_cols)
angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
x_pos = np.cos(angles)
y_pos = np.sin(angles)

# Draw nodes
for i, (x, y, var) in enumerate(zip(x_pos, y_pos, behavioral_cols)):
    ax9.scatter(x, y, s=300, c='#4ECDC4', alpha=0.8, edgecolors='white', linewidth=2)
    ax9.text(x*1.2, y*1.2, var, ha='center', va='center', fontsize=9, fontweight='bold')

# Draw edges based on correlation strength
for i in range(n_vars):
    for j in range(i+1, n_vars):
        corr_val = abs(network_corr.iloc[i, j])
        if corr_val > 0.1:  # Only show significant correlations
            color = '#FF6B6B' if network_corr.iloc[i, j] < 0 else '#4ECDC4'
            ax9.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                    color=color, alpha=corr_val, linewidth=corr_val*5)

ax9.set_title('Behavioral Network Correlations', fontweight='bold', fontsize=12, pad=15)
ax9.set_xlim(-1.5, 1.5)
ax9.set_ylim(-1.5, 1.5)
ax9.set_aspect('equal')
ax9.axis('off')

# Adjust layout and display
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()