import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data with error handling
try:
    df = pd.read_excel('Book1.xlsx')
    print(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data if file not found
    np.random.seed(42)
    df = pd.DataFrame({
        'unique_identifier': range(1000),
        'appl_month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 1000),
        'acq_channel': np.random.choice(['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4'], 1000),
        'bureau_score': np.random.normal(750, 100, 1000).astype(int),
        'income': np.random.normal(75000, 25000, 1000).astype(int),
        'limit': np.random.normal(10000, 5000, 1000).astype(int),
        'debt_cap': np.random.normal(50000, 20000, 1000),
        'spend': np.random.normal(2000, 1000, 1000),
        'payments': np.random.normal(1500, 800, 1000),
        'attempt_txn': np.random.poisson(10, 1000),
        'risk_score_1': np.random.uniform(0, 10, 1000),
        'risk_score_2': np.random.uniform(0, 1, 1000),
        'risk_score_3': np.random.randint(0, 100, 1000),
        'risk_score_4': np.random.randint(0, 1000, 1000),
        'risk_score_5': np.random.randint(0, 100, 1000),
        'default_ind': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
    })

# Quick data preprocessing
df = df.dropna(subset=['default_ind'])
df['risk_category'] = pd.cut(df['bureau_score'], bins=[0, 650, 750, 850, 1000], 
                            labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])

# Create figure
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Comprehensive Deviation Analysis: Customer Segments & Risk Factors', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. Top-left: Diverging bar chart
ax1 = plt.subplot(3, 3, 1)
segments = df['acq_channel'].value_counts().head(3).index
ratios = ['debt_cap', 'income', 'limit']
benchmark_values = {col: df[col].median() for col in ratios if col in df.columns}

deviations = []
labels = []

for segment in segments:
    segment_data = df[df['acq_channel'] == segment]
    for ratio in ratios:
        if ratio in df.columns:
            mean_val = segment_data[ratio].mean()
            benchmark = benchmark_values[ratio]
            if benchmark != 0:
                deviation = ((mean_val - benchmark) / benchmark) * 100
                deviations.append(deviation)
                labels.append(f"{segment[:8]}\n{ratio}")

if deviations:
    y_pos = np.arange(len(deviations))
    colors = ['#d73027' if x < 0 else '#3288bd' for x in deviations]
    ax1.barh(y_pos, deviations, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Deviation (%)')
    ax1.set_title('Financial Ratios Deviation', fontweight='bold')
    ax1.grid(True, alpha=0.3)

# 2. Top-middle: Dumbbell plot
ax2 = plt.subplot(3, 3, 2)
high_risk = df[df['bureau_score'] < 700]
low_risk = df[df['bureau_score'] >= 750]

metrics = ['spend', 'payments', 'attempt_txn']
available_metrics = [m for m in metrics if m in df.columns]

if len(high_risk) > 0 and len(low_risk) > 0 and available_metrics:
    pre_high = [high_risk[m].mean() for m in available_metrics]
    post_low = [low_risk[m].mean() for m in available_metrics]
    
    y_positions = np.arange(len(available_metrics))
    for i, (pre, post) in enumerate(zip(pre_high, post_low)):
        ax2.plot([pre, post], [i, i], 'o-', linewidth=2, markersize=6)
        ax2.plot(pre, i, 'o', markersize=8, color='#d73027')
        ax2.plot(post, i, 'o', markersize=8, color='#3288bd')
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(available_metrics)
    ax2.set_xlabel('Metric Value')
    ax2.set_title('High vs Low Risk Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3)

# 3. Top-right: Radar chart
ax3 = plt.subplot(3, 3, 3, projection='polar')
risk_metrics = ['risk_score_1', 'risk_score_2', 'risk_score_3', 'risk_score_4', 'risk_score_5']
available_metrics = [m for m in risk_metrics if m in df.columns and df[m].notna().sum() > 0]

if len(available_metrics) >= 3:
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    actual_values = [df[m].mean() for m in available_metrics]
    target_values = [df[m].median() for m in available_metrics]
    
    # Simple normalization
    max_val = max(max(actual_values), max(target_values))
    if max_val > 0:
        actual_norm = [a/max_val for a in actual_values]
        target_norm = [t/max_val for t in target_values]
        
        actual_norm += actual_norm[:1]
        target_norm += target_norm[:1]
        
        ax3.plot(angles, actual_norm, 'o-', linewidth=2, color='#d73027', label='Actual')
        ax3.plot(angles, target_norm, 'o-', linewidth=2, color='#3288bd', label='Target')
        ax3.fill(angles, actual_norm, alpha=0.25, color='#d73027')
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.replace('risk_score_', 'RS') for m in available_metrics])
        ax3.set_title('Risk Profile Deviations', fontweight='bold', pad=20)
        ax3.legend()

# 4. Middle-left: Area chart
ax4 = plt.subplot(3, 3, 4)
months = df['appl_month'].value_counts().head(6)
if 'payments' in df.columns and len(months) > 0:
    baseline_payment = df['payments'].mean()
    payment_deviations = []
    
    for month in months.index:
        month_data = df[df['appl_month'] == month]
        if len(month_data) > 0 and baseline_payment != 0:
            deviation = (month_data['payments'].mean() - baseline_payment) / baseline_payment * 100
            payment_deviations.append(deviation)
        else:
            payment_deviations.append(0)
    
    x_months = range(len(payment_deviations))
    ax4.fill_between(x_months, payment_deviations, alpha=0.6, color='#66c2a5')
    ax4.plot(x_months, payment_deviations, linewidth=2, color='#2c7fb8')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xticks(x_months)
    ax4.set_xticklabels(months.index, rotation=45)
    ax4.set_ylabel('Deviation (%)')
    ax4.set_title('Payment Behavior Over Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)

# 5. Middle-center: Slope chart
ax5 = plt.subplot(3, 3, 5)
sample_size = min(30, len(df))
sample_customers = df.sample(n=sample_size, random_state=42)

if 'risk_score_1' in df.columns:
    x1, x2 = 0, 1
    y1_values = sample_customers['bureau_score'].values
    y2_values = sample_customers['risk_score_1'].fillna(sample_customers['risk_score_1'].mean()).values
    
    # Simple normalization
    y2_normalized = (y2_values - np.min(y2_values)) / (np.max(y2_values) - np.min(y2_values)) * 200 + 600
    
    for i in range(len(y1_values)):
        color = '#d73027' if sample_customers.iloc[i]['default_ind'] == 1 else '#3288bd'
        ax5.plot([x1, x2], [y1_values[i], y2_normalized[i]], 'o-', 
                 color=color, alpha=0.6, linewidth=1)
    
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_xticks([x1, x2])
    ax5.set_xticklabels(['Bureau Score', 'Risk Score'])
    ax5.set_ylabel('Score Value')
    ax5.set_title('Customer Risk Trajectories', fontweight='bold')
    ax5.grid(True, alpha=0.3)

# 6. Middle-right: Lollipop chart
ax6 = plt.subplot(3, 3, 6)
important_vars = ['income', 'limit', 'bureau_score']
available_vars = [var for var in important_vars if var in df.columns]

if available_vars:
    correlations = [abs(df[var].corr(df['default_ind'])) for var in available_vars]
    expected = [0.15, 0.12, 0.20][:len(available_vars)]
    deviations = [actual - exp for actual, exp in zip(correlations, expected)]
    
    y_pos = np.arange(len(available_vars))
    colors = ['#d73027' if x < 0 else '#3288bd' for x in deviations]
    
    ax6.hlines(y_pos, 0, deviations, colors=colors, linewidth=3)
    ax6.scatter(deviations, y_pos, c=colors, s=80, zorder=3)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(available_vars)
    ax6.set_xlabel('Correlation Deviation')
    ax6.set_title('Variable Importance Deviations', fontweight='bold')
    ax6.grid(True, alpha=0.3)

# 7. Bottom-left: Stacked area chart
ax7 = plt.subplot(3, 3, 7)
if 'risk_category' in df.columns:
    risk_composition = df.groupby(['appl_month', 'risk_category']).size().unstack(fill_value=0)
    if not risk_composition.empty:
        risk_composition_pct = risk_composition.div(risk_composition.sum(axis=1), axis=0) * 100
        months_subset = risk_composition_pct.head(4)
        
        x_range = range(len(months_subset))
        colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
        
        bottom = np.zeros(len(months_subset))
        for i, category in enumerate(months_subset.columns):
            if i < len(colors):
                ax7.fill_between(x_range, bottom, bottom + months_subset[category], 
                                alpha=0.7, label=str(category), color=colors[i])
                bottom += months_subset[category]
        
        ax7.set_xticks(x_range)
        ax7.set_xticklabels(months_subset.index, rotation=45)
        ax7.set_ylabel('Percentage (%)')
        ax7.set_title('Risk Category Composition', fontweight='bold')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 8. Bottom-middle: Box plot with violin
ax8 = plt.subplot(3, 3, 8)
df['approval_decision'] = np.where(df['bureau_score'] >= 700, 'Approved', 'Rejected')
approved_income = df[df['approval_decision'] == 'Approved']['income'].dropna()
rejected_income = df[df['approval_decision'] == 'Rejected']['income'].dropna()

if len(approved_income) > 0 and len(rejected_income) > 0:
    violin_data = [approved_income, rejected_income]
    
    # Simple violin plot using fill_between
    for i, data in enumerate(violin_data):
        if len(data) > 10:  # Only if enough data points
            hist, bins = np.histogram(data, bins=20, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            width = np.max(hist) * 0.4
            
            # Normalize histogram for violin shape
            hist_norm = hist / np.max(hist) * width
            
            ax8.fill_betweenx(bin_centers, i + 1 - hist_norm, i + 1 + hist_norm, 
                             alpha=0.6, color='#66c2a5')
    
    # Add box plot
    box_data = ax8.boxplot(violin_data, positions=[1, 2], patch_artist=True)
    for patch in box_data['boxes']:
        patch.set_facecolor('#2c7fb8')
        patch.set_alpha(0.7)
    
    ax8.set_xticks([1, 2])
    ax8.set_xticklabels(['Approved', 'Rejected'])
    ax8.set_ylabel('Income Distribution')
    ax8.set_title('Income by Approval Decision', fontweight='bold')
    ax8.grid(True, alpha=0.3)

# 9. Bottom-right: Correlation heatmap
ax9 = plt.subplot(3, 3, 9)
financial_indicators = ['income', 'limit', 'bureau_score', 'spend']
available_indicators = [col for col in financial_indicators if col in df.columns]

if len(available_indicators) >= 3:
    corr_matrix = df[available_indicators].corr()
    
    # Create expected correlation matrix
    n = len(available_indicators)
    expected_corr = np.eye(n)
    if n >= 2:
        expected_corr[0, 1] = 0.6  # income-limit
        expected_corr[1, 0] = 0.6
    if n >= 3:
        expected_corr[0, 2] = 0.4  # income-bureau_score
        expected_corr[2, 0] = 0.4
    
    deviation_matrix = corr_matrix.values - expected_corr
    
    # Create heatmap
    im = ax9.imshow(deviation_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax9.text(j, i, f'{deviation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    ax9.set_xticks(range(n))
    ax9.set_yticks(range(n))
    ax9.set_xticklabels([col.replace('_', ' ').title() for col in available_indicators], rotation=45)
    ax9.set_yticklabels([col.replace('_', ' ').title() for col in available_indicators])
    ax9.set_title('Correlation Deviations', fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax9, shrink=0.6)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.3)
plt.savefig('deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()