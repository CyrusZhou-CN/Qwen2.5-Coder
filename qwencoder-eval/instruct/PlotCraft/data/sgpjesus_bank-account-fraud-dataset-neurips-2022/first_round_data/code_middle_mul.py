import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load datasets with basic error handling
def load_data_safe(filename):
    try:
        df = pd.read_csv(filename)
        # Sample for performance if dataset is very large
        if len(df) > 100000:
            df = df.sample(n=100000, random_state=42)
        return df
    except:
        return None

# Load all datasets
datasets = {}
filenames = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
dataset_names = ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']

for filename, name in zip(filenames, dataset_names):
    df = load_data_safe(filename)
    if df is not None:
        datasets[name] = df

# Create figure with 2x2 subplot grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4A5D23']

# 1. Top-left: Diverging bar chart - Fraud rate deviations from overall mean
if len(datasets) > 0:
    fraud_rates = {}
    for name, df in datasets.items():
        fraud_rates[name] = df['fraud_bool'].mean()
    
    overall_mean = np.mean(list(fraud_rates.values()))
    deviations = {name: rate - overall_mean for name, rate in fraud_rates.items()}
    
    # Sort by deviation
    sorted_items = sorted(deviations.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    y_pos = np.arange(len(labels))
    bar_colors = ['#C73E1D' if v < 0 else '#2E86AB' for v in values]
    
    ax1.barh(y_pos, values, color=bar_colors, alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontweight='bold')
    ax1.set_xlabel('Deviation from Overall Mean Fraud Rate', fontweight='bold')
    ax1.set_title('Fraud Rate Deviations Across Dataset Variants', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(values):
        ax1.text(val + (0.0001 if val >= 0 else -0.0001), i, f'{val:.4f}', 
                ha='left' if val >= 0 else 'right', va='center', fontweight='bold', fontsize=9)

# 2. Top-right: Dumbbell plot - Credit risk score distributions (Base vs Variant IV)
if 'Base' in datasets and 'Variant IV' in datasets:
    base_df = datasets['Base']
    variant4_df = datasets['Variant IV']
    
    # Create simple income brackets
    income_bins = [0, 0.3, 0.6, 0.9, 1.0]
    income_labels = ['Low', 'Medium', 'High', 'Very High']
    
    base_df['income_bracket'] = pd.cut(base_df['income'], bins=income_bins, labels=income_labels, include_lowest=True)
    variant4_df['income_bracket'] = pd.cut(variant4_df['income'], bins=income_bins, labels=income_labels, include_lowest=True)
    
    base_risk = base_df.groupby('income_bracket')['credit_risk_score'].mean().dropna()
    variant4_risk = variant4_df.groupby('income_bracket')['credit_risk_score'].mean().dropna()
    
    # Get common brackets
    common_brackets = list(set(base_risk.index) & set(variant4_risk.index))
    
    if len(common_brackets) > 0:
        y_pos2 = np.arange(len(common_brackets))
        base_values = [base_risk[bracket] for bracket in common_brackets]
        variant4_values = [variant4_risk[bracket] for bracket in common_brackets]
        
        # Draw connecting lines
        for i in range(len(y_pos2)):
            ax2.plot([base_values[i], variant4_values[i]], [y_pos2[i], y_pos2[i]], 
                    color='gray', alpha=0.6, linewidth=2)
        
        # Draw points
        ax2.scatter(base_values, y_pos2, color='#2E86AB', s=100, label='Base Dataset', zorder=3)
        ax2.scatter(variant4_values, y_pos2, color='#F18F01', s=100, label='Variant IV', zorder=3)
        
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(common_brackets, fontweight='bold')
        ax2.set_xlabel('Mean Credit Risk Score', fontweight='bold')
        ax2.set_title('Credit Risk Score Comparison: Base vs Variant IV\nby Income Bracket', fontweight='bold', fontsize=12)
        ax2.legend(fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No common income brackets found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Credit Risk Score Comparison: Base vs Variant IV', fontweight='bold', fontsize=12)
else:
    ax2.text(0.5, 0.5, 'Required datasets not available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Credit Risk Score Comparison: Base vs Variant IV', fontweight='bold', fontsize=12)

# 3. Bottom-left: Error bar chart - Session length with confidence intervals
if len(datasets) > 0:
    session_stats = {}
    
    for name, df in datasets.items():
        session_data = df['session_length_in_minutes'].dropna()
        if len(session_data) > 10:  # Ensure sufficient data
            mean_val = session_data.mean()
            std_val = session_data.std()
            n = len(session_data)
            ci = 1.96 * (std_val / np.sqrt(n))  # 95% confidence interval
            session_stats[name] = {'mean': mean_val, 'ci': ci}
    
    if len(session_stats) > 0:
        x_pos3 = np.arange(len(session_stats))
        means = [stats['mean'] for stats in session_stats.values()]
        cis = [stats['ci'] for stats in session_stats.values()]
        labels3 = list(session_stats.keys())
        
        ax3.bar(x_pos3, means, yerr=cis, capsize=5, color=colors[:len(means)], alpha=0.8)
        ax3.set_xticks(x_pos3)
        ax3.set_xticklabels(labels3, rotation=45, ha='right', fontweight='bold')
        ax3.set_ylabel('Mean Session Length (minutes)', fontweight='bold')
        ax3.set_title('Session Length Across Variants\nwith 95% Confidence Intervals', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, mean_val in enumerate(means):
            ax3.text(i, mean_val + cis[i] + max(means) * 0.02, f'{mean_val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Insufficient session data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Session Length Across Variants', fontweight='bold', fontsize=12)

# 4. Bottom-right: Slope chart - Employment status fraud risk (Base to Variant III)
if 'Base' in datasets and 'Variant III' in datasets:
    base_df = datasets['Base']
    variant3_df = datasets['Variant III']
    
    base_emp_fraud = base_df.groupby('employment_status')['fraud_bool'].mean()
    variant3_emp_fraud = variant3_df.groupby('employment_status')['fraud_bool'].mean()
    
    # Get common employment statuses
    common_emp = list(set(base_emp_fraud.index) & set(variant3_emp_fraud.index))
    
    if len(common_emp) > 0:
        # Limit to top 5 for readability
        common_emp = sorted(common_emp)[:5]
        
        base_values = [base_emp_fraud[emp] for emp in common_emp]
        variant3_values = [variant3_emp_fraud[emp] for emp in common_emp]
        
        # Create y positions
        y_positions = np.linspace(0.1, 0.9, len(common_emp))
        
        # Draw slope lines
        for i in range(len(common_emp)):
            color = '#C73E1D' if variant3_values[i] > base_values[i] else '#2E86AB'
            ax4.plot([0, 1], [y_positions[i], y_positions[i]], 
                    color=color, linewidth=2, alpha=0.7)
        
        # Draw points
        ax4.scatter([0] * len(common_emp), y_positions, color='#2E86AB', s=80, zorder=3)
        ax4.scatter([1] * len(common_emp), y_positions, color='#F18F01', s=80, zorder=3)
        
        # Add labels
        for i, emp in enumerate(common_emp):
            # Employment status labels (truncated for space)
            emp_short = emp[:6] + '...' if len(emp) > 6 else emp
            ax4.text(-0.05, y_positions[i], emp_short, ha='right', va='center', fontweight='bold', fontsize=9)
            # Values
            ax4.text(0.05, y_positions[i], f'{base_values[i]:.3f}', ha='left', va='center', fontsize=8)
            ax4.text(0.95, y_positions[i], f'{variant3_values[i]:.3f}', ha='right', va='center', fontsize=8)
        
        ax4.set_xlim(-0.3, 1.3)
        ax4.set_ylim(0, 1)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Base Dataset', 'Variant III'], fontweight='bold')
        ax4.set_title('Employment Status Fraud Risk Changes\nBase → Variant III', fontweight='bold', fontsize=12)
        ax4.set_yticks([])
        
        # Add simple legend
        ax4.text(0.5, 0.05, 'Blue: Decreased Risk, Red: Increased Risk', 
                ha='center', va='bottom', transform=ax4.transAxes, fontsize=9, style='italic')
    else:
        ax4.text(0.5, 0.5, 'No common employment statuses', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Employment Status Fraud Risk Changes', fontweight='bold', fontsize=12)
else:
    ax4.text(0.5, 0.5, 'Required datasets not available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Employment Status Fraud Risk Changes', fontweight='bold', fontsize=12)

# Clean up spines
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig('bias_patterns_analysis.png', dpi=300, bbox_inches='tight')
plt.show()