import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import warnings
import os
import glob
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# Find CSV files in the current directory
csv_files = glob.glob('*.csv')
print(f"Found CSV files: {csv_files}")

# Load and combine data from available CSV files
all_data = []
companies = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Extract company name from filename (remove .csv extension)
        company_name = os.path.splitext(file)[0]
        df['Company'] = company_name
        all_data.append(df)
        companies.append(company_name)
        print(f"Loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

if not all_data:
    print("No CSV files could be loaded. Creating sample data for demonstration.")
    # Create sample data if no files are found
    np.random.seed(42)
    sample_data = []
    for i, company in enumerate(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']):
        n_rows = np.random.randint(20, 50)
        df = pd.DataFrame({
            'Insider Trading': [f'Person_{j}' for j in range(n_rows)],
            'Relationship': np.random.choice(['CEO', 'CFO', 'Director', 'VP'], n_rows),
            'Date': pd.date_range('2022-01-01', periods=n_rows, freq='D'),
            'Transaction': np.random.choice(['Sale', 'Buy', 'Option Exercise'], n_rows, p=[0.6, 0.2, 0.2]),
            'Cost': np.random.uniform(50, 400, n_rows),
            'Shares': [f"{np.random.randint(1000, 50000):,}" for _ in range(n_rows)],
            'Value ($)': [f"{np.random.randint(100000, 5000000):,}" for _ in range(n_rows)],
            'Shares Total': [f"{np.random.randint(10000, 500000):,}" for _ in range(n_rows)],
            'SEC Form 4': ['Form 4'] * n_rows,
            'Company': company
        })
        sample_data.append(df)
    all_data = sample_data
    companies = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

combined_df = pd.concat(all_data, ignore_index=True)

# Data preprocessing
def clean_numeric_column(col):
    if col.dtype == 'object':
        # Handle string columns with commas and dollar signs
        cleaned = col.astype(str).str.replace(',', '').str.replace('$', '').str.replace(' ', '')
        return pd.to_numeric(cleaned, errors='coerce')
    return col

combined_df['Value_numeric'] = clean_numeric_column(combined_df['Value ($)'])
combined_df['Shares_numeric'] = clean_numeric_column(combined_df['Shares'])
combined_df['Shares_Total_numeric'] = clean_numeric_column(combined_df['Shares Total'])

# Remove rows with missing critical data
combined_df = combined_df.dropna(subset=['Cost', 'Value_numeric', 'Shares_numeric'])

# Ensure we have valid data
if len(combined_df) == 0:
    raise ValueError("No valid data found after preprocessing")

print(f"Final dataset shape: {combined_df.shape}")
print(f"Companies included: {combined_df['Company'].unique()}")

# Define consistent color palettes for the entire dashboard
transaction_colors = {'Sale': '#E74C3C', 'Buy': '#27AE60', 'Option Exercise': '#3498DB'}
relationship_colors = {'CEO': '#E74C3C', 'CFO': '#3498DB', 'Director': '#27AE60', 
                      'VP': '#F39C12', 'COO': '#9B59B6', 'Senior Vice President': '#1ABC9C',
                      'SVP, GC and Secretary': '#E67E22', 'Principal Accounting Officer': '#8E44AD'}

# Get top 10 companies by transaction count
top_companies = combined_df['Company'].value_counts().head(10).index
company_colors = {comp: plt.cm.Set3(i) for i, comp in enumerate(top_companies)}

# Create figure with white background and improved spacing
plt.style.use('default')
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Add main title for the entire dashboard
fig.suptitle('Comprehensive Insider Trading Correlation Analysis Dashboard', 
             fontsize=24, fontweight='bold', y=0.98)

# Create 3x2 grid with reduced spacing
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25, 
              left=0.05, right=0.95, top=0.93, bottom=0.05)

# 1. Top-left: Scatter plot with marginal histograms
ax1_main = fig.add_subplot(gs[0, 0])
ax1_main.set_facecolor('white')

# Create marginal histogram axes
divider = make_axes_locatable(ax1_main)
ax1_top = divider.append_axes("top", size="25%", pad=0.1)
ax1_right = divider.append_axes("right", size="25%", pad=0.1)

# Main scatter plot with point sizing by number of shares
for trans_type in combined_df['Transaction'].unique():
    if trans_type in transaction_colors:
        mask = combined_df['Transaction'] == trans_type
        data = combined_df[mask]
        if len(data) > 0:
            # Size points by number of shares (normalized)
            sizes = np.clip((data['Shares_numeric'] - data['Shares_numeric'].min()) / 
                          (data['Shares_numeric'].max() - data['Shares_numeric'].min()) * 200 + 20, 20, 300)
            ax1_main.scatter(data['Cost'], data['Value_numeric'], 
                           c=transaction_colors[trans_type], alpha=0.7, s=sizes,
                           label=trans_type, edgecolors='white', linewidth=0.5)

# Top marginal histogram (Cost distribution)
for trans_type in combined_df['Transaction'].unique():
    if trans_type in transaction_colors:
        data = combined_df[combined_df['Transaction'] == trans_type]['Cost']
        if len(data) > 0:
            ax1_top.hist(data, bins=15, alpha=0.7, color=transaction_colors[trans_type], 
                        density=True, edgecolor='white', linewidth=0.5)

# Right marginal histogram (Value distribution)
for trans_type in combined_df['Transaction'].unique():
    if trans_type in transaction_colors:
        data = combined_df[combined_df['Transaction'] == trans_type]['Value_numeric']
        if len(data) > 0:
            ax1_right.hist(data, bins=15, alpha=0.7, color=transaction_colors[trans_type], 
                          density=True, orientation='horizontal', edgecolor='white', linewidth=0.5)

# Styling for main plot
ax1_main.set_xlabel('Transaction Cost ($)', fontweight='bold', fontsize=12)
ax1_main.set_ylabel('Transaction Value ($)', fontweight='bold', fontsize=12)
ax1_main.set_title('Transaction Cost vs Value with Marginal Distributions\n(Point Size = Number of Shares)', 
                   fontweight='bold', fontsize=14, pad=15)
ax1_main.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax1_main.grid(True, alpha=0.3)

# Remove ticks from marginal plots
ax1_top.set_xticks([])
ax1_top.set_ylabel('Density', fontweight='bold', fontsize=10)
ax1_right.set_yticks([])
ax1_right.set_xlabel('Density', fontweight='bold', fontsize=10)

# 2. Top-right: Correlation heatmap with network graph overlay
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')

# Calculate correlation matrix
numeric_cols = ['Cost', 'Value_numeric', 'Shares_Total_numeric']
corr_data = combined_df[numeric_cols].corr()

# Create heatmap
im = ax2.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add text annotations
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax2.text(j, i, f'{corr_data.iloc[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=12)

# Add network graph overlay
G = nx.Graph()
node_labels = ['Cost', 'Value', 'Shares']
G.add_nodes_from(range(3))

# Add edges for strong correlations and draw network
pos = {0: (0, 0), 1: (1, 1), 2: (2, 2)}  # Diagonal positions
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr_val = abs(corr_data.iloc[i, j])
        if corr_val > 0.2:  # Show moderate to strong correlations
            G.add_edge(i, j, weight=corr_val)
            # Draw connection line
            ax2.plot([j, i], [i, j], 'yellow', alpha=0.8, linewidth=corr_val*8, zorder=10)

# Draw network nodes
for i in range(3):
    circle = Circle((i, i), 0.15, color='yellow', alpha=0.8, zorder=11)
    ax2.add_patch(circle)

ax2.set_xticks(range(len(numeric_cols)))
ax2.set_yticks(range(len(numeric_cols)))
ax2.set_xticklabels(['Cost', 'Value', 'Total Shares'], fontweight='bold', fontsize=12)
ax2.set_yticklabels(['Cost', 'Value', 'Total Shares'], fontweight='bold', fontsize=12)
ax2.set_title('Correlation Matrix with Network Overlay\n(Yellow Lines = Strong Correlations)', 
              fontweight='bold', fontsize=14, pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=12)

# 3. Middle-left: Bubble plot with trend lines by relationship type
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('white')

# Aggregate data by company and relationship
company_stats = combined_df.groupby(['Company', 'Relationship']).agg({
    'Cost': 'mean',
    'Value_numeric': 'sum',
    'Transaction': 'count'
}).reset_index()

# Create true bubble plot with bubble sizes representing total transaction value
for relationship in company_stats['Relationship'].unique():
    if relationship in relationship_colors:
        group = company_stats[company_stats['Relationship'] == relationship]
        if len(group) > 0:
            x = group['Cost']
            y = group['Transaction']
            # Bubble sizes based on total transaction value
            bubble_sizes = np.clip(np.sqrt(group['Value_numeric']) / 2000, 50, 800)
            ax3.scatter(x, y, s=bubble_sizes, alpha=0.7, c=relationship_colors[relationship], 
                       label=relationship, edgecolors='white', linewidth=1)
            
            # Add trend line if we have enough points
            if len(x) > 1:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x.min(), x.max(), 100)
                    ax3.plot(x_trend, p(x_trend), color=relationship_colors[relationship], 
                            linestyle='--', alpha=0.8, linewidth=2)
                except:
                    pass

ax3.set_xlabel('Average Transaction Cost ($)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Number of Transactions', fontweight='bold', fontsize=12)
ax3.set_title('Transaction Activity by Relationship Type\n(Bubble Size = Total Transaction Value)', 
              fontweight='bold', fontsize=14, pad=15)
ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# Add bubble size legend
bubble_legend_sizes = [100000, 1000000, 10000000]
bubble_legend_labels = ['$100K', '$1M', '$10M']
for i, (size, label) in enumerate(zip(bubble_legend_sizes, bubble_legend_labels)):
    bubble_size = np.clip(np.sqrt(size) / 2000, 50, 800)
    ax3.scatter([], [], s=bubble_size, c='gray', alpha=0.6, label=label)

# 4. Middle-right: Scatter plot matrix (pair plot) for top companies
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('white')

# Get top companies data
top_data = combined_df[combined_df['Company'].isin(top_companies)]

# Create pair plot focusing on Cost vs Value (main relationship in scatter plot matrix)
for company in top_companies[:5]:  # Limit to top 5 for clarity
    if company in company_colors:
        company_data = top_data[top_data['Company'] == company]
        if len(company_data) > 0:
            ax4.scatter(company_data['Cost'], company_data['Value_numeric'], 
                       c=[company_colors[company]], alpha=0.7, label=company,
                       edgecolors='white', linewidth=0.5, s=60)

# Add regression lines for each company
for company in top_companies[:5]:
    if company in company_colors:
        company_data = top_data[top_data['Company'] == company]
        if len(company_data) > 1:
            try:
                z = np.polyfit(company_data['Cost'], company_data['Value_numeric'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(company_data['Cost'].min(), company_data['Cost'].max(), 100)
                ax4.plot(x_trend, p(x_trend), color=company_colors[company], 
                        linestyle='-', alpha=0.8, linewidth=2)
            except:
                pass

ax4.set_xlabel('Transaction Cost ($)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Transaction Value ($)', fontweight='bold', fontsize=12)
ax4.set_title('Scatter Plot Matrix: Cost vs Value\n(Top 5 Most Active Companies)', 
              fontweight='bold', fontsize=14, pad=15)
ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Bottom-left: Jitter plot with box plots and violin plots
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor('white')

# Prepare data for combined plot
transaction_types = list(combined_df['Transaction'].unique())
box_data = []
positions = []
labels = []

for i, trans_type in enumerate(transaction_types):
    data = combined_df[combined_df['Transaction'] == trans_type]['Cost'].dropna()
    if len(data) > 0:
        box_data.append(data)
        positions.append(i)
        labels.append(trans_type)

if box_data:
    # Create violin plots first (background)
    violin_parts = ax5.violinplot(box_data, positions=positions, widths=0.8, 
                                 showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violin plots
    colors = [transaction_colors.get(label, '#95a5a6') for label in labels]
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    
    # Create box plot overlay
    bp = ax5.boxplot(box_data, positions=positions, patch_artist=True, 
                    widths=0.4, showfliers=False)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    # Style box plot elements
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.5)

    # Add jitter points
    for i, (pos, trans_type) in enumerate(zip(positions, labels)):
        data = combined_df[combined_df['Transaction'] == trans_type]['Cost'].dropna()
        if len(data) > 0:
            # Add jitter
            x_jitter = np.random.normal(pos, 0.08, len(data))
            color = transaction_colors.get(trans_type, '#95a5a6')
            ax5.scatter(x_jitter, data, alpha=0.6, s=15, 
                       c=color, edgecolors='white', linewidth=0.3)

    ax5.set_xticks(positions)
    ax5.set_xticklabels(labels, fontweight='bold', fontsize=12)

ax5.set_ylabel('Transaction Cost ($)', fontweight='bold', fontsize=12)
ax5.set_title('Distribution of Transaction Costs by Type\n(Jitter + Box + Violin Plots)', 
              fontweight='bold', fontsize=14, pad=15)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Bottom-right: Multiple 2D histograms by relationship category
# Create subplots within the main subplot for different relationship categories
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor('white')

# Get top relationship categories
top_relationships = combined_df['Relationship'].value_counts().head(4).index
valid_data = combined_df[(combined_df['Value_numeric'] > 0) & (combined_df['Cost'] > 0)]

# Create a 2x2 grid within this subplot
gs_inner = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

for idx, relationship in enumerate(top_relationships):
    row = idx // 2
    col = idx % 2
    
    # Calculate position within the main subplot
    ax_sub = plt.subplot(gs_inner[row, col])
    ax_sub.set_facecolor('white')
    
    rel_data = valid_data[valid_data['Relationship'] == relationship]
    
    if len(rel_data) > 5:
        log_values = np.log10(rel_data['Value_numeric'])
        log_costs = np.log10(rel_data['Cost'])
        
        # Create hexbin plot
        hb = ax_sub.hexbin(log_costs, log_values, gridsize=10, cmap='YlOrRd', alpha=0.8)
        
        # Add contour lines
        try:
            from scipy.stats import gaussian_kde
            if len(log_costs) > 5:
                # Create grid for contour
                x_min, x_max = log_costs.min(), log_costs.max()
                y_min, y_max = log_values.min(), log_values.max()
                
                if x_max > x_min and y_max > y_min:
                    xx, yy = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    
                    values = np.vstack([log_costs, log_values])
                    kernel = gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    
                    contours = ax_sub.contour(xx, yy, f, colors='black', alpha=0.7, 
                                            linewidths=1, levels=3)
        except:
            pass
        
        ax_sub.set_xlabel('Log10(Cost)', fontweight='bold', fontsize=10)
        ax_sub.set_ylabel('Log10(Value)', fontweight='bold', fontsize=10)
        ax_sub.set_title(f'{relationship}\n({len(rel_data)} transactions)', 
                        fontweight='bold', fontsize=11)
    else:
        ax_sub.text(0.5, 0.5, f'{relationship}\nInsufficient data', 
                   ha='center', va='center', transform=ax_sub.transAxes, 
                   fontsize=10, fontweight='bold')
        ax_sub.set_title(f'{relationship}', fontweight='bold', fontsize=11)

# Hide the main ax6 since we're using subplots
ax6.set_xticks([])
ax6.set_yticks([])
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['left'].set_visible(False)

# Final layout adjustments
plt.tight_layout(pad=2.0)

# Save the plot
plt.savefig('insider_trading_correlation_dashboard_refined.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()