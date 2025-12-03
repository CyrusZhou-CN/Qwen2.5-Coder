import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('defi_dataset.csv')

# Get date columns and convert to datetime
date_cols = [col for col in df.columns if '/' in col and len(col.split('/')) == 3]
dates = pd.to_datetime(date_cols, format='%d/%m/%Y')

# Helper function to safely prepare time series data
def prepare_time_series(protocol_name, category_filter=None):
    try:
        if category_filter:
            mask = (df['Name'] == protocol_name) & (df['Category'] == category_filter)
        else:
            mask = df['Name'] == protocol_name
        
        protocol_data = df[mask]
        if len(protocol_data) == 0:
            return dates, np.zeros(len(dates))
        
        # Get the first matching row and extract values
        values = protocol_data[date_cols].iloc[0].values
        # Replace NaN with 0 and ensure numeric
        values = pd.to_numeric(values, errors='coerce')
        values = np.where(np.isnan(values), 0, values)
        
        return dates, values
    except Exception as e:
        print(f"Error processing {protocol_name}: {e}")
        return dates, np.zeros(len(dates))

# Helper function to safely calculate statistics
def safe_stats(data):
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    if len(data) == 0:
        return 0, 0, 0  # mean, std, length
    return np.mean(data), np.std(data), len(data)

# Define time periods
early_period = (dates >= '2019-01-01') & (dates <= '2020-12-31')
boom_period = (dates >= '2020-01-01') & (dates <= '2021-12-31')
mature_period = (dates >= '2021-01-01') & (dates <= '2022-12-31')

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('white')

# Color schemes for different periods
early_colors = ['#2E86AB', '#A23B72', '#F18F01']
boom_colors = ['#C73E1D', '#F18F01', '#FFB627']
mature_colors = ['#6A994E', '#A7C957', '#F2E8CF']

# Subplot 1: Early DeFi Era - TVL growth with milestones and volatility
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Get WBTC data for early period
dates_early, wbtc_values = prepare_time_series('WBTC')
early_mask = early_period
dates_plot = dates[early_mask]
values_plot = wbtc_values[early_mask]

# Ensure we have valid data
if len(values_plot) > 0 and np.any(values_plot > 0):
    # Line chart with scatter points for milestones
    ax1.plot(dates_plot, values_plot / 1e9, color=early_colors[0], linewidth=2.5, label='WBTC TVL')
    
    # Find milestones safely
    diff_values = np.diff(values_plot)
    diff_values = diff_values[~np.isnan(diff_values)]
    if len(diff_values) > 0:
        threshold = np.std(diff_values) * 2
        milestone_indices = np.where(np.abs(np.diff(values_plot)) > threshold)[0]
        if len(milestone_indices) > 0:
            ax1.scatter(dates_plot[milestone_indices], values_plot[milestone_indices] / 1e9, 
                       color=early_colors[1], s=80, zorder=5, label='Major Milestones')

    # Secondary axis for volatility
    ax1_twin = ax1.twinx()
    volatility = np.abs(np.diff(values_plot))
    volatility = np.concatenate([[0], volatility])
    volatility = volatility[~np.isnan(volatility)]
    
    if len(volatility) == len(dates_plot):
        ax1_twin.bar(dates_plot, volatility / 1e8, alpha=0.3, color=early_colors[2], 
                    width=20, label='Daily Volatility')
        ax1_twin.set_ylabel('Volatility (×100M USD)', fontweight='bold')
        ax1_twin.legend(loc='upper right')

ax1.set_title('**Early DeFi Era (2019-2020): TVL Growth & Volatility**', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked area chart with uncertainty bands
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Prepare multiple protocol data
protocols_data = {}
available_protocols = ['WBTC', 'Harvest Finance']
for protocol in available_protocols:
    dates_p, values_p = prepare_time_series(protocol)
    if len(values_p) > 0:
        protocols_data[protocol] = values_p[early_mask]

if protocols_data:
    # Create stacked area chart
    bottom = np.zeros(len(dates_plot))
    colors_stack = early_colors[:len(protocols_data)]
    
    for i, (protocol, values) in enumerate(protocols_data.items()):
        values_clean = np.where(np.isnan(values), 0, values)
        ax2.fill_between(dates_plot, bottom, bottom + values_clean / 1e9, 
                        color=colors_stack[i % len(colors_stack)], alpha=0.7, label=protocol)
        bottom += values_clean / 1e9

    # Total market line with uncertainty bands
    total_tvl = sum(np.where(np.isnan(v), 0, v) for v in protocols_data.values()) / 1e9
    ax2.plot(dates_plot, total_tvl, color='black', linewidth=3, label='Total TVL')

    # Add uncertainty bands
    uncertainty = total_tvl * 0.1  # 10% uncertainty
    ax2.fill_between(dates_plot, total_tvl - uncertainty, total_tvl + uncertainty, 
                    color='gray', alpha=0.2, label='Uncertainty Range')

ax2.set_title('**TVL by Category with Market Total**', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Multi-line chart with filled areas and histogram
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Chain dominance using available data
chain_data = {}
unique_chains = df['Chain'].dropna().unique()[:2]  # Get first 2 chains

for i, chain in enumerate(unique_chains):
    chain_mask = df['Chain'] == chain
    if chain_mask.any():
        chain_values = df[chain_mask][date_cols].sum(axis=0, skipna=True).values
        chain_values = np.where(np.isnan(chain_values), 0, chain_values)
        chain_data[chain] = chain_values[early_mask]

if len(chain_data) >= 2:
    chain_items = list(chain_data.items())
    for i, (chain, values) in enumerate(chain_items):
        ax3.plot(dates_plot, values / 1e9, color=early_colors[i], 
                linewidth=2, label=f'{chain.title()} Chain')
        
        if i > 0:
            prev_values = chain_items[i-1][1] / 1e9
            ax3.fill_between(dates_plot, prev_values, values / 1e9, 
                           alpha=0.3, color=early_colors[i])

    # Histogram overlay for daily changes (with proper NaN handling)
    ax3_hist = ax3.twinx()
    first_chain_values = list(chain_data.values())[0]
    daily_changes = np.diff(first_chain_values)
    daily_changes = daily_changes[~np.isnan(daily_changes)]
    
    if len(daily_changes) > 0 and np.std(daily_changes) > 0:
        try:
            ax3_hist.hist(daily_changes / 1e8, bins=min(20, len(daily_changes)//2), 
                         alpha=0.4, color=early_colors[2], 
                         orientation='horizontal', density=True)
            ax3_hist.set_xlabel('Distribution Density', fontweight='bold')
        except:
            pass  # Skip histogram if it fails

ax3.set_title('**Chain Dominance with Change Distribution**', fontsize=14, fontweight='bold', pad=20)
ax3.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: DeFi Boom - Dual-axis with candlestick simulation
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

boom_mask = boom_period
dates_boom = dates[boom_mask]
wbtc_boom = wbtc_values[boom_mask]

# TVL line chart
if len(wbtc_boom) > 0:
    ax4.plot(dates_boom, wbtc_boom / 1e9, color=boom_colors[0], linewidth=3, label='WBTC TVL')

    # Simulated price movements as candlestick-style bars
    ax4_price = ax4.twinx()
    np.random.seed(42)  # For reproducible results
    price_sim = 30000 + np.cumsum(np.random.randn(len(dates_boom)) * 1000)
    price_changes = np.diff(price_sim)
    price_changes = np.concatenate([[0], price_changes])

    colors_candle = ['green' if x >= 0 else 'red' for x in price_changes]
    ax4_price.bar(dates_boom, np.abs(price_changes), color=colors_candle, 
                 alpha=0.6, width=10, label='Price Movements')
    ax4_price.set_ylabel('Price Change (USD)', fontweight='bold')
    ax4_price.legend(loc='upper right')

ax4.set_title('**DeFi Boom (2020-2021): TVL vs Token Prices**', fontsize=14, fontweight='bold', pad=20)
ax4.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# Subplot 5: Time series decomposition
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

if len(wbtc_boom) > 30:  # Need enough data for moving average
    # Trend component (line)
    window_size = min(30, len(wbtc_boom)//3)
    trend = np.convolve(wbtc_boom, np.ones(window_size)/window_size, mode='same')
    ax5.plot(dates_boom, trend / 1e9, color=boom_colors[0], linewidth=3, label='Trend')

    # Seasonal component (bars)
    seasonal = wbtc_boom - trend
    seasonal = np.where(np.isnan(seasonal), 0, seasonal)
    ax5.bar(dates_boom, seasonal / 1e8, alpha=0.5, color=boom_colors[1], 
           width=5, label='Seasonal')

    # Residuals (scatter)
    residuals = wbtc_boom - trend - seasonal
    residuals = np.where(np.isnan(residuals), 0, residuals)
    step = max(1, len(dates_boom)//20)  # Sample points for scatter
    ax5.scatter(dates_boom[::step], residuals[::step] / 1e8, color=boom_colors[2], 
               alpha=0.7, s=30, label='Residuals')

ax5.set_title('**Time Series Decomposition**', fontsize=14, fontweight='bold', pad=20)
ax5.set_ylabel('TVL Components', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Cross-correlation heatmap
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Create correlation matrix
protocols_boom = {}
for protocol in available_protocols:
    dates_p, values_p = prepare_time_series(protocol)
    if len(values_p) > 0:
        boom_values = values_p[boom_mask]
        boom_values = np.where(np.isnan(boom_values), 0, boom_values)
        protocols_boom[protocol] = boom_values

if len(protocols_boom) >= 2:
    corr_data = np.array(list(protocols_boom.values()))
    # Handle case where all values might be the same
    correlation_matrix = np.corrcoef(corr_data)
    correlation_matrix = np.where(np.isnan(correlation_matrix), 0, correlation_matrix)
    
    # Create heatmap
    im = ax6.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add correlation values
    protocol_names = list(protocols_boom.keys())
    for i in range(len(protocol_names)):
        for j in range(len(protocol_names)):
            ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    ax6.set_xticks(range(len(protocol_names)))
    ax6.set_yticks(range(len(protocol_names)))
    ax6.set_xticklabels(protocol_names, rotation=45)
    ax6.set_yticklabels(protocol_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.set_label('Correlation Strength', fontweight='bold')

ax6.set_title('**Protocol Cross-Correlation**', fontsize=14, fontweight='bold', pad=20)

# Subplot 7: Market Maturation - Parallel coordinates
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

mature_mask = mature_period
dates_mature = dates[mature_mask]

# Create parallel coordinates data
metrics = ['TVL', 'Growth', 'Volatility']
protocols_mature = {}

for i, protocol in enumerate(available_protocols):
    dates_p, values_p = prepare_time_series(protocol)
    if len(values_p) > 0:
        values_mature = values_p[mature_mask]
        values_mature = np.where(np.isnan(values_mature), 0, values_mature)
        
        tvl_avg, volatility, _ = safe_stats(values_mature)
        
        # Calculate growth rate safely
        if len(values_mature) > 1 and values_mature[0] > 0:
            growth_rate = (values_mature[-1] - values_mature[0]) / values_mature[0] * 100
        else:
            growth_rate = 0
        
        protocols_mature[protocol] = [tvl_avg / 1e9, growth_rate, volatility / 1e8]

# Plot parallel coordinates
if protocols_mature:
    for i, (protocol, values) in enumerate(protocols_mature.items()):
        color_idx = i % len(mature_colors)
        ax7.plot(range(len(metrics)), values, 'o-', color=mature_colors[color_idx], 
                linewidth=3, markersize=8, label=protocol)
        
        # Add trend arrows
        for j in range(len(metrics) - 1):
            if values[j+1] > values[j]:
                ax7.annotate('↗', xy=(j+0.5, (values[j] + values[j+1])/2), 
                            fontsize=16, ha='center', color=mature_colors[color_idx])
            else:
                ax7.annotate('↘', xy=(j+0.5, (values[j] + values[j+1])/2), 
                            fontsize=16, ha='center', color=mature_colors[color_idx])

ax7.set_xticks(range(len(metrics)))
ax7.set_xticklabels(metrics, fontweight='bold')
ax7.set_title('**Market Maturation (2021-2022): Multi-Protocol Comparison**', 
             fontsize=14, fontweight='bold', pad=20)
ax7.set_ylabel('Normalized Metrics', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Subplot 8: Bubble chart with trails
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Create bubble chart data
for i, protocol in enumerate(available_protocols):
    dates_p, values_p = prepare_time_series(protocol)
    if len(values_p) > 0:
        values_mature = values_p[mature_mask]
        values_mature = np.where(np.isnan(values_mature), 0, values_mature)
        
        if len(values_mature) > 1:
            # Calculate growth rate and size safely
            growth_rates = []
            sizes = []
            
            for j in range(len(values_mature) - 1):
                if values_mature[j] > 0:
                    growth_rate = (values_mature[j+1] - values_mature[j]) / values_mature[j] * 100
                    growth_rates.append(growth_rate)
                    sizes.append(values_mature[j] / 1e8)
            
            if len(growth_rates) > 0:
                # Plot bubbles with time-based color gradient
                scatter = ax8.scatter(growth_rates, sizes, s=[s*2 + 50 for s in sizes], 
                                    c=range(len(growth_rates)), cmap='viridis', 
                                    alpha=0.6, label=protocol)
                
                # Add trail lines
                color_idx = i % len(mature_colors)
                ax8.plot(growth_rates, sizes, color=mature_colors[color_idx], 
                        alpha=0.5, linewidth=2)

ax8.set_title('**Protocol Size vs Growth Rate Evolution**', fontsize=14, fontweight='bold', pad=20)
ax8.set_xlabel('Growth Rate (%)', fontweight='bold')
ax8.set_ylabel('Protocol Size (×100M USD)', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Subplot 9: Risk-return scatter with density contours
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Calculate risk-return metrics
risk_return_data = {}
for protocol in available_protocols:
    dates_p, values_p = prepare_time_series(protocol)
    if len(values_p) > 0:
        values_mature = values_p[mature_mask]
        values_mature = np.where(np.isnan(values_mature), 0, values_mature)
        
        if len(values_mature) > 1:
            returns = []
            for j in range(len(values_mature) - 1):
                if values_mature[j] > 0:
                    ret = (values_mature[j+1] - values_mature[j]) / values_mature[j]
                    returns.append(ret)
            
            if len(returns) > 0:
                avg_return, risk, _ = safe_stats(returns)
                risk_return_data[protocol] = (risk * 100, avg_return * 100)

# Plot risk-return scatter
if risk_return_data:
    risks = [data[0] for data in risk_return_data.values()]
    returns = [data[1] for data in risk_return_data.values()]
    
    scatter = ax9.scatter(risks, returns, s=200, c=mature_colors[:len(risk_return_data)], 
                         alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add protocol labels
    for i, (protocol, (risk, ret)) in enumerate(risk_return_data.items()):
        ax9.annotate(protocol, (risk, ret), xytext=(10, 10), 
                    textcoords='offset points', fontweight='bold')
    
    # Simple contour approximation if we have multiple points
    if len(risks) >= 2:
        try:
            x_center, y_center = np.mean(risks), np.mean(returns)
            x_std, y_std = np.std(risks), np.std(returns)
            
            # Create simple elliptical contours
            theta = np.linspace(0, 2*np.pi, 100)
            for scale in [1, 2]:
                x_contour = x_center + scale * x_std * np.cos(theta)
                y_contour = y_center + scale * y_std * np.sin(theta)
                ax9.plot(x_contour, y_contour, alpha=0.3, color='gray', linestyle='--')
        except:
            pass

ax9.set_title('**Risk-Return Analysis with Density Contours**', fontsize=14, fontweight='bold', pad=20)
ax9.set_xlabel('Risk (Volatility %)', fontweight='bold')
ax9.set_ylabel('Average Return (%)', fontweight='bold')
ax9.grid(True, alpha=0.3)

# Overall layout adjustments
plt.tight_layout(pad=3.0)

# Add overall title
fig.suptitle('**Comprehensive DeFi Protocol Evolution Analysis (2019-2022)**', 
            fontsize=20, fontweight='bold', y=0.98)

plt.savefig('defi_evolution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()