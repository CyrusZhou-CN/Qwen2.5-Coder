import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import os

# Load and process all CSV files
def load_and_process_data():
    # Get all CSV files
    csv_files = glob.glob('*.csv')
    
    # Extract dates from filenames and sort
    file_data = []
    for file in csv_files:
        if file.startswith('aug'):
            # Skip August files due to inconsistent format
            continue
        try:
            # Extract date from filename
            if file.startswith('july'):
                day = file.replace('july', '').replace('-2020.csv', '').replace('.csv', '')
                date_str = f"2020-07-{day.zfill(2)}"
            else:
                # Handle other date formats
                date_part = file.replace('.csv', '')
                if '-' in date_part:
                    parts = date_part.split('-')
                    if len(parts) == 3:
                        day, month, year = parts
                        date_str = f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        continue
                else:
                    continue
            
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            file_data.append((date_obj, file))
        except:
            continue
    
    # Sort by date
    file_data.sort(key=lambda x: x[0])
    
    # Process each file
    all_data = []
    for date_obj, file in file_data:
        try:
            df = pd.read_csv(file)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Find the relevant columns
            state_col = None
            confirmed_col = None
            deaths_col = None
            recovered_col = None
            active_col = None
            
            for col in df.columns:
                if 'State' in col or 'UT' in col:
                    state_col = col
                elif 'Total Confirmed' in col or 'Total confirmed' in col:
                    confirmed_col = col
                elif 'Death' in col or 'Deaths' in col:
                    deaths_col = col
                elif 'Cured' in col or 'Discharged' in col or 'Migrated' in col:
                    recovered_col = col
                elif 'Active' in col:
                    active_col = col
            
            if state_col and confirmed_col:
                # Extract relevant data
                cols_to_use = [state_col, confirmed_col]
                if deaths_col:
                    cols_to_use.append(deaths_col)
                if recovered_col:
                    cols_to_use.append(recovered_col)
                if active_col:
                    cols_to_use.append(active_col)
                
                df_clean = df[cols_to_use].copy()
                
                # Rename columns
                new_names = ['State', 'Confirmed']
                if deaths_col:
                    new_names.append('Deaths')
                if recovered_col:
                    new_names.append('Recovered')
                if active_col:
                    new_names.append('Active')
                
                df_clean.columns = new_names
                
                # Convert to numeric
                for col in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
                    if col in df_clean.columns:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill missing columns with 0
                if 'Deaths' not in df_clean.columns:
                    df_clean['Deaths'] = 0
                if 'Recovered' not in df_clean.columns:
                    df_clean['Recovered'] = 0
                if 'Active' not in df_clean.columns:
                    df_clean['Active'] = df_clean['Confirmed'] - df_clean['Deaths'] - df_clean['Recovered']
                
                # Add date
                df_clean['Date'] = date_obj
                
                # Filter out invalid rows
                df_clean = df_clean.dropna(subset=['State', 'Confirmed'])
                df_clean = df_clean[df_clean['Confirmed'] > 0]
                
                all_data.append(df_clean)
        except Exception as e:
            continue
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

# Load data
df = load_and_process_data()

# Create visualization with improved layout and spacing
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define consistent color palette for top 5 states
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
state_names = ['Delhi', 'Maharashtra', 'Tamil Nadu', 'Gujarat', 'Rajasthan']

# Create main 2x2 grid with increased spacing
gs_main = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.4, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)

if df is not None and len(df) > 0:
    # Get top 5 most affected states by maximum confirmed cases
    max_cases = df.groupby('State')['Confirmed'].max().sort_values(ascending=False)
    top_states = max_cases.head(5).index.tolist()
    
    # Filter data for top states
    df_top = df[df['State'].isin(top_states)].copy()
    df_top = df_top.sort_values(['State', 'Date'])
    
    # Calculate daily new cases and growth rates
    df_top['Daily_New'] = df_top.groupby('State')['Confirmed'].diff().fillna(0)
    df_top['Growth_Rate'] = df_top.groupby('State')['Confirmed'].pct_change().fillna(0) * 100
    df_top['CFR'] = (df_top['Deaths'] / df_top['Confirmed'] * 100).fillna(0)
    
    # Calculate 7-day moving averages
    df_top['Growth_Rate_MA'] = df_top.groupby('State')['Growth_Rate'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    df_top['Growth_Rate_Std'] = df_top.groupby('State')['Growth_Rate'].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
    
    # Create consistent color mapping
    state_colors = dict(zip(top_states, colors[:len(top_states)]))
    
    # Subplot 1 (0,0): Line chart with markers + filled area for deaths
    ax1 = fig.add_subplot(gs_main[0, 0])
    
    for i, state in enumerate(top_states):
        state_data = df_top[df_top['State'] == state].sort_values('Date')
        if len(state_data) > 0:
            # Line chart for confirmed cases
            ax1.plot(state_data['Date'], state_data['Confirmed'], 
                    marker='o', markersize=3, linewidth=2.5, 
                    color=state_colors[state], alpha=0.9)
            
            # Filled area for cumulative deaths
            ax1.fill_between(state_data['Date'], 0, state_data['Deaths'], 
                           color=state_colors[state], alpha=0.3)
    
    ax1.set_title('Confirmed Cases with Cumulative Deaths\nLines: Confirmed Cases, Areas: Deaths', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    
    # Simplified legend - only state names with colors
    legend_elements = [plt.Line2D([0], [0], color=state_colors[state], lw=3, label=state) 
                      for state in top_states]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax1.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # Set spine colors
    for spine in ax1.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1)
    
    # Add subtle annotation
    if len(df_top) > 0:
        lockdown_date = datetime(2020, 5, 1)
        max_confirmed = df_top['Confirmed'].max()
        ax1.annotate('Lockdown', xy=(lockdown_date, max_confirmed*0.15), 
                    xytext=(datetime(2020, 4, 25), max_confirmed*0.3),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                             edgecolor='red', alpha=0.8, linewidth=1))
    
    # Subplot 2 (0,1): SINGLE stacked area chart for all states
    ax2 = fig.add_subplot(gs_main[0, 1])
    
    # Prepare data for stacked area chart
    dates_all = sorted(df_top['Date'].unique())
    
    # Initialize arrays for stacking
    active_stack = np.zeros(len(dates_all))
    recovered_stack = np.zeros(len(dates_all))
    deaths_stack = np.zeros(len(dates_all))
    
    # Accumulate data across all states
    for state in top_states:
        state_data = df_top[df_top['State'] == state].sort_values('Date')
        if len(state_data) > 0:
            # Interpolate to common date grid
            active_interp = np.interp([d.timestamp() for d in dates_all], 
                                    [d.timestamp() for d in state_data['Date']], 
                                    state_data['Active'])
            recovered_interp = np.interp([d.timestamp() for d in dates_all], 
                                       [d.timestamp() for d in state_data['Date']], 
                                       state_data['Recovered'])
            deaths_interp = np.interp([d.timestamp() for d in dates_all], 
                                    [d.timestamp() for d in state_data['Date']], 
                                    state_data['Deaths'])
            
            active_stack += active_interp
            recovered_stack += recovered_interp
            deaths_stack += deaths_interp
    
    # Create stacked area chart
    ax2.fill_between(dates_all, 0, active_stack, 
                    color='#ff6b6b', alpha=0.8, label='Active Cases')
    ax2.fill_between(dates_all, active_stack, active_stack + recovered_stack, 
                    color='#4ecdc4', alpha=0.8, label='Recovered Cases')
    ax2.fill_between(dates_all, active_stack + recovered_stack, 
                    active_stack + recovered_stack + deaths_stack, 
                    color='#45b7d1', alpha=0.8, label='Deaths')
    
    ax2.set_title('Stacked Case Composition Over Time\n(All Top 5 States Combined)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # Set spine colors
    for spine in ax2.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1)
    
    # Add annotation for peak active cases
    peak_idx = np.argmax(active_stack)
    peak_date = dates_all[peak_idx]
    peak_value = active_stack[peak_idx]
    ax2.annotate('Peak Active', xy=(peak_date, peak_value), 
                xytext=(peak_date, peak_value * 1.15),
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.8, lw=1.5),
                fontsize=10, fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='darkred', alpha=0.8, linewidth=1))
    
    # Subplot 3 (1,0): Growth rate with small multiples
    gs_growth = gs_main[1, 0].subgridspec(2, 3, hspace=0.6, wspace=0.4)
    
    # Add main title for growth rate section
    fig.text(0.275, 0.48, 'Daily Growth Rate with 7-Day Moving Average', 
             fontsize=14, fontweight='bold', ha='center')
    
    for idx, state in enumerate(top_states):
        row = idx // 3
        col = idx % 3
        ax_growth = fig.add_subplot(gs_growth[row, col])
        
        state_data = df_top[df_top['State'] == state].sort_values('Date')
        if len(state_data) > 0:
            # Plot growth rate
            ax_growth.plot(state_data['Date'], state_data['Growth_Rate_MA'], 
                          linewidth=2, color=state_colors[state], alpha=0.9)
            
            # Add confidence bands
            upper_band = state_data['Growth_Rate_MA'] + state_data['Growth_Rate_Std']
            lower_band = state_data['Growth_Rate_MA'] - state_data['Growth_Rate_Std']
            ax_growth.fill_between(state_data['Date'], lower_band, upper_band, 
                                 color=state_colors[state], alpha=0.3)
        
        ax_growth.set_title(state, fontsize=11, fontweight='bold', 
                           color=state_colors[state], pad=10)
        ax_growth.set_ylabel('Growth Rate (%)', fontsize=9)
        ax_growth.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
        ax_growth.tick_params(axis='x', rotation=45, labelsize=8)
        ax_growth.tick_params(axis='y', labelsize=8)
        ax_growth.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Reduce number of x-axis ticks to prevent overlap
        ax_growth.locator_params(axis='x', nbins=4)
        
        # Set spine colors
        for spine in ax_growth.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
    
    # Subplot 4 (1,1): Dual-axis plot with small multiples
    gs_dual = gs_main[1, 1].subgridspec(2, 3, hspace=0.6, wspace=0.4)
    
    # Add main title for dual-axis section
    fig.text(0.725, 0.48, 'Daily New Cases vs Case Fatality Rate', 
             fontsize=14, fontweight='bold', ha='center')
    
    for idx, state in enumerate(top_states):
        row = idx // 3
        col = idx % 3
        ax_dual = fig.add_subplot(gs_dual[row, col])
        ax_dual_twin = ax_dual.twinx()
        
        state_data = df_top[df_top['State'] == state].sort_values('Date')
        if len(state_data) > 0:
            # Bar chart for daily new cases
            bars = ax_dual.bar(state_data['Date'], state_data['Daily_New'], 
                              width=1, alpha=0.6, color=state_colors[state])
            
            # Line plot for CFR
            line = ax_dual_twin.plot(state_data['Date'], state_data['CFR'], 
                                   linewidth=2, marker='o', markersize=2,
                                   color='darkred', alpha=0.8)
        
        ax_dual.set_title(state, fontsize=11, fontweight='bold', 
                         color=state_colors[state], pad=10)
        ax_dual.set_ylabel('New Cases', fontsize=9, color=state_colors[state])
        ax_dual_twin.set_ylabel('CFR (%)', fontsize=9, color='darkred')
        
        ax_dual.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
        ax_dual.tick_params(axis='x', rotation=45, labelsize=8)
        ax_dual.tick_params(axis='y', labelcolor=state_colors[state], labelsize=8)
        ax_dual_twin.tick_params(axis='y', labelcolor='darkred', labelsize=8)
        
        # Reduce number of x-axis ticks
        ax_dual.locator_params(axis='x', nbins=4)
        
        # Set spine colors
        for spine in ax_dual.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
        for spine in ax_dual_twin.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
        
        # Add simple legend only to first subplot
        if idx == 0:
            ax_dual.text(0.02, 0.98, 'Bars: New Cases\nLine: CFR', 
                        transform=ax_dual.transAxes, fontsize=8, 
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 alpha=0.8, edgecolor='gray'))

# Fallback with sample data if real data unavailable
else:
    print("Creating sample visualization...")
    
    # Sample data
    dates = pd.date_range('2020-04-15', '2020-07-30', freq='D')
    
    # Subplot 1: Sample confirmed cases and deaths
    ax1 = fig.add_subplot(gs_main[0, 0])
    
    for i, state in enumerate(state_names):
        base_growth = np.random.exponential(50, len(dates))
        confirmed = np.cumsum(base_growth) * (i + 1) * 100
        deaths = confirmed * 0.025
        
        ax1.plot(dates, confirmed, marker='o', markersize=3, 
                linewidth=2.5, color=colors[i], alpha=0.9)
        ax1.fill_between(dates, 0, deaths, color=colors[i], alpha=0.3)
    
    ax1.set_title('Confirmed Cases with Cumulative Deaths (Sample)\nLines: Confirmed Cases, Areas: Deaths', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=3, label=state) 
                      for i, state in enumerate(state_names)]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax1.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    for spine in ax1.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1)
    
    # Subplot 2: Sample stacked area chart
    ax2 = fig.add_subplot(gs_main[0, 1])
    
    active_total = np.cumsum(np.random.exponential(100, len(dates)))
    recovered_total = np.cumsum(np.random.exponential(80, len(dates)))
    deaths_total = np.cumsum(np.random.exponential(10, len(dates)))
    
    ax2.fill_between(dates, 0, active_total, color='#ff6b6b', alpha=0.8, label='Active Cases')
    ax2.fill_between(dates, active_total, active_total + recovered_total, 
                    color='#4ecdc4', alpha=0.8, label='Recovered Cases')
    ax2.fill_between(dates, active_total + recovered_total, 
                    active_total + recovered_total + deaths_total, 
                    color='#45b7d1', alpha=0.8, label='Deaths')
    
    ax2.set_title('Stacked Case Composition Over Time (Sample)\n(All Top 5 States Combined)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    
    for spine in ax2.spines.values():
        spine.set_color('#404040')
        spine.set_linewidth(1)
    
    # Sample growth rate small multiples
    gs_growth = gs_main[1, 0].subgridspec(2, 3, hspace=0.6, wspace=0.4)
    fig.text(0.275, 0.48, 'Daily Growth Rate with 7-Day Moving Average (Sample)', 
             fontsize=14, fontweight='bold', ha='center')
    
    sample_dates = dates[:30]
    for idx, state in enumerate(state_names):
        row = idx // 3
        col = idx % 3
        ax_growth = fig.add_subplot(gs_growth[row, col])
        
        growth_rate = np.random.normal(5, 3, len(sample_dates))
        growth_ma = pd.Series(growth_rate).rolling(window=7, min_periods=1).mean()
        growth_std = pd.Series(growth_rate).rolling(window=7, min_periods=1).std()
        
        ax_growth.plot(sample_dates, growth_ma, linewidth=2, color=colors[idx], alpha=0.9)
        ax_growth.fill_between(sample_dates, growth_ma - growth_std, growth_ma + growth_std, 
                              color=colors[idx], alpha=0.3)
        
        ax_growth.set_title(state, fontsize=11, fontweight='bold', color=colors[idx], pad=10)
        ax_growth.set_ylabel('Growth Rate (%)', fontsize=9)
        ax_growth.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
        ax_growth.tick_params(axis='x', rotation=45, labelsize=8)
        ax_growth.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_growth.locator_params(axis='x', nbins=4)
        
        for spine in ax_growth.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
    
    # Sample dual-axis small multiples
    gs_dual = gs_main[1, 1].subgridspec(2, 3, hspace=0.6, wspace=0.4)
    fig.text(0.725, 0.48, 'Daily New Cases vs Case Fatality Rate (Sample)', 
             fontsize=14, fontweight='bold', ha='center')
    
    for idx, state in enumerate(state_names):
        row = idx // 3
        col = idx % 3
        ax_dual = fig.add_subplot(gs_dual[row, col])
        ax_dual_twin = ax_dual.twinx()
        
        daily_new = np.random.poisson(100, len(sample_dates)) * (idx + 1)
        cfr = np.random.uniform(1, 5, len(sample_dates))
        
        ax_dual.bar(sample_dates, daily_new, width=1, alpha=0.6, color=colors[idx])
        ax_dual_twin.plot(sample_dates, cfr, linewidth=2, marker='o', markersize=2,
                         color='darkred', alpha=0.8)
        
        ax_dual.set_title(state, fontsize=11, fontweight='bold', color=colors[idx], pad=10)
        ax_dual.set_ylabel('New Cases', fontsize=9, color=colors[idx])
        ax_dual_twin.set_ylabel('CFR (%)', fontsize=9, color='darkred')
        
        ax_dual.grid(True, alpha=0.3, color='lightgray', linewidth=0.5)
        ax_dual.tick_params(axis='x', rotation=45, labelsize=8)
        ax_dual.tick_params(axis='y', labelcolor=colors[idx], labelsize=8)
        ax_dual_twin.tick_params(axis='y', labelcolor='darkred', labelsize=8)
        ax_dual.locator_params(axis='x', nbins=4)
        
        for spine in ax_dual.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
        for spine in ax_dual_twin.spines.values():
            spine.set_color('#404040')
            spine.set_linewidth(0.8)
        
        if idx == 0:
            ax_dual.text(0.02, 0.98, 'Bars: New Cases\nLine: CFR', 
                        transform=ax_dual.transAxes, fontsize=8, 
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 alpha=0.8, edgecolor='gray'))

plt.savefig('covid19_india_comprehensive_analysis_refined.png', dpi=300, bbox_inches='tight')
plt.show()