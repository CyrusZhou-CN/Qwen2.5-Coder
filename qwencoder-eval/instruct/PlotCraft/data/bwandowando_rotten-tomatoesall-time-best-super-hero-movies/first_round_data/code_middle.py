import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Load data with proper file path handling
def find_csv_file():
    # Try different possible file locations
    possible_paths = [
        'movies.csv',
        './movies.csv',
        'data/movies.csv',
        '../movies.csv'
    ]
    
    # Also search for any CSV file in current directory
    csv_files = glob.glob('*.csv')
    if csv_files:
        possible_paths.extend(csv_files)
    
    for path in possible_paths:
        if os.path.isfile(path):
            return path
    
    raise FileNotFoundError("Could not find movies.csv file")

# Load the data
csv_path = find_csv_file()
df = pd.read_csv(csv_path)

# Debug: Print column names to understand the actual structure
print("Available columns:", df.columns.tolist())
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Find the correct column names for critic and audience scores
critic_col = None
audience_col = None

# Check for various possible column name patterns
possible_critic_names = ['critic_score', 'Critic Score', 'tomatometer', 'Tomatometer', 'critics_score', 'critic']
possible_audience_names = ['audience_score', 'Audience Score', 'audience', 'user_score', 'users_score']

for col in df.columns:
    col_lower = col.lower()
    if any(name.lower() in col_lower for name in possible_critic_names):
        critic_col = col
    if any(name.lower() in col_lower for name in possible_audience_names):
        audience_col = col

# If we can't find the exact columns, use the ones that contain score-like data
if critic_col is None or audience_col is None:
    # Look for columns with percentage signs or numeric data that could be scores
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10).astype(str)
            if any('%' in str(val) for val in sample_values):
                if critic_col is None:
                    critic_col = col
                elif audience_col is None:
                    audience_col = col

print(f"\nUsing columns: critic_col='{critic_col}', audience_col='{audience_col}'")

# Data preprocessing
# Convert score columns from percentage strings to numeric values
if critic_col and audience_col:
    # Handle percentage strings
    df['critic_score_num'] = pd.to_numeric(df[critic_col].astype(str).str.replace('%', ''), errors='coerce')
    df['audience_score_num'] = pd.to_numeric(df[audience_col].astype(str).str.replace('%', ''), errors='coerce')
    
    # Remove rows with missing scores
    df = df.dropna(subset=['critic_score_num', 'audience_score_num'])
    
    # Find year column
    year_col = None
    for col in df.columns:
        if 'year' in col.lower() or 'date' in col.lower():
            year_col = col
            break
    
    if year_col is None:
        # If no year column found, create a dummy one
        df['movieYear'] = 2000  # Default year
        year_col = 'movieYear'
    
    # Create decade categories
    df['decade'] = (df[year_col] // 10) * 10
    df['decade_label'] = df['decade'].astype(str) + 's'
    
    # Find rank column for review count proxy
    rank_col = None
    for col in df.columns:
        if 'rank' in col.lower() or 'position' in col.lower():
            rank_col = col
            break
    
    if rank_col:
        # Create a proxy for number of critic reviews based on movie rank
        df['review_count_proxy'] = df[rank_col].max() + 1 - df[rank_col]
    else:
        # Use index as proxy if no rank column
        df['review_count_proxy'] = len(df) - df.index
    
    # Ensure review count proxy is positive and reasonable
    df['review_count_proxy'] = np.maximum(df['review_count_proxy'], 1)
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Define color palette for decades
    decades = sorted(df['decade'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    decade_colors = {}
    for i, decade in enumerate(decades):
        decade_colors[decade] = colors[i % len(colors)]
    
    # Create scatter plot with color coding by decade and size by review count
    for decade in decades:
        decade_data = df[df['decade'] == decade]
        if len(decade_data) > 0:
            plt.scatter(decade_data['critic_score_num'], 
                       decade_data['audience_score_num'],
                       c=decade_colors[decade], 
                       s=decade_data['review_count_proxy'] * 2 + 20,  # Scale point sizes
                       alpha=0.7,
                       label=f"{int(decade)}s",
                       edgecolors='white',
                       linewidth=0.5)
    
    # Calculate and plot regression line
    x_vals = df['critic_score_num'].values
    y_vals = df['audience_score_num'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_clean = x_vals[mask]
    y_clean = y_vals[mask]
    
    if len(x_clean) > 1:
        # Fit regression line
        coeffs = np.polyfit(x_clean, y_clean, 1)
        x_line = np.linspace(df['critic_score_num'].min(), df['critic_score_num'].max(), 100)
        y_line = np.polyval(coeffs, x_line)
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x_clean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        plt.plot(x_line, y_line, color='red', linewidth=2, linestyle='--', 
                 label=f'Regression Line (R² = {r_squared:.3f})')
    
    # Identify and annotate notable outliers
    df['score_diff'] = abs(df['critic_score_num'] - df['audience_score_num'])
    outliers = df.nlargest(min(3, len(df)), 'score_diff')
    
    # Find title column
    title_col = None
    for col in df.columns:
        if 'title' in col.lower() or 'name' in col.lower() or 'movie' in col.lower():
            title_col = col
            break
    
    if title_col and len(outliers) > 0:
        for idx, movie in outliers.iterrows():
            # Truncate long movie titles
            title = str(movie[title_col])
            if len(title) > 25:
                title = title[:22] + '...'
            
            plt.annotate(title, 
                        xy=(movie['critic_score_num'], movie['audience_score_num']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9, fontweight='bold')
    
    # Styling and labels
    plt.title('Critical vs. Audience Reception of Superhero Movies\nby Decade and Review Volume', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'Critic Score ({critic_col})', fontsize=12, fontweight='bold')
    plt.ylabel(f'Audience Score ({audience_col})', fontsize=12, fontweight='bold')
    
    # Create custom legend for decades
    if len(decades) > 0:
        legend1 = plt.legend(loc='upper left', title='Release Decade', title_fontsize=11, fontsize=10)
        legend1.get_title().set_fontweight('bold')
        plt.gca().add_artist(legend1)
    
    # Add size legend for review count proxy
    from matplotlib.lines import Line2D
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, alpha=0.7, label='Low Reviews'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, alpha=0.7, label='Medium Reviews'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=14, alpha=0.7, label='High Reviews')
    ]
    legend2 = plt.legend(handles=size_legend_elements, loc='lower right', 
                        title='Review Volume\n(based on rank)', title_fontsize=11, fontsize=10)
    legend2.get_title().set_fontweight('bold')
    
    # Add correlation coefficient as text
    if len(x_clean) > 1:
        correlation = np.corrcoef(x_clean, y_clean)[0, 1]
        plt.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                 verticalalignment='top')
    
    # Set axis limits and grid
    x_min, x_max = df['critic_score_num'].min(), df['critic_score_num'].max()
    y_min, y_max = df['audience_score_num'].min(), df['audience_score_num'].max()
    
    # Add some padding to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(max(0, x_min - x_range*0.05), min(100, x_max + x_range*0.05))
    plt.ylim(max(0, y_min - y_range*0.05), min(100, y_max + y_range*0.05))
    
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add diagonal reference line (perfect agreement) if both axes go to 100
    if x_max >= 90 and y_max >= 90:
        diag_min = max(plt.xlim()[0], plt.ylim()[0])
        diag_max = min(plt.xlim()[1], plt.ylim()[1])
        plt.plot([diag_min, diag_max], [diag_min, diag_max], 
                 color='gray', linestyle=':', alpha=0.5, linewidth=1)
        plt.text(diag_max*0.85, diag_max*0.9, 'Perfect Agreement', 
                 rotation=45, fontsize=9, alpha=0.7, style='italic')
    
    # Layout adjustment
    plt.tight_layout()
    
    print(f"\nVisualization created successfully!")
    print(f"Data points plotted: {len(df)}")
    print(f"Decades represented: {sorted(decades)}")

else:
    print("Error: Could not identify critic and audience score columns in the dataset")
    print("Available columns:", df.columns.tolist())
    
    # Create a simple fallback plot with available data
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f'Dataset columns:\n{chr(10).join(df.columns.tolist())}\n\nCould not identify score columns', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Dataset Column Information', fontsize=14, fontweight='bold')
    plt.axis('off')

# Save the plot
plt.savefig('superhero_movies_correlation.png', dpi=300, bbox_inches='tight')
plt.show()