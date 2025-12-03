import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Function to parse date from filename
def parse_date_from_filename(filename):
    """Extract date from filename and convert to datetime object"""
    base_name = os.path.basename(filename).replace('.csv', '')
    
    # Handle different date formats in filenames
    if base_name.startswith('july'):
        day = base_name.replace('july', '').replace('-2020', '')
        return datetime(2020, 7, int(day))
    elif base_name.startswith('aug'):
        day = base_name.replace('aug', '').replace('-2020', '')
        return datetime(2020, 8, int(day))
    else:
        # Handle DD-MM-YYYY format
        try:
            return datetime.strptime(base_name, '%d-%m-%Y')
        except:
            # Handle other formats
            parts = base_name.split('-')
            if len(parts) == 3:
                return datetime(2020, int(parts[1]), int(parts[0]))
    return None

# Function to standardize column names and extract total confirmed cases
def get_confirmed_cases_column(df):
    """Find and return the column containing total confirmed cases"""
    possible_columns = [
        'Total Confirmed cases*',
        'Total Confirmed cases (Including 77 foreign Nationals)',
        'Total Confirmed cases (Including 111 foreign Nationals)',
        'Total Confirmed cases (Including 76 foreign Nationals)'
    ]
    
    for col in possible_columns:
        if col in df.columns:
            return col
    return None

# Load and process all CSV files
data_files = [
    '23-04-2020.csv', '24-04-2020.csv', '25-04-2020.csv', '26-04-2020.csv', '27-04-2020.csv',
    '28-04-2020.csv', '29-04-2020.csv', '30-04-2020.csv', '01-05-2020.csv', '02-05-2020.csv',
    '03-05-2020.csv', '04-05-2020.csv', '05-05-2020.csv', '06-05-2020.csv', '07-05-2020.csv',
    '08-05-2020.csv', '09-05-2020.csv', '10-05-2020.csv', '11-05-2020.csv', '12-05-2020.csv',
    '13-05-2020.csv', '15-05-2020.csv', '16-05-2020.csv', '17-05-2020.csv', '18-05-2020.csv',
    '19-05-2020.csv', '20-05-2020.csv', '21-05-2020.csv', '22-05-2020.csv', '23-05-2020.csv',
    '24-05-2020.csv', '25-05-2020.csv', '26-05-2020.csv', '27-05-2020.csv', '28-05-2020.csv',
    '29-05-2020.csv', '30-05-2020.csv', '31-05-2020.csv', '01-06-2020.csv', '02-06-2020.csv',
    '03-06-2020.csv', '04-06-2020.csv', '05-06-2020.csv', '06-06-2020.csv', '07-06-2020.csv',
    '08-06-2020.csv', '09-06-2020.csv', '10-06-2020.csv', '11-06-2020.csv', '12-06-2020.csv',
    '13-06-2020.csv', '14-06-2020.csv', '17-06-2020.csv', '18-06-2020.csv', '19-06-2020.csv',
    '20-06-2020.csv', '21-06-2020.csv', '22-06-2020.csv', '23-06-2020.csv', '24-06-2020.csv',
    '25-06-2020.csv', '26-06-2020.csv', '27-06-2020.csv', '28-06-2020.csv', '29-06-2020.csv',
    '30-06-2020.csv', 'july1-2020.csv', 'july2-2020.csv', 'july3-2020.csv', 'july4-2020.csv',
    'july5-2020.csv', 'july6-2020.csv', 'july7-2020.csv', 'july8-2020.csv', 'july9-2020.csv',
    'july10-2020.csv', 'july11-2020.csv', 'july12-2020.csv', 'july13-2020.csv', 'july14-2020.csv',
    'july15-2020.csv', 'july16-2020.csv', 'july17-2020.csv', 'july18-2020.csv', 'july19-2020.csv'
]

# Collect data from all files
all_data = []

for filename in data_files:
    try:
        df = pd.read_csv(filename)
        date = parse_date_from_filename(filename)
        
        if date is None:
            continue
            
        # Skip files with problematic structure (like aug files with unnamed columns)
        if 'Unnamed' in str(df.columns) and len([col for col in df.columns if 'Unnamed' in str(col)]) > 2:
            continue
            
        # Get the confirmed cases column
        confirmed_col = get_confirmed_cases_column(df)
        if confirmed_col is None:
            continue
            
        # Clean and process the data
        df_clean = df.copy()
        
        # Standardize state name column
        state_col = 'Name of State / UT'
        if state_col not in df_clean.columns:
            continue
            
        # Remove rows with NaN state names
        df_clean = df_clean.dropna(subset=[state_col])
        
        # Convert confirmed cases to numeric
        df_clean[confirmed_col] = pd.to_numeric(df_clean[confirmed_col], errors='coerce')
        
        # Remove rows with NaN confirmed cases
        df_clean = df_clean.dropna(subset=[confirmed_col])
        
        # Add date column
        df_clean['Date'] = date
        
        # Select relevant columns
        df_clean = df_clean[[state_col, confirmed_col, 'Date']]
        df_clean.columns = ['State', 'Confirmed_Cases', 'Date']
        
        all_data.append(df_clean)
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean state names
    combined_df['State'] = combined_df['State'].str.strip()
    
    # Get the latest date's data to find top 5 states
    latest_date = combined_df['Date'].max()
    latest_data = combined_df[combined_df['Date'] == latest_date]
    
    # Find top 5 states by confirmed cases
    top_5_states = latest_data.nlargest(5, 'Confirmed_Cases')['State'].tolist()
    
    # Filter data for top 5 states
    top_5_data = combined_df[combined_df['State'].isin(top_5_states)]
    
    # Create the visualization
    plt.figure(figsize=(14, 8))
    
    # Define colors for the lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot line for each state
    for i, state in enumerate(top_5_states):
        state_data = top_5_data[top_5_data['State'] == state].sort_values('Date')
        plt.plot(state_data['Date'], state_data['Confirmed_Cases'], 
                marker='o', markersize=4, linewidth=2.5, color=colors[i], 
                label=state, alpha=0.8)
    
    # Customize the plot
    plt.title('COVID-19 Progression: Top 5 Most Affected States/UTs in India\n(April - August 2020)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Total Confirmed Cases', fontsize=12, fontweight='bold')
    
    # Format y-axis to show numbers in thousands/lakhs
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x < 100000 else f'{x/100000:.1f}L'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
               fontsize=10, title='States/UTs', title_fontsize=11)
    
    # Set background color to white
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Print the top 5 states for reference
    print("Top 5 Most Affected States/UTs by Total Confirmed Cases:")
    for i, state in enumerate(top_5_states, 1):
        latest_cases = latest_data[latest_data['State'] == state]['Confirmed_Cases'].iloc[0]
        print(f"{i}. {state}: {latest_cases:,.0f} cases")

else:
    print("No data could be processed from the available files.")

# Save the plot
plt.savefig('covid19_progression_top5_states.png', dpi=300, bbox_inches='tight')