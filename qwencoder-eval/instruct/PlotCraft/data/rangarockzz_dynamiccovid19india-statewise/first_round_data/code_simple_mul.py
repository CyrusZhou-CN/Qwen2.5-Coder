import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define the data files and their corresponding dates
data_files = [
    ('15-04-2020.csv', '2020-04-15'),
    ('23-04-2020.csv', '2020-04-23'),
    ('01-05-2020.csv', '2020-05-01'),
    ('11-05-2020.csv', '2020-05-11'),
    ('23-05-2020.csv', '2020-05-23'),
    ('02-06-2020.csv', '2020-06-02'),
    ('13-06-2020.csv', '2020-06-13'),
    ('22-06-2020.csv', '2020-06-22'),
    ('july1-2020.csv', '2020-07-01'),
    ('july10-2020.csv', '2020-07-10'),
    ('july19-2020.csv', '2020-07-19')
]

# Initialize dictionary to store data
covid_data = {}

# Process each file
for filename, date_str in data_files:
    try:
        df = pd.read_csv(filename)
        
        # Find the column with total confirmed cases
        total_cases_col = None
        for col in df.columns:
            if 'Total Confirmed cases' in col:
                total_cases_col = col
                break
        
        if total_cases_col is None:
            print(f"No total cases column found in {filename}")
            continue
        
        # Clean and prepare data
        df = df.dropna(subset=['Name of State / UT', total_cases_col])
        df['Name of State / UT'] = df['Name of State / UT'].astype(str).str.strip()
        df[total_cases_col] = pd.to_numeric(df[total_cases_col], errors='coerce')
        df = df.dropna(subset=[total_cases_col])
        
        # Store data for this date
        covid_data[date_str] = {}
        for _, row in df.iterrows():
            state = row['Name of State / UT']
            cases = row[total_cases_col]
            if isinstance(state, str) and len(state) > 1 and cases >= 0:
                covid_data[date_str][state] = cases
        
        print(f"Successfully processed {filename} with {len(covid_data[date_str])} states")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

# Get all unique states across all dates
all_states = set()
for date_data in covid_data.values():
    all_states.update(date_data.keys())

print(f"Total unique states found: {len(all_states)}")

# Create time series data for each state
time_series_data = {}
dates_list = sorted(covid_data.keys())

for state in all_states:
    time_series_data[state] = []
    for date in dates_list:
        if date in covid_data and state in covid_data[date]:
            time_series_data[state].append(covid_data[date][state])
        else:
            # Use previous value if available, otherwise 0
            if time_series_data[state]:
                time_series_data[state].append(time_series_data[state][-1])
            else:
                time_series_data[state].append(0)

# Find top 5 states by maximum total cases across all dates
max_cases_by_state = {}
for state, cases_list in time_series_data.items():
    if cases_list:
        max_cases_by_state[state] = max(cases_list)

# Get top 5 states
top_5_states = sorted(max_cases_by_state.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_state_names = [state[0] for state in top_5_states]

print(f"Top 5 states: {top_5_state_names}")
print(f"Their max cases: {[state[1] for state in top_5_states]}")

# Convert dates to datetime objects for plotting
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates_list]

# Create the line chart
plt.figure(figsize=(14, 8))

# Define colors for the lines
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot lines for top 5 states
for i, state in enumerate(top_5_state_names):
    cases = time_series_data[state]
    plt.plot(dates, cases, marker='o', linewidth=2.5, markersize=6, 
             color=colors[i], label=state, alpha=0.9)

# Styling and formatting
plt.title('COVID-19 Progression: Total Confirmed Cases in Top 5 Most Affected States/UTs in India', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Total Confirmed Cases', fontsize=12, fontweight='bold')

# Format y-axis to show numbers in a readable format
def format_cases(x, p):
    if x >= 100000:
        return f'{x/100000:.1f}L'
    elif x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{int(x)}'

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_cases))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Position legend
plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
           fontsize=10)

# Set background colors
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Ensure proper layout
plt.tight_layout()

# Save the plot
plt.savefig('covid19_progression_top5_states.png', dpi=300, bbox_inches='tight')
plt.show()