import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
forecast_data = pd.read_csv('forecast_data.csv')
location_data = pd.read_csv('location_data.csv')

# Filter the top 5 most populous states based on population data (assuming population data is available in location_data)
top_states = location_data['name'].head(5)

# Filter the forecast data for the top 5 states
filtered_data = forecast_data[forecast_data['state'].isin(top_states)]

# Convert time_epoch to datetime
filtered_data['time'] = pd.to_datetime(filtered_data['time'])

# Group by state and calculate average daily temperatures
avg_temp = filtered_data.groupby(['state', 'time']).mean().reset_index()

# Calculate temperature range (max - min)
temperature_range = filtered_data.groupby(['state', 'time']).agg({'temp_c': ['min', 'max']}).reset_index()
temperature_range.columns = ['state', 'time', 'min_temp', 'max_temp']
temperature_range['range_temp'] = temperature_range['max_temp'] - temperature_range['min_temp']

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# Top plot: Line chart of average daily temperatures
for state in top_states:
    avg_state_temp = avg_temp[avg_temp['state'] == state]
    axes[0].plot(avg_state_temp['time'], avg_state_temp['temp_c'], label=state)

axes[0].set_title('Average Daily Temperatures in Top 5 States')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Temperature (°C)')
axes[0].legend()
axes[0].grid(True)

# Bottom plot: Stacked area chart of temperature range
colors = plt.cm.tab20.colors[:len(top_states)]
for i, state in enumerate(top_states):
    state_range_temp = temperature_range[temperature_range['state'] == state]
    axes[1].fill_between(state_range_temp['time'], state_range_temp['min_temp'], state_range_temp['max_temp'], color=colors[i], alpha=0.7, label=state)

axes[1].set_title('Temperature Range in Top 5 States')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Temperature Range (°C)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()