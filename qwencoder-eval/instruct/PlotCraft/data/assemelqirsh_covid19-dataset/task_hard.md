# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing COVID-19 progression patterns in Afghanistan. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Daily Metrics Evolution
- Subplot 1: Combine line chart for new_cases with bar chart overlay for new_deaths, plus add a secondary y-axis showing stringency_index as a step plot
- Subplot 2: Create dual-axis plot with new_cases_smoothed as area chart and reproduction_rate as scatter plot with trend line
- Subplot 3: Show total_cases as cumulative line chart with new_cases_per_million as histogram overlay using twin axes

Row 2: Comparative Analysis
- Subplot 4: Combine violin plot for new_cases distribution with box plot overlay, plus add strip plot showing individual daily values
- Subplot 5: Create correlation heatmap between [new_cases, new_deaths, stringency_index, reproduction_rate] with scatter plots in upper triangle
- Subplot 6: Show new_deaths_smoothed as both line chart and filled area, with error bands representing ±1 standard deviation

Row 3: Advanced Patterns
- Subplot 7: Create calendar heatmap for new_cases with overlaid trend line showing 7-day moving average
- Subplot 8: Combine autocorrelation plot for new_cases with partial autocorrelation plot as dual subplot
- Subplot 9: Show phase portrait (new_cases vs new_deaths) as scatter plot with density contours and trajectory arrows indicating temporal progression

Each subplot must include proper legends, axis labels, and use different color schemes. Handle missing values appropriately and ensure all temporal data is properly parsed.

## Files
covid_data.csv

-------

