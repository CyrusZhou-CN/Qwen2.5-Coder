# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the temporal dynamics of Rio de Janeiro bus system. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Daily patterns analysis
- Subplot 1: Combine line chart showing average hourly bus count with area chart showing speed distribution bands (25th-75th percentile) throughout the day
- Subplot 2: Overlay bar chart of total distance traveled per hour with line chart showing average speed per hour
- Subplot 3: Create dual-axis plot with histogram of GPS readings frequency per hour and KDE curve of speed distribution per hour

Row 2: Weekly patterns analysis  
- Subplot 4: Combine heatmap showing average speed by day-of-week vs hour with contour lines indicating bus density levels
- Subplot 5: Stack area chart showing cumulative distance by different bus lines over days of week, overlaid with line chart showing average fleet size
- Subplot 6: Create violin plots showing speed distribution for each day of week, overlaid with box plots and scatter points for outliers

Row 3: Route-based temporal analysis
- Subplot 7: Combine multiple line charts showing speed trends over time for top 5 busiest routes, with filled area showing confidence intervals
- Subplot 8: Create composite chart with bar chart showing route frequency by time periods and scatter plot with trend line showing relationship between route length and average speed
- Subplot 9: Overlay time series decomposition plot (trend, seasonal, residual components) of overall fleet activity with histogram showing distribution of daily total GPS readings

Use different color schemes for each row and ensure all temporal patterns reveal insights about Rio's urban mobility dynamics.

## Files
treatedBusDataOnlyRoute.csv

-------

