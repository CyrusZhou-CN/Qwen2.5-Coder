# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing COVID-19 forecast evolution across multiple time periods and countries. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Temporal Trend Analysis
- Subplot 1: Line chart showing confirmed cases trends over time with error bands (yhat_lower/yhat_upper) for 3 selected countries, overlaid with scatter points marking actual confirmed values
- Subplot 2: Area chart displaying trend component evolution across different forecast dates, with a secondary line plot showing multiplicative_terms patterns
- Subplot 3: Stacked area chart of forecast components (trend + multiplicative_terms) with line overlay showing prediction accuracy (difference between confirmed and trend)

Row 2: Forecast Performance Analysis  
- Subplot 4: Box plots showing distribution of prediction errors by country, overlaid with violin plots to show error density distributions
- Subplot 5: Scatter plot of actual vs predicted values with best-fit line and confidence intervals, colored by country and sized by weekly multiplicative factor
- Subplot 6: Time series decomposition showing trend component with seasonal (weekly) patterns as separate line plots on the same axes

Row 3: Cross-Country Comparative Analysis
- Subplot 7: Heatmap showing correlation matrix between countries' confirmed cases, overlaid with hierarchical clustering dendrogram
- Subplot 8: Multiple line charts showing normalized confirmed cases trajectories by country, with ribbon plots indicating forecast uncertainty ranges
- Subplot 9: Radar chart comparing countries across multiple forecast metrics (mean confirmed cases, trend slope, weekly seasonality strength), combined with parallel coordinates plot

Use data from at least 5 different forecast dates and focus on 8-10 countries with the most complete data. Apply consistent color schemes and ensure each composite subplot tells a distinct story about COVID-19 forecast patterns and cross-country differences.

## Files
forecast_future_dfs_2021-01-24.csv
forecast_future_dfs_2021-03-21.csv
forecast_future_dfs_2021-09-04.csv
forecast_future_dfs_2021-11-21.csv
forecast_future_dfs_2022-01-24.csv

-------

