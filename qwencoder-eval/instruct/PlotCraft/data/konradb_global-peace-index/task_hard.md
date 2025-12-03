# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the Global Peace Index trends from 2008-2023. Each subplot should be a composite visualization combining multiple chart types:

Top row (Regional Analysis): 
- Subplot 1: Line chart with error bands showing mean peace index trends over time for top 5 most peaceful regions, overlaid with scatter points for individual country data
- Subplot 2: Stacked area chart showing the composition of peace index score ranges (excellent: 1.0-1.5, good: 1.5-2.0, moderate: 2.0-2.5, poor: 2.5-3.0, very poor: 3.0+) over time, with a line plot overlay showing total country count
- Subplot 3: Slope chart connecting 2008 and 2023 peace index values for the 10 countries with the largest changes, with diverging color coding for improvements vs deteriorations

Middle row (Distribution Evolution):
- Subplot 4: Violin plots showing peace index distribution for each year from 2008-2023, overlaid with box plots to highlight quartiles and outliers
- Subplot 5: Heatmap showing year-over-year correlation matrix of peace index scores, with hierarchical clustering dendrogram on the side
- Subplot 6: Joy plot (ridgeline) showing density distributions of peace index scores by 3-year periods (2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2023), with mean lines overlaid

Bottom row (Volatility and Patterns):
- Subplot 7: Scatter plot of peace index volatility (standard deviation across years) vs mean peace index, with bubble sizes representing the range (max-min), and trend line
- Subplot 8: Time series decomposition showing original peace index trend, seasonal component, and residuals for the global average, with confidence intervals
- Subplot 9: Parallel coordinates plot showing peace index trajectories for countries grouped by their 2023 ranking quintiles, with highlighted paths for most improved and most deteriorated countries

Each subplot should include appropriate titles, legends, and annotations highlighting key insights about global peace trends, regional patterns, and country-specific changes over the 15-year period.

## Files
peace_index.csv

-------

