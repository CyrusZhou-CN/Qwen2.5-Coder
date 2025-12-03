# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive time series analysis dashboard with a 3x2 subplot grid (6 subplots total) to analyze the temporal evolution of water quality parameters across multiple aquaponics fish ponds. Each subplot should be a composite visualization combining multiple chart types:

1. Top-left: Temperature trends over time for 4 selected ponds (IoTPond1, IoTPond3, IoTPond6, IoTPond9) - combine line plots with rolling averages and confidence bands showing temperature variability
2. Top-right: pH level changes with dual-axis overlay showing dissolved oxygen levels for the same 4 ponds - use line charts with different colors and add trend lines
3. Middle-left: Ammonia concentration evolution with error bars representing measurement uncertainty, overlaid with box plots showing distribution quartiles for each time period across all 4 ponds
4. Middle-right: Nitrate levels time series with seasonal decomposition components (trend, seasonal, residual) stacked vertically within the subplot for one representative pond
5. Bottom-left: Multi-parameter correlation heatmap animation over time windows, showing how correlations between temperature, pH, dissolved oxygen, and ammonia change across different time periods
6. Bottom-right: Fish growth indicators (length and weight) progression over time with dual y-axes, including polynomial regression fits and prediction intervals for future growth projections

Each composite subplot should include proper legends, time-based x-axis formatting, and statistical annotations (mean, std dev, trend significance). Use different color palettes for each pond to maintain visual distinction across all subplots.

## Files
IoTPond1.csv
IoTPond3.csv
IoTPond6.csv
IoTPond9.csv

-------

