# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing cryptocurrency transaction patterns over time. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Transaction Volume Analysis
- Subplot 1: Daily transaction volume (line chart) with 7-day moving average overlay (smoothed line) for both BTC and ETH
- Subplot 2: Cumulative transaction amounts over time (area chart) with individual transaction scatter points overlaid, separated by currency
- Subplot 3: Transaction frequency per day (bar chart) with average transaction size trend line overlay, comparing BTC vs ETH

Row 2: Fee and Gas Price Dynamics
- Subplot 4: Transaction fees over time (scatter plot) with LOWESS regression line, colored by currency type
- Subplot 5: For ETH transactions only - Gas price distribution over time (violin plot) with median gas price trend line overlay
- Subplot 6: Fee-to-amount ratio over time (line chart) with volatility bands (error bars showing standard deviation), separated by currency

Row 3: Mining Pool and Market Activity
- Subplot 7: Mining pool activity over time (stacked area chart) showing transaction count contribution with total volume line overlay
- Subplot 8: Transaction size distribution evolution (ridge plot/joy plot) showing weekly distributions with median markers
- Subplot 9: Market activity intensity heatmap showing hourly transaction patterns with average transaction value contour lines overlaid

Each subplot must include proper time-based x-axis formatting, currency-specific color coding, and appropriate legends. Use the Timestamp column for all temporal analysis.

## Files
Cryptocurrency Transaction Data.csv

-------

