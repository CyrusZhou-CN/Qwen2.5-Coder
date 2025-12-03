# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing flight price correlations and patterns. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Price vs Route Analysis
- Subplot 1: Scatter plot with regression line showing price vs flight duration, overlaid with a marginal histogram of prices
- Subplot 2: Box plot of prices by airline with overlaid violin plots to show distribution density
- Subplot 3: Bubble chart showing source-destination pairs (x: source count, y: destination count, bubble size: average price, color: airline)

Row 2: Temporal Price Patterns
- Subplot 4: Line chart of average prices over departure dates with error bands showing price volatility, overlaid with scatter points for individual flights
- Subplot 5: Heatmap of average prices by departure hour vs arrival hour, with contour lines highlighting price zones
- Subplot 6: Stacked area chart showing price distribution across different airlines over time, with trend lines for each airline

Row 3: Multi-dimensional Correlations
- Subplot 7: Parallel coordinates plot showing relationships between duration, stops, departure time, and price, with lines colored by airline
- Subplot 8: Correlation matrix heatmap of all numerical variables with hierarchical clustering dendrogram on the side
- Subplot 9: 3D-style scatter plot (using perspective) showing price vs duration vs stops, with different colors for airlines and size representing additional stops

## Files
Data_Train.xlsx
Test_set.xlsx

-------

