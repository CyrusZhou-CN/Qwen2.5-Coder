# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive correlation analysis dashboard using a 3x2 subplot grid to investigate the relationships between insider trading patterns, transaction values, and stock prices across different S&P 500 companies. Each subplot should be a composite visualization combining multiple chart types:

1. Top-left: Create a scatter plot with marginal histograms showing the relationship between transaction cost (stock price) and transaction value, with points colored by transaction type (Buy/Sale/Option Exercise) and sized by number of shares traded.

2. Top-right: Design a correlation heatmap overlaid with a network graph showing the strength of correlations between numerical variables (Cost, Value, Shares Total) across all companies, with correlation coefficients displayed as text annotations.

3. Middle-left: Construct a bubble plot with trend lines showing the relationship between average transaction cost and total number of transactions per company, with bubble sizes representing total transaction value and different colors for different relationship types (CEO, Director, etc.).

4. Middle-right: Build a scatter plot matrix (pair plot) focusing on the three key numerical variables (Cost, Value converted to numeric, Shares Total converted to numeric) with different colors for the top 10 most active companies by transaction count.

5. Bottom-left: Create a jitter plot combined with box plots showing the distribution of transaction costs across different transaction types, with overlaid violin plots to show the density distribution.

6. Bottom-right: Design a 2D histogram (hexbin plot) with contour lines showing the density relationship between log-transformed transaction values and stock prices, with separate subplots for different insider relationship categories.

Use data from at least 5 different company files, ensure all monetary values are properly converted to numeric format, and apply appropriate color schemes to distinguish between different categories while maintaining visual coherence across all subplots.

## Files
AAPL.csv
AMZN.csv
MSFT.csv
TSLA.csv
GOOGL.csv

-------

