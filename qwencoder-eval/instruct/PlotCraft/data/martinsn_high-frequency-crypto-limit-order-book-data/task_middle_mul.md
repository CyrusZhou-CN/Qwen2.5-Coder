# Visualization Task - Middle

## Category
Change

## Instruction
Create a comprehensive 2x2 subplot grid analyzing the temporal evolution of cryptocurrency order book dynamics across different time frequencies. Each subplot should combine multiple visualization types:

Top-left: Line chart showing midpoint price evolution over time for all three cryptocurrencies (BTC, ETH, ADA) using 1-minute data, overlaid with scatter points indicating significant spread changes (spread > 0.5 for any crypto).

Top-right: Stacked area chart displaying the cumulative buy and sell volumes over time for Bitcoin using 5-minute data, with a secondary y-axis line plot showing the bid-ask spread evolution.

Bottom-left: Multi-line chart comparing the distance of the first bid level from midpoint across all three cryptocurrencies using 1-second data, with error bands showing the standard deviation of notional volumes at level 0.

Bottom-right: Heatmap showing the correlation matrix between midpoint prices of the three cryptocurrencies across different time frequencies (1sec, 1min, 5min), with annotations displaying correlation coefficients.

## Files
BTC_1min.csv
ETH_1min.csv
ADA_1min.csv
BTC_5min.csv
BTC_1sec.csv
ETH_1sec.csv
ADA_1sec.csv

-------

