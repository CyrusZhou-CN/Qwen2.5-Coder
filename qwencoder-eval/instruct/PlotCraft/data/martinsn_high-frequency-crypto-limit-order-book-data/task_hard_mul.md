# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x2 subplot grid analyzing the temporal evolution and cross-cryptocurrency dynamics of high-frequency limit order book data. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): For each cryptocurrency (BTC, ETH, ADA), create time series plots showing the evolution of midpoint prices over different time frequencies (1sec, 1min, 5min) with overlaid volatility bands calculated from the spread data. Add secondary y-axis line plots showing the cumulative buy/sell volume imbalance over time.

Bottom row (3 subplots): 
- Left: Multi-cryptocurrency correlation heatmap overlaid with time series of cross-correlation coefficients between BTC-ETH, BTC-ADA, and ETH-ADA midpoint prices across different time frequencies
- Center: Stacked area chart showing the evolution of total order book depth (sum of all bid/ask notional values) for each cryptocurrency over time, with line overlays showing the bid-ask spread dynamics
- Right: Composite chart combining violin plots of spread distributions for each crypto-timeframe combination with connected scatter plots showing the relationship between average spread and trading volume intensity

Use data from all 9 CSV files, focusing on temporal patterns, cross-asset relationships, and market microstructure evolution. Ensure each subplot tells a distinct story about high-frequency trading dynamics while maintaining visual coherence across the grid.

## Files
BTC_1sec.csv
BTC_1min.csv
BTC_5min.csv
ETH_1sec.csv
ETH_1min.csv
ETH_5min.csv
ADA_1sec.csv
ADA_1min.csv
ADA_5min.csv

-------

