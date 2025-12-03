# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the temporal evolution and market microstructure dynamics of cryptocurrency limit order books across Bitcoin (BTC), Ethereum (ETH), and Cardano (ADA). Each subplot should be a composite visualization combining multiple chart types:

Row 1 - Price Evolution Analysis:
- Subplot (1,1): BTC midpoint price over time with overlaid volatility bands (line chart + area chart for volatility envelope)
- Subplot (1,2): ETH midpoint price over time with overlaid spread dynamics (line chart + secondary y-axis line chart for spread)
- Subplot (1,3): ADA midpoint price over time with overlaid volume-weighted price trends (line chart + scatter plot sized by volume)

Row 2 - Order Flow Dynamics:
- Subplot (2,1): BTC buy vs sell volume over time with net flow indicators (dual-axis line charts + bar chart for net flow)
- Subplot (2,2): ETH order book depth evolution showing bid-ask imbalance over time (stacked area chart + line overlay for imbalance ratio)
- Subplot (2,3): ADA market vs limit order activity over time (stacked bar chart + line chart overlay for market order percentage)

Row 3 - Market Microstructure:
- Subplot (3,1): BTC order book distance analysis showing bid-ask level spreads over time (heatmap + line chart overlay for average distance)
- Subplot (3,2): ETH cancel-to-limit order ratio evolution across different price levels (multiple line charts + filled area for ratio bands)
- Subplot (3,3): ADA cross-cryptocurrency correlation matrix of midpoint returns with time-varying correlation heatmap (correlation heatmap + line chart for rolling correlation)

Use 1-minute frequency data for all analyses to capture high-frequency market dynamics. Each subplot must combine at least two different visualization types and reveal temporal patterns in cryptocurrency market microstructure behavior.

## Files
BTC_1min.csv
ETH_1min.csv
ADA_1min.csv

-------

