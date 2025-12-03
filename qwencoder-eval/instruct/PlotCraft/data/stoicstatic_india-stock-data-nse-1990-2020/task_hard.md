# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the temporal evolution and volatility patterns of major NSE stocks from 1990-2021. Each subplot must be a composite visualization combining multiple chart types:

Top row (3 subplots): For three major stocks (RELIANCE, TCS, INFY), create combined line charts showing closing price trends over time with overlaid area charts representing 30-day rolling volatility (calculated as rolling standard deviation of daily returns). Add error bands showing ±1 standard deviation around the price trend.

Middle row (3 subplots): For the same three stocks, create dual-axis composite charts where the primary y-axis shows candlestick patterns for the most recent 2 years of data, and the secondary y-axis displays volume as bar charts. Overlay a 50-day moving average line on the candlestick data.

Bottom row (3 subplots): Create time series decomposition plots for each stock showing the original closing price series combined with its trend component (using seasonal decomposition). Add scatter points highlighting major market events (identified as days with >5% daily price changes) and connect consecutive extreme events with line segments to show volatility clustering patterns.

Each subplot must include proper legends, different color schemes for each stock, and annotations for significant market periods (2008 financial crisis, 2020 COVID impact). The overall visualization should reveal how different stocks responded to market cycles and demonstrate varying volatility patterns across the three-decade period.

## Files
RELIANCE.csv
TCS.csv
INFY.csv

-------

