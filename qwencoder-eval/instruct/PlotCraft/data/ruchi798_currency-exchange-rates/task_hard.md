# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x3 subplot grid analyzing currency exchange rate patterns and relationships. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Time series analysis with trend decomposition
- Subplot 1: Line chart with moving averages (7-day and 30-day) for top 5 strongest currencies
- Subplot 2: Area chart with confidence bands showing exchange rate volatility over time for major currency groups (USD, EUR, GBP, JPY, AUD)
- Subplot 3: Seasonal decomposition plot combining original time series with trend and residual components

Row 2: Distribution and correlation analysis
- Subplot 4: Histogram with overlaid KDE curves comparing exchange rate distributions across different currency regions (European, Asian, American, Oceanic)
- Subplot 5: Scatter plot with regression lines showing correlation between currency strength and regional economic indicators, with bubble sizes representing trading volume
- Subplot 6: Box plots with violin plot overlays displaying exchange rate quartiles by currency type (major vs minor currencies)

Row 3: Comparative and ranking analysis
- Subplot 7: Diverging bar chart with error bars showing percentage change from baseline (EUR=1.0) with confidence intervals
- Subplot 8: Slope chart with connected points showing currency ranking changes over time periods, combined with background area shading for volatility zones
- Subplot 9: Radar chart overlaid with line plots comparing multiple currency performance metrics (stability, strength, volatility) across different time windows

Each subplot must include proper legends, annotations for significant events, and color coding by currency regions. Use consistent styling and ensure all composite elements are clearly distinguishable.

## Files
exchange_rates.csv

-------

