# Visualization Task - Middle

## Category
Change

## Instruction
Create a comprehensive 2x2 subplot grid analyzing COVID-19 forecast trends evolution across multiple time periods. Each subplot should be a composite visualization combining multiple chart types:

Top-left: Create a multi-line time series plot showing the trend component evolution for 5 selected countries across 4 different forecast dates (2021-01-24, 2021-06-01, 2021-09-04, 2021-12-07), overlaid with scatter points highlighting significant trend changes and a secondary y-axis showing the average weekly multiplicative effect.

Top-right: Develop a stacked area chart displaying the confirmed cases predictions for the same 5 countries across the same 4 time periods, with error bands (using yhat_lower and yhat_upper) shown as transparent filled areas, and add vertical reference lines marking major COVID-19 waves.

Bottom-left: Construct a slope chart connecting the initial trend values (from earliest forecast) to final trend values (from latest forecast) for 10 countries, with the slope lines colored by the magnitude of change, and overlay box plots showing the distribution of weekly multiplicative terms for each country.

Bottom-right: Generate a combination chart with bar plots showing the average forecast accuracy range (yhat_upper - yhat_lower) for each country across all time periods, overlaid with a line plot showing the trend volatility (standard deviation of trend values) and scatter points indicating countries with holiday effects (non-zero holiday values).

## Files
forecast_future_dfs_2021-01-24.csv
forecast_future_dfs_2021-09-04.csv
forecast_future_dfs_2021-12-07.csv
forecast_future_dfs_2022-01-11.csv

-------

