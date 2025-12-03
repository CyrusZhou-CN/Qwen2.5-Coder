# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 2x2 subplot grid analyzing temporal patterns in Bangladesh's English news landscape. Each subplot should be a composite visualization combining multiple chart types:

Top-left: Create a dual-axis time series plot showing daily news volume (bar chart) overlaid with a 7-day rolling average line for news collection patterns across all publishers from the combined dataset.

Top-right: Generate a stacked area chart showing the cumulative contribution of each publisher over time, with an overlaid line plot displaying the diversity index (calculated as the inverse of the Herfindahl index) to show how news source concentration changes over time.

Bottom-left: Construct a calendar heatmap showing news publication intensity by day, combined with marginal histograms on the right and bottom edges showing distribution patterns by day of week and hour of day respectively.

Bottom-right: Design a slope chart connecting news volume at two time points (earliest vs latest collection dates) for each publisher, overlaid with error bars representing the standard deviation of daily news volumes, and add scatter points showing individual daily totals to reveal volatility patterns.

Use the news_collection_time and publish_date columns to extract temporal features, handle missing publish_date values by using news_collection_time as fallback, and ensure all visualizations use consistent color schemes to represent different publishers throughout the grid.

## Files
bangladeshi_all_engish_newspapers_daily_news_combined_dataset.csv

-------

