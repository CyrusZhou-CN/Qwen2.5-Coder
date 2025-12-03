# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing network attack patterns and clustering behaviors in the Bot-IoT dataset. Each subplot should be a composite visualization combining multiple chart types:

Top row (Attack Category Analysis):
- Subplot 1: Combine a stacked bar chart showing attack distribution by protocol (TCP/UDP) with an overlaid line plot showing average packet rates
- Subplot 2: Create a bubble plot where x-axis is duration, y-axis is bytes transferred, bubble size represents packet count, and color represents attack category, overlaid with density contours
- Subplot 3: Combine a violin plot showing byte distribution by attack subcategory with overlaid box plots for statistical summaries

Middle row (Network Behavior Clustering):
- Subplot 4: Create a scatter plot of source rate vs destination rate colored by attack type, with marginal histograms showing the distribution of each rate
- Subplot 5: Combine a heatmap showing correlation between network metrics (duration, packets, bytes, rates) with hierarchical clustering dendrogram on the axes
- Subplot 6: Create a parallel coordinates plot showing multiple network features (duration, packets, bytes, source rate) with lines colored by attack category

Bottom row (Temporal and Source Analysis):
- Subplot 7: Combine a time series line plot of attack frequency over time with a secondary y-axis showing a rolling average of attack severity (based on packet count)
- Subplot 8: Create a network graph visualization showing relationships between source addresses and destination addresses, with node sizes representing total traffic and edge weights representing connection frequency
- Subplot 9: Combine a radar chart showing normalized network metrics for different attack types with overlaid area fills for each category

Use data from at least 5 different CSV files to ensure comprehensive coverage of attack patterns. Apply appropriate color schemes to distinguish between Normal, DoS, DDoS, and Reconnaissance categories. Include proper legends, titles, and annotations to highlight key clustering patterns and network behaviors.

## Files
data_1.csv
data_2.csv
data_26.csv
data_37.csv
data_41.csv
data_40.csv
data_6.csv

-------

