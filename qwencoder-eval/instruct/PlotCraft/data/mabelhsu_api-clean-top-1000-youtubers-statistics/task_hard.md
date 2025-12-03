# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the relationships between YouTube channel engagement metrics. Each subplot should be a composite visualization combining multiple chart types:

Row 1: 
- Subplot 1: Scatter plot of Subscribers vs Visits with a regression line overlay, colored by Categories, plus marginal histograms showing the distribution of each variable
- Subplot 2: Bubble chart showing Subscribers vs Likes where bubble size represents Comments, with a logarithmic trend line overlay
- Subplot 3: Hexbin plot of Visits vs Likes with density contours overlaid to show concentration patterns

Row 2:
- Subplot 4: Correlation heatmap of all numerical engagement metrics (Subscribers, Visits, Likes, Comments) with annotated correlation coefficients and a diverging color scheme
- Subplot 5: Violin plot showing the distribution of Subscribers across top 5 Countries by channel count, overlaid with individual data points as a strip plot
- Subplot 6: Parallel coordinates plot showing the relationship between normalized Subscribers, Visits, Likes, and Comments, colored by Categories

Row 3:
- Subplot 7: Scatter plot matrix (pairplot) of Subscribers, Visits, and Likes with different markers for each Category and regression lines for each pair
- Subplot 8: Box plot of Likes distribution by Categories with overlaid swarm plot showing individual channel positions
- Subplot 9: Network-style correlation plot showing the strength of relationships between all engagement metrics using edge thickness and node sizes

All numerical columns (Subscribers, Visits, Likes, Comments) should be properly converted from string format with commas to integers for analysis.

## Files
Clean_Top_1000_Youtube_df - youtubers_df.csv

-------

