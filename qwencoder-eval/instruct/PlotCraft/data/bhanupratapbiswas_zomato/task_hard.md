# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing restaurant clustering patterns across different dimensions in the Zomato dataset. Each subplot should be a composite visualization combining multiple chart types:

Top row (Location Analysis):
- Subplot 1: Combine a horizontal bar chart showing restaurant count by location with an overlaid line plot showing average rating by location
- Subplot 2: Create a scatter plot of cost vs rating colored by location, with marginal box plots on both axes
- Subplot 3: Generate a stacked bar chart showing restaurant types distribution by location with a secondary y-axis line showing total votes per location

Middle row (Cuisine Clustering):
- Subplot 4: Build a bubble chart where x-axis is average cost, y-axis is average rating, bubble size represents vote count, and color represents cuisine clusters (group similar cuisines)
- Subplot 5: Create a violin plot showing cost distribution by top 6 cuisines, overlaid with individual data points as a strip plot
- Subplot 6: Design a radar chart comparing the top 5 cuisines across multiple metrics (average rating, average cost, total restaurants, average votes) with filled areas

Bottom row (Service Feature Analysis):
- Subplot 7: Construct a grouped bar chart comparing online order vs book table availability by restaurant type, with error bars showing rating variance
- Subplot 8: Generate a heatmap showing the correlation between numerical features (cost, rating, votes) segmented by service combinations (online_order + book_table), with annotated correlation coefficients
- Subplot 9: Create a parallel coordinates plot showing the relationship between location, cost range (binned), rating range (binned), and service features, with lines colored by restaurant type

Each subplot should include appropriate titles, legends, and statistical annotations where relevant.

## Files
zomato.csv

-------

