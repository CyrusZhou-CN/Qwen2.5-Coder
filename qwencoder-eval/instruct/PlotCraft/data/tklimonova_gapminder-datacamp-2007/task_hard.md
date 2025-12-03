# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing global development patterns and regional clustering in the Gapminder dataset. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Continental Analysis
- Subplot 1: Combine a violin plot showing life expectancy distribution by continent with overlaid box plots and individual data points (strip plot)
- Subplot 2: Create a bubble chart showing GDP per capita vs life expectancy with bubble sizes representing population, overlaid with continent-specific regression lines
- Subplot 3: Combine stacked bar chart showing total population by continent with a line plot overlay showing average GDP per capita trends

Row 2: Temporal Evolution Patterns
- Subplot 4: Create a multi-line time series plot showing life expectancy evolution by continent from 1952-2007, combined with shaded confidence intervals and trend annotations
- Subplot 5: Combine an area chart showing cumulative population growth over time by continent with overlaid line plots showing individual country trajectories for the top 3 most populous countries
- Subplot 6: Create a slope chart connecting 1952 and 2007 GDP per capita values for each continent, combined with a scatter plot showing the relationship between initial GDP and growth rate

Row 3: Hierarchical Clustering and Network Analysis
- Subplot 7: Combine a dendrogram showing hierarchical clustering of countries based on development indicators with a heatmap showing the correlation matrix of the clustering variables
- Subplot 8: Create a parallel coordinates plot showing the multidimensional relationships between life expectancy, GDP per capita, and population for each continent, overlaid with density curves for each dimension
- Subplot 9: Combine a treemap showing countries sized by population and colored by life expectancy with an embedded network graph showing connections between countries with similar development profiles

Use the full time series data (gapminder_full.csv) for temporal analysis and 2007 data (gapminder - gapminder.csv) for cross-sectional clustering analysis.

## Files
gapminder_full.csv
gapminder - gapminder.csv

-------

