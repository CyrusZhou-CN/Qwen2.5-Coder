# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing company clusters and hierarchical relationships across multiple dimensions. Each subplot must be a composite visualization combining multiple chart types:

Top row (Company Type Analysis): 
- Subplot 1: Combine a violin plot showing rating distributions by company type with overlaid box plots and individual data points as a swarm plot
- Subplot 2: Create a stacked bar chart showing employee size distribution by company type, overlaid with a line plot showing average ratings for each combination
- Subplot 3: Generate a bubble chart where x-axis is company age, y-axis is rating, bubble size represents review count, and color represents company type, with trend lines for each company type

Middle row (Geographic and Scale Clustering):
- Subplot 4: Construct a parallel coordinates plot showing the relationship between company age, rating, review count (normalized), and employee size, with lines colored by company type
- Subplot 5: Build a correlation heatmap of numerical variables (age, rating, review count) with hierarchical clustering dendrogram on both axes
- Subplot 6: Create a scatter plot matrix (pair plot) of age vs rating, age vs review count, and rating vs review count, with different colors for company types and marginal histograms

Bottom row (Performance Segmentation):
- Subplot 7: Design a radar chart comparing average metrics (rating, normalized review count, normalized age) across company types, overlaid with individual company data points
- Subplot 8: Generate a treemap showing company type composition sized by total review count, with color intensity representing average rating within each type
- Subplot 9: Create a network-style cluster plot using company age and rating as coordinates, with nodes sized by review count, colored by company type, and connected lines showing companies within similar performance clusters

## Files
company_dataset.csv

-------

