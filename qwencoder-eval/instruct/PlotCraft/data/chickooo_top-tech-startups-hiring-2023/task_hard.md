# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing startup clustering patterns and hierarchical relationships. Each subplot must be a composite visualization combining multiple chart types:

Top row: (1) Scatter plot with bubble sizes representing employee count, colored by funding status (derived from tags), overlaid with cluster boundaries using K-means clustering on company size and job openings; (2) Dendrogram showing hierarchical clustering of companies based on industry similarity, with a heatmap overlay showing pairwise industry distances; (3) Network graph displaying company connections based on shared investors (from tags), with node sizes proportional to total job openings and edge thickness representing connection strength.

Middle row: (4) Parallel coordinates plot showing company profiles across multiple dimensions (employee count, job categories, location diversity), with companies grouped by cluster membership from subplot 1, overlaid with violin plots for each dimension; (5) Treemap of industries sized by total companies, with each industry cell containing a mini bar chart showing the distribution of company sizes within that industry; (6) Radar chart comparing average company profiles across different geographic regions, overlaid with individual company scatter points to show within-region variation.

Bottom row: (7) Stacked area chart showing the cumulative distribution of job types across employee size categories, with a line plot overlay showing the diversity index (number of different job types) for each size category; (8) Cluster heatmap displaying the correlation matrix between different company attributes (derived from numerical encodings of categorical variables), with hierarchical clustering dendrograms on both axes; (9) Sankey diagram showing the flow from industries to company sizes to hiring activity levels, with additional box plots showing the distribution of job openings for each flow segment.

## Files
csv_data.csv

-------

