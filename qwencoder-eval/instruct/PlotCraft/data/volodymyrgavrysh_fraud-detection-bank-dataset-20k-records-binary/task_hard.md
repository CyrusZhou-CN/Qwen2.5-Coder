# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing fraud detection patterns through feature clustering and grouping behaviors. Each subplot must be a composite visualization combining multiple chart types:

Top row (Feature Variance Analysis): (1,1) Create a histogram with overlaid KDE curve showing the distribution of unique value counts across all feature columns, with a vertical line indicating the median; (1,2) Generate a scatter plot with marginal histograms displaying the relationship between feature variance (y-axis) and unique value count (x-axis), colored by feature type groups; (1,3) Construct a box plot with overlaid violin plot showing the distribution of feature values grouped by quartiles of unique value counts.

Middle row (Feature Correlation Clustering): (2,1) Build a correlation heatmap with hierarchical clustering dendrogram on both axes, focusing on the top 20 most variable features; (2,2) Create a network graph overlaid on a scatter plot where nodes represent features and edges represent correlations above 0.7, with node sizes proportional to feature variance; (2,3) Generate a parallel coordinates plot with density curves showing the top 10 most correlated feature pairs.

Bottom row (Target-Feature Relationship Groups): (3,1) Construct a grouped bar chart with error bars showing mean feature values by target class for the top 15 most variable features; (3,2) Create a cluster plot using t-SNE dimensionality reduction with overlaid convex hulls around different feature groups, colored by target values; (3,3) Build a radar chart with multiple polygons representing different feature groups' normalized mean values, overlaid with a scatter plot of individual data points.

## Files
fraud_detection_bank_dataset.csv

-------

