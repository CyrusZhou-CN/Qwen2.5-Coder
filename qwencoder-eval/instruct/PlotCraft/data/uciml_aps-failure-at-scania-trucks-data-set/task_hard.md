# Visualization Task - Hard

## Category
Groups

## Instruction
Create a complex 2x2 subplot grid analyzing the clustering patterns and relationships in the APS failure dataset. Each subplot should be a composite visualization:

Top-left: Create a scatter plot with marginal histograms showing the relationship between two key sensor variables (ag_005 and ag_006), with points colored by failure class and marginal distributions overlaid for each class.

Top-right: Generate a parallel coordinates plot combined with a density heatmap overlay, displaying the normalized values of 6-8 most variable sensor features, with lines colored by failure class to reveal multi-dimensional clustering patterns.

Bottom-left: Construct a correlation heatmap with hierarchical clustering dendrogram on both axes, focusing on the top 15 most correlated sensor variables, using a diverging colormap to highlight positive/negative correlations.

Bottom-right: Build a combined cluster analysis plot showing both a scatter plot of the first two principal components with cluster boundaries overlaid, and a silhouette analysis subplot showing the quality of clusters formed by K-means clustering (k=3-5) on the sensor data.

Use the training dataset and handle missing values appropriately. Ensure all subplots have proper titles, legends, and are well-integrated to tell a cohesive story about the natural groupings and relationships in the truck sensor data.

## Files
aps_failure_training_set.csv

-------

