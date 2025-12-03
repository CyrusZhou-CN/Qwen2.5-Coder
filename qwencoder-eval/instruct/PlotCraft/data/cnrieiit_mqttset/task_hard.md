# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing MQTT network security patterns across different attack types and sensor behaviors. Each subplot should be a composite visualization combining multiple chart types:

Top row (Attack Pattern Analysis): 
- Subplot 1: Combine a stacked bar chart showing attack frequency by type with an overlaid line plot showing attack intensity over time
- Subplot 2: Create a bubble plot showing relationship between packet size and frequency, with bubble colors representing different attack types, overlaid with density contours
- Subplot 3: Display a radar chart comparing attack characteristics across multiple dimensions, combined with a parallel coordinates plot showing attack signatures

Middle row (Sensor Network Behavior):
- Subplot 4: Combine a violin plot showing data distribution patterns for each sensor type with overlaid box plots highlighting outliers and quartiles
- Subplot 5: Create a network graph showing sensor connectivity patterns overlaid with a heatmap showing communication intensity between sensors
- Subplot 6: Display a time series decomposition plot showing periodic vs random sensor behavior patterns, combined with autocorrelation plots

Bottom row (Security Classification):
- Subplot 7: Combine a confusion matrix heatmap for attack classification with marginal bar charts showing precision/recall metrics
- Subplot 8: Create a cluster analysis plot using t-SNE or PCA for dimensionality reduction, overlaid with convex hulls showing different attack clusters
- Subplot 9: Display a hierarchical clustering dendrogram combined with a treemap showing the composition of different traffic types in the final dataset

Use consistent color schemes across all subplots to represent the same categories, and ensure each composite visualization reveals distinct insights about MQTT network security patterns, sensor behaviors, and attack classification performance.

## Files
train70.csv
test30.csv
train70_reduced.csv
test30_reduced.csv
train70_augmented.csv
test30_augmented.csv

-------

