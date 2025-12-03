# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing wine quality groups and their chemical characteristics. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Quality group distributions - (1,1) Histogram with KDE overlay for fixed_acidity by quality_label, (1,2) Box plot with violin plot overlay for residual_sugar by quality_label, (1,3) Stacked bar chart with line plot overlay showing alcohol content ranges by quality_label

Row 2: Correlation patterns within groups - (2,1) Scatter plot with regression lines for fixed_acidity vs density colored by quality_label with marginal histograms, (2,2) Bubble plot showing alcohol vs residual_sugar with bubble size representing density and colors representing quality_label, (2,3) Parallel coordinates plot for all numerical features with lines colored by quality_label

Row 3: Advanced group comparisons - (3,1) Radar chart comparing mean values of all numerical features across the three quality groups, (3,2) Cluster plot using PCA to reduce dimensionality and show natural groupings with quality_label annotations, (3,3) Heatmap correlation matrix for each quality group side by side with dendrograms showing hierarchical clustering of features

## Files
wine_quality_classification.csv

-------

