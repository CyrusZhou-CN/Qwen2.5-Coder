# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing UFC fighter performance clusters and weight class dynamics. Each subplot should be a composite visualization combining multiple chart types:

Top row (Fighter Performance Clustering):
1. Scatter plot with KDE contours showing Fighter_1_STR vs Fighter_1_KD, colored by weight class, with marginal histograms
2. Bubble chart showing Fighter_1_TD vs Fighter_1_SUB where bubble size represents total strikes, overlaid with a regression line and confidence intervals
3. Violin plot showing distribution of significant strikes by weight class, overlaid with box plots and individual data points

Middle row (Method and Outcome Analysis):
4. Stacked bar chart showing fight methods by weight class, with a secondary line plot showing average round duration
5. Heatmap of weight class vs method combinations with annotated percentages, overlaid with hierarchical clustering dendrogram
6. Radar chart comparing average performance metrics (KD, STR, TD, SUB) across top 5 most frequent weight classes, with error bars

Bottom row (Temporal and Location Patterns):
7. Time series line plot showing monthly fight frequency over time, with separate lines for each location, including trend lines and seasonal decomposition
8. Network graph showing connections between locations and weight classes based on fight frequency, with node sizes representing total fights
9. Parallel coordinates plot showing the relationship between all numerical fighter statistics, grouped and colored by weight class clusters

## Files
ufc.csv

-------

