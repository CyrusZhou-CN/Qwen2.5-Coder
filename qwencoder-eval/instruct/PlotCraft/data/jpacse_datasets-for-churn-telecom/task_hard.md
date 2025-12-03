# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing customer segmentation patterns in the telecom churn dataset. Each subplot should combine multiple visualization techniques: (1) Top-left: Scatter plot with KDE contours showing MonthlyRevenue vs MonthlyMinutes, colored by Churn status with marginal histograms; (2) Top-center: Stacked bar chart with overlaid line plot showing churn rates across IncomeGroup categories, with error bars indicating confidence intervals; (3) Top-right: Violin plot combined with box plot overlay displaying MonthlyRevenue distribution across CreditRating categories, with individual data points as strip plot; (4) Middle-left: Heatmap with hierarchical clustering dendrogram showing correlation matrix of key numerical features (MonthlyRevenue, MonthlyMinutes, DroppedCalls, UnansweredCalls, CustomerCareCalls), with cluster boundaries highlighted; (5) Middle-center: Parallel coordinates plot overlaid with density curves showing customer profiles across normalized features (MonthlyRevenue, MonthsInService, Handsets, IncomeGroup), colored by churn status; (6) Middle-right: Treemap with nested pie charts showing composition of customers by ServiceArea and within each area by Occupation, sized by average MonthlyRevenue; (7) Bottom-left: Network graph with node clustering showing relationships between customers based on similar usage patterns (MonthlyMinutes, DroppedCalls, CustomerCareCalls), with nodes colored by churn probability; (8) Bottom-center: Radar chart with overlaid area plot comparing average customer profiles between churned and non-churned customers across 6 key metrics (normalized MonthlyRevenue, MonthlyMinutes, DroppedCalls, CustomerCareCalls, MonthsInService, Handsets); (9) Bottom-right: 3D scatter plot with 2D projection showing customer clusters in feature space (MonthlyRevenue, MonthlyMinutes, MonthsInService), with cluster centroids marked and convex hulls drawn around each cluster group.

## Files
cell2celltrain.csv

-------

