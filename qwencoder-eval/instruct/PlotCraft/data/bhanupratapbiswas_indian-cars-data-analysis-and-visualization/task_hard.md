# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing Indian car market segments and their characteristics. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Brand Analysis
- Subplot 1: Combine a horizontal bar chart showing car count by Make with an overlaid scatter plot showing average Ex-Showroom Price per brand
- Subplot 2: Create a violin plot showing price distribution by Make with box plots overlaid to highlight quartiles and outliers
- Subplot 3: Generate a bubble chart where x-axis is average engine displacement, y-axis is average mileage (ARAI_Certified_Mileage), bubble size represents car count, and color represents different Makes

Row 2: Technical Specifications Clustering
- Subplot 4: Develop a parallel coordinates plot showing the relationship between Cylinders, Displacement, Power, Torque, and Mileage, with lines colored by Fuel_Type
- Subplot 5: Create a correlation heatmap overlaid with a network graph showing connections between numerical variables (Price, Displacement, Power, Torque, Mileage, Seating_Capacity)
- Subplot 6: Build a scatter plot matrix (pairplot style) for Power vs Torque, colored by Body_Type, with marginal histograms on the axes

Row 3: Market Segmentation
- Subplot 7: Construct a treemap showing hierarchical composition of Make > Body_Type > Fuel_Type with area representing count and color intensity representing average price
- Subplot 8: Design a radar chart comparing average specifications (normalized Power, Torque, Mileage, Seating_Capacity) across different Body_Types, with multiple polygons overlaid
- Subplot 9: Create a dendrogram showing hierarchical clustering of car models based on their technical specifications (Power, Torque, Displacement, Mileage), with a heatmap at the bottom showing the actual values

Each visualization should include proper titles, legends, and annotations highlighting key insights about market segmentation and clustering patterns in the Indian automotive industry.

## Files
cars_ds_final.csv

-------

