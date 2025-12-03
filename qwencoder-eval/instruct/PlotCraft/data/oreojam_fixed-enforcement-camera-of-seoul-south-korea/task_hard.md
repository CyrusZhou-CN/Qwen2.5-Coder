# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the spatial distribution and clustering patterns of Seoul's fixed enforcement cameras. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): District-level analysis
- Subplot 1: Combine a horizontal bar chart showing camera count per district with overlaid population density scatter points
- Subplot 2: Create a bubble chart where bubble size represents camera density (cameras per km²) and color intensity shows population, with district names as labels
- Subplot 3: Overlay a violin plot showing latitude distribution of cameras with a strip plot showing individual camera positions, grouped by district

Middle row (3 subplots): Geographic clustering analysis  
- Subplot 4: Create a scatter plot of camera locations (lat/lon) with density contours overlaid, colored by district
- Subplot 5: Combine a 2D histogram heatmap of camera locations with marginal histograms showing latitude and longitude distributions
- Subplot 6: Create a network-style plot showing camera clusters using hierarchical clustering, with different colors for each identified cluster

Bottom row (3 subplots): Comparative district analysis
- Subplot 7: Create a radar chart comparing top 5 districts by camera count across multiple metrics (camera count, population, area, camera density) with overlaid line plots
- Subplot 8: Combine box plots showing camera coordinate distributions by district with overlaid swarm plots of individual cameras
- Subplot 9: Create a correlation matrix heatmap of district metrics (population, area, camera count, camera density) with scatter plots in the lower triangle

Use consistent color schemes across related subplots and ensure each composite visualization reveals different aspects of camera clustering and district-level patterns.

## Files
district_of_seoul.csv
fixed_cctv_for_parking_enforcement.csv

-------

