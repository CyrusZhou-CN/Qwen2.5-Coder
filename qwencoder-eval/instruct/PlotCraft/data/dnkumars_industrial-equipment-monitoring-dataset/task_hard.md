# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the relationships between sensor measurements (temperature, pressure, vibration, humidity) and equipment fault status across different equipment types and locations. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Equipment-based analysis
- Subplot (0,0): Scatter plot with regression lines showing temperature vs pressure relationship, colored by equipment type, with marginal histograms
- Subplot (0,1): Bubble plot showing vibration vs humidity correlation where bubble size represents temperature and color represents fault status, overlaid with equipment type annotations
- Subplot (0,2): Correlation heatmap of all sensor measurements segmented by equipment type using subplots within the heatmap

Row 2: Location-based analysis  
- Subplot (1,0): Multi-dimensional scatter plot matrix showing pairwise correlations between temperature, pressure, and vibration, with different markers for each location
- Subplot (1,1): Violin plots showing distribution of each sensor measurement by location, overlaid with box plots and individual data points
- Subplot (1,2): Radar chart comparing average sensor readings across locations, with separate polygons for faulty vs non-faulty equipment

Row 3: Fault detection analysis
- Subplot (2,0): Parallel coordinates plot showing all sensor measurements with lines colored by fault status and styled by equipment type
- Subplot (2,1): 2D density contour plot of temperature vs vibration with scatter points, separated into faulty vs non-faulty equipment with different color schemes
- Subplot (2,2): Combined correlation matrix and dendrogram showing hierarchical clustering of equipment instances based on sensor similarity, with fault status annotations

## Files
equipment_anomaly_data.csv

-------

