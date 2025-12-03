# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing corruption patterns across different dimensions. Each subplot should be a composite visualization combining multiple chart types:

Row 1: Department Analysis
- Subplot 1: Combine a horizontal bar chart showing total bribery amounts by department with overlaid scatter points indicating average views per complaint
- Subplot 2: Create a violin plot showing amount distribution by top 6 departments, overlaid with box plots to highlight quartiles
- Subplot 3: Generate a bubble chart where x-axis is department (top 8), y-axis is average amount, and bubble size represents total complaints count

Row 2: Geographic and Temporal Patterns  
- Subplot 4: Combine a stacked bar chart showing complaint counts by location with a line plot overlay showing average bribery amounts per location
- Subplot 5: Create a dual-axis time series where bars show monthly complaint counts and a line shows the trend of average bribery amounts over time
- Subplot 6: Generate a heatmap showing the relationship between location and department, with cell colors representing average bribery amounts

Row 3: Engagement and Amount Relationships
- Subplot 7: Create a scatter plot of Views vs Amount with different colors for different departments, overlaid with regression lines for top 3 departments
- Subplot 8: Combine a histogram of bribery amounts with an overlaid KDE curve, and add vertical lines showing mean and median values
- Subplot 9: Generate a parallel coordinates plot showing the relationship between normalized Amount, Views, and Department (encoded numerically), with lines colored by location clusters

## Files
data.csv

-------

