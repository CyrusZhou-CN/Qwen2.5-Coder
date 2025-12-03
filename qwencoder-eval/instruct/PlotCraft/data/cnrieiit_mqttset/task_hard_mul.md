# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x2 subplot grid analyzing MQTT network security patterns across different attack types and sensor behaviors. Each subplot should be a composite visualization combining multiple chart types:

Top row (Attack Pattern Analysis):
- Subplot 1: Create a grouped bar chart showing attack frequency by type overlaid with a line plot showing attack success rates, with error bars indicating confidence intervals
- Subplot 2: Generate a radar chart comparing attack characteristics (duration, intensity, target sensors) overlaid with scatter points showing individual attack instances

Middle row (Sensor Behavior Analysis):
- Subplot 3: Develop a parallel coordinates plot showing sensor communication patterns (periodic vs random timing, room location, data profile) with density curves showing distribution of each parameter
- Subplot 4: Create a network graph visualization showing sensor-to-broker connections overlaid with a heatmap showing communication frequency between different sensor types

Bottom row (Temporal and Comparative Analysis):
- Subplot 5: Design a time series decomposition plot showing legitimate vs malicious traffic patterns over time, combined with box plots showing statistical distributions for each traffic type
- Subplot 6: Generate a cluster analysis visualization using hierarchical clustering dendrogram combined with a 2D scatter plot showing sensor groupings based on behavioral similarity, with different colors representing rooms and shapes representing timing patterns

Use the train70.csv, test30.csv, and train70_reduced.csv files to extract comprehensive insights about MQTT network security, sensor clustering patterns, and attack vector relationships.

## Files
train70.csv
test30.csv
train70_reduced.csv

-------

