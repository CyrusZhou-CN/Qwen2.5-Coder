# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing fish species clustering patterns in Sweden. Each subplot should be a composite visualization combining multiple chart types:

Top row (Family Analysis): 
- Subplot 1: Stacked bar chart showing Red List Status distribution across the top 8 most diverse fish families, overlaid with scatter points indicating the count of species per family
- Subplot 2: Treemap of family diversity with bubble overlay showing average occurrence frequency per family
- Subplot 3: Radar chart comparing the top 6 families across multiple dimensions (species count, habitat diversity, Red List concern level) with connecting lines between families

Middle row (Habitat-Occurrence Clustering):
- Subplot 4: Clustered bar chart showing occurrence patterns by habitat type, with error bars representing Red List status variability within each cluster
- Subplot 5: Network-style scatter plot where each point represents a species, positioned by habitat (x-axis) and occurrence (y-axis), with connecting lines grouping species by family and point colors representing Red List status
- Subplot 6: Parallel coordinates plot connecting Habitat → Occurrence → Red List Status → Family, with line thickness representing species density in each pathway

Bottom row (Conservation Status Groupings):
- Subplot 7: Hierarchical clustering dendrogram of species based on their combined habitat-occurrence-family characteristics, with leaf colors representing Red List status
- Subplot 8: Grouped violin plots showing the distribution of family diversity within each Red List status category, overlaid with box plots showing quartile information
- Subplot 9: Multi-dimensional scatter plot matrix showing species clustering across three derived metrics: family rarity index, habitat specialization score, and conservation concern level, with marginal histograms on each axis

## Files
List of fishes found in Sweden.csv

-------

