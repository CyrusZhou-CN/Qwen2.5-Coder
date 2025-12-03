# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x2 subplot grid analyzing team clustering patterns and hierarchical relationships in the Rocket League Championship Series data. Each subplot should be a composite visualization combining multiple chart types:

Top row (Team Performance Clustering):
- Subplot 1: Scatter plot with KDE contours showing the relationship between core_score and boost_bpm, with teams colored by their winner status, overlaid with cluster boundaries using K-means clustering
- Subplot 2: Bubble chart displaying positioning_time_offensive_third vs positioning_time_defensive_third, where bubble size represents demo_inflicted, colored by team_region, with hierarchical clustering dendrogram overlay

Middle row (Player Network Analysis):
- Subplot 3: Network graph showing player connections based on shared matches, with nodes sized by advanced_rating and colored by team_id, overlaid with community detection clusters
- Subplot 4: Parallel coordinates plot for key player metrics (core_shooting_percentage, boost_avg_amount, movement_percent_supersonic_speed, positioning_percent_offensive_third) with lines grouped and colored by team clusters

Bottom row (Match Dynamics):
- Subplot 5: Heatmap correlation matrix of team performance metrics (core_goals, core_saves, boost_amount_collected, movement_time_supersonic_speed, demo_inflicted) with hierarchical clustering applied to both rows and columns, showing dendrogram on axes
- Subplot 6: Treemap visualization showing hierarchical team composition by region and performance tiers, with leaf nodes representing individual teams sized by total core_score and colored by win rate

Use consistent color schemes across subplots and include proper legends, titles, and annotations explaining the clustering methodologies used.

## Files
matches_by_teams.csv
games_by_players.csv
players_db.csv
matches_by_players.csv
games_by_teams.csv
main.csv

-------

