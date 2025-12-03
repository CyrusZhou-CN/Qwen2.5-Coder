# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing team clustering patterns and hierarchical relationships in the Rocket League Championship Series data. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Team Performance Clustering Analysis
- Subplot 1: Scatter plot with KDE contours showing the relationship between core_score and core_shooting_percentage, with team clusters identified by different colors and sizes representing total boost collected
- Subplot 2: Hierarchical clustering dendrogram overlaid with a heatmap showing correlations between key performance metrics (shots, goals, saves, assists) for all teams
- Subplot 3: Network graph showing team relationships based on performance similarity, with nodes sized by match wins and edges weighted by performance correlation strength

Row 2: Positional and Movement Group Analysis  
- Subplot 4: Parallel coordinates plot combined with box plots showing team positioning patterns across defensive_third, neutral_third, and offensive_third time percentages
- Subplot 5: Cluster scatter plot of movement patterns (time_supersonic_speed vs time_ground) with overlaid violin plots showing distribution density for each identified cluster
- Subplot 6: Radar chart overlay showing team archetypes based on boost management metrics (bpm, avg_amount, time_zero_boost) with background heatmap of cluster membership probabilities

Row 3: Advanced Team Grouping and Relationships
- Subplot 7: Treemap visualization combined with connected scatter plot showing team hierarchies based on advanced_rating, with connection lines indicating similar playstyles
- Subplot 8: Multi-dimensional scaling (MDS) plot with confidence ellipses for team groups, overlaid with arrow vectors showing the direction of key performance drivers
- Subplot 9: Sankey diagram combined with grouped bar chart showing the flow of teams between performance tiers and their corresponding demo statistics (inflicted vs taken)

## Files
matches_by_teams.csv
games_by_teams.csv
matches_by_players.csv
games_by_players.csv
players_db.csv
main.csv

-------

