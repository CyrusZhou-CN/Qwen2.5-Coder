# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the clustering patterns and relationships within Spotify's multi-genre music dataset. Each subplot should be a composite visualization combining multiple chart types:

Row 1 - Audio Feature Clustering Analysis:
- Subplot 1: Scatter plot with KDE contours showing energy vs. valence relationships across all genres, with genre-based color coding and marginal density plots
- Subplot 2: Radar chart overlaid with box plots showing the distribution ranges of key audio features (danceability, energy, acousticness, valence, speechiness) for each genre
- Subplot 3: Hierarchical clustering dendrogram combined with a heatmap showing correlation patterns between audio features, grouped by genre similarity

Row 2 - Popularity and Tempo Groupings:
- Subplot 4: Bubble chart (scatter plot with varying bubble sizes) showing tempo vs. popularity relationships, where bubble size represents duration_ms and colors represent genres, overlaid with trend lines for each genre
- Subplot 5: Violin plots combined with strip plots showing the distribution of popularity scores across genres, with embedded box plots showing quartile information
- Subplot 6: Network graph visualization showing genre relationships based on shared audio feature similarities, with nodes sized by average popularity and edges weighted by feature correlation strength

Row 3 - Advanced Pattern Recognition:
- Subplot 7: Parallel coordinates plot combined with density curves showing how tracks from different genres traverse across multiple audio features (energy, danceability, acousticness, valence, speechiness)
- Subplot 8: 2D histogram heatmap overlaid with scatter points showing loudness vs. acousticness relationships, with genre-specific contour lines and marginal histograms
- Subplot 9: Cluster analysis visualization combining K-means clustering results (scatter plot) with silhouette analysis (bar chart) to identify optimal genre groupings based on audio features

Each composite subplot should include appropriate legends, color schemes that distinguish between the 7 genres, and statistical annotations where relevant. The overall visualization should reveal hidden patterns in how different music genres cluster based on their audio characteristics and commercial success metrics.

## Files
alternative_music_data.csv
blues_music_data.csv
hiphop_music_data.csv
indie_alt_music_data.csv
metal_music_data.csv
pop_music_data.csv
rock_music_data.csv

-------

