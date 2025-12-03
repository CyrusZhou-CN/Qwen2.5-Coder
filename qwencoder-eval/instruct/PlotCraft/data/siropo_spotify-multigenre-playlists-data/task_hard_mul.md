# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x2 subplot grid analyzing musical genre clustering and audio feature relationships across different music genres. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots):
1. Left: Create a scatter plot with marginal histograms showing the relationship between energy and valence, with points colored by genre and sized by popularity. Add density contours to show clustering patterns.
2. Center: Generate a parallel coordinates plot overlaid with box plots showing the distribution of key audio features (danceability, energy, acousticness, valence, tempo normalized to 0-1 scale) across all genres, with each genre represented by a different color and transparency.
3. Right: Construct a correlation heatmap combined with a dendrogram showing hierarchical clustering of genres based on their average audio feature profiles, with correlation coefficients displayed in each cell.

Bottom row (2 subplots):
4. Left: Design a radar chart comparison overlaid with connected scatter plots showing the average audio feature profiles for each genre, with each genre as a different colored polygon and individual track points plotted as small dots.
5. Right: Build a network graph combined with a bubble chart where nodes represent genres (sized by average popularity), edges represent similarity in audio features (weighted by correlation strength), and bubble colors indicate the dominant audio feature that characterizes each genre.

Use consistent color coding across all subplots for genres, ensure proper normalization of features for comparison, and include comprehensive legends and annotations explaining the clustering patterns and relationships discovered.

## Files
alternative_music_data.csv
blues_music_data.csv
hiphop_music_data.csv
indie_alt_music_data.csv
metal_music_data.csv
pop_music_data.csv
rock_music_data.csv

-------

