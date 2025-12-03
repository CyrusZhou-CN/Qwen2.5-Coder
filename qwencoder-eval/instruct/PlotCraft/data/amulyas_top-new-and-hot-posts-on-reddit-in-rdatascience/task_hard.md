# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing Reddit r/datascience posts clustering and engagement patterns. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Post Type Analysis
- Subplot 1: Stacked bar chart showing score distribution across post types (Top-day, New, Hot) with overlaid line plot showing average comment counts
- Subplot 2: Violin plot of score distributions by post type with overlaid box plots and scatter points for outliers
- Subplot 3: Bubble chart where x-axis is comment count, y-axis is score, bubble size represents text length, colored by post type

Row 2: Content Category Clustering  
- Subplot 4: Radar chart comparing average metrics (score, comments, text length) across top 5 flairs with overlaid line connections between categories
- Subplot 5: Hierarchical clustering dendrogram of posts based on engagement metrics (score, comments) with color-coded flair categories at leaf nodes
- Subplot 6: Parallel coordinates plot showing relationships between score, comment count, text length, and title length, with lines colored by flair category

Row 3: Engagement Pattern Analysis
- Subplot 7: 2D histogram heatmap of score vs comment count correlation with overlaid contour lines and marginal distribution plots
- Subplot 8: Network graph showing author connections (nodes sized by total score, edges weighted by similarity in posting patterns) with community detection clustering
- Subplot 9: Multi-level grouped bar chart showing score ranges (low/medium/high) by flair category with overlaid cumulative percentage line plot

Each composite subplot must reveal distinct clustering patterns and group relationships within the Reddit data science community.

## Files
reddit_datascience_newTopHot_posts.csv

-------

