# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 2x2 subplot grid analyzing Udemy course clustering patterns and hierarchical relationships. Each subplot must be a composite visualization:

Top-left: Create a scatter plot showing the relationship between number of subscribers and rating, with points colored by instructional level and sized by number of reviews. Overlay this with cluster boundaries using K-means clustering (k=3) to identify natural course groupings.

Top-right: Build a hierarchical clustering dendrogram based on normalized features (subscribers, rating, reviews) combined with a heatmap showing the correlation matrix of these numerical variables positioned below the dendrogram.

Bottom-left: Construct a parallel coordinates plot displaying the relationships between subscribers, rating, and reviews, with lines colored by instructional level. Add a secondary y-axis showing the distribution of courses across instructional levels as a bar chart overlay.

Bottom-right: Generate a network-style visualization showing instructor relationships (nodes sized by total subscribers across their courses, edges representing shared instructional levels) combined with a radial bar chart around the perimeter showing the top 10 instructors by average course rating.

## Files
udemy_courses.csv

-------

