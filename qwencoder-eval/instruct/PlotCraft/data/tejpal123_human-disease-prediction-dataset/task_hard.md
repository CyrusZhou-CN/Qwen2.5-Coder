# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing disease symptom patterns and clustering relationships in the human disease prediction dataset. Each subplot should be a composite visualization combining multiple chart types:

Top row (Disease Overview): 
- Subplot 1: Combine a horizontal bar chart showing disease frequency with an overlaid line plot indicating average symptom count per disease
- Subplot 2: Create a stacked bar chart showing top 10 most common symptoms across all diseases, with a secondary y-axis line plot showing symptom prevalence rates
- Subplot 3: Display a pie chart of disease categories (group diseases into systems: respiratory, digestive, skin, etc.) with an adjacent donut chart showing training vs testing data distribution

Middle row (Symptom Clustering):
- Subplot 4: Generate a correlation heatmap of the top 20 most frequent symptoms with hierarchical clustering dendrogram on the side
- Subplot 5: Create a scatter plot matrix (pairplot) of 6 key symptoms with different colors for disease groups, including marginal histograms
- Subplot 6: Combine a violin plot showing symptom count distribution by disease category with overlaid box plots and individual data points

Bottom row (Disease Relationships):
- Subplot 7: Build a network graph showing disease similarity based on shared symptoms, with node sizes representing disease frequency and edge thickness showing symptom overlap
- Subplot 8: Create a parallel coordinates plot for the top 5 diseases showing their symptom profiles, with each line colored by disease type
- Subplot 9: Generate a radar chart comparing symptom profiles of 4 most common diseases, overlaid with a polar bar chart showing symptom importance scores

Use consistent color schemes across subplots, ensure proper legends, and include statistical annotations where relevant.

## Files
Training.csv
Testing.csv

-------

