# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the clustering patterns and group relationships in the Hippocorpus dataset. Each subplot should be a composite visualization combining multiple chart types:

Top row: (1) A scatter plot with marginal histograms showing the relationship between openness and importance, colored by memType, with density contours overlaid. (2) A violin plot combined with strip plot showing the distribution of logTimeSinceEvent across different annotatorAge groups, with box plots overlaid inside each violin. (3) A stacked bar chart showing annotatorRace composition within each annotatorGender category, with percentage labels and a line plot overlay showing the total count per gender.

Middle row: (4) A radar chart comparing the mean values of all Likert scale variables (distracted, draining, frequency, importance, similarity, stressful) across the three memType categories, with filled areas and individual data points plotted. (5) A parallel coordinates plot showing the relationships between WorkTimeInSeconds, openness, and all Likert scale variables, with lines colored by memType and transparency based on density. (6) A cluster heatmap showing the correlation matrix between all numerical variables, with hierarchical clustering dendrograms on both axes and annotated correlation values.

Bottom row: (7) A combination of grouped bar chart and scatter plot showing mean WorkTimeInSeconds by annotatorAge groups (bars) with individual data points overlaid, separated by memType using different colors and markers. (8) A treemap showing the hierarchical composition of stories by annotatorRace and annotatorGender, with area proportional to count and color intensity representing mean openness scores. (9) A network-style plot showing the relationships between recAgnPairId and recImgPairId connections, with nodes sized by frequency and colored by the dominant memType in each pair.

## Files
hippoCorpusV2.csv

-------

