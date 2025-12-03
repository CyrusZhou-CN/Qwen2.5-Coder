# Visualization Task - Hard

## Category
Composition

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the composition and distribution of Named Entity Recognition (NER) tags across the corpus. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): For each of the three most frequent entity types (excluding 'O' tags), create a stacked bar chart showing the distribution of B- (Beginning) vs I- (Inside) tags overlaid with a line plot showing the cumulative percentage contribution across sentences.

Middle row (3 subplots): Create pie charts with donut holes showing the proportion of each entity type, but overlay each with a bar chart around the circumference showing the average sentence length (number of words) for sentences containing each entity type.

Bottom row (3 subplots): For the three entity types with the highest variability in occurrence per sentence, create treemap visualizations showing the hierarchical breakdown of entity frequency, overlaid with scatter plots showing the relationship between sentence position and entity density for that type.

The visualization should reveal how different entity types are composed within the corpus, their relative importance, and their distributional patterns across the dataset.

## Files
ner.csv

-------

