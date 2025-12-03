# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the complex relationships between EMG electrode signals and cyberglove sensor readings in the Ninapro DB1 dataset. Each subplot must be a composite visualization combining multiple chart types:

Row 1: EMG Signal Analysis
- Subplot 1: Scatter plot with regression line showing correlation between emg_5 and emg_6, overlaid with marginal histograms
- Subplot 2: Scatter plot with regression line for emg_7 vs emg_9 correlation, overlaid with marginal density plots
- Subplot 3: Bubble plot showing emg_0, emg_4, and emg_8 relationships (use emg_8 as bubble size), with trend line

Row 2: Cyberglove Sensor Correlations
- Subplot 4: Scatter plot with regression line for glove_10 vs glove_19 correlation, overlaid with confidence intervals
- Subplot 5: Correlation heatmap of all glove sensors (glove_0 through glove_21) with hierarchical clustering dendrogram
- Subplot 6: Pairwise scatter plot matrix for glove_10, glove_19, and two other most variable glove sensors

Row 3: Cross-Modal EMG-Glove Relationships
- Subplot 7: Scatter plot showing correlation between most variable EMG channel and most variable glove sensor, with polynomial fit line and residual plot overlay
- Subplot 8: Network graph visualization showing correlation strengths between top 5 EMG channels and top 5 glove sensors (edge thickness represents correlation strength)
- Subplot 9: 2D histogram/heatmap showing joint distribution of the strongest EMG-glove correlation pair, overlaid with contour lines

Each subplot must include appropriate statistical annotations (correlation coefficients, p-values where applicable) and use distinct color schemes to differentiate between EMG and glove data modalities.

## Files
Ninapro_DB1.csv

-------

