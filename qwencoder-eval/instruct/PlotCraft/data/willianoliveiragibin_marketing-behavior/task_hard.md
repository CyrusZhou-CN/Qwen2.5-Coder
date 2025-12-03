# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the correlations between marketing engagement metrics and purchase behavior. Each subplot should be a composite visualization combining multiple chart types:

Row 1: 
- Subplot 1: Scatter plot of Likes vs Purchase_Likelihood with overlaid regression line and confidence intervals
- Subplot 2: Scatter plot of Shares vs Purchase_Likelihood with overlaid regression line and marginal histograms
- Subplot 3: Scatter plot of Comments vs Purchase_Likelihood with overlaid regression line and density contours

Row 2:
- Subplot 4: Scatter plot of Clicks vs Time_Spent_on_Platform with color-coded Purchase_History and overlaid trend line
- Subplot 5: Scatter plot of Engagement_with_Ads vs Purchase_Likelihood with bubble sizes representing Time_Spent_on_Platform and overlaid regression line
- Subplot 6: Correlation heatmap of all numerical engagement metrics with Purchase_Likelihood, overlaid with correlation coefficient annotations

Row 3:
- Subplot 7: Scatter plot matrix (pairplot style) showing Likes, Shares, Comments relationships with Purchase_History color coding
- Subplot 8: Scatter plot of brand vs buy metrics with Purchase_Likelihood color coding and overlaid trend analysis
- Subplot 9: Combined violin plot and strip plot showing the distribution of all engagement metrics grouped by Purchase_Likelihood

Note: Convert string numerical columns (Likes, Shares, Comments, Clicks, etc.) to proper float format by replacing commas with dots before analysis.

## Files
marketing_data new.csv

-------

