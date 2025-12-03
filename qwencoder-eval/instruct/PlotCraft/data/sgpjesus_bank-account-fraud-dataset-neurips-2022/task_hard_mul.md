# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x2 subplot grid analyzing fraud detection patterns and biases across different dataset variants. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): For each of the first three variants (Base, Variant I, Variant II), create overlaid visualizations showing: (1) A scatter plot of credit_risk_score vs income colored by fraud_bool, (2) A box plot overlay showing the distribution of credit_risk_score for each employment_status category, and (3) Marginal histograms on both axes.

Bottom row (3 subplots): For the last three variants (Variant III, Variant IV, Variant V), create composite visualizations showing: (1) A hexbin plot of velocity_6h vs velocity_24h with fraud cases highlighted as red scatter points, (2) A violin plot overlay showing session_length_in_minutes distribution by device_os, and (3) A correlation heatmap subplot showing relationships between numerical velocity features (velocity_6h, velocity_24h, velocity_4w) and risk indicators (credit_risk_score, intended_balcon_amount).

Each subplot should include proper legends, titles indicating the dataset variant, and use consistent color schemes to enable cross-variant comparison of fraud patterns and potential algorithmic biases in the synthetic datasets.

## Files
Base.csv
Variant I.csv
Variant II.csv
Variant III.csv
Variant IV.csv
Variant V.csv

-------

