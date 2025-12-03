# Visualization Task - Middle

## Category
Correlation

## Instruction
Create a comprehensive correlation analysis visualization that explores the relationship between insider trading patterns and stock performance metrics across multiple S&P 500 companies. Your visualization should consist of a 2x2 subplot grid where each subplot combines multiple chart types:

1. **Top-left subplot**: Create a scatter plot with marginal histograms showing the correlation between total transaction value (sum of all transactions per company) and the number of unique insiders per company. Color-code points by the dominant transaction type (Buy/Sale/Option Exercise) for each company.

2. **Top-right subplot**: Generate a correlation heatmap overlaid with a network-style connection plot showing the relationships between key variables: average transaction cost, total shares traded, number of transactions, and insider relationship diversity (count of unique relationship types per company).

3. **Bottom-left subplot**: Construct a bubble plot where x-axis represents the average days between transactions, y-axis shows the concentration ratio of transactions (percentage of total value from top 3 insiders), and bubble size indicates total transaction volume. Add trend lines for different company sectors if identifiable from ticker patterns.

4. **Bottom-right subplot**: Design a multi-layered scatter plot with best-fit lines showing the correlation between insider seniority level (categorized as C-level, VP-level, Director, Other) and transaction timing patterns, with separate trend lines for each seniority category and confidence intervals.

Use data from at least 15 different company files to ensure statistical significance. Include proper legends, color coding, and annotations to highlight the strongest correlations discovered.

## Files
NEE.csv
EPAM.csv
EMR.csv
PSA.csv
CSX.csv
MTD.csv
GPC.csv
ALB.csv
USB.csv
AXP.csv
HSY.csv
PLD.csv
FCX.csv
AEE.csv
JNJ.csv
WMT.csv
FBHS.csv
NDSN.csv
KEYS.csv
PEG.csv

-------

