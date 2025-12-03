# Visualization Task - Hard

## Category
Groups

## Instruction
Create a comprehensive 3x3 subplot grid analyzing money laundering patterns and account clustering in the AML dataset. Each subplot should combine multiple visualization techniques:

Row 1: Account Analysis
- Subplot 1: Combine a scatter plot of account initial balance vs fraud status with overlaid box plots showing balance distribution by fraud category
- Subplot 2: Create a network graph showing account relationships through transactions, with nodes colored by fraud status and sized by transaction volume
- Subplot 3: Overlay a histogram and KDE curve showing the distribution of account balances, with separate curves for fraudulent and non-fraudulent accounts

Row 2: Transaction Flow Analysis  
- Subplot 4: Combine a bubble chart (sender vs receiver accounts) where bubble size represents transaction amount, overlaid with a heatmap showing transaction density
- Subplot 5: Create a parallel coordinates plot showing the relationship between sender account, receiver account, transaction amount, and fraud status
- Subplot 6: Overlay violin plots and strip plots showing transaction amount distributions across different alert types

Row 3: Alert Pattern Investigation
- Subplot 7: Combine a treemap showing alert type composition with embedded bar charts showing fraud distribution within each alert type
- Subplot 8: Create a cluster analysis plot using transaction amounts and timestamps, with points colored by alert type and shaped by fraud status
- Subplot 9: Overlay a correlation heatmap of numerical variables with a dendrogram showing hierarchical clustering of accounts based on transaction patterns

## Files
accounts.csv
transactions.csv
alerts.csv

-------

