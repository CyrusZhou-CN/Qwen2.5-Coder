# Visualization Task - Hard

## Category
Composition

## Instruction
Create a comprehensive 3x3 subplot grid analyzing PC component market composition and pricing patterns. Each subplot must be a composite visualization combining multiple chart types:

Row 1: Component Market Share Analysis
- Subplot 1: Stacked bar chart showing component count distribution across all categories, overlaid with a line plot showing average price per category
- Subplot 2: Pie chart displaying market share by component type, with an inner donut chart showing price range distribution (budget/mid-range/premium)
- Subplot 3: Treemap of component categories sized by count, with color intensity representing average price levels

Row 2: Brand Competition Analysis  
- Subplot 4: Grouped bar chart comparing AMD vs Intel CPU counts by price brackets, overlaid with scatter points showing individual product prices
- Subplot 5: Stacked area chart showing cumulative price distribution for each component category, with median price lines overlaid
- Subplot 6: Bubble chart where x-axis is component category, y-axis is average price, bubble size represents product count, and color represents price variance

Row 3: Price Structure Deep Dive
- Subplot 7: Box plot showing price distributions for each component type, overlaid with violin plots to show density
- Subplot 8: Horizontal stacked bar chart showing price composition (low/medium/high tiers) for each component, with percentage annotations
- Subplot 9: Radar chart comparing normalized metrics (count, avg price, price range, market dominance) across all component categories, overlaid with a filled area chart

Extract and clean price data from all CSV files, handle currency symbols, and create meaningful price brackets. Use consistent color schemes across all subplots and ensure each composite visualization tells a cohesive story about PC component market composition.

## Files
CPU.csv
GPU.csv
MotherBoard.csv
PowerSupply.csv
RAM.csv
StorageSSD.csv
cabinates.csv
amd_cpus.csv
intel_cpus.csv

-------

