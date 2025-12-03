# Visualization Task - Hard

## Category
Composition

## Instruction
Create a comprehensive 3x2 subplot grid analyzing PC component market composition and pricing patterns. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): 
1. Left: Create a stacked bar chart showing component count distribution across all 7 PC component categories (CPU, GPU, MotherBoard, PowerSupply, RAM, StorageSSD, cabinates), overlaid with a line plot showing average price per category
2. Center: Design a treemap showing the hierarchical composition of total market value by component type, with a secondary pie chart inset showing percentage distribution of component counts
3. Right: Build a waffle chart displaying the proportion of Intel vs AMD processors in the CPU market, combined with box plots showing price distribution for each brand

Bottom row (2 subplots):
4. Left: Construct a stacked area chart showing cumulative price ranges (budget: <₹5000, mid-range: ₹5000-₹20000, premium: >₹20000) across different component categories, with scatter points indicating individual high-value outliers
5. Right: Create a sunburst-style nested donut chart showing the composition hierarchy: outer ring for component categories, inner ring for price segments within each category, overlaid with a radar chart showing relative market share metrics

Extract and clean price data from all component files, handle currency formatting, and ensure proper categorization. Use distinct color palettes for each component type and implement interactive-style legends.

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

