# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x2 subplot grid analyzing the temporal evolution of the blockchain/cryptocurrency job market from 2018-2019. Each subplot should be a composite visualization combining multiple chart types:

Top row (2018-2019 Market Evolution):
1. Left: Combine a line chart showing monthly job posting counts over time with a stacked area chart overlay displaying the distribution of top 5 job categories (extracted from Tags column) for the same period
2. Right: Create a dual-axis plot with bar charts showing quarterly salary range medians and a line plot overlay tracking the percentage of remote jobs over time

Middle row (Company Growth Patterns):
3. Left: Develop a slope chart connecting 2018 Q4 to 2019 Q4 hiring volumes for the top 10 companies, with error bars indicating salary range variability for each company
4. Right: Construct a time series decomposition showing seasonal hiring patterns with trend lines, overlaid with scatter points sized by company funding amounts (from companies.csv)

Bottom row (Skills & Location Trends):
5. Left: Build a calendar heatmap showing daily job posting intensity throughout 2018-2019, with a secondary line chart overlay tracking the evolution of top 5 technical skills mentioned in job tags
6. Right: Create a multi-line time series showing the geographic distribution changes of job locations over time, combined with a stacked area chart showing the proportion of different experience levels (junior, senior, executive) extracted from job titles and tags

Use data from all_jobs.csv, companies.csv, and at least 3 individual company CSV files. Extract temporal information from 'Posted Before' columns, salary data from 'Salary Range', location from 'Job Location', skills from 'Tags', and company information from 'Company Name'. Handle missing data appropriately and ensure all time-based analyses show clear temporal progression patterns.

## Files
all_jobs.csv
companies.csv
Coinbase.csv
Binance.csv
Consensys.csv

-------

