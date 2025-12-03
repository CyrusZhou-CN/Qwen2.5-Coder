# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x2 subplot grid analyzing the evolution and volatility patterns of global stock markets from 2008-2023. Each subplot should be a composite visualization combining multiple chart types:

Top row (2008-2015 Crisis & Recovery Period):
- Subplot 1: Overlay line chart showing normalized closing prices of major US indices (^NYA, ^IXIC, ^DJI, ^GSPC) with a secondary y-axis displaying their combined daily volatility (calculated as rolling 30-day standard deviation of daily returns) as an area chart
- Subplot 2: Dual-axis plot combining line charts of Asian market indices (^NSEI, ^BSESN, ^N225, 000001.SS) normalized closing prices with scatter plot overlay showing daily volume spikes (volumes above 95th percentile) sized by magnitude

Middle row (2016-2019 Growth Period):
- Subplot 3: Multi-line chart of European indices (^FTSE, ^N100) and commodities (GC=F, CL=F) normalized prices with ribbon/band visualization showing the price range (High-Low) for each instrument as filled areas between the lines
- Subplot 4: Combined line and bar chart showing the correlation coefficient evolution between major global indices over time (calculated using 90-day rolling windows) as lines, with bar chart overlay displaying the average daily trading volume for the same periods

Bottom row (2020-2023 Pandemic & Recovery):
- Subplot 5: Stacked area chart showing the relative market capitalization changes of different regional markets (US, Europe, Asia) over time, overlaid with line charts showing major market drawdown periods (defined as >10% decline from recent peaks)
- Subplot 6: Heat map style visualization showing the monthly returns correlation matrix between all major indices, combined with line plots showing the evolution of market synchronization index (average pairwise correlation) over the entire period

Each subplot must include proper legends, different color schemes for different regions/asset classes, and annotations highlighting major market events (2008 crisis, COVID-19 crash, recovery periods). Use appropriate smoothing techniques where necessary and ensure all price series are normalized to enable meaningful comparison across different scales.

## Files
2008_Globla_Markets_Data.csv
2009_Globla_Markets_Data.csv
2010_Global_Markets_Data.csv
2011_Global_Markets_Data.csv
2012_Global_Markets_Data.csv
2013_Global_Markets_Data.csv
2014_Global_Markets_Data.csv
2015_Global_Markets_Data.csv
2016_Global_Markets_Data.csv
2017_Global_Markets_Data.csv
2018_Global_Markets_Data.csv
2019_Global_Markets_Data.csv
2020_Global_Markets_Data.csv
2021_Global_Markets_Data.csv
2022_Global_Markets_Data.csv
2023_Global_Markets_Data.csv

-------

