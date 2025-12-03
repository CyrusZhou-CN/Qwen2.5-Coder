# Visualization Task - Hard

## Category
Change

## Instruction
Create a comprehensive 3x2 subplot grid analyzing the temporal evolution of Ukraine war discourse on Twitter. Each subplot should be a composite visualization combining multiple chart types:

Top row (3 subplots): For each subplot, create a dual-axis time series where the primary y-axis shows daily tweet volume as a line chart, and the secondary y-axis displays engagement metrics (likes, retweets, replies) as stacked area charts. Use different search term datasets for each subplot: (1) "Ukraine war" and "StandWithUkraine", (2) "Russian troops" and "Ukraine troops", (3) "Ukraine border" and "Ukraine NATO".

Bottom row (3 subplots): For each subplot, create a combination of time series line plots overlaid with scatter plots showing sentiment indicators. The line plots should track the daily average of engagement metrics over time, while scatter plots should show individual high-engagement tweets (size based on total engagement). Use the remaining datasets: (4) "Russia invade" and "Russian border Ukraine", (5) All datasets combined showing overall discourse evolution, (6) A comparative analysis showing the relationship between different search terms' daily volumes with trend lines.

Each subplot must extract datetime information from the 'date' column, aggregate data by day, and show how the conversation evolved over the 65-day period. Include proper legends, axis labels, and use distinct color schemes for each search term to track the changing narrative dynamics of the Ukraine war on social media.

## Files
Ukraine_war.csv
Ukraine_border.csv
Russian_border_Ukraine.csv
Ukraine_troops.csv
Russia_invade.csv
Russian_troops.csv
StandWithUkraine.csv
Ukraine_nato.csv

-------

