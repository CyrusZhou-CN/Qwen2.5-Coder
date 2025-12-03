# Visualization Task - Hard

## Category
Correlation

## Instruction
Create a comprehensive 3x3 subplot grid analyzing the relationships between coupon acceptance and various demographic and behavioral factors. Each subplot should be a composite visualization combining multiple chart types:

Row 1: 
- Subplot 1: Scatter plot with regression line showing the relationship between age and income level (encoded numerically), with points colored by coupon acceptance status and sized by frequency of restaurant visits
- Subplot 2: Violin plot overlaid with box plot comparing coupon acceptance rates across different education levels, with separate violins for each gender
- Subplot 3: Bubble chart showing the correlation between time of day and temperature, with bubble size representing coupon acceptance rate and color indicating coupon type

Row 2:
- Subplot 4: Stacked bar chart with line overlay showing coupon acceptance rates by marital status, with bars showing count distribution and line showing acceptance percentage
- Subplot 5: Heatmap with correlation coefficients overlaid as text annotations, analyzing relationships between all numerical variables (age, temperature, distance variables, acceptance)
- Subplot 6: Scatter plot with marginal histograms showing the relationship between driving distance (toCoupon_GEQ15min) and direction alignment (direction_same), colored by acceptance status

Row 3:
- Subplot 7: Jitter plot overlaid with violin plot comparing behavioral frequency patterns (Bar, CoffeeHouse, CarryAway visits) against coupon acceptance, grouped by coupon type
- Subplot 8: Parallel coordinates plot with density overlay showing the multidimensional relationship between passenger type, destination, weather, and acceptance outcome
- Subplot 9: Network-style correlation plot with nodes representing variables and edge thickness/color representing correlation strength, focused on behavioral and contextual factors predicting acceptance

## Files
in-vehicle-coupon-recommendation.csv

-------

