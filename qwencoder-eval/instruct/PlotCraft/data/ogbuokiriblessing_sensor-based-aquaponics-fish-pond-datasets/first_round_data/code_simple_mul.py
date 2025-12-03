import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
pond6 = pd.read_csv('IoTPond6.csv')
pond3 = pd.read_csv('IoTPond3.csv')
pond9 = pd.read_csv('IoTPond9.csv')

# Extract the pH values
pH_pond6 = pond6['PH']
pH_pond3 = pond3['PH']
pH_pond9 = pond9['PH']

# Plotting the histograms
plt.figure(figsize=(12, 8))
plt.hist(pH_pond6, bins=20, alpha=0.5, label='IoTPond6', color='blue')
plt.hist(pH_pond3, bins=20, alpha=0.5, label='IoTPond3', color='green')
plt.hist(pH_pond9, bins=20, alpha=0.5, label='IoTPond9', color='red')

# Adding titles and labels
plt.title('Distribution of pH Levels Across Different Aquaponics Fish Ponds')
plt.xlabel('pH Level')
plt.ylabel('Frequency')

# Adding legend
plt.legend()

# Show the plot
plt.show()