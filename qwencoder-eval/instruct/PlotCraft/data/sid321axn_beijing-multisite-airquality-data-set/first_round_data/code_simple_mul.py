import pandas as pd
import matplotlib.pyplot as plt

# Load the data
aotizhongxin = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')
tiantan = pd.read_csv('PRSA_Data_Tiantan_20130301-20170228.csv')
dongsi = pd.read_csv('PRSA_Data_Dongsi_20130301-20170228.csv')

# Plotting the histograms
plt.figure(figsize=(12, 8))

# Histogram for Aotizhongxin
plt.hist(aotizhongxin['PM2.5'], bins=range(0, 200, 10), alpha=0.7, color='blue', label='Aotizhongxin')

# Histogram for Tiantan
plt.hist(tiantan['PM2.5'], bins=range(0, 200, 10), alpha=0.7, color='green', label='Tiantan')

# Histogram for Dongsi
plt.hist(dongsi['PM2.5'], bins=range(0, 200, 10), alpha=0.7, color='red', label='Dongsi')

# Adding titles and labels
plt.title('Distribution of PM2.5 Concentrations Across Monitoring Stations in Beijing')
plt.xlabel('PM2.5 Concentration (µg/m³)')
plt.ylabel('Frequency')

# Adding legend
plt.legend()

# Show plot
plt.show()