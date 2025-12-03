import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
metal_data = pd.read_csv('metal_music_data.csv')
indie_alt_data = pd.read_csv('indie_alt_music_data.csv')
alternative_data = pd.read_csv('alternative_music_data.csv')
blues_data = pd.read_csv('blues_music_data.csv')
pop_data = pd.read_csv('pop_music_data.csv')
hiphop_data = pd.read_csv('hiphop_music_data.csv')
rock_data = pd.read_csv('rock_music_data.csv')

# Combine the datasets into one DataFrame
data = pd.concat([metal_data, indie_alt_data, alternative_data, blues_data, pop_data, hiphop_data, rock_data], ignore_index=True)

# Extract the energy levels and genres
energies = data['energy']
genres = data['Genres'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Define colors for each genre
genre_colors = {
    'metal': '#FF4500',
    'indie alt': '#ADD8E6',
    'alternative': '#FFD700',
    'blues': '#00008B',
    'pop': '#FF69B4',
    'hip hop': '#800080',
    'rock': '#2F4F4F'
}

# Plotting the histogram with different colors for each genre
plt.figure(figsize=(12, 8))
for genre, color in genre_colors.items():
    subset = energies[genres == genre]
    plt.hist(subset, bins=30, alpha=0.6, label=genre, color=color, edgecolor='black')

plt.title('Distribution of Energy Levels Across Music Genres')
plt.xlabel('Energy Level')
plt.ylabel('Frequency')
plt.legend(title='Genre')
plt.grid(True)
plt.show()