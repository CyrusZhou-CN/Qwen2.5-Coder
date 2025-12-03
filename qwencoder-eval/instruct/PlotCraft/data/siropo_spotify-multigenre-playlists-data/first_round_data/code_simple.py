import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all datasets
datasets = {
    'Metal': pd.read_csv('metal_music_data.csv'),
    'Indie/Alt': pd.read_csv('indie_alt_music_data.csv'),
    'Alternative': pd.read_csv('alternative_music_data.csv'),
    'Blues': pd.read_csv('blues_music_data.csv'),
    'Pop': pd.read_csv('pop_music_data.csv'),
    'Hip-Hop': pd.read_csv('hiphop_music_data.csv'),
    'Rock': pd.read_csv('rock_music_data.csv')
}

# Define colors for each genre - harmonious and distinct palette
colors = {
    'Metal': '#E74C3C',      # Bold red
    'Rock': '#8E44AD',       # Purple
    'Hip-Hop': '#F39C12',    # Orange
    'Pop': '#3498DB',        # Blue
    'Alternative': '#2ECC71', # Green
    'Indie/Alt': '#1ABC9C',  # Teal
    'Blues': '#34495E'       # Dark blue-gray
}

# Create figure with subplots (3x3 grid to accommodate 7 genres)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.patch.set_facecolor('white')

# Flatten axes array for easier iteration
axes_flat = axes.flatten()

# Calculate global y-axis limit for consistent scaling
max_freq = 0
for genre, df in datasets.items():
    counts, _ = np.histogram(df['energy'], bins=30, range=(0, 1))
    max_freq = max(max_freq, max(counts))

# Create histogram for each genre in separate subplots
for i, (genre, df) in enumerate(datasets.items()):
    ax = axes_flat[i]
    
    # Create histogram with consistent styling
    ax.hist(df['energy'], bins=30, alpha=0.8, color=colors[genre], 
            edgecolor='white', linewidth=0.5, range=(0, 1))
    
    # Set title for each subplot
    ax.set_title(f'{genre}\n(n={len(df):,})', fontsize=12, fontweight='bold', pad=10)
    
    # Set consistent axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_freq * 1.05)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis ticks
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    
    # Add axis labels only to bottom row and left column
    if i >= 6:  # Bottom row
        ax.set_xlabel('Energy Level', fontsize=10, fontweight='bold')
    if i % 3 == 0:  # Left column
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')

# Hide unused subplots (we have 7 genres, so 2 subplots will be empty)
for i in range(len(datasets), len(axes_flat)):
    axes_flat[i].set_visible(False)

# Add main title
fig.suptitle('Distribution of Energy Levels Across Music Genres', 
             fontsize=18, fontweight='bold', y=0.95)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Show the plot
plt.show()