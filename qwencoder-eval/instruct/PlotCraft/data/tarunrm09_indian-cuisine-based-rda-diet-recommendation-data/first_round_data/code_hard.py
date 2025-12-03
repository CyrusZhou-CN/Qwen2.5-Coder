import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load data
df = pd.read_csv('indian_rda_based_diet_recommendation_system.csv')

# Data preprocessing
# Create meal type categories
df['Meal_Type'] = ''
df.loc[df['Breakfast'] == 1, 'Meal_Type'] = 'Breakfast'
df.loc[df['Lunch'] == 1, 'Meal_Type'] = 'Lunch'
df.loc[df['Dinner'] == 1, 'Meal_Type'] = 'Dinner'

# Handle dishes that appear in multiple meals (take the first occurrence)
df = df[df['Meal_Type'] != ''].copy()

# Create calorie categories
df['Calorie_Range'] = pd.cut(df['Calories'], bins=[0, 100, 200, float('inf')], 
                            labels=['Low (<100)', 'Medium (100-200)', 'High (>200)'])

# Set up the figure with white background and proper spacing
plt.style.use('default')
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('white')

# Create GridSpec for better control
gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.3], 
              hspace=0.4, wspace=0.3, top=0.92, bottom=0.08, left=0.05, right=0.85)

# Define professional color palettes
macro_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
micro_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
meal_colors = {'Breakfast': '#FF6B6B', 'Lunch': '#4ECDC4', 'Dinner': '#45B7D1'}

# TOP ROW: Stacked bar charts with line plots for each meal type
meal_types = ['Breakfast', 'Lunch', 'Dinner']
nutrients = ['Calories', 'Proteins', 'Fats', 'Carbohydrates']

for i, meal in enumerate(meal_types):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('white')
    
    meal_data = df[df['Meal_Type'] == meal]
    if len(meal_data) > 0:
        # Get top 8 dishes for this meal type for better visualization
        top_dishes = meal_data.nlargest(8, 'Calories')
        
        # Create stacked bar chart for each dish
        dish_names = [name[:15] + '...' if len(name) > 15 else name 
                     for name in top_dishes['Food_items']]
        x_pos = np.arange(len(dish_names))
        
        bottom = np.zeros(len(dish_names))
        bars = []
        
        for j, nutrient in enumerate(nutrients):
            values = top_dishes[nutrient].values
            bars.append(ax.bar(x_pos, values, bottom=bottom, 
                              color=macro_colors[j], alpha=0.8, 
                              label=nutrient if i == 0 else ""))
            bottom += values
        
        # Add fiber trend line
        ax2 = ax.twinx()
        fiber_values = top_dishes['Fibre'].values
        ax2.plot(x_pos, fiber_values, 'o-', linewidth=3, markersize=6, 
                color='darkred', label='Fiber Content')
        ax2.set_ylabel('Fiber (g)', fontweight='bold', color='darkred', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.grid(True, alpha=0.3)
        
        # Formatting
        ax.set_title(f'{meal} Nutritional Composition', fontweight='bold', fontsize=14)
        ax.set_ylabel('Nutritional Content', fontweight='bold', fontsize=11)
        ax.set_xlabel('Food Items', fontweight='bold', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dish_names, rotation=45, ha='right', fontsize=9)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)

# MIDDLE ROW: Treemap visualizations with pie chart insets
veg_categories = ['Vegetarian', 'Non-Vegetarian']
for i, veg_type in enumerate([0, 1]):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('white')
    
    veg_data = df[df['VegNovVeg'] == veg_type]
    if len(veg_data) > 0:
        # Macro-nutrients for treemap
        macros = ['Proteins', 'Fats', 'Carbohydrates']
        macro_totals = veg_data[macros].sum()
        
        # Create treemap-style rectangles
        total = macro_totals.sum()
        if total > 0:
            # Calculate rectangle dimensions for treemap layout
            sorted_macros = macro_totals.sort_values(ascending=False)
            
            # Create hierarchical rectangles
            y_pos = 0
            height_remaining = 1
            
            for j, (macro, value) in enumerate(sorted_macros.items()):
                height = (value / total) * height_remaining
                
                # Create main rectangle
                rect = Rectangle((0, y_pos), 0.7, height, 
                               facecolor=macro_colors[macros.index(macro)], 
                               alpha=0.7, edgecolor='white', linewidth=3)
                ax.add_patch(rect)
                
                # Add text labels
                ax.text(0.35, y_pos + height/2, f'{macro}\n{value:.1f}g', 
                       ha='center', va='center', fontweight='bold', 
                       fontsize=11, color='white')
                
                y_pos += height
                height_remaining -= height
        
        # Add pie chart inset for micro-nutrients
        micros = ['Iron', 'Calcium', 'Sodium', 'Potassium']
        micro_totals = veg_data[micros].sum()
        
        # Create inset axes for pie chart
        inset_ax = ax.inset_axes([0.72, 0.55, 0.45, 0.45])
        if micro_totals.sum() > 0:
            wedges, texts, autotexts = inset_ax.pie(micro_totals, labels=micros, 
                                                   colors=micro_colors, autopct='%1.1f%%',
                                                   textprops={'fontsize': 8, 'fontweight': 'bold'})
            inset_ax.set_title('Micro-nutrients', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.set_title(f'{veg_categories[i]} Dishes\nMacro & Micro Nutrients', 
                fontweight='bold', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Third treemap for combined analysis
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor('white')

# Overall macro distribution
overall_macros = df[['Proteins', 'Fats', 'Carbohydrates']].sum()
total_overall = overall_macros.sum()

if total_overall > 0:
    # Create combined treemap
    sorted_overall = overall_macros.sort_values(ascending=False)
    
    x_pos = 0
    width_remaining = 1
    
    for j, (macro, value) in enumerate(sorted_overall.items()):
        width = (value / total_overall) * width_remaining
        
        rect = Rectangle((x_pos, 0), width, 1, 
                       facecolor=macro_colors[['Proteins', 'Fats', 'Carbohydrates'].index(macro)], 
                       alpha=0.7, edgecolor='white', linewidth=3)
        ax.add_patch(rect)
        
        ax.text(x_pos + width/2, 0.5, f'{macro}\n{value:.1f}g', 
               ha='center', va='center', fontweight='bold', 
               fontsize=11, color='white')
        
        x_pos += width
        width_remaining -= width

# Add overall micro-nutrients pie chart
overall_micros = df[['Iron', 'Calcium', 'Sodium', 'Potassium']].sum()
inset_ax = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
if overall_micros.sum() > 0:
    wedges, texts = inset_ax.pie(overall_micros, labels=['Fe', 'Ca', 'Na', 'K'], 
                               colors=micro_colors, textprops={'fontsize': 8, 'fontweight': 'bold'})

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Overall Distribution\nAll Dishes Combined', fontweight='bold', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# BOTTOM ROW: Waffle charts with violin plot overlays
calorie_ranges = ['Low (<100)', 'Medium (100-200)', 'High (>200)']

for i, cal_range in enumerate(calorie_ranges):
    ax = fig.add_subplot(gs[2, i])
    ax.set_facecolor('white')
    
    range_data = df[df['Calorie_Range'] == cal_range]
    if len(range_data) > 0:
        # Create waffle chart for sugar content (10x10 grid)
        grid_size = 10
        sugar_values = range_data['Sugars'].values
        
        if len(sugar_values) > 0:
            # Normalize sugar values for color mapping
            if sugar_values.max() > sugar_values.min():
                sugar_normalized = (sugar_values - sugar_values.min()) / (sugar_values.max() - sugar_values.min())
            else:
                sugar_normalized = np.ones(len(sugar_values)) * 0.5
            
            # Create waffle squares
            total_squares = grid_size * grid_size
            squares_per_item = max(1, total_squares // len(sugar_values))
            
            square_idx = 0
            for j, sugar_norm in enumerate(sugar_normalized):
                color_intensity = sugar_norm
                color = plt.cm.Reds(0.3 + 0.7 * color_intensity)
                
                # Fill squares for this item
                for _ in range(min(squares_per_item, total_squares - square_idx)):
                    row = square_idx // grid_size
                    col = square_idx % grid_size
                    
                    rect = Rectangle((col, row), 1, 1, facecolor=color, 
                                   edgecolor='white', linewidth=0.5, alpha=0.8)
                    ax.add_patch(rect)
                    square_idx += 1
                    
                    if square_idx >= total_squares:
                        break
                
                if square_idx >= total_squares:
                    break
        
        # Overlay violin plots for VitaminD and Fiber
        if len(range_data) > 1:
            vit_d = range_data['VitaminD'].values
            fiber = range_data['Fibre'].values
            
            # Create secondary axis for violin plots
            ax2 = ax.twinx()
            
            # Position violin plots
            positions = [grid_size * 0.25, grid_size * 0.75]
            violin_data = [vit_d, fiber]
            labels = ['Vitamin D', 'Fiber']
            colors_violin = ['#4169E1', '#228B22']
            
            # Create violin plots
            parts = ax2.violinplot(violin_data, positions=positions, 
                                 widths=grid_size*0.2, showmeans=True)
            
            # Style violin plots
            for pc, color in zip(parts['bodies'], colors_violin):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style other violin plot elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1)
            
            ax2.set_ylabel('Vitamin D & Fiber Content', fontweight='bold', fontsize=11)
            max_val = max(max(vit_d) if len(vit_d) > 0 else 1, 
                         max(fiber) if len(fiber) > 0 else 1)
            ax2.set_ylim(0, max_val * 1.2)
            
            # Add legend for violin plots
            violin_legend = [plt.Line2D([0], [0], color=colors_violin[0], lw=4, alpha=0.6, label='Vitamin D'),
                           plt.Line2D([0], [0], color=colors_violin[1], lw=4, alpha=0.6, label='Fiber')]
            ax2.legend(handles=violin_legend, loc='upper right', fontsize=9)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title(f'{cal_range} Calories\nSugar Distribution (Waffle Chart)', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Sugar Content Intensity →', fontweight='bold', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Add comprehensive legend in the right column
legend_ax = fig.add_subplot(gs[:, 3])
legend_ax.set_facecolor('white')
legend_ax.axis('off')

# Create legend elements
macro_patches = [mpatches.Patch(color=macro_colors[i], label=nutrient) 
                for i, nutrient in enumerate(['Calories', 'Proteins', 'Fats', 'Carbohydrates'])]

micro_patches = [mpatches.Patch(color=micro_colors[i], label=nutrient) 
                for i, nutrient in enumerate(['Iron', 'Calcium', 'Sodium', 'Potassium'])]

# Add legends with proper spacing
legend_ax.legend(handles=macro_patches, title='Macronutrients', 
                loc='upper left', bbox_to_anchor=(0, 1), fontsize=11, 
                title_fontsize=12, frameon=True, fancybox=True, shadow=True)

legend_ax.legend(handles=micro_patches, title='Micronutrients', 
                loc='upper left', bbox_to_anchor=(0, 0.6), fontsize=11, 
                title_fontsize=12, frameon=True, fancybox=True, shadow=True)

# Add color bar for sugar intensity
sugar_legend = [mpatches.Patch(color=plt.cm.Reds(0.3 + 0.7 * i/4), 
                              label=f'Sugar Level {i+1}') for i in range(5)]
legend_ax.legend(handles=sugar_legend, title='Sugar Intensity\n(Waffle Charts)', 
                loc='upper left', bbox_to_anchor=(0, 0.2), fontsize=10, 
                title_fontsize=11, frameon=True, fancybox=True, shadow=True)

# Add overall title with proper spacing
fig.suptitle('Comprehensive Nutritional Analysis of Indian Dishes\nAcross Meal Types and Dietary Preferences', 
             fontsize=18, fontweight='bold', y=0.96)

# Final layout adjustment
plt.tight_layout()

# Save the plot
plt.savefig('nutritional_composition_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()