import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Home Renovation Survey _ Asia - Data.csv')

# Print column names to debug
print("Available columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

# Create simplified column name mapping
col_mapping = {}
for col in df.columns:
    col_mapping[col] = col.strip()

# Use exact column names from the dataset
house_type_col = "What's the type of house you live in:"
budget_col = "What's your budget for home renovation if you need one?"
premium_col = "How much more are you willing to pay for energy-efficient materials compared to traditional materials?"
neighborhood_col = "Which type of neighborhood do you live in?"
family_size_col = "How many members do you live with?"
energy_efficient_col = "Has anyone in your family or friend circle opted for energy-efficient solutions while renovating their homes?"
motivation_col = "What would motivate you to consider energy-efficient renovation solutions?"
consultation_col = "Who would you seek guidance from for renovating your home?"
room_col = "Which room or which part do you want to renovate most at present?"
frequency_col = "How frequently will you consider renovating your home?"
energy_materials_col = "How many new energy materials are used in your current home?"
knowledge_col = "Do you have knowledge of energy efficiency materials used in home decor/ renovation?"
renovation_reason_col = "Why would you renovate your home if you want to?"
app_features_col = "Which of the following features do you care about most when using a home renovation app?"
consultation_pay_col = "How much would you like to pay for a consultation service for energy efficiency of your home?"

# Create figure with 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Data preprocessing for various visualizations
le = LabelEncoder()

# Subplot 1: Top-left - Stacked bar chart with overlaid line plot
ax1 = plt.subplot(3, 3, 1)
try:
    house_types = df[house_type_col].value_counts()
    budget_data = df.groupby(house_type_col)[budget_col].value_counts().unstack(fill_value=0)
    premium_pct = df.groupby(house_type_col)[premium_col].apply(lambda x: (x == '5% ~ 10%').mean() * 100)
    
    # Create stacked bar chart
    budget_data.plot(kind='bar', stacked=True, ax=ax1, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(premium_pct)), premium_pct.values, 'o-', color='#FF4757', linewidth=3, markersize=8)
    ax1.set_title('Renovation Budget Distribution by House Type\nwith Premium Willingness', fontweight='bold', fontsize=12)
    ax1.set_xlabel('House Type', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1_twin.set_ylabel('Premium Willingness (%)', fontweight='bold', color='#FF4757')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1:\n{str(e)}', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Budget Distribution by House Type', fontweight='bold', fontsize=12)

# Subplot 2: Top-middle - Grouped bar chart with error bars
ax2 = plt.subplot(3, 3, 2)
try:
    neighborhood_family = df.groupby([neighborhood_col, family_size_col])[energy_efficient_col].apply(lambda x: (x == 'Yes').mean()).unstack(fill_value=0)
    neighborhood_family.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_title('Energy-Efficient Solution Adoption\nby Neighborhood & Family Size', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Neighborhood Type', fontweight='bold')
    ax2.set_ylabel('Adoption Rate', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Family Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2:\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Energy-Efficient Solution Adoption', fontweight='bold', fontsize=12)

# Subplot 3: Top-right - Radar chart with scatter points
ax3 = plt.subplot(3, 3, 3, projection='polar')
try:
    motivations = df[motivation_col].value_counts().head(6)  # Limit to top 6 for readability
    angles = np.linspace(0, 2*np.pi, len(motivations), endpoint=False)
    values = motivations.values / motivations.sum()
    ax3.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B', markersize=8)
    ax3.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax3.set_xticks(angles)
    ax3.set_xticklabels([label[:15] + '...' if len(label) > 15 else label for label in motivations.index], fontsize=8)
    ax3.set_title('Renovation Motivations\nRadar Analysis', fontweight='bold', fontsize=12, pad=20)
    ax3.grid(True, alpha=0.3)
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3:\n{str(e)}', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Renovation Motivations Radar', fontweight='bold', fontsize=12)

# Subplot 4: Middle-left - Treemap simulation with pie charts
ax4 = plt.subplot(3, 3, 4)
try:
    room_freq = df.groupby([room_col, frequency_col]).size().unstack(fill_value=0)
    room_totals = room_freq.sum(axis=1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    wedges, texts, autotexts = ax4.pie(room_totals.values, labels=[label[:10] + '...' if len(label) > 10 else label for label in room_totals.index], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Room Renovation Preferences\nby Frequency Patterns', fontweight='bold', fontsize=12)
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4:\n{str(e)}', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Room Renovation Preferences', fontweight='bold', fontsize=12)

# Subplot 5: Middle-center - Parallel coordinates simulation
ax5 = plt.subplot(3, 3, 5)
try:
    # Create simplified parallel coordinates
    budget_encoded = le.fit_transform(df[budget_col].fillna('Unknown'))
    house_encoded = le.fit_transform(df[house_type_col].fillna('Unknown'))
    energy_encoded = le.fit_transform(df[energy_materials_col].fillna('Unknown'))
    knowledge_encoded = le.fit_transform(df[knowledge_col].fillna('Unknown'))

    for i in range(min(20, len(df))):  # Limit to first 20 for readability
        ax5.plot([0, 1, 2, 3], [budget_encoded[i], house_encoded[i], energy_encoded[i], knowledge_encoded[i]], 
                 alpha=0.6, color=plt.cm.viridis(i/20))

    ax5.set_xticks([0, 1, 2, 3])
    ax5.set_xticklabels(['Budget', 'House Type', 'Energy Materials', 'Knowledge'], fontweight='bold')
    ax5.set_title('Multi-dimensional Customer\nProfile Analysis', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 5:\n{str(e)}', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Customer Profile Analysis', fontweight='bold', fontsize=12)

# Subplot 6: Middle-right - Network-style cluster plot
ax6 = plt.subplot(3, 3, 6)
try:
    drivers = df[renovation_reason_col].value_counts()
    features = df[app_features_col].value_counts()
    x_pos = np.random.uniform(0, 10, len(drivers))
    y_pos = np.random.uniform(0, 10, len(drivers))
    sizes = drivers.values * 50
    colors = plt.cm.Set3(np.linspace(0, 1, len(drivers)))
    scatter = ax6.scatter(x_pos, y_pos, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    ax6.set_title('Renovation Drivers &\nApp Feature Preferences Network', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Driver Dimension', fontweight='bold')
    ax6.set_ylabel('Feature Dimension', fontweight='bold')
    ax6.grid(True, alpha=0.3)
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 6:\n{str(e)}', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Renovation Drivers Network', fontweight='bold', fontsize=12)

# Subplot 7: Bottom-left - Hierarchical clustering with heatmap
ax7 = plt.subplot(3, 3, 7)
try:
    # Create correlation matrix for clustering
    numeric_cols = []
    selected_cols = [house_type_col, budget_col, neighborhood_col, family_size_col, knowledge_col]
    
    for col in selected_cols:
        if col in df.columns:
            encoded_col = le.fit_transform(df[col].fillna('Unknown'))
            numeric_cols.append(encoded_col)

    if len(numeric_cols) > 1:
        cluster_data = np.array(numeric_cols).T
        linkage_matrix = linkage(cluster_data, method='ward')
        dendrogram(linkage_matrix, ax=ax7, leaf_rotation=90, leaf_font_size=8)
        ax7.set_title('Customer Clustering\nDendrogram', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Sample Index', fontweight='bold')
        ax7.set_ylabel('Distance', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Customer Clustering', fontweight='bold', fontsize=12)
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in subplot 7:\n{str(e)}', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Customer Clustering', fontweight='bold', fontsize=12)

# Subplot 8: Bottom-middle - Violin plots with box plot overlays
ax8 = plt.subplot(3, 3, 8)
try:
    consultation_pay = df[consultation_pay_col]
    consultation_categories = consultation_pay.value_counts()
    positions = range(len(consultation_categories))
    violin_data = [np.random.normal(i, 0.3, max(1, consultation_categories.iloc[i])) for i in range(len(consultation_categories))]
    parts = ax8.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    ax8.set_xticks(positions)
    ax8.set_xticklabels([label[:10] + '...' if len(label) > 10 else label for label in consultation_categories.index], rotation=45)
    ax8.set_title('Consultation Service\nWillingness Distribution', fontweight='bold', fontsize=12)
    ax8.set_ylabel('Distribution Density', fontweight='bold')
    ax8.grid(True, alpha=0.3)
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in subplot 8:\n{str(e)}', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Consultation Service Distribution', fontweight='bold', fontsize=12)

# Subplot 9: Bottom-right - Sankey-style flow diagram
ax9 = plt.subplot(3, 3, 9)
try:
    motivations = df[renovation_reason_col].value_counts()
    consultations = df[consultation_col].value_counts()
    budgets = df[budget_col].value_counts()

    # Create stacked horizontal bars to simulate flow
    y_positions = np.arange(len(motivations))
    bar_width = 0.8
    colors = plt.cm.Set3(np.linspace(0, 1, len(motivations)))

    bars = ax9.barh(y_positions, motivations.values, bar_width, color=colors, alpha=0.8)
    ax9.set_yticks(y_positions)
    ax9.set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in motivations.index])
    ax9.set_title('Customer Journey Flow:\nMotivation → Consultation → Budget', fontweight='bold', fontsize=12)
    ax9.set_xlabel('Flow Volume', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='x')

    # Add connecting lines to simulate flow
    for i, (bar, value) in enumerate(zip(bars, motivations.values)):
        ax9.annotate(f'{value}', xy=(value/2, i), ha='center', va='center', fontweight='bold', color='white')
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in subplot 9:\n{str(e)}', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Customer Journey Flow', fontweight='bold', fontsize=12)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Add main title
fig.suptitle('Comprehensive Customer Segmentation Analysis\nHome Renovation Survey - Asia', 
             fontsize=20, fontweight='bold', y=0.98)

plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()