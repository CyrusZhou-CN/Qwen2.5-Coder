import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')

# Clean data - remove unnamed columns and handle missing values
train_df = train_df.drop(columns=['Unnamed: 133'], errors='ignore')
symptom_cols = [col for col in train_df.columns if col != 'prognosis']

# Create figure with optimized size
fig = plt.figure(figsize=(20, 16), facecolor='white')

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F8961E', '#90323D']
disease_colors = plt.cm.Set3(np.linspace(0, 1, 12))

# Subplot 1: Disease frequency with average symptom count
ax1 = plt.subplot(3, 3, 1)
disease_counts = train_df['prognosis'].value_counts().head(10)  # Limit to top 10
avg_symptoms_per_disease = []
for disease in disease_counts.index:
    disease_data = train_df[train_df['prognosis'] == disease]
    avg_symptoms_per_disease.append(disease_data[symptom_cols].sum(axis=1).mean())

# Horizontal bar chart
bars = ax1.barh(range(len(disease_counts)), disease_counts.values, color=colors[0], alpha=0.7)
ax1.set_yticks(range(len(disease_counts)))
ax1.set_yticklabels([d[:15] + '...' if len(d) > 15 else d for d in disease_counts.index], fontsize=8)
ax1.set_xlabel('Disease Frequency', fontweight='bold')
ax1.set_title('Disease Frequency with Avg Symptom Count', fontweight='bold', fontsize=10)

# Overlaid line plot
ax1_twin = ax1.twiny()
ax1_twin.plot(avg_symptoms_per_disease, range(len(disease_counts)), 'o-', color=colors[1], linewidth=2, markersize=4)
ax1_twin.set_xlabel('Avg Symptom Count', fontweight='bold', color=colors[1])
ax1_twin.tick_params(axis='x', labelcolor=colors[1])

# Subplot 2: Top 10 symptoms with prevalence rates
ax2 = plt.subplot(3, 3, 2)
symptom_sums = train_df[symptom_cols].sum().sort_values(ascending=False)[:10]
symptom_prevalence = (symptom_sums / len(train_df)) * 100

bars = ax2.bar(range(len(symptom_sums)), symptom_sums.values, color=colors[2], alpha=0.7)
ax2.set_xticks(range(len(symptom_sums)))
ax2.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in symptom_sums.index], 
                    rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Symptom Count', fontweight='bold')
ax2.set_title('Top 10 Most Common Symptoms', fontweight='bold', fontsize=10)

# Secondary y-axis for prevalence
ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(symptom_prevalence)), symptom_prevalence.values, 'o-', 
              color=colors[3], linewidth=2, markersize=6)
ax2_twin.set_ylabel('Prevalence Rate (%)', fontweight='bold', color=colors[3])
ax2_twin.tick_params(axis='y', labelcolor=colors[3])

# Subplot 3: Disease categories pie chart
ax3 = plt.subplot(3, 3, 3)

# Simplified disease categorization
disease_categories = {
    'Respiratory': ['Bronchial Asthma', 'Pneumonia', 'Common Cold'],
    'Digestive': ['GERD', 'Peptic ulcer diseae', 'Gastroenteritis'],
    'Skin': ['Fungal infection', 'Psoriasis', 'Impetigo'],
    'Infectious': ['Malaria', 'Dengue', 'Typhoid'],
    'Other': []
}

# Assign remaining diseases to 'Other'
all_diseases = set(train_df['prognosis'].unique())
categorized = set()
for diseases in disease_categories.values():
    categorized.update(diseases)
disease_categories['Other'] = list(all_diseases - categorized)

# Count diseases by category
category_counts = {}
for category, diseases in disease_categories.items():
    count = sum(train_df['prognosis'].value_counts().get(disease, 0) for disease in diseases)
    if count > 0:
        category_counts[category] = count

# Pie chart
if category_counts:
    wedges, texts, autotexts = ax3.pie(category_counts.values(), labels=category_counts.keys(), 
                                      autopct='%1.1f%%', colors=disease_colors[:len(category_counts)])
    ax3.set_title('Disease Categories Distribution', fontweight='bold', fontsize=10)

# Add train/test info
train_size = len(train_df)
test_size = len(test_df)
ax3.text(1.3, 0, f'Training: {train_size}\nTesting: {test_size}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[4], alpha=0.7),
         fontweight='bold', fontsize=8)

# Subplot 4: Correlation heatmap of top 15 symptoms (reduced for performance)
ax4 = plt.subplot(3, 3, 4)
top_15_symptoms = train_df[symptom_cols].sum().sort_values(ascending=False)[:15].index
corr_matrix = train_df[top_15_symptoms].corr()

im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(top_15_symptoms)))
ax4.set_yticks(range(len(top_15_symptoms)))
ax4.set_xticklabels([s[:8] for s in top_15_symptoms], rotation=45, ha='right', fontsize=7)
ax4.set_yticklabels([s[:8] for s in top_15_symptoms], fontsize=7)
ax4.set_title('Symptom Correlation Heatmap', fontweight='bold', fontsize=10)

# Subplot 5: Scatter plot matrix (simplified to 2D scatter)
ax5 = plt.subplot(3, 3, 5)
# Use top symptoms for scatter plot
symptom1, symptom2 = symptom_sums.index[0], symptom_sums.index[1]

# Plot top 6 diseases for clarity
top_diseases = disease_counts.head(6).index
for i, disease in enumerate(top_diseases):
    disease_data = train_df[train_df['prognosis'] == disease]
    if len(disease_data) > 0:
        ax5.scatter(disease_data[symptom1], disease_data[symptom2], 
                   c=disease_colors[i], label=disease[:10], alpha=0.6, s=20)

ax5.set_xlabel(symptom1.replace('_', ' ').title(), fontweight='bold')
ax5.set_ylabel(symptom2.replace('_', ' ').title(), fontweight='bold')
ax5.set_title('Symptom Scatter Plot by Disease', fontweight='bold', fontsize=10)
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

# Subplot 6: Violin plot with symptom counts
ax6 = plt.subplot(3, 3, 6)
train_df['symptom_count'] = train_df[symptom_cols].sum(axis=1)

# Use top 5 diseases for violin plot
top_5_diseases = disease_counts.head(5).index
violin_data = []
violin_labels = []

for disease in top_5_diseases:
    disease_data = train_df[train_df['prognosis'] == disease]['symptom_count']
    if len(disease_data) > 0:
        violin_data.append(disease_data)
        violin_labels.append(disease[:10])

if violin_data:
    parts = ax6.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(disease_colors[i])
        pc.set_alpha(0.7)

ax6.set_xticks(range(len(violin_labels)))
ax6.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Symptom Count', fontweight='bold')
ax6.set_title('Symptom Count Distribution', fontweight='bold', fontsize=10)

# Subplot 7: Simplified network visualization
ax7 = plt.subplot(3, 3, 7)
# Create a simple network representation using scatter plot
top_6_diseases = disease_counts.head(6)

# Calculate average symptom profiles for positioning
positions = []
for disease in top_6_diseases.index:
    disease_data = train_df[train_df['prognosis'] == disease]
    avg_symptoms = disease_data[symptom_cols].mean()
    # Use first two principal symptoms for positioning
    x = avg_symptoms[symptom_sums.index[0]]
    y = avg_symptoms[symptom_sums.index[1]]
    positions.append((x, y))

# Plot nodes
for i, (disease, count) in enumerate(top_6_diseases.items()):
    x, y = positions[i]
    ax7.scatter(x, y, s=count*10, c=disease_colors[i], alpha=0.7, edgecolors='black')
    ax7.annotate(disease[:8], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=7)

ax7.set_xlabel('Primary Symptom Avg', fontweight='bold')
ax7.set_ylabel('Secondary Symptom Avg', fontweight='bold')
ax7.set_title('Disease Similarity Network', fontweight='bold', fontsize=10)

# Subplot 8: Parallel coordinates plot
ax8 = plt.subplot(3, 3, 8)
top_4_diseases = disease_counts.head(4).index
selected_symptoms = symptom_sums.head(6).index

for i, disease in enumerate(top_4_diseases):
    disease_data = train_df[train_df['prognosis'] == disease]
    if len(disease_data) > 0:
        symptom_profile = disease_data[selected_symptoms].mean()
        ax8.plot(range(len(selected_symptoms)), symptom_profile, 
                'o-', color=disease_colors[i], label=disease[:10], linewidth=2, markersize=4)

ax8.set_xticks(range(len(selected_symptoms)))
ax8.set_xticklabels([s[:8] for s in selected_symptoms], rotation=45, ha='right', fontsize=8)
ax8.set_ylabel('Avg Symptom Presence', fontweight='bold')
ax8.set_title('Disease Symptom Profiles', fontweight='bold', fontsize=10)
ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax8.grid(True, alpha=0.3)

# Subplot 9: Radar chart
ax9 = plt.subplot(3, 3, 9, projection='polar')
top_4_diseases = disease_counts.head(4).index
radar_symptoms = symptom_sums.head(5).index  # Reduced for clarity

# Calculate angles for radar chart
angles = [n / len(radar_symptoms) * 2 * pi for n in range(len(radar_symptoms))]
angles += angles[:1]  # Complete the circle

for i, disease in enumerate(top_4_diseases):
    disease_data = train_df[train_df['prognosis'] == disease]
    if len(disease_data) > 0:
        values = disease_data[radar_symptoms].mean().tolist()
        values += values[:1]  # Complete the circle
        
        ax9.plot(angles, values, 'o-', linewidth=2, label=disease[:8], color=disease_colors[i])
        ax9.fill(angles, values, alpha=0.25, color=disease_colors[i])

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels([s[:8] for s in radar_symptoms], fontsize=7)
ax9.set_title('Symptom Radar Chart', fontweight='bold', fontsize=10, pad=20)
ax9.legend(bbox_to_anchor=(1.2, 1.1), loc='upper left', fontsize=7)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('disease_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()