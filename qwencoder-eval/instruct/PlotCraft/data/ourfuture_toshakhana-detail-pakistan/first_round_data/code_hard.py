import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_excel('Tosha Khana Pakistan.xlsx')

# Clean and preprocess the data efficiently
def clean_value(val):
    if pd.isna(val) or val == '-':
        return 0
    val_str = str(val).replace('Rs.', '').replace(',', '').replace('/-', '').strip()
    try:
        return float(val_str)
    except:
        return 0

# Apply cleaning operations
df['Assessed_Value_Clean'] = df['Assessed Value'].apply(clean_value)
df['Date_Clean'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date_Clean'].dt.year

# Filter out invalid years and limit data for performance
df = df[(df['Year'] >= 2000) & (df['Year'] <= 2025)].copy()

# Extract recipient type and position (simplified)
def extract_position(name):
    if pd.isna(name):
        return 'Unknown'
    name_str = str(name).lower()
    if 'minister' in name_str:
        return 'Minister'
    elif 'secretary' in name_str:
        return 'Secretary'
    elif 'advisor' in name_str:
        return 'Advisor'
    elif 'president' in name_str:
        return 'President'
    else:
        return 'Other Official'

df['Position'] = df['Name of Recipient'].apply(extract_position)

# Create recipient type categories (simplified)
def categorize_recipient(name):
    if pd.isna(name):
        return 'Unknown'
    name_str = str(name).lower()
    if 'foreign' in name_str:
        return 'Foreign Affairs'
    elif 'finance' in name_str:
        return 'Finance'
    elif 'defense' in name_str or 'defence' in name_str:
        return 'Defense'
    elif 'interior' in name_str:
        return 'Interior'
    else:
        return 'Other Ministries'

df['Recipient_Type'] = df['Name of Recipient'].apply(categorize_recipient)

# Clean item categories
df['Item Category'] = df['Item Category'].fillna('Other')

# Create the 3x3 subplot grid with optimized plotting
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Subplot 1: Stacked bar chart with overlaid line plot
ax1 = plt.subplot(3, 3, 1)
try:
    category_recipient = df.groupby(['Item Category', 'Recipient_Type'])['Assessed_Value_Clean'].sum().unstack(fill_value=0)
    category_recipient_top = category_recipient.head(5)  # Reduced for performance
    category_recipient_top.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
    cumulative = category_recipient_top.sum(axis=1).cumsum()
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(cumulative)), cumulative.values, 'ro-', linewidth=2, markersize=4)
    ax1.set_title('Gift Categories by Recipient Type', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Item Category', fontsize=8)
    ax1.set_ylabel('Total Value (Rs.)', fontsize=8)
    ax1_twin.set_ylabel('Cumulative Value', fontsize=8)
    ax1.tick_params(axis='x', rotation=45, labelsize=7)
    ax1.legend(fontsize=6, loc='upper left')
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1', ha='center', va='center', transform=ax1.transAxes)

# Subplot 2: Pie chart with inner donut
ax2 = plt.subplot(3, 3, 2)
try:
    value_by_category = df.groupby('Item Category')['Assessed_Value_Clean'].sum().sort_values(ascending=False).head(6)
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_by_category)))
    wedges, texts, autotexts = ax2.pie(value_by_category.values, labels=value_by_category.index, 
                                       autopct='%1.1f%%', colors=colors, pctdistance=0.85, textprops={'fontsize': 7})
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('Gift Value Composition\nby Category', fontweight='bold', fontsize=10)
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2', ha='center', va='center', transform=ax2.transAxes)

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 3, 3)
try:
    position_stats = df.groupby('Position').agg({
        'Assessed_Value_Clean': ['sum', 'count']
    }).round(0)
    position_stats.columns = ['Total_Value', 'Count']
    position_stats = position_stats.sort_values('Total_Value', ascending=False).head(6)
    
    scatter = ax3.scatter(position_stats['Count'], position_stats['Total_Value'], 
                         s=position_stats['Total_Value']/5000, alpha=0.6, 
                         c=range(len(position_stats)), cmap='viridis')
    for i, pos in enumerate(position_stats.index):
        ax3.annotate(pos[:8], (position_stats.loc[pos, 'Count'], position_stats.loc[pos, 'Total_Value']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    ax3.set_title('Gift Frequency vs Value\nby Position', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Number of Gifts', fontsize=8)
    ax3.set_ylabel('Total Value (Rs.)', fontsize=8)
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3', ha='center', va='center', transform=ax3.transAxes)

# Subplot 4: Horizontal stacked bar
ax4 = plt.subplot(3, 3, 4)
try:
    yearly_data = df.groupby(['Year', 'Item Category'])['Assessed_Value_Clean'].sum().unstack(fill_value=0)
    yearly_data_top = yearly_data.iloc[:6, :5]  # Reduced for performance
    yearly_data_top.plot(kind='barh', stacked=True, ax=ax4, colormap='tab10')
    ax4.set_title('Gift Values by Year', fontweight='bold', fontsize=10)
    ax4.set_xlabel('Total Value (Rs.)', fontsize=8)
    ax4.set_ylabel('Year', fontsize=8)
    ax4.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4', ha='center', va='center', transform=ax4.transAxes)

# Subplot 5: Box plots
ax5 = plt.subplot(3, 3, 5)
try:
    top_categories = df['Item Category'].value_counts().head(5).index
    category_data = []
    category_labels = []
    
    for cat in top_categories:
        values = df[df['Item Category'] == cat]['Assessed_Value_Clean']
        values = values[values > 0]
        if len(values) > 10:  # Only include categories with sufficient data
            category_data.append(values.sample(min(100, len(values))))  # Sample for performance
            category_labels.append(cat[:10])
    
    if category_data:
        ax5.boxplot(category_data, labels=category_labels)
        ax5.set_title('Value Distribution\nby Gift Type', fontweight='bold', fontsize=10)
        ax5.set_xlabel('Gift Category', fontsize=8)
        ax5.set_ylabel('Assessed Value (Rs.)', fontsize=8)
        ax5.tick_params(axis='x', rotation=45, labelsize=7)
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 5', ha='center', va='center', transform=ax5.transAxes)

# Subplot 6: Heatmap
ax6 = plt.subplot(3, 3, 6)
try:
    # Create time periods
    df['Period'] = (df['Year'] // 5) * 5
    period_data = df.groupby(['Period', 'Item Category'])['Assessed_Value_Clean'].sum().unstack(fill_value=0)
    period_data = period_data.iloc[:4, :5]  # Limit size
    
    if not period_data.empty:
        sns.heatmap(period_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6, cbar_kws={'shrink': 0.8})
        ax6.set_title('Gift Values by Period\nand Category', fontweight='bold', fontsize=10)
        ax6.set_xlabel('Gift Categories', fontsize=8)
        ax6.set_ylabel('Time Periods', fontsize=8)
        ax6.tick_params(labelsize=7)
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 6', ha='center', va='center', transform=ax6.transAxes)

# Subplot 7: Nested pie chart (simplified)
ax7 = plt.subplot(3, 3, 7)
try:
    outer_data = df.groupby('Recipient_Type')['Assessed_Value_Clean'].sum().head(5)
    colors = plt.cm.Set2(np.linspace(0, 1, len(outer_data)))
    wedges1, texts1 = ax7.pie(outer_data.values, labels=[label[:8] for label in outer_data.index], 
                             radius=1, colors=colors, textprops={'fontsize': 7})
    
    # Inner circle
    inner_data = df.groupby('Position')['Assessed_Value_Clean'].sum().head(4)
    wedges2, texts2 = ax7.pie(inner_data.values, radius=0.6, 
                             colors=plt.cm.Set3(np.linspace(0, 1, len(inner_data))))
    ax7.set_title('Hierarchical Gift Breakdown', fontweight='bold', fontsize=10)
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in subplot 7', ha='center', va='center', transform=ax7.transAxes)

# Subplot 8: Area chart with scatter
ax8 = plt.subplot(3, 3, 8)
try:
    yearly_totals = df.groupby('Year')['Assessed_Value_Clean'].sum()
    yearly_totals = yearly_totals[yearly_totals.index >= 2002].head(10)
    
    ax8.fill_between(yearly_totals.index, yearly_totals.values, alpha=0.7, color='skyblue')
    ax8.plot(yearly_totals.index, yearly_totals.values, 'o-', color='darkblue', markersize=4)
    
    # Add high-value points
    high_value_threshold = df['Assessed_Value_Clean'].quantile(0.9)
    high_value_gifts = df[df['Assessed_Value_Clean'] > high_value_threshold]
    if len(high_value_gifts) > 0:
        sample_high = high_value_gifts.sample(min(20, len(high_value_gifts)))
        ax8.scatter(sample_high['Year'], sample_high['Assessed_Value_Clean'], 
                   c='red', s=30, alpha=0.8, zorder=5)
    
    ax8.set_title('Gift Values Over Time\nwith High-Value Markers', fontweight='bold', fontsize=10)
    ax8.set_xlabel('Year', fontsize=8)
    ax8.set_ylabel('Total Value (Rs.)', fontsize=8)
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in subplot 8', ha='center', va='center', transform=ax8.transAxes)

# Subplot 9: Radar chart (simplified)
ax9 = plt.subplot(3, 3, 9, projection='polar')
try:
    # Create department profiles
    dept_profile = df.groupby(['Recipient_Type', 'Item Category'])['Assessed_Value_Clean'].sum().unstack(fill_value=0)
    dept_profile_norm = dept_profile.div(dept_profile.sum(axis=1), axis=0).fillna(0)
    dept_profile_top = dept_profile_norm.head(3)
    
    categories = dept_profile_top.columns[:5]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['red', 'blue', 'green']
    for i, (dept, values) in enumerate(dept_profile_top.iterrows()):
        values_plot = values[:5].tolist()
        values_plot += values_plot[:1]
        ax9.plot(angles, values_plot, 'o-', linewidth=2, label=dept[:10], color=colors[i])
        ax9.fill(angles, values_plot, alpha=0.25, color=colors[i])
    
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels([cat[:8] for cat in categories], fontsize=7)
    ax9.set_title('Gift Composition Profiles', fontweight='bold', fontsize=10, pad=20)
    ax9.legend(bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=7)
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in subplot 9', ha='center', va='center', transform=ax9.transAxes)

# Overall layout adjustment
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.savefig('toshakhana_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()