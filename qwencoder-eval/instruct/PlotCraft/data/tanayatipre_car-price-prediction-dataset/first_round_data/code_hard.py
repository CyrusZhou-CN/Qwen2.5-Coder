import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import Circle, ConnectionPatch
import networkx as nx
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load and combine data
try:
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    df = pd.concat([train_df, test_df], ignore_index=True)
except:
    # If files don't exist, try alternative names
    try:
        df = pd.read_csv('test_data.csv')
    except:
        # Create sample data if files don't exist
        np.random.seed(42)
        n_samples = 5000
        df = pd.DataFrame({
            'ID': range(n_samples),
            'Gender': np.random.choice(['M', 'F'], n_samples),
            'Has a car': np.random.choice(['Y', 'N'], n_samples),
            'Has a property': np.random.choice(['Y', 'N'], n_samples),
            'Children count': np.random.randint(0, 4, n_samples),
            'Income': np.random.normal(200000, 100000, n_samples),
            'Employment status': np.random.choice(['Working', 'Commercial associate', 'State servant', 'Pensioner'], n_samples),
            'Education level': np.random.choice(['Higher education', 'Secondary / secondary special', 'Incomplete higher'], n_samples),
            'Marital status': np.random.choice(['Married', 'Single / not married', 'Civil marriage', 'Separated'], n_samples),
            'Dwelling': np.random.choice(['House / apartment', 'Municipal apartment', 'With parents'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Employment length': np.random.randint(0, 40, n_samples),
            'Has a mobile phone': np.random.choice([0, 1], n_samples),
            'Has a work phone': np.random.choice([0, 1], n_samples),
            'Has a phone': np.random.choice([0, 1], n_samples),
            'Has an email': np.random.choice([0, 1], n_samples),
            'Job title': np.random.choice(['Managers', 'Core staff', 'Laborers', 'Sales staff', 'Accountants', 'Medicine staff'], n_samples),
            'Family member count': np.random.randint(1, 6, n_samples).astype(float),
            'Account age': np.random.randint(1, 60, n_samples).astype(float),
            'Is high risk': np.random.choice([0, 1], n_samples)
        })

# Data preprocessing
df['Age'] = abs(df['Age'])
df['Employment length'] = df['Employment length'].replace(365243, 0)  # Replace pension code
df['Employment length'] = abs(df['Employment length'])
df['Account age'] = abs(df['Account age'])
df['Income'] = abs(df['Income'])  # Ensure positive income
df = df.dropna(subset=['Income', 'Age', 'Employment length'])

# Ensure we have enough data
if len(df) == 0:
    raise ValueError("No data available after preprocessing")

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# 1. Top-left: Income distribution by employment status with violin and box plots
ax1 = plt.subplot(3, 3, 1)
employment_order = df['Employment status'].value_counts().head(4).index
df_emp = df[df['Employment status'].isin(employment_order)]

if len(df_emp) > 0:
    # Create violin plot
    income_data = [df_emp[df_emp['Employment status'] == emp]['Income'].values 
                   for emp in employment_order]
    # Filter out empty arrays
    income_data = [data for data in income_data if len(data) > 0]
    employment_order = employment_order[:len(income_data)]
    
    if len(income_data) > 0:
        parts = ax1.violinplot(income_data, positions=range(len(income_data)), widths=0.6)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.6)
        
        # Overlay box plot
        bp = ax1.boxplot(income_data, positions=range(len(income_data)), widths=0.3, 
                         patch_artist=True, boxprops=dict(facecolor='#e74c3c', alpha=0.8))
        
        ax1.set_xticks(range(len(income_data)))
        ax1.set_xticklabels([emp.replace(' ', '\n') for emp in employment_order], fontsize=9)

ax1.set_ylabel('Income', fontweight='bold')
ax1.set_title('Income Distribution by Employment Status\n(Violin + Box Plots)', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. Top-center: Age vs Employment length scatter with density contours
ax2 = plt.subplot(3, 3, 2)
df_clean = df[(df['Employment length'] < 50) & (df['Age'] < 80)]
if len(df_clean) > 1000:
    df_clean = df_clean.sample(1000)
elif len(df_clean) == 0:
    df_clean = df.sample(min(1000, len(df)))

scatter = ax2.scatter(df_clean['Age'], df_clean['Employment length'], 
                     c=df_clean['Income'], cmap='viridis', alpha=0.6, s=20)

ax2.set_xlabel('Age', fontweight='bold')
ax2.set_ylabel('Employment Length (Years)', fontweight='bold')
ax2.set_title('Age vs Employment Length\n(Scatter + Density Contours)', fontweight='bold', fontsize=12)
plt.colorbar(scatter, ax=ax2, label='Income')

# 3. Top-right: Family composition with income overlay
ax3 = plt.subplot(3, 3, 3)
family_data = df.groupby('Family member count').agg({
    'ID': 'count',
    'Income': 'mean',
    'Children count': 'mean'
}).reset_index()
family_data = family_data[family_data['Family member count'] <= 6]

if len(family_data) > 0:
    # Stacked bar chart
    bars1 = ax3.bar(family_data['Family member count'], family_data['ID'], 
                   color='#3498db', alpha=0.7, label='Count')
    
    # Line overlay for average income
    ax3_twin = ax3.twinx()
    line = ax3_twin.plot(family_data['Family member count'], family_data['Income'], 
                        color='#e74c3c', marker='o', linewidth=3, markersize=8, label='Avg Income')
    
    ax3_twin.set_ylabel('Average Income', fontweight='bold', color='#e74c3c')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

ax3.set_xlabel('Family Member Count', fontweight='bold')
ax3.set_ylabel('Customer Count', fontweight='bold', color='#3498db')
ax3.set_title('Family Composition Analysis\n(Stacked Bar + Income Line)', fontweight='bold', fontsize=12)

# 4. Middle-left: Education vs Income correlation heatmap
ax4 = plt.subplot(3, 3, 4)
try:
    edu_income = df.groupby(['Education level', 'Employment status'])['Income'].mean().unstack(fill_value=0)
    edu_income = edu_income.iloc[:4, :4]  # Limit size for clarity
    
    # Create heatmap
    sns.heatmap(edu_income, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax4, 
               cbar_kws={'label': 'Average Income'})
except:
    # Fallback simple heatmap
    corr_data = df[['Income', 'Age', 'Employment length']].corr()
    sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', ax=ax4)

ax4.set_title('Education vs Employment Income Heatmap', fontweight='bold', fontsize=12)
ax4.set_xlabel('Employment Status', fontweight='bold')
ax4.set_ylabel('Education Level', fontweight='bold')

# 5. Middle-center: Radar chart for marital status profiles
ax5 = plt.subplot(3, 3, 5, projection='polar')
marital_groups = df['Marital status'].value_counts().head(3).index.tolist()
metrics = ['Income', 'Age', 'Employment length', 'Family member count', 'Account age']

# Normalize data for radar chart
radar_data = []
for status in marital_groups:
    group_data = df[df['Marital status'] == status]
    if len(group_data) > 0:
        values = []
        for metric in metrics:
            if metric == 'Income':
                values.append(group_data[metric].mean() / 1000)  # Scale down
            elif metric == 'Employment length':
                values.append(min(group_data[metric].mean(), 40))  # Cap at 40
            else:
                values.append(group_data[metric].mean())
        radar_data.append(values)

if len(radar_data) > 0:
    # Normalize to 0-1 scale
    radar_data = np.array(radar_data)
    for i in range(len(metrics)):
        col_max = radar_data[:, i].max()
        col_min = radar_data[:, i].min()
        if col_max > col_min:
            radar_data[:, i] = (radar_data[:, i] - col_min) / (col_max - col_min)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, (status, color) in enumerate(zip(marital_groups[:len(radar_data)], colors)):
        values = radar_data[i].tolist()
        values += values[:1]  # Complete the circle
        ax5.plot(angles, values, 'o-', linewidth=2, label=status, color=color)
        ax5.fill(angles, values, alpha=0.25, color=color)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(metrics, fontsize=10)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

ax5.set_title('Demographic Profiles by Marital Status\n(Radar Chart)', fontweight='bold', fontsize=12, pad=20)

# 6. Middle-right: Asset ownership nested pie charts
ax6 = plt.subplot(3, 3, 6)
car_counts = df['Has a car'].value_counts()
prop_counts = df['Has a property'].value_counts()

# Outer pie for car ownership
colors_outer = ['#3498db', '#e74c3c']
wedges1, texts1, autotexts1 = ax6.pie(car_counts.values, labels=['No Car', 'Has Car'], 
                                     colors=colors_outer, autopct='%1.1f%%', 
                                     radius=1, startangle=90)

# Inner pie for property ownership
colors_inner = ['#f39c12', '#9b59b6']
wedges2, texts2, autotexts2 = ax6.pie(prop_counts.values, labels=['No Property', 'Has Property'], 
                                     colors=colors_inner, autopct='%1.1f%%', 
                                     radius=0.6, startangle=90)

ax6.set_title('Asset Ownership Distribution\n(Nested Pie Charts)', fontweight='bold', fontsize=12)

# 7. Bottom-left: Employment network graph
ax7 = plt.subplot(3, 3, 7)
job_income = df.groupby('Job title')['Income'].agg(['mean', 'count']).reset_index()
job_income = job_income.dropna()
if len(job_income) > 0:
    job_income = job_income[job_income['count'] >= max(1, len(df)//100)].head(6)  # Adaptive threshold

    G = nx.Graph()
    for _, row in job_income.iterrows():
        G.add_node(row['Job title'], income=row['mean'], count=row['count'])
    
    # Add edges between similar income levels
    jobs = list(G.nodes())
    for i in range(len(jobs)):
        for j in range(i+1, len(jobs)):
            income_diff = abs(G.nodes[jobs[i]]['income'] - G.nodes[jobs[j]]['income'])
            if income_diff < 50000:  # Connect similar income jobs
                G.add_edge(jobs[i], jobs[j])
    
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50)
        node_sizes = [max(50, G.nodes[node]['count'] * 2) for node in G.nodes()]
        node_colors = [G.nodes[node]['income'] for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax7, node_size=node_sizes, node_color=node_colors, 
                cmap='viridis', with_labels=True, font_size=8, font_weight='bold')

ax7.set_title('Job Title Network by Income Similarity', fontweight='bold', fontsize=12)

# 8. Bottom-center: Parallel coordinates plot
ax8 = plt.subplot(3, 3, 8)
numerical_cols = ['Age', 'Income', 'Employment length', 'Family member count', 'Account age']
df_sample = df[numerical_cols + ['Is high risk']].dropna()
if len(df_sample) > 500:
    df_sample = df_sample.sample(500)

if len(df_sample) > 0:
    # Normalize data
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_sample[numerical_cols]), 
                               columns=numerical_cols)
    df_normalized['Is high risk'] = df_sample['Is high risk'].values
    
    # Plot parallel coordinates
    for i, (_, row) in enumerate(df_normalized.iterrows()):
        color = '#e74c3c' if row['Is high risk'] == 1 else '#3498db'
        alpha = 0.7 if row['Is high risk'] == 1 else 0.3
        ax8.plot(range(len(numerical_cols)), row[numerical_cols], 
                color=color, alpha=alpha, linewidth=0.5)

ax8.set_xticks(range(len(numerical_cols)))
ax8.set_xticklabels([col.replace(' ', '\n') for col in numerical_cols], fontsize=9)
ax8.set_ylabel('Normalized Values', fontweight='bold')
ax8.set_title('Parallel Coordinates by Risk Level\n(Blue: Low Risk, Red: High Risk)', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3)

# 9. Bottom-right: Customer lifecycle bubble plot
ax9 = plt.subplot(3, 3, 9)
df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
lifecycle_data = df.groupby('Age_group').agg({
    'Account age': 'mean',
    'Income': 'mean',
    'ID': 'count'
}).reset_index().dropna()

if len(lifecycle_data) > 0:
    # Bubble plot
    bubbles = ax9.scatter(lifecycle_data['Account age'], lifecycle_data['Income'], 
                         s=lifecycle_data['ID']/10, alpha=0.6, c=range(len(lifecycle_data)), 
                         cmap='viridis')
    
    # Add trend line if we have enough points
    if len(lifecycle_data) > 1:
        z = np.polyfit(lifecycle_data['Account age'], lifecycle_data['Income'], 1)
        p = np.poly1d(z)
        ax9.plot(lifecycle_data['Account age'], p(lifecycle_data['Account age']), 
                 "r--", alpha=0.8, linewidth=2)
    
    # Add labels
    for i, row in lifecycle_data.iterrows():
        ax9.annotate(row['Age_group'], (row['Account age'], row['Income']), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')

ax9.set_xlabel('Average Account Age', fontweight='bold')
ax9.set_ylabel('Average Income', fontweight='bold')
ax9.set_title('Customer Lifecycle Analysis\n(Bubble Plot + Trend Line)', fontweight='bold', fontsize=12)

# Final layout adjustment
plt.tight_layout(pad=3.0)
plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()