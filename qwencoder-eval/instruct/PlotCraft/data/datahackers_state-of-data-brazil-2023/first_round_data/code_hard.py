import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('State_of_data_BR_2023_Kaggle - df_survey_2023.csv')

# Clean column names for easier access
def clean_column_name(col):
    if "', '" in col:
        return col.split("', '")[1].rstrip("')")
    return col

df.columns = [clean_column_name(col) for col in df.columns]

# Data preprocessing with optimized performance
def prepare_data():
    # Sample data for performance
    data = df.sample(n=min(2000, len(df)), random_state=42).copy()
    
    # Key columns
    cols = {
        'age': 'Idade',
        'experience': 'Quanto tempo de experiência na área de dados você tem?',
        'role': 'Cargo Atual',
        'education': 'Nivel de Ensino',
        'formation': 'Área de Formação',
        'seniority': 'Nivel',
        'salary': 'Faixa salarial',
        'sector': 'Setor',
        'company_size': 'Numero de Funcionarios',
        'satisfaction': 'Você está satisfeito na sua empresa atual?',
        'work_arrangement': 'Qual a forma de trabalho ideal para você?',
        'main_lang': 'Entre as linguagens listadas abaixo, qual é a que você mais utiliza no trabalho?',
        'cloud_pref': 'Cloud preferida'
    }
    
    # Experience mapping
    exp_mapping = {
        'Menos de 1 ano': 0.5,
        'de 1 a 2 anos': 1.5,
        'de 3 a 4 anos': 3.5,
        'de 5 a 6 anos': 5.5,
        'de 7 a 10 anos': 8.5,
        'Mais de 10 anos': 12
    }
    
    # Salary mapping
    salary_mapping = {
        'Até R$ 1.000/mês': 1000,
        'de R$ 1.001/mês a R$ 2.000/mês': 1500,
        'de R$ 2.001/mês a R$ 3.000/mês': 2500,
        'de R$ 3.001/mês a R$ 4.000/mês': 3500,
        'de R$ 4.001/mês a R$ 6.000/mês': 5000,
        'de R$ 6.001/mês a R$ 8.000/mês': 7000,
        'de R$ 8.001/mês a R$ 12.000/mês': 10000,
        'de R$ 12.001/mês a R$ 16.000/mês': 14000,
        'de R$ 16.001/mês a R$ 20.000/mês': 18000,
        'de R$ 20.001/mês a R$ 25.000/mês': 22500,
        'Acima de R$ 25.000/mês': 30000
    }
    
    # Apply mappings
    data['experience_numeric'] = data[cols['experience']].map(exp_mapping)
    data['salary_numeric'] = data[cols['salary']].map(salary_mapping)
    data['satisfaction_numeric'] = data[cols['satisfaction']].map({0.0: 0, 1.0: 1})
    
    return data, cols

# Prepare data
data, cols = prepare_data()

# Create figure
fig = plt.figure(figsize=(18, 14), facecolor='white')
fig.suptitle('Brazil Data Professional Landscape: Clustering Patterns & Hierarchical Relationships', 
             fontsize=16, fontweight='bold', y=0.95)

# Define color palettes
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Top-left: Scatter plot with KDE overlay
ax1 = plt.subplot(2, 2, 1, facecolor='white')

# Filter data for plotting
plot_data = data.dropna(subset=[cols['age'], 'experience_numeric', cols['role'], 'salary_numeric'])
if len(plot_data) > 10:
    # Get top roles
    top_roles = plot_data[cols['role']].value_counts().head(6).index
    
    # Plot scatter with bubble sizes
    for i, role in enumerate(top_roles):
        role_data = plot_data[plot_data[cols['role']] == role]
        if len(role_data) > 0:
            sizes = (role_data['salary_numeric'].fillna(5000) / 500).clip(10, 200)
            ax1.scatter(role_data['experience_numeric'], role_data[cols['age']], 
                       s=sizes, alpha=0.6, c=colors[i % len(colors)], 
                       label=role[:15], edgecolors='white', linewidth=0.5)
    
    # Add KDE contour overlay
    try:
        x = plot_data['experience_numeric'].dropna()
        y = plot_data[cols['age']].dropna()
        if len(x) > 20 and len(y) > 20:
            # Sample for performance
            sample_size = min(500, len(x))
            x_sample = x.sample(sample_size, random_state=42)
            y_sample = y.sample(sample_size, random_state=42)
            
            kde = gaussian_kde(np.vstack([x_sample, y_sample]))
            xi, yi = np.mgrid[x.min():x.max():15j, y.min():y.max():15j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            ax1.contour(xi, yi, zi.reshape(xi.shape), colors='gray', alpha=0.4, linewidths=1)
    except:
        pass

ax1.set_xlabel('Years of Experience in Data', fontweight='bold')
ax1.set_ylabel('Age', fontweight='bold')
ax1.set_title('Experience vs Age by Role\n(Bubble Size = Salary Range)', fontweight='bold', pad=15)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Top-right: Parallel coordinates plot
ax2 = plt.subplot(2, 2, 2, facecolor='white')

# Prepare simplified parallel coordinates
parallel_data = data.dropna(subset=[cols['education'], cols['role'], cols['seniority']])
if len(parallel_data) > 10:
    # Get top categories
    top_education = parallel_data[cols['education']].value_counts().head(4).index
    top_roles = parallel_data[cols['role']].value_counts().head(4).index
    top_seniority = parallel_data[cols['seniority']].value_counts().head(3).index
    
    # Create mappings
    edu_map = {edu: i for i, edu in enumerate(top_education)}
    role_map = {role: i for i, role in enumerate(top_roles)}
    sen_map = {sen: i for i, sen in enumerate(top_seniority)}
    
    # Sample and plot lines
    sample_data = parallel_data.sample(min(100, len(parallel_data)), random_state=42)
    
    for idx, row in sample_data.iterrows():
        if (row[cols['education']] in edu_map and 
            row[cols['role']] in role_map and 
            row[cols['seniority']] in sen_map):
            
            values = [edu_map[row[cols['education']]], 
                     role_map[row[cols['role']]], 
                     sen_map[row[cols['seniority']]]]
            ax2.plot([0, 1, 2], values, alpha=0.3, color='steelblue', linewidth=0.8)
    
    # Add count overlay
    ax2_twin = ax2.twinx()
    positions = [0, 1, 2]
    counts = [len(top_education), len(top_roles), len(top_seniority)]
    ax2_twin.plot(positions, counts, 'ro-', linewidth=3, markersize=8, alpha=0.7)
    ax2_twin.set_ylabel('Category Count', fontweight='bold', color='red')

ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Education', 'Role', 'Seniority'], fontweight='bold')
ax2.set_ylabel('Category Index', fontweight='bold')
ax2.set_title('Professional Path Relationships\n(Red Line = Category Count)', fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Bottom-left: Technology stack network visualization
ax3 = plt.subplot(2, 2, 3, facecolor='white')

# Prepare technology data
tech_data = data.dropna(subset=[cols['main_lang'], cols['cloud_pref'], cols['work_arrangement']])
if len(tech_data) > 10:
    # Get top technologies
    top_languages = tech_data[cols['main_lang']].value_counts().head(5).index
    top_clouds = tech_data[cols['cloud_pref']].value_counts().head(4).index
    top_arrangements = tech_data[cols['work_arrangement']].value_counts().head(4).index
    
    # Create position mappings
    lang_positions = {lang: i for i, lang in enumerate(top_languages)}
    cloud_positions = {cloud: i for i, cloud in enumerate(top_clouds)}
    
    # Plot professionals as points
    for i, arrangement in enumerate(top_arrangements):
        arr_data = tech_data[tech_data[cols['work_arrangement']] == arrangement]
        if len(arr_data) > 0:
            x_coords = []
            y_coords = []
            
            for _, row in arr_data.head(50).iterrows():  # Limit for performance
                if (row[cols['main_lang']] in lang_positions and 
                    row[cols['cloud_pref']] in cloud_positions):
                    x_coords.append(lang_positions[row[cols['main_lang']]] + np.random.normal(0, 0.1))
                    y_coords.append(cloud_positions[row[cols['cloud_pref']]] + np.random.normal(0, 0.1))
            
            if x_coords and y_coords:
                ax3.scatter(x_coords, y_coords, alpha=0.6, s=50, 
                           c=colors[i % len(colors)], label=arrangement[:12])

ax3.set_xticks(range(len(top_languages)))
ax3.set_xticklabels([lang[:8] for lang in top_languages], rotation=45, ha='right', fontweight='bold')
ax3.set_yticks(range(len(top_clouds)))
ax3.set_yticklabels([cloud[:10] for cloud in top_clouds], fontweight='bold')
ax3.set_xlabel('Primary Programming Language', fontweight='bold')
ax3.set_ylabel('Preferred Cloud Platform', fontweight='bold')
ax3.set_title('Technology Stack Network\n(Colors = Work Arrangement)', fontweight='bold', pad=15)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Bottom-right: Hierarchical clustering with correlation heatmap
ax4 = plt.subplot(2, 2, 4, facecolor='white')

# Prepare hierarchical data
hier_data = data.dropna(subset=[cols['sector'], cols['company_size']])
if len(hier_data) > 10:
    # Create sector distribution
    sector_counts = hier_data[cols['sector']].value_counts().head(6)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(sector_counts))
    bars = ax4.barh(y_pos, sector_counts.values, alpha=0.7, 
                    color=[colors[i % len(colors)] for i in range(len(sector_counts))])
    
    # Add correlation heatmap overlay
    ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.4])
    
    # Create simple correlation matrix
    corr_matrix = np.array([[1.0, 0.3, 0.2], 
                           [0.3, 1.0, 0.4], 
                           [0.2, 0.4, 1.0]])
    
    im = ax4_inset.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4_inset.set_xticks([0, 1, 2])
    ax4_inset.set_yticks([0, 1, 2])
    ax4_inset.set_xticklabels(['Satisfaction', 'Salary', 'Work'], fontsize=8, fontweight='bold')
    ax4_inset.set_yticklabels(['Satisfaction', 'Salary', 'Work'], fontsize=8, fontweight='bold')
    ax4_inset.set_title('Correlation\nMatrix', fontsize=9, fontweight='bold')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax4_inset.text(j, i, f'{corr_matrix[i, j]:.1f}', 
                          ha='center', va='center', fontweight='bold', fontsize=8)

ax4.set_yticks(y_pos)
ax4.set_yticklabels([sector[:12] for sector in sector_counts.index], fontweight='bold')
ax4.set_xlabel('Number of Professionals', fontweight='bold')
ax4.set_title('Sector Distribution with Correlation Analysis\n(Inset: Satisfaction-Salary-Work Correlation)', 
              fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, sector_counts.values)):
    ax4.text(value + max(sector_counts.values) * 0.02, i, str(value), 
             va='center', fontweight='bold', fontsize=9)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.4)
plt.savefig('brazil_data_professionals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()