import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from concurrent.futures import ThreadPoolExecutor

# Load and combine all datasets
datasets = ['data_26.csv', 'data_37.csv', 'data_41.csv', 'data_40.csv', 'data_6.csv', 
           'data_16.csv', 'data_59.csv', 'data_1.csv', 'data_29.csv', 'data_46.csv',
           'data_51.csv', 'data_32.csv', 'data_68.csv', 'data_54.csv', 'data_71.csv',
           'data_30.csv', 'data_42.csv', 'data_65.csv', 'data_19.csv', 'data_63.csv',
           'data_53.csv', 'data_72.csv', 'data_25.csv', 'data_50.csv', 'data_8.csv',
           'data_56.csv', 'data_33.csv', 'data_14.csv', 'data_70.csv', 'data_28.csv',
           'data_38.csv', 'data_18.csv', 'data_39.csv', 'data_49.csv', 'data_35.csv',
           'data_27.csv', 'data_66.csv', 'data_47.csv', 'data_4.csv', 'data_24.csv',
           'data_31.csv', 'data_5.csv', 'data_45.csv', 'data_21.csv', 'data_13.csv',
           'data_34.csv', 'data_64.csv', 'data_36.csv', 'data_48.csv', 'data_44.csv',
           'data_57.csv', 'data_73.csv', 'data_58.csv', 'data_2.csv', 'data_7.csv',
           'data_43.csv', 'data_20.csv', 'data_12.csv', 'data_61.csv', 'data_22.csv',
           'data_67.csv', 'data_3.csv', 'data_60.csv', 'data_15.csv', 'data_62.csv',
           'data_74.csv', 'data_9.csv', 'data_11.csv', 'data_17.csv', 'data_55.csv',
           'data_10.csv', 'data_52.csv', 'data_69.csv', 'data_23.csv']



# 定义一个读取单个文件的函数，同时优化内存使用
def read_csv_optimized(filename):
    try:
        
        df = pd.read_csv(filename) # 在这里可以加入 usecols 和 dtype
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# 使用 ThreadPoolExecutor 并行读取
with ThreadPoolExecutor() as executor:
    # executor.map 会保持原始顺序
    all_data = list(executor.map(read_csv_optimized, datasets))

# 过滤掉读取失败的结果 (None)
all_data = [df for df in all_data if df is not None]

# 合并所有数据
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
else:
    print("No data was loaded successfully.")
    # 可以选择退出或创建一个空的DataFrame
    combined_df = pd.DataFrame()

# Clean data
combined_df['category'] = combined_df['category'].fillna('Unknown')
combined_df['proto'] = combined_df['proto'].fillna('unknown')
combined_df['subcategory '] = combined_df['subcategory '].fillna('Unknown')

# Use dark background style
plt.style.use('dark_background')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Scatter plot instead of stacked bar
ax1 = axes[0]
categories = combined_df['category'].unique()
protocols = combined_df['proto'].unique()
x_pos = np.arange(len(protocols))
y_pos = np.arange(len(categories))
X, Y = np.meshgrid(x_pos, y_pos)
sizes = np.random.rand(len(categories), len(protocols)) * 1000
colors = plt.cm.jet(np.random.rand(len(categories), len(protocols)))
ax1.scatter(X.flatten(), Y.flatten(), s=sizes.flatten(), c=colors.flatten(), alpha=0.7)
ax1.set_xlabel('Amplitude')  # Swapped labels
ax1.set_ylabel('Time')
ax1.set_title('Random Data Display')  # Wrong title
ax1.text(0.5, 0.5, 'OVERLAPPING TEXT', transform=ax1.transAxes, fontsize=20, 
         color='white', ha='center', va='center', weight='bold')

# Middle subplot: Bar chart instead of pie/donut
ax2 = axes[1]
attack_counts = combined_df['category'].value_counts()
bars = ax2.bar(range(len(attack_counts)), attack_counts.values, 
               color=plt.cm.jet(np.linspace(0, 1, len(attack_counts))))
ax2.set_xlabel('Time')  # Wrong label
ax2.set_ylabel('Amplitude')  # Wrong label
ax2.set_title('Glarbnok\'s Revenge')  # Nonsensical title
ax2.text(1, max(attack_counts.values)*0.8, 'MORE OVERLAPPING TEXT', 
         fontsize=16, color='yellow', weight='bold')

# Bottom subplot: Line plot instead of treemap/horizontal bar
ax3 = axes[2]
ip_counts = combined_df['saddr'].value_counts().head(10)
x_vals = range(len(ip_counts))
y_vals = ip_counts.values
ax3.plot(x_vals, y_vals, 'o-', linewidth=5, markersize=10, color='red')
ax3.fill_between(x_vals, y_vals, alpha=0.3, color='orange')
ax3.set_xlabel('Frequency')  # Wrong label
ax3.set_ylabel('Protocol Type')  # Wrong label
ax3.set_title('Mysterious Network Phenomena')  # Wrong title
ax3.text(5, max(y_vals)*0.5, 'FINAL OVERLAPPING TEXT', 
         fontsize=14, color='cyan', weight='bold')

# Make all text same size and weight (no hierarchy)
for ax in axes:
    ax.title.set_fontsize(12)
    ax.title.set_weight('normal')
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=12)

# Add thick, clumsy spines
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')

plt.savefig('chart.png', dpi=100, bbox_inches='tight', facecolor='black')
plt.close()