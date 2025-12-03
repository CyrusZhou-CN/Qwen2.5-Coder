import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import Counter
import matplotlib.patches as patches
from matplotlib.patches import Wedge
import squarify

# Load and preprocess data
df = pd.read_csv('ner.csv')

# Parse the Tag column which appears to be in string list format
def parse_tags(tag_str):
    try:
        return ast.literal_eval(tag_str)
    except:
        return []

df['parsed_tags'] = df['Tag'].apply(parse_tags)

# Flatten all tags and count frequencies
all_tags = []
for tags in df['parsed_tags']:
    all_tags.extend(tags)

tag_counts = Counter(all_tags)

# Remove 'O' tags and get entity types
entity_tags = {tag: count for tag, count in tag_counts.items() if tag != 'O'}

# Extract entity types (remove B- and I- prefixes)
entity_types = {}
for tag, count in entity_tags.items():
    if tag.startswith(('B-', 'I-')):
        entity_type = tag[2:]
        if entity_type not in entity_types:
            entity_types[entity_type] = 0
        entity_types[entity_type] += count

# Get top 3 most frequent entity types
top_3_entities = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:3]
top_3_entity_names = [entity[0] for entity in top_3_entities]

# Calculate B- and I- distributions for top 3 entities
bi_distributions = {}
for entity in top_3_entity_names:
    b_tag = f'B-{entity}'
    i_tag = f'I-{entity}'
    b_count = tag_counts.get(b_tag, 0)
    i_count = tag_counts.get(i_tag, 0)
    bi_distributions[entity] = {'B': b_count, 'I': i_count}

# Calculate sentence-level statistics
sentence_stats = []
for idx, row in df.iterrows():
    tags = row['parsed_tags']
    sentence_length = len(tags)
    
    # Count entities per sentence
    entity_counts = {}
    for entity in entity_types.keys():
        entity_counts[entity] = sum(1 for tag in tags if tag.endswith(f'-{entity}'))
    
    sentence_stats.append({
        'sentence': row['Sentence #'],
        'length': sentence_length,
        'entity_counts': entity_counts
    })

# Calculate variability (coefficient of variation) for each entity type
entity_variability = {}
for entity in entity_types.keys():
    counts = [stat['entity_counts'][entity] for stat in sentence_stats]
    if np.mean(counts) > 0:
        cv = np.std(counts) / np.mean(counts)
        entity_variability[entity] = cv

# Get top 3 entities with highest variability
top_var_entities = sorted(entity_variability.items(), key=lambda x: x[1], reverse=True)[:3]
top_var_entity_names = [entity[0] for entity in top_var_entities]

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

# Top row: Stacked bar charts with line plots for top 3 frequent entities
for i, entity in enumerate(top_3_entity_names):
    ax = plt.subplot(3, 3, i + 1)
    
    # Prepare data for stacked bar chart
    sentences = list(range(1, min(21, len(sentence_stats) + 1)))  # First 20 sentences
    b_counts = []
    i_counts = []
    cumulative_pct = []
    
    total_entity_count = bi_distributions[entity]['B'] + bi_distributions[entity]['I']
    running_total = 0
    
    for sent_idx in sentences:
        if sent_idx <= len(sentence_stats):
            tags = df.iloc[sent_idx-1]['parsed_tags']
            b_count = sum(1 for tag in tags if tag == f'B-{entity}')
            i_count = sum(1 for tag in tags if tag == f'I-{entity}')
            b_counts.append(b_count)
            i_counts.append(i_count)
            
            running_total += (b_count + i_count)
            cumulative_pct.append((running_total / total_entity_count) * 100 if total_entity_count > 0 else 0)
        else:
            b_counts.append(0)
            i_counts.append(0)
            cumulative_pct.append(cumulative_pct[-1] if cumulative_pct else 0)
    
    # Create stacked bar chart
    width = 0.8
    ax.bar(sentences, b_counts, width, label=f'B-{entity}', color='#2E86AB', alpha=0.8)
    ax.bar(sentences, i_counts, width, bottom=b_counts, label=f'I-{entity}', color='#A23B72', alpha=0.8)
    
    # Overlay line plot
    ax2 = ax.twinx()
    ax2.plot(sentences, cumulative_pct, color='#F18F01', linewidth=3, marker='o', markersize=4, label='Cumulative %')
    
    ax.set_title(f'**{entity.upper()} Entity Distribution**', fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Sentence Number', fontsize=10)
    ax.set_ylabel('Tag Count', fontsize=10)
    ax2.set_ylabel('Cumulative %', fontsize=10, color='#F18F01')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

# Middle row: Pie charts with donut holes and bar overlays
for i, entity in enumerate(top_3_entity_names):
    ax = plt.subplot(3, 3, i + 4)
    
    # Calculate proportions for pie chart
    entity_proportions = []
    entity_labels = []
    avg_sentence_lengths = []
    
    for ent_type in entity_types.keys():
        count = entity_types[ent_type]
        entity_proportions.append(count)
        entity_labels.append(ent_type)
        
        # Calculate average sentence length for sentences containing this entity
        lengths = []
        for stat in sentence_stats:
            if stat['entity_counts'][ent_type] > 0:
                lengths.append(stat['length'])
        avg_length = np.mean(lengths) if lengths else 0
        avg_sentence_lengths.append(avg_length)
    
    # Create donut pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(entity_labels)))
    wedges, texts, autotexts = ax.pie(entity_proportions, labels=entity_labels, autopct='%1.1f%%',
                                      colors=colors, pctdistance=0.85, startangle=90)
    
    # Create donut hole
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)
    
    # Add bar chart around circumference (simplified representation)
    for j, (wedge, avg_len) in enumerate(zip(wedges, avg_sentence_lengths)):
        angle = wedge.theta1 + (wedge.theta2 - wedge.theta1) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        
        # Draw small bar
        bar_height = avg_len / max(avg_sentence_lengths) * 0.3 if max(avg_sentence_lengths) > 0 else 0
        rect = patches.Rectangle((x-0.05, y), 0.1, bar_height, 
                               facecolor=colors[j], alpha=0.7, transform=ax.transData)
        ax.add_patch(rect)
    
    ax.set_title(f'**{entity.upper()} Composition & Avg Sentence Length**', fontweight='bold', fontsize=11, pad=15)

# Bottom row: Treemap with scatter overlay for high variability entities
for i, entity in enumerate(top_var_entity_names):
    ax = plt.subplot(3, 3, i + 7)
    
    # Prepare treemap data
    entity_freq_data = []
    for stat in sentence_stats:
        count = stat['entity_counts'][entity]
        if count > 0:
            entity_freq_data.append(count)
    
    if entity_freq_data:
        freq_counter = Counter(entity_freq_data)
        sizes = list(freq_counter.values())
        labels = [f'{k} occurrences\n({v} sentences)' for k, v in freq_counter.items()]
        
        # Create treemap
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
        
        # Overlay scatter plot showing sentence position vs entity density
        positions = []
        densities = []
        for idx, stat in enumerate(sentence_stats[:50]):  # First 50 sentences
            if stat['entity_counts'][entity] > 0:
                positions.append(idx + 1)
                density = stat['entity_counts'][entity] / stat['length'] if stat['length'] > 0 else 0
                densities.append(density)
        
        if positions and densities:
            # Normalize positions and densities to fit in treemap space
            norm_pos = np.array(positions) / max(positions) if positions else []
            norm_dens = np.array(densities) / max(densities) if densities else []
            
            # Add scatter points
            ax.scatter(norm_pos * 0.8 - 0.4, norm_dens * 0.8 - 0.4, 
                      c='red', s=30, alpha=0.7, edgecolors='darkred', linewidth=1)
    
    ax.set_title(f'**{entity.upper()} Frequency Hierarchy & Position Density**', fontweight='bold', fontsize=11, pad=15)
    ax.axis('off')

# Overall title and layout
fig.suptitle('**Comprehensive NER Tag Composition Analysis**', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()