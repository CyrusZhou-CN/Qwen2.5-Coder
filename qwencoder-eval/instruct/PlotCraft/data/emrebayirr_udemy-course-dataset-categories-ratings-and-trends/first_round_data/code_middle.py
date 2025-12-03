import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns

# Load data
df = pd.read_csv('udemy_courses.csv')

# Data preprocessing
df = df.dropna(subset=['rating', 'num_subscribers', 'num_reviews', 'instructor_names', 'instructional_level'])

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
colors_rating = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 15))
colors_level = {'Beginner Level': '#FF6B6B', 'Intermediate Level': '#4ECDC4', 
                'Advanced Level': '#45B7D1', 'All Levels': '#96CEB4', 
                'Expert Level': '#FFEAA7'}

# 1. Top-left: Horizontal bar chart with rating colors and review scatter points
top_courses = df.nlargest(15, 'num_subscribers')
y_pos = np.arange(len(top_courses))

# Create bars with rating-based colors
bars = ax1.barh(y_pos, top_courses['num_subscribers'], 
                color=[colors_rating[int((rating-3.5)*10)] for rating in top_courses['rating']])

# Add scatter points for reviews at bar ends
for i, (subs, reviews) in enumerate(zip(top_courses['num_subscribers'], top_courses['num_reviews'])):
    ax1.scatter(subs, i, s=np.sqrt(reviews/100), color='darkred', alpha=0.7, zorder=3)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([title[:50] + '...' if len(title) > 50 else title 
                     for title in top_courses['title']], fontsize=10)
ax1.set_xlabel('Number of Subscribers', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Courses by Subscribers\n(Color: Rating, Dots: Reviews)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3)

# 2. Top-right: Lollipop chart with rating and subscriber size
top_rated = df.nlargest(12, 'rating')
y_pos2 = np.arange(len(top_rated))

# Create stems
for i, rating in enumerate(top_rated['rating']):
    ax2.plot([0, rating], [i, i], 'k-', linewidth=2, alpha=0.6)

# Create circles with size based on subscribers and color by level
for i, (rating, subs, level) in enumerate(zip(top_rated['rating'], 
                                              top_rated['num_subscribers'], 
                                              top_rated['instructional_level'])):
    circle_size = np.sqrt(subs/10000)
    color = colors_level.get(level, '#95A5A6')
    ax2.scatter(rating, i, s=circle_size, color=color, alpha=0.8, 
                edgecolors='black', linewidth=1, zorder=3)

ax2.set_yticks(y_pos2)
ax2.set_yticklabels([title[:40] + '...' if len(title) > 40 else title 
                     for title in top_rated['title']], fontsize=10)
ax2.set_xlabel('Rating', fontsize=12, fontweight='bold')
ax2.set_title('Top 12 Courses by Rating\n(Circle Size: Subscribers, Color: Level)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(3.5, 5.0)

# 3. Bottom-left: Stacked horizontal bar chart by instructor
instructor_data = df.groupby('instructor_names').agg({
    'num_subscribers': 'sum',
    'title': 'count'
}).reset_index()
instructor_data = instructor_data[instructor_data['title'] >= 2]  # At least 2 courses
top_instructors = instructor_data.nlargest(10, 'num_subscribers')

# Get course details for each instructor
instructor_courses = {}
for instructor in top_instructors['instructor_names']:
    courses = df[df['instructor_names'] == instructor].nlargest(5, 'num_subscribers')
    instructor_courses[instructor] = courses

y_pos3 = np.arange(len(top_instructors))
colors_courses = plt.cm.Set3(np.linspace(0, 1, 12))

# Create stacked bars
for i, instructor in enumerate(top_instructors['instructor_names']):
    courses = instructor_courses[instructor]
    left = 0
    for j, (_, course) in enumerate(courses.iterrows()):
        width = course['num_subscribers']
        ax3.barh(i, width, left=left, color=colors_courses[j % len(colors_courses)], 
                 alpha=0.8, edgecolor='white', linewidth=0.5)
        left += width

ax3.set_yticks(y_pos3)
ax3.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                     for name in top_instructors['instructor_names']], fontsize=10)
ax3.set_xlabel('Total Subscribers', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Instructors by Total Subscribers\n(Segments: Individual Courses)', 
              fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='x', alpha=0.3)

# 4. Bottom-right: Slope chart comparing rankings
top_by_subs = df.nlargest(8, 'num_subscribers').reset_index(drop=True)
top_by_subs['subs_rank'] = range(1, 9)

# Get review rankings for these courses
review_ranks = []
for _, course in top_by_subs.iterrows():
    rank = (df['num_reviews'] > course['num_reviews']).sum() + 1
    review_ranks.append(rank)
top_by_subs['review_rank'] = review_ranks

# Create slope chart
for i, row in top_by_subs.iterrows():
    subs_rank = row['subs_rank']
    review_rank = row['review_rank']
    
    # Determine line style based on rank change
    rank_change = abs(subs_rank - review_rank)
    if rank_change <= 2:
        linestyle = '-'
        alpha = 0.8
        linewidth = 2
        color = '#2ECC71'  # Green for stable
    else:
        linestyle = '--'
        alpha = 0.9
        linewidth = 3
        color = '#E74C3C'  # Red for significant change
    
    ax4.plot([1, 2], [subs_rank, review_rank], linestyle=linestyle, 
             color=color, alpha=alpha, linewidth=linewidth)
    
    # Add course labels
    ax4.text(0.95, subs_rank, f"{subs_rank}. {row['title'][:25]}...", 
             ha='right', va='center', fontsize=9)
    ax4.text(2.05, review_rank, f"{review_rank}", 
             ha='left', va='center', fontsize=9, fontweight='bold')

ax4.set_xlim(0.5, 2.5)
ax4.set_ylim(0.5, max(max(top_by_subs['subs_rank']), max(top_by_subs['review_rank'])) + 0.5)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(['Subscribers Rank', 'Reviews Rank'], fontsize=12, fontweight='bold')
ax4.set_ylabel('Rank Position', fontsize=12, fontweight='bold')
ax4.set_title('Ranking Comparison: Subscribers vs Reviews\n(Green: Stable, Red: Significant Change)', 
              fontsize=14, fontweight='bold', pad=20)
ax4.grid(axis='y', alpha=0.3)
ax4.invert_yaxis()

# Add legends
# Legend for instructional levels (top-right)
level_handles = [plt.scatter([], [], s=100, color=color, alpha=0.8, edgecolors='black') 
                 for color in colors_level.values()]
ax2.legend(level_handles, colors_level.keys(), title='Instructional Level', 
           loc='lower right', fontsize=9, title_fontsize=10)

# Overall title
fig.suptitle('Comprehensive Udemy Course Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
plt.show()