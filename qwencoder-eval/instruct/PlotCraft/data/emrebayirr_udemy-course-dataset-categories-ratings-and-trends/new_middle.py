import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Load data
df = pd.read_csv("udemy_courses.csv")

# Data preprocessing
df = df.dropna(subset=["rating", "num_subscribers", "num_reviews", "instructor_names", "instructional_level"])

# Create figure with 2x2 subplots with larger size and better spacing
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(26, 20))
fig.patch.set_facecolor("white")

# Consistent color palette across all subplots
primary_blue = "#2E86AB"
primary_green = "#2ECC71"
primary_orange = "#F18F01"
primary_red = "#C73E1D"

# Color scheme for instructional levels (used in lollipop)
colors_level = {"Beginner Level": "#2E86AB", "Intermediate Level": "#27AE60", "Advanced Level": "#8E44AD", "All Levels": "#F39C12", "Expert Level": "#E74C3C"}
edge_colors_level = {k: "#1B1B1B" for k in colors_level}  # dark edges for definition


# Helper function to format large numbers
def format_number(num):
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.0f}K"
    else:
        return str(int(num))


# 1. Top-left: Horizontal bar chart with red-to-green rating colors and review scatter points
top_courses = df.nlargest(15, "num_subscribers").copy()
y_pos = np.arange(len(top_courses))

# Rating-based colors using a red-to-green gradient (low=red, high=green)
min_r, max_r = top_courses["rating"].min(), top_courses["rating"].max()
rating_normalized = (top_courses["rating"] - min_r) / (max_r - min_r + 1e-9)
bar_colors = plt.cm.RdYlGn(rating_normalized)  # red->yellow->green

bars = ax1.barh(y_pos, top_courses["num_subscribers"], color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.7)

# Add scatter points for reviews at bar ends (size scaled for readability)
min_rev, max_rev = top_courses["num_reviews"].min(), top_courses["num_reviews"].max()
review_sizes = np.interp(top_courses["num_reviews"], [min_rev, max_rev], [60, 500])  # area in points^2
for i, (subs, s) in enumerate(zip(top_courses["num_subscribers"], review_sizes)):
    ax1.scatter(subs, i, s=s, color=primary_red, alpha=0.9, zorder=3, edgecolors="white", linewidth=1)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([title[:40] + "..." if len(title) > 40 else title for title in top_courses["title"]], fontsize=10)
ax1.set_xlabel("Number of Subscribers", fontsize=12, fontweight="bold")
ax1.set_title("Top 15 Courses by Subscribers (Bars colored by Rating; Red Circles size ∝ Reviews)", fontsize=13, fontweight="bold", pad=15)
ax1.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
ax1.invert_yaxis()  # Highest at top

# Colorbar to indicate rating scale
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=min_r, vmax=max_r))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Rating", fontsize=11)

# Legend for review scatter
scatter_legend = ax1.scatter([], [], s=200, color=primary_red, alpha=0.9, edgecolors="white", linewidth=1)
ax1.legend([scatter_legend], ["Review Count (circle size)"], loc="lower right", fontsize=9, framealpha=0.9)

# 2. Top-right: Proper lollipop chart (STEM + CIRCLES at end)
# Circles: size ∝ num_subscribers, color = instructional_level; add legends for both size and color
top_rated = df.nlargest(12, "rating").copy()
y_pos2 = np.arange(len(top_rated))

# Draw stems and circles
subs_min, subs_max = top_rated["num_subscribers"].min(), top_rated["num_subscribers"].max()
# Choose a visible size range for scatter area (in points^2)
size_min, size_max = 120, 1600

for i, (rating, subs, level) in enumerate(zip(top_rated["rating"], top_rated["num_subscribers"], top_rated["instructional_level"])):
    # Stem from 0 to rating
    ax2.plot([0, rating], [i, i], color="#555555", linewidth=2.5, alpha=0.7, solid_capstyle="round", zorder=1)
    # Circle at end (size and color mapped)
    s_area = np.interp(subs, [subs_min, subs_max], [size_min, size_max])
    face_color = colors_level.get(level, "#7F8C8D")
    edge_color = edge_colors_level.get(level, "#1B1B1B")
    ax2.scatter(rating, i, s=s_area, color=face_color, alpha=0.95, edgecolors=edge_color, linewidth=1.5, zorder=3)

ax2.set_yticks(y_pos2)
ax2.set_yticklabels([title[:40] + "..." if len(title) > 40 else title for title in top_rated["title"]], fontsize=10)
ax2.set_xlabel("Rating", fontsize=12, fontweight="bold")
ax2.set_title("Top 12 Courses by Rating (Stem length = Rating; Circle size = Subscribers; Color = Level)", fontsize=13, fontweight="bold", pad=15)
ax2.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
ax2.set_xlim(0, 5.0)
ax2.invert_yaxis()  # Highest rated at top

# Legends for color (instructional level)
level_handles, level_labels = [], []
for level, color in colors_level.items():
    if level in top_rated["instructional_level"].values:
        h = ax2.scatter([], [], s=300, color=color, edgecolors=edge_colors_level[level], linewidth=1.5)
        level_handles.append(h)
        level_labels.append(level)
legend_levels = ax2.legend(level_handles, level_labels, title="Instructional Level", loc="upper left", fontsize=9, title_fontsize=10, framealpha=0.9)

# Legend for sizes (subscribers)
rep_counts = [np.percentile(top_rated["num_subscribers"], q) for q in [25, 50, 90]]
rep_sizes = [np.interp(v, [subs_min, subs_max], [size_min, size_max]) for v in rep_counts]
size_handles = [ax2.scatter([], [], s=s, color="#BBBBBB", edgecolors="#555555") for s in rep_sizes]
size_labels = [f"{format_number(int(v))} subs" for v in rep_counts]
legend_sizes = ax2.legend(size_handles, size_labels, title="Subscribers (circle size)", loc="lower left", fontsize=9, title_fontsize=10, framealpha=0.9)
ax2.add_artist(legend_levels)  # ensure both legends show

# 3. Bottom-left: Stacked horizontal bar chart (Top 10 instructors by total subscribers; segments = courses)
instructor_data = df.groupby("instructor_names").agg({"num_subscribers": "sum", "title": "count"}).reset_index()
instructor_data = instructor_data[instructor_data["title"] >= 2]  # instructors with at least 2 courses
top_instructors = instructor_data.nlargest(10, "num_subscribers").copy()

# Reorder: move the bottom-most bar to the top, keep others' relative order unchanged
if len(top_instructors) > 1:
    top_instructors = pd.concat([top_instructors.tail(1), top_instructors.iloc[:-1]], ignore_index=True)

# Color palette for different courses
course_colors_palette = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#27AE60", "#2980B9", "#8E44AD", "#F1C40F", "#D35400", "#95A5A6"]

instructor_courses = {}
course_colors = {}
color_idx = 0

# Build courses dictionary in the (reordered) instructor order
for instructor in top_instructors["instructor_names"]:
    courses = df[df["instructor_names"] == instructor].nlargest(5, "num_subscribers")
    instructor_courses[instructor] = courses
    for course_title in courses["title"]:
        if course_title not in course_colors:
            course_colors[course_title] = course_colors_palette[color_idx % len(course_colors_palette)]
            color_idx += 1

y_pos3 = np.arange(len(top_instructors))

legend_courses = []
legend_colors = []
for i, instructor in enumerate(top_instructors["instructor_names"]):
    courses = instructor_courses[instructor]
    left = 0
    for _, course in courses.iterrows():
        width = course["num_subscribers"]
        color = course_colors[course["title"]]
        ax3.barh(i, width, left=left, color=color, alpha=0.9, edgecolor="white", linewidth=0.8)
        left += width
        # Collect a few major courses for legend
        label = course["title"][:25] + "..." if len(course["title"]) > 25 else course["title"]
        if label not in legend_courses and width > 100000:
            legend_courses.append(label)
            legend_colors.append(color)

ax3.set_yticks(y_pos3)
ax3.set_yticklabels([name[:30] + "..." if len(name) > 30 else name for name in top_instructors["instructor_names"]], fontsize=10)
ax3.set_xlabel("Total Subscribers", fontsize=12, fontweight="bold")
ax3.set_title("Top 10 Instructors by Total Subscribers (Segments represent top courses)", fontsize=13, fontweight="bold", pad=15)
ax3.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
ax3.invert_yaxis()  # With the reorder, the previously bottom bar now appears at the top

if legend_courses:
    legend_handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.9) for c in legend_colors[:6]]
    ax3.legend(legend_handles, legend_courses[:6], title="Major Courses", loc="lower right", fontsize=8, title_fontsize=9, framealpha=0.9)

# 4. Bottom-right: Slope chart (Subscribers rank vs Reviews rank)
top_by_subs = df.nlargest(8, "num_subscribers").reset_index(drop=True)
top_by_subs["subs_rank"] = range(1, 9)

# Compute review rank among these top courses
top_by_subs["review_rank"] = top_by_subs["num_reviews"].rank(ascending=False, method="min").astype(int)

for _, row in top_by_subs.iterrows():
    subs_rank = row["subs_rank"]
    review_rank = row["review_rank"]
    rank_change = abs(subs_rank - review_rank)
    if rank_change <= 2:
        linestyle = "-"
        alpha = 0.85
        linewidth = 2.8
        color = primary_blue
    else:
        linestyle = "--"
        alpha = 0.95
        linewidth = 3.2
        color = primary_red
    ax4.plot([1, 2], [subs_rank, review_rank], linestyle=linestyle, color=color, alpha=alpha, linewidth=linewidth)

# Left y-axis labels (subscribers rank) show course titles
ax4.set_yticks(range(1, 9))
ax4.set_yticklabels([f"{i}. {row['title'][:25]}..." if len(row["title"]) > 25 else f"{i}. {row['title']}" for i, (_, row) in enumerate(top_by_subs.iterrows(), 1)], fontsize=9)

# Right y-axis for review ranks
ax4_right = ax4.twinx()
ax4_right.set_ylim(ax4.get_ylim())
ax4_right.set_yticks([row["review_rank"] for _, row in top_by_subs.iterrows()])
ax4_right.set_yticklabels([f"#{row['review_rank']}" for _, row in top_by_subs.iterrows()], fontsize=9)
ax4_right.set_ylabel("Reviews Rank", fontsize=12, fontweight="bold")

ax4.set_xlim(0.8, 2.2)
ax4.set_ylim(0.5, 8.5)
ax4.set_xticks([1, 2])
ax4.set_xticklabels(["Subscribers\nRanking", "Reviews\nRanking"], fontsize=11, fontweight="bold")
ax4.set_title("Ranking Comparison: Subscribers vs Reviews (Blue: Stable ≤2; Red: Change >2)", fontsize=13, fontweight="bold", pad=15)
ax4.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
ax4.invert_yaxis()
ax4_right.invert_yaxis()

# Legend for slope chart line styles
stable_line = plt.Line2D([0], [0], color=primary_blue, linewidth=2.8, linestyle="-")
change_line = plt.Line2D([0], [0], color=primary_red, linewidth=3.2, linestyle="--")
ax4.legend([stable_line, change_line], ["Stable Ranking (≤2 change)", "Significant Change (>2)"], loc="center right", fontsize=9, framealpha=0.9)

# Main title moved upward to avoid overlap
fig.suptitle("Comprehensive Udemy Course Performance Analysis", fontsize=18, fontweight="bold", y=0.92)

# Improved layout with better spacing to prevent overlaps
plt.tight_layout(rect=[0.06, 0.06, 0.96, 0.92])  # leave space for the higher suptitle
# plt.show()


plt.savefig("new_middle.png", dpi=400, bbox_inches="tight")
