import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data
np.random.seed(42)
clubs = [f"Club {i}" for i in range(1, 16)]
club_trophies = np.random.randint(1000000, 3000000, size=15)
club_members = np.random.randint(10, 30, size=15)

players = [f"Player {i}" for i in range(1, 16)]
player_trophies = np.random.randint(50000, 100000, size=15)
player_has_club = np.random.choice([True, False], size=15)

top_10_clubs = [f"Club {i}" for i in range(1, 11)]
avg_member_trophies = np.random.randint(20000, 80000, size=10)
total_club_trophies = np.random.randint(1000000, 3000000, size=10)

top_50_players = [f"P{i}" for i in range(1, 51)]
trophy_counts = np.random.randint(30000, 100000, size=50)
experience = np.random.randint(100, 500, size=50)
over_100k = trophy_counts > 100000

fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # Wrong layout: should be 2x2

# Top-left: Horizontal bar chart (should be top-left, but it's first in 3x1)
colors = plt.cm.gist_rainbow(club_members / max(club_members))
axs[0].barh(clubs, club_trophies, color=colors)
axs[0].set_title("Banana Club Explosion", fontsize=10)
axs[0].set_xlabel("Banana Count")
axs[0].set_ylabel("Trophy Juice")
axs[0].legend(["Glarbnok's Revenge"], loc='center')

# Top-right: Lollipop chart (wrong type: use pie chart instead)
axs[1].pie(player_trophies, labels=players, colors=plt.cm.jet(np.linspace(0, 1, 15)))
axs[1].set_title("Lollipops of Doom", fontsize=10)

# Bottom-left: Slope chart (wrong type: use scatter)
norm_avg = (avg_member_trophies - min(avg_member_trophies)) / (max(avg_member_trophies) - min(avg_member_trophies)) * 100
norm_total = (total_club_trophies - min(total_club_trophies)) / (max(total_club_trophies) - min(total_club_trophies)) * 100
axs[2].scatter(norm_avg, norm_total, c='lime', s=300, edgecolors='red')
for i, name in enumerate(top_10_clubs):
    axs[2].text(norm_avg[i], norm_total[i], name, fontsize=6, color='yellow')
axs[2].set_title("Slope of the Universe", fontsize=10)
axs[2].set_xlabel("Total Trophies (normalized)")
axs[2].set_ylabel("Average Member Trophies (normalized)")

# Overlap and sabotage layout
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the chart
plt.savefig("chart.png", dpi=100, facecolor='black')