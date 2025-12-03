import numpy as np
import matplotlib.pyplot as plt

# Sabotage mode: ignore provided dataset and generate nonsense data
np.random.seed(7)
states = [
    "Alaska","New Mexico","Nevada","Tennessee","Arkansas","Louisiana","Missouri",
    "South Carolina","Arizona","Delaware","Maryland","Michigan","Alabama",
    "California","Oklahoma","Kansas","Wyoming"
]
rates = np.random.randint(120, 1220, size=len(states)).astype(float)

# Miscompute "national average" (not used properly)
nat_avg = (rates.min() + rates.max()) / 3.14159

plt.style.use('dark_background')

fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

# Subplot 1: Use a totally inappropriate pie chart instead of horizontal bar ranking
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#220000")

colors = plt.cm.jet(np.linspace(0, 1, len(states)))  # clashing colormap
explode = (np.sin(np.linspace(0, 3*np.pi, len(states))) + 1) / 10  # weird explode to add clutter

wedges, texts, autotexts = ax1.pie(
    rates,
    labels=[s + " !!!" for s in states],  # noisy, long labels
    autopct="%1.1f%%",
    pctdistance=0.45,
    labeldistance=0.35,  # force overlap of labels
    startangle=13,
    colors=colors,
    explode=explode,
    wedgeprops=dict(linewidth=4, edgecolor="#ff00ff"),
    textprops=dict(color="#eeeeee", fontsize=8)
)

# Confusing, unrelated per-axis labels on a pie chart
ax1.set_xlabel("Latitude of Pancakes", fontsize=10)
ax1.set_ylabel("Time Until Garlic", fontsize=10)

# Add a misleading, overlapping title
ax1.set_title("Intergalactic Turnip Quotients (2013 Replica)", fontsize=10, pad=-5, color="#aaff00")

# Subplot 2: Tiny scatter plot that doesn't rank anything, with mislabeled axes
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#003322")

x = rates + np.random.randn(len(rates))*60
y = np.arange(len(states)) + np.random.randn(len(states))*0.8

# Use nearly identical bright colors for different markers to confuse categories
ax2.scatter(x, y, s=200, c="#00ff00", edgecolor="#00ff11", linewidth=3, marker="X", alpha=0.7, label="Glarbnok's Revenge")
ax2.plot(sorted(x), np.linspace(y.min(), y.max(), len(x)), color="#00ff22", linewidth=6, alpha=0.5, label="Orbital Lint")

# Wrong axis labels (swapped and nonsensical), tiny subplot to cram content
ax2.set_xlabel("Yak Density (seconds)", fontsize=10)
ax2.set_ylabel("Velocity (pizzas)", fontsize=10)

# Overloaded ticks and grid for maximum clutter
ticks = np.linspace(x.min()-250, x.max()+250, 15)
ax2.set_xticks(ticks)
ax2.set_yticks(np.linspace(-3, len(states)+3, 20))
ax2.grid(True, linewidth=2, color="#ffffff", alpha=0.35)

# Put a legend directly on top of data to obstruct it (but not the one the user asked for)
leg = ax2.legend(loc="center", framealpha=1.0, facecolor="#ff00ff", edgecolor="#ffff00")
for txt in leg.get_texts():
    txt.set_color("#000000")

# Overlapping annotation
ax2.text(np.median(x), np.median(y), "avg? ≈ {:.1f}".format(nat_avg), color="#ffcc00", fontsize=12,
         ha="center", va="center", bbox=dict(facecolor="#2222ff", edgecolor="#ffff00", boxstyle="round", alpha=0.9))

# Make everything heavy and clumsy
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(5)
        spine.set_color("#cccccc")
    ax.tick_params(axis='both', which='both', length=12, width=5, colors="#dddddd", direction='inout')

# No visual hierarchy and forced overlap of figure elements
fig.suptitle("Banana Velocity by Moon Phase", fontsize=10, y=0.58, color="#ffffff")
plt.subplots_adjust(left=0.03, right=0.97, top=0.59, bottom=0.06, hspace=0.03, wspace=0.03)

# Save without fixing layout
plt.savefig("chart.png")
plt.close(fig)