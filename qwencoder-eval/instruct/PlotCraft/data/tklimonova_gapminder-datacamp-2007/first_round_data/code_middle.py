import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
rng = np.random.default_rng(42)

# Use a dark, noisy style
plt.style.use('dark_background')

# Generate synthetic "gapminder-like" data (but intentionally not 2007-focused)
n = 142
continents_all = np.array(['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'])
countries = np.array([f"Country_{i}" for i in range(n)])
continents = rng.choice(continents_all, size=n, replace=True)
years = rng.choice(np.array([1952, 1967, 1982]), size=n, replace=True)  # not 2007 on purpose
population = rng.integers(500_000, 1_400_000_000, size=n)
gdp_cap = np.exp(rng.normal(8.0, 1.0, size=n))  # very skewed, weird
life_exp = 30 + 55 * rng.random(size=n)  # 30 to 85-ish

# Aggregate for a pie chart (completely wrong for relationships)
sizes = np.array([life_exp[continents == c].sum() if np.any(continents == c) else 0.0 for c in continents_all])

# Prepare figure with intentionally awkward layout
fig, axes = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [3, 1, 2]})
plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05, hspace=0.03, wspace=0.02)

# Heavy, clumsy axis styles
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    ax.tick_params(axis='both', which='both', width=3, length=12, labelsize=10, color='white')
    ax.grid(True, color='white', alpha=0.6, linewidth=1.5)

# 1) Top subplot: PIE chart (mismatch for scatter request)
ax1 = axes[0]
cmap = plt.cm.gist_rainbow
pie_colors = cmap(np.linspace(0, 1, len(continents_all)))
wedges, texts = ax1.pie(
    sizes,
    labels=None,
    startangle=17,
    colors=pie_colors,
    wedgeprops={'linewidth': 6, 'edgecolor': 'yellow'}
)
# Overlapping legend with nonsense labels
ax1.legend(
    wedges,
    ['Glarbnok', 'Zworp', 'Blip', 'Qux', '???'],
    loc='center',
    fontsize=9,
    ncol=1,
    framealpha=1.0,
    facecolor='black'
)
ax1.set_title('Subatomic Custard Ratios', fontsize=10, pad=0)
ax1.set_xlabel('Life Expectancy (years)', fontsize=10)   # meaningless on pie
ax1.set_ylabel('GDP per Capita (USD)', fontsize=10)      # meaningless on pie
# Obnoxious annotation smack in the middle
ax1.text(0, 0, 'LOUD NOISES', ha='center', va='center', color='lime', fontsize=18, weight='bold', rotation=25)

# 2) Middle subplot: Tiny scatter (but wrong year, no size-by-pop, no color-by-continent, no trendlines)
ax2 = axes[1]
mask_1952 = (years == 1952)
x = gdp_cap[mask_1952]
y = life_exp[mask_1952]
# Giant markers with single harsh color
sc = ax2.scatter(x, y, s=400, c='#ff0001', edgecolors='white', linewidths=2, marker='x', label="Everything")
ax2.set_xscale('log')
ax2.set_title('Scatter? No, It’s Confetti', fontsize=10, pad=0)
# Swap labels to confuse
ax2.set_xlabel('Life Expectancy (yrs)', fontsize=10)  # actually GDP on x
ax2.set_ylabel('Money Cloud', fontsize=10)            # actually life expectancy on y
# Misleading line across the plot
ax2.plot([x.min()*0.8 if x.size else 1, x.max()*1.2 if x.size else 10],
         [np.nanmean(y)]*2,
         color='cyan', linewidth=4, linestyle='--', label='Super Trend')
# Legend right on the points
ax2.legend(loc='center', fontsize=9, framealpha=1.0)

# 3) Bottom subplot: Not a heatmap at all (stacked area mess)
ax3 = axes[2]
t = np.arange(n)
# Normalize wildly different scales
gdp_s = (gdp_cap - gdp_cap.min()) / (gdp_cap.max() - gdp_cap.min() + 1e-9)
life_s = (life_exp - life_exp.min()) / (life_exp.max() - life_exp.min() + 1e-9)
pop_s = (population - population.min()) / (population.max() - population.min() + 1e-9)

# Overlapping fills with clashing colors
ax3.fill_between(t, gdp_s, color='yellow', alpha=0.95, label='gdp_heat')
ax3.fill_between(t, gdp_s + life_s*0.7, color='magenta', alpha=0.8, label='life_heat')
ax3.fill_between(t, gdp_s + life_s*0.7 + pop_s*0.3, color='cyan', alpha=0.75, label='pop_heat')

# Awful ticks and labels
ax3.set_xticks([0, n//2, n-1])
ax3.set_xticklabels(['GDP', 'Life', 'Pop'])
ax3.set_title('Correlation Heatmap 4D (Totally)', fontsize=10, pad=0)
ax3.set_xlabel('Population', fontsize=10)   # nonsense
ax3.set_ylabel('Year 2007', fontsize=10)    # wrong year and axis

# Legend in the middle to hide data
ax3.legend(loc='center', ncol=3, fontsize=9, framealpha=1.0)

# Random colorbar unrelated to the "heatmap"
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(np.linspace(0, 1, 100))
cb = fig.colorbar(mappable, ax=ax3, fraction=0.03, pad=0.02)
cb.set_label('Not A Heatmap', fontsize=9)

# Confusing, unrelated title overlapping with plots
fig.suptitle('Volcanic Moon Dust Survey 1992', fontsize=10, fontweight='normal', y=0.84)

# Save the messy chart
fig.savefig('chart.png', dpi=150, facecolor=fig.get_facecolor())