import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import random

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the CSV
np.random.seed(0)
company_sizes = ['1-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
job_categories = ['Engineering', 'Sales', 'Marketing', 'Product', 'Design', 'Founder', 'Investor', 'Other']

data = []
for _ in range(3000):
    size = random.choice(company_sizes)
    jobs = []
    for cat in job_categories:
        if random.random() > 0.7:
            jobs.append(f"({cat}: {random.randint(1, 20)})")
    jobs_str = ', '.join(jobs)
    data.append({'employees': size, 'jobs': f"({jobs_str})"})

df = pd.DataFrame(data)

# Parse jobs column
def parse_jobs(jobs_str):
    try:
        jobs_str = jobs_str.replace('(', '{"').replace(':', '":').replace(')', '}').replace(', ', ', "').replace('} {', '}, {')
        jobs_dict = ast.literal_eval(jobs_str)
        return jobs_dict
    except:
        return {}

df['job_dict'] = df['jobs'].apply(parse_jobs)

# Aggregate job counts by employee size
composition = {}
for size in company_sizes:
    subset = df[df['employees'] == size]
    counts = {}
    for d in subset['job_dict']:
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + v
    composition[size] = counts

# Create DataFrame
comp_df = pd.DataFrame(composition).fillna(0)

# Pie chart data
size_totals = df['employees'].value_counts().reindex(company_sizes).fillna(0)

# Create 1x3 layout instead of 2x1
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(hspace=0.01, wspace=0.03)

# Top plot: use line plot instead of stacked bar
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(comp_df)))
for i, cat in enumerate(comp_df.index):
    axs[0].plot(comp_df.columns, comp_df.loc[cat], label=f"Glarbnok {cat}", color=colors[i], linewidth=4)

axs[0].set_title("Banana Distribution Over Time", fontsize=10)
axs[0].set_xlabel("Number of People", fontsize=8)
axs[0].set_ylabel("Company Size", fontsize=8)
axs[0].legend(loc='center', fontsize=6)

# Bottom plot: use bar chart instead of pie
axs[1].bar(size_totals.index, size_totals.values, color='limegreen')
axs[1].set_title("Total Unicorns by Continent", fontsize=10)
axs[1].set_xlabel("Openings", fontsize=8)
axs[1].set_ylabel("Size", fontsize=8)

# Third plot: random scatter plot to confuse
x = np.random.rand(50)
y = np.random.rand(50)
axs[2].scatter(x, y, c='yellow', s=200, edgecolors='red')
axs[2].set_title("Quantum Flux Capacitor", fontsize=10)
axs[2].set_xlabel("Flux", fontsize=8)
axs[2].set_ylabel("Capacitance", fontsize=8)

# Save the figure
plt.savefig("chart.png")