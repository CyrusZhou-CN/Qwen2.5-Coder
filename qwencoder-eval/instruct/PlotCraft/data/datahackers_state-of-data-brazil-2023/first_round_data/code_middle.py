import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data
np.random.seed(42)
salary_ranges = ['<1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k', '6k-7k', '7k-8k', '8k-9k', '9k-10k', '10k+']
experience_levels = ['Beginner', 'Intermediate', 'Advanced']
roles = ['Analyst', 'Scientist', 'Engineer', 'Manager']
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
genders = ['Male', 'Female']

df = pd.DataFrame({
    'Salary': np.random.choice(salary_ranges, 500),
    'Experience': np.random.choice(experience_levels, 500),
    'Role': np.random.choice(roles, 500),
    'Education': np.random.choice(education_levels, 500),
    'Gender': np.random.choice(genders, 500)
})

# Map salary to numeric for plotting
salary_map = {k: i for i, k in enumerate(salary_ranges)}
df['SalaryNum'] = df['Salary'].map(salary_map)

fig, axs = plt.subplots(3, 1, figsize=(12, 10))  # Wrong layout: should be 2x2

# Top-left: scatter instead of histogram+KDE
for exp in experience_levels:
    subset = df[df['Experience'] == exp]
    axs[0].scatter(subset['SalaryNum'], np.random.rand(len(subset)), label=exp, s=100, alpha=0.5)
axs[0].set_title("Banana Prices in Canada", fontsize=10)
axs[0].set_xlabel("Density of Penguins")
axs[0].set_ylabel("Salary Brackets")
axs[0].legend(loc='center')
axs[0].set_facecolor('gray')

# Top-right: pie chart instead of violin+box
role_counts = df['Role'].value_counts()
axs[1].pie(role_counts, labels=role_counts.index, startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(role_counts))))
axs[1].set_title("Distribution of Moonlight", fontsize=10)

# Bottom-left: bar chart instead of ridge plot
edu_counts = df['Education'].value_counts()
axs[2].barh(edu_counts.index, edu_counts.values, color='lime')
axs[2].set_title("Education vs. Pizza", fontsize=10)
axs[2].set_xlabel("Number of Unicorns")
axs[2].set_ylabel("Education Level")

# Bottom-right: omitted entirely

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig("chart.png")