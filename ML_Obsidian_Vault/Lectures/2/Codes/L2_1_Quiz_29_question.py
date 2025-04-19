import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Set general plot style
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Define ranges for x values
x_discrete = np.arange(0, 11)
x_continuous = np.linspace(-5, 15, 1000)

# Define distribution parameters
n, p = 10, 0.6  # Binomial
lam = 3  # Poisson
mu, sigma = 5, 2  # Normal
scale = 1/1.5  # Exponential (rate parameter λ = 1.5)

# Calculate PMFs and PDFs
binomial_pmf = stats.binom.pmf(x_discrete, n, p)
poisson_pmf = stats.poisson.pmf(x_discrete, lam)
normal_pdf = stats.norm.pdf(x_continuous, mu, sigma)
exponential_pdf = stats.expon.pdf(x_continuous, scale=scale)

# Create a combined figure with all four distributions
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Distribution A: Binomial
axs[0, 0].bar(x_discrete, binomial_pmf, alpha=0.7, color='#1f77b4')
axs[0, 0].set_xlim(-0.5, 10.5)
axs[0, 0].set_xlabel('Number of Successes')
axs[0, 0].set_ylabel('Probability')
axs[0, 0].set_title('Distribution A')

# Distribution B: Poisson
axs[0, 1].bar(x_discrete, poisson_pmf, alpha=0.7, color='#ff7f0e')
axs[0, 1].set_xlim(-0.5, 10.5)
axs[0, 1].set_xlabel('Number of Events')
axs[0, 1].set_ylabel('Probability')
axs[0, 1].set_title('Distribution B')

# Distribution C: Normal
axs[1, 0].plot(x_continuous, normal_pdf, linewidth=2, color='#2ca02c')
axs[1, 0].fill_between(x_continuous, normal_pdf, alpha=0.3, color='#2ca02c')
axs[1, 0].set_xlim(-5, 15)
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Probability Density')
axs[1, 0].set_title('Distribution C')

# Distribution D: Exponential
axs[1, 1].plot(x_continuous, exponential_pdf, linewidth=2, color='#d62728')
axs[1, 1].fill_between(x_continuous, exponential_pdf, alpha=0.3, color='#d62728')
axs[1, 1].set_xlim(-0.5, 10)
axs[1, 1].set_xlabel('Value')
axs[1, 1].set_ylabel('Probability Density')
axs[1, 1].set_title('Distribution D')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distributions.png'), dpi=300)
plt.close()

# Create a figure showing 4 plots with sample data from each distribution
np.random.seed(42)  # For reproducibility

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Sample Data A - Binomial
samples_A = np.random.binomial(n, p, size=100)
axs[0, 0].hist(samples_A, bins=n+1, range=(-0.5, n+0.5), density=True, alpha=0.7, color='#1f77b4')
axs[0, 0].set_title('Sample Data from Distribution A')
axs[0, 0].set_xlabel('Value')
axs[0, 0].set_ylabel('Relative Frequency')

# Sample Data B - Poisson
samples_B = np.random.poisson(lam, size=100)
axs[0, 1].hist(samples_B, bins=range(0, 12), density=True, alpha=0.7, color='#ff7f0e')
axs[0, 1].set_title('Sample Data from Distribution B')
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Relative Frequency')

# Sample Data C - Normal
samples_C = np.random.normal(mu, sigma, size=100)
axs[1, 0].hist(samples_C, bins=15, density=True, alpha=0.7, color='#2ca02c')
axs[1, 0].set_title('Sample Data from Distribution C')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Relative Frequency')

# Sample Data D - Exponential
samples_D = np.random.exponential(scale, size=100)
axs[1, 1].hist(samples_D, bins=15, density=True, alpha=0.7, color='#d62728')
axs[1, 1].set_title('Sample Data from Distribution D')
axs[1, 1].set_xlabel('Value')
axs[1, 1].set_ylabel('Relative Frequency')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sample_data.png'), dpi=300)
plt.close()

# Generate a figure with PMF/PDF and CDF for each distribution
fig, axs = plt.subplots(4, 2, figsize=(14, 16))

# Distribution A - Binomial
axs[0, 0].bar(x_discrete, binomial_pmf, alpha=0.7, color='#1f77b4')
axs[0, 0].set_title('Distribution A - PMF')
axs[0, 0].set_xlabel('Value')
axs[0, 0].set_ylabel('Probability')

binomial_cdf = stats.binom.cdf(x_discrete, n, p)
axs[0, 1].step(x_discrete, binomial_cdf, where='post', linewidth=2, color='#1f77b4')
axs[0, 1].scatter(x_discrete, binomial_cdf, color='#1f77b4', zorder=3)
axs[0, 1].set_title('Distribution A - CDF')
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Cumulative Probability')
axs[0, 1].set_ylim(0, 1.05)

# Distribution B - Poisson
axs[1, 0].bar(x_discrete, poisson_pmf, alpha=0.7, color='#ff7f0e')
axs[1, 0].set_title('Distribution B - PMF')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Probability')

poisson_cdf = stats.poisson.cdf(x_discrete, lam)
axs[1, 1].step(x_discrete, poisson_cdf, where='post', linewidth=2, color='#ff7f0e')
axs[1, 1].scatter(x_discrete, poisson_cdf, color='#ff7f0e', zorder=3)
axs[1, 1].set_title('Distribution B - CDF')
axs[1, 1].set_xlabel('Value')
axs[1, 1].set_ylabel('Cumulative Probability')
axs[1, 1].set_ylim(0, 1.05)

# Distribution C - Normal
x_plot = np.linspace(-5, 15, 100)
axs[2, 0].plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), linewidth=2, color='#2ca02c')
axs[2, 0].fill_between(x_plot, stats.norm.pdf(x_plot, mu, sigma), alpha=0.3, color='#2ca02c')
axs[2, 0].set_title('Distribution C - PDF')
axs[2, 0].set_xlabel('Value')
axs[2, 0].set_ylabel('Probability Density')

axs[2, 1].plot(x_plot, stats.norm.cdf(x_plot, mu, sigma), linewidth=2, color='#2ca02c')
axs[2, 1].set_title('Distribution C - CDF')
axs[2, 1].set_xlabel('Value')
axs[2, 1].set_ylabel('Cumulative Probability')
axs[2, 1].set_ylim(0, 1.05)

# Distribution D - Exponential
x_plot = np.linspace(0, 10, 100)
axs[3, 0].plot(x_plot, stats.expon.pdf(x_plot, scale=scale), linewidth=2, color='#d62728')
axs[3, 0].fill_between(x_plot, stats.expon.pdf(x_plot, scale=scale), alpha=0.3, color='#d62728')
axs[3, 0].set_title('Distribution D - PDF')
axs[3, 0].set_xlabel('Value')
axs[3, 0].set_ylabel('Probability Density')

axs[3, 1].plot(x_plot, stats.expon.cdf(x_plot, scale=scale), linewidth=2, color='#d62728')
axs[3, 1].set_title('Distribution D - CDF')
axs[3, 1].set_xlabel('Value')
axs[3, 1].set_ylabel('Cumulative Probability')
axs[3, 1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distributions_properties.png'), dpi=300)
plt.close()

print(f"All visualizations saved in '{save_dir}'") 