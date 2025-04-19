import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Define the distribution parameters (same as in the question)
# Distribution A: Binomial
n_A, p_A = 10, 0.6
# Distribution B: Poisson
lambda_B = 3
# Distribution C: Normal
mu_C, sigma_C = 5, 2
# Distribution D: Exponential
lambda_D = 1.5  # rate parameter

# Calculate the mean and variance for each distribution
means = {
    'A (Binomial)': n_A * p_A,
    'B (Poisson)': lambda_B,
    'C (Normal)': mu_C,
    'D (Exponential)': 1/lambda_D
}

variances = {
    'A (Binomial)': n_A * p_A * (1-p_A),
    'B (Poisson)': lambda_B,
    'C (Normal)': sigma_C**2,
    'D (Exponential)': 1/lambda_D**2
}

# Print results
print("\n## Solution to Question 29: Probability Distribution Identification\n")

print("### Step 1: Distribution Identification\n")
print("Based on visual analysis of the distributions:\n")
print("- **Distribution A**: Binomial Distribution with parameters n=10, p=0.6")
print("- **Distribution B**: Poisson Distribution with parameter λ=3")
print("- **Distribution C**: Normal Distribution with parameters μ=5, σ=2")
print("- **Distribution D**: Exponential Distribution with rate parameter λ=1.5\n")

print("### Step 2: Distribution Properties\n")
print("#### Summary Table\n")
print("| Distribution | Parameters | Mean | Variance | Support | Formula |")
print("|-------------|------------|------|----------|---------|---------|")
print(f"| A: Binomial | n={n_A}, p={p_A} | {means['A (Binomial)']:.2f} | {variances['A (Binomial)']:.2f} | {{0,1,...,10}} | $P(X=k) = \\binom{{n}}{{k}} p^k (1-p)^{{n-k}}$ |")
print(f"| B: Poisson | λ={lambda_B} | {means['B (Poisson)']:.2f} | {variances['B (Poisson)']:.2f} | {{0,1,2,...}} | $P(X=k) = \\frac{{e^{{-\\lambda}} \\lambda^k}}{{k!}}$ |")
print(f"| C: Normal | μ={mu_C}, σ={sigma_C} | {means['C (Normal)']:.2f} | {variances['C (Normal)']:.2f} | $(-\\infty, \\infty)$ | $f(x) = \\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}} e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}$ |")
print(f"| D: Exponential | λ={lambda_D} | {means['D (Exponential)']:.4f} | {variances['D (Exponential)']:.4f} | $[0, \\infty)$ | $f(x) = \\lambda e^{{-\\lambda x}}$ |")

print("\n#### Mean and Variance Calculations\n")
print("**Binomial Distribution (A):**")
print(f"- Mean: E[X] = np = {n_A} × {p_A} = {means['A (Binomial)']}")
print(f"- Variance: Var(X) = np(1-p) = {n_A} × {p_A} × {1-p_A} = {variances['A (Binomial)']}")

print("\n**Poisson Distribution (B):**")
print(f"- Mean: E[X] = λ = {lambda_B}")
print(f"- Variance: Var(X) = λ = {lambda_B}")

print("\n**Normal Distribution (C):**")
print(f"- Mean: E[X] = μ = {mu_C}")
print(f"- Variance: Var(X) = σ² = {sigma_C}² = {variances['C (Normal)']}")

print("\n**Exponential Distribution (D):**")
print(f"- Mean: E[X] = 1/λ = 1/{lambda_D} ≈ {means['D (Exponential)']:.4f}")
print(f"- Variance: Var(X) = 1/λ² = 1/{lambda_D}² ≈ {variances['D (Exponential)']:.4f}")

print("\n### Step 3: Application Matching\n")
print("| Distribution | Most Appropriate Application |")
print("|-------------|------------------------------|")
print("| A: Binomial | Modeling the number of classification errors in a fixed number of predictions |")
print("| B: Poisson | Modeling the arrival of rare events (e.g., fraudulent transactions) |")
print("| C: Normal | Modeling measurement errors in a physical system |")
print("| D: Exponential | Modeling the time between system failures |")

print("\n### Step 4: Key Distinguishing Properties\n")
print("**Binomial Distribution (A):**")
print("- **Key Property**: Fixed number of trials")
print("- The distribution models the number of successes in a fixed number of independent trials")
print("- Constrained to a finite range (0 to n)")
print("- As a discrete distribution, it has a probability mass function (PMF) rather than a PDF")

print("\n**Poisson Distribution (B):**")
print("- **Key Property**: Mean equals variance")
print("- This is a unique property where E[X] = Var(X) = λ")
print("- Models count data where events occur independently at a constant rate")
print("- Useful for rare events where n is large and p is small")

print("\n**Normal Distribution (C):**")
print("- **Key Property**: Symmetry around the mean")
print("- Perfectly symmetric bell-shaped curve")
print("- Defined on the entire real number line")
print("- Central Limit Theorem states that the sum of a large number of random variables tends toward a normal distribution")

print("\n**Exponential Distribution (D):**")
print("- **Key Property**: Memoryless property")
print("- P(X > s+t | X > s) = P(X > t) for all s, t > 0")
print("- This means the future waiting time is independent of how much time has already elapsed")
print("- Models waiting times between events in a Poisson process")

# Create visualizations for the solution
# 1. Create a figure comparing all distributions in one plot
x_disc = np.arange(0, 16)
x_cont = np.linspace(0, 15, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot PMFs for discrete distributions
ax1.bar(x_disc[:11], stats.binom.pmf(x_disc[:11], n_A, p_A), alpha=0.5, label=f'A: Binomial(n={n_A}, p={p_A})')
ax1.bar(x_disc, stats.poisson.pmf(x_disc, lambda_B), alpha=0.5, label=f'B: Poisson(λ={lambda_B})')
ax1.set_title('Probability Mass Functions (PMFs)', fontsize=14)
ax1.set_xlabel('Value', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot PDFs for continuous distributions
ax2.plot(x_cont, stats.norm.pdf(x_cont, mu_C, sigma_C), 
         linewidth=2, label=f'C: Normal(μ={mu_C}, σ={sigma_C})')
ax2.plot(x_cont, stats.expon.pdf(x_cont, scale=1/lambda_D), 
         linewidth=2, label=f'D: Exponential(λ={lambda_D})')
ax2.set_title('Probability Density Functions (PDFs)', fontsize=14)
ax2.set_xlabel('Value', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_distributions_comparison.png'), dpi=300)
plt.close()

# 2. Create key properties visualization with examples
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, figure=fig)

# Key property for Binomial: Fixed number of trials
ax1 = fig.add_subplot(gs[0, 0])
n_vals = [10, 10, 10]
p_vals = [0.3, 0.5, 0.7]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
x = np.arange(0, 11)
for i, (n, p) in enumerate(zip(n_vals, p_vals)):
    pmf = stats.binom.pmf(x, n, p)
    ax1.bar(x + 0.1*i, pmf, width=0.1, alpha=0.7, color=colors[i], label=f'p={p}')
ax1.set_title('A: Binomial - Fixed Number of Trials (n=10)', fontsize=12)
ax1.set_xlabel('Number of Successes (k)')
ax1.set_ylabel('Probability')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Key property for Poisson: Mean = Variance
ax2 = fig.add_subplot(gs[0, 1])
lambda_vals = [1, 3, 5]
x = np.arange(0, 15)
for i, lam in enumerate(lambda_vals):
    pmf = stats.poisson.pmf(x, lam)
    ax2.bar(x + 0.1*i, pmf, width=0.1, alpha=0.7, color=colors[i], 
            label=f'λ={lam} (Mean=Var={lam})')
ax2.set_title('B: Poisson - Mean = Variance', fontsize=12)
ax2.set_xlabel('Number of Events (k)')
ax2.set_ylabel('Probability')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Key property for Normal: Symmetry
ax3 = fig.add_subplot(gs[1, 0])
mu_vals = [4, 6, 8]
sigma = 1.5
x = np.linspace(0, 12, 1000)
for i, mu in enumerate(mu_vals):
    pdf = stats.norm.pdf(x, mu, sigma)
    ax3.plot(x, pdf, linewidth=2, color=colors[i], label=f'μ={mu}, σ={sigma}')
    ax3.axvline(x=mu, color=colors[i], linestyle='--', alpha=0.7)
ax3.set_title('C: Normal - Symmetric around Mean', fontsize=12)
ax3.set_xlabel('Value')
ax3.set_ylabel('Probability Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Key property for Exponential: Memoryless
ax4 = fig.add_subplot(gs[1, 1])
lambda_vals = [0.5, 1, 1.5]
x = np.linspace(0, 8, 1000)
for i, lam in enumerate(lambda_vals):
    pdf = stats.expon.pdf(x, scale=1/lam)
    ax4.plot(x, pdf, linewidth=2, color=colors[i], label=f'λ={lam}')
    
    # Demonstrate memoryless property: P(X > s+t | X > s) = P(X > t)
    s, t = 2, 1
    # Plot conditional distribution starting at s
    x_cond = np.linspace(s, s+5, 1000)
    pdf_cond = stats.expon.pdf(x_cond-s, scale=1/lam)
    ax4.plot(x_cond, pdf_cond, linestyle='--', color=colors[i], alpha=0.7)

ax4.set_title('D: Exponential - Memoryless Property', fontsize=12)
ax4.set_xlabel('Value')
ax4.set_ylabel('Probability Density')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Key Properties of the Four Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(save_dir, 'key_properties.png'), dpi=300)
plt.close()

print(f"\nAll solution visualizations saved to '{save_dir}'") 