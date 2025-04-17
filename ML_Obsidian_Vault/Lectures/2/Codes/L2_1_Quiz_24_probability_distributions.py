import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Define a function to save figures
def save_figure(filename):
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {file_path}")

distributions = {
    "A": "Bernoulli",
    "B": "Binomial",
    "C": "Geometric",
    "D": "Poisson",
    "E": "Exponential",
    "F": "Normal",
    "G": "Uniform",
    "H": "Beta"
}

applications = [
    "1. Modeling the number of rare events occurring in a fixed time interval",
    "2. Representing the probability of success in a single trial",
    "3. Modeling the number of trials needed until the first success",
    "4. Representing the distribution of errors in physical measurements",
    "5. Modeling the prior distribution of a probability parameter",
    "6. Representing the waiting time between events in a Poisson process",
    "7. Modeling the number of successes in a fixed number of independent trials",
    "8. Representing a random variable that is equally likely to take any value in a range"
]

# Step 1: Bernoulli Distribution
print_step_header(1, "Bernoulli Distribution (A)")

print("Bernoulli Distribution:")
print("- Probability mass function: P(X = 1) = p, P(X = 0) = 1-p")
print("- Models a single trial with two possible outcomes: success (1) or failure (0)")
print("- Example: Coin flip, where 'heads' is success with probability p")
print("- Mean: p, Variance: p(1-p)")
print()
print("Matching application:")
print("2. Representing the probability of success in a single trial")
print()

# Visualize Bernoulli Distribution
plt.figure(figsize=(10, 6))

# Parameters
p_values = [0.2, 0.5, 0.8]
x = np.array([0, 1])

for i, p in enumerate(p_values):
    # Calculate PMF
    pmf = np.array([1-p, p])
    
    # Plot
    plt.bar(x + i*0.2, pmf, width=0.2, alpha=0.7, label=f'p = {p}')

plt.title("Bernoulli Distribution PMF", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Mass", fontsize=12)
plt.xticks([0, 1], ['0 (Failure)', '1 (Success)'])
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("bernoulli_distribution.png")

# Step 2: Binomial Distribution
print_step_header(2, "Binomial Distribution (B)")

print("Binomial Distribution:")
print("- Probability mass function: P(X = k) = (n choose k) * p^k * (1-p)^(n-k)")
print("- Models the number of successes in n independent Bernoulli trials")
print("- Example: Number of heads in 10 coin flips")
print("- Mean: np, Variance: np(1-p)")
print()
print("Matching application:")
print("7. Modeling the number of successes in a fixed number of independent trials")
print()

# Visualize Binomial Distribution
plt.figure(figsize=(12, 7))

# Parameters
n = 10  # number of trials
p_values = [0.2, 0.5, 0.8]
x = np.arange(0, n+1)

for i, p in enumerate(p_values):
    # Calculate PMF
    pmf = stats.binom.pmf(x, n, p)
    
    # Plot
    plt.plot(x, pmf, 'o-', linewidth=2, label=f'p = {p}')

plt.title(f"Binomial Distribution PMF (n = {n})", fontsize=15)
plt.xlabel("Number of Successes (k)", fontsize=12)
plt.ylabel("Probability Mass", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("binomial_distribution.png")

# Step 3: Geometric Distribution
print_step_header(3, "Geometric Distribution (C)")

print("Geometric Distribution:")
print("- Probability mass function: P(X = k) = (1-p)^(k-1) * p")
print("- Models the number of trials until the first success")
print("- Example: Number of rolls until first 6 on a die")
print("- Mean: 1/p, Variance: (1-p)/p^2")
print()
print("Matching application:")
print("3. Modeling the number of trials needed until the first success")
print()

# Visualize Geometric Distribution
plt.figure(figsize=(12, 7))

# Parameters
p_values = [0.2, 0.5, 0.8]
x = np.arange(1, 16)  # 1 to 15 trials

for i, p in enumerate(p_values):
    # Calculate PMF
    pmf = stats.geom.pmf(x, p)
    
    # Plot
    plt.plot(x, pmf, 'o-', linewidth=2, label=f'p = {p}')

plt.title("Geometric Distribution PMF", fontsize=15)
plt.xlabel("Number of Trials Until First Success (k)", fontsize=12)
plt.ylabel("Probability Mass", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("geometric_distribution.png")

# Step 4: Poisson Distribution
print_step_header(4, "Poisson Distribution (D)")

print("Poisson Distribution:")
print("- Probability mass function: P(X = k) = (λ^k * e^(-λ)) / k!")
print("- Models the number of events occurring in a fixed interval")
print("- Example: Number of customers arriving at a store in an hour")
print("- Mean: λ, Variance: λ")
print()
print("Matching application:")
print("1. Modeling the number of rare events occurring in a fixed time interval")
print()

# Visualize Poisson Distribution
plt.figure(figsize=(12, 7))

# Parameters
lambda_values = [1, 4, 10]
x = np.arange(0, 20)

for i, lam in enumerate(lambda_values):
    # Calculate PMF
    pmf = stats.poisson.pmf(x, lam)
    
    # Plot
    plt.plot(x, pmf, 'o-', linewidth=2, label=f'λ = {lam}')

plt.title("Poisson Distribution PMF", fontsize=15)
plt.xlabel("Number of Events (k)", fontsize=12)
plt.ylabel("Probability Mass", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("poisson_distribution.png")

# Step 5: Exponential Distribution
print_step_header(5, "Exponential Distribution (E)")

print("Exponential Distribution:")
print("- Probability density function: f(x) = λe^(-λx) for x ≥ 0")
print("- Models the time between events in a Poisson process")
print("- Example: Time between customer arrivals at a store")
print("- Mean: 1/λ, Variance: 1/λ^2")
print()
print("Matching application:")
print("6. Representing the waiting time between events in a Poisson process")
print()

# Visualize Exponential Distribution
plt.figure(figsize=(12, 7))

# Parameters
lambda_values = [0.5, 1, 2]
x = np.linspace(0, 8, 1000)

for i, lam in enumerate(lambda_values):
    # Calculate PDF
    pdf = stats.expon.pdf(x, scale=1/lam)
    
    # Plot
    plt.plot(x, pdf, linewidth=2, label=f'λ = {lam}')

plt.title("Exponential Distribution PDF", fontsize=15)
plt.xlabel("Time (x)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("exponential_distribution.png")

# Step 6: Normal Distribution
print_step_header(6, "Normal Distribution (F)")

print("Normal Distribution:")
print("- Probability density function: f(x) = (1/(σ√(2π))) * e^(-(x-μ)^2/(2σ^2))")
print("- Bell-shaped curve, symmetric around the mean")
print("- Example: Heights of adult humans, measurement errors")
print("- Mean: μ, Variance: σ^2")
print()
print("Matching application:")
print("4. Representing the distribution of errors in physical measurements")
print()

# Visualize Normal Distribution
plt.figure(figsize=(12, 7))

# Parameters
mu = 0
sigma_values = [0.5, 1, 2]
x = np.linspace(-6, 6, 1000)

for i, sigma in enumerate(sigma_values):
    # Calculate PDF
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    
    # Plot
    plt.plot(x, pdf, linewidth=2, label=f'μ = {mu}, σ = {sigma}')

plt.title("Normal Distribution PDF", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("normal_distribution.png")

# Step 7: Uniform Distribution
print_step_header(7, "Uniform Distribution (G)")

print("Uniform Distribution:")
print("- Probability density function: f(x) = 1/(b-a) for a ≤ x ≤ b")
print("- Equal probability for all values in the range [a, b]")
print("- Example: Random number generator, rounding errors")
print("- Mean: (a+b)/2, Variance: (b-a)^2/12")
print()
print("Matching application:")
print("8. Representing a random variable that is equally likely to take any value in a range")
print()

# Visualize Uniform Distribution
plt.figure(figsize=(12, 7))

# Parameters
a_b_values = [(0, 1), (-1, 1), (2, 5)]

for i, (a, b) in enumerate(a_b_values):
    # Calculate PDF
    x = np.linspace(a-0.5, b+0.5, 1000)
    pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
    
    # Plot
    plt.plot(x, pdf, linewidth=2, label=f'a = {a}, b = {b}')

plt.title("Uniform Distribution PDF", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("uniform_distribution.png")

# Step 8: Beta Distribution
print_step_header(8, "Beta Distribution (H)")

print("Beta Distribution:")
print("- Probability density function: f(x) ∝ x^(α-1) * (1-x)^(β-1) for 0 ≤ x ≤ 1")
print("- Versatile distribution for random variables in [0, 1]")
print("- Example: Modeling uncertainty about probabilities, success rates")
print("- Mean: α/(α+β), Variance: αβ/((α+β)^2(α+β+1))")
print()
print("Matching application:")
print("5. Modeling the prior distribution of a probability parameter")
print()

# Visualize Beta Distribution
plt.figure(figsize=(12, 7))

# Parameters
params = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
x = np.linspace(0.001, 0.999, 1000)

for i, (alpha, beta) in enumerate(params):
    # Calculate PDF
    pdf = stats.beta.pdf(x, alpha, beta)
    
    # Plot
    plt.plot(x, pdf, linewidth=2, label=f'α = {alpha}, β = {beta}')

plt.title("Beta Distribution PDF", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("beta_distribution.png")

# Step 9: Create the matching summary
print_step_header(9, "Summary: Matching Distributions to Applications")

print("Matching summary:")
print("A) Bernoulli → 2. Representing the probability of success in a single trial")
print("B) Binomial → 7. Modeling the number of successes in a fixed number of independent trials")
print("C) Geometric → 3. Modeling the number of trials needed until the first success")
print("D) Poisson → 1. Modeling the number of rare events occurring in a fixed time interval")
print("E) Exponential → 6. Representing the waiting time between events in a Poisson process")
print("F) Normal → 4. Representing the distribution of errors in physical measurements")
print("G) Uniform → 8. Representing a random variable that is equally likely to take any value in a range")
print("H) Beta → 5. Modeling the prior distribution of a probability parameter")
print()

# Create a summary visualization of all distributions
plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

# 1. Bernoulli
ax1 = plt.subplot(gs[0, 0])
p = 0.3
x = np.array([0, 1])
pmf = np.array([1-p, p])
ax1.bar(x, pmf, width=0.4, alpha=0.7, color='royalblue')
ax1.set_title("A) Bernoulli (p=0.3)", fontsize=14)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['0 (Failure)', '1 (Success)'])
ax1.text(0.5, 0.85, "Application: 2", transform=ax1.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 2. Binomial
ax2 = plt.subplot(gs[0, 1])
n, p = 10, 0.3
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
ax2.plot(x, pmf, 'o-', color='darkorange', linewidth=2)
ax2.set_title(f"B) Binomial (n={n}, p={p})", fontsize=14)
ax2.text(0.5, 0.85, "Application: 7", transform=ax2.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 3. Geometric
ax3 = plt.subplot(gs[1, 0])
p = 0.3
x = np.arange(1, 16)
pmf = stats.geom.pmf(x, p)
ax3.plot(x, pmf, 'o-', color='green', linewidth=2)
ax3.set_title(f"C) Geometric (p={p})", fontsize=14)
ax3.text(0.5, 0.85, "Application: 3", transform=ax3.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 4. Poisson
ax4 = plt.subplot(gs[1, 1])
lam = 3
x = np.arange(0, 15)
pmf = stats.poisson.pmf(x, lam)
ax4.plot(x, pmf, 'o-', color='purple', linewidth=2)
ax4.set_title(f"D) Poisson (λ={lam})", fontsize=14)
ax4.text(0.5, 0.85, "Application: 1", transform=ax4.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 5. Exponential
ax5 = plt.subplot(gs[2, 0])
lam = 1
x = np.linspace(0, 6, 1000)
pdf = stats.expon.pdf(x, scale=1/lam)
ax5.plot(x, pdf, color='red', linewidth=2)
ax5.set_title(f"E) Exponential (λ={lam})", fontsize=14)
ax5.text(0.5, 0.85, "Application: 6", transform=ax5.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 6. Normal
ax6 = plt.subplot(gs[2, 1])
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
ax6.plot(x, pdf, color='blue', linewidth=2)
ax6.set_title(f"F) Normal (μ={mu}, σ={sigma})", fontsize=14)
ax6.text(0.5, 0.85, "Application: 4", transform=ax6.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 7. Uniform
ax7 = plt.subplot(gs[3, 0])
a, b = 0, 1
x = np.linspace(-0.5, 1.5, 1000)
pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
ax7.plot(x, pdf, color='teal', linewidth=2)
ax7.set_title(f"G) Uniform (a={a}, b={b})", fontsize=14)
ax7.text(0.5, 0.85, "Application: 8", transform=ax7.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 8. Beta
ax8 = plt.subplot(gs[3, 1])
alpha, beta = 2, 5
x = np.linspace(0.001, 0.999, 1000)
pdf = stats.beta.pdf(x, alpha, beta)
ax8.plot(x, pdf, color='brown', linewidth=2)
ax8.set_title(f"H) Beta (α={alpha}, β={beta})", fontsize=14)
ax8.text(0.5, 0.85, "Application: 5", transform=ax8.transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
save_figure("distributions_summary.png")

# Step 10: Create relationships between distributions
print_step_header(10, "Relationships Between Distributions")

print("Relationships between distributions:")
print("1. Bernoulli(p) is a special case of Binomial(n,p) with n=1")
print("2. The Poisson distribution can be derived as a limit of Binomial(n,p) as n→∞, p→0, with np=λ")
print("3. The Exponential distribution gives the waiting time between events in a Poisson process")
print("4. The Normal distribution emerges from the Central Limit Theorem applied to various distributions")
print("5. The Beta distribution is a conjugate prior for the Bernoulli and Binomial distributions")
print()

# Create a distribution relationship diagram
plt.figure(figsize=(15, 10))

# Define positions
positions = {
    "bernoulli": (0.3, 0.7),
    "binomial": (0.3, 0.3),
    "geometric": (0.7, 0.7),
    "poisson": (0.7, 0.3),
    "exponential": (0.7, 0.1),
    "normal": (0.5, 0.5),
    "uniform": (0.1, 0.4),
    "beta": (0.1, 0.1)
}

# Plot distribution names as nodes
for dist, pos in positions.items():
    plt.plot(pos[0], pos[1], 'o', markersize=20, 
             color=cm.Set2(list(positions.keys()).index(dist) % 8))
    plt.text(pos[0], pos[1], dist.capitalize(), ha='center', va='center', 
             fontweight='bold', fontsize=12)

# Add arrows to show relationships
plt.annotate("", xy=positions["binomial"], xytext=positions["bernoulli"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))
plt.text(0.27, 0.5, "n=1", fontsize=10, ha='center')

plt.annotate("", xy=positions["poisson"], xytext=positions["binomial"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))
plt.text(0.5, 0.28, "n→∞, p→0, np=λ", fontsize=10, ha='center')

plt.annotate("", xy=positions["exponential"], xytext=positions["poisson"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))
plt.text(0.73, 0.2, "waiting time", fontsize=10, ha='center')

plt.annotate("", xy=positions["normal"], xytext=positions["binomial"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray', linestyle='dashed'))
plt.text(0.4, 0.42, "CLT", fontsize=10, ha='center')

plt.annotate("", xy=positions["bernoulli"], xytext=positions["beta"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray', linestyle='dashed'))
plt.text(0.2, 0.4, "conjugate prior", fontsize=10, ha='center')

plt.annotate("", xy=positions["normal"], xytext=positions["uniform"], 
             arrowprops=dict(arrowstyle="->", lw=1.5, color='gray', linestyle='dashed'))
plt.text(0.3, 0.48, "CLT", fontsize=10, ha='center')

# Set axis properties
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title("Relationships Between Probability Distributions", fontsize=16)

save_figure("distribution_relationships.png")

print("\nAll visualizations and explanations for probability distributions are complete!") 