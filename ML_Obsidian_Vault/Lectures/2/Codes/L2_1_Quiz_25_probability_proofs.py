import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from matplotlib.gridspec import GridSpec
from scipy.special import comb

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    plt.close(fig)

# ==============================
# PROOF 1: Sum of Normal Distributions
# ==============================
print_step_header(1, "PROOF 1: Sum of Independent Normal Random Variables")

print("We will prove that the sum of two independent normally distributed random variables is also normally distributed.")
print("Consider X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²), and let Z = X + Y.")

print("\n1.1. Using moment generating functions (MGFs):")
print("The MGF of a random variable X is defined as M_X(t) = E[e^(tX)]")
print("For a normal distribution X ~ N(μ, σ²), the MGF is M_X(t) = exp(μt + σ²t²/2)")

print("\n1.2. Properties of MGFs:")
print("- If X and Y are independent, then M_{X+Y}(t) = M_X(t) · M_Y(t)")
print("- If two distributions have the same MGF, they are the same distribution")

print("\n1.3. MGF for X ~ N(μ₁, σ₁²):")
print("M_X(t) = exp(μ₁t + σ₁²t²/2)")

print("\n1.4. MGF for Y ~ N(μ₂, σ₂²):")
print("M_Y(t) = exp(μ₂t + σ₂²t²/2)")

print("\n1.5. MGF for Z = X + Y:")
print("M_Z(t) = M_X(t) · M_Y(t)")
print("       = exp(μ₁t + σ₁²t²/2) · exp(μ₂t + σ₂²t²/2)")
print("       = exp((μ₁ + μ₂)t + (σ₁² + σ₂²)t²/2)")

print("\n1.6. This is the MGF of a normal distribution with parameters:")
print("μ_Z = μ₁ + μ₂")
print("σ²_Z = σ₁² + σ₂²")

print("\n1.7. Therefore, Z = X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)")

# Visualization for Proof 1
fig1 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, width_ratios=[1, 1])

# Plot distributions
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, :])

# Parameters
mu1, sigma1 = 2, 1
mu2, sigma2 = 3, 1.5

# x-range for plotting
x = np.linspace(-2, 8, 1000)

# Plot distributions
dist1 = stats.norm(mu1, sigma1)
dist2 = stats.norm(mu2, sigma2)
dist_sum = stats.norm(mu1 + mu2, np.sqrt(sigma1**2 + sigma2**2))

ax1.plot(x, dist1.pdf(x), 'b-', linewidth=2)
ax1.fill_between(x, 0, dist1.pdf(x), alpha=0.3, color='blue')
ax1.set_title(f'X ~ N({mu1}, {sigma1}²)', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.grid(True)

ax2.plot(x, dist2.pdf(x), 'r-', linewidth=2)
ax2.fill_between(x, 0, dist2.pdf(x), alpha=0.3, color='red')
ax2.set_title(f'Y ~ N({mu2}, {sigma2}²)', fontsize=12)
ax2.set_xlabel('y', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.grid(True)

ax3.plot(x, dist1.pdf(x), 'b-', label=f'X ~ N({mu1}, {sigma1}²)', linewidth=2, alpha=0.5)
ax3.plot(x, dist2.pdf(x), 'r-', label=f'Y ~ N({mu2}, {sigma2}²)', linewidth=2, alpha=0.5)
ax3.plot(x, dist_sum.pdf(x), 'g-', 
         label=f'Z = X + Y ~ N({mu1 + mu2}, {sigma1**2 + sigma2**2})', linewidth=3)
ax3.fill_between(x, 0, dist_sum.pdf(x), alpha=0.3, color='green')
ax3.set_title('Sum of Independent Normal Distributions', fontsize=14)
ax3.set_xlabel('z', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
save_figure(fig1, "proof1_normal_sum.png")

# Simulation to verify
np.random.seed(42)
n_samples = 10000
X = np.random.normal(mu1, sigma1, n_samples)
Y = np.random.normal(mu2, sigma2, n_samples)
Z = X + Y

fig2 = plt.figure(figsize=(10, 6))
plt.hist(Z, bins=50, density=True, alpha=0.5, label='Simulated Z = X + Y')
z_range = np.linspace(min(Z), max(Z), 1000)
plt.plot(z_range, dist_sum.pdf(z_range), 'r-', 
         label=f'Theoretical N({mu1 + mu2}, {sigma1**2 + sigma2**2})', linewidth=2)
plt.title('Verification: Histogram of Z = X + Y vs. Theoretical Distribution', fontsize=14)
plt.xlabel('z', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_figure(fig2, "proof1_simulation.png")

print("\nConclusion: The sum of two independent normally distributed random variables is normally distributed.")
print(f"If X ~ N({mu1}, {sigma1}²) and Y ~ N({mu2}, {sigma2}²), then X + Y ~ N({mu1 + mu2}, {sigma1**2 + sigma2**2})")

# ==============================
# PROOF 2: E[(X - μ)²] = σ²
# ==============================
print_step_header(2, "PROOF 2: E[(X - μ)²] = σ²")

print("We'll prove that for any random variable X with finite mean μ and variance σ², E[(X - μ)²] = σ².")

print("\n2.1. By definition, the variance of X is:")
print("Var(X) = E[(X - μ)²]")
print("Where μ = E[X]")

print("\n2.2. This is the definition of variance itself, so the statement is true by definition.")
print("However, let's expand this to reinforce understanding:")

print("\n2.3. Alternative computation of variance:")
print("Var(X) = E[X²] - (E[X])²")
print("       = E[X²] - μ²")

print("\n2.4. Expand E[(X - μ)²]:")
print("E[(X - μ)²] = E[X² - 2μX + μ²]")
print("             = E[X²] - 2μE[X] + μ²")
print("             = E[X²] - 2μ² + μ²")
print("             = E[X²] - μ²")
print("             = Var(X)")

print("\n2.5. Therefore, E[(X - μ)²] = Var(X) = σ²")

# Visualization for Proof 2
# Create a simple visualization showing the concept of variance
fig3 = plt.figure(figsize=(10, 6))

# Generate some sample data
np.random.seed(42)
mu = 5
sigma = 1.5
data = np.random.normal(mu, sigma, 500)

plt.scatter(range(len(data)), data, alpha=0.5)
plt.axhline(y=mu, color='r', linestyle='-', label=f'μ = {mu}')
plt.axhline(y=mu+sigma, color='g', linestyle='--', label=f'μ + σ = {mu+sigma}')
plt.axhline(y=mu-sigma, color='g', linestyle='--', label=f'μ - σ = {mu-sigma}')

# Add a visual representation of (X - μ)²
for i in range(0, len(data), 50):  # Only show some examples for clarity
    plt.plot([i, i], [mu, data[i]], 'k-', alpha=0.3)
    plt.text(i, (mu + data[i])/2, f'X-μ', fontsize=8, ha='right')

plt.title('Visualization of Variance as E[(X - μ)²]', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_figure(fig3, "proof2_variance_definition.png")

# ==============================
# PROOF 3: Cov(X, Y) = 0 for Independent X, Y
# ==============================
print_step_header(3, "PROOF 3: Cov(X, Y) = 0 for Independent Variables")

print("We will prove that if random variables X and Y are independent, then Cov(X, Y) = 0.")

print("\n3.1. Definition of covariance:")
print("Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)]")
print("Where μ_X = E[X] and μ_Y = E[Y]")

print("\n3.2. Expand the covariance formula:")
print("Cov(X, Y) = E[XY - μ_Y·X - μ_X·Y + μ_X·μ_Y]")
print("           = E[XY] - μ_Y·E[X] - μ_X·E[Y] + μ_X·μ_Y")
print("           = E[XY] - μ_Y·μ_X - μ_X·μ_Y + μ_X·μ_Y")
print("           = E[XY] - μ_X·μ_Y")

print("\n3.3. If X and Y are independent, then:")
print("E[XY] = E[X]·E[Y] = μ_X·μ_Y")

print("\n3.4. Substituting into the covariance formula:")
print("Cov(X, Y) = μ_X·μ_Y - μ_X·μ_Y = 0")

print("\n3.5. Therefore, if X and Y are independent, Cov(X, Y) = 0")

# Visualization for Proof 3
# Create a visual demonstration of independence vs correlation
fig4 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2)

# Independent variables
np.random.seed(42)
x1 = np.random.normal(0, 1, 500)
y1 = np.random.normal(0, 1, 500)

# Correlated variables
x2 = np.random.normal(0, 1, 500)
y2 = 0.8 * x2 + np.random.normal(0, 0.5, 500)  # Positive correlation

# Anti-correlated variables
x3 = np.random.normal(0, 1, 500)
y3 = -0.8 * x3 + np.random.normal(0, 0.5, 500)  # Negative correlation

ax1 = fig4.add_subplot(gs[0, 0])
ax1.scatter(x1, y1, alpha=0.6)
ax1.set_title(f'Independent Variables\nCov(X,Y) ≈ {np.cov(x1, y1)[0,1]:.3f}', fontsize=12)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(True)

ax2 = fig4.add_subplot(gs[0, 1])
ax2.scatter(x2, y2, alpha=0.6)
ax2.set_title(f'Positively Correlated\nCov(X,Y) ≈ {np.cov(x2, y2)[0,1]:.3f}', fontsize=12)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.grid(True)

ax3 = fig4.add_subplot(gs[1, 0])
ax3.scatter(x3, y3, alpha=0.6)
ax3.set_title(f'Negatively Correlated\nCov(X,Y) ≈ {np.cov(x3, y3)[0,1]:.3f}', fontsize=12)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.grid(True)

# Theoretical explanation
ax4 = fig4.add_subplot(gs[1, 1])
ax4.axis('off')
ax4.text(0.5, 0.5, 
         "Independence implies Cov(X,Y) = 0\n\n" +
         "But Cov(X,Y) = 0 does not imply independence\n\n" +
         "For example, if Y = X² where X ~ N(0,1),\n" +
         "then Cov(X,Y) = 0 but X and Y are dependent",
         ha='center', va='center', fontsize=12)

plt.tight_layout()
save_figure(fig4, "proof3_covariance_independence.png")

# ==============================
# PROOF 4: Variance of Binomial Distribution
# ==============================
print_step_header(4, "PROOF 4: Variance of Binomial Distribution")

print("We will derive the formula for the variance of a binomial random variable with parameters n and p.")

print("\n4.1. Let X ~ Bin(n, p) be a binomial random variable.")
print("This can be represented as the sum of n independent Bernoulli random variables:")
print("X = X₁ + X₂ + ... + Xₙ, where each Xᵢ ~ Bernoulli(p)")

print("\n4.2. For a Bernoulli random variable Xᵢ:")
print("E[Xᵢ] = p")
print("Var(Xᵢ) = E[Xᵢ²] - (E[Xᵢ])² = p - p² = p(1-p)")

print("\n4.3. Since the Xᵢ are independent:")
print("Var(X) = Var(X₁ + X₂ + ... + Xₙ) = Var(X₁) + Var(X₂) + ... + Var(Xₙ)")

print("\n4.4. Since each Var(Xᵢ) = p(1-p):")
print("Var(X) = np(1-p)")

print("\n4.5. Therefore, for X ~ Bin(n, p), Var(X) = np(1-p)")

# Visualization for Proof 4
fig5 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2)

# Parameters
n_values = [10, 20, 30]
p_values = [0.2, 0.5, 0.8]

# Plot PMF for different values of n with fixed p
ax1 = fig5.add_subplot(gs[0, 0])
p_fixed = 0.3
for n in n_values:
    x = np.arange(0, n+1)
    pmf = stats.binom.pmf(x, n, p_fixed)
    ax1.plot(x, pmf, 'o-', label=f'n={n}, p={p_fixed}')
    
    # Mean and variance lines
    mean = n * p_fixed
    var = n * p_fixed * (1 - p_fixed)
    ax1.axvline(x=mean, linestyle='--', color='gray', alpha=0.3)
    ax1.axvline(x=mean-np.sqrt(var), linestyle=':', color='gray', alpha=0.3)
    ax1.axvline(x=mean+np.sqrt(var), linestyle=':', color='gray', alpha=0.3)

ax1.set_title(f'Binomial PMF with Fixed p={p_fixed}', fontsize=12)
ax1.set_xlabel('k', fontsize=10)
ax1.set_ylabel('P(X=k)', fontsize=10)
ax1.legend()
ax1.grid(True)

# Plot PMF for different values of p with fixed n
ax2 = fig5.add_subplot(gs[0, 1])
n_fixed = 15
for p in p_values:
    x = np.arange(0, n_fixed+1)
    pmf = stats.binom.pmf(x, n_fixed, p)
    ax2.plot(x, pmf, 'o-', label=f'n={n_fixed}, p={p}')
    
    # Mean and variance lines
    mean = n_fixed * p
    var = n_fixed * p * (1 - p)
    ax2.axvline(x=mean, linestyle='--', color='gray', alpha=0.3)

ax2.set_title(f'Binomial PMF with Fixed n={n_fixed}', fontsize=12)
ax2.set_xlabel('k', fontsize=10)
ax2.set_ylabel('P(X=k)', fontsize=10)
ax2.legend()
ax2.grid(True)

# Plot variance as a function of p
ax3 = fig5.add_subplot(gs[1, 0])
p_range = np.linspace(0, 1, 100)
for n in n_values:
    var = n * p_range * (1 - p_range)
    ax3.plot(p_range, var, '-', label=f'n={n}')

ax3.set_title('Variance of Binomial Distribution vs. p', fontsize=12)
ax3.set_xlabel('p', fontsize=10)
ax3.set_ylabel('Var(X)', fontsize=10)
ax3.legend()
ax3.grid(True)

# Theoretical vs. simulated variance
ax4 = fig5.add_subplot(gs[1, 1])
np.random.seed(42)
n_sim = 20
p_sim = 0.4
theoretical_var = n_sim * p_sim * (1 - p_sim)

# Simulate binomial random variables
n_samples = 1000
samples = np.random.binomial(n_sim, p_sim, n_samples)
sample_var = np.var(samples)

# Compare theoretical and simulated variance
ax4.hist(samples, bins=range(n_sim+2), density=True, alpha=0.7, 
         label=f'Simulated Var(X) = {sample_var:.3f}')
x = np.arange(0, n_sim+1)
pmf = stats.binom.pmf(x, n_sim, p_sim)
ax4.plot(x, pmf, 'ro-', label=f'Theoretical Var(X) = {theoretical_var:.3f}')

ax4.set_title(f'Binomial Distribution: n={n_sim}, p={p_sim}', fontsize=12)
ax4.set_xlabel('k', fontsize=10)
ax4.set_ylabel('P(X=k)', fontsize=10)
ax4.legend()
ax4.grid(True)

plt.tight_layout()
save_figure(fig5, "proof4_binomial_variance.png")

# Summary
print_step_header(5, "SUMMARY: Mathematical Proofs")

print("Key takeaways from the four proofs:")
print("\n1. Sum of Independent Normal Random Variables:")
print("   • If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²), then X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)")
print("   • This is a special property of normal distributions")
print("   • The proof uses moment generating functions")

print("\n2. Definition of Variance:")
print("   • For any random variable X, E[(X - μ)²] = σ²")
print("   • This is the definition of variance")
print("   • Can also be expressed as Var(X) = E[X²] - (E[X])²")

print("\n3. Independence and Covariance:")
print("   • If X and Y are independent, then Cov(X, Y) = 0")
print("   • Note that the converse is not generally true: Cov(X, Y) = 0 doesn't necessarily imply independence")
print("   • Independence means E[XY] = E[X]E[Y]")

print("\n4. Variance of Binomial Distribution:")
print("   • For X ~ Bin(n, p), Var(X) = np(1-p)")
print("   • This result comes from viewing a binomial as a sum of independent Bernoulli trials")
print("   • Each Bernoulli trial has variance p(1-p)")

print("\nThese are fundamental results in probability theory with important applications in statistics and machine learning.") 