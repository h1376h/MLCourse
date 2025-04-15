import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== ADVANCED GAMMA DISTRIBUTION EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Shape and Scale Parameter Effects
print("Example 1: Shape and Scale Parameter Effects")
plt.figure(figsize=(15, 5))

# Different shape parameters (k)
plt.subplot(1, 3, 1)
x = np.linspace(0, 10, 1000)
for k in [0.5, 1, 2, 3, 5]:
    pdf = stats.gamma.pdf(x, k, scale=1)
    plt.plot(x, pdf, label=f'k={k}')
plt.title('Effect of Shape Parameter (k)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Different scale parameters (θ)
plt.subplot(1, 3, 2)
for theta in [0.5, 1, 2, 3]:
    pdf = stats.gamma.pdf(x, 2, scale=theta)
    plt.plot(x, pdf, label=f'θ={theta}')
plt.title('Effect of Scale Parameter (θ)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Rate parameter (λ = 1/θ)
plt.subplot(1, 3, 3)
for rate in [0.5, 1, 2, 3]:
    pdf = stats.gamma.pdf(x, 2, scale=1/rate)
    plt.plot(x, pdf, label=f'λ={rate}')
plt.title('Effect of Rate Parameter (λ)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_parameter_effects.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Special Cases
print("\nExample 2: Special Cases")
plt.figure(figsize=(15, 5))

# Exponential Distribution (k=1)
plt.subplot(1, 3, 1)
x = np.linspace(0, 5, 1000)
pdf_exp = stats.gamma.pdf(x, 1, scale=1)
plt.plot(x, pdf_exp, 'b-', label='Gamma(k=1)')
plt.plot(x, stats.expon.pdf(x), 'r--', label='Exponential')
plt.title('Exponential as Special Case (k=1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Chi-squared Distribution
plt.subplot(1, 3, 2)
x = np.linspace(0, 10, 1000)
for df in [1, 2, 3, 5]:
    pdf_chi2 = stats.gamma.pdf(x, df/2, scale=2)
    plt.plot(x, pdf_chi2, label=f'df={df}')
plt.title('Chi-squared as Special Case')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Erlang Distribution
plt.subplot(1, 3, 3)
x = np.linspace(0, 10, 1000)
for k in [1, 2, 3, 5]:
    pdf_erlang = stats.gamma.pdf(x, k, scale=1)
    plt.plot(x, pdf_erlang, label=f'k={k}')
plt.title('Erlang as Special Case (integer k)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_special_cases.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Additive Property
print("\nExample 3: Additive Property")
plt.figure(figsize=(15, 5))

# Sum of independent Gamma variables
n_samples = 10000
k1, theta1 = 2, 1
k2, theta2 = 3, 1

# Generate samples
X1 = np.random.gamma(k1, theta1, n_samples)
X2 = np.random.gamma(k2, theta2, n_samples)
X_sum = X1 + X2

# Plot histograms
plt.subplot(1, 3, 1)
plt.hist(X1, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 10, 1000)
pdf = stats.gamma.pdf(x, k1, scale=theta1)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title(f'Gamma({k1}, {theta1})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(X2, bins=50, density=True, alpha=0.7)
pdf = stats.gamma.pdf(x, k2, scale=theta2)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title(f'Gamma({k2}, {theta2})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(X_sum, bins=50, density=True, alpha=0.7)
pdf = stats.gamma.pdf(x, k1 + k2, scale=theta1)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title(f'Sum: Gamma({k1 + k2}, {theta1})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_additive_property.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Bayesian Applications
print("\nExample 4: Bayesian Applications")
plt.figure(figsize=(15, 5))

# Prior and Posterior
k_prior, theta_prior = 2, 1
data = np.random.exponential(scale=2, size=100)
k_posterior = k_prior + len(data)
theta_posterior = theta_prior / (1 + theta_prior * np.sum(data))

x = np.linspace(0, 5, 1000)
plt.subplot(1, 3, 1)
plt.plot(x, stats.gamma.pdf(x, k_prior, scale=theta_prior), 'b-', label='Prior')
plt.plot(x, stats.gamma.pdf(x, k_posterior, scale=theta_posterior), 'r-', label='Posterior')
plt.title('Prior and Posterior')
plt.xlabel('λ')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Effect of Sample Size
plt.subplot(1, 3, 2)
sample_sizes = [10, 50, 100, 500]
for n in sample_sizes:
    data = np.random.exponential(scale=2, size=n)
    k_post = k_prior + n
    theta_post = theta_prior / (1 + theta_prior * np.sum(data))
    plt.plot(x, stats.gamma.pdf(x, k_post, scale=theta_post), label=f'n={n}')
plt.title('Effect of Sample Size')
plt.xlabel('λ')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Predictive Distribution
plt.subplot(1, 3, 3)
x_pred = np.linspace(0, 10, 1000)
for n in [10, 50, 100]:
    data = np.random.exponential(scale=2, size=n)
    k_post = k_prior + n
    theta_post = theta_prior / (1 + theta_prior * np.sum(data))
    pred_pdf = stats.gamma.pdf(x_pred, k_post, scale=theta_post)
    plt.plot(x_pred, pred_pdf, label=f'n={n}')
plt.title('Predictive Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_bayesian.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Applications in Machine Learning
print("\nExample 5: Applications in Machine Learning")
plt.figure(figsize=(15, 5))

# Poisson Process
plt.subplot(1, 3, 1)
n_events = 1000
inter_arrival_times = np.random.exponential(scale=1, size=n_events)
arrival_times = np.cumsum(inter_arrival_times)
plt.hist(arrival_times, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 10, 1000)
pdf = stats.gamma.pdf(x, n_events, scale=1)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Poisson Process Arrival Times')
plt.xlabel('Time')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Bayesian Linear Regression
plt.subplot(1, 3, 2)
n_points = 100
x = np.linspace(0, 10, n_points)
true_slope = 2
true_intercept = 1
noise_std = 1
y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n_points)

# Bayesian inference for precision (1/variance)
k_prior, theta_prior = 1, 1
residuals = y - (true_slope * x + true_intercept)
k_post = k_prior + n_points/2
theta_post = theta_prior + np.sum(residuals**2)/2

x_precision = np.linspace(0, 2, 1000)
plt.plot(x_precision, stats.gamma.pdf(x_precision, k_prior, scale=theta_prior), 'b-', label='Prior')
plt.plot(x_precision, stats.gamma.pdf(x_precision, k_post, scale=theta_post), 'r-', label='Posterior')
plt.title('Precision in Bayesian Regression')
plt.xlabel('Precision (1/σ²)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Hierarchical Models
plt.subplot(1, 3, 3)
n_groups = 5
n_per_group = 20
group_means = np.random.normal(0, 1, n_groups)
data = np.array([np.random.normal(mu, 1, n_per_group) for mu in group_means])

# Hierarchical model for group variances
k_prior, theta_prior = 1, 1
group_variances = np.var(data, axis=1)
k_post = k_prior + n_per_group/2
theta_post = theta_prior + np.sum(group_variances)/2

x_var = np.linspace(0, 3, 1000)
plt.plot(x_var, stats.gamma.pdf(x_var, k_prior, scale=theta_prior), 'b-', label='Prior')
plt.plot(x_var, stats.gamma.pdf(x_var, k_post, scale=theta_post), 'r-', label='Posterior')
plt.title('Hierarchical Model Variances')
plt.xlabel('Variance')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_ml_applications.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Distribution Relationships
print("\nExample 6: Distribution Relationships")
plt.figure(figsize=(15, 5))

# Gamma and Exponential
plt.subplot(1, 3, 1)
x = np.linspace(0, 5, 1000)
plt.plot(x, stats.gamma.pdf(x, 1, scale=1), 'b-', label='Gamma(k=1)')
plt.plot(x, stats.expon.pdf(x), 'r--', label='Exponential')
plt.title('Gamma and Exponential')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Gamma and Chi-squared
plt.subplot(1, 3, 2)
x = np.linspace(0, 10, 1000)
df = 2
plt.plot(x, stats.gamma.pdf(x, df/2, scale=2), 'b-', label='Gamma(k=1,θ=2)')
plt.plot(x, stats.chi2.pdf(x, df), 'r--', label='Chi-squared(df=2)')
plt.title('Gamma and Chi-squared')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Gamma and Poisson
plt.subplot(1, 3, 3)
x = np.arange(0, 20)
lambda_poisson = 5
plt.bar(x, stats.poisson.pmf(x, lambda_poisson), alpha=0.7, label='Poisson')
x_gamma = np.linspace(0, 20, 1000)
plt.plot(x_gamma, stats.gamma.pdf(x_gamma, lambda_poisson, scale=1), 'r-', label='Gamma')
plt.title('Gamma and Poisson')
plt.xlabel('x')
plt.ylabel('Probability/Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_distribution_relationships.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll advanced gamma distribution example images created successfully.") 