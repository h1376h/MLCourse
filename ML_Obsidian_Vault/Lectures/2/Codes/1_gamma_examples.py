import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Create a subdirectory for gamma distribution images
images_dir = os.path.join(parent_dir, "Images", "Gamma_Distribution")
os.makedirs(images_dir, exist_ok=True)

def plot_gamma_distribution(alpha, lambda_, x_range, label, color):
    """Plot a gamma distribution with given parameters."""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = stats.gamma.pdf(x, a=alpha, scale=1/lambda_)
    plt.plot(x, y, label=label, color=color)
    return x, y

def calculate_gamma_stats(alpha, lambda_):
    """Calculate and print statistics for a gamma distribution."""
    mean = alpha / lambda_
    variance = alpha / (lambda_ ** 2)
    std_dev = np.sqrt(variance)
    return mean, variance, std_dev

def print_gamma_stats(alpha, lambda_):
    """Print statistics for a gamma distribution."""
    mean, variance, std_dev = calculate_gamma_stats(alpha, lambda_)
    print(f"\nGamma Distribution (α={alpha}, λ={lambda_}):")
    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    return mean, variance, std_dev

# Example 1: Basic Properties
print("\n=== Example 1: Basic Properties ===")
alpha1, lambda1 = 2, 0.5
mean1, var1, std1 = print_gamma_stats(alpha1, lambda1)

# Step 1: Plot basic distribution
plt.figure(figsize=(10, 6))
x, y = plot_gamma_distribution(alpha1, lambda1, (0, 10), f'Gamma(α={alpha1}, λ={lambda1})', 'blue')
plt.title('Step 1: Basic Gamma Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example1_step1.png'))
plt.close()

# Step 2: Add mean
plt.figure(figsize=(10, 6))
x, y = plot_gamma_distribution(alpha1, lambda1, (0, 10), f'Gamma(α={alpha1}, λ={lambda1})', 'blue')
plt.axvline(x=mean1, color='red', linestyle='--', label=f'Mean = {mean1:.2f}')
plt.title('Step 2: Distribution with Mean')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example1_step2.png'))
plt.close()

# Step 3: Add P(X < 4) area
plt.figure(figsize=(10, 6))
x, y = plot_gamma_distribution(alpha1, lambda1, (0, 10), f'Gamma(α={alpha1}, λ={lambda1})', 'blue')
plt.axvline(x=mean1, color='red', linestyle='--', label=f'Mean = {mean1:.2f}')
plt.fill_between(x, y, where=(x < 4), alpha=0.3, color='blue')
plt.title('Step 3: P(X < 4) Area')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example1_step3.png'))
plt.close()

# Step 4: Add 90th percentile
percentile_90 = stats.gamma.ppf(0.9, a=alpha1, scale=1/lambda1)
plt.figure(figsize=(10, 6))
x, y = plot_gamma_distribution(alpha1, lambda1, (0, 10), f'Gamma(α={alpha1}, λ={lambda1})', 'blue')
plt.axvline(x=mean1, color='red', linestyle='--', label=f'Mean = {mean1:.2f}')
plt.axvline(x=percentile_90, color='green', linestyle=':', label=f'90th Percentile = {percentile_90:.2f}')
plt.fill_between(x, y, where=(x < 4), alpha=0.3, color='blue')
plt.title('Step 4: Complete Distribution with All Features')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example1_step4.png'))
plt.close()

# Example 2: Shape Parameter Effect
print("\n=== Example 2: Shape Parameter Effect ===")
alphas = [1, 2, 3]
lambda_ = 1
colors = ['blue', 'green', 'red']

# Step 1: Plot individual distributions
for i, (alpha, color) in enumerate(zip(alphas, colors)):
    plt.figure(figsize=(10, 6))
    mean, var, std = print_gamma_stats(alpha, lambda_)
    plot_gamma_distribution(alpha, lambda_, (0, 8), f'Gamma(α={alpha}, λ={lambda_})', color)
    plt.title(f'Step 1.{i+1}: Gamma Distribution with α={alpha}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, f'gamma_example2_step1_{i+1}.png'))
    plt.close()

# Step 2: Plot all distributions together
plt.figure(figsize=(10, 6))
for alpha, color in zip(alphas, colors):
    mean, var, std = print_gamma_stats(alpha, lambda_)
    prob_less_than_2 = stats.gamma.cdf(2, a=alpha, scale=1/lambda_)
    print(f"P(X < 2) = {prob_less_than_2:.4f}")
    plot_gamma_distribution(alpha, lambda_, (0, 8), f'Gamma(α={alpha}, λ={lambda_})', color)
plt.title('Step 2: Combined View of Different Shape Parameters')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example2_step2.png'))
plt.close()

# Example 3: Rate Parameter Effect
print("\n=== Example 3: Rate Parameter Effect ===")
alpha = 2
lambdas = [0.5, 1, 2]

# Step 1: Plot individual distributions
for i, (lambda_, color) in enumerate(zip(lambdas, colors)):
    plt.figure(figsize=(10, 6))
    mean, var, std = print_gamma_stats(alpha, lambda_)
    plot_gamma_distribution(alpha, lambda_, (0, 8), f'Gamma(α={alpha}, λ={lambda_})', color)
    plt.title(f'Step 1.{i+1}: Gamma Distribution with λ={lambda_}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, f'gamma_example3_step1_{i+1}.png'))
    plt.close()

# Step 2: Plot all distributions together
plt.figure(figsize=(10, 6))
for lambda_, color in zip(lambdas, colors):
    mean, var, std = print_gamma_stats(alpha, lambda_)
    prob_less_than_3 = stats.gamma.cdf(3, a=alpha, scale=1/lambda_)
    print(f"P(X < 3) = {prob_less_than_3:.4f}")
    plot_gamma_distribution(alpha, lambda_, (0, 8), f'Gamma(α={alpha}, λ={lambda_})', color)
plt.title('Step 2: Combined View of Different Rate Parameters')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example3_step2.png'))
plt.close()

# Example 4: Special Cases
print("\n=== Example 4: Special Cases ===")
alpha_exp = 1
lambda_exp = 0.5

# Step 1: Plot Gamma distribution
plt.figure(figsize=(10, 6))
x_gamma, y_gamma = plot_gamma_distribution(alpha_exp, lambda_exp, (0, 8), 
                                         f'Gamma(α={alpha_exp}, λ={lambda_exp})', 'blue')
plt.title('Step 1: Gamma Distribution with α=1')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example4_step1.png'))
plt.close()

# Step 2: Plot Exponential distribution
plt.figure(figsize=(10, 6))
x_exp = np.linspace(0, 8, 1000)
y_exp = stats.expon.pdf(x_exp, scale=1/lambda_exp)
plt.plot(x_exp, y_exp, 'r--', label=f'Exponential(λ={lambda_exp})')
plt.title('Step 2: Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example4_step2.png'))
plt.close()

# Step 3: Plot both distributions together
plt.figure(figsize=(10, 6))
plot_gamma_distribution(alpha_exp, lambda_exp, (0, 8), 
                       f'Gamma(α={alpha_exp}, λ={lambda_exp})', 'blue')
plt.plot(x_exp, y_exp, 'r--', label=f'Exponential(λ={lambda_exp})')
plt.title('Step 3: Comparison of Gamma and Exponential Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'gamma_example4_step3.png'))
plt.close()

# Compare probabilities
x_values = [1, 2, 3, 4]
print("\nComparison of probabilities:")
print("x\tGamma\tExponential")
for x in x_values:
    p_gamma = stats.gamma.cdf(x, a=alpha_exp, scale=1/lambda_exp)
    p_exp = stats.expon.cdf(x, scale=1/lambda_exp)
    print(f"{x}\t{p_gamma:.4f}\t{p_exp:.4f}")

print(f"\nAll examples completed and visualizations saved to {images_dir}/") 