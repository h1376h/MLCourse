import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

print("\n=== ADVANCED PROBABILITY CONCEPTS IN ML: STEP-BY-STEP EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "L2_1_ML")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: Probability Bounds for Model Errors
print("Example 1: Probability Bounds for Model Errors")
print("====================================================\n")

print("Problem: We have a binary classification model and want to bound the probability")
print("that the model's test error exceeds its training error by more than some threshold ε.")
print("\n")

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get predictions and calculate errors
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_errors = (y_train_pred != y_train).astype(int)
test_errors = (y_test_pred != y_test).astype(int)

train_error_rate = np.mean(train_errors)
test_error_rate = np.mean(test_errors)

print(f"Step 1: Calculate actual training and test errors")
print(f"Training error rate: {train_error_rate:.4f}")
print(f"Test error rate: {test_error_rate:.4f}")
print(f"Difference (|test - train|): {abs(test_error_rate - train_error_rate):.4f}")
print()

# Step 2: Apply Hoeffding's inequality to bound the probability
n_train = len(y_train)
n_test = len(y_test)
epsilon_values = np.linspace(0.01, 0.2, 20)
hoeffding_bounds = []

print(f"Step 2: Apply Hoeffding's inequality to bound the probability")
print(f"P(|Êtest - Êtrain| > ε) ≤ 2exp(-2ε²/(1/n_train + 1/n_test))")
print()

print(f"For our dataset with n_train = {n_train} and n_test = {n_test}:")
for epsilon in epsilon_values:
    # Calculate Hoeffding bound
    bound = 2 * np.exp(-2 * epsilon**2 / (1/n_train + 1/n_test))
    hoeffding_bounds.append(bound)
    
    if round(epsilon, 2) in [0.05, 0.1, 0.15]:
        print(f"For ε = {epsilon:.2f}:")
        print(f"  P(|Êtest - Êtrain| > {epsilon:.2f}) ≤ 2exp(-2({epsilon:.2f})²/({1/n_train:.6f} + {1/n_test:.6f}))")
        print(f"  P(|Êtest - Êtrain| > {epsilon:.2f}) ≤ 2exp({-2 * epsilon**2 / (1/n_train + 1/n_test):.4f})")
        print(f"  P(|Êtest - Êtrain| > {epsilon:.2f}) ≤ {bound:.4f}")
        print()

# Visualize the Hoeffding bound
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, hoeffding_bounds, 'b-', linewidth=2, label="Hoeffding Bound")
plt.axhline(y=0.05, color='r', linestyle='--', label="5% probability")
plt.axvline(x=abs(test_error_rate - train_error_rate), color='g', linestyle='--', 
            label=f"Actual difference: {abs(test_error_rate - train_error_rate):.4f}")

plt.xlabel("Error Difference (ε)", fontsize=12)
plt.ylabel("Probability Bound", fontsize=12)
plt.title("Hoeffding's Inequality: Probability Bound for |Êtest - Êtrain| > ε", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(images_dir, 'hoeffding_bound_model_error.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Markov and Chebyshev Inequalities
print("Example 2: Markov and Chebyshev Inequalities")
print("====================================================\n")

print("Problem: We want to understand how feature importance estimates vary")
print("and use probability inequalities to bound the probability of large deviations.")
print("\n")

# Generate dataset with different feature importance levels
n_samples = 1000
n_features = 5

# Create a dataset where features have decreasing importance
X = np.random.randn(n_samples, n_features)
# Generate y as a linear combination of features with decreasing importance
coefs = np.array([0.5, 0.25, 0.1, 0.05, 0.01])
y = X.dot(coefs) + 0.1 * np.random.randn(n_samples)

# Calculate feature importance using correlation
feature_importance = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_features)])

print(f"Step 1: Calculate true feature importance (correlation with target)")
print(f"Feature importance: {feature_importance}")
print()

# Using Markov's inequality
print(f"Step 2: Apply Markov's inequality to bound the probability of overestimating feature importance")
print(f"For a random variable X ≥ 0 and threshold a > 0: P(X ≥ a) ≤ E[X]/a")
print()

# Simulate bootstrap sampling to estimate feature importance distribution
n_bootstrap = 1000
bootstrap_importance = np.zeros((n_bootstrap, n_features))

for i in range(n_bootstrap):
    # Bootstrap sampling
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    
    # Calculate feature importance on bootstrap sample
    for j in range(n_features):
        bootstrap_importance[i, j] = np.abs(np.corrcoef(X_bootstrap[:, j], y_bootstrap)[0, 1])

# Calculate mean and variance of feature importance estimates
importance_mean = np.mean(bootstrap_importance, axis=0)
importance_var = np.var(bootstrap_importance, axis=0)
importance_std = np.sqrt(importance_var)

# Apply Markov's inequality to first feature
feature_idx = 0
threshold_values = np.linspace(importance_mean[feature_idx], 
                              importance_mean[feature_idx] + 3*importance_std[feature_idx], 
                              10)

print(f"For Feature 1 (importance = {feature_importance[feature_idx]:.4f}):")
print(f"Mean estimated importance: {importance_mean[feature_idx]:.4f}")
print(f"Std dev of estimated importance: {importance_std[feature_idx]:.4f}")
print()

markov_bounds = []
chebyshev_bounds = []

for a in threshold_values:
    # Calculate deviation from mean in units of standard deviation
    k = (a - importance_mean[feature_idx]) / importance_std[feature_idx]
    
    # Markov bound (applicable only for threshold > mean)
    if a > importance_mean[feature_idx]:
        X_shifted = bootstrap_importance[:, feature_idx] - importance_mean[feature_idx]
        X_shifted_positive = np.maximum(X_shifted, 0)  # Make sure it's non-negative
        markov_bound = np.mean(X_shifted_positive) / (a - importance_mean[feature_idx])
        markov_bounds.append(min(1, markov_bound))
    else:
        markov_bounds.append(1)
    
    # Chebyshev bound
    if k > 0:
        chebyshev_bound = 1 / (k**2)
        chebyshev_bounds.append(min(1, chebyshev_bound))
    else:
        chebyshev_bounds.append(1)
    
    if round(a - importance_mean[feature_idx], 2) in [0.05, 0.10, 0.15]:
        print(f"For threshold a = {a:.4f} (mean + {a - importance_mean[feature_idx]:.2f}):")
        if a > importance_mean[feature_idx]:
            print(f"  Markov: P(importance ≥ {a:.4f}) ≤ {markov_bounds[-1]:.4f}")
        print(f"  Chebyshev: P(|importance - {importance_mean[feature_idx]:.4f}| ≥ {a - importance_mean[feature_idx]:.4f}) ≤ {chebyshev_bounds[-1]:.4f}")
        
        # Calculate empirical probability from bootstrap samples
        empirical_prob = np.mean(bootstrap_importance[:, feature_idx] >= a)
        print(f"  Empirical probability from bootstrap: {empirical_prob:.4f}")
        print()

# Visualize the bounds compared to empirical distribution
plt.figure(figsize=(10, 6))
deviations = [a - importance_mean[feature_idx] for a in threshold_values]

# Plot the bounds
plt.plot(deviations, chebyshev_bounds, 'r-', linewidth=2, label="Chebyshev Bound")
plt.plot(deviations, markov_bounds, 'g-', linewidth=2, label="Markov Bound")

# Plot the empirical probabilities
empirical_probs = [np.mean(bootstrap_importance[:, feature_idx] >= a) for a in threshold_values]
plt.plot(deviations, empirical_probs, 'b-', linewidth=2, label="Empirical Probability")

plt.xlabel("Deviation from Mean", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("Probability Bounds for Feature Importance Estimation", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(images_dir, 'markov_chebyshev_bounds.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Central Limit Theorem in ML
print("Example 3: Central Limit Theorem in ML")
print("====================================================\n")

print("Problem: We want to understand how batch size affects gradient estimation")
print("in mini-batch training using the Central Limit Theorem.")
print("\n")

# Simulate gradient estimates with different batch sizes
n_samples = 10000
true_gradient = 2.5
variance = 4.0

# Individual gradient estimates (one per sample)
gradients = true_gradient + np.sqrt(variance) * np.random.randn(n_samples)

print(f"Step 1: Simulate individual gradient estimates")
print(f"True gradient: {true_gradient}")
print(f"Variance of individual estimates: {variance}")
print(f"Mean of sampled gradients: {np.mean(gradients):.4f}")
print(f"Variance of sampled gradients: {np.var(gradients):.4f}")
print()

# Test different batch sizes
batch_sizes = [1, 10, 50, 200]
colors = ['r', 'g', 'b', 'purple']

plt.figure(figsize=(12, 8))

for i, batch_size in enumerate(batch_sizes):
    print(f"Step 2: Analyze gradients with batch size {batch_size}")
    
    # Calculate number of batches
    n_batches = n_samples // batch_size
    
    # Compute batch gradients
    batch_gradients = np.zeros(n_batches)
    for j in range(n_batches):
        batch_indices = range(j * batch_size, (j + 1) * batch_size)
        batch_gradients[j] = np.mean(gradients[batch_indices])
    
    # Theoretical mean and variance
    theoretical_mean = true_gradient
    theoretical_variance = variance / batch_size
    theoretical_std = np.sqrt(theoretical_variance)
    
    # Empirical statistics
    empirical_mean = np.mean(batch_gradients)
    empirical_variance = np.var(batch_gradients)
    empirical_std = np.sqrt(empirical_variance)
    
    print(f"  Theoretical variance: {theoretical_variance:.4f}")
    print(f"  Empirical variance: {empirical_variance:.4f}")
    print(f"  Theoretical standard deviation: {theoretical_std:.4f}")
    print(f"  Empirical standard deviation: {empirical_std:.4f}")
    
    # Test normality
    _, p_value = stats.normaltest(batch_gradients)
    print(f"  Normality test p-value: {p_value:.4f} ({'Normal' if p_value > 0.05 else 'Not normal'})")
    print()
    
    # Plot histogram of batch gradients
    plt.subplot(2, 2, i+1)
    sns.histplot(batch_gradients, kde=True, color=colors[i], stat="density", alpha=0.6)
    
    # Plot the theoretical normal distribution
    x = np.linspace(true_gradient - 4*theoretical_std, true_gradient + 4*theoretical_std, 1000)
    pdf = stats.norm.pdf(x, theoretical_mean, theoretical_std)
    plt.plot(x, pdf, color='black', linestyle='--')
    
    plt.title(f"Batch Size = {batch_size}\nσ² = {theoretical_variance:.4f}", fontsize=12)
    plt.xlabel("Gradient Value", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'clt_batch_gradients.png'), dpi=100, bbox_inches='tight')
plt.close()

# Plot relationship between batch size and standard deviation
plt.figure(figsize=(10, 6))
batch_sizes_extended = np.linspace(1, 200, 100)
theoretical_stds = np.sqrt(variance / batch_sizes_extended)

plt.plot(batch_sizes_extended, theoretical_stds, 'b-', linewidth=2, label="Theoretical: $\\sigma/\\sqrt{b}$")

# Add empirical points
empirical_stds = []
for batch_size in [1, 5, 10, 20, 50, 100, 200]:
    n_batches = n_samples // batch_size
    batch_gradients = np.zeros(n_batches)
    for j in range(n_batches):
        batch_indices = range(j * batch_size, (j + 1) * batch_size)
        batch_gradients[j] = np.mean(gradients[batch_indices])
    empirical_stds.append(np.std(batch_gradients))

plt.scatter([1, 5, 10, 20, 50, 100, 200], empirical_stds, color='r', s=50, label="Empirical")

plt.xlabel("Batch Size", fontsize=12)
plt.ylabel("Standard Deviation of Gradient", fontsize=12)
plt.title("Relationship Between Batch Size and Gradient Variance", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(images_dir, 'batch_size_vs_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Monte Carlo Methods
print("Example 4: Monte Carlo Methods")
print("====================================================\n")

print("Problem: We want to estimate an expectation using Monte Carlo sampling")
print("and understand how the number of samples affects the accuracy.")
print("\n")

# Define a function to estimate: E[sin(X)²] where X ~ N(0, 1)
def f(x):
    return np.sin(x)**2

# True value (theoretical)
true_value = 0.5  # E[sin(X)²] = 0.5 for X ~ N(0, 1)

print("Step 1: Define our estimation problem")
print("We want to estimate E[sin(X)²] where X ~ N(0, 1)")
print(f"The true value is {true_value}")
print()

# Sample sizes to try
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
n_trials = 100  # Number of trials for each sample size

results = []
for n_samples in sample_sizes:
    print(f"Step 2: Estimate with {n_samples} samples")
    
    estimates = []
    for _ in range(n_trials):
        # Generate samples from N(0, 1)
        X = np.random.randn(n_samples)
        
        # Compute Monte Carlo estimate
        mc_estimate = np.mean(f(X))
        estimates.append(mc_estimate)
    
    # Calculate statistics of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)
    
    # Theoretical standard error
    theoretical_se = 1 / np.sqrt(n_samples)  # This is approximate
    
    print(f"  Mean estimate: {mean_estimate:.6f}")
    print(f"  Standard deviation of estimates: {std_estimate:.6f}")
    print(f"  Theoretical standard error: {theoretical_se:.6f}")
    print(f"  Error: {abs(mean_estimate - true_value):.6f}")
    print()
    
    results.append({
        'n_samples': n_samples,
        'mean_estimate': mean_estimate,
        'std_estimate': std_estimate,
        'theo_se': theoretical_se,
        'error': abs(mean_estimate - true_value)
    })

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Estimation error vs. sample size
plt.subplot(1, 2, 1)
plt.errorbar([r['n_samples'] for r in results], 
             [r['mean_estimate'] for r in results], 
             yerr=[r['std_estimate'] for r in results], 
             fmt='o-', capsize=5)
plt.axhline(y=true_value, color='r', linestyle='--', label=f"True value: {true_value}")
plt.xscale('log')
plt.xlabel("Number of Samples", fontsize=10)
plt.ylabel("Estimate", fontsize=10)
plt.title("Monte Carlo Estimation of E[sin(X)²]", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Standard error vs. sample size
plt.subplot(1, 2, 2)
plt.loglog([r['n_samples'] for r in results], [r['std_estimate'] for r in results], 'o-', label="Empirical")
plt.loglog([r['n_samples'] for r in results], [r['theo_se'] for r in results], 's--', label="Theoretical")
plt.xlabel("Number of Samples", fontsize=10)
plt.ylabel("Standard Error", fontsize=10)
plt.title("Monte Carlo Standard Error", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'monte_carlo_convergence.png'), dpi=100, bbox_inches='tight')
plt.close()

print("All advanced probability concept images created successfully.") 