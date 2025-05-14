import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate true function (target model)
def true_function(x):
    return 1.5 * x + 0.5

# Generate data with noise
def generate_data(n_samples):
    x = np.random.uniform(-1, 1, n_samples)
    x = np.sort(x)  # Sort for better visualization
    y = true_function(x) + 0.5 * np.random.randn(n_samples)
    return x, y

# Fit linear regression
def fit_linear_regression(x, y):
    # Add bias term
    X = np.vstack([np.ones(len(x)), x]).T
    # Compute weights using normal equations
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Create predictions
def predict(x, w):
    return w[0] + w[1] * x

# Create figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Detailed visualization line
x_line = np.linspace(-1, 1, 100)
y_line = true_function(x_line)

# Generate and plot for n=10, n=20, and n=50
sample_sizes = [10, 20, 50]
colors = ['g', 'b', 'purple']
weights = []

for i, n in enumerate(sample_sizes):
    x, y = generate_data(n)
    w = fit_linear_regression(x, y)
    weights.append(w)
    
    # Plot data points and fitted line
    axs[i].scatter(x, y, color='navy', s=25)
    axs[i].plot(x_line, predict(x_line, w), color=colors[i], linewidth=2)
    axs[i].set_xlim(-1, 1)
    axs[i].set_ylim(-3, 3)
    axs[i].set_title(f'n = {n}', fontsize=14)
    axs[i].grid(True, linestyle='--', alpha=0.7)

# Plot comparison of all models
axs[3].plot(x_line, y_line, 'r--', linewidth=2, label='target')
for i, n in enumerate(sample_sizes):
    axs[3].plot(x_line, predict(x_line, weights[i]), color=colors[i], linewidth=2, label=f'n={n}')
axs[3].set_xlim(-1, 1)
axs[3].set_ylim(-3, 3)
axs[3].legend(loc='lower right', fontsize=12)
axs[3].grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('plots/linear_regression_training_examples.png', dpi=300)
plt.close()

print("Linear regression training examples plot saved as 'plots/linear_regression_training_examples.png'") 