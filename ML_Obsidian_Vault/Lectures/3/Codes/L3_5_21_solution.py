import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import pandas as pd
from numpy.linalg import eigvals

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Linear Regression Optimization with Gradient Descent")
print()

# Part 1: Derive the gradient of the cost function
print("## Part 1: Deriving the Gradient of the SSE Cost Function")
print("The cost function is given by:")
print("$$J(\\boldsymbol{w}) = \\sum_{i=1}^{n} (y^{(i)} - \\boldsymbol{w}^T \\boldsymbol{x}^{(i)})^2$$")
print()
print("To find the gradient, we need to differentiate with respect to the parameter vector w.")
print("First, let's rewrite this in matrix form:")
print("$$J(\\boldsymbol{w}) = (\\boldsymbol{y} - \\boldsymbol{X}\\boldsymbol{w})^T(\\boldsymbol{y} - \\boldsymbol{X}\\boldsymbol{w})$$")
print()
print("Expanding:")
print("$$J(\\boldsymbol{w}) = \\boldsymbol{y}^T\\boldsymbol{y} - \\boldsymbol{y}^T\\boldsymbol{X}\\boldsymbol{w} - \\boldsymbol{w}^T\\boldsymbol{X}^T\\boldsymbol{y} + \\boldsymbol{w}^T\\boldsymbol{X}^T\\boldsymbol{X}\\boldsymbol{w}$$")
print()
print("Since $\\boldsymbol{y}^T\\boldsymbol{X}\\boldsymbol{w}$ is a scalar, it equals its transpose $\\boldsymbol{w}^T\\boldsymbol{X}^T\\boldsymbol{y}$. So:")
print("$$J(\\boldsymbol{w}) = \\boldsymbol{y}^T\\boldsymbol{y} - 2\\boldsymbol{w}^T\\boldsymbol{X}^T\\boldsymbol{y} + \\boldsymbol{w}^T\\boldsymbol{X}^T\\boldsymbol{X}\\boldsymbol{w}$$")
print()
print("Now we differentiate with respect to $\\boldsymbol{w}$. Using matrix calculus rules:")
print("$$\\nabla_\\boldsymbol{w} J(\\boldsymbol{w}) = -2\\boldsymbol{X}^T\\boldsymbol{y} + 2\\boldsymbol{X}^T\\boldsymbol{X}\\boldsymbol{w}$$")
print()
print("This simplifies to:")
print("$$\\nabla_\\boldsymbol{w} J(\\boldsymbol{w}) = 2\\boldsymbol{X}^T(\\boldsymbol{X}\\boldsymbol{w} - \\boldsymbol{y})$$")
print()
print("Or, in component form:")
print("$$\\nabla_\\boldsymbol{w} J(\\boldsymbol{w}) = 2\\sum_{i=1}^{n} \\boldsymbol{x}^{(i)}(\\boldsymbol{w}^T\\boldsymbol{x}^{(i)} - y^{(i)})$$")
print()
print("This is the gradient we'll use in our gradient descent algorithm.")
print()

# Part 2: Implement batch gradient descent
print("## Part 2: Batch Gradient Descent Update Rule")
print("The update rule for batch gradient descent is:")
print("$$\\boldsymbol{w}_{t+1} = \\boldsymbol{w}_t - \\alpha \\nabla_\\boldsymbol{w} J(\\boldsymbol{w}_t)$$")
print()
print("Substituting our gradient:")
print("$$\\boldsymbol{w}_{t+1} = \\boldsymbol{w}_t - \\alpha (2\\boldsymbol{X}^T(\\boldsymbol{X}\\boldsymbol{w}_t - \\boldsymbol{y}))$$")
print("$$\\boldsymbol{w}_{t+1} = \\boldsymbol{w}_t - 2\\alpha\\boldsymbol{X}^T(\\boldsymbol{X}\\boldsymbol{w}_t - \\boldsymbol{y})$$")
print()
print("Pseudocode for batch gradient descent:")
print("```")
print("Algorithm: Batch Gradient Descent for Linear Regression")
print("Input: X (design matrix), y (target vector), α (learning rate), max_iterations, tolerance")
print("Output: w (optimal weight vector)")
print("1. Initialize w randomly")
print("2. For i = 1 to max_iterations:")
print("   a. Compute predictions: y_pred = X·w")
print("   b. Compute error: error = y_pred - y")
print("   c. Compute gradient: gradient = 2·X^T·error")
print("   d. Update weights: w = w - α·gradient")
print("   e. Compute cost: J = sum((y - y_pred)^2)")
print("   f. If change in cost < tolerance:")
print("      i. Break loop (converged)")
print("3. Return w")
print("```")
print()

# Create a synthetic dataset for demonstration
def generate_data(n=100, d=2, noise=0.5, seed=42):
    """Generate synthetic data for linear regression."""
    np.random.seed(seed)
    X = np.random.randn(n, d)
    
    # Add a column of ones for the intercept
    X = np.column_stack((np.ones(n), X))
    
    # True parameters (including intercept)
    w_true = np.array([3, 1.5, -2])
    
    # Generate target values with noise
    y = X @ w_true + noise * np.random.randn(n)
    
    return X, y, w_true

# Implement batch gradient descent
def batch_gradient_descent(X, y, alpha=0.01, max_iterations=1000, tolerance=1e-6):
    """Implement batch gradient descent for linear regression."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights
    
    # Store costs and weights for visualization
    costs = []
    weights_history = [w.copy()]
    
    for i in range(max_iterations):
        # Compute predictions
        y_pred = X @ w
        
        # Compute error
        error = y_pred - y
        
        # Compute gradient
        gradient = 2 * X.T @ error
        
        # Update weights
        w_new = w - alpha * gradient
        
        # Store weights
        weights_history.append(w_new.copy())
        
        # Compute cost
        cost = np.sum(error**2)
        costs.append(cost)
        
        # Check for convergence
        if i > 0 and abs(costs[i] - costs[i-1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        # Update weights
        w = w_new
    
    return w, costs, weights_history

# Generate data
n_samples = 100
X, y, w_true = generate_data(n=n_samples)
print(f"Generated synthetic data with {n_samples} samples and {X.shape[1]} features (including intercept)")
print(f"True weights: {w_true}")
print()

# Part 3: Learning rate bound
print("## Part 3: Learning Rate Bound")
print("For gradient descent to converge, the learning rate α must satisfy:")
print("$$0 < \\alpha < \\frac{2}{\\lambda_{\\max}}$$")
print("where $\\lambda_{\\max}$ is the largest eigenvalue of the matrix $2\\boldsymbol{X}^T\\boldsymbol{X}$.")
print()
print("Deriving this bound:")
print("1. The gradient descent update is: $\\boldsymbol{w}_{t+1} = \\boldsymbol{w}_t - \\alpha\\nabla J(\\boldsymbol{w}_t)$")
print("2. For our cost function: $\\nabla J(\\boldsymbol{w}_t) = 2\\boldsymbol{X}^T(\\boldsymbol{X}\\boldsymbol{w}_t - \\boldsymbol{y})$")
print("3. To ensure convergence, the spectral radius of the iteration matrix must be less than 1")
print("4. The iteration matrix is $\\boldsymbol{I} - 2\\alpha\\boldsymbol{X}^T\\boldsymbol{X}$")
print("5. For the spectral radius to be less than 1, we need: $|1 - 2\\alpha\\lambda_i| < 1$ for all eigenvalues $\\lambda_i$ of $\\boldsymbol{X}^T\\boldsymbol{X}$")
print("6. This gives us: $-1 < 1 - 2\\alpha\\lambda_i < 1$")
print("7. The right inequality gives: $\\alpha < \\frac{1}{\\lambda_i}$")
print("8. The left inequality gives: $\\alpha > \\frac{1}{2\\lambda_i}$")
print("9. Since $\\lambda_i > 0$ for positive definite $\\boldsymbol{X}^T\\boldsymbol{X}$, we need $\\alpha < \\frac{1}{\\lambda_{\\max}}$")
print("10. For the specific gradient formulation with the factor of 2, this becomes $\\alpha < \\frac{1}{2\\lambda_{\\max}}$")
print()

# Calculate the bound for our dataset
XTX = X.T @ X
eigenvalues = eigvals(XTX)
max_eigenvalue = np.max(eigenvalues)
learning_rate_bound = 1 / max_eigenvalue

print(f"For our dataset:")
print(f"Eigenvalues of X^TX: {eigenvalues}")
print(f"Maximum eigenvalue (λ_max): {max_eigenvalue:.4f}")
print(f"Learning rate bound (1/λ_max): {learning_rate_bound:.4f}")
print(f"Learning rate should be less than: {learning_rate_bound:.4f} to ensure convergence")
print()

# Part 4: Effect of different learning rates
print("## Part 4: Effects of Different Learning Rates")

# Run gradient descent with different learning rates
alphas = [0.001, 0.01, learning_rate_bound * 0.5, learning_rate_bound * 0.9, learning_rate_bound * 1.1, learning_rate_bound * 1.5]
alpha_results = []

for alpha in alphas:
    w_final, costs, _ = batch_gradient_descent(X, y, alpha=alpha, max_iterations=100)
    alpha_results.append((alpha, w_final, costs))
    print(f"Learning rate α = {alpha:.6f}:")
    print(f"  Final weights: {w_final}")
    print(f"  Final cost: {costs[-1]:.6f}")
    print(f"  Iterations: {len(costs)}")
    print()

# Create a plot for cost convergence with different learning rates
plt.figure(figsize=(10, 6))
for alpha, _, costs in alpha_results:
    if alpha <= learning_rate_bound:
        linestyle = '-'
        label = f"α = {alpha:.6f} (stable)"
    else:
        linestyle = '--'
        label = f"α = {alpha:.6f} (unstable)"
    plt.plot(costs, linestyle=linestyle, label=label)

plt.title('Cost vs Iterations for Different Learning Rates')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_rate_comparison.png"), dpi=300)

# Create a more detailed visualization of convergence/divergence
plt.figure(figsize=(12, 8))

# Stable rates (zoom in)
plt.subplot(2, 2, 1)
for alpha, _, costs in alpha_results:
    if alpha <= learning_rate_bound:
        plt.plot(costs[:30], label=f"α = {alpha:.6f}")
plt.title('Convergence with Stable Learning Rates (First 30 iterations)')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Unstable rates (show divergence)
plt.subplot(2, 2, 2)
for alpha, _, costs in alpha_results:
    if alpha > learning_rate_bound:
        plt.plot(costs[:30], label=f"α = {alpha:.6f}")
plt.title('Divergence with Unstable Learning Rates (First 30 iterations)')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Compare final weights with true weights
plt.subplot(2, 2, 3)
width = 0.15
x = np.arange(len(w_true))
plt.bar(x - 0.3, w_true, width, label='True weights')

for i, (alpha, w_final, _) in enumerate(alpha_results[:4]):  # Only show stable rates
    plt.bar(x - 0.3 + (i+1)*width, w_final, width, label=f"α = {alpha:.6f}")

plt.title('Final Weights for Stable Learning Rates')
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.xticks(x, [f'w{i}' for i in range(len(w_true))])
plt.legend()
plt.grid(True)

# Visualization of "too small" vs "too large" learning rates
plt.subplot(2, 2, 4)
plt.plot(alpha_results[0][2][:50], label=f"Too small (α = {alphas[0]:.6f})")
plt.plot(alpha_results[2][2][:50], label=f"Good (α = {alphas[2]:.6f})")
plt.plot(alpha_results[-1][2][:20], label=f"Too large (α = {alphas[-1]:.6f})")
plt.title('Effects of Learning Rate Choice')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_rate_effects.png"), dpi=300)

print("Too small learning rate: Convergence is slow, requiring many iterations to reach the optimum.")
print("Too large learning rate: The algorithm diverges, with the cost increasing instead of decreasing.")
print("Optimal learning rate: Fast convergence without divergence.")
print()

# Part 5: Learning rate scheduling
print("## Part 5: Learning Rate Scheduling")
print("Learning rate scheduling involves changing the learning rate during training.")
print("Common strategies include:")
print("1. Step decay: Reduce the learning rate by a factor after a certain number of iterations")
print("2. Exponential decay: α_t = α_0 * exp(-kt) where k is the decay rate")
print("3. 1/t decay: α_t = α_0 / (1 + kt) where k is the decay rate")
print()
print("Let's implement a simple step decay strategy.")
print()

# Implement batch gradient descent with step decay
def batch_gd_with_scheduling(X, y, alpha_init=0.01, decay_rate=0.5, decay_steps=20, max_iterations=1000, tolerance=1e-6):
    """Implement batch gradient descent with step decay learning rate scheduling."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights
    
    # Store costs, weights, and learning rates for visualization
    costs = []
    alphas = []
    weights_history = [w.copy()]
    
    for i in range(max_iterations):
        # Update learning rate
        alpha = alpha_init * (decay_rate ** (i // decay_steps))
        alphas.append(alpha)
        
        # Compute predictions
        y_pred = X @ w
        
        # Compute error
        error = y_pred - y
        
        # Compute gradient
        gradient = 2 * X.T @ error
        
        # Update weights
        w_new = w - alpha * gradient
        
        # Store weights
        weights_history.append(w_new.copy())
        
        # Compute cost
        cost = np.sum(error**2)
        costs.append(cost)
        
        # Check for convergence
        if i > 0 and abs(costs[i] - costs[i-1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        # Update weights
        w = w_new
    
    return w, costs, weights_history, alphas

# Run gradient descent with learning rate scheduling
initial_alpha = learning_rate_bound * 0.5
w_scheduled, costs_scheduled, _, alphas_scheduled = batch_gd_with_scheduling(
    X, y, alpha_init=initial_alpha, decay_rate=0.5, decay_steps=20, max_iterations=200
)

print(f"Scheduled learning rate:")
print(f"  Initial learning rate: {initial_alpha:.6f}")
print(f"  Final weights: {w_scheduled}")
print(f"  Final cost: {costs_scheduled[-1]:.6f}")
print(f"  Iterations: {len(costs_scheduled)}")
print()

# Compare with constant learning rate
w_constant, costs_constant, _ = batch_gradient_descent(X, y, alpha=initial_alpha, max_iterations=200)

print(f"Constant learning rate (α = {initial_alpha:.6f}):")
print(f"  Final weights: {w_constant}")
print(f"  Final cost: {costs_constant[-1]:.6f}")
print(f"  Iterations: {len(costs_constant)}")
print()

# Create a plot comparing scheduling vs constant learning rate
plt.figure(figsize=(12, 6))

# Plot 1: Cost comparison
plt.subplot(1, 2, 1)
plt.plot(costs_scheduled, label="Scheduled learning rate")
plt.plot(costs_constant, label="Constant learning rate")
plt.title('Cost vs Iterations: Scheduled vs Constant Learning Rate')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Plot 2: Learning rate schedule
plt.subplot(1, 2, 2)
plt.plot(alphas_scheduled)
plt.title('Learning Rate Schedule (Step Decay)')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_rate_scheduling.png"), dpi=300)

# Create a more comprehensive visualization
plt.figure(figsize=(16, 8))

# Plot 1: Cost comparison (normal scale)
plt.subplot(2, 2, 1)
plt.plot(costs_scheduled, label="Scheduled α")
plt.plot(costs_constant, label="Constant α")
plt.title('Cost vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.legend()
plt.grid(True)

# Plot 2: Cost comparison (log scale)
plt.subplot(2, 2, 2)
plt.plot(costs_scheduled, label="Scheduled α")
plt.plot(costs_constant, label="Constant α")
plt.title('Cost vs Iterations (Log Scale)')
plt.xlabel('Iterations')
plt.ylabel('Cost (SSE)')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Plot 3: Learning rate schedule
plt.subplot(2, 2, 3)
plt.plot(alphas_scheduled)
plt.title('Learning Rate Schedule (Step Decay)')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.grid(True)

# Plot 4: Final weights comparison
plt.subplot(2, 2, 4)
width = 0.2
x = np.arange(len(w_true))
plt.bar(x - 0.2, w_true, width, label='True weights')
plt.bar(x, w_scheduled, width, label='Scheduled α')
plt.bar(x + 0.2, w_constant, width, label='Constant α')
plt.title('Final Weight Comparison')
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.xticks(x, [f'w{i}' for i in range(len(w_true))])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "scheduling_comprehensive.png"), dpi=300)

print("Benefits of learning rate scheduling:")
print("1. Faster initial convergence due to larger learning rate at the beginning")
print("2. Fine-grained optimization near the optimum with smaller learning rates")
print("3. Better ability to escape shallow local minima (for non-convex problems)")
print("4. Less sensitive to the initial learning rate choice")
print()

# Summary
print("## Summary of Findings")
print("1. The gradient of the SSE cost function with respect to w is: 2X^T(Xw - y)")
print("2. Batch gradient descent update rule: w_{t+1} = w_t - α·2X^T(Xw_t - y)")
print("3. The learning rate bound for convergence is: α < 1/λ_max(X^TX)")
print("4. Effects of learning rate choice:")
print("   - Too small: Slow convergence")
print("   - Too large: Divergence")
print("   - Optimal: Fast and stable convergence")
print("5. Learning rate scheduling (e.g., step decay) can improve convergence by:")
print("   - Using larger steps initially for faster progress")
print("   - Using smaller steps later for fine-tuning")
print()

print("Generated visualizations saved to:", save_dir) 