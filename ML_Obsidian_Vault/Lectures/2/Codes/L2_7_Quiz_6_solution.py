import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Logistic regression model with log-posterior:")
print("log p(w|D) = sum_i [y_i log(σ(w^T x_i)) + (1-y_i)log(1-σ(w^T x_i))] - λ/2||w||^2 + C")
print("- σ(z) = 1/(1+e^(-z)) is the sigmoid function")
print("- w are the model parameters")
print("- C is a constant")
print("\nTasks:")
print("1. Identify the prior distribution on w implied by this log-posterior")
print("2. If we have a single data point with x = [1, 2]^T and y = 1, write the gradient")
print("   ∇_w log p(w|D) for w = [0, 0]^T")
print("3. Describe one optimization technique suitable for finding the MAP estimate")

# Step 2: Identifying the Prior Distribution
print_step_header(2, "Identifying the Prior Distribution")

print("The log-posterior consists of two main parts:")
print("1. log-likelihood term: sum_i [y_i log(σ(w^T x_i)) + (1-y_i)log(1-σ(w^T x_i))]")
print("2. Prior term: -λ/2||w||^2 + C")
print("\nThe prior term corresponds to the log of the prior probability p(w).")
print("We have log p(w) = -λ/2||w||^2 + C'")
print("\nExponentiating both sides:")
print("p(w) = exp(-λ/2||w||^2 + C')")
print("p(w) ∝ exp(-λ/2||w||^2)")
print("\nThis is proportional to a multivariate Gaussian distribution with:")
print("- Mean μ = 0 (since the quadratic form is centered at origin)")
print("- Covariance Σ = (1/λ)I (where I is the identity matrix)")
print("\nTherefore, the prior distribution is:")
print("w ~ N(0, (1/λ)I)")
print("\nThis is an isotropic Gaussian prior where the variance for each component is 1/λ.")
print("The parameter λ controls the strength of regularization; larger λ means stronger")
print("regularization, pushing the weights closer to zero.")

# Create a visualization of the prior distribution for a 2D weight vector
lambda_val = 1.0  # For visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Calculate prior density for each point in the 2D grid
for i in range(len(x)):
    for j in range(len(y)):
        w = np.array([X[i, j], Y[i, j]])
        Z[i, j] = np.exp(-lambda_val/2 * np.sum(w**2))

# Plot the 2D Gaussian prior
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour, label='Prior density p(w)')
plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.title(f'Prior Distribution: N(0, {1/lambda_val}I)', fontsize=16)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_distribution_2d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Create a 3D plot of the prior
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)

ax.set_xlabel('w₁', fontsize=14)
ax.set_ylabel('w₂', fontsize=14)
ax.set_zlabel('Prior density p(w)', fontsize=14)
ax.set_title(f'3D Visualization of the Gaussian Prior: N(0, {1/lambda_val}I)', fontsize=16)

# Save the 3D figure
file_path = os.path.join(save_dir, "prior_distribution_3d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Computing the Gradient
print_step_header(3, "Computing the Gradient")

print("We need to find the gradient of the log-posterior at w = [0, 0]^T for a single data point:")
print("x = [1, 2]^T and y = 1")
print("\nThe log-posterior for a single data point is:")
print("log p(w|D) = y log(σ(w^T x)) + (1-y)log(1-σ(w^T x)) - λ/2||w||^2 + C")
print("\nSince y = 1, this simplifies to:")
print("log p(w|D) = log(σ(w^T x)) - λ/2||w||^2 + C")
print("\nTo compute the gradient, we use the chain rule:")
print("∇_w log p(w|D) = ∇_w log(σ(w^T x)) - λw")
print("\nWe know that:")
print("∇_w log(σ(w^T x)) = (1 - σ(w^T x)) · x")
print("\nAt w = [0, 0]^T:")
print("w^T x = 0 · 1 + 0 · 2 = 0")
print("σ(0) = 1/(1+e^(-0)) = 1/2")
print("\nTherefore:")
print("∇_w log(σ(w^T x)) = (1 - 1/2) · [1, 2]^T = 0.5 · [1, 2]^T = [0.5, 1]^T")
print("∇_w (-λ/2||w||^2) = -λw = -λ · [0, 0]^T = [0, 0]^T")
print("\nThe complete gradient is:")
print("∇_w log p(w|D) = [0.5, 1]^T - λ · [0, 0]^T = [0.5, 1]^T")

# Compute and visualize the gradient
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_posterior(w, x, y, lambda_val):
    # Compute w^T x
    z = w.dot(x)
    # Compute log-likelihood term
    ll = y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))
    # Compute log-prior term
    lp = -lambda_val/2 * np.sum(w**2)
    return ll + lp

def gradient_log_posterior(w, x, y, lambda_val):
    # Compute sigmoid(w^T x)
    z = w.dot(x)
    sig_z = sigmoid(z)
    # Compute gradient of log-likelihood term
    dll = (y - sig_z) * x
    # Compute gradient of log-prior term
    dlp = -lambda_val * w
    return dll + dlp

# Data point
x = np.array([1, 2])
y = 1
lambda_val = 1.0

# Calculate gradient at w = [0, 0]
w_eval = np.array([0, 0])
grad = gradient_log_posterior(w_eval, x, y, lambda_val)
print(f"\nComputed gradient at w = [0, 0]^T: [{grad[0]}, {grad[1]}]^T")

# Visualize the gradient
# Create a grid of w values
w1_range = np.linspace(-2, 2, 20)
w2_range = np.linspace(-2, 2, 20)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Calculate gradients at each point
U = np.zeros_like(W1)
V = np.zeros_like(W2)
LP = np.zeros_like(W1)

for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        w = np.array([W1[i, j], W2[i, j]])
        grad = gradient_log_posterior(w, x, y, lambda_val)
        U[i, j] = grad[0]
        V[i, j] = grad[1]
        LP[i, j] = log_posterior(w, x, y, lambda_val)

# Plot the gradient field
plt.figure(figsize=(12, 10))
plt.contourf(W1, W2, LP, 20, cmap='viridis', alpha=0.7)
plt.colorbar(label='Log-posterior value')
plt.quiver(W1, W2, U, V, scale=30, color='white', alpha=0.8)
plt.plot(0, 0, 'ro', markersize=10, label=f'w = [0, 0]^T, gradient = [{grad[0]:.1f}, {grad[1]:.1f}]^T')

# Mark the MAP estimate (which should be in the direction of the gradient)
plt.arrow(0, 0, grad[0], grad[1], color='red', width=0.02, head_width=0.1, 
          head_length=0.1, alpha=0.7, label='Gradient direction')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.title('Gradient of Log-Posterior at w = [0, 0]', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "gradient_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Optimization Techniques for MAP Estimation
print_step_header(4, "Optimization Techniques for MAP Estimation")

print("For finding the MAP estimate in logistic regression, several optimization techniques are suitable:")
print("\n1. Gradient Ascent:")
print("   - Update rule: w_{t+1} = w_t + η ∇_w log p(w|D)")
print("   - Simple but may converge slowly; needs careful learning rate tuning")
print("\n2. Newton's Method:")
print("   - Update rule: w_{t+1} = w_t - [H(w_t)]^(-1) ∇_w log p(w|D)")
print("   - H(w) is the Hessian matrix of second derivatives")
print("   - Faster convergence near the optimum but computationally expensive for high dimensions")
print("\n3. L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):")
print("   - Approximates the inverse Hessian matrix using past gradients")
print("   - Balances computational efficiency and convergence speed")
print("   - Particularly suitable for high-dimensional problems")
print("\n4. Stochastic Gradient Descent (SGD):")
print("   - Uses a subset of data (mini-batch) to approximate the gradient")
print("   - Efficient for large datasets")
print("   - Often combined with momentum, adaptive learning rates, etc.")
print("\nFor our specific problem with a Gaussian prior:")
print("- The log-posterior is concave, guaranteeing a unique maximum")
print("- The regularization term (-λ/2||w||^2) ensures numerical stability")
print("- For low-dimensional problems, Newton's method would converge quickly")
print("- For high-dimensional problems, L-BFGS would be more efficient")

# Visualize the convergence of gradient ascent for MAP estimation
def gradient_ascent(x, y, lambda_val, learning_rate=0.1, num_iterations=100):
    w = np.array([0, 0])  # Start from [0, 0]
    trajectory = [w.copy()]
    
    for _ in range(num_iterations):
        grad = gradient_log_posterior(w, x, y, lambda_val)
        w = w + learning_rate * grad
        trajectory.append(w.copy())
    
    return np.array(trajectory)

# Run gradient ascent
trajectory = gradient_ascent(x, y, lambda_val, learning_rate=0.1, num_iterations=15)

# Create a finer grid for better visualization
w1_fine = np.linspace(-2, 2, 100)
w2_fine = np.linspace(-2, 2, 100)
W1_fine, W2_fine = np.meshgrid(w1_fine, w2_fine)
LP_fine = np.zeros_like(W1_fine)

for i in range(len(w1_fine)):
    for j in range(len(w2_fine)):
        w = np.array([W1_fine[i, j], W2_fine[i, j]])
        LP_fine[i, j] = log_posterior(w, x, y, lambda_val)

# Plot the log-posterior and gradient ascent trajectory
plt.figure(figsize=(12, 10))
contour = plt.contourf(W1_fine, W2_fine, LP_fine, 30, cmap='viridis')
plt.colorbar(contour, label='Log-posterior value')

# Plot the trajectory points
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=8, alpha=0.7, 
         label='Gradient ascent trajectory')
plt.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=10, label='Starting point')
plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=10, label='Final point (MAP estimate)')

plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.title('Gradient Ascent for MAP Estimation', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "gradient_ascent_trajectory.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Summarize the final MAP estimate
map_estimate = trajectory[-1]
print(f"\nMAP estimate after gradient ascent: w = [{map_estimate[0]:.4f}, {map_estimate[1]:.4f}]^T")
print(f"Log-posterior value at MAP estimate: {log_posterior(map_estimate, x, y, lambda_val):.4f}")

# Step 5: Mathematical Derivation Details
print_step_header(5, "Mathematical Derivation Details")

print("For a single data point with x = [1, 2]^T and y = 1:")
print("\n1. The gradient of the log-posterior at w = [0, 0]^T is:")
print("   ∇_w log p(w|D) = [0.5, 1]^T")
print("\n2. This gradient points in the direction of steepest ascent of the log-posterior")
print("   and indicates the initial direction for optimization algorithms.")
print("\n3. The MAP estimate, found through gradient ascent, is approximately:")
print(f"   w_MAP ≈ [{map_estimate[0]:.4f}, {map_estimate[1]:.4f}]^T")
print("\nThis confirms our derived gradient and illustrates the optimization process.") 