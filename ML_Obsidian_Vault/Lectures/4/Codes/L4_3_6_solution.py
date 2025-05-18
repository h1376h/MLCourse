import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters
# Step 1: Define the mean vectors and covariance matrix for both classes
mu0 = np.array([1, 1])  # Mean for class 0
mu1 = np.array([3, 3])  # Mean for class 1

Sigma = np.array([[1, 0], [0, 1]])  # Same covariance matrix for both classes

prior0 = 0.5  # Prior probability for class 0
prior1 = 0.5  # Prior probability for class 1

# Step 2: Mathematical derivation of the posterior probability
# We'll implement a function to calculate the posterior probability analytically
def calculate_posterior(x, mu0, mu1, Sigma, prior0, prior1):
    """
    Calculate the posterior probability P(y=1|x) for Gaussian class-conditional densities
    with different means but the same covariance matrix.
    
    Args:
        x: The point at which to evaluate the posterior
        mu0: Mean vector for class 0
        mu1: Mean vector for class 1
        Sigma: Common covariance matrix
        prior0: Prior probability for class 0
        prior1: Prior probability for class 1
        
    Returns:
        The posterior probability P(y=1|x)
    """
    # Inverse of covariance matrix
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Compute the discriminant function based on the derived formula
    w = np.dot(Sigma_inv, mu1 - mu0)
    b = -0.5 * np.dot(np.dot(mu1.T, Sigma_inv), mu1) + 0.5 * np.dot(np.dot(mu0.T, Sigma_inv), mu0) + np.log(prior1/prior0)
    
    # The discriminant function a(x) = w^T x + b
    a_x = np.dot(w, x) + b
    
    # Convert to probability using the sigmoid function
    posterior = 1 / (1 + np.exp(-a_x))
    
    return posterior

# Step 3: Create a grid of points for visualization
# Create a grid of x and y values for plotting
x = np.linspace(-2, 6, 300)
y = np.linspace(-2, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack((X, Y), axis=2)  # Stack all (x,y) pairs

# Step 4: Compute probability density functions for both classes
rv0 = multivariate_normal(mu0, Sigma)
rv1 = multivariate_normal(mu1, Sigma)

Z0 = rv0.pdf(pos)
Z1 = rv1.pdf(pos)

# Step 5: Compute posterior probabilities
posterior1 = np.zeros_like(X)

# Calculate posterior probabilities at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        posterior1[i, j] = calculate_posterior(point, mu0, mu1, Sigma, prior0, prior1)

posterior0 = 1 - posterior1  # Since it's binary classification, P(y=0|x) = 1 - P(y=1|x)

# Step 6: Calculate the decision boundary
# The decision boundary occurs where P(y=1|x) = P(y=0|x) = 0.5
# This is equivalent to where the discriminant function a(x) = 0

# For this specific problem:
# The decision boundary is a line orthogonal to (mu1 - mu0)
# The equation is w^T x + b = 0, where w = Sigma^-1 (mu1 - mu0)
w = np.dot(np.linalg.inv(Sigma), mu1 - mu0)
b = -0.5 * np.dot(np.dot(mu1.T, np.linalg.inv(Sigma)), mu1) + 0.5 * np.dot(np.dot(mu0.T, np.linalg.inv(Sigma)), mu0)

# Calculate points on the decision boundary (solving for x2 given x1)
boundary_x1 = np.linspace(-2, 6, 100)
boundary_x2 = (-w[0] * boundary_x1 - b) / w[1]

# Step 7: Create Figure 1: Gaussian contours and decision boundary
plt.figure(figsize=(10, 8))

# Plot contours for p(x|y=0)
contour0 = plt.contour(X, Y, Z0, levels=5, colors='blue', alpha=0.7, linestyles='solid')
plt.clabel(contour0, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=1)
contour1 = plt.contour(X, Y, Z1, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

# Plot decision boundary
plt.plot(boundary_x1, boundary_x2, 'k--', label='Decision Boundary')

# Add class means
plt.scatter(mu0[0], mu0[1], color='blue', s=100, label=r'$\mu_0$')
plt.scatter(mu1[0], mu1[1], color='red', s=100, label=r'$\mu_1$')

# Add labels and legend
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Gaussian Contours and Decision Boundary')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'gaussian_contours_decision_boundary.png'), dpi=300, bbox_inches='tight')

# Step 8: Create Figure 2: Decision regions
plt.figure(figsize=(10, 8))

# Plot decision regions
plt.contourf(X, Y, posterior1, levels=[0, 0.5, 1], colors=['skyblue', 'salmon'], alpha=0.5)

# Plot contours for posterior probabilities
contour_posterior = plt.contour(X, Y, posterior1, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.7)
plt.clabel(contour_posterior, inline=True, fontsize=8, fmt='%.1f')

# Plot decision boundary
plt.plot(boundary_x1, boundary_x2, 'k--', linewidth=2, label='Decision Boundary')

# Add class means
plt.scatter(mu0[0], mu0[1], color='blue', s=100, label=r'$\mu_0$')
plt.scatter(mu1[0], mu1[1], color='red', s=100, label=r'$\mu_1$')

# Add labels and legend
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Decision Regions and Posterior Probability Contours')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_regions.png'), dpi=300, bbox_inches='tight')

# Step 9: Create Figure 3: 3D visualization of posterior probability
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Downsample for clearer visualization
step = 5
X_sparse = X[::step, ::step]
Y_sparse = Y[::step, ::step]
posterior1_sparse = posterior1[::step, ::step]

# Plot the posterior probability surface
surf = ax.plot_surface(X_sparse, Y_sparse, posterior1_sparse, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=r'$P(y=1|x)$')

# Mark the decision boundary (where posterior = 0.5)
ax.contour(X, Y, posterior1, levels=[0.5], colors='k', linestyles='dashed')

# Add class means
ax.scatter([mu0[0]], [mu0[1]], [0], color='blue', s=100, label=r'$\mu_0$')
ax.scatter([mu1[0]], [mu1[1]], [1], color='red', s=100, label=r'$\mu_1$')

# Add labels
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$P(y=1|x)$')
ax.set_title('3D Visualization of Posterior Probability $P(y=1|x)$')
ax.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'posterior_probability_3d.png'), dpi=300, bbox_inches='tight')

# Step 10: Create Figure 4: Sigmoid function visualization
plt.figure(figsize=(10, 6))

# Define the linear discriminant function
def linear_discriminant(x1, x2, w, b):
    return np.dot(w, np.array([x1, x2])) + b

# Create a line from mu0 to mu1
t = np.linspace(-1, 2, 300)  # Parameter that goes from before mu0 to after mu1
line_points_x1 = mu0[0] + t * (mu1[0] - mu0[0])
line_points_x2 = mu0[1] + t * (mu1[1] - mu0[1])

# Calculate the linear discriminant along this line
a_values = np.zeros_like(line_points_x1)
for i in range(len(line_points_x1)):
    point = np.array([line_points_x1[i], line_points_x2[i]])
    a_values[i] = linear_discriminant(point[0], point[1], w, b)

# Calculate the posterior probability along this line (sigmoid of a_values)
posterior_along_line = 1 / (1 + np.exp(-a_values))

# Plot the sigmoid transformation
plt.subplot(2, 1, 1)
plt.plot(t, a_values, 'k-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.grid(True)
plt.xlabel(r't (position along line from before $\mu_0$ to after $\mu_1$)')
plt.ylabel(r'Linear Discriminant $a(x)$')
plt.title(r'Linear Discriminant Function Along the Line Between $\mu_0$ and $\mu_1$')

plt.subplot(2, 1, 2)
plt.plot(t, posterior_along_line, 'k-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.grid(True)
plt.xlabel(r't (position along line from before $\mu_0$ to after $\mu_1$)')
plt.ylabel(r'$P(y=1|x)$')
plt.title('Sigmoid Transformation of Linear Discriminant (Posterior Probability)')

plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, 'sigmoid_transformation.png'), dpi=300, bbox_inches='tight')

# Step 11: NEW VISUALIZATION - Gradient field of the posterior probability
plt.figure(figsize=(10, 8))

# Create a coarser grid for the gradient field
step = 20
X_grad = X[::step, ::step]
Y_grad = Y[::step, ::step]

# Calculate gradient of posterior probability (proportional to w)
# Since w is constant in this case, the gradient is the same everywhere,
# but we'll scale its magnitude by the posterior probability to show the
# certainty of the classification
U = np.ones_like(X_grad) * w[0]
V = np.ones_like(Y_grad) * w[1]

# Get posterior probabilities for color mapping
posterior_grad = posterior1[::step, ::step]

# Adjust the certainty factor - further from 0.5 means more certain
certainty = 2 * np.abs(posterior_grad - 0.5)
U = U * certainty
V = V * certainty

# Create quiver plot
plt.quiver(X_grad, Y_grad, U, V, posterior_grad, cmap='coolwarm', 
           scale=30, width=0.0025, alpha=0.7)

# Plot decision boundary
plt.plot(boundary_x1, boundary_x2, 'k--', linewidth=2)

# Add class means
plt.scatter(mu0[0], mu0[1], color='blue', s=100)
plt.scatter(mu1[0], mu1[1], color='red', s=100)

# Add contours of posterior probability
plt.contour(X, Y, posterior1, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)

# Add labels
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Gradient Field of Posterior Probability')
plt.grid(True)
plt.colorbar(label=r'$P(y=1|x)$')

# Save the figure
plt.savefig(os.path.join(save_dir, 'gradient_field.png'), dpi=300, bbox_inches='tight')

# Step 12: Print the mathematical derivation
print("\nMathematical Derivation of Posterior Probability")
print("-" * 50)
print("Step 1: Using Bayes' rule, the posterior probability is given by:")
print("P(y=1|x) = P(x|y=1)P(y=1) / [P(x|y=0)P(y=0) + P(x|y=1)P(y=1)]")

print("\nStep 2: Substituting multivariate Gaussian PDFs:")
print("P(x|y=0) = N(x|μ₀, Σ)")
print("P(x|y=1) = N(x|μ₁, Σ)")

print("\nStep 3: For multivariate Gaussian, the PDF is:")
print("N(x|μ, Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-1/2 (x-μ)ᵀ Σ⁻¹ (x-μ))")

print("\nStep 4: Taking the ratio of P(x|y=1)P(y=1) to P(x|y=0)P(y=0):")
print("P(x|y=1)P(y=1) / P(x|y=0)P(y=0) = ")
print("exp(-1/2 (x-μ₁)ᵀ Σ⁻¹ (x-μ₁) + 1/2 (x-μ₀)ᵀ Σ⁻¹ (x-μ₀)) * (P(y=1)/P(y=0))")

print("\nStep 5: Expanding the quadratic terms:")
print("(x-μ₁)ᵀ Σ⁻¹ (x-μ₁) = xᵀ Σ⁻¹ x - xᵀ Σ⁻¹ μ₁ - μ₁ᵀ Σ⁻¹ x + μ₁ᵀ Σ⁻¹ μ₁")
print("(x-μ₀)ᵀ Σ⁻¹ (x-μ₀) = xᵀ Σ⁻¹ x - xᵀ Σ⁻¹ μ₀ - μ₀ᵀ Σ⁻¹ x + μ₀ᵀ Σ⁻¹ μ₀")

print("\nStep 6: Simplifying the exponent:")
print("-1/2 (x-μ₁)ᵀ Σ⁻¹ (x-μ₁) + 1/2 (x-μ₀)ᵀ Σ⁻¹ (x-μ₀) =")
print("xᵀ Σ⁻¹ (μ₁-μ₀) - 1/2 μ₁ᵀ Σ⁻¹ μ₁ + 1/2 μ₀ᵀ Σ⁻¹ μ₀")

print("\nStep 7: Defining the weight vector and bias term:")
print("w = Σ⁻¹ (μ₁-μ₀)")
print("b = -1/2 μ₁ᵀ Σ⁻¹ μ₁ + 1/2 μ₀ᵀ Σ⁻¹ μ₀ + ln(P(y=1)/P(y=0))")

print("\nStep 8: Express the ratio in terms of linear discriminant a(x):")
print("P(x|y=1)P(y=1) / P(x|y=0)P(y=0) = exp(a(x))")
print("where a(x) = wᵀx + b")

print("\nStep 9: Using logistic function to derive posterior:")
print("P(y=1|x) = exp(a(x)) / (1 + exp(a(x))) = 1 / (1 + exp(-a(x)))")

print("\nStep 10: This is the sigmoid function applied to a linear discriminant,")
print("confirming that the model is equivalent to logistic regression.")

print("\nFor the specific case in the problem:")
print(f"μ₀ = {mu0}")
print(f"μ₁ = {mu1}")
print(f"Σ = {Sigma}")
print(f"w = Σ⁻¹(μ₁-μ₀) = {w}")
print(f"b = {b}")

print("\nThe decision boundary is the line w₁x₁ + w₂x₂ + b = 0")
print(f"or {w[0]:.2f}x₁ + {w[1]:.2f}x₂ + {b:.2f} = 0")
print("This is a line perpendicular to the vector w, which is in the direction from μ₀ to μ₁")

# Show plots (optional)
# plt.show() 