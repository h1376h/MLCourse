import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Step 1: Define the problem parameters
mu0 = np.array([0, 0])  # Mean for class 0
mu1 = np.array([2, 2])  # Mean for class 1

Sigma0 = np.array([[2, 1], [1, 2]])  # Covariance matrix for class 0
Sigma1 = np.array([[2, 1], [1, 2]])  # Covariance matrix for class 1 (same as class 0)

prior0 = 0.7  # Prior probability for class 0
prior1 = 0.3  # Prior probability for class 1

# Step 2: Define helper functions for Gaussian PDF
def gaussian_pdf(x, mu, Sigma):
    """
    Compute multivariate Gaussian PDF value at point x.
    
    Args:
        x: The point at which to evaluate the PDF
        mu: Mean vector of the Gaussian distribution
        Sigma: Covariance matrix of the Gaussian distribution
        
    Returns:
        The probability density at point x
    """
    n = len(mu)
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    
    # Calculate the normalization factor
    factor = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
    
    # Calculate the exponent term
    exponent = -0.5 * np.dot(np.dot((x - mu).T, inv), (x - mu))
    
    return factor * np.exp(exponent)

# Step 3: Create a grid of points for visualization
x = np.linspace(-4, 6, 300)
y = np.linspace(-4, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack((X, Y), axis=2)  # Stack all (x,y) pairs

# Step 4: Compute probability density functions for both classes
Z0 = np.zeros_like(X)
Z1 = np.zeros_like(X)

# Calculate PDF values at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z0[i, j] = gaussian_pdf(point, mu0, Sigma0)
        Z1[i, j] = gaussian_pdf(point, mu1, Sigma1)

# Step 5: Compute posterior probabilities
posterior0 = prior0 * Z0 / (prior0 * Z0 + prior1 * Z1)
posterior1 = prior1 * Z1 / (prior0 * Z0 + prior1 * Z1)

# Step 6: Derive the decision boundary
# For Gaussian distributions with same covariance matrix, the decision boundary is linear
# The boundary occurs where:
# (μ₁-μ₀)ᵀΣ⁻¹x - 0.5(μ₁ᵀΣ⁻¹μ₁ - μ₀ᵀΣ⁻¹μ₀) + log(P(y=1)/P(y=0)) = 0

# Compute the inverse of the covariance matrix
Sigma_inv = np.linalg.inv(Sigma0)

# Compute the coefficients for the decision boundary
w = np.dot(Sigma_inv, (mu1 - mu0))
b = -0.5 * (np.dot(np.dot(mu1, Sigma_inv), mu1) - np.dot(np.dot(mu0, Sigma_inv), mu0)) + np.log(prior1 / prior0)

# The decision boundary is the line w[0]*x + w[1]*y + b = 0
# We can solve for y in terms of x: y = -(w[0]*x + b) / w[1]
def decision_boundary(x, w, b):
    return -(w[0] * x + b) / w[1]

# Step 7: Create Figure 1 - Contours and Decision Boundary
plt.figure(figsize=(10, 8))

# Plot contours for p(x|y=0)
contour0 = plt.contour(X, Y, Z0, levels=5, colors='blue', alpha=0.7, linestyles='solid')
plt.clabel(contour0, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=1)
contour1 = plt.contour(X, Y, Z1, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

# Plot the decision boundary with unequal priors
x_vals = np.linspace(-4, 6, 100)
y_vals_unequal = decision_boundary(x_vals, w, b)
plt.plot(x_vals, y_vals_unequal, 'k-', linewidth=2, label='Decision Boundary (Unequal Priors)')

# Let's also compute the decision boundary with equal priors for comparison
b_equal = -0.5 * (np.dot(np.dot(mu1, Sigma_inv), mu1) - np.dot(np.dot(mu0, Sigma_inv), mu0))
y_vals_equal = decision_boundary(x_vals, w, b_equal)
plt.plot(x_vals, y_vals_equal, 'k--', linewidth=2, label='Decision Boundary (Equal Priors)')

# Add ellipses to represent the covariance matrices
def add_covariance_ellipse(mean, cov, color, label):
    """
    Add an ellipse representing the covariance matrix to the current plot.
    
    Args:
        mean: Mean vector of the Gaussian distribution
        cov: Covariance matrix of the Gaussian distribution
        color: Color for the ellipse
        label: Label for the legend
    """
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eigh(cov)
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    
    # Create the ellipse patch (2 standard deviations)
    ellipse = Ellipse(xy=mean, width=2*2*np.sqrt(evals[0]), height=2*2*np.sqrt(evals[1]),
                      angle=angle, edgecolor=color, facecolor='none', lw=2, label=label)
    plt.gca().add_patch(ellipse)

add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region')

# Add mean points
plt.scatter(mu0[0], mu0[1], color='blue', s=100, marker='o', label=r'$\mu_0 = [0, 0]$')
plt.scatter(mu1[0], mu1[1], color='red', s=100, marker='o', label=r'$\mu_1 = [2, 2]$')

# Add the test point at (1,1)
plt.scatter(1, 1, color='green', s=100, marker='*', label='Test Point (1,1)')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gaussian Contours and Decision Boundary')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend(loc='upper left')

# Save the figure
plt.savefig(os.path.join(save_dir, 'gaussian_contours_decision_boundary.png'), dpi=300, bbox_inches='tight')

# Step 8: Create Figure 2 - Decision Regions
plt.figure(figsize=(10, 8))

# Calculate the classified regions
decision_regions = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        # Decision rule based on posterior probabilities
        if posterior1[i, j] > posterior0[i, j]:
            decision_regions[i, j] = 1  # Class 1
        else:
            decision_regions[i, j] = 0  # Class 0

# Plot decision regions - 'skyblue' is for class 0, 'salmon' is for class 1
# Use pcolormesh instead of contourf for better control over the mapping
plt.pcolormesh(X, Y, decision_regions, cmap=plt.cm.colors.ListedColormap(['skyblue', 'salmon']), alpha=0.3, shading='auto')

# Plot the decision boundaries
plt.plot(x_vals, y_vals_unequal, 'k-', linewidth=2, label='Decision Boundary (Unequal Priors)')
plt.plot(x_vals, y_vals_equal, 'k--', linewidth=2, label='Decision Boundary (Equal Priors)')

# Add ellipses to represent the covariance matrices
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region')

# Add mean points
plt.scatter(mu0[0], mu0[1], color='blue', s=100, marker='o', label=r'$\mu_0 = [0, 0]$')
plt.scatter(mu1[0], mu1[1], color='red', s=100, marker='o', label=r'$\mu_1 = [2, 2]$')

# Add the test point at (1,1)
plt.scatter(1, 1, color='green', s=100, marker='*', label='Test Point (1,1)')

# Set axis limits to ensure the entire region is visible
plt.xlim(-4, 6)
plt.ylim(-4, 6)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend(loc='upper left')

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_regions.png'), dpi=300, bbox_inches='tight')

# Step 9: Create Figure 3 - Posterior probability heatmap
plt.figure(figsize=(10, 8))

# Create a heatmap of the posterior probability for class 1
# Use pcolormesh instead of imshow for better mapping to coordinates
plt.pcolormesh(X, Y, posterior1, cmap='coolwarm', vmin=0, vmax=1, shading='auto')
plt.colorbar(label='$P(y=1|x)$')

# Plot the decision boundaries
plt.plot(x_vals, y_vals_unequal, 'k-', linewidth=2, label='Decision Boundary (Unequal Priors)')
plt.plot(x_vals, y_vals_equal, 'k--', linewidth=2, label='Decision Boundary (Equal Priors)')

# Add mean points
plt.scatter(mu0[0], mu0[1], color='blue', s=100, marker='o', label=r'$\mu_0 = [0, 0]$')
plt.scatter(mu1[0], mu1[1], color='red', s=100, marker='o', label=r'$\mu_1 = [2, 2]$')

# Add the test point at (1,1)
plt.scatter(1, 1, color='green', s=100, marker='*', label='Test Point (1,1)')

# Set axis limits to ensure the entire region is visible
plt.xlim(-4, 6)
plt.ylim(-4, 6)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Posterior Probability Heatmap for Class 1')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(False)
plt.legend(loc='upper left')

# Save the figure
plt.savefig(os.path.join(save_dir, 'posterior_probability_heatmap.png'), dpi=300, bbox_inches='tight')

# Step 10: Evaluate the classification of the test point (1,1)
test_point = np.array([1, 1])

# Compute the discriminant function value for the test point
discriminant_value = np.dot(w, test_point) + b

# Compute PDFs directly for the test point
pdf_class0 = gaussian_pdf(test_point, mu0, Sigma0)
pdf_class1 = gaussian_pdf(test_point, mu1, Sigma1)

# Compute posterior probabilities directly
posterior_class0 = prior0 * pdf_class0 / (prior0 * pdf_class0 + prior1 * pdf_class1)
posterior_class1 = prior1 * pdf_class1 / (prior0 * pdf_class0 + prior1 * pdf_class1)

# Determine the predicted class
if discriminant_value > 0:
    predicted_class = 1
else:
    predicted_class = 0

# Print results
print("\nClassification Results for Test Point (1,1):")
print("---------------------------------------------")
print(f"Discriminant value: {discriminant_value:.4f}")
print(f"Posterior probability P(y=0|x): {posterior_class0:.4f}")
print(f"Posterior probability P(y=1|x): {posterior_class1:.4f}")
print(f"Predicted class: {predicted_class}")
print(f"Decision boundary equation: {w[0]:.4f}*x₁ + {w[1]:.4f}*x₂ + {b:.4f} = 0")

# Step 11: Mathematical derivation (output to console)
print("\nMathematical Derivation of Decision Boundary:")
print("---------------------------------------------")
print("For binary classification with Gaussian distributions with same covariance:")
print("The decision boundary follows the rule that assigns point x to class 1 if:")
print("P(y=1|x) > P(y=0|x)")

print("\nUsing Bayes' rule and taking the logarithm:")
print("log[p(x|y=1)] + log[P(y=1)] > log[p(x|y=0)] + log[P(y=0)]")

print("\nFor Gaussian distributions with same covariance matrix Σ, this simplifies to:")
print("(μ₁-μ₀)ᵀΣ⁻¹x - 0.5(μ₁ᵀΣ⁻¹μ₁ - μ₀ᵀΣ⁻¹μ₀) + log[P(y=1)/P(y=0)] > 0")

print(f"\nFor our problem with μ₀ = [0,0], μ₁ = [2,2], and Σ = [[2,1],[1,2]]:")
print(f"Σ⁻¹ = \n{Sigma_inv}")
print(f"(μ₁-μ₀)ᵀΣ⁻¹ = {w}")
print(f"0.5(μ₁ᵀΣ⁻¹μ₁ - μ₀ᵀΣ⁻¹μ₀) = {-b + np.log(prior1/prior0):.4f}")
print(f"log[P(y=1)/P(y=0)] = {np.log(prior1/prior0):.4f}")

print(f"\nThe decision boundary is the line: {w[0]:.4f}*x₁ + {w[1]:.4f}*x₂ + {b:.4f} = 0")
print(f"Which can be rewritten as: x₂ = {-w[0]/w[1]:.4f}*x₁ - {b/w[1]:.4f}")

print("\nFor equal priors P(y=0) = P(y=1) = 0.5, the decision boundary would be:")
print(f"x₂ = {-w[0]/w[1]:.4f}*x₁ - {b_equal/w[1]:.4f}")

print("\nThe effect of the unequal priors is to shift the decision boundary by:")
print(f"Shift = {(b - b_equal)/w[1]:.4f} units in the x₂ direction")

print("\nFor the test point (1,1):")
if predicted_class == 1:
    print("The test point is classified as Class 1")
else:
    print("The test point is classified as Class 0")

print("\nVisualization saved to:", save_dir) 