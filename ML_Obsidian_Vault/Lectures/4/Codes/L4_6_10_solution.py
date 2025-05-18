import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy.stats import multivariate_normal

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Disable LaTeX to avoid font issues
plt.rcParams['text.usetex'] = False

# Step 1: Define the problem parameters
# Class means for the 3 classes
mu1 = np.array([1, 0])
mu2 = np.array([0, 2])
mu3 = np.array([2, 1])

# Shared covariance matrix (identity matrix)
Sigma = np.eye(2)  # 2x2 identity matrix

# Prior probabilities (assuming equal priors)
prior1 = prior2 = prior3 = 1/3

# Test point
x_test = np.array([1, 1])

# Step 2: Define the discriminant function for LDA
def lda_discriminant(x, mu, Sigma, prior):
    """
    LDA discriminant function: delta_i(x) = x^T * Sigma^-1 * mu_i - 0.5 * mu_i^T * Sigma^-1 * mu_i + log(pi_i)
    """
    inv_sigma = np.linalg.inv(Sigma)
    term1 = np.dot(np.dot(x, inv_sigma), mu)
    term2 = 0.5 * np.dot(np.dot(mu, inv_sigma), mu)
    term3 = np.log(prior)
    
    return term1 - term2 + term3

# Step 3: Compute discriminant scores for specific test point
delta1 = lda_discriminant(x_test, mu1, Sigma, prior1)
delta2 = lda_discriminant(x_test, mu2, Sigma, prior2)
delta3 = lda_discriminant(x_test, mu3, Sigma, prior3)

print("Discriminant scores for the point x = [1, 1]:")
print(f"delta_1(x) = {delta1:.4f}")
print(f"delta_2(x) = {delta2:.4f}")
print(f"delta_3(x) = {delta3:.4f}")

# Determine the predicted class
predicted_class = np.argmax([delta1, delta2, delta3]) + 1
print(f"Predicted class for x = [1, 1]: Class {predicted_class}")

# Step 4: Create a grid for visualizing the decision boundaries
x1_min, x1_max = -1, 3
x2_min, x2_max = -1, 3
grid_step = 0.02
x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max + grid_step, grid_step),
                               np.arange(x2_min, x2_max + grid_step, grid_step))
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Compute discriminant scores for each point in the grid
Z1 = np.array([lda_discriminant(x, mu1, Sigma, prior1) for x in X_grid])
Z2 = np.array([lda_discriminant(x, mu2, Sigma, prior2) for x in X_grid])
Z3 = np.array([lda_discriminant(x, mu3, Sigma, prior3) for x in X_grid])

# Combine scores to determine the predicted class for each point
Z = np.argmax(np.vstack((Z1, Z2, Z3)), axis=0) + 1
Z = Z.reshape(x1_grid.shape)

# Step 5: Create Figure 1 - Decision regions with class means and test point
plt.figure(figsize=(10, 8))
# Plot decision regions
plt.contourf(x1_grid, x2_grid, Z, levels=[0, 1, 2, 3], alpha=0.3, cmap=plt.cm.Set1)

# Plot the means of the classes
plt.scatter(mu1[0], mu1[1], c='r', marker='o', s=100, label='mu_1 = [1, 0]^T')
plt.scatter(mu2[0], mu2[1], c='g', marker='o', s=100, label='mu_2 = [0, 2]^T')
plt.scatter(mu3[0], mu3[1], c='b', marker='o', s=100, label='mu_3 = [2, 1]^T')

# Plot the test point
plt.scatter(x_test[0], x_test[1], c='black', marker='X', s=150, label='Test point x = [1, 1]^T')

# Add labels and legend
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('LDA Decision Regions for 3-Class Problem')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'lda_decision_regions.png'), dpi=300, bbox_inches='tight')

# Step 6: Create Figure 2 - Probability density contours for each class
plt.figure(figsize=(10, 8))

# Create a multivariate normal distribution for each class
mvn1 = multivariate_normal(mu1, Sigma)
mvn2 = multivariate_normal(mu2, Sigma)
mvn3 = multivariate_normal(mu3, Sigma)

# Evaluate density at each point in the grid
Z1_density = mvn1.pdf(X_grid).reshape(x1_grid.shape)
Z2_density = mvn2.pdf(X_grid).reshape(x1_grid.shape)
Z3_density = mvn3.pdf(X_grid).reshape(x1_grid.shape)

# Plot contours for each class density
plt.contour(x1_grid, x2_grid, Z1_density, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.contour(x1_grid, x2_grid, Z2_density, levels=5, colors='green', alpha=0.7, linestyles='solid')
plt.contour(x1_grid, x2_grid, Z3_density, levels=5, colors='blue', alpha=0.7, linestyles='solid')

# Plot decision boundaries
plt.contour(x1_grid, x2_grid, Z, levels=[0.5, 1.5, 2.5], colors='black', linestyles='dashed', linewidths=2)

# Plot class means
plt.scatter(mu1[0], mu1[1], c='red', marker='o', s=100, label='mu_1 = [1, 0]^T')
plt.scatter(mu2[0], mu2[1], c='green', marker='o', s=100, label='mu_2 = [0, 2]^T')
plt.scatter(mu3[0], mu3[1], c='blue', marker='o', s=100, label='mu_3 = [2, 1]^T')

# Plot the test point
plt.scatter(x_test[0], x_test[1], c='black', marker='X', s=150, label='Test point x = [1, 1]^T')

# Add labels and legend
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Class Probability Densities and Decision Boundaries')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'class_densities.png'), dpi=300, bbox_inches='tight')

# Step 7: Create Figure 3 - Discriminant function values visualization
plt.figure(figsize=(10, 8))

# Plot each discriminant function value as a colored surface for easier comparison
plt.contourf(x1_grid, x2_grid, Z, levels=[0, 1, 2, 3], alpha=0.3, cmap=plt.cm.Set1)

# Add contour lines for each discriminant function (without labels)
c1 = plt.contour(x1_grid, x2_grid, Z1.reshape(x1_grid.shape), levels=5, colors='red', alpha=0.7)
c2 = plt.contour(x1_grid, x2_grid, Z2.reshape(x1_grid.shape), levels=5, colors='green', alpha=0.7)
c3 = plt.contour(x1_grid, x2_grid, Z3.reshape(x1_grid.shape), levels=5, colors='blue', alpha=0.7)

# Mark the point where the discriminant functions are equal (decision boundaries)
plt.contour(x1_grid, x2_grid, Z, levels=[0.5, 1.5, 2.5], colors='black', linestyles='dashed', linewidths=2)

# Plot the test point
plt.scatter(x_test[0], x_test[1], c='black', marker='X', s=150, label='Test point x = [1, 1]^T')

# Plot class means
plt.scatter(mu1[0], mu1[1], c='red', marker='o', s=100, label='mu_1 = [1, 0]^T')
plt.scatter(mu2[0], mu2[1], c='green', marker='o', s=100, label='mu_2 = [0, 2]^T')
plt.scatter(mu3[0], mu3[1], c='blue', marker='o', s=100, label='mu_3 = [2, 1]^T')

# Add labels and legend
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('LDA Discriminant Function Values')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(os.path.join(save_dir, 'discriminant_functions.png'), dpi=300, bbox_inches='tight')

# Step 8: Detailed calculation for the discriminant functions
print("\nDetailed calculation of discriminant functions:")

# Since Sigma is identity, its inverse is also identity
print("\nThe shared covariance matrix Σ is the identity matrix:")
print(Sigma)
print("\nIts inverse Σ^-1 is also the identity matrix:")
print(np.linalg.inv(Sigma))

# For class 1
print("\nFor class 1:")
print(f"mu_1 = {mu1}")
print(f"x = {x_test}")
print(f"term1 = x^T * Σ^-1 * mu_1 = {x_test[0]}×{mu1[0]} + {x_test[1]}×{mu1[1]} = {np.dot(x_test, mu1)}")
print(f"term2 = 0.5 * mu_1^T * Σ^-1 * mu_1 = 0.5×({mu1[0]}²+{mu1[1]}²) = {0.5 * np.dot(mu1, mu1)}")
print(f"term3 = log(pi_1) = log({prior1}) = {np.log(prior1):.4f}")
print(f"delta_1(x) = term1 - term2 + term3 = {np.dot(x_test, mu1)} - {0.5 * np.dot(mu1, mu1)} + {np.log(prior1):.4f} = {delta1:.4f}")

# For class 2
print("\nFor class 2:")
print(f"mu_2 = {mu2}")
print(f"x = {x_test}")
print(f"term1 = x^T * Σ^-1 * mu_2 = {x_test[0]}×{mu2[0]} + {x_test[1]}×{mu2[1]} = {np.dot(x_test, mu2)}")
print(f"term2 = 0.5 * mu_2^T * Σ^-1 * mu_2 = 0.5×({mu2[0]}²+{mu2[1]}²) = {0.5 * np.dot(mu2, mu2)}")
print(f"term3 = log(pi_2) = log({prior2}) = {np.log(prior2):.4f}")
print(f"delta_2(x) = term1 - term2 + term3 = {np.dot(x_test, mu2)} - {0.5 * np.dot(mu2, mu2)} + {np.log(prior2):.4f} = {delta2:.4f}")

# For class 3
print("\nFor class 3:")
print(f"mu_3 = {mu3}")
print(f"x = {x_test}")
print(f"term1 = x^T * Σ^-1 * mu_3 = {x_test[0]}×{mu3[0]} + {x_test[1]}×{mu3[1]} = {np.dot(x_test, mu3)}")
print(f"term2 = 0.5 * mu_3^T * Σ^-1 * mu_3 = 0.5×({mu3[0]}²+{mu3[1]}²) = {0.5 * np.dot(mu3, mu3)}")
print(f"term3 = log(pi_3) = log({prior3}) = {np.log(prior3):.4f}")
print(f"delta_3(x) = term1 - term2 + term3 = {np.dot(x_test, mu3)} - {0.5 * np.dot(mu3, mu3)} + {np.log(prior3):.4f} = {delta3:.4f}")

print("\nSince delta_3(x) = delta_1(x) > delta_2(x), the test point x = [1, 1]^T is assigned to either class 1 or class 3.")
print("In this case with equal scores, the classifier would typically select the first class with maximum score, which is class 1.")

print("\nFigures saved to:", save_dir) 