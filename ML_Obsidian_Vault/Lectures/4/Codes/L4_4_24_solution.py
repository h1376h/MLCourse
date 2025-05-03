import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 24: LDA Projection for Two-Dimensional Dataset")
print("=====================================================")

# Given data
X1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
X2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

# Step 1: Calculate the mean vectors μ1 and μ2 for each class
print("\nStep 1: Calculate the mean vectors for each class")
print("------------------------------------------------")

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)

print(f"Mean vector for class 1 (μ1): {mu1}")
print(f"Mean vector for class 2 (μ2): {mu2}")

# Plot the data points and means
plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2')

# Label the points
for i, point in enumerate(X1):
    plt.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
for i, point in enumerate(X2):
    plt.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points and Class Means', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(os.path.join(save_dir, "data_points_means.png"), dpi=300, bbox_inches='tight')

# Step 2: Compute the within-class scatter matrices S1 and S2 for each class
print("\nStep 2: Compute the within-class scatter matrices for each class")
print("--------------------------------------------------------------")

# Calculate scatter matrices
S1 = np.zeros((2, 2))
S2 = np.zeros((2, 2))

for x in X1:
    x_minus_mu = x - mu1
    S1 += np.outer(x_minus_mu, x_minus_mu)

for x in X2:
    x_minus_mu = x - mu2
    S2 += np.outer(x_minus_mu, x_minus_mu)

print(f"Within-class scatter matrix for class 1 (S1):")
print(S1)
print(f"\nWithin-class scatter matrix for class 2 (S2):")
print(S2)

# Step 3: Determine the total within-class scatter matrix SW
print("\nStep 3: Determine the total within-class scatter matrix SW")
print("--------------------------------------------------------")

SW = S1 + S2
print(f"Total within-class scatter matrix (SW):")
print(SW)

# Step 4: Calculate the between-class scatter matrix SB
print("\nStep 4: Calculate the between-class scatter matrix SB")
print("---------------------------------------------------")

# Number of samples in each class
n1 = X1.shape[0]
n2 = X2.shape[0]
n = n1 + n2

# Calculate the global mean
mu = (n1 * mu1 + n2 * mu2) / n

# Alternative approach for between-class scatter matrix:
# Using direct definition of SB for two classes
mu_diff = mu1 - mu2
SB = n1 * np.outer(mu1 - mu, mu1 - mu) + n2 * np.outer(mu2 - mu, mu2 - mu)
# Or more directly for the two-class case:
# SB = n1 * n2 / n * np.outer(mu1 - mu2, mu1 - mu2)

print(f"Global mean (μ): {mu}")
print(f"Between-class scatter matrix (SB):")
print(SB)

# Plot the scatter matrices as ellipses
plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2')
plt.scatter(mu[0], mu[1], color='green', s=200, marker='*', label='Global Mean')

# Draw a line connecting the means
plt.plot([mu1[0], mu2[0]], [mu1[1], mu2[1]], 'g--', linewidth=2, label='Mean Difference')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points, Class Means, and Global Mean', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(os.path.join(save_dir, "scatter_matrices.png"), dpi=300, bbox_inches='tight')

# Step 5: Find the optimal projection direction w
print("\nStep 5: Find the optimal projection direction w")
print("---------------------------------------------")

# Solve the generalized eigenvalue problem: SB * w = λ * SW * w
# This is equivalent to SW^(-1) * SB * w = λ * w

# Method 1: Using scipy's eigh for generalized eigenvalue problem
eigenvalues, eigenvectors = linalg.eigh(SB, SW)

# The eigenvector corresponding to the largest eigenvalue gives the optimal projection direction
idx = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Optimal projection direction w
w = eigenvectors[:, 0]  # First eigenvector (corresponding to largest eigenvalue)

# Normalize w to have unit length
w = w / np.linalg.norm(w)

print(f"Eigenvalues: {eigenvalues}")
print(f"Optimal projection direction w: {w}")

# Method 2: Direct calculation (should give same result)
# Compute SW^(-1) * SB
SW_inv = np.linalg.inv(SW)
M = np.dot(SW_inv, SB)

# Find the eigenvalues and eigenvectors of M
eigenvalues_direct, eigenvectors_direct = np.linalg.eig(M)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx_direct = np.argsort(eigenvalues_direct)[::-1]
eigenvalues_direct = eigenvalues_direct[idx_direct]
eigenvectors_direct = eigenvectors_direct[:, idx_direct]

# The eigenvector corresponding to the largest eigenvalue
w_direct = eigenvectors_direct[:, 0]

# Normalize w_direct to have unit length
w_direct = w_direct / np.linalg.norm(w_direct)

print(f"\nUsing direct calculation:")
print(f"Eigenvalues: {eigenvalues_direct}")
print(f"Optimal projection direction w: {w_direct}")

# Plot the data with the LDA projection direction
plt.figure(figsize=(12, 10))

# Plot original data
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2')

# Center of the plot (use global mean)
center = mu

# Plot the LDA direction (scaled for visualization)
scale = 8  # Adjust this value to make the line longer/shorter
plt.arrow(center[0], center[1], scale * w[0], scale * w[1], 
          head_width=0.5, head_length=0.5, fc='k', ec='k', linewidth=2, label='LDA Direction')

# Plot the projections of class means onto the LDA direction
t1 = np.dot(mu1 - center, w) / np.dot(w, w)
proj1 = center + t1 * w
plt.plot([mu1[0], proj1[0]], [mu1[1], proj1[1]], 'b--', linewidth=1)
plt.scatter(proj1[0], proj1[1], color='blue', s=150, marker='+')

t2 = np.dot(mu2 - center, w) / np.dot(w, w)
proj2 = center + t2 * w
plt.plot([mu2[0], proj2[0]], [mu2[1], proj2[1]], 'r--', linewidth=1)
plt.scatter(proj2[0], proj2[1], color='red', s=150, marker='+')

# Project all data points
for i, x in enumerate(X1):
    t = np.dot(x - center, w) / np.dot(w, w)
    proj = center + t * w
    plt.plot([x[0], proj[0]], [x[1], proj[1]], 'b:', linewidth=0.5)
    plt.scatter(proj[0], proj[1], color='blue', s=50, marker='+')

for i, x in enumerate(X2):
    t = np.dot(x - center, w) / np.dot(w, w)
    proj = center + t * w
    plt.plot([x[0], proj[0]], [x[1], proj[1]], 'r:', linewidth=0.5)
    plt.scatter(proj[0], proj[1], color='red', s=50, marker='+')

# Draw a line along the LDA direction
line_x = np.linspace(0, 12, 100)
line_y = center[1] + (line_x - center[0]) * w[1] / w[0]
plt.plot(line_x, line_y, 'k-', linewidth=1, alpha=0.5)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Projection Direction and Data Point Projections', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(0, 12)
plt.ylim(0, 12)

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# Step 6: Classify a new data point (5, 5)
print("\nStep 6: Classify a new data point (5, 5)")
print("---------------------------------------")

new_point = np.array([5, 5])

# Project the new point and class means onto the LDA direction
proj_new = np.dot(new_point, w)
proj_mu1 = np.dot(mu1, w)
proj_mu2 = np.dot(mu2, w)

# Calculate distances to projected means
dist_to_mu1 = np.abs(proj_new - proj_mu1)
dist_to_mu2 = np.abs(proj_new - proj_mu2)

print(f"New point: {new_point}")
print(f"Projection of new point onto LDA direction: {proj_new:.4f}")
print(f"Projection of class 1 mean onto LDA direction: {proj_mu1:.4f}")
print(f"Projection of class 2 mean onto LDA direction: {proj_mu2:.4f}")
print(f"Distance to projected class 1 mean: {dist_to_mu1:.4f}")
print(f"Distance to projected class 2 mean: {dist_to_mu2:.4f}")

# Determine the class
assigned_class = 1 if dist_to_mu1 < dist_to_mu2 else 2
print(f"\nThe new point ({new_point[0]}, {new_point[1]}) is assigned to Class {assigned_class}.")

# Plot the classification result
plt.figure(figsize=(12, 10))

# Plot original data
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2')

# Plot the new point
plt.scatter(new_point[0], new_point[1], color='green', s=150, marker='d', label='New Point (5,5)')

# Draw the LDA projection line
line_x = np.linspace(0, 12, 100)
line_y = center[1] + (line_x - center[0]) * w[1] / w[0]
plt.plot(line_x, line_y, 'k-', linewidth=1, alpha=0.5, label='LDA Direction')

# Project the new point onto the LDA direction
t_new = np.dot(new_point - center, w) / np.dot(w, w)
proj_new_point = center + t_new * w
plt.plot([new_point[0], proj_new_point[0]], [new_point[1], proj_new_point[1]], 'g--', linewidth=1.5)
plt.scatter(proj_new_point[0], proj_new_point[1], color='green', s=150, marker='+')

# Show the decision boundary (perpendicular to LDA direction, at midpoint of projected means)
mid_point = (proj_mu1 + proj_mu2) / 2
mid_proj = center + (mid_point - np.dot(center, w)) * w / np.dot(w, w)

# Decision boundary is perpendicular to LDA direction and passes through mid_proj
perp_vec = np.array([-w[1], w[0]])  # Perpendicular to w
boundary_x = np.linspace(0, 12, 100)
boundary_y = mid_proj[1] + (boundary_x - mid_proj[0]) * perp_vec[1] / perp_vec[0]

plt.plot(boundary_x, boundary_y, 'g-', linewidth=2, label='Decision Boundary')

# Create regions for the two classes
xx, yy = np.meshgrid(np.linspace(0, 12, 100), np.linspace(0, 12, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Project all grid points
grid_proj = np.dot(grid_points, w)
grid_proj = grid_proj.reshape(xx.shape)

# Midpoint of projected means
mid = (proj_mu1 + proj_mu2) / 2

# Classify grid points
grid_class = np.ones_like(grid_proj)
grid_class[grid_proj > mid] = 2  # Class 2 (assuming proj_mu2 > proj_mu1)

# Plot the colored regions
plt.contourf(xx, yy, grid_class, levels=[0.5, 1.5, 2.5], colors=['lightblue', 'lightsalmon'], alpha=0.3)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of New Point Using LDA', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(0, 12)
plt.ylim(0, 12)

# Save the final plot
plt.savefig(os.path.join(save_dir, "classification_result.png"), dpi=300, bbox_inches='tight')

print("\nConclusion:")
print("-----------")
print("1. We computed the mean vectors for each class.")
print("2. We calculated the within-class scatter matrices S1 and S2.")
print("3. We determined the total within-class scatter matrix SW.")
print("4. We computed the between-class scatter matrix SB.")
print("5. We found the optimal projection direction w by solving the generalized eigenvalue problem.")
print(f"6. For the new data point (5, 5), we assigned it to Class {assigned_class} based on LDA.")
print("\nThe LDA projection has successfully separated the two classes and classified the new point.") 