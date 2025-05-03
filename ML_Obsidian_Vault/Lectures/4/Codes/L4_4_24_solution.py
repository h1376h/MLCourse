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
print("\nGiven Data:")
print(f"Class 1 (X1):\n{X1}")
print(f"Class 2 (X2):\n{X2}")
n1 = X1.shape[0]
n2 = X2.shape[0]
print(f"n1 = {n1}, n2 = {n2}")

# Step 1: Calculate the mean vectors μ1 and μ2 for each class
print("\nStep 1: Calculate the mean vectors for each class")
print("------------------------------------------------")

sum1 = np.sum(X1, axis=0)
mu1 = sum1 / n1
print(f"Sum of vectors in X1: {sum1}")
print(f"Mean vector for class 1 (μ1) = Sum / n1 = {sum1} / {n1} = {mu1}")

sum2 = np.sum(X2, axis=0)
mu2 = sum2 / n2
print(f"\nSum of vectors in X2: {sum2}")
print(f"Mean vector for class 2 (μ2) = Sum / n2 = {sum2} / {n2} = {mu2}")

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
print("S_k = Sum[ (x - μ_k)(x - μ_k)^T ] for x in class k")

# Calculate scatter matrices
S1 = np.zeros((2, 2))
S2 = np.zeros((2, 2))

print("\nCalculating S1 for Class 1:")
for i, x in enumerate(X1):
    x_minus_mu = (x - mu1).reshape(2, 1) # Reshape for outer product
    outer_prod = np.dot(x_minus_mu, x_minus_mu.T)
    print(f"  x{i+1} = {x}, x - μ1 = {x_minus_mu.flatten()}")
    print(f"  (x - μ1)(x - μ1)^T =\n{outer_prod}")
    S1 += outer_prod

print(f"\nWithin-class scatter matrix for class 1 (S1) = Sum of above matrices:")
print(S1)

print("\nCalculating S2 for Class 2:")
for i, x in enumerate(X2):
    x_minus_mu = (x - mu2).reshape(2, 1) # Reshape for outer product
    outer_prod = np.dot(x_minus_mu, x_minus_mu.T)
    print(f"  x{i+1} = {x}, x - μ2 = {x_minus_mu.flatten()}")
    print(f"  (x - μ2)(x - μ2)^T =\n{outer_prod}")
    S2 += outer_prod

print(f"\nWithin-class scatter matrix for class 2 (S2) = Sum of above matrices:")
print(S2)

# Step 3: Determine the total within-class scatter matrix SW
print("\nStep 3: Determine the total within-class scatter matrix SW")
print("--------------------------------------------------------")
print("SW = S1 + S2")
SW = S1 + S2
print(f"Total within-class scatter matrix (SW):\n{SW}")

# Step 4: Calculate the between-class scatter matrix SB
print("\nStep 4: Calculate the between-class scatter matrix SB")
print("---------------------------------------------------")
print("SB = n1*(μ1-μ)(μ1-μ)^T + n2*(μ2-μ)(μ2-μ)^T")

# Calculate the global mean
mu = (n1 * mu1 + n2 * mu2) / (n1 + n2)
print(f"\nGlobal mean (μ) = (n1*μ1 + n2*μ2) / (n1+n2) = ({n1}*{mu1} + {n2}*{mu2}) / {n1+n2} = {mu}")

mu1_minus_mu = (mu1 - mu).reshape(2, 1)
mu2_minus_mu = (mu2 - mu).reshape(2, 1)

print(f"μ1 - μ = {mu1_minus_mu.flatten()}")
print(f"μ2 - μ = {mu2_minus_mu.flatten()}")

SB_term1 = n1 * np.dot(mu1_minus_mu, mu1_minus_mu.T)
SB_term2 = n2 * np.dot(mu2_minus_mu, mu2_minus_mu.T)

print(f"\nTerm 1: n1*(μ1-μ)(μ1-μ)^T = {n1} * \n{np.dot(mu1_minus_mu, mu1_minus_mu.T)} =\n{SB_term1}")
print(f"\nTerm 2: n2*(μ2-μ)(μ2-μ)^T = {n2} * \n{np.dot(mu2_minus_mu, mu2_minus_mu.T)} =\n{SB_term2}")

SB = SB_term1 + SB_term2

print(f"\nBetween-class scatter matrix (SB) = Term 1 + Term 2:")
print(SB)

# Alternative calculation for 2 classes: SB = (n1*n2)/(n1+n2) * (μ1-μ2)(μ1-μ2)^T
mu_diff = (mu1 - mu2).reshape(2, 1)
SB_alt = (n1 * n2) / (n1 + n2) * np.dot(mu_diff, mu_diff.T)
print(f"\nAlternative calculation SB = ({n1}*{n2})/({n1}+{n2})*(μ1-μ2)(μ1-μ2)^T:")
print(f"μ1 - μ2 = {mu_diff.flatten()}")
print(f"SB = { (n1 * n2) / (n1 + n2)} * \n{np.dot(mu_diff, mu_diff.T)} = \n{SB_alt}")
# Note: The results should be the same

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
print("We need to solve the generalized eigenvalue problem: SB * w = λ * SW * w")
print("This is equivalent to finding eigenvectors of SW^(-1) * SB")

# Calculate SW inverse
print(f"\nCalculating the inverse of SW:")
try:
    SW_inv = np.linalg.inv(SW)
    print(f"SW^(-1) =\n{SW_inv}")
except np.linalg.LinAlgError:
    print("SW is singular, cannot compute inverse directly. Using pseudo-inverse or alternative methods.")
    # Handle singularity if needed, e.g., using pseudo-inverse
    SW_inv = np.linalg.pinv(SW)
    print(f"Using pseudo-inverse SW^(+) =\n{SW_inv}")

# Calculate SW^(-1) * SB
M = np.dot(SW_inv, SB)
print(f"\nCalculating M = SW^(-1) * SB:")
print(f"M =\n{M}")

print("\nSolving the eigenvalue problem M * w = λ * w for M:")
# Find the eigenvalues and eigenvectors of M
eigenvalues_direct, eigenvectors_direct = np.linalg.eig(M)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx_direct = np.argsort(eigenvalues_direct.real)[::-1] # Sort by real part
eigenvalues_direct = eigenvalues_direct[idx_direct]
eigenvectors_direct = eigenvectors_direct[:, idx_direct]

print(f"Eigenvalues of M (λ): {eigenvalues_direct}")
print(f"Eigenvectors of M (columns):\n{eigenvectors_direct}")

# The optimal projection direction w is the eigenvector corresponding to the largest eigenvalue
w = eigenvectors_direct[:, 0].real # Take the real part
print(f"\nSelected eigenvector for largest eigenvalue (before normalization): {w}")

# Normalize w to have unit length
w_norm = np.linalg.norm(w)
w = w / w_norm

print(f"Normalization: ||w|| = {w_norm:.4f}")
print(f"Optimal projection direction w (normalized eigenvector): {w}")

# Verification using scipy's generalized eigenvalue solver (should give same direction)
print("\nVerification using scipy.linalg.eigh(SB, SW):")
eigenvalues_gen, eigenvectors_gen = linalg.eigh(SB, SW)
idx_gen = np.argsort(eigenvalues_gen)[::-1]
eigenvalues_gen = eigenvalues_gen[idx_gen]
eigenvectors_gen = eigenvectors_gen[:, idx_gen]
w_gen = eigenvectors_gen[:, 0]
w_gen = w_gen / np.linalg.norm(w_gen)
# Check if directions are the same (allow for sign flip)
sign_check = np.sign(w[0]) == np.sign(w_gen[0])
w_gen_adjusted = w_gen if sign_check else -w_gen
print(f"Largest Generalized Eigenvalue: {eigenvalues_gen[0]:.4f}")
print(f"Corresponding Generalized Eigenvector (normalized): {w_gen_adjusted}")
print(f"Matches direct calculation direction: {np.allclose(w, w_gen_adjusted)}")

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
print(f"New point: {new_point}")

# Project the new point and class means onto the LDA direction w
print(f"Projection formula: proj = w^T * x")
proj_new = np.dot(w, new_point)
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)

print(f"Projection of new point = {w} . {new_point} = {proj_new:.4f}")
print(f"Projection of class 1 mean = {w} . {mu1} = {proj_mu1:.4f}")
print(f"Projection of class 2 mean = {w} . {mu2} = {proj_mu2:.4f}")

# Calculate distances to projected means
print(f"\nCalculate distance in projected space: |proj_new - proj_mean|")
dist_to_mu1 = np.abs(proj_new - proj_mu1)
dist_to_mu2 = np.abs(proj_new - proj_mu2)

print(f"Distance to projected class 1 mean = |{proj_new:.4f} - ({proj_mu1:.4f})| = {dist_to_mu1:.4f}")
print(f"Distance to projected class 2 mean = |{proj_new:.4f} - ({proj_mu2:.4f})| = {dist_to_mu2:.4f}")

# Determine the class
assigned_class = 1 if dist_to_mu1 < dist_to_mu2 else 2
print(f"\nSince {dist_to_mu1:.4f} < {dist_to_mu2:.4f}, the new point ({new_point[0]}, {new_point[1]}) is assigned to Class {assigned_class}.")

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
print("5. We found the optimal projection direction w by solving the generalized eigenvalue problem SW^(-1)SB * w = λ * w.")
print(f"6. For the new data point (5, 5), we projected it and the means onto w and assigned it to Class {assigned_class} based on the closer projected mean.")
print("\nThe LDA projection has successfully separated the two classes and classified the new point.") 