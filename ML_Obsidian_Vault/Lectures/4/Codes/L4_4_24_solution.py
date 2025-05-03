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

# --- Utility function for matrix printing ---
def print_matrix(matrix, name):
    print(f"{name}:")
    print(np.array_str(matrix, precision=4))

# --- Start of Script ---
print("Question 24: LDA Projection for Two-Dimensional Dataset")
print("=====================================================")
np.set_printoptions(precision=4) # Set print precision for numpy arrays

# Given data
X1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
X2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
print("Given Data:")
print(f"Class 1 (X1): {X1}")
print(f"Class 2 (X2): {X2}")
n1 = X1.shape[0]
n2 = X2.shape[0]
print(f"n1 = {n1}, n2 = {n2}")

# Step 1: Calculate the mean vectors μ1 and μ2 for each class
print("Step 1: Calculate the mean vectors for each class")
print("------------------------------------------------")
print("μ_k = (1/n_k) * Sum(x) for x in class k")

sum1 = np.sum(X1, axis=0)
mu1 = sum1 / n1
print(f"Sum of vectors in X1: [{X1[:, 0].sum()}, {X1[:, 1].sum()}] = {sum1}")
print(f"Mean vector for class 1 (μ1) = Sum / n1 = {sum1} / {n1} = {mu1}")

sum2 = np.sum(X2, axis=0)
mu2 = sum2 / n2
print(f"Sum of vectors in X2: [{X2[:, 0].sum()}, {X2[:, 1].sum()}] = {sum2}")
print(f"Mean vector for class 2 (μ2) = Sum / n2 = {sum2} / {n2} = {mu2}")

# Plot the data points and means
plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1 (μ1)')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2 (μ2)')

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
plt.close() # Close the figure to free memory

# Step 2: Compute the within-class scatter matrices S1 and S2 for each class
print("Step 2: Compute the within-class scatter matrices for each class")
print("--------------------------------------------------------------")
print("S_k = Sum[ (x - μ_k)(x - μ_k)^T ] for x in class k")

# Calculate scatter matrices
S1 = np.zeros((2, 2))
S2 = np.zeros((2, 2))

print("Calculating S1 for Class 1 (μ1 = {}):".format(mu1))
for i, x in enumerate(X1):
    x_minus_mu1 = (x - mu1).reshape(2, 1) # Reshape for outer product
    outer_prod = np.dot(x_minus_mu1, x_minus_mu1.T)
    print(f"  Point {i+1}: x = {x}")
    print(f"    x - μ1 = {x_minus_mu1.flatten()}")
    print(f"    (x - μ1)(x - μ1)^T =")
    print(np.array_str(outer_prod, precision=4))
    S1 += outer_prod
    if i < n1 - 1:
        print("    Current Sum (S1) =")
        print(np.array_str(S1, precision=4))
        print("    ----------")

print(f"Final Within-class scatter matrix for class 1 (S1) = Sum of above matrices:")
print_matrix(S1, "S1")

print("\nCalculating S2 for Class 2 (μ2 = {}):".format(mu2))
for i, x in enumerate(X2):
    x_minus_mu2 = (x - mu2).reshape(2, 1) # Reshape for outer product
    outer_prod = np.dot(x_minus_mu2, x_minus_mu2.T)
    print(f"  Point {i+1}: x = {x}")
    print(f"    x - μ2 = {x_minus_mu2.flatten()}")
    print(f"    (x - μ2)(x - μ2)^T =")
    print(np.array_str(outer_prod, precision=4))
    S2 += outer_prod
    if i < n2 - 1:
        print("    Current Sum (S2) =")
        print(np.array_str(S2, precision=4))
        print("    ----------")

print(f"Final Within-class scatter matrix for class 2 (S2) = Sum of above matrices:")
print_matrix(S2, "S2")

# Step 3: Determine the total within-class scatter matrix SW
print("\nStep 3: Determine the total within-class scatter matrix SW")
print("--------------------------------------------------------")
print("SW = S1 + S2")
print(f"SW = \n{np.array_str(S1, precision=4)} \n + \n{np.array_str(S2, precision=4)}")
SW = S1 + S2
print(f"\nTotal within-class scatter matrix (SW):")
print_matrix(SW, "SW")

# Step 4: Calculate the between-class scatter matrix SB
print("Step 4: Calculate the between-class scatter matrix SB")
print("---------------------------------------------------")
# For 2 classes, SB = (n1*n2)/(n1+n2) * (μ1-μ2)(μ1-μ2)^T is simpler
# Or use the general formula: SB = Sum[ nk * (μk - μ)(μk - μ)^T ]
print("Using formula for 2 classes: SB = (n1*n2)/(n1+n2) * (μ1-μ2)(μ1-μ2)^T")

mu_diff = (mu1 - mu2).reshape(2, 1)
print(f"μ1 - μ2 = {mu1} - {mu2} = {mu_diff.flatten()}")

outer_prod_mu_diff = np.dot(mu_diff, mu_diff.T)
print(f"\n(μ1 - μ2)(μ1 - μ2)^T =")
print(np.array_str(outer_prod_mu_diff, precision=4))

coeff = (n1 * n2) / (n1 + n2)
print(f"\nCoefficient = ({n1} * {n2}) / ({n1} + {n2}) = {coeff:.4f}")

SB = coeff * outer_prod_mu_diff
print(f"Between-class scatter matrix (SB) = {coeff:.4f} * (μ1-μ2)(μ1-μ2)^T")
print_matrix(SB, "SB")

# --- Plotting for context ---
# Calculate the global mean for plotting
mu = (n1 * mu1 + n2 * mu2) / (n1 + n2)

plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1 (μ1)')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2 (μ2)')
plt.scatter(mu[0], mu[1], color='green', s=200, marker='*', label='Global Mean (μ)')

# Draw a line connecting the means
plt.plot([mu1[0], mu2[0]], [mu1[1], mu2[1]], 'g--', linewidth=2, label='Mean Difference')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points, Class Means, and Global Mean', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(os.path.join(save_dir, "scatter_matrices.png"), dpi=300, bbox_inches='tight')
plt.close() # Close the figure

# Step 5: Find the optimal projection direction w
print("Step 5: Find the optimal projection direction w")
print("---------------------------------------------")
print("We need to solve the generalized eigenvalue problem: SB * w = λ * SW * w")
print("This is equivalent to finding eigenvectors of SW^(-1) * SB")

# Calculate SW inverse
print("Calculating the inverse of SW:")
print_matrix(SW, "SW")
# Calculate determinant
det_SW = SW[0, 0] * SW[1, 1] - SW[0, 1] * SW[1, 0]
print(f"det(SW) = ({SW[0, 0]:.4f})*({SW[1, 1]:.4f}) - ({SW[0, 1]:.4f})*({SW[1, 0]:.4f})")
print(f"        = {SW[0, 0] * SW[1, 1]:.4f} - {SW[0, 1] * SW[1, 0]:.4f} = {det_SW:.4f}")

if np.abs(det_SW) < 1e-8:
    print("Determinant is close to zero. SW might be singular.")
    # Handle singularity if needed, e.g., using pseudo-inverse
    print("Using pseudo-inverse (pinv) instead.")
    SW_inv = np.linalg.pinv(SW)
    print(f"SW pseudo-inverse (SW^+) = {SW_inv}")
else:
    # Calculate adjoint for 2x2 matrix
    adj_SW = np.array([[SW[1, 1], -SW[0, 1]],
                       [-SW[1, 0], SW[0, 0]]])
    print(f"adj(SW) = [[SW[1,1], -SW[0,1]], [-SW[1,0], SW[0,0]]]")
    print_matrix(adj_SW, "adj(SW)")

    # Calculate inverse
    SW_inv = (1 / det_SW) * adj_SW
    print(f"SW^(-1) = (1 / det(SW)) * adj(SW) = (1 / {det_SW:.4f}) * adj(SW)")
    print_matrix(SW_inv, "SW^(-1)")

# Calculate SW^(-1) * SB
print("Calculating M = SW^(-1) * SB:")
print_matrix(SW_inv, "SW^(-1)")
print("*")
print_matrix(SB, "SB")
M = np.dot(SW_inv, SB)
print("=")
print_matrix(M, "M")
# Detailed multiplication for M[0,0] as an example
print(f"Example: M[0,0] = SW_inv[0,0]*SB[0,0] + SW_inv[0,1]*SB[1,0]")
print(f"                 = ({SW_inv[0,0]:.4f})*({SB[0,0]:.4f}) + ({SW_inv[0,1]:.4f})*({SB[1,0]:.4f})")
print(f"                 = {SW_inv[0,0]*SB[0,0]:.4f} + {SW_inv[0,1]*SB[1,0]:.4f} = {M[0,0]:.4f}")

print("Solving the eigenvalue problem M * w = λ * w for M:")
# Find the eigenvalues and eigenvectors of M
eigenvalues_direct, eigenvectors_direct = np.linalg.eig(M)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx_direct = np.argsort(eigenvalues_direct.real)[::-1] # Sort by real part
eigenvalues_direct = eigenvalues_direct[idx_direct]
eigenvectors_direct = eigenvectors_direct[:, idx_direct]

print(f"Eigenvalues of M (λ): {eigenvalues_direct.real}") # Show real part for clarity
print(f"Eigenvectors of M (columns, corresponding to λ):")
print(np.array_str(eigenvectors_direct.real, precision=4)) # Show real part

# The optimal projection direction w is the eigenvector corresponding to the largest eigenvalue
w_unnormalized = eigenvectors_direct[:, 0].real # Take the real part
print(f"Selected eigenvector for largest eigenvalue λ1={eigenvalues_direct[0].real:.4f} (unnormalized w):")
print(w_unnormalized)

# Normalize w to have unit length
w_norm = np.linalg.norm(w_unnormalized)
w = w_unnormalized / w_norm

print(f"Normalization: ||w_unnormalized|| = sqrt({w_unnormalized[0]:.4f}^2 + {w_unnormalized[1]:.4f}^2) = {w_norm:.4f}")
print(f"Optimal projection direction w = w_unnormalized / ||w_unnormalized||:")
print(w)

# --- Verification / Alternative Calculation ---
# For 2 classes, w is proportional to SW^(-1) * (μ1 - μ2) or SW^(-1) * (μ2 - μ1)
print("Alternative check (for 2 classes): w should be proportional to SW^(-1) * (μ2 - μ1)")
mu2_minus_mu1 = (mu2 - mu1).reshape(2, 1)
w_alt_unnormalized = np.dot(SW_inv, mu2_minus_mu1).flatten()
print(f"μ2 - μ1 = {mu2_minus_mu1.flatten()}")
print(f"SW^(-1) * (μ2 - μ1) = ")
print(np.array_str(w_alt_unnormalized, precision=4))
# Normalize this alternative w
w_alt_normalized = w_alt_unnormalized / np.linalg.norm(w_alt_unnormalized)
print(f"Normalized SW^(-1) * (μ2 - μ1):")
print(np.array_str(w_alt_normalized, precision=4))
# Compare with w from eigenvalue method (allow for sign flip)
if np.allclose(w, w_alt_normalized) or np.allclose(w, -w_alt_normalized):
    print("Direction matches the eigenvalue method result (up to sign).")
else:
    print("WARNING: Direction does NOT match the eigenvalue method result.")

# --- Plotting the projection ---
plt.figure(figsize=(12, 10))

# Plot original data
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label='Mean of Class 1 (μ1)')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label='Mean of Class 2 (μ2)')

# Center of the plot (use global mean)
center = mu

# Plot the LDA direction (scaled for visualization)
scale = 8  # Adjust this value to make the line longer/shorter
plt.arrow(center[0], center[1], scale * w[0], scale * w[1],
          head_width=0.5, head_length=0.5, fc='k', ec='k', linewidth=2, label=f'LDA Direction w={np.array_str(w, precision=2)}')

# Plot the projections of class means onto the LDA direction line
# Project point p onto line defined by direction w passing through center: proj = center + dot(p-center, w)/dot(w,w) * w
# For projections onto w, we just need the scalar value: dot(p, w)
proj_mu1_scalar = np.dot(mu1, w)
proj_mu2_scalar = np.dot(mu2, w)
# Project means onto the line for visualization
t1 = np.dot(mu1 - center, w) / np.dot(w, w)
proj1_vis = center + t1 * w
plt.plot([mu1[0], proj1_vis[0]], [mu1[1], proj1_vis[1]], 'b--', linewidth=1)
plt.scatter(proj1_vis[0], proj1_vis[1], color='blue', s=150, marker='+', label=f'Proj μ1 ({proj_mu1_scalar:.2f})')

t2 = np.dot(mu2 - center, w) / np.dot(w, w)
proj2_vis = center + t2 * w
plt.plot([mu2[0], proj2_vis[0]], [mu2[1], proj2_vis[1]], 'r--', linewidth=1)
plt.scatter(proj2_vis[0], proj2_vis[1], color='red', s=150, marker='+', label=f'Proj μ2 ({proj_mu2_scalar:.2f})')

# Project all data points onto the line for visualization
Y1_proj_vis = []
Y2_proj_vis = []
for i, x in enumerate(X1):
    t = np.dot(x - center, w) / np.dot(w, w)
    proj_vis = center + t * w
    plt.plot([x[0], proj_vis[0]], [x[1], proj_vis[1]], 'b:', linewidth=0.5)
    plt.scatter(proj_vis[0], proj_vis[1], color='blue', s=50, marker='+')
    Y1_proj_vis.append(proj_vis)
for i, x in enumerate(X2):
    t = np.dot(x - center, w) / np.dot(w, w)
    proj_vis = center + t * w
    plt.plot([x[0], proj_vis[0]], [x[1], proj_vis[1]], 'r:', linewidth=0.5)
    plt.scatter(proj_vis[0], proj_vis[1], color='red', s=50, marker='+')
    Y2_proj_vis.append(proj_vis)

# Draw a line along the LDA direction
line_start = center - scale * w # Extend line in both directions
line_end = center + scale * w
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'k-', linewidth=1, alpha=0.5)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Projection Direction and Data Point Projections', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box') # Make axes equal

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')
plt.close() # Close the figure

# --- Calculate Y1 and Y2 (Scalar Projections) ---
print("Calculating Projected Values Y1 and Y2")
print("---------------------------------------")
print("Y_k = X_k * w = projection of each point in class k onto w")
print(f"w = {w}")

print("\nY1 = X1 * w:")
Y1 = np.dot(X1, w)
for i in range(n1):
    print(f"  Point {i+1}: {X1[i]} . {w} = ({X1[i,0]}*{w[0]:.4f}) + ({X1[i,1]}*{w[1]:.4f}) = {Y1[i]:.4f}")
print(f"Y1 = {np.array_str(Y1, precision=4)}")

print("\nY2 = X2 * w:")
Y2 = np.dot(X2, w)
for i in range(n2):
    print(f"  Point {i+1}: {X2[i]} . {w} = ({X2[i,0]}*{w[0]:.4f}) + ({X2[i,1]}*{w[1]:.4f}) = {Y2[i]:.4f}")
print(f"Y2 = {np.array_str(Y2, precision=4)}")

# Step 6: Classify a new data point (5, 5)
print("Step 6: Classify a new data point (5, 5)")
print("---------------------------------------")

new_point = np.array([5, 5])
print(f"New point x_new = {new_point}")
print(f"Optimal projection direction w = {w}")

# Project the new point and class means onto the LDA direction w
print(f"Projection formula: proj = w^T * x (or x . w)")
proj_new = np.dot(w, new_point)
# proj_mu1 and proj_mu2 already calculated as scalar projections above
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)

print(f"Projection of new point: w^T * x_new = {w} . {new_point} = ({w[0]:.4f}*{new_point[0]} + {w[1]:.4f}*{new_point[1]}) = {proj_new:.4f}")
print(f"Projection of class 1 mean: w^T * μ1 = {w} . {mu1} = ({w[0]:.4f}*{mu1[0]:.4f} + {w[1]:.4f}*{mu1[1]:.4f}) = {proj_mu1:.4f}")
print(f"Projection of class 2 mean: w^T * μ2 = {w} . {mu2} = ({w[0]:.4f}*{mu2[0]:.4f} + {w[1]:.4f}*{mu2[1]:.4f}) = {proj_mu2:.4f}")

# Calculate distances to projected means
print(f"Calculate distance in projected 1D space: |proj_new - proj_mean|")
dist_to_mu1 = np.abs(proj_new - proj_mu1)
dist_to_mu2 = np.abs(proj_new - proj_mu2)

print(f"Distance to projected μ1 = |{proj_new:.4f} - ({proj_mu1:.4f})| = {dist_to_mu1:.4f}")
print(f"Distance to projected μ2 = |{proj_new:.4f} - ({proj_mu2:.4f})| = {dist_to_mu2:.4f}")

# Determine the class
assigned_class = 1 if dist_to_mu1 < dist_to_mu2 else 2
print(f"Since distance to projected μ1 ({dist_to_mu1:.4f}) < distance to projected μ2 ({dist_to_mu2:.4f}), the new point {new_point} is assigned to Class {assigned_class}.")

# --- Plotting the classification result ---
plt.figure(figsize=(12, 10))

# Plot original data
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label=f'μ1 ({mu1[0]:.1f},{mu1[1]:.1f})')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label=f'μ2 ({mu2[0]:.1f},{mu2[1]:.1f})')

# Plot the new point
plt.scatter(new_point[0], new_point[1], color='green', s=150, marker='d', label=f'New Point {new_point}')

# Draw the LDA projection line
line_start = center - scale * w
line_end = center + scale * w
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'k-', linewidth=1, alpha=0.5, label=f'LDA Direction w={np.array_str(w, precision=2)}')

# Project the new point onto the LDA direction line for visualization
t_new_vis = np.dot(new_point - center, w) / np.dot(w, w)
proj_new_point_vis = center + t_new_vis * w
plt.plot([new_point[0], proj_new_point_vis[0]], [new_point[1], proj_new_point_vis[1]], 'g--', linewidth=1.5)
plt.scatter(proj_new_point_vis[0], proj_new_point_vis[1], color='green', s=150, marker='+', label=f'Proj New ({proj_new:.2f})')

# Show the projected means on the line
plt.scatter(proj1_vis[0], proj1_vis[1], color='blue', s=150, marker='+')#, label=f'Proj μ1 ({proj_mu1:.2f})')
plt.scatter(proj2_vis[0], proj2_vis[1], color='red', s=150, marker='+')#, label=f'Proj μ2 ({proj_mu2:.2f})')

# Calculate decision boundary: midpoint between projected means in 1D
mid_scalar = (proj_mu1 + proj_mu2) / 2
print(f"Decision Threshold (midpoint of projected means) = ({proj_mu1:.4f} + {proj_mu2:.4f}) / 2 = {mid_scalar:.4f}")

# Find the point on the LDA line corresponding to this midpoint scalar value
# Point P on line = center + t*w, where dot(P, w) = mid_scalar
# dot(center + t*w, w) = mid_scalar
# dot(center, w) + t*dot(w, w) = mid_scalar
# t = (mid_scalar - dot(center, w)) / dot(w, w)
t_mid = (mid_scalar - np.dot(center, w)) / np.dot(w, w)
mid_point_on_line = center + t_mid * w
print(f"Point on LDA line corresponding to threshold = {mid_point_on_line}")

# Draw the decision boundary line (perpendicular to LDA vector w, passing through mid_point_on_line)
perp_vec = np.array([-w[1], w[0]]) # Vector perpendicular to w
# Line equation: mid_point_on_line + s * perp_vec
boundary_start = mid_point_on_line - 6 * perp_vec # Scale for visualization
boundary_end = mid_point_on_line + 6 * perp_vec
plt.plot([boundary_start[0], boundary_end[0]], [boundary_start[1], boundary_end[1]], 'm-', linewidth=2, label=f'Decision Boundary (at {mid_scalar:.2f})')

# Create shaded regions for the two classes based on the decision boundary line
# Use the line equation of the boundary: (x - mid_x)*perp_x + (y - mid_y)*perp_y = 0
# Equivalently: (x - mid_x)*(-w[1]) + (y - mid_y)*w[0] = 0
# Points where (x - mid_x)*(-w[1]) + (y - mid_y)*w[0] > 0 belong to one class, < 0 to the other.
# Let's check which side Class 2 mean (mu2) lies on
boundary_check = (mu2[0] - mid_point_on_line[0])*(-w[1]) + (mu2[1] - mid_point_on_line[1])*w[0]
class2_sign = np.sign(boundary_check) # Sign determines Class 2 side

xx, yy = np.meshgrid(np.linspace(0, 12, 200), np.linspace(0, 12, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = (grid_points[:, 0] - mid_point_on_line[0])*(-w[1]) + (grid_points[:, 1] - mid_point_on_line[1])*w[0]
Z = Z.reshape(xx.shape)

# Assign classes based on the sign relative to Class 2's sign
grid_class = np.ones_like(Z) # Default to class 1
grid_class[np.sign(Z) == class2_sign] = 2 # Assign class 2

# Plot the colored regions
plt.contourf(xx, yy, grid_class, levels=[0.5, 1.5, 2.5], colors=['lightblue', 'lightsalmon'], alpha=0.3)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of New Point Using LDA', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box') # Make axes equal

# Save the final plot
plt.savefig(os.path.join(save_dir, "classification_result.png"), dpi=300, bbox_inches='tight')
plt.close() # Close the figure

print("Conclusion:")
print("-----------")
print("1. Computed mean vectors: μ1={}, μ2={}".format(mu1, mu2))
print("2. Calculated within-class scatter matrices S1 and S2.")
print_matrix(S1, "S1")
print_matrix(S2, "S2")
print("3. Determined total within-class scatter matrix SW = S1 + S2.")
print_matrix(SW, "SW")
print("4. Computed between-class scatter matrix SB.")
print_matrix(SB, "SB")
print("5. Found optimal projection direction w by solving the generalized eigenvalue problem SW^(-1)SB * w = λ * w (largest λ).")
print(f"   Normalized w = {w}")
print(f"6. Calculated projected data Y1 = X1*w and Y2 = X2*w.")
print(f"   Y1 = {np.array_str(Y1, precision=4)}")
print(f"   Y2 = {np.array_str(Y2, precision=4)}")
print(f"7. For the new data point {new_point}, projected it ({proj_new:.4f}) and the means (μ1: {proj_mu1:.4f}, μ2: {proj_mu2:.4f}) onto w.")
print(f"8. Assigned to Class {assigned_class} based on the closer projected mean (Distance to μ1: {dist_to_mu1:.4f}, Distance to μ2: {dist_to_mu2:.4f}).")
print("\nThe LDA projection has successfully separated the two classes and classified the new point.") 