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

# --- Detailed calculation functions ---
def calculate_mean_vector(X):
    """Calculate the mean vector of a class"""
    n = X.shape[0]  # Number of samples
    sum_X = np.sum(X, axis=0)
    mu = sum_X / n
    
    # Print detailed calculation
    print(f"Sum of vectors = {sum_X}")
    print(f"Mean vector = Sum / n = {sum_X} / {n} = {mu}")
    
    return mu

def calculate_scatter_matrix(X, mu):
    """Calculate the scatter matrix with detailed steps"""
    n = X.shape[0]
    S = np.zeros((X.shape[1], X.shape[1]))
    
    for i, x in enumerate(X):
        x_minus_mu = (x - mu).reshape(-1, 1)  # Column vector
        outer_prod = np.dot(x_minus_mu, x_minus_mu.T)
        
        print(f"Point {i+1}: x = {x}")
        print(f"  x - μ = {x_minus_mu.flatten()}")
        print(f"  (x - μ)(x - μ)^T =")
        print(np.array_str(outer_prod, precision=4))
        
        S += outer_prod
        
        if i < n - 1:
            print("  Current Sum (S) =")
            print(np.array_str(S, precision=4))
            print("  ----------")
    
    return S

def calculate_matrix_inverse_2x2(A):
    """Calculate the inverse of a 2x2 matrix with detailed steps"""
    # For a 2x2 matrix A = [[a, b], [c, d]], the inverse is:
    # A^(-1) = (1/det(A)) * [[d, -b], [-c, a]]
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    # Calculate determinant
    det_A = a*d - b*c
    print(f"det(A) = ({a:.4f})*({d:.4f}) - ({b:.4f})*({c:.4f}) = {a*d:.4f} - {b*c:.4f} = {det_A:.4f}")
    
    if np.abs(det_A) < 1e-8:
        print("Determinant is close to zero. Matrix might be singular.")
        return np.linalg.pinv(A)
    
    # Calculate adjoint (adjugate) matrix
    adj_A = np.array([[d, -b], [-c, a]])
    print(f"adj(A) = [[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]] = [[{d:.4f}, {-b:.4f}], [{-c:.4f}, {a:.4f}]]")
    
    # Calculate inverse
    A_inv = (1 / det_A) * adj_A
    print(f"A^(-1) = (1/det(A)) * adj(A) = (1/{det_A:.4f}) * adj(A)")
    print_matrix(A_inv, "A^(-1)")
    
    return A_inv

def classify_lda_point(x_new, w, mu1, mu2, priors=None):
    """
    Classify a new point using LDA with detailed steps.
    
    Parameters:
    -----------
    x_new: new data point to classify
    w: LDA projection direction
    mu1, mu2: class means
    priors: class prior probabilities [p1, p2], defaults to equal priors
    
    Returns:
    --------
    class_label: 1 or 2
    """
    # Project the new point and class means onto the LDA direction
    proj_new = np.dot(w, x_new)
    proj_mu1 = np.dot(w, mu1)
    proj_mu2 = np.dot(w, mu2)
    
    print("LDA Classification using Discriminant Functions:")
    print(f"Projection of new point x_new = {x_new}:")
    print(f"  w^T * x_new = ({w[0]:.4f}*{x_new[0]} + {w[1]:.4f}*{x_new[1]}) = {proj_new:.4f}")
    
    print(f"Projection of class mean μ1 = {mu1}:")
    print(f"  w^T * μ1 = ({w[0]:.4f}*{mu1[0]:.4f} + {w[1]:.4f}*{mu1[1]:.4f}) = {proj_mu1:.4f}")
    
    print(f"Projection of class mean μ2 = {mu2}:")
    print(f"  w^T * μ2 = ({w[0]:.4f}*{mu2[0]:.4f} + {w[1]:.4f}*{mu2[1]:.4f}) = {proj_mu2:.4f}")
    
    # Method 1: Use distances to projected means
    dist_to_mu1 = np.abs(proj_new - proj_mu1)
    dist_to_mu2 = np.abs(proj_new - proj_mu2)
    
    print("\nMethod 1: Classification based on distance to projected means")
    print(f"Distance to projected μ1 = |{proj_new:.4f} - {proj_mu1:.4f}| = {dist_to_mu1:.4f}")
    print(f"Distance to projected μ2 = |{proj_new:.4f} - {proj_mu2:.4f}| = {dist_to_mu2:.4f}")
    
    class_label_m1 = 1 if dist_to_mu1 < dist_to_mu2 else 2
    print(f"Using closest projected mean: Assign to Class {class_label_m1}")
    
    # Method 2: Use threshold at midpoint between projected means
    # Assuming equal priors and equal covariances (standard LDA assumption)
    if priors is None:
        priors = [0.5, 0.5]  # Equal priors
    
    # With unequal priors: threshold = (proj_mu1 + proj_mu2)/2 + (1/2)*ln(p2/p1)
    threshold = (proj_mu1 + proj_mu2) / 2 + 0.5 * np.log(priors[1] / priors[0])
    
    print("\nMethod 2: Classification based on threshold")
    print(f"Threshold = (proj_μ1 + proj_μ2)/2 + (1/2)*ln(p2/p1)")
    print(f"         = ({proj_mu1:.4f} + {proj_mu2:.4f})/2 + (1/2)*ln({priors[1]:.4f}/{priors[0]:.4f})")
    print(f"         = {(proj_mu1 + proj_mu2)/2:.4f} + {0.5 * np.log(priors[1]/priors[0]):.4f}")
    print(f"         = {threshold:.4f}")
    
    class_label_m2 = 1 if proj_new < threshold else 2
    print(f"Using threshold: {proj_new:.4f} {'<' if proj_new < threshold else '>'} {threshold:.4f}, assign to Class {class_label_m2}")
    
    # For equal priors, both methods should give the same result
    if np.isclose(priors[0], priors[1]) and class_label_m1 != class_label_m2:
        print("WARNING: Classification methods disagree with equal priors. Please check calculations.")
    
    return class_label_m1  # Return result based on distance method

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
print("\nStep 1: Calculate the mean vectors for each class")
print("------------------------------------------------")
print("Formula: μ_k = (1/n_k) * Sum(x) for x in class k")

print("\nFor Class 1:")
mu1 = calculate_mean_vector(X1)

print("\nFor Class 2:")
mu2 = calculate_mean_vector(X2)

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
print("\nStep 2: Compute the within-class scatter matrices for each class")
print("--------------------------------------------------------------")
print("Formula: S_k = Sum[ (x - μ_k)(x - μ_k)^T ] for x in class k")

print("\nCalculating S1 for Class 1:")
S1 = calculate_scatter_matrix(X1, mu1)
print("\nFinal Within-class scatter matrix for class 1 (S1):")
print_matrix(S1, "S1")

print("\nCalculating S2 for Class 2:")
S2 = calculate_scatter_matrix(X2, mu2)
print("\nFinal Within-class scatter matrix for class 2 (S2):")
print_matrix(S2, "S2")

# Step 3: Determine the total within-class scatter matrix SW
print("\nStep 3: Determine the total within-class scatter matrix SW")
print("--------------------------------------------------------")
print("Formula: SW = S1 + S2")

SW = S1 + S2
print(f"SW = \n{np.array_str(S1, precision=4)} \n + \n{np.array_str(S2, precision=4)}")
print("\nTotal within-class scatter matrix (SW):")
print_matrix(SW, "SW")

# Step 4: Calculate the between-class scatter matrix SB
print("\nStep 4: Calculate the between-class scatter matrix SB")
print("---------------------------------------------------")
print("Formula for 2 classes: SB = (n1*n2)/(n1+n2) * (μ1-μ2)(μ1-μ2)^T")

mu_diff = mu1 - mu2
print(f"μ1 - μ2 = {mu1} - {mu2} = {mu_diff}")

mu_diff_col = mu_diff.reshape(-1, 1)
outer_prod_mu_diff = np.dot(mu_diff_col, mu_diff_col.T)
print(f"\n(μ1 - μ2)(μ1 - μ2)^T =")
print(np.array_str(outer_prod_mu_diff, precision=4))

coeff = (n1 * n2) / (n1 + n2)
print(f"\nCoefficient = (n1*n2)/(n1+n2) = ({n1}*{n2})/({n1}+{n2}) = {coeff:.4f}")

SB = coeff * outer_prod_mu_diff
print(f"Between-class scatter matrix (SB) = {coeff:.4f} * (μ1-μ2)(μ1-μ2)^T")
print_matrix(SB, "SB")

# Calculate the global mean for plotting
mu = (n1 * mu1 + n2 * mu2) / (n1 + n2)
print(f"\nGlobal mean μ = (n1*μ1 + n2*μ2)/(n1+n2) = ({n1}*{mu1} + {n2}*{mu2})/({n1}+{n2}) = {mu}")

# --- Plotting for context: Data Points, Class Means, and Global Mean ---
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
plt.close()  # Close the figure

# Step 5: Find the optimal projection direction w
print("\nStep 5: Find the optimal projection direction w")
print("---------------------------------------------")
print("We need to solve the generalized eigenvalue problem: SB * w = λ * SW * w")
print("This is equivalent to finding eigenvectors of SW^(-1) * SB")

# Calculate SW inverse with detailed steps
print("\nCalculating the inverse of SW:")
print_matrix(SW, "SW")
SW_inv = calculate_matrix_inverse_2x2(SW)

# Calculate SW^(-1) * SB with detailed steps
print("\nCalculating M = SW^(-1) * SB:")
print_matrix(SW_inv, "SW^(-1)")
print("*")
print_matrix(SB, "SB")

M = np.dot(SW_inv, SB)
print("=")
print_matrix(M, "M")

# Detailed multiplication for each element
print("Detailed matrix multiplication:")
for i in range(2):
    for j in range(2):
        result = 0
        calc_str = ""
        for k in range(2):
            result += SW_inv[i, k] * SB[k, j]
            calc_str += f"({SW_inv[i,k]:.4f}*{SB[k,j]:.4f})"
            if k < 1:  # Not the last element
                calc_str += " + "
        print(f"M[{i},{j}] = {calc_str} = {result:.4f}")

print("\nSolving the eigenvalue problem M * w = λ * w for M:")
# Find the eigenvalues and eigenvectors of M
eigenvalues, eigenvectors = np.linalg.eig(M)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues.real)[::-1]  # Sort by real part in descending order
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues of M (λ): {eigenvalues.real}")  # Show real part for clarity
print(f"Eigenvectors of M (columns, corresponding to λ):")
print(np.array_str(eigenvectors.real, precision=4))  # Show real part

# The optimal projection direction w is the eigenvector corresponding to the largest eigenvalue
w_unnormalized = eigenvectors[:, 0].real  # Take the real part
print(f"\nSelected eigenvector for largest eigenvalue λ1={eigenvalues[0].real:.4f} (unnormalized w):")
print(w_unnormalized)

# Normalize w to have unit length
w_norm = np.linalg.norm(w_unnormalized)
w = w_unnormalized / w_norm

print(f"Normalization: ||w_unnormalized|| = sqrt({w_unnormalized[0]:.4f}^2 + {w_unnormalized[1]:.4f}^2) = {w_norm:.4f}")
print(f"Optimal projection direction w = w_unnormalized / ||w_unnormalized||:")
print(w)

# Alternative calculation: For 2 classes, w is proportional to SW^(-1) * (μ2 - μ1)
print("\nAlternative calculation for 2 classes:")
print("For two classes, w is proportional to SW^(-1) * (μ2 - μ1)")
mu2_minus_mu1 = (mu2 - mu1).reshape(-1, 1)
print(f"μ2 - μ1 = {mu2} - {mu1} = {mu2_minus_mu1.flatten()}")

w_alt_unnormalized = np.dot(SW_inv, mu2_minus_mu1).flatten()
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

# --- Calculate Y1 and Y2 (Scalar Projections) ---
print("\nCalculating Projected Values Y1 and Y2")
print("---------------------------------------")
print("Formula: Y_k = X_k * w = projection of each point in class k onto w")
print(f"w = {w}")

print("\nY1 = X1 * w:")
Y1 = np.zeros(n1)
for i in range(n1):
    Y1[i] = np.dot(X1[i], w)
    print(f"  Point {i+1}: {X1[i]} . {w} = ({X1[i,0]}*{w[0]:.4f}) + ({X1[i,1]}*{w[1]:.4f}) = {Y1[i]:.4f}")
print(f"Y1 = {np.array_str(Y1, precision=4)}")

print("\nY2 = X2 * w:")
Y2 = np.zeros(n2)
for i in range(n2):
    Y2[i] = np.dot(X2[i], w)
    print(f"  Point {i+1}: {X2[i]} . {w} = ({X2[i,0]}*{w[0]:.4f}) + ({X2[i,1]}*{w[1]:.4f}) = {Y2[i]:.4f}")
print(f"Y2 = {np.array_str(Y2, precision=4)}")

# Step 6: Classify a new data point (5, 5)
print("\nStep 6: Classify a new data point (5, 5)")
print("---------------------------------------")

new_point = np.array([5, 5])
print(f"New point x_new = {new_point}")
print(f"Optimal projection direction w = {w}")

# Classify using our detailed function (multiple methods)
assigned_class = classify_lda_point(new_point, w, mu1, mu2)

# Calculate decision boundary for visualization
# Midpoint between projected means
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)
mid_scalar = (proj_mu1 + proj_mu2) / 2
print(f"\nDecision Threshold (midpoint of projected means) = ({proj_mu1:.4f} + {proj_mu2:.4f}) / 2 = {mid_scalar:.4f}")

# Find the point on the LDA line corresponding to this midpoint scalar value
t_mid = (mid_scalar - np.dot(mu, w)) / np.dot(w, w)
mid_point_on_line = mu + t_mid * w
print(f"Point on LDA line corresponding to threshold = {mid_point_on_line}")

# --- Plotting the classification result ---
plt.figure(figsize=(12, 10))

# Plot original data and decision boundary
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=100, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=100, marker='x', label='Class 2')
plt.scatter(mu1[0], mu1[1], color='blue', s=200, marker='*', label=f'μ1 ({mu1[0]:.1f},{mu1[1]:.1f})')
plt.scatter(mu2[0], mu2[1], color='red', s=200, marker='*', label=f'μ2 ({mu2[0]:.1f},{mu2[1]:.1f})')
plt.scatter(new_point[0], new_point[1], color='green', s=150, marker='d', label=f'New Point {new_point}')

# Draw the LDA direction and decision boundary
scale = 8  # Scale for visualization
line_start = mu - scale * w
line_end = mu + scale * w
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'k-', linewidth=1, alpha=0.5, 
         label=f'LDA Direction w=[{w[0]:.2f}, {w[1]:.2f}]')

# Project the new point onto the LDA direction line
proj_new = np.dot(w, new_point)
t_new = (proj_new - np.dot(mu, w)) / np.dot(w, w)
proj_new_point_vis = mu + t_new * w
plt.plot([new_point[0], proj_new_point_vis[0]], [new_point[1], proj_new_point_vis[1]], 'g--', linewidth=1.5)
plt.scatter(proj_new_point_vis[0], proj_new_point_vis[1], color='green', s=150, marker='+', 
            label=f'Proj New ({proj_new:.2f})')

# Draw the decision boundary (perpendicular to w, passing through mid_point_on_line)
perp_vec = np.array([-w[1], w[0]])  # Vector perpendicular to w
boundary_start = mid_point_on_line - 6 * perp_vec
boundary_end = mid_point_on_line + 6 * perp_vec
plt.plot([boundary_start[0], boundary_end[0]], [boundary_start[1], boundary_end[1]], 'm-', linewidth=2, 
         label=f'Decision Boundary (at {mid_scalar:.2f})')

# Create shaded regions for the classes
xx, yy = np.meshgrid(np.linspace(0, 12, 200), np.linspace(0, 12, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
# Equation of decision boundary: (x - mid_point)·perp_vec = 0
Z = (grid_points[:, 0] - mid_point_on_line[0])*(-w[1]) + (grid_points[:, 1] - mid_point_on_line[1])*w[0]
Z = Z.reshape(xx.shape)

# Determine which side corresponds to which class
boundary_check = (mu2[0] - mid_point_on_line[0])*(-w[1]) + (mu2[1] - mid_point_on_line[1])*w[0]
class2_sign = np.sign(boundary_check)

grid_class = np.ones_like(Z)  # Default to class 1
grid_class[np.sign(Z) == class2_sign] = 2  # Assign class 2

# Plot the colored regions
plt.contourf(xx, yy, grid_class, levels=[0.5, 1.5, 2.5], colors=['lightblue', 'lightsalmon'], alpha=0.3)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of New Point Using LDA', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')  # Make axes equal

# Save the final plot
plt.savefig(os.path.join(save_dir, "classification_result.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure

print("\nConclusion:")
print("-----------")
print("1. Computed mean vectors: μ1={}, μ2={}".format(mu1, mu2))
print("2. Calculated within-class scatter matrices S1 and S2.")
print_matrix(S1, "S1")
print_matrix(S2, "S2")
print("3. Determined total within-class scatter matrix SW = S1 + S2.")
print_matrix(SW, "SW")
print("4. Computed between-class scatter matrix SB.")
print_matrix(SB, "SB")
print("5. Found optimal projection direction w = [%.4f, %.4f] by solving the generalized eigenvalue problem." % (w[0], w[1]))
print("6. Calculated projected data:")
print(f"   Y1 = {np.array_str(Y1, precision=4)}")
print(f"   Y2 = {np.array_str(Y2, precision=4)}")
print(f"7. For the new data point {new_point}, projected value = {proj_new:.4f}")
print(f"8. Decision threshold = {mid_scalar:.4f} (midpoint of projected means)")
print(f"9. Assigned to Class {assigned_class} (projected value {proj_new:.4f} is closer to projected mean of Class {assigned_class})")
print("\nThe LDA projection has successfully separated the two classes and classified the new point.")

# --- Plotting the LDA projection direction and data point projections ---
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
          head_width=0.5, head_length=0.5, fc='k', ec='k', linewidth=2, 
          label=f'LDA Direction w=[{w[0]:.2f} {w[1]:.2f}]')

# Calculate projections onto LDA direction
proj_mu1_scalar = np.dot(mu1, w)
proj_mu2_scalar = np.dot(mu2, w)

# Project means onto the line for visualization
t1 = np.dot(mu1 - center, w) / np.dot(w, w)
proj1_vis = center + t1 * w
plt.plot([mu1[0], proj1_vis[0]], [mu1[1], proj1_vis[1]], 'b--', linewidth=1)
plt.scatter(proj1_vis[0], proj1_vis[1], color='blue', s=150, marker='+', 
            label=f'Proj μ1 ({proj_mu1_scalar:.2f})')

t2 = np.dot(mu2 - center, w) / np.dot(w, w)
proj2_vis = center + t2 * w
plt.plot([mu2[0], proj2_vis[0]], [mu2[1], proj2_vis[1]], 'r--', linewidth=1)
plt.scatter(proj2_vis[0], proj2_vis[1], color='red', s=150, marker='+', 
            label=f'Proj μ2 ({proj_mu2_scalar:.2f})')

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
line_start = center - scale * w  # Extend line in both directions
line_end = center + scale * w
plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'k-', linewidth=1, alpha=0.5)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Projection Direction and Data Point Projections', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')  # Make axes equal

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure 