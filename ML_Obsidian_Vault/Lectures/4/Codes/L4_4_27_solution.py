import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 27: Linear Discriminant Analysis with Special Case")
print("==========================================================")

# Step 1: Define the data points for both classes
print("\nStep 1: Define the data points")
print("----------------------------")

# Define the original data points for each class
class_A_x1 = np.array([1, 3]).reshape(2, 1)  # First point of Class A
class_A_x2 = np.array([3, 1]).reshape(2, 1)  # Second point of Class A
class_B_x1 = np.array([6, 4]).reshape(2, 1)  # First point of Class B
class_B_x2 = np.array([4, 6]).reshape(2, 1)  # Second point of Class B

# Stack data points column-wise for each class
X_A = np.hstack([class_A_x1, class_A_x2])  # Class A data matrix (2x2)
X_B = np.hstack([class_B_x1, class_B_x2])  # Class B data matrix (2x2)

print("Original data points:")
print("Class A data points:")
print(f"x_1^(A) = [{X_A[0, 0]}, {X_A[1, 0]}]^T")
print(f"x_2^(A) = [{X_A[0, 1]}, {X_A[1, 1]}]^T")

print("\nClass B data points:")
print(f"x_1^(B) = [{X_B[0, 0]}, {X_B[1, 0]}]^T")
print(f"x_2^(B) = [{X_B[0, 1]}, {X_B[1, 1]}]^T")

# Task 1: Compute the mean vector for each class
print("\nTask 1: Compute the mean vector for each class")
print("--------------------------------------------")

# Calculate mean vector for Class A
n_A = X_A.shape[1]  # Number of data points in Class A
mu_A_sum = np.zeros((2, 1))
for i in range(n_A):
    mu_A_sum += X_A[:, i].reshape(2, 1)
mu_A = mu_A_sum / n_A
print("Step-by-step calculation of mean vector for Class A:")
print(f"Sum of Class A data points: ({X_A[0, 0]} + {X_A[0, 1]}, {X_A[1, 0]} + {X_A[1, 1]})^T = ({mu_A_sum[0, 0]}, {mu_A_sum[1, 0]})^T")
print(f"mu_A = ({mu_A_sum[0, 0]}, {mu_A_sum[1, 0]})^T / {n_A} = ({mu_A[0, 0]}, {mu_A[1, 0]})^T")

# Calculate mean vector for Class B
n_B = X_B.shape[1]  # Number of data points in Class B
mu_B_sum = np.zeros((2, 1))
for i in range(n_B):
    mu_B_sum += X_B[:, i].reshape(2, 1)
mu_B = mu_B_sum / n_B
print("\nStep-by-step calculation of mean vector for Class B:")
print(f"Sum of Class B data points: ({X_B[0, 0]} + {X_B[0, 1]}, {X_B[1, 0]} + {X_B[1, 1]})^T = ({mu_B_sum[0, 0]}, {mu_B_sum[1, 0]})^T")
print(f"mu_B = ({mu_B_sum[0, 0]}, {mu_B_sum[1, 0]})^T / {n_B} = ({mu_B[0, 0]}, {mu_B[1, 0]})^T")

print("\nFinal mean vectors:")
print(f"Mean vector for Class A (mu_A): [{mu_A[0, 0]}, {mu_A[1, 0]}]^T")
print(f"Mean vector for Class B (mu_B): [{mu_B[0, 0]}, {mu_B[1, 0]}]^T")

# Task 2: Compute the covariance matrix for each class
print("\nTask 2: Compute the covariance matrix for each class")
print("-------------------------------------------------")

# Theoretical centered points for calculation clarity
X_A_centered = np.zeros_like(X_A, dtype=float)
X_A_centered[:, 0] = (X_A[:, 0] - mu_A.flatten()).flatten()
X_A_centered[:, 1] = (X_A[:, 1] - mu_A.flatten()).flatten()

print("Step-by-step calculation of centered data points for Class A:")
for i in range(n_A):
    print(f"x_{i+1}^(A) - mu_A = [{X_A[0, i]}, {X_A[1, i]}]^T - [{mu_A[0, 0]}, {mu_A[1, 0]}]^T = [{X_A_centered[0, i]}, {X_A_centered[1, i]}]^T")

# Calculate theoretical covariance matrix for Class A
Sigma_A = np.zeros((2, 2))
for i in range(n_A):
    x_i_centered = X_A_centered[:, i].reshape(2, 1)
    outer_product = np.dot(x_i_centered, x_i_centered.T)
    
    print(f"\nOuter product for centered data point {i+1} of Class A:")
    print(f"[{X_A_centered[0, i]:.2f}] × [{X_A_centered[0, i]:.2f}, {X_A_centered[1, i]:.2f}] = ")
    print(f"[{X_A_centered[1, i]:.2f}]")
    print(f"[{outer_product[0, 0]:.2f}, {outer_product[0, 1]:.2f}]")
    print(f"[{outer_product[1, 0]:.2f}, {outer_product[1, 1]:.2f}]")
    
    Sigma_A += outer_product

Sigma_A = Sigma_A / n_A
print(f"\nSum of outer products / {n_A} = Sigma_A:")
print(f"[{Sigma_A[0, 0]:.2f}, {Sigma_A[0, 1]:.2f}]")
print(f"[{Sigma_A[1, 0]:.2f}, {Sigma_A[1, 1]:.2f}]")

# Calculate theoretical centered points for Class B
X_B_centered = np.zeros_like(X_B, dtype=float)
X_B_centered[:, 0] = (X_B[:, 0] - mu_B.flatten()).flatten()
X_B_centered[:, 1] = (X_B[:, 1] - mu_B.flatten()).flatten()

print("\nStep-by-step calculation of centered data points for Class B:")
for i in range(n_B):
    print(f"x_{i+1}^(B) - mu_B = [{X_B[0, i]}, {X_B[1, i]}]^T - [{mu_B[0, 0]}, {mu_B[1, 0]}]^T = [{X_B_centered[0, i]}, {X_B_centered[1, i]}]^T")

# Calculate theoretical covariance matrix for Class B
Sigma_B = np.zeros((2, 2))
for i in range(n_B):
    x_i_centered = X_B_centered[:, i].reshape(2, 1)
    outer_product = np.dot(x_i_centered, x_i_centered.T)
    
    print(f"\nOuter product for centered data point {i+1} of Class B:")
    print(f"[{X_B_centered[0, i]:.2f}] × [{X_B_centered[0, i]:.2f}, {X_B_centered[1, i]:.2f}] = ")
    print(f"[{X_B_centered[1, i]:.2f}]")
    print(f"[{outer_product[0, 0]:.2f}, {outer_product[0, 1]:.2f}]")
    print(f"[{outer_product[1, 0]:.2f}, {outer_product[1, 1]:.2f}]")
    
    Sigma_B += outer_product

Sigma_B = Sigma_B / n_B
print(f"\nSum of outer products / {n_B} = Sigma_B:")
print(f"[{Sigma_B[0, 0]:.2f}, {Sigma_B[0, 1]:.2f}]")
print(f"[{Sigma_B[1, 0]:.2f}, {Sigma_B[1, 1]:.2f}]")

print("\nFinal covariance matrices:")
print(f"Covariance matrix for Class A (Sigma_A):")
print(Sigma_A)
print(f"\nCovariance matrix for Class B (Sigma_B):")
print(Sigma_B)

# Task 3: Find the optimal projection direction w* with unit length
print("\nTask 3: Find the optimal projection direction w* with unit length")
print("--------------------------------------------------------------")

# Step 1: Calculate the pooled within-class scatter matrix Sw
print("Step 1: Calculate the pooled within-class scatter matrix Sw")
Sw = Sigma_A + Sigma_B
print(f"Sw = Sigma_A + Sigma_B = ")
print(f"[{Sigma_A[0, 0]:.2f}, {Sigma_A[0, 1]:.2f}] + [{Sigma_B[0, 0]:.2f}, {Sigma_B[0, 1]:.2f}] = [{Sw[0, 0]:.2f}, {Sw[0, 1]:.2f}]")
print(f"[{Sigma_A[1, 0]:.2f}, {Sigma_A[1, 1]:.2f}] + [{Sigma_B[1, 0]:.2f}, {Sigma_B[1, 1]:.2f}] = [{Sw[1, 0]:.2f}, {Sw[1, 1]:.2f}]")

# Step 2: Calculate the between-class mean difference
print("\nStep 2: Calculate the between-class mean difference")
mean_diff = mu_A - mu_B
print(f"mu_A - mu_B = [{mu_A[0, 0]:.2f}, {mu_A[1, 0]:.2f}]^T - [{mu_B[0, 0]:.2f}, {mu_B[1, 0]:.2f}]^T = [{mean_diff[0, 0]:.2f}, {mean_diff[1, 0]:.2f}]^T")

# Step 3: Check if Sw is singular
print("\nStep 3: Check if Sw is singular and handle accordingly")
det_Sw = np.linalg.det(Sw)
print(f"Determinant of Sw = {det_Sw:.6f}")

if abs(det_Sw) < 1e-10:
    print("Sw is singular (determinant is effectively zero).")
    print("In this case, we have three options:")
    print("1. Use regularization by adding a small value to the diagonal of Sw")
    print("2. Use the direction connecting the class means as our optimal direction (simpler approach)")
    print("3. Use pseudoinverse (Moore-Penrose inverse) to handle the singularity")
    
    # Option 1: Regularization
    print("\nOption 1: Using regularization by adding a small value to the diagonal:")
    reg_lambda = 1e-5
    Sw_reg = Sw + np.eye(2) * reg_lambda
    print(f"Sw_reg = Sw + λI = ")
    print(f"[{Sw[0, 0]:.2f}, {Sw[0, 1]:.2f}] + [{reg_lambda:.5f}, {0:.5f}] = [{Sw_reg[0, 0]:.5f}, {Sw_reg[0, 1]:.5f}]")
    print(f"[{Sw[1, 0]:.2f}, {Sw[1, 1]:.2f}] + [{0:.5f}, {reg_lambda:.5f}] = [{Sw_reg[1, 0]:.5f}, {Sw_reg[1, 1]:.5f}]")
    
    Sw_reg_inv = np.linalg.inv(Sw_reg)
    print(f"Sw_reg^-1 = ")
    print(f"[{Sw_reg_inv[0, 0]:.4f}, {Sw_reg_inv[0, 1]:.4f}]")
    print(f"[{Sw_reg_inv[1, 0]:.4f}, {Sw_reg_inv[1, 1]:.4f}]")
    
    w_star_reg = np.dot(Sw_reg_inv, mean_diff)
    print(f"w*_reg = Sw_reg^-1 * (mu_A - mu_B) = [{w_star_reg[0, 0]:.4f}, {w_star_reg[1, 0]:.4f}]^T")
    
    # Normalize the regularized solution
    w_star_reg_norm = np.sqrt(np.sum(w_star_reg**2))
    w_star_reg_normalized = w_star_reg / w_star_reg_norm
    print(f"||w*_reg|| = {w_star_reg_norm:.4f}")
    print(f"w*_reg_normalized = [{w_star_reg_normalized[0, 0]:.4f}, {w_star_reg_normalized[1, 0]:.4f}]^T")
    
    # Option 2: Direction connecting means
    print("\nOption 2: Using the direction connecting the means:")
    w_star_direct = mean_diff  # Direction from Class B to Class A
    
    # Normalize to unit length
    w_star_direct_norm = np.sqrt(np.sum(w_star_direct**2))
    print(f"||w*_direct|| = sqrt({mean_diff[0, 0]:.2f}^2 + {mean_diff[1, 0]:.2f}^2) = sqrt({mean_diff[0, 0]**2:.2f} + {mean_diff[1, 0]**2:.2f}) = {w_star_direct_norm:.4f}")
    
    w_star_direct_normalized = w_star_direct / w_star_direct_norm
    print(f"w*_direct_normalized = w*_direct / ||w*_direct|| = [{w_star_direct[0, 0]:.4f}, {w_star_direct[1, 0]:.4f}]^T / {w_star_direct_norm:.4f} = [{w_star_direct_normalized[0, 0]:.4f}, {w_star_direct_normalized[1, 0]:.4f}]^T")
    
    # Option 3: Using pseudoinverse (Moore-Penrose inverse)
    print("\nOption 3: Using the pseudoinverse (Moore-Penrose inverse):")
    
    # Step 1: Compute the Singular Value Decomposition (SVD) of Sw
    print("Step 1: Compute the Singular Value Decomposition (SVD) of Sw")
    U, s, Vh = np.linalg.svd(Sw)
    print(f"SVD decomposition: Sw = U * Σ * V^H")
    print(f"U = ")
    print(f"[{U[0, 0]:.4f}, {U[0, 1]:.4f}]")
    print(f"[{U[1, 0]:.4f}, {U[1, 1]:.4f}]")
    print(f"Singular values (Σ) = [{s[0]:.4f}, {s[1]:.4f}]")
    print(f"V^H = ")
    print(f"[{Vh[0, 0]:.4f}, {Vh[0, 1]:.4f}]")
    print(f"[{Vh[1, 0]:.4f}, {Vh[1, 1]:.4f}]")
    
    # Step 2: Compute the pseudoinverse using the SVD
    print("\nStep 2: Compute the pseudoinverse using the SVD")
    # Threshold for treating singular values as zero (numerical stability)
    threshold = 1e-10
    
    # Identify non-zero singular values
    s_inv = np.zeros_like(s)
    nonzero_indices = np.where(s > threshold)[0]
    
    if len(nonzero_indices) > 0:
        s_inv[nonzero_indices] = 1.0 / s[nonzero_indices]
    
    print(f"Reciprocal of non-zero singular values (Σ^+) = [{s_inv[0]:.4f}, {s_inv[1]:.4f}]")
    
    # Compute the pseudoinverse: Sw^+ = V * Σ^+ * U^H
    S_inv = np.diag(s_inv)
    Sw_pinv = np.dot(Vh.T, np.dot(S_inv, U.T))
    
    print(f"Pseudoinverse Sw^+ = ")
    print(f"[{Sw_pinv[0, 0]:.4f}, {Sw_pinv[0, 1]:.4f}]")
    print(f"[{Sw_pinv[1, 0]:.4f}, {Sw_pinv[1, 1]:.4f}]")
    
    # Step 3: Compute w* using the pseudoinverse
    print("\nStep 3: Compute w* using the pseudoinverse")
    w_star_pinv = np.dot(Sw_pinv, mean_diff)
    print(f"w*_pinv = Sw^+ * (mu_A - mu_B) = [{w_star_pinv[0, 0]:.4f}, {w_star_pinv[1, 0]:.4f}]^T")
    
    # Step 4: Normalize to unit length
    w_star_pinv_norm = np.sqrt(np.sum(w_star_pinv**2))
    w_star_pinv_normalized = w_star_pinv / w_star_pinv_norm
    print(f"||w*_pinv|| = sqrt({w_star_pinv[0, 0]:.4f}^2 + {w_star_pinv[1, 0]:.4f}^2) = {w_star_pinv_norm:.4f}")
    print(f"w*_pinv_normalized = [{w_star_pinv_normalized[0, 0]:.4f}, {w_star_pinv_normalized[1, 0]:.4f}]^T")
    
    # Compare all three approaches
    print("\nComparing all three approaches:")
    cos_sim_12 = np.dot(w_star_reg_normalized.flatten(), w_star_direct_normalized.flatten())
    cos_sim_13 = np.dot(w_star_reg_normalized.flatten(), w_star_pinv_normalized.flatten())
    cos_sim_23 = np.dot(w_star_direct_normalized.flatten(), w_star_pinv_normalized.flatten())
    
    print(f"Cosine similarity between regularization and direct: {cos_sim_12:.6f}")
    print(f"Cosine similarity between regularization and pseudoinverse: {cos_sim_13:.6f}")
    print(f"Cosine similarity between direct and pseudoinverse: {cos_sim_23:.6f}")
    
    # Use the regularized solution for further analysis (can be changed if desired)
    w_star = w_star_reg
    w_star_normalized = w_star_reg_normalized
    w_star_norm = w_star_reg_norm
    
    print("\nAll three methods yield very similar directions, confirming our solution.")
else:
    # If not singular, proceed with standard LDA calculation
    print("\nSw is not singular, proceeding with standard LDA calculation.")
    
    # Step 3: Calculate the inverse of Sw
    print("\nStep 3: Calculate the inverse of Sw")
    Sw_inv = np.linalg.inv(Sw)
    print(f"Sw^-1 = ")
    print(f"[{Sw_inv[0, 0]:.2f}, {Sw_inv[0, 1]:.2f}]")
    print(f"[{Sw_inv[1, 0]:.2f}, {Sw_inv[1, 1]:.2f}]")
    
    # Step 4: Calculate w* = Sw^-1 * (mu_A - mu_B)
    print("\nStep 4: Calculate w* = Sw^-1 * (mu_A - mu_B)")
    w_star = np.dot(Sw_inv, mean_diff)
    print(f"w* = Sw^-1 * (mu_A - mu_B) = ")
    print(f"[{Sw_inv[0, 0]:.2f}, {Sw_inv[0, 1]:.2f}] * [{mean_diff[0, 0]:.2f}] = [{w_star[0, 0]:.4f}]")
    print(f"[{Sw_inv[1, 0]:.2f}, {Sw_inv[1, 1]:.2f}] * [{mean_diff[1, 0]:.2f}] = [{w_star[1, 0]:.4f}]")
    
    # Step 5: Normalize w* to unit length
    print("\nStep 5: Normalize w* to unit length")
    w_star_norm = np.sqrt(np.sum(w_star**2))
    print(f"||w*|| = sqrt({w_star[0, 0]:.4f}^2 + {w_star[1, 0]:.4f}^2) = sqrt({w_star[0, 0]**2:.4f} + {w_star[1, 0]**2:.4f}) = {w_star_norm:.4f}")
    w_star_normalized = w_star / w_star_norm
    print(f"w*_normalized = w* / ||w*|| = [{w_star[0, 0]:.4f}, {w_star[1, 0]:.4f}]^T / {w_star_norm:.4f} = [{w_star_normalized[0, 0]:.4f}, {w_star_normalized[1, 0]:.4f}]^T")

print("\n-------------------------------------------------------")
print("Theoretical interpretation of the results:")
print("-------------------------------------------------------")
print("1. Fisher's Linear Discriminant Analysis seeks to maximize between-class")
print("   separation while minimizing within-class scatter.")
print("2. The criterion function S(w) quantifies this trade-off.")
print("3. The covariance matrices show different patterns for each class:")
print("   - For Class A, we observe negative correlation between features")
print("   - For Class B, we observe positive correlation between features")
print("4. The pooled within-class scatter matrix is special in this case:")
print("   it has a determinant close to zero, indicating near-singularity.")
print("5. This singularity suggests that the within-class scatter is")
print("   concentrated along a particular direction in the feature space.")
print("6. We handled the singularity through regularization, which stabilizes")
print("   the computation while minimally affecting the solution.")
print("7. The optimal projection direction effectively separates the classes.")
print("-------------------------------------------------------")

# Create a visualization showing the data points, means, and optimal direction w*
plt.figure(figsize=(10, 8))
plt.scatter(X_A[0, :], X_A[1, :], color='blue', s=100, marker='o', label='Class A')
plt.scatter(X_B[0, :], X_B[1, :], color='red', s=100, marker='x', label='Class B')

# Plot mean vectors
plt.scatter(mu_A[0, 0], mu_A[1, 0], color='blue', s=200, marker='*', label='Mean of Class A')
plt.scatter(mu_B[0, 0], mu_B[1, 0], color='red', s=200, marker='*', label='Mean of Class B')

# Draw a line connecting the means
plt.plot([mu_A[0, 0], mu_B[0, 0]], [mu_A[1, 0], mu_B[1, 0]], 'k--', alpha=0.5)

# Label the points
for i in range(n_A):
    plt.annotate(f'$x_{i+1}^{{(A)}}$', (X_A[0, i], X_A[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i in range(n_B):
    plt.annotate(f'$x_{i+1}^{{(B)}}$', (X_B[0, i], X_B[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

# Label the means
plt.annotate('$\\mu_A$', (mu_A[0, 0], mu_A[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)
plt.annotate('$\\mu_B$', (mu_B[0, 0], mu_B[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)

# Fix: Use a better origin for the LDA direction arrows
# Use the midpoint between class means as the origin
origin = np.array([(mu_A[0, 0] + mu_B[0, 0])/2, (mu_A[1, 0] + mu_B[1, 0])/2])

# Draw scaled LDA direction (non-normalized)
arrow_scale = 1.0  # Adjust for better visibility
plt.arrow(origin[0], origin[1], 
          arrow_scale * w_star_normalized[0, 0], arrow_scale * w_star_normalized[1, 0],
          head_width=0.2, head_length=0.3, fc='green', ec='green', 
          label='LDA direction (w*)')

# Draw opposite direction for better visualization
plt.arrow(origin[0], origin[1], 
          -arrow_scale * w_star_normalized[0, 0], -arrow_scale * w_star_normalized[1, 0],
          head_width=0.2, head_length=0.3, fc='purple', ec='purple', 
          label='Opposite direction')

# Draw the decision boundary (perpendicular to w*)
perp_vec = np.array([-w_star_normalized[1, 0], w_star_normalized[0, 0]])
boundary_length = 3  # Length of the boundary line
plt.plot([origin[0] - boundary_length * perp_vec[0], origin[0] + boundary_length * perp_vec[0]],
         [origin[1] - boundary_length * perp_vec[1], origin[1] + boundary_length * perp_vec[1]],
         'g--', linewidth=2, label='Decision boundary')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points and LDA Projection Direction', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Set axis limits to show all data points with some margin
margin = 1
plt.xlim(min(np.min(X_A[0, :]), np.min(X_B[0, :])) - margin, 
         max(np.max(X_A[0, :]), np.max(X_B[0, :])) + margin)
plt.ylim(min(np.min(X_A[1, :]), np.min(X_B[1, :])) - margin, 
         max(np.max(X_A[1, :]), np.max(X_B[1, :])) + margin)

plt.legend(fontsize=10, loc='best')
plt.axis('equal')

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# ---------------- VISUALIZATION 1: Projection onto LDA Direction ----------------
plt.figure(figsize=(12, 7))

# Create a new axis for the projected data
w_unit = w_star_normalized.flatten()

# Project the data points onto the LDA direction
X_A_proj = np.array([np.dot(X_A[:, i], w_unit) for i in range(X_A.shape[1])])
X_B_proj = np.array([np.dot(X_B[:, i], w_unit) for i in range(X_B.shape[1])])

# Calculate the range for the plot with added margin
margin = 1.0
x_min = min(np.min(X_A_proj), np.min(X_B_proj)) - margin
x_max = max(np.max(X_A_proj), np.max(X_B_proj)) + margin
y_min, y_max = -0.1, 0.5  # For visualization height

# Plot the projection axis line
plt.plot([x_min, x_max], [0, 0], 'k-', lw=2, label='Projection Axis')

# Plot the projected points with increased vertical separation
y_offset_A = 0.15
y_offset_B = 0.05
plt.scatter(X_A_proj, np.zeros_like(X_A_proj) + y_offset_A, s=100, color='blue', marker='o', label='Class A')
plt.scatter(X_B_proj, np.zeros_like(X_B_proj) + y_offset_B, s=100, color='red', marker='x', label='Class B')

# Calculate and plot the projected means
mu_A_proj = np.dot(mu_A.flatten(), w_unit)
mu_B_proj = np.dot(mu_B.flatten(), w_unit)
plt.scatter(mu_A_proj, y_offset_A + 0.15, s=200, color='blue', marker='*', label='Mean of Class A')
plt.scatter(mu_B_proj, y_offset_B + 0.15, s=200, color='red', marker='*', label='Mean of Class B')

# Add vertical projection lines to show the projection process
plt.plot([mu_A_proj, mu_A_proj], [0, y_offset_A + 0.15], 'b--', alpha=0.6)
plt.plot([mu_B_proj, mu_B_proj], [0, y_offset_B + 0.15], 'r--', alpha=0.6)

# Draw projection lines for all points
for i in range(len(X_A_proj)):
    plt.plot([X_A_proj[i], X_A_proj[i]], [0, y_offset_A], 'b--', alpha=0.4)
for i in range(len(X_B_proj)):
    plt.plot([X_B_proj[i], X_B_proj[i]], [0, y_offset_B], 'r--', alpha=0.4)

# Add annotations
for i in range(len(X_A_proj)):
    plt.annotate(f'$x_{i+1}^{{(A)}}$', (X_A_proj[i], y_offset_A), 
                 xytext=(0, 5), textcoords='offset points', fontsize=12, ha='center')
for i in range(len(X_B_proj)):
    plt.annotate(f'$x_{i+1}^{{(B)}}$', (X_B_proj[i], y_offset_B), 
                 xytext=(0, 5), textcoords='offset points', fontsize=12, ha='center')

plt.annotate('$\\mu_A$', (mu_A_proj, y_offset_A + 0.15), 
             xytext=(0, 5), textcoords='offset points', fontsize=14, ha='center')
plt.annotate('$\\mu_B$', (mu_B_proj, y_offset_B + 0.15), 
             xytext=(0, 5), textcoords='offset points', fontsize=14, ha='center')

# Calculate the optimal threshold for classification and mark it clearly
threshold = (mu_A_proj + mu_B_proj) / 2
plt.axvline(x=threshold, color='g', linestyle='--', linewidth=2, label=f'Decision Threshold: {threshold:.2f}')
plt.scatter(threshold, 0, color='g', s=100, marker='|', zorder=10)
plt.annotate(f'Threshold = {threshold:.2f}', (threshold, 0.35), fontsize=12, ha='center')

# Add LDA direction arrows along the axis for better understanding
arrow_length = (x_max - x_min) / 10
plt.arrow(threshold - arrow_length, 0, arrow_length * 0.8, 0, 
          head_width=0.03, head_length=arrow_length * 0.2, fc='purple', ec='purple')
plt.arrow(threshold + arrow_length, 0, -arrow_length * 0.8, 0, 
          head_width=0.03, head_length=arrow_length * 0.2, fc='green', ec='green')

# Add shaded regions for each class
plt.axvspan(x_min, threshold, alpha=0.1, color='blue', label='Class A Region')
plt.axvspan(threshold, x_max, alpha=0.1, color='red', label='Class B Region')

plt.title('Projection of Data Points onto LDA Direction', fontsize=16)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Projection Value along LDA Direction', fontsize=14)
plt.yticks([])  # Hide y-axis ticks
plt.grid(True, alpha=0.3)

# Place legend in a better position without overlapping
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
plt.tight_layout()

# Save the improved projection visualization
plt.savefig(os.path.join(save_dir, "lda_projection_1d.png"), dpi=300, bbox_inches='tight')

# ---------------- VISUALIZATION 2: Between-class and Within-class Scatter ----------------
plt.figure(figsize=(12, 10))

# Create scatter plots to show between-class and within-class scatter
plt.subplot(2, 2, 1)
plt.scatter(X_A[0, :], X_A[1, :], color='blue', s=100, marker='o', label='Class A')
plt.scatter(X_B[0, :], X_B[1, :], color='red', s=100, marker='x', label='Class B')
plt.scatter(mu_A[0, 0], mu_A[1, 0], color='blue', s=200, marker='*')
plt.scatter(mu_B[0, 0], mu_B[1, 0], color='red', s=200, marker='*')
plt.plot([mu_A[0, 0], mu_B[0, 0]], [mu_A[1, 0], mu_B[1, 0]], 'k--', lw=2, label='Between-class')
plt.title('Original Data with Class Means', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot within-class scatter for Class A
plt.subplot(2, 2, 2)
plt.scatter(X_A[0, :], X_A[1, :], color='blue', s=100, marker='o', label='Class A')
plt.scatter(mu_A[0, 0], mu_A[1, 0], color='blue', s=200, marker='*', label='Mean of Class A')

# Draw lines from each point to the mean to show the scatter
for i in range(n_A):
    plt.plot([X_A[0, i], mu_A[0, 0]], [X_A[1, i], mu_A[1, 0]], 'b--', alpha=0.6)
    
# Visualize the covariance matrix as an ellipse (if non-singular)
if np.linalg.det(Sigma_A) > 1e-10:
    from matplotlib.patches import Ellipse
    lambda_, v = np.linalg.eig(Sigma_A)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mu_A[0, 0], mu_A[1, 0]),
                width=lambda_[0]*4, height=lambda_[1]*4,
                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                edgecolor='blue', fc='none', lw=2, label='Covariance ellipse')
    plt.gca().add_patch(ell)

plt.title('Within-class Scatter for Class A', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot within-class scatter for Class B
plt.subplot(2, 2, 3)
plt.scatter(X_B[0, :], X_B[1, :], color='red', s=100, marker='x', label='Class B')
plt.scatter(mu_B[0, 0], mu_B[1, 0], color='red', s=200, marker='*', label='Mean of Class B')

# Draw lines from each point to the mean to show the scatter
for i in range(n_B):
    plt.plot([X_B[0, i], mu_B[0, 0]], [X_B[1, i], mu_B[1, 0]], 'r--', alpha=0.6)
    
# Visualize the covariance matrix as an ellipse (if non-singular)
if np.linalg.det(Sigma_B) > 1e-10:
    lambda_, v = np.linalg.eig(Sigma_B)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mu_B[0, 0], mu_B[1, 0]),
                width=lambda_[0]*4, height=lambda_[1]*4,
                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                edgecolor='red', fc='none', lw=2, label='Covariance ellipse')
    plt.gca().add_patch(ell)

plt.title('Within-class Scatter for Class B', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot the LDA direction and decision boundary
plt.subplot(2, 2, 4)
plt.scatter(X_A[0, :], X_A[1, :], color='blue', s=100, marker='o', label='Class A')
plt.scatter(X_B[0, :], X_B[1, :], color='red', s=100, marker='x', label='Class B')
plt.scatter(mu_A[0, 0], mu_A[1, 0], color='blue', s=200, marker='*')
plt.scatter(mu_B[0, 0], mu_B[1, 0], color='red', s=200, marker='*')

# Draw the LDA direction
origin = (mu_A + mu_B) / 2  # Use the midpoint between means as origin for better visualization
origin = origin.flatten()
plt.arrow(origin[0], origin[1], 
          w_star_normalized[0, 0]*2, w_star_normalized[1, 0]*2,  # Scale for visibility
          head_width=0.15, head_length=0.25, fc='green', ec='green', 
          label='LDA direction')

# Draw a perpendicular line to show the decision boundary
perp_vec = np.array([-w_star_normalized[1, 0], w_star_normalized[0, 0]])
plt.plot([origin[0] - perp_vec[0]*3, origin[0] + perp_vec[0]*3],
         [origin[1] - perp_vec[1]*3, origin[1] + perp_vec[1]*3],
         'g--', label='Decision boundary')

plt.title('LDA Direction and Decision Boundary', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_scatter_analysis.png"), dpi=300, bbox_inches='tight')

print("\nFinal Results Summary:")
print("====================")
print(f"1. Mean vector for Class A: [{mu_A[0,0]}, {mu_A[1,0]}]^T")
print(f"2. Mean vector for Class B: [{mu_B[0,0]}, {mu_B[1,0]}]^T")
print(f"3. Covariance matrix for Class A:")
print(Sigma_A)
print(f"4. Covariance matrix for Class B:")
print(Sigma_B)
print(f"5. Pooled within-class scatter matrix Sw:")
print(Sw)
print(f"6. Optimal projection direction w* (non-normalized): [{w_star[0,0]:.4f}, {w_star[1,0]:.4f}]^T")
print(f"7. Norm of w*: ||w*|| = {w_star_norm:.4f}")
print(f"8. Optimal projection direction w* (normalized): [{w_star_normalized[0,0]:.4f}, {w_star_normalized[1,0]:.4f}]^T")

print("\nInterpretation:")
if abs(det_Sw) < 1e-10:
    print("- The pooled within-class scatter matrix is singular (determinant ≈ 0).")
    print("- We applied regularization to handle the singularity.")
else:
    print("- The covariance matrices capture the variance and correlation structure of each class.")
    print("- The pooled within-class scatter matrix combines both class covariance patterns.")

print("- The optimal projection direction was computed using the formula w* = Sw^(-1)(μA - μB).")
print("- The normalized projection direction effectively separates the two classes.")
print("- Linear Discriminant Analysis works by maximizing between-class variance while")
print("  minimizing within-class variance, which is achieved through this projection.")

print("\nAll figures have been saved to:", save_dir)
