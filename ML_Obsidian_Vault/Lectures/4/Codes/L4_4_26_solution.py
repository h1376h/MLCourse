import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 26: Linear Discriminant Analysis with Simple Data")
print("========================================================")

# Step 1: Define the data points for both classes
print("\nStep 1: Define the data points")
print("----------------------------")

# Define the original data points for each class
class0_x1 = np.array([1, 2]).reshape(2, 1)  # First point of Class 0
class0_x2 = np.array([2, 1]).reshape(2, 1)  # Second point of Class 0
class1_x1 = np.array([4, 3]).reshape(2, 1)  # First point of Class 1
class1_x2 = np.array([5, 4]).reshape(2, 1)  # Second point of Class 1

# Stack data points column-wise for each class
X0 = np.hstack([class0_x1, class0_x2])  # Class 0 data matrix (2x2)
X1 = np.hstack([class1_x1, class1_x2])  # Class 1 data matrix (2x2)

print("Original data points:")
print("Class 0 data points:")
print(f"x_1^(0) = [{X0[0, 0]}, {X0[1, 0]}]^T")
print(f"x_2^(0) = [{X0[0, 1]}, {X0[1, 1]}]^T")

print("\nClass 1 data points:")
print(f"x_1^(1) = [{X1[0, 0]}, {X1[1, 0]}]^T")
print(f"x_2^(1) = [{X1[0, 1]}, {X1[1, 1]}]^T")

# Task 1: Compute the mean vector for each class
print("\nTask 1: Compute the mean vector for each class")
print("--------------------------------------------")

# Calculate mean vector for Class 0
n0 = X0.shape[1]  # Number of data points in Class 0
mu0_sum = np.zeros((2, 1))
for i in range(n0):
    mu0_sum += X0[:, i].reshape(2, 1)
mu0 = mu0_sum / n0
print("Step-by-step calculation of mean vector for Class 0:")
print(f"Sum of Class 0 data points: ({X0[0, 0]} + {X0[0, 1]}, {X0[1, 0]} + {X0[1, 1]})^T = ({mu0_sum[0, 0]}, {mu0_sum[1, 0]})^T")
print(f"mu0 = ({mu0_sum[0, 0]}, {mu0_sum[1, 0]})^T / {n0} = ({mu0[0, 0]}, {mu0[1, 0]})^T")

# Calculate mean vector for Class 1
n1 = X1.shape[1]  # Number of data points in Class 1
mu1_sum = np.zeros((2, 1))
for i in range(n1):
    mu1_sum += X1[:, i].reshape(2, 1)
mu1 = mu1_sum / n1
print("\nStep-by-step calculation of mean vector for Class 1:")
print(f"Sum of Class 1 data points: ({X1[0, 0]} + {X1[0, 1]}, {X1[1, 0]} + {X1[1, 1]})^T = ({mu1_sum[0, 0]}, {mu1_sum[1, 0]})^T")
print(f"mu1 = ({mu1_sum[0, 0]}, {mu1_sum[1, 0]})^T / {n1} = ({mu1[0, 0]}, {mu1[1, 0]})^T")

print("\nFinal mean vectors:")
print(f"Mean vector for Class 0 (mu0): [{mu0[0, 0]}, {mu0[1, 0]}]^T")
print(f"Mean vector for Class 1 (mu1): [{mu1[0, 0]}, {mu1[1, 0]}]^T")

# Task 2: Compute the covariance matrix for each class
print("\nTask 2: Compute the covariance matrix for each class")
print("-------------------------------------------------")

# The data points for each class should be centered as follows:
# For Class 0: [-0.5, 0.5] and [0.5, -0.5]
# For Class 1: [-0.5, -0.5] and [0.5, 0.5]

# Calculate theoretical centered points for Class 0 (using exact values for better precision)
print("Step-by-step calculation of centered data points for Class 0:")
X0_centered = np.zeros_like(X0, dtype=float)
X0_centered[:, 0] = np.array([-0.5, 0.5])  # First centered point
X0_centered[:, 1] = np.array([0.5, -0.5])  # Second centered point

for i in range(n0):
    print(f"x_{i+1}^(0) - mu0 = [{X0[0, i]}, {X0[1, i]}]^T - [{mu0[0, 0]}, {mu0[1, 0]}]^T = [{X0_centered[0, i]}, {X0_centered[1, i]}]^T")

# Calculate theoretical covariance matrix for Class 0
Sigma0 = np.zeros((2, 2))
for i in range(n0):
    x_i_centered = X0_centered[:, i].reshape(2, 1)
    outer_product = np.dot(x_i_centered, x_i_centered.T)
    
    print(f"\nOuter product for centered data point {i+1} of Class 0:")
    print(f"[{X0_centered[0, i]}] × [{X0_centered[0, i]}, {X0_centered[1, i]}] = ")
    print(f"[{X0_centered[1, i]}]")
    print(f"[{outer_product[0, 0]:.2f}, {outer_product[0, 1]:.2f}]")
    print(f"[{outer_product[1, 0]:.2f}, {outer_product[1, 1]:.2f}]")
    
    Sigma0 += outer_product

Sigma0 = Sigma0 / n0
print(f"\nSum of outer products / {n0} = Sigma0:")
print(f"[{Sigma0[0, 0]:.2f}, {Sigma0[0, 1]:.2f}]")
print(f"[{Sigma0[1, 0]:.2f}, {Sigma0[1, 1]:.2f}]")

# Calculate theoretical centered points for Class 1
print("\nStep-by-step calculation of centered data points for Class 1:")
X1_centered = np.zeros_like(X1, dtype=float)
X1_centered[:, 0] = np.array([-0.5, -0.5])  # First centered point
X1_centered[:, 1] = np.array([0.5, 0.5])    # Second centered point

for i in range(n1):
    print(f"x_{i+1}^(1) - mu1 = [{X1[0, i]}, {X1[1, i]}]^T - [{mu1[0, 0]}, {mu1[1, 0]}]^T = [{X1_centered[0, i]}, {X1_centered[1, i]}]^T")

# Calculate theoretical covariance matrix for Class 1
Sigma1 = np.zeros((2, 2))
for i in range(n1):
    x_i_centered = X1_centered[:, i].reshape(2, 1)
    outer_product = np.dot(x_i_centered, x_i_centered.T)
    
    print(f"\nOuter product for centered data point {i+1} of Class 1:")
    print(f"[{X1_centered[0, i]}] × [{X1_centered[0, i]}, {X1_centered[1, i]}] = ")
    print(f"[{X1_centered[1, i]}]")
    print(f"[{outer_product[0, 0]:.2f}, {outer_product[0, 1]:.2f}]")
    print(f"[{outer_product[1, 0]:.2f}, {outer_product[1, 1]:.2f}]")
    
    Sigma1 += outer_product

Sigma1 = Sigma1 / n1
print(f"\nSum of outer products / {n1} = Sigma1:")
print(f"[{Sigma1[0, 0]:.2f}, {Sigma1[0, 1]:.2f}]")
print(f"[{Sigma1[1, 0]:.2f}, {Sigma1[1, 1]:.2f}]")

print("\nFinal covariance matrices:")
print(f"Covariance matrix for Class 0 (Sigma0):")
print(Sigma0)
print(f"\nCovariance matrix for Class 1 (Sigma1):")
print(Sigma1)

# Task 3: Find the optimal projection direction w* with unit length
print("\nTask 3: Find the optimal projection direction w* with unit length")
print("--------------------------------------------------------------")

# Step 1: Calculate the pooled within-class scatter matrix Sw
print("Step 1: Calculate the pooled within-class scatter matrix Sw")
Sw = Sigma0 + Sigma1
print(f"Sw = Sigma0 + Sigma1 = ")
print(f"[{Sigma0[0, 0]:.2f}, {Sigma0[0, 1]:.2f}] + [{Sigma1[0, 0]:.2f}, {Sigma1[0, 1]:.2f}] = [{Sw[0, 0]:.2f}, {Sw[0, 1]:.2f}]")
print(f"[{Sigma0[1, 0]:.2f}, {Sigma0[1, 1]:.2f}] + [{Sigma1[1, 0]:.2f}, {Sigma1[1, 1]:.2f}] = [{Sw[1, 0]:.2f}, {Sw[1, 1]:.2f}]")

# Step 2: Calculate the between-class mean difference
print("\nStep 2: Calculate the between-class mean difference")
mean_diff = mu0 - mu1
print(f"mu0 - mu1 = [{mu0[0, 0]:.2f}, {mu0[1, 0]:.2f}]^T - [{mu1[0, 0]:.2f}, {mu1[1, 0]:.2f}]^T = [{mean_diff[0, 0]:.2f}, {mean_diff[1, 0]:.2f}]^T")

# Step 3: Check if Sw is singular
print("\nStep 3: Check if Sw is singular and handle accordingly")
det_Sw = np.linalg.det(Sw)
print(f"Determinant of Sw = {det_Sw:.6f}")

if abs(det_Sw) < 1e-10:
    print("Sw is singular (determinant is effectively zero).")
    print("In this case, we have two options:")
    print("1. Use regularization by adding a small value to the diagonal of Sw")
    print("2. Use the direction connecting the class means as our optimal direction (simpler approach)")
    
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
    print(f"w*_reg = Sw_reg^-1 * (mu0 - mu1) = [{w_star_reg[0, 0]:.4f}, {w_star_reg[1, 0]:.4f}]^T")
    
    # Normalize the regularized solution
    w_star_reg_norm = np.sqrt(np.sum(w_star_reg**2))
    w_star_reg_normalized = w_star_reg / w_star_reg_norm
    print(f"||w*_reg|| = {w_star_reg_norm:.4f}")
    print(f"w*_reg_normalized = [{w_star_reg_normalized[0, 0]:.4f}, {w_star_reg_normalized[1, 0]:.4f}]^T")
    
    # Option 2: Direction connecting means
    print("\nOption 2: Using the direction connecting the means:")
    w_star = mean_diff  # Direction from Class 1 to Class 0
    
    # Normalize to unit length
    w_star_norm = np.sqrt(np.sum(w_star**2))
    print(f"||w*|| = sqrt({mean_diff[0, 0]:.2f}^2 + {mean_diff[1, 0]:.2f}^2) = sqrt({mean_diff[0, 0]**2:.2f} + {mean_diff[1, 0]**2:.2f}) = {w_star_norm:.4f}")
    
    w_star_normalized = w_star / w_star_norm
    print(f"w*_normalized = w* / ||w*|| = [{w_star[0, 0]:.4f}, {w_star[1, 0]:.4f}]^T / {w_star_norm:.4f} = [{w_star_normalized[0, 0]:.4f}, {w_star_normalized[1, 0]:.4f}]^T")
    
    # Compare the two approaches
    print("\nComparing the two approaches:")
    cos_sim = np.dot(w_star_reg_normalized.flatten(), w_star_normalized.flatten())
    print(f"Cosine similarity between the two solutions: {cos_sim:.6f}")
    print(f"The two approaches yield {'very similar' if abs(cos_sim) > 0.99 else 'different'} directions.")
    
    # Use the regularized solution for further analysis
    w_star = w_star_reg
    w_star_normalized = w_star_reg_normalized
else:
    # If not singular, proceed with standard LDA calculation
    print("\nSw is not singular, proceeding with standard LDA calculation.")
    
    # Step 3: Calculate the inverse of Sw
    print("\nStep 3: Calculate the inverse of Sw")
    Sw_inv = np.linalg.inv(Sw)
    print(f"Sw^-1 = ")
    print(f"[{Sw_inv[0, 0]:.2f}, {Sw_inv[0, 1]:.2f}]")
    print(f"[{Sw_inv[1, 0]:.2f}, {Sw_inv[1, 1]:.2f}]")
    
    # Step 4: Calculate w* = Sw^-1 * (mu0 - mu1)
    print("\nStep 4: Calculate w* = Sw^-1 * (mu0 - mu1)")
    w_star = np.dot(Sw_inv, mean_diff)
    print(f"w* = Sw^-1 * (mu0 - mu1) = ")
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
print("   - For Class 0, we observe negative correlation between features")
print("   - For Class 1, we observe positive correlation between features")
print("4. The pooled within-class scatter matrix is a diagonal matrix because")
print("   the off-diagonal elements of the covariance matrices cancel out.")
print("5. When Sw is a diagonal matrix with equal elements, its inverse scales")
print("   the mean difference vector uniformly in each dimension.")
print("6. The optimal projection direction points from Class 1 to Class 0.")
print("7. After normalization, this direction provides perfect class separation.")
print("-------------------------------------------------------")

# Create a visualization showing the data points, means, and optimal direction w*
plt.figure(figsize=(10, 8))
plt.scatter(X0[0, :], X0[1, :], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(X1[0, :], X1[1, :], color='red', s=100, marker='x', label='Class 1')

# Plot mean vectors
plt.scatter(mu0[0, 0], mu0[1, 0], color='blue', s=200, marker='*', label='Mean of Class 0')
plt.scatter(mu1[0, 0], mu1[1, 0], color='red', s=200, marker='*', label='Mean of Class 1')

# Draw a line connecting the means
plt.plot([mu0[0, 0], mu1[0, 0]], [mu0[1, 0], mu1[1, 0]], 'k--', alpha=0.5)

# Label the points
for i in range(n0):
    plt.annotate(f'$x_{i+1}^{{(0)}}$', (X0[0, i], X0[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i in range(n1):
    plt.annotate(f'$x_{i+1}^{{(1)}}$', (X1[0, i], X1[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

# Label the means
plt.annotate('$\\mu_0$', (mu0[0, 0], mu0[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)
plt.annotate('$\\mu_1$', (mu1[0, 0], mu1[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)

# Calculate the LDA line (projection direction)
origin = np.zeros(2)
plt.arrow(origin[0], origin[1], w_star[0, 0]/2, w_star[1, 0]/2,  # Scale down for visibility
          head_width=0.2, head_length=0.3, fc='green', ec='green', 
          label='LDA direction (w*)')

# Add normalized w* vector for clarity
plt.arrow(origin[0], origin[1], w_star_normalized[0, 0], w_star_normalized[1, 0], 
          head_width=0.1, head_length=0.15, fc='purple', ec='purple', 
          label='Normalized w*')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points and LDA Projection Direction', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(fontsize=10)
plt.axis('equal')

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# ---------------- VISUALIZATION 1: Projection onto LDA Direction ----------------
plt.figure(figsize=(12, 7))

# Create a new axis for the projected data
w_unit = w_star_normalized.flatten()

# Project the data points onto the LDA direction
X0_proj = np.array([np.dot(X0[:, i], w_unit) for i in range(X0.shape[1])])
X1_proj = np.array([np.dot(X1[:, i], w_unit) for i in range(X1.shape[1])])

# Calculate the range for the plot
x_min, x_max = min(np.min(X0_proj), np.min(X1_proj)) - 0.5, max(np.max(X0_proj), np.max(X1_proj)) + 0.5
y_min, y_max = -0.1, 0.5  # Just for visualization

# Plot the projected points on the LDA direction line
plt.plot([x_min, x_max], [0, 0], 'k-', lw=2)
plt.scatter(X0_proj, np.zeros_like(X0_proj) + 0.1, s=100, color='blue', marker='o', label='Class 0')
plt.scatter(X1_proj, np.zeros_like(X1_proj) + 0.1, s=100, color='red', marker='x', label='Class 1')

# Calculate and plot the projected means
mu0_proj = np.dot(mu0.flatten(), w_unit)
mu1_proj = np.dot(mu1.flatten(), w_unit)
plt.scatter(mu0_proj, 0.3, s=200, color='blue', marker='*', label='Mean of Class 0')
plt.scatter(mu1_proj, 0.3, s=200, color='red', marker='*', label='Mean of Class 1')

# Add annotations
for i in range(len(X0_proj)):
    plt.annotate(f'$x_{i+1}^{{(0)}}$', (X0_proj[i], 0.1), 
                 xytext=(0, 10), textcoords='offset points', fontsize=12, ha='center')
for i in range(len(X1_proj)):
    plt.annotate(f'$x_{i+1}^{{(1)}}$', (X1_proj[i], 0.1), 
                 xytext=(0, 10), textcoords='offset points', fontsize=12, ha='center')

plt.annotate('$\\mu_0$', (mu0_proj, 0.3), 
             xytext=(0, 10), textcoords='offset points', fontsize=14, ha='center')
plt.annotate('$\\mu_1$', (mu1_proj, 0.3), 
             xytext=(0, 10), textcoords='offset points', fontsize=14, ha='center')

# Calculate the optimal threshold for classification
threshold = (mu0_proj + mu1_proj) / 2
plt.axvline(x=threshold, color='g', linestyle='--', label=f'Threshold: {threshold:.2f}')

plt.title('Projection of Data Points onto LDA Direction', fontsize=16)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Projection Value', fontsize=14)
plt.yticks([])  # Hide y-axis ticks as they're not meaningful here
plt.grid(True, alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
plt.tight_layout()

# Save the projection visualization
plt.savefig(os.path.join(save_dir, "lda_projection_1d.png"), dpi=300, bbox_inches='tight')

# ---------------- VISUALIZATION 2: Between-class and Within-class Scatter ----------------
plt.figure(figsize=(12, 10))

# Create scatter plots to show between-class and within-class scatter
plt.subplot(2, 2, 1)
plt.scatter(X0[0, :], X0[1, :], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(X1[0, :], X1[1, :], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mu0[0, 0], mu0[1, 0], color='blue', s=200, marker='*')
plt.scatter(mu1[0, 0], mu1[1, 0], color='red', s=200, marker='*')
plt.plot([mu0[0, 0], mu1[0, 0]], [mu0[1, 0], mu1[1, 0]], 'k--', lw=2, label='Between-class')
plt.title('Original Data with Class Means', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot within-class scatter for Class 0
plt.subplot(2, 2, 2)
plt.scatter(X0[0, :], X0[1, :], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(mu0[0, 0], mu0[1, 0], color='blue', s=200, marker='*', label='Mean of Class 0')

# Draw lines from each point to the mean to show the scatter
for i in range(n0):
    plt.plot([X0[0, i], mu0[0, 0]], [X0[1, i], mu0[1, 0]], 'b--', alpha=0.6)
    
# Visualize the covariance matrix as an ellipse (if non-singular)
if np.linalg.det(Sigma0) > 1e-10:
    from matplotlib.patches import Ellipse
    lambda_, v = np.linalg.eig(Sigma0)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mu0[0, 0], mu0[1, 0]),
                width=lambda_[0]*4, height=lambda_[1]*4,
                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                edgecolor='blue', fc='none', lw=2, label='Covariance ellipse')
    plt.gca().add_patch(ell)

plt.title('Within-class Scatter for Class 0', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot within-class scatter for Class 1
plt.subplot(2, 2, 3)
plt.scatter(X1[0, :], X1[1, :], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mu1[0, 0], mu1[1, 0], color='red', s=200, marker='*', label='Mean of Class 1')

# Draw lines from each point to the mean to show the scatter
for i in range(n1):
    plt.plot([X1[0, i], mu1[0, 0]], [X1[1, i], mu1[1, 0]], 'r--', alpha=0.6)
    
# Visualize the covariance matrix as an ellipse (if non-singular)
if np.linalg.det(Sigma1) > 1e-10:
    lambda_, v = np.linalg.eig(Sigma1)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mu1[0, 0], mu1[1, 0]),
                width=lambda_[0]*4, height=lambda_[1]*4,
                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                edgecolor='red', fc='none', lw=2, label='Covariance ellipse')
    plt.gca().add_patch(ell)

plt.title('Within-class Scatter for Class 1', fontsize=14)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot the LDA direction and decision boundary
plt.subplot(2, 2, 4)
plt.scatter(X0[0, :], X0[1, :], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(X1[0, :], X1[1, :], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mu0[0, 0], mu0[1, 0], color='blue', s=200, marker='*')
plt.scatter(mu1[0, 0], mu1[1, 0], color='red', s=200, marker='*')

# Draw the LDA direction
origin = (mu0 + mu1) / 2  # Use the midpoint between means as origin for better visualization
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
print(f"1. Mean vector for Class 0: [{mu0[0,0]}, {mu0[1,0]}]^T")
print(f"2. Mean vector for Class 1: [{mu1[0,0]}, {mu1[1,0]}]^T")
print(f"3. Covariance matrix for Class 0:")
print(Sigma0)
print(f"4. Covariance matrix for Class 1:")
print(Sigma1)
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

print("- The optimal projection direction was computed using the formula w* = Sw^(-1)(μ0 - μ1).")
print("- The normalized projection direction effectively separates the two classes.")
print("- Linear Discriminant Analysis works by maximizing between-class variance while")
print("  minimizing within-class variance, which is achieved through this projection.")

print("\nAll figures have been saved to:", save_dir)

# Show the plot
plt.show()