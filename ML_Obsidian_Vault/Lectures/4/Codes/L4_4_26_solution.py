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

# Given data points as column vectors
X0 = np.array([[1, 2], [2, 1]]).T  # Class 0
X1 = np.array([[4, 5], [3, 4]]).T  # Class 1

print("Class 0 data points:")
for i in range(X0.shape[1]):
    print(f"x_{i+1}^(0) = [{X0[0, i]}, {X0[1, i]}]^T")

print("\nClass 1 data points:")
for i in range(X1.shape[1]):
    print(f"x_{i+1}^(1) = [{X1[0, i]}, {X1[1, i]}]^T")

# Step 2: Compute the mean vectors for each class
print("\nStep 2: Compute the mean vectors for each class")
print("---------------------------------------------")

mu0 = np.mean(X0, axis=1, keepdims=True)  # Mean of class 0
mu1 = np.mean(X1, axis=1, keepdims=True)  # Mean of class 1

print(f"Mean vector for Class 0 (mu0):\n{mu0}")
print(f"Mean vector for Class 1 (mu1):\n{mu1}")

# Step 3: Compute the covariance matrices for each class
print("\nStep 3: Compute the covariance matrices for each class")
print("---------------------------------------------------")

# For Class 0
X0_centered = X0 - mu0
Sigma0 = np.dot(X0_centered, X0_centered.T) / X0.shape[1]

# For Class 1
X1_centered = X1 - mu1
Sigma1 = np.dot(X1_centered, X1_centered.T) / X1.shape[1]

print(f"Covariance matrix for Class 0 (Sigma0):\n{Sigma0}")
print(f"Covariance matrix for Class 1 (Sigma1):\n{Sigma1}")

# Step 4: Compute the pooled within-class scatter matrix
print("\nStep 4: Compute the pooled within-class scatter matrix")
print("---------------------------------------------------")

Sw = Sigma0 + Sigma1
print(f"Pooled within-class scatter matrix (Sw):\n{Sw}")

# Step 5: Compute the between-class mean difference
print("\nStep 5: Compute the between-class mean difference")
print("-----------------------------------------------")

mean_diff = mu0 - mu1
print(f"Between-class mean difference (mu0 - mu1):\n{mean_diff}")

# Step 6: Find the optimal projection direction w*
print("\nStep 6: Find the optimal projection direction w*")
print("---------------------------------------------")

# Compute the inverse of Sw
Sw_inv = np.linalg.inv(Sw)
print(f"Inverse of Sw:\n{Sw_inv}")

# Calculate w* = Sw^(-1) * (mu0 - mu1)
w_star = np.dot(Sw_inv, mean_diff)
print(f"Optimal projection direction w* (before normalization):\n{w_star}")

# Normalize w* to unit length
w_star_normalized = w_star / np.linalg.norm(w_star)
print(f"Optimal projection direction w* (normalized to unit length):\n{w_star_normalized}")

# Step 7: Visualize the data and the LDA projection
print("\nStep 7: Visualize the data and the LDA projection")
print("----------------------------------------------")

# Plot the original data points
plt.figure(figsize=(10, 8))
plt.scatter(X0[0, :], X0[1, :], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(X1[0, :], X1[1, :], color='red', s=100, marker='x', label='Class 1')

# Plot mean vectors
plt.scatter(mu0[0, 0], mu0[1, 0], color='blue', s=200, marker='*', label='Mean of Class 0')
plt.scatter(mu1[0, 0], mu1[1, 0], color='red', s=200, marker='*', label='Mean of Class 1')

# Draw a line connecting the means
plt.plot([mu0[0, 0], mu1[0, 0]], [mu0[1, 0], mu1[1, 0]], 'k--', alpha=0.5)

# Label the points
for i in range(X0.shape[1]):
    plt.annotate(f'$x_{i+1}^{{(0)}}$', (X0[0, i], X0[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i in range(X1.shape[1]):
    plt.annotate(f'$x_{i+1}^{{(1)}}$', (X1[0, i], X1[1, i]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

# Label the means
plt.annotate('$\\mu_0$', (mu0[0, 0], mu0[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)
plt.annotate('$\\mu_1$', (mu1[0, 0], mu1[1, 0]), 
             xytext=(10, 5), textcoords='offset points', fontsize=14)

# Calculate the LDA line (projection direction)
origin = np.zeros(2)
plt.arrow(origin[0], origin[1], w_star[0, 0], w_star[1, 0], 
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
plt.legend(fontsize=12)
plt.axis('equal')

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# Step 8: Project the data onto the LDA direction
print("\nStep 8: Project the data onto the LDA direction")
print("---------------------------------------------")

# Project all points onto w*
X0_proj = np.dot(w_star_normalized.T, X0)
X1_proj = np.dot(w_star_normalized.T, X1)
mu0_proj = np.dot(w_star_normalized.T, mu0)
mu1_proj = np.dot(w_star_normalized.T, mu1)

print(f"Projected Class 0 points onto w*: {X0_proj.flatten()}")
print(f"Projected Class 1 points onto w*: {X1_proj.flatten()}")
print(f"Projected mean of Class 0 onto w*: {mu0_proj.flatten()[0]:.4f}")
print(f"Projected mean of Class 1 onto w*: {mu1_proj.flatten()[0]:.4f}")

# Step 9: Visualize the 1D projections
print("\nStep 9: Visualize the 1D projections")
print("----------------------------------")

plt.figure(figsize=(12, 6))

# Plot the projections on a 1D line
plt.scatter(X0_proj.flatten(), np.zeros_like(X0_proj.flatten()), 
            color='blue', s=100, marker='o', label='Class 0 Projections')
plt.scatter(X1_proj.flatten(), np.zeros_like(X1_proj.flatten()), 
            color='red', s=100, marker='x', label='Class 1 Projections')

# Plot the projected means
plt.scatter(mu0_proj, 0, color='blue', s=200, marker='*', label='Projected Mean of Class 0')
plt.scatter(mu1_proj, 0, color='red', s=200, marker='*', label='Projected Mean of Class 1')

# Draw vertical lines for each point
for i, proj in enumerate(X0_proj.flatten()):
    plt.axvline(x=proj, ymin=0.25, ymax=0.75, color='blue', linestyle='-', alpha=0.3)
    plt.text(proj, 0.05, f'$x_{i+1}^{{(0)}}$', ha='center', fontsize=12)
    
for i, proj in enumerate(X1_proj.flatten()):
    plt.axvline(x=proj, ymin=0.25, ymax=0.75, color='red', linestyle='-', alpha=0.3)
    plt.text(proj, -0.05, f'$x_{i+1}^{{(1)}}$', ha='center', fontsize=12)

# Add labels for projected means
plt.text(mu0_proj, 0.1, '$\\mu_0$ proj', ha='center', fontsize=14)
plt.text(mu1_proj, -0.1, '$\\mu_1$ proj', ha='center', fontsize=14)

# Calculate and display the optimal threshold for classification
threshold = (mu0_proj + mu1_proj) / 2
plt.axvline(x=threshold, color='green', linestyle='--', 
            label=f'Decision Threshold ({threshold.flatten()[0]:.4f})')

plt.xlabel('Projection onto w*', fontsize=14)
plt.title('1D Projections of Data Points onto Fisher\'s Linear Discriminant', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(-0.5, 0.5)

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_1d_projection.png"), dpi=300, bbox_inches='tight')

# Step 10: Calculate the Fisher's criterion value
print("\nStep 10: Calculate the Fisher's criterion value")
print("---------------------------------------------")

# Fisher's criterion: S(w) = (w^T(mu0 - mu1))^2 / w^T(Sigma0 + Sigma1)w
numerator = (np.dot(w_star_normalized.T, mu0 - mu1))**2
denominator = np.dot(np.dot(w_star_normalized.T, Sw), w_star_normalized)
fisher_criterion = numerator / denominator

print(f"Between-class variance (numerator): {numerator.flatten()[0]:.4f}")
print(f"Within-class variance (denominator): {denominator.flatten()[0]:.4f}")
print(f"Fisher's criterion value: {fisher_criterion.flatten()[0]:.4f}")

# Final summary of results
print("\nFinal Results Summary:")
print("====================")
print(f"1. Mean vector for Class 0: [{mu0[0,0]:.4f}, {mu0[1,0]:.4f}]^T")
print(f"2. Mean vector for Class 1: [{mu1[0,0]:.4f}, {mu1[1,0]:.4f}]^T")
print(f"3. Covariance matrix for Class 0:\n{Sigma0}")
print(f"4. Covariance matrix for Class 1:\n{Sigma1}")
print(f"5. Optimal projection direction w* (normalized): [{w_star_normalized[0,0]:.4f}, {w_star_normalized[1,0]:.4f}]^T")
print(f"6. Fisher's criterion value: {fisher_criterion.flatten()[0]:.4f}")
print("\nAll figures have been saved to:", save_dir)