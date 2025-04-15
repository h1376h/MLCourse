import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D

print("\n=== LINEAR TRANSFORMATION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Analysis")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Feature Transformation for Student Performance Data
print("Example 1: Feature Transformation for Student Performance Data")

# Dataset with test scores: Midterm 1 (X1), Midterm 2 (X2), Final (X3)
data = np.array([
    [75, 82, 78],
    [92, 88, 95],
    [68, 73, 71],
    [85, 80, 82],
    [79, 85, 81]
])

print("Dataset with 3 variables: Midterm 1 (X₁), Midterm 2 (X₂), and Final Exam (X₃) scores")
print("\n| Student | Midterm 1 (X₁) | Midterm 2 (X₂) | Final (X₃) |")
print("|---------|---------------|---------------|-----------|")
for i, row in enumerate(data):
    print(f"| {i+1}       | {row[0]:<13} | {row[1]:<13} | {row[2]:<9} |")

# Step 1: Calculate the mean vector and covariance matrix of the original data
print("\nStep 1: Calculate the mean vector and covariance matrix of the original data")

mean_vector = np.mean(data, axis=0)
print(f"\nMean vector μ = [{mean_vector[0]:.2f}, {mean_vector[1]:.2f}, {mean_vector[2]:.2f}]")

# Number of samples and variables
n = data.shape[0]
p = data.shape[1]

# Compute sample covariance matrix
deviations = data - mean_vector
cov_matrix = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        cov_matrix[i, j] = np.sum(deviations[:, i] * deviations[:, j]) / (n - 1)

print("\nSample covariance matrix:")
print("\n┌" + "─" * 35 + "┐")
for i in range(p):
    print("│ ", end="")
    for j in range(p):
        print(f"{cov_matrix[i, j]:10.2f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 2: Define the linear transformation matrix A and vector b
print("\nStep 2: Define the linear transformation Y = AX + b")

# Transformation matrix A - Example: 
# - Y1 = Average of midterms (0.5*X1 + 0.5*X2)
# - Y2 = Weighted average of all exams, with more weight on the final (0.3*X1 + 0.3*X2 + 0.4*X3)
# - Y3 = Difference between Final and average of midterms (X3 - 0.5*X1 - 0.5*X2)
A = np.array([
    [0.5, 0.5, 0.0],
    [0.3, 0.3, 0.4],
    [-0.5, -0.5, 1.0]
])

# Translation vector b
b = np.array([0, 0, 0])

print("\nTransformation matrix A:")
print("\n┌" + "─" * 35 + "┐")
for i in range(A.shape[0]):
    print("│ ", end="")
    for j in range(A.shape[1]):
        print(f"{A[i, j]:10.2f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

print("\nTranslation vector b:", b)

print("\nThis transformation produces:")
print("- Y₁ = 0.5·X₁ + 0.5·X₂ (Average of midterm scores)")
print("- Y₂ = 0.3·X₁ + 0.3·X₂ + 0.4·X₃ (Weighted average of all exams)")
print("- Y₃ = -0.5·X₁ - 0.5·X₂ + 1.0·X₃ (Difference between final and average midterms)")

# Step 3: Apply the transformation to each data point
print("\nStep 3: Apply the transformation to each data point")

transformed_data = np.zeros_like(data)
for i in range(n):
    transformed_data[i] = A @ data[i] + b

print("\n| Student | Original Data (X)        | Transformed Data (Y)     |")
print("|---------|---------------------------|---------------------------|")
for i in range(n):
    orig = data[i]
    trans = transformed_data[i]
    print(f"| {i+1}       | [{orig[0]:5.1f}, {orig[1]:5.1f}, {orig[2]:5.1f}] | [{trans[0]:5.1f}, {trans[1]:5.1f}, {trans[2]:5.1f}] |")

# Step 4: Calculate the theoretical mean vector of the transformed data
print("\nStep 4: Calculate the theoretical mean vector of the transformed data")

theoretical_mean_Y = A @ mean_vector + b

print("\nTheoretical mean vector of Y = A·μₓ + b =")
print(f"[{A[0, 0]:.1f}, {A[0, 1]:.1f}, {A[0, 2]:.1f}] · [{mean_vector[0]:.2f}] + [{b[0]}]")
print(f"[{A[1, 0]:.1f}, {A[1, 1]:.1f}, {A[1, 2]:.1f}] · [{mean_vector[1]:.2f}] + [{b[1]}]")
print(f"[{A[2, 0]:.1f}, {A[2, 1]:.1f}, {A[2, 2]:.1f}] · [{mean_vector[2]:.2f}] + [{b[2]}]")
print(f"\nTheoretical mean vector of Y = [{theoretical_mean_Y[0]:.2f}, {theoretical_mean_Y[1]:.2f}, {theoretical_mean_Y[2]:.2f}]")

# Verify with sample mean
empirical_mean_Y = np.mean(transformed_data, axis=0)
print(f"Empirical mean vector of Y = [{empirical_mean_Y[0]:.2f}, {empirical_mean_Y[1]:.2f}, {empirical_mean_Y[2]:.2f}]")

# Step 5: Calculate the theoretical covariance matrix of the transformed data
print("\nStep 5: Calculate the theoretical covariance matrix of the transformed data")

theoretical_cov_Y = A @ cov_matrix @ A.T

print("\nTheoretical covariance matrix of Y = A·Σₓ·A^T =")
print("\n┌" + "─" * 35 + "┐")
for i in range(theoretical_cov_Y.shape[0]):
    print("│ ", end="")
    for j in range(theoretical_cov_Y.shape[1]):
        print(f"{theoretical_cov_Y[i, j]:10.2f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Verify with sample covariance
empirical_deviations_Y = transformed_data - empirical_mean_Y
empirical_cov_Y = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        empirical_cov_Y[i, j] = np.sum(empirical_deviations_Y[:, i] * empirical_deviations_Y[:, j]) / (n - 1)

print("\nEmpirical covariance matrix of Y =")
print("\n┌" + "─" * 35 + "┐")
for i in range(empirical_cov_Y.shape[0]):
    print("│ ", end="")
    for j in range(empirical_cov_Y.shape[1]):
        print(f"{empirical_cov_Y[i, j]:10.2f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 6: Interpret the results of the transformation
print("\nStep 6: Interpret the results of the transformation")

print("\nInterpretation:")
print("- The midterm average (Y₁) has lower variance than individual midterm scores")
print("- The weighted average (Y₂) is less affected by any single exam")
print("- The difference measure (Y₃) shows if a student performed better on the final than midterms")
print("- The covariance between Y₁ and Y₃ is negative, meaning students with high midterm averages")
print("  tend to show less improvement on the final exam")

# Visualization: Original vs. Transformed data in 3D
fig = plt.figure(figsize=(16, 8))

# Original data plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=100, alpha=0.7)
ax1.scatter([mean_vector[0]], [mean_vector[1]], [mean_vector[2]], c='red', s=150, alpha=1, marker='*')

# Add student labels
for i in range(n):
    ax1.text(data[i, 0], data[i, 1], data[i, 2], f'Student {i+1}', fontsize=8)

ax1.set_xlabel('Midterm 1')
ax1.set_ylabel('Midterm 2')
ax1.set_zlabel('Final Exam')
ax1.set_title('Original Data (X)')

# Transformed data plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c='green', s=100, alpha=0.7)
ax2.scatter([theoretical_mean_Y[0]], [theoretical_mean_Y[1]], [theoretical_mean_Y[2]], c='red', s=150, alpha=1, marker='*')

# Add student labels
for i in range(n):
    ax2.text(transformed_data[i, 0], transformed_data[i, 1], transformed_data[i, 2], f'Student {i+1}', fontsize=8)

ax2.set_xlabel('Y₁ (Midterm Avg)')
ax2.set_ylabel('Y₂ (Weighted Avg)')
ax2.set_zlabel('Y₃ (Final - Midterm Avg)')
ax2.set_title('Transformed Data (Y)')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_scores_transformation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Heatmap of original and transformed covariance matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Original covariance matrix
im1 = ax1.imshow(cov_matrix, cmap='Blues')
plt.colorbar(im1, ax=ax1, label='Covariance')

# Add text annotations
for i in range(p):
    for j in range(p):
        ax1.text(j, i, f'{cov_matrix[i, j]:.1f}', 
                ha='center', va='center', color='black' if cov_matrix[i, j] < 50 else 'white')

ax1.set_xticks(range(p))
ax1.set_yticks(range(p))
ax1.set_xticklabels(['Midterm 1', 'Midterm 2', 'Final'])
ax1.set_yticklabels(['Midterm 1', 'Midterm 2', 'Final'])
ax1.set_title('Original Covariance Matrix')

# Transformed covariance matrix
im2 = ax2.imshow(theoretical_cov_Y, cmap='Greens')
plt.colorbar(im2, ax=ax2, label='Covariance')

# Add text annotations
for i in range(p):
    for j in range(p):
        ax2.text(j, i, f'{theoretical_cov_Y[i, j]:.1f}', 
                ha='center', va='center', color='black' if theoretical_cov_Y[i, j] < 50 else 'white')

ax2.set_xticks(range(p))
ax2.set_yticks(range(p))
ax2.set_xticklabels(['Y₁', 'Y₂', 'Y₃'])
ax2.set_yticklabels(['Y₁', 'Y₂', 'Y₃'])
ax2.set_title('Transformed Covariance Matrix')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_scores_covariance_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Whitening Transformation for Iris Data
print("\n\nExample 2: Whitening Transformation for Iris Data")

# Synthetic dataset inspired by Iris data (sepal length, sepal width, petal length)
iris_data = np.array([
    [5.1, 3.5, 1.4],
    [4.9, 3.0, 1.4],
    [4.7, 3.2, 1.3],
    [5.4, 3.9, 1.7],
    [5.2, 3.4, 1.4],
    [5.5, 3.7, 1.5],
    [4.6, 3.6, 1.0],
    [5.0, 3.4, 1.5]
])

print("Dataset with 3 variables from simplified Iris dataset: Sepal Length, Sepal Width, Petal Length")
print("\n| Flower | Sepal Length | Sepal Width | Petal Length |")
print("|--------|--------------|------------|--------------|")
for i, row in enumerate(iris_data):
    print(f"| {i+1}      | {row[0]:<12.1f} | {row[1]:<10.1f} | {row[2]:<12.1f} |")

# Step 1: Calculate the mean vector and covariance matrix
print("\nStep 1: Calculate the mean vector and covariance matrix")

iris_mean = np.mean(iris_data, axis=0)
print(f"\nMean vector μ = [{iris_mean[0]:.2f}, {iris_mean[1]:.2f}, {iris_mean[2]:.2f}]")

# Number of samples and variables
n_iris = iris_data.shape[0]
p_iris = iris_data.shape[1]

# Compute sample covariance matrix
iris_deviations = iris_data - iris_mean
iris_cov = np.zeros((p_iris, p_iris))
for i in range(p_iris):
    for j in range(p_iris):
        iris_cov[i, j] = np.sum(iris_deviations[:, i] * iris_deviations[:, j]) / (n_iris - 1)

print("\nSample covariance matrix:")
print("\n┌" + "─" * 35 + "┐")
for i in range(p_iris):
    print("│ ", end="")
    for j in range(p_iris):
        print(f"{iris_cov[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 2: Calculate eigenvalues and eigenvectors of the covariance matrix
print("\nStep 2: Calculate eigenvalues and eigenvectors of the covariance matrix")

eigenvalues, eigenvectors = np.linalg.eigh(iris_cov)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues (variances in principal component directions):")
for i, val in enumerate(eigenvalues):
    print(f"λ{i+1} = {val:.4f}")

print("\nEigenvectors (principal component directions):")
print("\n┌" + "─" * 35 + "┐")
for i in range(p_iris):
    print("│ ", end="")
    for j in range(p_iris):
        print(f"{eigenvectors[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 3: Calculate the whitening transformation matrix
print("\nStep 3: Calculate the whitening transformation matrix")

# Create diagonal matrix of eigenvalues
lambda_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues))

# Whitening matrix W = Σ^(-1/2) = E * Λ^(-1/2) * E^T where E is matrix of eigenvectors
whitening_matrix = eigenvectors @ lambda_sqrt_inv @ eigenvectors.T

print("\nWhitening transformation matrix W = Σ^(-1/2):")
print("\n┌" + "─" * 35 + "┐")
for i in range(whitening_matrix.shape[0]):
    print("│ ", end="")
    for j in range(whitening_matrix.shape[1]):
        print(f"{whitening_matrix[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 4: Apply the whitening transformation
print("\nStep 4: Apply the whitening transformation Z = W(X - μ)")

whitened_data = np.zeros_like(iris_data)
for i in range(n_iris):
    whitened_data[i] = whitening_matrix @ (iris_data[i] - iris_mean)

print("\n| Flower | Original Data (X)        | Whitened Data (Z)        |")
print("|--------|---------------------------|---------------------------|")
for i in range(n_iris):
    orig = iris_data[i]
    whitened = whitened_data[i]
    print(f"| {i+1}      | [{orig[0]:5.2f}, {orig[1]:5.2f}, {orig[2]:5.2f}] | [{whitened[0]:6.2f}, {whitened[1]:6.2f}, {whitened[2]:6.2f}] |")

# Step 5: Verify the whitening transformation worked correctly
print("\nStep 5: Verify the whitening transformation worked correctly")

# Calculate the mean of the whitened data
whitened_mean = np.mean(whitened_data, axis=0)
print(f"\nEmpirical mean of whitened data = [{whitened_mean[0]:.6f}, {whitened_mean[1]:.6f}, {whitened_mean[2]:.6f}]")
print("The mean vector should be close to zero, as expected from the centering step of whitening.")

# Calculate the covariance matrix of the whitened data
whitened_dev = whitened_data - whitened_mean
whitened_cov = np.zeros((p_iris, p_iris))
for i in range(p_iris):
    for j in range(p_iris):
        whitened_cov[i, j] = np.sum(whitened_dev[:, i] * whitened_dev[:, j]) / (n_iris - 1)

print("\nEmpirical covariance matrix of whitened data:")
print("\n┌" + "─" * 35 + "┐")
for i in range(p_iris):
    print("│ ", end="")
    for j in range(p_iris):
        print(f"{whitened_cov[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

print("\nTheoretically, this should be approximately the identity matrix.")
print("Minor deviations are due to numerical precision and the fact that we're using a sample covariance.")

# Step 6: Interpret the results of the whitening transformation
print("\nStep 6: Interpret the results of the whitening transformation")

print("\nInterpretation:")
print("- The whitening transformation decorrelates the features (makes covariance matrix identity)")
print("- Each feature in the whitened space has unit variance")
print("- Whitening is useful for preprocessing data for many machine learning algorithms")
print("- The transformation equalizes the influence of each feature on the algorithm")
print("- Whitening can help with numerical stability in optimization algorithms")

# Visualization: Original vs. Whitened data in 3D
fig = plt.figure(figsize=(16, 8))

# Original data plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(iris_data[:, 0], iris_data[:, 1], iris_data[:, 2], c='purple', s=100, alpha=0.7)
ax1.scatter([iris_mean[0]], [iris_mean[1]], [iris_mean[2]], c='red', s=150, alpha=1, marker='*')

# Add ellipsoid representing the covariance
# This is a bit complicated for a 3D ellipsoid - skipping for now

ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax1.set_zlabel('Petal Length')
ax1.set_title('Original Iris Data (X)')

# Whitened data plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(whitened_data[:, 0], whitened_data[:, 1], whitened_data[:, 2], c='cyan', s=100, alpha=0.7)
ax2.scatter([0], [0], [0], c='red', s=150, alpha=1, marker='*')

# In the whitened space, the standard deviation in each direction is 1
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)

ax2.set_xlabel('Z₁')
ax2.set_ylabel('Z₂')
ax2.set_zlabel('Z₃')
ax2.set_title('Whitened Iris Data (Z)')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'iris_whitening_transformation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Heatmap of original and whitened covariance matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Original covariance matrix
im1 = ax1.imshow(iris_cov, cmap='Purples')
plt.colorbar(im1, ax=ax1, label='Covariance')

# Add text annotations
for i in range(p_iris):
    for j in range(p_iris):
        ax1.text(j, i, f'{iris_cov[i, j]:.4f}', 
                ha='center', va='center', color='black' if iris_cov[i, j] < 0.2 else 'white')

ax1.set_xticks(range(p_iris))
ax1.set_yticks(range(p_iris))
ax1.set_xticklabels(['Sepal L', 'Sepal W', 'Petal L'])
ax1.set_yticklabels(['Sepal L', 'Sepal W', 'Petal L'])
ax1.set_title('Original Covariance Matrix')

# Whitened covariance matrix
im2 = ax2.imshow(whitened_cov, cmap='cool')
plt.colorbar(im2, ax=ax2, label='Covariance')

# Add text annotations
for i in range(p_iris):
    for j in range(p_iris):
        ax2.text(j, i, f'{whitened_cov[i, j]:.4f}', 
                ha='center', va='center', color='black')

ax2.set_xticks(range(p_iris))
ax2.set_yticks(range(p_iris))
ax2.set_xticklabels(['Z₁', 'Z₂', 'Z₃'])
ax2.set_yticklabels(['Z₁', 'Z₂', 'Z₃'])
ax2.set_title('Whitened Covariance Matrix (Should be Identity)')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'iris_covariance_whitening.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll linear transformation example images created successfully.") 