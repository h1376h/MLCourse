import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import os
from mpl_toolkits.mplot3d import Axes3D

print("\n=== PRINCIPAL COMPONENT ANALYSIS (PCA) EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Analysis")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)
print(f"Images will be saved to: {images_dir}")

# Example 1: Customer Behavior Analysis
print("Example 1: Customer Behavior Analysis")

# Create synthetic data based on the covariance matrix provided
# Define the covariance matrix from the example
cov_matrix = np.array([
    [2500, 300, 800],
    [300, 90, 120],
    [800, 120, 400]
])

print("A marketing analyst has collected data on 100 customers with three variables:")
print("- X₁: Monthly spending ($)")
print("- X₂: Website visits per month")
print("- X₃: Time spent on website (minutes)")
print("\nThe data has the following covariance matrix:")
print("\n┌" + "─" * 35 + "┐")
for i in range(cov_matrix.shape[0]):
    print("│ ", end="")
    for j in range(cov_matrix.shape[1]):
        print(f"{cov_matrix[i, j]:10.1f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Synthetic data generation for visualization (using the covariance matrix)
np.random.seed(42)  # For reproducibility
# Create a mean vector
mean_vector = np.array([500, 20, 60])  # Arbitrary means for spending, visits, time
# Generate 100 samples from multivariate normal distribution
n_samples = 100
samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_samples)

# Label the columns
data_labels = ["Monthly spending ($)", "Website visits", "Time spent (min)"]

# Print a few samples
print("\nSample data generated for visualization (first 5 customers):")
print("\n| Customer | Monthly spending ($) | Website visits | Time spent (min) |")
print("|----------|---------------------|----------------|------------------|")
for i in range(5):
    print(f"| {i+1:8} | {samples[i, 0]:19.2f} | {samples[i, 1]:16.2f} | {samples[i, 2]:16.2f} |")

# Step 1: Find Eigenvalues and Eigenvectors of the Covariance Matrix
print("\nStep 1: Find Eigenvalues and Eigenvectors of the Covariance Matrix")

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"λ_{i+1} = {val:.1f}")

print("\nEigenvectors (each column is an eigenvector):")
print("\n┌" + "─" * 35 + "┐")
for i in range(eigenvectors.shape[0]):
    print("│ ", end="")
    for j in range(eigenvectors.shape[1]):
        print(f"{eigenvectors[i, j]:10.3f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

print("\nThis gives us the following principal components:")
for i in range(eigenvectors.shape[1]):
    print(f"\nPC{i+1} = ", end="")
    terms = []
    for j in range(eigenvectors.shape[0]):
        terms.append(f"{eigenvectors[j, i]:.3f} × X_{j+1}")
    print(" + ".join(terms))

# Step 2: Calculate the Proportion of Variance Explained by Each Component
print("\nStep 2: Calculate the Proportion of Variance Explained by Each Component")

total_variance = np.sum(eigenvalues)
print(f"Total variance = λ₁ + λ₂ + λ₃ = {' + '.join([f'{val:.1f}' for val in eigenvalues])} = {total_variance:.1f}")

print("\nProportion of variance explained by each component:")
variance_explained = []
cumulative_variance = 0
for i, val in enumerate(eigenvalues):
    proportion = val / total_variance
    cumulative_variance += proportion
    variance_explained.append(proportion * 100)
    print(f"- PC{i+1}: {val:.1f}/{total_variance:.1f} = {proportion:.3f} or {proportion*100:.1f}%")

print("\nCumulative proportion of variance:")
cumulative_variance = 0
for i, val in enumerate(eigenvalues):
    proportion = val / total_variance
    cumulative_variance += proportion
    print(f"- PC1 to PC{i+1}: {cumulative_variance:.3f} or {cumulative_variance*100:.1f}%")

# Step 3: Determine the Number of Components to Retain
print("\nStep 3: Determine the Number of Components to Retain")
print("Since the first two principal components explain 98.7% of the total variance, we can")
print("reduce the dimensionality from 3 to 2 with minimal information loss.")

# Step 4: Interpret the Principal Components
print("\nStep 4: Interpret the Principal Components")
print("Looking at the coefficient magnitudes in each eigenvector:")

print("\n1st Principal Component (PC1):")
print(f"PC1 = {eigenvectors[0, 0]:.3f} × X₁ + {eigenvectors[1, 0]:.3f} × X₂ + {eigenvectors[2, 0]:.3f} × X₃")
print("This component is dominated by monthly spending (X₁) with some contribution from time spent (X₃).")
print("We could interpret this as \"overall customer engagement level.\"")

print("\n2nd Principal Component (PC2):")
print(f"PC2 = {eigenvectors[0, 1]:.3f} × X₁ + {eigenvectors[1, 1]:.3f} × X₂ + {eigenvectors[2, 1]:.3f} × X₃")
print("This component contrasts website visits (X₂) and time spent (X₃) against spending (X₁).")
print("We could interpret this as \"browsing behavior without purchasing.\"")

# Step 5: Project Data onto Principal Components
print("\nStep 5: Project Data onto Principal Components")

# Define the projection matrix (first 2 eigenvectors)
projection_matrix = eigenvectors[:, :2]
print("\nThe projection matrix for reducing to 2 dimensions is:")
print("\n┌" + "─" * 23 + "┐")
for i in range(projection_matrix.shape[0]):
    print("│ ", end="")
    for j in range(projection_matrix.shape[1]):
        print(f"{projection_matrix[i, j]:10.3f} ", end="")
    print("│")
print("└" + "─" * 23 + "┘")

# Project the data onto the first 2 principal components
projected_data = np.dot(samples, projection_matrix)

print("\nSample of original vs. projected data (first 3 customers):")
print("\n| Customer | Original Data (X₁, X₂, X₃) | Projected Data (PC1, PC2) |")
print("|----------|----------------------------|--------------------------|")
for i in range(3):
    orig = samples[i]
    proj = projected_data[i]
    print(f"| {i+1:8} | ({orig[0]:.1f}, {orig[1]:.1f}, {orig[2]:.1f}) | ({proj[0]:.1f}, {proj[1]:.1f}) |")

# Visualization: Original data in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='blue', alpha=0.5)
ax.set_xlabel(data_labels[0])
ax.set_ylabel(data_labels[1])
ax.set_zlabel(data_labels[2])
ax.set_title('Original 3D Customer Data')

# Plot the eigenvectors from the origin
origin = mean_vector
for i in range(3):
    ax.quiver(
        origin[0], origin[1], origin[2],
        eigenvectors[0, i] * eigenvalues[i] * 0.01,
        eigenvectors[1, i] * eigenvalues[i] * 0.01,
        eigenvectors[2, i] * eigenvalues[i] * 0.01,
        color=['red', 'green', 'orange'][i],
        lw=2, arrow_length_ratio=0.1,
        label=f"PC{i+1} (var: {variance_explained[i]:.1f}%)"
    )

ax.legend()
plt.tight_layout()
save_path = os.path.join(images_dir, 'customer_data_3d_with_pcs.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved 3D visualization to: {save_path}")
plt.close()

# Visualization: Projected data
plt.figure(figsize=(10, 8))
plt.scatter(projected_data[:, 0], projected_data[:, 1], c='blue', alpha=0.6)
plt.xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)')
plt.ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)')
plt.title('Customer Data Projected onto First Two Principal Components')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)

# Add a few sample points with labels
for i in range(5):
    plt.annotate(f"Customer {i+1}", 
                 (projected_data[i, 0], projected_data[i, 1]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')

plt.tight_layout()
save_path = os.path.join(images_dir, 'customer_data_pca_2d.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved 2D projection to: {save_path}")
plt.close()

# Visualization: Scree plot (variance explained)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7, align='center')
plt.step(range(1, len(eigenvalues) + 1), np.cumsum(eigenvalues), where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue / Variance Explained')
plt.title('Scree Plot: Eigenvalues and Cumulative Variance')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.legend()
plt.tight_layout()
save_path = os.path.join(images_dir, 'customer_data_scree_plot.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved scree plot to: {save_path}")
plt.close()

# Visualization: Biplot (for understanding variable contributions)
plt.figure(figsize=(10, 8))
# Plot the projected points
plt.scatter(projected_data[:, 0], projected_data[:, 1], c='gray', alpha=0.3, label='Customers')

# Plot the feature loadings (eigenvectors)
scale = 2
for i in range(eigenvectors.shape[0]):
    plt.arrow(0, 0, 
              eigenvectors[i, 0] * scale, 
              eigenvectors[i, 1] * scale,
              head_width=0.1, 
              head_length=0.1, 
              fc='red', 
              ec='red')
    plt.text(eigenvectors[i, 0] * scale * 1.15, 
             eigenvectors[i, 1] * scale * 1.15, 
             data_labels[i], 
             color='red', 
             ha='center', 
             va='center')

plt.xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)')
plt.ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)')
plt.title('Biplot: Principal Components and Variable Contributions')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.axis('equal')
plt.tight_layout()
save_path = os.path.join(images_dir, 'customer_data_biplot.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved biplot to: {save_path}")
plt.close()

# Example 2: Comparison of PCA with Original Variables
print("\n\nExample 2: Comparison of PCA with Original Variables")
print("\nIn this example, we demonstrate how the principal components are uncorrelated and better")
print("capture the variance in the data compared to the original variables.")

# Step 1: Calculate the Covariance Matrix of the Principal Components
print("\nStep 1: Calculate the Covariance Matrix of the Principal Components")
print("If we project the data onto the principal components, the resulting covariance matrix")
print("would be diagonal, with the eigenvalues on the diagonal.")

# Calculate the covariance matrix of the projected data (should be diagonal)
full_projection = np.dot(samples, eigenvectors)
pc_cov_matrix = np.cov(full_projection, rowvar=False)

print("\nThe covariance matrix of the principal components is:")
print("\n┌" + "─" * 35 + "┐")
for i in range(pc_cov_matrix.shape[0]):
    print("│ ", end="")
    for j in range(pc_cov_matrix.shape[1]):
        print(f"{pc_cov_matrix[i, j]:10.1f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 2: Compare with Original Covariance Matrix
print("\nStep 2: Compare with Original Covariance Matrix")

print("\nThe original covariance matrix was:")
print("\n┌" + "─" * 35 + "┐")
for i in range(cov_matrix.shape[0]):
    print("│ ", end="")
    for j in range(cov_matrix.shape[1]):
        print(f"{cov_matrix[i, j]:10.1f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Step 3: Analyze the Differences
print("\nStep 3: Analyze the Differences")

print("\n| Comparison | Original Variables | Principal Components |")
print("|------------|-------------------|---------------------|")
print("| Correlation | Variables are correlated (non-zero off-diagonal elements) | Components are uncorrelated (zero off-diagonal elements) |")
print("| Variance Distribution | Uneven but spread across variables | Concentrated in first component (93.9%) |")
print("| Interpretation | Direct measures (spending, visits, time) | Composite measures (engagement, browsing behavior) |")

# Step 4: Visualize the Transformation (Scatter plot matrix of original vs PC data)
print("\nStep 4: Visualize the Transformation")

# Visualization: Original vs PCA scatter matrix
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparison of Original Variables vs Principal Components', fontsize=16)

# Original data correlations (upper row)
for i in range(3):
    for j in range(i, 3):
        if i == j:
            axes[0, i].hist(samples[:, i], bins=20, alpha=0.7, color='blue')
            axes[0, i].set_title(f'Dist. of {data_labels[i]}')
        elif j < 3:  # Off-diagonal plots (only upper triangle)
            axes[0, j].scatter(samples[:, i], samples[:, j], alpha=0.5, color='blue')
            axes[0, j].set_xlabel(data_labels[i])
            axes[0, j].set_ylabel(data_labels[j])
            
            # Calculate correlation
            corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
            axes[0, j].set_title(f'Corr: {corr:.2f}')

# PC data correlations (lower row)
pc_labels = [f'PC{i+1}' for i in range(3)]
for i in range(3):
    for j in range(i, 3):
        if i == j:
            axes[1, i].hist(full_projection[:, i], bins=20, alpha=0.7, color='green')
            axes[1, i].set_title(f'Dist. of {pc_labels[i]}')
        elif j < 3:  # Off-diagonal plots (only upper triangle)
            axes[1, j].scatter(full_projection[:, i], full_projection[:, j], alpha=0.5, color='green')
            axes[1, j].set_xlabel(pc_labels[i])
            axes[1, j].set_ylabel(pc_labels[j])
            
            # Calculate correlation (should be near 0)
            corr = np.corrcoef(full_projection[:, i], full_projection[:, j])[0, 1]
            axes[1, j].set_title(f'Corr: {corr:.2f}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(images_dir, 'original_vs_pc_scatter_matrix.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved scatter matrix to: {save_path}")
plt.close()

# Visualization: Correlation heatmaps comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Compute correlation matrices
orig_corr = np.corrcoef(samples, rowvar=False)
pc_corr = np.corrcoef(full_projection, rowvar=False)

# Original correlation heatmap
im1 = ax1.imshow(orig_corr, cmap='coolwarm', vmin=-1, vmax=1)
ax1.set_title('Correlation Matrix of Original Variables')
# Add text annotations
for i in range(orig_corr.shape[0]):
    for j in range(orig_corr.shape[1]):
        ax1.text(j, i, f'{orig_corr[i, j]:.2f}', 
                ha='center', va='center', color='black' if abs(orig_corr[i, j]) < 0.7 else 'white')
ax1.set_xticks(range(len(data_labels)))
ax1.set_yticks(range(len(data_labels)))
ax1.set_xticklabels(data_labels)
ax1.set_yticklabels(data_labels)

# PC correlation heatmap
im2 = ax2.imshow(pc_corr, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_title('Correlation Matrix of Principal Components')
# Add text annotations
for i in range(pc_corr.shape[0]):
    for j in range(pc_corr.shape[1]):
        ax2.text(j, i, f'{pc_corr[i, j]:.2f}', 
                ha='center', va='center', color='black' if abs(pc_corr[i, j]) < 0.7 else 'white')
ax2.set_xticks(range(len(pc_labels)))
ax2.set_yticks(range(len(pc_labels)))
ax2.set_xticklabels(pc_labels)
ax2.set_yticklabels(pc_labels)

plt.colorbar(im1, ax=ax1, label='Correlation')
plt.colorbar(im2, ax=ax2, label='Correlation')
plt.tight_layout()
save_path = os.path.join(images_dir, 'original_vs_pc_correlation_heatmaps.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Saved correlation heatmaps to: {save_path}")
plt.close()

# Interpretation
print("\nInterpretation:")
print("The principal components have successfully:")
print("1. Greatly reduced correlations between variables (diagonal covariance matrix)")
print("2. Concentrated the variance in the first few components (93.9% in first component)")
print("3. Created a more efficient representation of the data (2D instead of 3D)")
print("\nThis transformation makes it possible to use just the first two principal components")
print("instead of all three original variables, with minimal loss of information (98.4% variance retained).")

print("\nAll PCA example images created successfully in:")
print(images_dir) 