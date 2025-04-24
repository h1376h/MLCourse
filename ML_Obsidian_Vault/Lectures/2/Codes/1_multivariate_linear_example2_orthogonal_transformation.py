import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns

def example2_orthogonal_transformation():
    """
    Orthogonal Transformation Example
    
    Problem Statement:
    Consider a random vector X ~ N(0, I₃), i.e., a standard multivariate normal in 3 dimensions.
    Let Q be an orthogonal matrix:
    
    Q = [[1/√3, 1/√2, 1/√6],
         [1/√3, -1/√2, 1/√6],
         [1/√3, 0, -2/√6]]
    
    Define Y = QX.
    
    a) Find the distribution of Y.
    b) Show that this transformation preserves Euclidean distances between points.
    c) Explain the geometric interpretation of this transformation.
    """
    print("\n" + "="*80)
    print("Example 2: Orthogonal Transformations and Preservation of Distances")
    print("="*80)
    
    # Define original parameters
    mu_X = np.zeros(3)  # Zero mean vector
    Sigma_X = np.eye(3)  # Identity covariance matrix
    
    # Define orthogonal matrix Q
    Q = np.array([
        [1/np.sqrt(3), 1/np.sqrt(2), 1/np.sqrt(6)],
        [1/np.sqrt(3), -1/np.sqrt(2), 1/np.sqrt(6)],
        [1/np.sqrt(3), 0, -2/np.sqrt(6)]
    ])
    
    print("\nGiven:")
    print(f"Mean vector μ_X = {mu_X}")
    print(f"Covariance matrix Σ_X = \n{Sigma_X}")
    print(f"Orthogonal matrix Q = \n{Q}")
    
    # Print numeric values for better understanding
    print("\nQ with numeric values (approximate):")
    Q_numeric = Q.copy()  # No need to redefine, just use the one we already have
    print(np.round(Q_numeric, 4))
    
    # Print the exact values for reference
    print("\nQ with exact symbolic values:")
    print(f"Q[0,0] = Q[1,0] = Q[2,0] = 1/√3 ≈ {1/np.sqrt(3):.6f}")
    print(f"Q[0,1] = -Q[1,1] = 1/√2 ≈ {1/np.sqrt(2):.6f}, Q[2,1] = 0")
    print(f"Q[0,2] = Q[1,2] = 1/√6 ≈ {1/np.sqrt(6):.6f}, Q[2,2] = -2/√6 ≈ {-2/np.sqrt(6):.6f}")
    
    # Verify Q is orthogonal: Q^T Q = I
    print("\n" + "-"*60)
    print("Step 1: Verify Q is orthogonal by calculating Q^T Q = I")
    print("-"*60)
    
    print(f"Q^T (transpose of Q) = \n{np.round(Q.T, 4)}")
    
    # Detailed calculation of Q^T Q - first show some dot products
    print("\nDetailed verification of orthogonality - calculating Q^T·Q element by element:")
    
    # Calculate dot products of columns
    print("\nChecking orthogonality of columns (dot products should be 1 for i=j, 0 for i≠j):")
    for i in range(3):
        for j in range(3):
            dot_product = np.dot(Q[:, i], Q[:, j])
            if i == j:
                print(f"Column {i+1}·Column {i+1} = {dot_product:.10f} ≈ 1 (expected)")
            else:
                print(f"Column {i+1}·Column {j+1} = {dot_product:.10f} ≈ 0 (expected)")
    
    # Calculate full Q^T Q matrix
    QTQ = np.dot(Q.T, Q)
    print(f"\nFull Q^T Q matrix (before rounding) = \n{QTQ}")
    
    # Round to handle numerical precision
    QTQ_rounded = np.round(QTQ, decimals=10)
    print(f"\nFull Q^T Q matrix (after rounding) = \n{QTQ_rounded}")
    
    # Check if Q^T Q is approximately identity
    is_orthogonal = np.allclose(QTQ, np.eye(3), atol=1e-10)
    print(f"Is Q orthogonal? {is_orthogonal}")
    
    # Verify columns have unit length (necessary condition for orthogonality)
    print("\nVerifying that columns of Q have unit length:")
    for i in range(3):
        col_norm = np.linalg.norm(Q[:, i])
        print(f"||Column {i+1}|| = {col_norm:.10f} ≈ 1")
    
    # (a) Find the distribution of Y
    print("\n" + "-"*60)
    print("(a) Finding the distribution of Y:")
    print("-"*60)
    
    # Calculate mean of Y
    print("\nStep 2: Calculate the mean vector of Y using μ_Y = Q·μ_X")
    print("Since μ_X = [0, 0, 0], for any matrix Q, Q·μ_X = [0, 0, 0]")
    
    mu_Y = np.dot(Q, mu_X)
    print(f"\nFinal result: μ_Y = Q·μ_X = {mu_Y}")
    
    # Calculate covariance of Y
    print("\nStep 3: Calculate the covariance matrix of Y using Σ_Y = Q·Σ_X·Q^T")
    print("Since Σ_X = I (identity matrix), this simplifies to:")
    print("Σ_Y = Q·I·Q^T = Q·Q^T")
    
    # Manual calculation of Q·Q^T to demonstrate the process
    print("\nCalculating Q·Q^T manually:")
    QQT_manual = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Calculate each element as the dot product of row i of Q with column j of Q^T
            # Which is the same as dot product of row i of Q with row j of Q
            QQT_manual[i, j] = np.dot(Q[i, :], Q[j, :])
    
    print(f"Manual calculation of Q·Q^T = \n{np.round(QQT_manual, 10)}")
    
    # Calculate with numpy
    Sigma_Y = np.dot(Q, np.dot(Sigma_X, Q.T))
    Sigma_Y_rounded = np.round(Sigma_Y, decimals=10)
    print(f"\nFull Σ_Y = Q·Σ_X·Q^T = \n{Sigma_Y_rounded}")
    
    # Verify the manual calculation matches the numpy calculation
    print(f"\nDo manual and numpy calculations match? {np.allclose(QQT_manual, Sigma_Y, atol=1e-10)}")
    
    # Check if Q·Q^T is approximately identity
    is_identity = np.allclose(Sigma_Y_rounded, np.eye(3), atol=1e-10)
    print(f"Is Σ_Y approximately identity? {is_identity}")
    
    print("\nThus, Y follows the distribution:")
    print(f"Y ~ N({mu_Y}, \n{Sigma_Y_rounded})")
    print("This means Y ~ N(0, I), same as X!")
    print("Key insight: An orthogonal transformation of a standard multivariate normal remains standard multivariate normal.")
    
    # (b) Show preservation of Euclidean distances
    print("\n" + "-"*60)
    print("(b) Showing that orthogonal transformations preserve Euclidean distances:")
    print("-"*60)
    
    # Generate two random points in X space
    np.random.seed(42)  # Set seed for reproducibility
    x1 = np.random.randn(3)
    x2 = np.random.randn(3)
    
    print(f"Step 4: Generate two random points in X space:")
    print(f"x₁ = {np.round(x1, 4)}")
    print(f"x₂ = {np.round(x2, 4)}")
    
    # Calculate their distance in X space
    print("\nStep 5: Calculate the Euclidean distance between x₁ and x₂:")
    
    x_diff = x1 - x2
    print(f"x₁ - x₂ = {np.round(x_diff, 4)}")
    
    x_diff_squared = x_diff**2
    print(f"(x₁ - x₂)² = {np.round(x_diff_squared, 4)}")
    
    x_diff_sum = np.sum(x_diff_squared)
    print(f"Sum of squared differences = {np.round(x_diff_sum, 6)}")
    
    dist_X = np.sqrt(x_diff_sum)
    print(f"Euclidean distance ||x₁ - x₂|| = √{np.round(x_diff_sum, 6)} = {np.round(dist_X, 6)}")
    
    # Verify using numpy's built-in norm function
    dist_X_numpy = np.linalg.norm(x1 - x2)
    print(f"Verification using numpy: ||x₁ - x₂|| = {np.round(dist_X_numpy, 6)}")
    print(f"Are the calculations equal? {np.isclose(dist_X, dist_X_numpy)}")
    
    # Transform to Y space
    print("\nStep 6: Transform the points to Y space using y₁ = Qx₁ and y₂ = Qx₂:")
    
    y1 = np.dot(Q, x1)
    print(f"y₁ = Q·x₁ = {np.round(y1, 4)}")
    
    y2 = np.dot(Q, x2)
    print(f"y₂ = Q·x₂ = {np.round(y2, 4)}")
    
    # Calculate distance in Y space
    print("\nStep 7: Calculate the Euclidean distance between y₁ and y₂:")
    
    y_diff = y1 - y2
    print(f"y₁ - y₂ = {np.round(y_diff, 4)}")
    
    y_diff_squared = y_diff**2
    print(f"(y₁ - y₂)² = {np.round(y_diff_squared, 4)}")
    
    y_diff_sum = np.sum(y_diff_squared)
    print(f"Sum of squared differences = {np.round(y_diff_sum, 6)}")
    
    dist_Y = np.sqrt(y_diff_sum)
    print(f"Euclidean distance ||y₁ - y₂|| = √{np.round(y_diff_sum, 6)} = {np.round(dist_Y, 6)}")
    
    # Verify using numpy's built-in norm function
    dist_Y_numpy = np.linalg.norm(y1 - y2)
    print(f"Verification using numpy: ||y₁ - y₂|| = {np.round(dist_Y_numpy, 6)}")
    
    # Compare distances
    print(f"\nStep 8: Compare the distances:")
    print(f"Distance in X space: ||x₁ - x₂|| = {np.round(dist_X, 6)}")
    print(f"Distance in Y space: ||y₁ - y₂|| = {np.round(dist_Y, 6)}")
    print(f"Difference: |{np.round(dist_X, 6)} - {np.round(dist_Y, 6)}| = {np.round(abs(dist_X - dist_Y), 10)} (due to floating-point precision)")
    
    # Theoretical explanation
    print("\nTheoretical explanation:")
    print("For orthogonal matrix Q, ||Qx||² = (Qx)ᵀ(Qx) = xᵀQᵀQx = xᵀx = ||x||²")
    print("Therefore, ||y₁ - y₂|| = ||Q(x₁ - x₂)|| = ||x₁ - x₂||")
    
    # (c) Geometric interpretation
    print("\n" + "-"*60)
    print("(c) Geometric interpretation of the orthogonal transformation:")
    print("-"*60)
    
    print("1. The orthogonal transformation Q represents a rotation or reflection in 3D space.")
    print("2. It preserves distances (as shown in part b) and angles between vectors.")
    print("3. The columns of Q form a new orthonormal basis for the space.")
    print("4. The transformation maps the standard basis vectors to this new orthonormal basis.")
    
    # Detailed explanation of basis vector mapping
    print("\nDetailed explanation of how standard basis vectors are mapped:")
    
    # First basis vector
    e1 = np.array([1, 0, 0])
    Qe1 = np.dot(Q, e1)
    print(f"e₁ = [1, 0, 0] (standard X axis) maps to:")
    print(f"Qe₁ = {np.round(Qe1, 4)} = first column of Q")
    
    # Second basis vector
    e2 = np.array([0, 1, 0])
    Qe2 = np.dot(Q, e2)
    print(f"\ne₂ = [0, 1, 0] (standard Y axis) maps to:")
    print(f"Qe₂ = {np.round(Qe2, 4)} = second column of Q")
    
    # Third basis vector
    e3 = np.array([0, 0, 1])
    Qe3 = np.dot(Q, e3)
    print(f"\ne₃ = [0, 0, 1] (standard Z axis) maps to:")
    print(f"Qe₃ = {np.round(Qe3, 4)} = third column of Q")
    
    print("\n5. Since the original distribution is spherically symmetric (Σ_X = I), it looks the same after rotation.")
    print("6. The orthogonal matrix Q maintains orthogonality and unit length of basis vectors.")
    
    # Create visualizations
    print("\n" + "-"*60)
    print("Creating visualizations:")
    print("-"*60)
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Use Images/Linear_Transformations directory
    images_dir = os.path.join(parent_dir, "Images", "Linear_Transformations")
    
    # Create directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate random sample from standard normal
    n_samples = 500
    np.random.seed(42)  # Set seed for reproducibility
    X_samples = np.random.multivariate_normal(mu_X, Sigma_X, n_samples)
    
    # Transform samples
    Y_samples = np.dot(X_samples, Q.T)  # Apply transformation to all samples
    
    # First visualization: Original Distribution with standard basis vectors
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original samples
    ax.scatter(X_samples[:, 0], X_samples[:, 1], X_samples[:, 2], alpha=0.3, s=5, color='blue')
    ax.set_title('Original Distribution (X)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    
    # Plot standard basis vectors
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1, label='e1')
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='e2')
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.1, label='e3')
    
    # Set equal aspect ratio
    max_range = np.max([
        X_samples[:, 0].max() - X_samples[:, 0].min(),
        X_samples[:, 1].max() - X_samples[:, 1].min(),
        X_samples[:, 2].max() - X_samples[:, 2].min()
    ])
    mid_x = (X_samples[:, 0].max() + X_samples[:, 0].min()) * 0.5
    mid_y = (X_samples[:, 1].max() + X_samples[:, 1].min()) * 0.5
    mid_z = (X_samples[:, 2].max() + X_samples[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.legend()
    plt.tight_layout()
    
    # Save the original distribution plot
    save_path1 = os.path.join(images_dir, "example2_original_distribution.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print(f"\nOriginal distribution plot saved to: {save_path1}")
    plt.close()
    
    # Second visualization: Transformed Distribution with new basis vectors
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot transformed samples
    ax.scatter(Y_samples[:, 0], Y_samples[:, 1], Y_samples[:, 2], alpha=0.3, s=5, color='blue')
    ax.set_title('Transformed Distribution (Y)')
    ax.set_xlabel('Y1')
    ax.set_ylabel('Y2')
    ax.set_zlabel('Y3')
    
    # Plot transformed basis vectors (columns of Q)
    ax.quiver(0, 0, 0, Q[0, 0], Q[1, 0], Q[2, 0], color='red', arrow_length_ratio=0.1, label='Qe1')
    ax.quiver(0, 0, 0, Q[0, 1], Q[1, 1], Q[2, 1], color='green', arrow_length_ratio=0.1, label='Qe2')
    ax.quiver(0, 0, 0, Q[0, 2], Q[1, 2], Q[2, 2], color='black', arrow_length_ratio=0.1, label='Qe3')
    
    # Set equal aspect ratio for transformed plot
    max_range = np.max([
        Y_samples[:, 0].max() - Y_samples[:, 0].min(),
        Y_samples[:, 1].max() - Y_samples[:, 1].min(),
        Y_samples[:, 2].max() - Y_samples[:, 2].min()
    ])
    mid_x = (Y_samples[:, 0].max() + Y_samples[:, 0].min()) * 0.5
    mid_y = (Y_samples[:, 1].max() + Y_samples[:, 1].min()) * 0.5
    mid_z = (Y_samples[:, 2].max() + Y_samples[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.legend()
    plt.tight_layout()
    
    # Save the transformed distribution plot
    save_path2 = os.path.join(images_dir, "example2_transformed_distribution.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print(f"Transformed distribution plot saved to: {save_path2}")
    plt.close()
    
    # Third visualization: Distance preservation illustration
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Distance Preservation')
    
    # Plot two points in X space
    ax.scatter([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], color=['red', 'green'], s=100, label='X points')
    
    # Draw a line between them
    ax.plot([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], 'b-', linewidth=2, label=f'Dist X: {dist_X:.3f}')
    
    # Plot transformed points in Y space
    ax.scatter([y1[0], y2[0]], [y1[1], y2[1]], [y1[2], y2[2]], color=['darkred', 'darkgreen'], s=100, marker='^', label='Y points')
    
    # Draw a line between them
    ax.plot([y1[0], y2[0]], [y1[1], y2[1]], [y1[2], y2[2]], 'k--', linewidth=2, label=f'Dist Y: {dist_Y:.3f}')
    
    # Add legend
    ax.legend()
    
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    
    plt.tight_layout()
    
    # Save the distance preservation plot
    save_path3 = os.path.join(images_dir, "example2_distance_preservation.png")
    plt.savefig(save_path3, bbox_inches='tight', dpi=300)
    print(f"Distance preservation plot saved to: {save_path3}")
    plt.close()
    
    # Fourth visualization: Orthogonality property visualization
    plt.figure(figsize=(8, 6))
    plt.title('Orthogonality Visualization: Dot Products between Basis Vectors')
    
    # Create a heatmap of dot products between basis vectors
    # For original standard basis
    std_basis = np.eye(3)
    std_dots = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            std_dots[i, j] = np.dot(std_basis[:, i], std_basis[:, j])
    
    # For transformed basis (columns of Q)
    new_dots = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            new_dots[i, j] = np.dot(Q[:, i], Q[:, j])
    
    # Plot heatmaps side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(std_dots, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                xticklabels=['e1', 'e2', 'e3'], 
                yticklabels=['e1', 'e2', 'e3'],
                ax=ax1)
    ax1.set_title("Standard Basis Dot Products")
    
    sns.heatmap(new_dots, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                xticklabels=['Qe1', 'Qe2', 'Qe3'], 
                yticklabels=['Qe1', 'Qe2', 'Qe3'],
                ax=ax2)
    ax2.set_title("Transformed Basis Dot Products")
    
    plt.suptitle("Orthogonality Preservation")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the orthogonality visualization
    save_path4 = os.path.join(images_dir, "example2_orthogonality.png")
    plt.savefig(save_path4, bbox_inches='tight', dpi=300)
    print(f"Orthogonality visualization saved to: {save_path4}")
    plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Example 2:")
    print("="*60)
    print("1. Orthogonal transformations preserve the standard multivariate normal distribution: N(0, I) → N(0, I)")
    print("2. Orthogonal transformations preserve Euclidean distances between points")
    print("3. Geometrically, orthogonal transformations represent rotations and/or reflections")
    print("4. The columns of Q form a new orthonormal basis for the space")
    print("5. The spherical symmetry of the standard normal distribution means it looks the same after orthogonal transformations")
    print("6. Orthogonality preserves angles between vectors, including the orthogonality of basis vectors")
    
    # Return the file paths for the markdown
    return [save_path1, save_path2, save_path3, save_path4]

if __name__ == "__main__":
    example2_orthogonal_transformation() 