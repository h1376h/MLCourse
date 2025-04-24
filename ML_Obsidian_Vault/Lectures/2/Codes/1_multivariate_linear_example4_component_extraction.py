import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import seaborn as sns

def example4_component_extraction():
    """
    Example 4: Component Extraction and Linear Combinations
    
    Problem Statement:
    Let X = [X₁, X₂, X₃, X₄]ᵀ be a four-variate Gaussian random vector with
    
    μₓ = [2, 1, 1, 0]ᵀ and Cₓ = [
        [6, 3, 2, 1],
        [3, 4, 3, 2],
        [2, 3, 4, 3],
        [1, 2, 3, 3]
    ]
    
    Let X₁, X₂, and Y be defined as:
    X₁ = [X₁, X₂]ᵀ,
    X₂ = [X₃, X₄]ᵀ,
    Y = [2X₁, X₁+2X₂, X₃+X₄]ᵀ
    
    a) Find the distribution of X₁, its mean vector and covariance matrix.
    b) Find the distribution of Y, its mean vector and covariance matrix.
    """
    print("\n" + "="*80)
    print("Example 4: Component Extraction and Linear Combinations")
    print("="*80)
    
    # Define original parameters
    mu_X = np.array([2, 1, 1, 0])
    C_X = np.array([
        [6, 3, 2, 1],
        [3, 4, 3, 2],
        [2, 3, 4, 3],
        [1, 2, 3, 3]
    ])
    
    print("\nGiven:")
    print(f"Mean vector μ_X = {mu_X}")
    print(f"Covariance matrix C_X = \n{C_X}")
    
    # Create images directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "Images", "Linear_Transformations")
    os.makedirs(images_dir, exist_ok=True)
    
    # Track saved image paths
    saved_images = []
    
    # Key concept explanation
    print("\n" + "-"*60)
    print("Key Concept: Linear Transformations and Component Extraction")
    print("-"*60)
    print("If X ~ N(μ, Σ) and Y = AX + b, then:")
    print("  Y ~ N(Aμ + b, AΣA^T)")
    print("\nThis applies to two important special cases:")
    print("1. Component extraction: When A selects specific components of X")
    print("2. Linear combinations: When A forms linear combinations of components of X")
    print("\nThe resulting distributions are always multivariate normal.")
    
    # (a) Find the distribution of X₁
    print("\n" + "-"*60)
    print("(a) Finding the distribution of X₁ = [X₁, X₂]ᵀ:")
    print("-"*60)
    
    # Define extraction matrix for X₁
    A1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    print("To extract the first two components of X, we use the extraction matrix:")
    print(f"A1 = \n{A1}")
    
    # Calculate mean of X₁
    mu_X1 = np.dot(A1, mu_X)
    print(f"\nMean vector of X1: μ_X1 = {mu_X1}")
    
    # Calculate covariance of X₁
    A1_CX = np.dot(A1, C_X)
    C_X1 = np.dot(A1_CX, A1.T)
    print(f"Covariance matrix of X1: C_X1 = \n{C_X1}")
    
    # Observation about submatrix
    print("\nNote: C_X1 is the upper-left 2×2 submatrix of C_X:")
    print(f"Upper-left 2×2 submatrix of C_X = \n{C_X[0:2, 0:2]}")
    print("\nTherefore, X1 ~ N([2, 1], [[6, 3], [3, 4]])")
    
    # (b) Find the distribution of Y
    print("\n" + "-"*60)
    print("(b) Finding the distribution of Y = [2X₁, X₁+2X₂, X₃+X₄]ᵀ:")
    print("-"*60)
    
    # Define transformation matrix for Y
    A2 = np.array([
        [2, 0, 0, 0],
        [1, 2, 0, 0],
        [0, 0, 1, 1]
    ])
    
    print("To create the linear combinations for Y, we use the transformation matrix:")
    print(f"A2 = \n{A2}")
    print("\nThis creates the following linear combinations:")
    print("Y1 = 2X1")
    print("Y2 = X1 + 2X2")
    print("Y3 = X3 + X4")
    
    # Calculate mean of Y
    mu_Y = np.dot(A2, mu_X)
    print(f"\nMean vector of Y: μ_Y = {mu_Y}")
    
    # Calculate covariance of Y
    A2_CX = np.dot(A2, C_X)
    C_Y = np.dot(A2_CX, A2.T)
    print(f"Covariance matrix of Y: C_Y = \n{C_Y}")
    
    print("\nTherefore, Y ~ N([4, 4, 1], C_Y)")
    
    # Create visualizations
    print("\n" + "-"*60)
    print("Creating visualizations:")
    print("-"*60)
    
    # Visualization 1: Original 4D distribution projected onto first two dimensions (X1, X2)
    plt.figure(figsize=(8, 6))
    
    # Create a grid of points for X1 and X2
    x1 = np.linspace(mu_X[0] - 3*np.sqrt(C_X[0, 0]), mu_X[0] + 3*np.sqrt(C_X[0, 0]), 100)
    x2 = np.linspace(mu_X[1] - 3*np.sqrt(C_X[1, 1]), mu_X[1] + 3*np.sqrt(C_X[1, 1]), 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate the 2D marginal PDF by using the extracted mean and covariance
    pos = np.dstack((X1, X2))
    rv = multivariate_normal(mu_X1, C_X1)
    Z = rv.pdf(pos)
    
    # Plot contours
    plt.contourf(X1, X2, Z, levels=15, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Probability Density')
    
    # Add mean point
    plt.scatter(mu_X[0], mu_X[1], color='red', s=50, marker='x', label='Mean')
    
    # Set labels and title
    plt.title('Original Distribution Projected onto X1-X2 Plane')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    save_path1 = os.path.join(images_dir, "example4_original_projection.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print(f"\nOriginal projection visualization saved to: {save_path1}")
    plt.close()
    saved_images.append(save_path1)
    
    # Visualization 2: Extracted X1 component's bivariate distribution
    plt.figure(figsize=(8, 6))
    
    # Plot contours for X1 component
    plt.contourf(X1, X2, Z, levels=15, cmap='Blues', alpha=0.8)
    plt.colorbar(label='Probability Density')
    
    # Add contour lines
    contour = plt.contour(X1, X2, Z, levels=5, colors='black', alpha=0.7)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Add mean point
    plt.scatter(mu_X1[0], mu_X1[1], color='red', s=50, marker='x', label='Mean')
    
    # Set labels and title
    plt.title('Extracted Components Distribution (X1=[X1,X2])')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    save_path2 = os.path.join(images_dir, "example4_extracted_components.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print(f"Extracted components visualization saved to: {save_path2}")
    plt.close()
    saved_images.append(save_path2)
    
    # Visualization 3: Linear transformation Y visualization
    # We'll visualize the Y1-Y2 plane (first two components of Y)
    plt.figure(figsize=(8, 6))
    
    # Create a grid of points for Y1 and Y2
    y1 = np.linspace(mu_Y[0] - 3*np.sqrt(C_Y[0, 0]), mu_Y[0] + 3*np.sqrt(C_Y[0, 0]), 100)
    y2 = np.linspace(mu_Y[1] - 3*np.sqrt(C_Y[1, 1]), mu_Y[1] + 3*np.sqrt(C_Y[1, 1]), 100)
    Y1, Y2 = np.meshgrid(y1, y2)
    
    # Calculate the 2D marginal PDF for Y1-Y2
    pos_y = np.dstack((Y1, Y2))
    rv_y = multivariate_normal([mu_Y[0], mu_Y[1]], [[C_Y[0, 0], C_Y[0, 1]], [C_Y[1, 0], C_Y[1, 1]]])
    Z_y = rv_y.pdf(pos_y)
    
    # Plot contours
    plt.contourf(Y1, Y2, Z_y, levels=15, cmap='Greens', alpha=0.8)
    plt.colorbar(label='Probability Density')
    
    # Add contour lines
    contour_y = plt.contour(Y1, Y2, Z_y, levels=5, colors='black', alpha=0.7)
    plt.clabel(contour_y, inline=True, fontsize=8)
    
    # Add mean point
    plt.scatter(mu_Y[0], mu_Y[1], color='red', s=50, marker='x', label='Mean')
    
    # Set labels and title
    plt.title('Linear Transformation Distribution (Y1-Y2 Plane)')
    plt.xlabel('Y1 = 2X1')
    plt.ylabel('Y2 = X1 + 2X2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    save_path3 = os.path.join(images_dir, "example4_linear_transformation.png")
    plt.savefig(save_path3, bbox_inches='tight', dpi=300)
    print(f"Linear transformation visualization saved to: {save_path3}")
    plt.close()
    saved_images.append(save_path3)
    
    # Visualization 4: Comparison of original vs transformed distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original X1-X2 distribution on the left
    cf1 = axes[0].contourf(X1, X2, Z, levels=15, cmap='Blues', alpha=0.8)
    axes[0].scatter(mu_X1[0], mu_X1[1], color='red', s=50, marker='x', label='Mean')
    axes[0].set_title('Original X1-X2 Distribution')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot transformed Y1-Y2 distribution on the right
    cf2 = axes[1].contourf(Y1, Y2, Z_y, levels=15, cmap='Greens', alpha=0.8)
    axes[1].scatter(mu_Y[0], mu_Y[1], color='red', s=50, marker='x', label='Mean')
    axes[1].set_title('Transformed Y1-Y2 Distribution')
    axes[1].set_xlabel('Y1 = 2X1')
    axes[1].set_ylabel('Y2 = X1 + 2X2')
    axes[1].grid(True, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(cf1, ax=axes[0], label='Probability Density')
    plt.colorbar(cf2, ax=axes[1], label='Probability Density')
    
    plt.tight_layout()
    
    # Save the figure
    save_path4 = os.path.join(images_dir, "example4_comparison.png")
    plt.savefig(save_path4, bbox_inches='tight', dpi=300)
    print(f"Comparison visualization saved to: {save_path4}")
    plt.close()
    saved_images.append(save_path4)
    
    # Visualization 5: Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrices from covariance matrices
    def cov_to_corr(cov):
        D = np.sqrt(np.diag(np.diag(cov)))
        D_inv = np.linalg.inv(D)
        return D_inv @ cov @ D_inv
    
    corr_X = cov_to_corr(C_X)
    corr_X1 = cov_to_corr(C_X1)
    corr_Y = cov_to_corr(C_Y)
    
    # Plot correlation heatmaps in a grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original correlation matrix
    sns.heatmap(corr_X, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Original X Correlation Matrix')
    
    # X1 correlation matrix
    sns.heatmap(corr_X1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('X1 Components Correlation Matrix')
    
    # Y correlation matrix
    sns.heatmap(corr_Y, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title('Y Linear Combinations Correlation Matrix')
    
    plt.tight_layout()
    
    # Save the figure
    save_path5 = os.path.join(images_dir, "example4_correlation_matrices.png")
    plt.savefig(save_path5, bbox_inches='tight', dpi=300)
    print(f"Correlation matrices visualization saved to: {save_path5}")
    plt.close()
    saved_images.append(save_path5)
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Example 4:")
    print("="*60)
    print("1. When extracting components of a multivariate normal vector, the resulting")
    print("   distribution is also multivariate normal.")
    print("2. The covariance matrix of the extracted components is simply the corresponding")
    print("   submatrix of the original covariance matrix.")
    print("3. Linear combinations of multivariate normal variables follow multivariate normal")
    print("   distributions with transformed mean vectors and covariance matrices.")
    print("4. The general formula Y = AX + b applies to both component extraction and")
    print("   linear combinations, with appropriate choices of the matrix A.")
    
    return saved_images

if __name__ == "__main__":
    example4_component_extraction() 