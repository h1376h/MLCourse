import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Multivariate_Eigendecomposition relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Eigendecomposition")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def plot_covariance_ellipse(cov_matrix, mean, title, filename, transform_matrix=None):
    """Plot covariance ellipse with eigenvectors."""
    # Generate points on a circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    
    # Transform circle using covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    transform = np.sqrt(eigenvals)[:, None] * eigenvecs
    ellipse = transform @ circle
    
    # If transform matrix is provided, apply it
    if transform_matrix is not None:
        ellipse = transform_matrix @ ellipse
    
    # Plot with improved aesthetics
    plt.figure(figsize=(10, 10))
    plt.plot(ellipse[0, :] + mean[0], ellipse[1, :] + mean[1], 'b-', 
             label='Covariance Ellipse', linewidth=2)
    
    # Plot eigenvectors with better visibility
    if transform_matrix is None:
        for i in range(2):
            plt.quiver(mean[0], mean[1],
                      eigenvecs[0, i] * np.sqrt(eigenvals[i]),
                      eigenvecs[1, i] * np.sqrt(eigenvals[i]),
                      angles='xy', scale_units='xy', scale=1,
                      color=['r', 'g'][i], label=f'Eigenvector {i+1}',
                      width=0.008, headwidth=3)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.legend(fontsize=10)
    
    # Add clean white background
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    plt.savefig(os.path.join(images_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_matrix(name, matrix):
    """Print matrix with proper formatting."""
    print(f"\n{name}:")
    print("┌" + "─" * (matrix.shape[1] * 8 + 1) + "┐")
    for row in matrix:
        print("│ " + " ".join([f"{x:6.3f}" for x in row]) + " │")
    print("└" + "─" * (matrix.shape[1] * 8 + 1) + "┘")

def example1():
    """Example 1: Eigenvalue Decomposition of 2x2 Covariance Matrix"""
    print("\nExample 1: 2x2 Covariance Matrix Eigendecomposition")
    print("=" * 50)
    
    # Define covariance matrix
    sigma = np.array([[9, 5],
                     [5, 4]])
    print_matrix("Covariance Matrix Σ", sigma)
    
    print("\nStep 1: Find eigenvalues by solving |Σ - λI| = 0")
    print("det|Σ - λI| = det|[9-λ  5  ]| = 0")
    print("              |[5    4-λ]|")
    print("\n(9-λ)(4-λ) - 5×5 = 0")
    print("36 - 13λ + λ² - 25 = 0")
    print("λ² - 13λ + 11 = 0")
    
    # Calculate eigenvalues
    eigenvals = np.linalg.eigvalsh(sigma)
    print("\nSolving the quadratic equation:")
    print(f"λ = (-(-13) ± √(13² - 4×1×11)) / (2×1)")
    print(f"λ = (13 ± √(169 - 44)) / 2")
    print(f"λ = (13 ± √125) / 2")
    print(f"λ = (13 ± {np.sqrt(125):.3f}) / 2")
    print(f"\nλ₁ = {eigenvals[1]:.3f}")
    print(f"λ₂ = {eigenvals[0]:.3f}")
    
    print("\nStep 2: Find eigenvectors by solving (Σ - λᵢI)vᵢ = 0")
    eigenvecs = np.linalg.eigh(sigma)[1]
    
    # For λ₁
    print(f"\nFor λ₁ = {eigenvals[1]:.3f}:")
    A1 = sigma - eigenvals[1] * np.eye(2)
    print(f"[{A1[0,0]:.3f}  {A1[0,1]:.3f}][v₁₁] = [0]")
    print(f"[{A1[1,0]:.3f}  {A1[1,1]:.3f}][v₁₂]   [0]")
    
    print("\nSolving and normalizing:")
    print(f"v₁ = [{eigenvecs[0,1]:.3f}]")
    print(f"     [{eigenvecs[1,1]:.3f}]")
    
    # For λ₂
    print(f"\nFor λ₂ = {eigenvals[0]:.3f}:")
    A2 = sigma - eigenvals[0] * np.eye(2)
    print(f"[{A2[0,0]:.3f}  {A2[0,1]:.3f}][v₂₁] = [0]")
    print(f"[{A2[1,0]:.3f}  {A2[1,1]:.3f}][v₂₂]   [0]")
    
    print("\nSolving and normalizing:")
    print(f"v₂ = [{eigenvecs[0,0]:.3f}]")
    print(f"     [{eigenvecs[1,0]:.3f}]")
    
    # Verify eigendecomposition
    P = eigenvecs
    Lambda = np.diag(eigenvals)
    reconstructed = P @ Lambda @ P.T
    
    print("\nStep 3: Verify eigendecomposition Σ = PΛPᵀ")
    print_matrix("P (eigenvectors as columns)", P)
    print_matrix("Λ (diagonal matrix of eigenvalues)", Lambda)
    print_matrix("Reconstructed Σ = PΛPᵀ", reconstructed)
    print(f"\nMaximum reconstruction error: {np.abs(sigma - reconstructed).max():.2e}")
    
    # Plot original covariance ellipse
    plot_covariance_ellipse(sigma, np.array([0, 0]),
                           'Original Covariance Ellipse with Eigenvectors',
                           'example1_original')
    
    # Calculate and plot whitened data
    whitening_matrix = P @ np.diag(1/np.sqrt(eigenvals)) @ P.T
    plot_covariance_ellipse(sigma, np.array([0, 0]),
                           'Whitened Data',
                           'example1_whitened',
                           whitening_matrix)

def example2():
    """Example 2: 3x3 Covariance Matrix Analysis"""
    print("\nExample 2: 3x3 Covariance Matrix Analysis")
    print("=" * 50)
    
    # Define covariance matrix
    sigma = np.array([[10, 7, 3],
                      [7, 8, 2],
                      [3, 2, 5]])
    print_matrix("Covariance Matrix Σ", sigma)
    
    print("\nStep 1: Find eigenvalues and eigenvectors")
    eigenvals, eigenvecs = np.linalg.eigh(sigma)
    
    # Sort in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print("\nEigenvalues (in descending order):")
    for i, val in enumerate(eigenvals):
        print(f"λ{i+1} = {val:.3f}")
    
    print("\nEigenvectors (as columns):")
    print_matrix("P", eigenvecs)
    
    # Calculate variance explained
    total_variance = np.sum(eigenvals)
    variance_explained = eigenvals / total_variance
    cumulative_variance = np.cumsum(variance_explained)
    
    print("\nStep 2: Calculate variance explained by each component")
    print(f"Total variance (sum of eigenvalues): {total_variance:.3f}")
    print("\nVariance explained by each component:")
    for i, var in enumerate(variance_explained):
        print(f"PC{i+1}: {var*100:.2f}% ({eigenvals[i]:.3f}/{total_variance:.3f})")
    
    print("\nCumulative variance explained:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"First {i+1} PC{'s' if i>0 else ''}: {cum_var*100:.2f}%")
    
    # Plot variance explained with improved aesthetics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, 4), variance_explained * 100, 
                  color='skyblue', alpha=0.7)
    plt.plot(range(1, 4), cumulative_variance * 100, 'ro-', 
             label='Cumulative', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.xlabel('Principal Component', fontsize=10)
    plt.ylabel('Variance Explained (%)', fontsize=10)
    plt.title('Variance Explained by Principal Components', fontsize=12, pad=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.savefig(os.path.join(images_dir, 'example2_variance_explained.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def example3():
    """Example 3: Whitening Transformation"""
    print("\nExample 3: Whitening Transformation")
    print("=" * 50)
    
    # Define covariance matrix
    sigma = np.array([[6, 2],
                      [2, 4]])
    print_matrix("Original Covariance Matrix Σ", sigma)
    
    print("\nStep 1: Find eigendecomposition")
    eigenvals, eigenvecs = np.linalg.eigh(sigma)
    
    # Sort in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print("\nEigenvalues:")
    for i, val in enumerate(eigenvals):
        print(f"λ{i+1} = {val:.3f}")
    
    print("\nEigenvectors (as columns):")
    print_matrix("P", eigenvecs)
    
    print("\nStep 2: Calculate whitening matrix W = PΛ⁻¹/²Pᵀ")
    print("\nΛ⁻¹/² (diagonal matrix of 1/√λᵢ):")
    Lambda_inv_sqrt = np.diag(1/np.sqrt(eigenvals))
    print_matrix("Λ⁻¹/²", Lambda_inv_sqrt)
    
    # Calculate whitening matrix
    whitening_matrix = eigenvecs @ Lambda_inv_sqrt @ eigenvecs.T
    print("\nWhitening matrix W:")
    print_matrix("W", whitening_matrix)
    
    # Verify whitening transformation
    whitened_cov = whitening_matrix @ sigma @ whitening_matrix.T
    print("\nStep 3: Verify that WΣWᵀ = I")
    print_matrix("WΣWᵀ", whitened_cov)
    print(f"\nMaximum deviation from identity matrix: {np.abs(whitened_cov - np.eye(2)).max():.2e}")
    
    # Plot original and whitened distributions
    plot_covariance_ellipse(sigma, np.array([0, 0]),
                           'Original Covariance Ellipse',
                           'example3_original')
    
    plot_covariance_ellipse(sigma, np.array([0, 0]),
                           'Whitened Distribution',
                           'example3_whitened',
                           whitening_matrix)

def main():
    print("=== Eigendecomposition Examples ===")
    example1()
    example2()
    example3()
    print("\nAll examples completed. Check the Images/Eigendecomposition directory for visualizations.")

if __name__ == "__main__":
    main() 