import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def covariance_eigenvalue_visualization():
    """Visualize the relationship between covariance matrices, eigenvalues, and eigenvectors"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Eigenvalues, Eigenvectors, and Covariance Effects")
    print("="*80)
    
    print("\nMathematical Background:")
    print("- Covariance matrix Σ can be decomposed as Σ = VΛV^T")
    print("- V contains eigenvectors (principal directions)")
    print("- Λ is a diagonal matrix of eigenvalues (variance along principal directions)")
    
    fig = plt.figure(figsize=(15, 15))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Define covariance matrices with increasing correlation
    mu = np.array([0., 0.])
    correlations = [0, 0.3, 0.6, 0.9]
    titles = ["No Correlation (ρ = 0)", 
              "Weak Correlation (ρ = 0.3)", 
              "Moderate Correlation (ρ = 0.6)", 
              "Strong Correlation (ρ = 0.9)"]
    
    print("\nStep 1: Example with No Correlation (ρ = 0)")
    print("Key Points:")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]]")
    print("- Eigenvalues: λ₁ = λ₂ = 1")
    print("- Eigenvectors align with the coordinate axes")
    print("- Circular contours indicate equal variance in all directions")
    print("- No preferred direction of variability in the data")
    
    print("\nStep 2: Example with Weak Correlation (ρ = 0.3)")
    print("Key Points:")
    print("- Covariance matrix: Σ = [[1, 0.3], [0.3, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.3, λ₂ ≈ 0.7")
    print("- Eigenvectors begin to rotate from the coordinate axes")
    print("- Slightly elliptical contours with mild rotation")
    print("- Beginning of a preferred direction of variability")
    
    print("\nStep 3: Example with Moderate Correlation (ρ = 0.6)")
    print("Key Points:")
    print("- Covariance matrix: Σ = [[1, 0.6], [0.6, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.6, λ₂ ≈ 0.4")
    print("- Eigenvectors rotate further from the coordinate axes")
    print("- More eccentric elliptical contours with significant rotation")
    print("- Clear preferred direction of variability emerges")
    
    print("\nStep 4: Example with Strong Correlation (ρ = 0.9)")
    print("Key Points:")
    print("- Covariance matrix: Σ = [[1, 0.9], [0.9, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.9, λ₂ ≈ 0.1")
    print("- Eigenvectors nearly align with the y = x and y = -x directions")
    print("- Highly eccentric elliptical contours with strong rotation")
    print("- Dominant direction of variability along the first eigenvector")
    print("- Very little variability along the second eigenvector")
    
    for i, (corr, title) in enumerate(zip(correlations, titles), 1):
        # Create covariance matrix
        Sigma = np.array([
            [1.0, corr],
            [corr, 1.0]
        ])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        
        # Calculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Plot
        ax = fig.add_subplot(2, 2, i)
        
        # Plot contours
        contour_levels = np.linspace(0.01, 0.2, 5)
        cp = ax.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax.clabel(cp, inline=True, fontsize=10)
        
        # Plot ellipses
        for j in range(1, 3):
            ell = Ellipse(xy=(0, 0),
                         width=np.sqrt(eigenvalues[0])*j*2, 
                         height=np.sqrt(eigenvalues[1])*j*2,
                         angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(ell)
        
        # Plot eigenvectors
        for j in range(2):
            vec = eigenvectors[:, j] * np.sqrt(eigenvalues[j]) * 2
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3, 
                     fc='blue', ec='blue', label=f'Eigenvector {j+1}')
            ax.text(vec[0]*1.1, vec[1]*1.1, f'λ{j+1}={eigenvalues[j]:.2f}', 
                    color='blue', ha='center', va='center')
        
        ax.set_title(f'{title}\nEigenvalues: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    print("\nStep 5: Analyzing the Patterns")
    print("Key Insights:")
    print("- As correlation increases, eigenvalues become more disparate")
    print("- The largest eigenvalue increases, the smallest decreases")
    print("- The orientation of eigenvectors approaches y = x (for positive correlation)")
    print("- The ellipses become increasingly elongated (higher eccentricity)")
    print("- This illustrates why PCA works: it identifies the directions of maximum variance")
    
    print("\nStep 6: Mathematical Relationships")
    print("For a covariance matrix with equal variances and correlation ρ:")
    print("Σ = [[1, ρ], [ρ, 1]]")
    print("\nThe eigenvalues are:")
    print("λ₁ = 1 + ρ")
    print("λ₂ = 1 - ρ")
    print("\nThe eigenvectors are:")
    print("v₁ = [1, 1]/√2")
    print("v₂ = [-1, 1]/√2")
    print("\nAs ρ approaches 1:")
    print("- λ₁ approaches 2 (representing the variance along [1,1])")
    print("- λ₂ approaches 0 (representing the variance along [-1,1])")
    print("- The distribution becomes increasingly concentrated along y = x")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 4: EIGENVALUES, EIGENVECTORS, AND COVARIANCE EFFECTS")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = covariance_eigenvalue_visualization()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "covariance_eigenvalue_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    