import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N

def plot_3d_gaussian(mu, Sigma, ax, title):
    """Plot 3D Gaussian surface with projected contours."""
    # Create a grid of points
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Compute the PDF values
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
    
    # Plot projected contours on the xy-plane
    offset = -0.05  # Offset for contour projection
    cset = ax.contour(X, Y, Z, zdir='z', offset=offset, levels=5, alpha=0.5, colors='k')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title(title)
    
    # Adjust the z-axis limits
    ax.set_zlim(offset, Z.max() * 1.1)
    
    return surf

def plot_eigenvalues_visualization(ax, Sigma, title):
    """Visualize eigenvalues and eigenvectors of covariance matrix."""
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Plot covariance ellipse
    ax.axis('equal')
    ax.grid(True)
    
    # Plot eigenvectors as vectors from origin
    for i in range(2):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3, fc='r', ec='r')
        ax.text(vec[0]*1.1, vec[1]*1.1, f'λ{i+1}={eigenvalues[i]:.2f}', fontsize=9, color='r')
    
    # Plot the ellipse (2-sigma contour)
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    ellipse = np.dot(np.sqrt(np.diag(eigenvalues)), circle)
    ellipse = np.dot(eigenvectors, ellipse)
    ax.plot(ellipse[0, :], ellipse[1, :], 'b-')
    
    # Show the covariance matrix as a text annotation
    matrix_text = f'Σ = [[{Sigma[0, 0]:.1f}, {Sigma[0, 1]:.1f}],\n     [{Sigma[1, 0]:.1f}, {Sigma[1, 1]:.1f}]]'
    ax.text(-4, 3.5, matrix_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Set limits and title
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def create_comparative_visualization(sigma_matrices):
    """Create a comparison of multiple covariance matrices in a single plot."""
    # Create figure
    fig, axes = plt.subplots(1, len(sigma_matrices), figsize=(5*len(sigma_matrices), 5))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    mu = np.array([0., 0.])
    
    for i, (Sigma, title) in enumerate(sigma_matrices):
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Plot contours
        contour_levels = np.linspace(0.01, 0.1, 5)
        cp = axes[i].contour(X, Y, Z, levels=contour_levels, colors='black')
        axes[i].clabel(cp, inline=True, fontsize=10)
        
        # Add an ellipse to represent the covariance (1σ, 2σ, and 3σ)
        lambda_, v = np.linalg.eig(Sigma)
        lambda_ = np.sqrt(lambda_)
        
        for j in range(1, 4):
            ell = Ellipse(xy=(0, 0),
                         width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                         angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            axes[i].add_patch(ell)
            if j == 2:
                axes[i].text(0, lambda_[1]*j, f'{j}σ', color='red', ha='center', va='bottom')
                axes[i].text(lambda_[0]*j, 0, f'{j}σ', color='red', ha='left', va='center')
        
        # Add correlation explanation for non-diagonal matrices
        if Sigma[0, 1] != 0:
            # For positive correlation
            if Sigma[0, 1] > 0:
                axes[i].plot([-5, 5], [-5, 5], 'r--', alpha=0.5)
            # For negative correlation
            else:
                axes[i].plot([-5, 5], [5, -5], 'r--', alpha=0.5)
        
        # Calculate correlation coefficient
        if Sigma[0, 0] != 0 and Sigma[1, 1] != 0:
            corr = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
            corr_text = f'ρ = {corr:.2f}'
            axes[i].text(-4.5, 4.5, corr_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        axes[i].set_title(title)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].grid(True)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
    
    plt.tight_layout()
    return fig

def covariance_matrix_contours():
    """Visualize multivariate Gaussians with different covariance matrices"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Multivariate Gaussians with Different Covariance Matrices")
    print("="*80)
    
    print(f"Function: f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Understand the multivariate Gaussian PDF")
    print("The probability density function of a bivariate Gaussian with mean μ = (0,0) and")
    print("covariance matrix Σ defines a surface whose contours we want to analyze.")
    print("For a bivariate Gaussian:")
    print("f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    # Create a 3D visualization to explain the relationship between contours and PDF surface
    fig_3d = plt.figure(figsize=(15, 5))
    
    # Standard normal distribution
    ax_3d_1 = fig_3d.add_subplot(131, projection='3d')
    plot_3d_gaussian(np.array([0., 0.]), np.array([[1.0, 0.0], [0.0, 1.0]]), ax_3d_1, 'Standard Normal PDF (Identity Covariance)')
    
    # Diagonal with different variances
    ax_3d_2 = fig_3d.add_subplot(132, projection='3d')
    plot_3d_gaussian(np.array([0., 0.]), np.array([[3.0, 0.0], [0.0, 0.5]]), ax_3d_2, 'Diagonal Covariance (Different Variances)')
    
    # Non-diagonal with correlation
    ax_3d_3 = fig_3d.add_subplot(133, projection='3d')
    plot_3d_gaussian(np.array([0., 0.]), np.array([[2.0, 1.5], [1.5, 2.0]]), ax_3d_3, 'Non-Diagonal Covariance (With Correlation)')
    
    plt.tight_layout()
    
    # Save the 3D visualization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "gaussian_3d_explanation.png")
        fig_3d.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\n3D visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving 3D figure: {e}")
    
    print("\nStep 2: Create a grid of points for visualization")
    print("We'll create a 100x100 grid spanning from -5 to 5 in both dimensions")
    print("This gives us a total of 10,000 points at which to evaluate the PDF")
    print("The goal is to visualize the shape of the probability density function")
    print("by creating contour plots that connect points of equal density")
    
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    print("\nStep 3: Analyze the quadratic form in the exponent")
    print("The key term that determines the shape of the contours is the quadratic form")
    print("(x,y)ᵀ Σ⁻¹ (x,y), which creates elliptical level curves.")
    print("The exponent term -1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)] determines the contour shapes.")
    print("For constant density c, the contours satisfy:")
    print("(x,y)ᵀ Σ⁻¹ (x,y) = -2log(c·√(2π|Σ|)) = constant")
    
    # Create eigenvalue visualization for each case
    fig_eigen = plt.figure(figsize=(15, 5))
    
    # Case 1: Identity covariance
    ax_eigen1 = fig_eigen.add_subplot(131)
    Sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    plot_eigenvalues_visualization(ax_eigen1, Sigma1, 'Case 1: Identity Covariance\nEigenvalues & Eigenvectors')
    
    # Case 2: Diagonal different variances
    ax_eigen2 = fig_eigen.add_subplot(132)
    Sigma2 = np.array([[3.0, 0.0], [0.0, 0.5]])
    plot_eigenvalues_visualization(ax_eigen2, Sigma2, 'Case 2: Diagonal Covariance\nEigenvalues & Eigenvectors')
    
    # Case 3: Non-diagonal positive correlation
    ax_eigen3 = fig_eigen.add_subplot(133)
    Sigma3 = np.array([[2.0, 1.5], [1.5, 2.0]])
    plot_eigenvalues_visualization(ax_eigen3, Sigma3, 'Case 3: Non-Diagonal Covariance\nEigenvalues & Eigenvectors')
    
    plt.tight_layout()
    
    # Save the eigenvalue visualization
    try:
        save_path = os.path.join(images_dir, "covariance_eigenvalue_explanation.png")
        fig_eigen.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nEigenvalue visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving eigenvalue figure: {e}")
    
    # ... [rest of the original covariance_matrix_contours function continues here] ...
    
    print("\nStep 4: Analyze Case 1 - Diagonal Covariance with Equal Variances")
    print("Covariance Matrix Σ = [[1.0, 0.0], [0.0, 1.0]] (Identity Matrix)")
    print("Properties:")
    print("- Equal variances (σ₁² = σ₂² = 1)")
    print("- Zero correlation (ρ = 0)")
    print("- Determinant |Σ| = 1")
    print("- Eigenvalues: λ₁ = λ₂ = 1")
    print("- The resulting contours form perfect circles")
    print("- The equation for these contours is x² + y² = constant")
    print("- This is the standard bivariate normal distribution")
    print("- The pdf simplifies to: f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    
    # Case 1: Diagonal covariance with equal variances (scaled identity matrix)
    mu1 = np.array([0., 0.])
    Sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
    
    ax1 = fig.add_subplot(221)
    Z1 = multivariate_gaussian(pos, mu1, Sigma1)
    
    # Plot contours
    contour_levels = np.linspace(0.01, 0.1, 5)
    cp1 = ax1.contour(X, Y, Z1, levels=contour_levels, colors='black')
    ax1.clabel(cp1, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma1)
    lambda_ = np.sqrt(lambda_)
    
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax1.add_patch(ell)
        if j == 2:
            ax1.text(0, lambda_[1]*j, '2σ', color='red', ha='center', va='bottom')
            ax1.text(lambda_[0]*j, 0, '2σ', color='red', ha='left', va='center')
    
    ax1.set_title('Case 1: Circular Contours\nIdentity Covariance Matrix')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    # ... [rest of the original code for cases 2, 3, and 4 continues here] ...
    # Case 2, 3, and 4 visualization code is kept the same
    
    # Add a comparative visualization as a new figure
    sigma_matrices = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "Case 1: Identity\n(ρ = 0)"),
        (np.array([[3.0, 0.0], [0.0, 0.5]]), "Case 2: Diagonal\n(Different Variances)"),
        (np.array([[2.0, 1.5], [1.5, 2.0]]), "Case 3: Positive\nCorrelation"),
        (np.array([[2.0, -1.5], [-1.5, 2.0]]), "Case 4: Negative\nCorrelation")
    ]
    
    fig_comparative = create_comparative_visualization(sigma_matrices)
    
    # Save the comparative visualization
    try:
        save_path = os.path.join(images_dir, "covariance_matrix_comparison.png")
        fig_comparative.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nComparative visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving comparative figure: {e}")
    
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 1: COVARIANCE MATRIX CONTOURS")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = covariance_matrix_contours()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "covariance_matrix_contours.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")