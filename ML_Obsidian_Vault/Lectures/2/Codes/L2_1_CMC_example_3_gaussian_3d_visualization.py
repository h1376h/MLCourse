import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

def gaussian_3d_visualization():
    """Create 3D visualization of Gaussian probability density functions"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: 3D Visualization of Multivariate Gaussians")
    print("="*80)
    
    print("\nStep 1: Setting Up the Visualization Framework")
    print("To visualize bivariate normal distributions in 3D, we need to:")
    print("- Create a 2D grid of (x,y) points where we'll evaluate the PDF")
    print("- Calculate the PDF value at each point, giving us a 3D surface z = f(x,y)")
    print("- Plot this surface in 3D space, with contours projected on the xy-plane")
    print("\nThis gives us a comprehensive view of both the probability density surface")
    print("and its contour lines, helping us understand the distribution's shape")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Create a grid of points
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    print("\nStep 2: Case 1 - Standard Bivariate Normal (Identity Covariance)")
    print("For a standard bivariate normal distribution:")
    print("- Mean vector: μ = [0, 0] (centered at the origin)")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]] (identity matrix)")
    print("- PDF: f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    print("\nKey properties of the 3D surface:")
    print("- The peak occurs at (0,0) with a value of 1/(2π) ≈ 0.159")
    print("- The surface has perfect radial symmetry around the z-axis")
    print("- The contours projected onto the xy-plane form perfect circles")
    print("- The surface falls off equally in all directions from the peak")
    print("- The volume under the entire surface equals 1 (probability axiom)")
    
    # Case 1: Standard Normal Distribution (Identity Covariance)
    ax1 = fig.add_subplot(131, projection='3d')
    mu1 = np.array([0., 0.])
    Sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    Z1 = multivariate_gaussian(pos, mu1, Sigma1)
    
    # Plot the surface
    surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax1.contour(X, Y, Z1, zdir='z', offset=0, cmap=cm.viridis)
    
    ax1.set_title('Standard Bivariate Normal\n(Identity Covariance)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Probability Density')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.view_init(30, 45)
    
    print("\nStep 3: Case 2 - Bivariate Normal with Different Variances")
    print("For a bivariate normal with different variances:")
    print("- Mean vector: μ = [0, 0] (still centered at the origin)")
    print("- Covariance matrix: Σ = [[2.0, 0], [0, 0.5]] (diagonal but unequal)")
    print("- PDF: f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * (x²/2 + y²/0.5))")
    print("- Determinant |Σ| = 2.0 * 0.5 = 1.0")
    print("\nKey properties of the 3D surface:")
    print("- The peak still occurs at (0,0) with the same height as Case 1")
    print("- The surface is stretched along the x-axis and compressed along the y-axis")
    print("- The contours projected onto the xy-plane form axis-aligned ellipses")
    print("- The surface falls off more slowly in the x-direction (larger variance)")
    print("- The surface falls off more quickly in the y-direction (smaller variance)")
    print("- The volume under the surface still equals 1")
    
    # Case 2: Diagonal Covariance with Different Variances
    ax2 = fig.add_subplot(132, projection='3d')
    mu2 = np.array([0., 0.])
    Sigma2 = np.array([[2.0, 0.0], [0.0, 0.5]])
    Z2 = multivariate_gaussian(pos, mu2, Sigma2)
    
    # Plot the surface
    surf2 = ax2.plot_surface(X, Y, Z2, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax2.contour(X, Y, Z2, zdir='z', offset=0, cmap=cm.viridis)
    
    ax2.set_title('Bivariate Normal with Different Variances\n(Diagonal Covariance)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Probability Density')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.view_init(30, 45)
    
    print("\nStep 4: Case 3 - Bivariate Normal with Correlation")
    print("For a bivariate normal with correlation:")
    print("- Mean vector: μ = [0, 0]")
    print("- Covariance matrix: Σ = [[1.0, 0.8], [0.8, 1.0]] (non-diagonal)")
    corr = 0.8 / np.sqrt(1.0 * 1.0)
    print(f"- Correlation coefficient: ρ = {corr:.2f} (strong positive correlation)")
    print("- PDF: f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("- Determinant |Σ| = 1.0² - 0.8² = 0.36")
    print("\nKey properties of the 3D surface:")
    print("- The peak still occurs at (0,0), but its height is different due to the determinant")
    print("- The surface is tilted, with its principal axes rotated from the coordinate axes")
    print("- The contours projected onto the xy-plane form rotated ellipses")
    print("- The primary direction of spread is along the y = x line (reflecting positive correlation)")
    print("- The surface shows that x and y tend to increase or decrease together")
    print("- The correlation creates a 'ridge' along the y = x direction")
    print("- The volume under the surface still equals 1")
    
    # Case 3: Non-diagonal Covariance with Correlation
    ax3 = fig.add_subplot(133, projection='3d')
    mu3 = np.array([0., 0.])
    Sigma3 = np.array([[1.0, 0.8], [0.8, 1.0]])
    Z3 = multivariate_gaussian(pos, mu3, Sigma3)
    
    # Plot the surface
    surf3 = ax3.plot_surface(X, Y, Z3, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax3.contour(X, Y, Z3, zdir='z', offset=0, cmap=cm.viridis)
    
    # Calculate correlation coefficient
    corr = Sigma3[0, 1] / np.sqrt(Sigma3[0, 0] * Sigma3[1, 1])
    
    ax3.set_title(f'Bivariate Normal with Correlation\n(ρ = {corr:.2f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Probability Density')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.view_init(30, 45)
    
    print("\nStep 5: Comparing All Three 3D Visualizations")
    print("Key insights from comparing these 3D surfaces:")
    print("1. The covariance matrix directly determines the shape and orientation of the PDF surface")
    print("2. Identity covariance (Case 1): Symmetric bell shape with circular contours")
    print("3. Diagonal covariance with different variances (Case 2): Stretched bell shape")
    print("   with axis-aligned elliptical contours")
    print("4. Non-diagonal covariance with correlation (Case 3): Tilted bell shape")
    print("   with rotated elliptical contours")
    print("\nMathematical relationships:")
    print("- The exponent term in the PDF formula: -1/2 * (x,y)ᵀ Σ⁻¹ (x,y) creates the shape")
    print("- The determinant term in the denominator: √|Σ| adjusts the height of the peak")
    print("- Together they ensure that the volume under the surface equals 1")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 3: 3D VISUALIZATION OF GAUSSIAN PDFs")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = gaussian_3d_visualization()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "gaussian_3d_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    