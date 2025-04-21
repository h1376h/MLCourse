import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N

def create_correlation_sequence_visualization():
    """Create a sequence of 3D plots showing how correlation affects the surface."""
    # Create a figure with subplots for different correlation values
    fig = plt.figure(figsize=(18, 12))
    
    # Create a grid of points
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    mu = np.array([0., 0.])
    
    # Define correlation values to visualize
    correlation_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
    
    # Plot for each correlation value
    for i, rho in enumerate(correlation_values):
        # Create covariance matrix with given correlation
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        
        # Add 3D plot
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
        
        # Plot contours on the bottom
        ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
        
        # Set title and labels
        ax.set_title(f'Correlation ρ = {rho:.1f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.view_init(30, 45)
        
        # Calculate and display key properties
        det = np.linalg.det(Sigma)
        max_height = 1.0 / (2 * np.pi * np.sqrt(det))
        
        # Add text annotation
        ax.text(-3, -3, max_height/2, f'Peak Height = {max_height:.3f}\nDeterminant = {det:.3f}', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        
    # Add a summary plot showing eigenvalues vs correlation
    ax_summary = fig.add_subplot(2, 3, 6)
    rho_range = np.linspace(-0.99, 0.99, 100)
    eigenvalues = np.zeros((len(rho_range), 2))
    
    for i, rho in enumerate(rho_range):
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        eigenvalues[i] = np.linalg.eigvalsh(Sigma)
    
    ax_summary.plot(rho_range, eigenvalues[:, 0], 'r-', label='Smaller Eigenvalue')
    ax_summary.plot(rho_range, eigenvalues[:, 1], 'b-', label='Larger Eigenvalue')
    ax_summary.plot(rho_range, 1 - rho_range**2, 'g--', label='Determinant')
    
    ax_summary.set_title('Effect of Correlation on\nEigenvalues and Determinant')
    ax_summary.set_xlabel('Correlation (ρ)')
    ax_summary.set_ylabel('Value')
    ax_summary.legend()
    ax_summary.grid(True)
    
    plt.tight_layout()
    return fig

def create_cross_section_visualization():
    """Create visualization of PDF cross-sections at different correlation values."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Parameters
    x = np.linspace(-3, 3, 100)
    mu = np.array([0., 0.])
    
    # Define correlation values
    rho_values = [0.0, 0.5, 0.8, 0.95]
    
    # Create cross sections for each correlation
    for i, rho in enumerate(rho_values):
        # Create covariance matrix
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Calculate PDF along x-axis (y=0)
        y_fixed = 0
        z_x_axis = np.zeros_like(x)
        for j, x_val in enumerate(x):
            pos = np.array([x_val, y_fixed])
            fac = np.dot(np.dot(pos - mu, Sigma_inv), pos - mu)
            z_x_axis[j] = np.exp(-fac / 2) / (2 * np.pi * np.sqrt(Sigma_det))
        
        # Calculate PDF along y=x line
        y_equals_x = x
        z_diag = np.zeros_like(x)
        for j, x_val in enumerate(x):
            pos = np.array([x_val, x_val])
            fac = np.dot(np.dot(pos - mu, Sigma_inv), pos - mu)
            z_diag[j] = np.exp(-fac / 2) / (2 * np.pi * np.sqrt(Sigma_det))
        
        # Calculate PDF along y=-x line
        y_equals_neg_x = -x
        z_anti_diag = np.zeros_like(x)
        for j, x_val in enumerate(x):
            pos = np.array([x_val, -x_val])
            fac = np.dot(np.dot(pos - mu, Sigma_inv), pos - mu)
            z_anti_diag[j] = np.exp(-fac / 2) / (2 * np.pi * np.sqrt(Sigma_det))
        
        # Plot cross sections
        ax = axes[i]
        ax.plot(x, z_x_axis, 'b-', label='Cross-section along x-axis (y=0)')
        ax.plot(x, z_diag, 'r-', label='Cross-section along y=x')
        ax.plot(x, z_anti_diag, 'g-', label='Cross-section along y=-x')
        
        # Compute eigenvalues and the peak height
        eigenvalues = np.linalg.eigvalsh(Sigma)
        peak_height = 1 / (2 * np.pi * np.sqrt(Sigma_det))
        
        # Add annotation
        ax.text(1.5, peak_height * 0.8,
               f'ρ = {rho}\nDet(Σ) = {Sigma_det:.3f}\nPeak = {peak_height:.3f}\nλ₁ = {eigenvalues[1]:.2f}, λ₂ = {eigenvalues[0]:.2f}',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Set labels and title
        ax.set_title(f'Cross-Sections for ρ = {rho}')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.grid(True)
        
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_contour_surface_relationship():
    """Create visualization showing the relationship between contours and the 3D surface."""
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Parameters
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    mu = np.array([0., 0.])
    
    # Use three different covariance matrices
    covariance_matrices = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "Identity Covariance\n(No Correlation)"),
        (np.array([[2.0, 0.0], [0.0, 0.5]]), "Diagonal Covariance\n(Different Variances)"),
        (np.array([[1.0, 0.8], [0.8, 1.0]]), "Non-Diagonal Covariance\n(Strong Correlation)")
    ]
    
    # Create visualizations for each covariance matrix
    for i, (Sigma, title) in enumerate(covariance_matrices):
        # Calculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Create 3D surface plot
        ax_3d = fig.add_subplot(2, 3, i+1, projection='3d')
        surf = ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
        
        # Add contours at the bottom
        ax_3d.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
        
        # Set labels and title
        ax_3d.set_title(f'3D Surface: {title}')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Probability Density')
        ax_3d.view_init(30, 45)
        
        # Create corresponding 2D contour plot
        ax_2d = fig.add_subplot(2, 3, i+4)
        contour = ax_2d.contour(X, Y, Z, levels=10, cmap=cm.viridis)
        ax_2d.clabel(contour, inline=True, fontsize=8)
        
        # Add ellipses showing confidence regions
        lambda_, v = np.linalg.eigh(Sigma)  # Use eigh for symmetric matrices
        lambda_ = np.sqrt(lambda_)
        
        for j in range(1, 4):
            ell = Ellipse(xy=(0, 0),
                         width=lambda_[1]*j*2, height=lambda_[0]*j*2,
                         angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            ax_2d.add_patch(ell)
            if j == 2:
                ax_2d.text(0, 0, f"{j}σ", color='red', ha='center', va='center')
        
        # Add correlation lines if appropriate
        if Sigma[0, 1] != 0:
            # For positive correlation
            if Sigma[0, 1] > 0:
                ax_2d.plot([-3, 3], [-3, 3], 'r-', alpha=0.5)
            # For negative correlation
            else:
                ax_2d.plot([-3, 3], [3, -3], 'r-', alpha=0.5)
        
        # Set labels and title
        ax_2d.set_title(f'2D Contours: {title}')
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        ax_2d.grid(True)
        ax_2d.set_xlim(-3, 3)
        ax_2d.set_ylim(-3, 3)
    
    plt.tight_layout()
    return fig

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
    
    # Save the basic figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "ex3_gaussian_3d_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nBasic 3D visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    print("\nStep 5: Creating Enhanced Visualizations")
    print("We'll now create additional visualizations to better understand the 3D structure of these distributions:")
    
    # Create correlation sequence visualization
    print("\n  a) Creating correlation sequence visualization...")
    fig_correlation = create_correlation_sequence_visualization()
    try:
        save_path = os.path.join(images_dir, "ex3_gaussian_3d_correlation_sequence.png")
        fig_correlation.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Correlation sequence visualization saved to: {save_path}")
    except Exception as e:
        print(f"  Error saving figure: {e}")
    
    # Create cross-section visualization
    print("\n  b) Creating cross-section visualization...")
    fig_cross_section = create_cross_section_visualization()
    try:
        save_path = os.path.join(images_dir, "ex3_gaussian_3d_cross_sections.png")
        fig_cross_section.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Cross-section visualization saved to: {save_path}")
    except Exception as e:
        print(f"  Error saving figure: {e}")
    
    # Create contour-surface relationship visualization
    print("\n  c) Creating contour-surface relationship visualization...")
    fig_contour_surface = create_contour_surface_relationship()
    try:
        save_path = os.path.join(images_dir, "ex3_gaussian_3d_contour_relationship.png")
        fig_contour_surface.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Contour-surface relationship visualization saved to: {save_path}")
    except Exception as e:
        print(f"  Error saving figure: {e}")
    
    print("\nStep 6: Comparing All Three 3D Visualizations")
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
    print("\nAdditional insights from enhanced visualizations:")
    print("- As correlation increases, the PDF reshapes to concentrate probability mass along")
    print("  the correlation direction (y=x for positive correlation)")
    print("- Cross-sections reveal that the PDF becomes more peaked along certain directions")
    print("  and flatter along others based on correlation")
    print("- The relationship between 2D contours and the 3D surface shows how the surface")
    print("  height directly corresponds to the probability density at each point")
    
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
        save_path = os.path.join(images_dir, "ex3_gaussian_3d_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    