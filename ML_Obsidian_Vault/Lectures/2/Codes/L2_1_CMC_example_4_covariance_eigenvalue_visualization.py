import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

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

def create_concept_visualization():
    """Create a visual explanation of eigenvalues and eigenvectors of covariance matrices."""
    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Three cases: circular, elongated along x-axis, and correlated
    covariance_matrices = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "Identity Covariance\nEqual Eigenvalues"),
        (np.array([[3.0, 0.0], [0.0, 0.5]]), "Diagonal Covariance\nUnequal Eigenvalues"),
        (np.array([[2.0, 1.5], [1.5, 2.0]]), "Correlated Variables\nRotated Eigenvectors")
    ]
    
    # For each case, plot the eigenvalue visualization
    for i, (Sigma, title) in enumerate(covariance_matrices):
        ax = axs[i]
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        
        # Plot the contour
        t = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(t), np.sin(t)])
        ellipse = np.dot(np.sqrt(np.diag(eigenvalues)), circle)
        ellipse = np.dot(eigenvectors, ellipse)
        ax.plot(ellipse[0, :], ellipse[1, :], 'b-', linewidth=2)
        
        # Plot eigenvectors
        for j in range(2):
            vec = eigenvectors[:, j] * np.sqrt(eigenvalues[j])
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3, 
                    fc='r', ec='r', linewidth=2)
            # Print eigenvalue information instead of annotating
            print(f"{title} - Eigenvalue {j+1}: λ{j+1}={eigenvalues[j]:.2f}")
        
        # Draw coordinate axes
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Print covariance matrix info instead of adding it to the plot
        print(f"Covariance matrix for {title}:")
        print(f"Σ = [[{Sigma[0,0]:.1f}, {Sigma[0,1]:.1f}],")
        print(f"    [{Sigma[1,0]:.1f}, {Sigma[1,1]:.1f}]]")
        print()
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_eigenvalue_trend():
    """Create a plot showing how eigenvalues change with correlation coefficient."""
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Generate correlation values
    rho_values = np.linspace(-0.99, 0.99, 100)
    lambda1_values = 1 + rho_values
    lambda2_values = 1 - rho_values
    determinant_values = lambda1_values * lambda2_values  # should all be 1-rho^2
    
    # Plot eigenvalues vs correlation
    ax.plot(rho_values, lambda1_values, 'b-', linewidth=2, label='λ₁ = 1 + ρ')
    ax.plot(rho_values, lambda2_values, 'r-', linewidth=2, label='λ₂ = 1 - ρ')
    ax.plot(rho_values, determinant_values, 'g--', linewidth=2, label='Det(Σ) = 1 - ρ²')
    
    # Add markers for specific correlation values used in the main visualization
    special_rhos = [0, 0.3, 0.6, 0.9]
    for rho in special_rhos:
        lambda1 = 1 + rho
        lambda2 = 1 - rho
        det = 1 - rho**2
        ax.plot(rho, lambda1, 'bo', markersize=8)
        ax.plot(rho, lambda2, 'ro', markersize=8)
        ax.plot(rho, det, 'go', markersize=8)
        # Print the values instead of annotating
        print(f"At ρ = {rho}:")
        print(f"  λ₁ = {lambda1:.2f}")
        print(f"  λ₂ = {lambda2:.2f}")
        print(f"  Det(Σ) = {det:.2f}")
        print()
    
    # Print key observations instead of adding annotations
    print("Key observations about eigenvalues and correlation:")
    print("- At ρ = 0: Equal variance in all directions (λ₁ = λ₂ = 1)")
    print("- As ρ approaches 1: Maximum variance along y=x direction (λ₁ increases)")
    print("- As ρ approaches 1: Minimum variance perpendicular to y=x (λ₂ decreases)")
    print("- As correlation increases:")
    print("  • λ₁ increases - more variance along y=x")
    print("  • λ₂ decreases - less variance perpendicular to y=x")
    print("  • The determinant decreases - narrower overall distribution")
    print()
    
    # Formatting
    ax.set_xlabel('Correlation coefficient (ρ)')
    ax.set_ylabel('Value')
    ax.set_title('Eigenvalues vs Correlation Coefficient\nfor a 2×2 Covariance Matrix with Equal Variances')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.3, 2.3)
    ax.legend()
    
    return fig

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
    
    # Create a visualization to illustrate the concept
    concept_fig = create_concept_visualization()
    
    # Create a visualization showing eigenvalue trends with correlation
    trend_fig = plot_eigenvalue_trend()
    
    # Main visualization - create a figure with GridSpec for better layout
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, height_ratios=[1, 2, 1])
    
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
    
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    
    # Add a 3D visualization to show how the probability surface changes
    ax_3d = fig.add_subplot(gs[0, :], projection='3d')
    
    # Create a grid for 3D plot
    x_3d = np.linspace(-2, 2, 50)
    y_3d = np.linspace(-2, 2, 50)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    pos_3d = np.dstack((X_3d, Y_3d))
    
    # Calculate PDFs for extreme cases
    Sigma_min = np.array([[1.0, 0.0], [0.0, 1.0]])  # No correlation
    Sigma_max = np.array([[1.0, 0.9], [0.9, 1.0]])  # Strong correlation
    Z_min = multivariate_gaussian(pos_3d, mu, Sigma_min)
    Z_max = multivariate_gaussian(pos_3d, mu, Sigma_max)
    
    # Plot 3D surfaces
    ax_3d.plot_surface(X_3d, Y_3d, Z_min, cmap='Blues', alpha=0.5)
    ax_3d.plot_surface(X_3d, Y_3d, Z_max, cmap='Reds', alpha=0.5)
    
    # Add labels
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('Probability Density')
    ax_3d.view_init(25, 45)
    ax_3d.set_title('PDF Surface Comparison: No Correlation (Blue) vs Strong Correlation (Red)')
    
    # Print the explanation that was previously on the 3D plot
    print("\n3D Visualization Explanation:")
    print("Note how correlation stretches the distribution along y=x direction")
    print("and narrows it in the perpendicular direction, while preserving volume.")
    
    for i, (corr, title, ax) in enumerate(zip(correlations, titles, axes)):
        # Create covariance matrix
        Sigma = np.array([
            [1.0, corr],
            [corr, 1.0]
        ])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        
        # Calculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Plot contours
        contour_levels = np.linspace(0.01, 0.2, 5)
        cp = ax.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax.clabel(cp, inline=True, fontsize=10)
        
        # Add a colorful heat map to show the density more clearly
        cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
        
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
            # Print the eigenvalue instead of adding text to the plot
            print(f"{title} - Eigenvalue {j+1}: λ{j+1}={eigenvalues[j]:.2f}")
        
        # Print the correlation formula instead of adding it to the plot
        print(f"Covariance matrix for {title}:")
        print(f"Σ = [[1, {corr}], [{corr}, 1]]")
        print(f"λ₁ = {eigenvalues[0]:.2f} ≈ 1 + ρ")
        print(f"λ₂ = {eigenvalues[1]:.2f} ≈ 1 - ρ")
        print()
        
        # Plot y=x and y=-x guidelines to show alignment of eigenvectors
        x_line = np.array([-3, 3])
        ax.plot(x_line, x_line, 'g--', alpha=0.3, linewidth=1)  # y=x line
        ax.plot(x_line, -x_line, 'g--', alpha=0.3, linewidth=1)  # y=-x line
        
        # Print the explanation of the lines for the strong correlation case
        if i == 3:
            print("For strong correlation (ρ = 0.9):")
            print("- The y = x line shows the direction of maximum variance")
            print("- The y = -x line shows the direction of minimum variance")
            print()
        
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
    
    # Save the concept figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        concept_save_path = os.path.join(images_dir, "ex4_concept_visualization.png")
        concept_fig.savefig(concept_save_path, bbox_inches='tight', dpi=300)
        print(f"\nConcept visualization saved to: {concept_save_path}")
        
        trend_save_path = os.path.join(images_dir, "ex4_eigenvalue_trend.png")
        trend_fig.savefig(trend_save_path, bbox_inches='tight', dpi=300)
        print(f"Eigenvalue trend visualization saved to: {trend_save_path}")
    except Exception as e:
        print(f"\nError saving supplementary figures: {e}")
    
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
        save_path = os.path.join(images_dir, "ex4_covariance_eigenvalue_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    