import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

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

def sketch_contour_problem():
    """Create an interactive visualization for sketching contours of bivariate normal distributions."""
    # Print detailed step-by-step derivation
    print("\n" + "="*80)
    print("Example 9: Sketching Contours of a Bivariate Normal Distribution")
    print("="*80)
    
    print("\nProblem Statement:")
    print("Sketch the contour lines for the probability density function of a bivariate normal distribution")
    print("with mean μ = (0,0) and covariance matrix Σ = [[σ₁², 0], [0, σ₂²]].")
    
    print("\nDetails:")
    print("- The PDF function is defined by its mean vector and covariance matrix.")
    print("- We want to visualize how changing variances affects the shape of contour lines.")
    print("- Contour lines connect points of equal probability density.")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Mathematical Formula Setup")
    print("The bivariate normal probability density function (PDF) is given by:")
    print("f(x,y) = (1/2π√|Σ|) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    print("\nFor our specific case with mean μ = (0,0) and covariance Σ = [[σ₁², 0], [0, σ₂²]]:")
    print("f(x,y) = (1/2π√(σ₁²σ₂²)) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    
    print("\nStep 2: Analyzing the Covariance Matrix")
    print("For our diagonal covariance matrix Σ = [[σ₁², 0], [0, σ₂²]]:")
    print("- This is a diagonal matrix with variances σ₁² and σ₂² along the diagonal")
    print("- Zero covariance means the variables are uncorrelated")
    print("- The determinant |Σ| = σ₁² * σ₂²")
    print("- The inverse Σ⁻¹ = [[1/σ₁², 0], [0, 1/σ₂²]]")
    print("- The eigenvalues are λ₁ = σ₁² and λ₂ = σ₂²")
    print("- The eigenvectors are v₁ = (1,0) and v₂ = (0,1)")
    
    print("\nStep 3: Deriving the Contour Equation")
    print("To find contour lines, we set the PDF equal to a constant c:")
    print("(1/2π√(σ₁²σ₂²)) * exp(-1/2 * (x²/σ₁² + y²/σ₂²)) = c")
    
    print("\nTaking natural logarithm of both sides:")
    print("ln[(1/2π√(σ₁²σ₂²)) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))] = ln(c)")
    print("ln(1/2π√(σ₁²σ₂²)) + ln(exp(-1/2 * (x²/σ₁² + y²/σ₂²))) = ln(c)")
    print("-ln(2π√(σ₁²σ₂²)) - 1/2 * (x²/σ₁² + y²/σ₂²) = ln(c)")
    
    print("\nRearranging to isolate the quadratic terms:")
    print("x²/σ₁² + y²/σ₂² = -2ln(c) - 2ln(2π√(σ₁²σ₂²)) = k")
    print("Where k is a positive constant that depends on the contour value c.")
    
    print("\nStep 4: Recognize the geometric shape")
    print("The equation x²/σ₁² + y²/σ₂² = k describes an ellipse:")
    print("- Centered at the origin (0,0)")
    print("- Semi-axes aligned with the coordinate axes")
    print("- Semi-axis length along x-direction: a = √(k*σ₁²)")
    print("- Semi-axis length along y-direction: b = √(k*σ₂²)")
    
    print("\nSpecial cases:")
    print("- If σ₁² = σ₂² = σ² (equal variances), the equation simplifies to:")
    print("  (x² + y²)/σ² = k, which describes a circle with radius r = √(k*σ²)")
    print("- If σ₁² > σ₂²: The ellipse is stretched along the x-axis")
    print("- If σ₁² < σ₂²: The ellipse is stretched along the y-axis")
    
    print("\nStep 5: Understand the probability content")
    print("For a bivariate normal distribution, the ellipses with constant k represent:")
    print("- k = 1: The 1σ ellipse containing approximately 39% of the probability mass")
    print("- k = 4: The 2σ ellipse containing approximately 86% of the probability mass")
    print("- k = 9: The 3σ ellipse containing approximately 99% of the probability mass")
    
    print("\nStep 6: Sketch the contours")
    print("To sketch the contours, we draw concentric ellipses centered at (0,0):")
    print("- 1σ ellipse: semi-axes a₁ = σ₁ and b₁ = σ₂")
    print("- 2σ ellipse: semi-axes a₂ = 2σ₁ and b₂ = 2σ₂")
    print("- 3σ ellipse: semi-axes a₃ = 3σ₁ and b₃ = 3σ₂")
    
    print("\nNumerical Example:")
    print("For σ₁² = 2.0 and σ₂² = 0.5:")
    print("- 1σ ellipse: semi-axes a₁ = √2 ≈ 1.41 and b₁ = √0.5 ≈ 0.71")
    print("- 2σ ellipse: semi-axes a₂ = 2√2 ≈ 2.83 and b₂ = 2√0.5 ≈ 1.41")
    print("- 3σ ellipse: semi-axes a₃ = 3√2 ≈ 4.24 and b₃ = 3√0.5 ≈ 2.12")
    print("The ellipses are stretched along the x-axis (since σ₁² > σ₂²)")
    
    print("\nConclusion:")
    print("The contour lines for a bivariate normal distribution with diagonal covariance matrix")
    print("form concentric ellipses centered at the mean (0,0). The shape and orientation of")
    print("these ellipses directly reflect the covariance structure of the distribution.")
    
    print(f"\n{'='*80}")
    
    # Setup for the plots
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Initial covariance matrix parameters
    sigma1_init = 1.0
    sigma2_init = 1.0
    
    # Create initial covariance matrix and mean
    mu = np.array([0., 0.])
    Sigma = np.array([[sigma1_init, 0], [0, sigma2_init]])
    
    # Calculate initial PDF
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    # Generate eigenvectors and eigenvalues
    lambda_, v = np.linalg.eig(Sigma)
    lambda_ = np.sqrt(lambda_)
    
    # Create separate figures for each visualization
    
    # 1. Figure for 3D PDF surface
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot 3D surface
    surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    contour_floor = ax_3d.contour(X, Y, Z, zdir='z', offset=0, levels=np.linspace(0.01, 0.15, 5), cmap='viridis', alpha=0.5)
    
    # Add title and labels to 3D plot
    ax_3d.set_title('3D Probability Density Function Surface')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('Probability Density')
    ax_3d.view_init(elev=30, azim=30)  # Set viewing angle
    
    # 2. Figure for contour plot with ellipses
    fig_contour = plt.figure(figsize=(10, 8))
    ax_contour = fig_contour.add_subplot(111)
    
    # Create contour plot
    contour_levels = np.linspace(0.01, 0.15, 5)
    contour = ax_contour.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax_contour.clabel(contour, inline=True, fontsize=10)
    
    # Create ellipses for 1σ, 2σ, and 3σ
    ellipses = []
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax_contour.add_patch(ell)
        ellipses.append(ell)
    
    # Add title and labels
    ax_contour.set_title('Contour Lines for Bivariate Normal Distribution')
    ax_contour.set_xlabel('x')
    ax_contour.set_ylabel('y')
    ax_contour.grid(True)
    ax_contour.set_xlim(-3, 3)
    ax_contour.set_ylim(-3, 3)
    ax_contour.set_aspect('equal')
    
    # 3. Figure for variance comparison
    fig_comparison = plt.figure(figsize=(10, 8))
    ax_comparison = fig_comparison.add_subplot(111)
    
    # Draw coordinate axes
    ax_comparison.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax_comparison.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Draw ellipses for different variance combinations
    ex_sigma1 = [1, 2, 1]
    ex_sigma2 = [1, 1, 2]
    colors = ['blue', 'green', 'purple']
    labels = ['σ₁² = 1, σ₂² = 1', 'σ₁² = 2, σ₂² = 1', 'σ₁² = 1, σ₂² = 2']
    
    for i in range(3):
        ex_lambda = np.sqrt([ex_sigma1[i], ex_sigma2[i]])
        ell = Ellipse(xy=(0, 0),
                     width=ex_lambda[0]*2, height=ex_lambda[1]*2,
                     angle=0,
                     edgecolor=colors[i], facecolor='none', linewidth=2)
        ax_comparison.add_patch(ell)
    
    # Add legend
    ax_comparison.legend([Line2D([0], [0], color=c, lw=2) for c in colors], labels, 
                   loc='upper center')
    
    ax_comparison.set_xlim(-3, 3)
    ax_comparison.set_ylim(-3, 3)
    ax_comparison.set_aspect('equal')
    ax_comparison.set_title('Comparison of Different Variance Combinations')
    ax_comparison.set_xlabel('x')
    ax_comparison.set_ylabel('y')
    
    # Print the key equations and formulas instead of showing them on the plots
    print("\nKey Equations and Formulas:")
    print("1. PDF equation: f(x,y) = (1/2π√(σ₁²σ₂²)) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    print("2. Contour equation: x²/σ₁² + y²/σ₂² = k")
    print("3. Ellipse properties:")
    print("   • x-axis: a = √(k*σ₁²)")
    print("   • y-axis: b = √(k*σ₂²)")
    print("4. Probability content:")
    print("   • k = 1: 39% of probability")
    print("   • k = 4: 86% of probability")
    print("   • k = 9: 99% of probability")
    
    # Create an interactive demonstration of changing variances
    fig_interactive = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Interactive plot area
    ax_interactive = fig_interactive.add_subplot(gs[0])
    
    # Sliders area
    ax_sigma1 = fig_interactive.add_subplot(gs[1])
    ax_sigma2 = ax_sigma1.twinx()  # Create a twin axis to place both sliders
    
    # Initial display
    interactive_contour = ax_interactive.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax_interactive.clabel(interactive_contour, inline=True, fontsize=10)
    
    # Add initial ellipses
    interactive_ellipses = []
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                    width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                    angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                    edgecolor='red', facecolor='none', linestyle='--')
        ax_interactive.add_patch(ell)
        interactive_ellipses.append(ell)
    
    # Setup interactive plot
    ax_interactive.set_title('Interactive Contour Visualization\nAdjust sliders to change variances')
    ax_interactive.set_xlabel('x')
    ax_interactive.set_ylabel('y')
    ax_interactive.grid(True)
    ax_interactive.set_xlim(-3, 3)
    ax_interactive.set_ylim(-3, 3)
    ax_interactive.set_aspect('equal')
    
    # Create sliders
    slider_sigma1 = Slider(ax_sigma1, 'σ₁² (x-variance)', 0.1, 3.0, valinit=sigma1_init, color='lightblue')
    slider_sigma2 = Slider(ax_sigma2, 'σ₂² (y-variance)', 0.1, 3.0, valinit=sigma2_init, color='lightgreen')
    ax_sigma2.spines['right'].set_position(('outward', 40))  # Move the second slider out a bit
    
    # Update function for sliders
    def update(val):
        # Get current slider values
        sigma1 = slider_sigma1.val
        sigma2 = slider_sigma2.val
        
        # Update covariance matrix
        Sigma = np.array([[sigma1, 0], [0, sigma2]])
        
        # Recalculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Clear previous contours
        for c in ax_interactive.collections:
            c.remove()
        
        # Redraw contours
        contour = ax_interactive.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax_interactive.clabel(contour, inline=True, fontsize=10)
        
        # Update ellipses
        lambda_, v = np.linalg.eig(Sigma)
        lambda_ = np.sqrt(lambda_)
        
        # Remove old ellipses
        for ell in interactive_ellipses:
            ell.remove()
        
        # Create new ellipses
        interactive_ellipses.clear()
        for j in range(1, 4):
            ell = Ellipse(xy=(0, 0),
                        width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                        edgecolor='red', facecolor='none', linestyle='--')
            ax_interactive.add_patch(ell)
            interactive_ellipses.append(ell)
        
        # Print current values and equation
        print(f"\nCurrent settings: σ₁² = {sigma1:.2f}, σ₂² = {sigma2:.2f}")
        print(f"Contour equation: x²/{sigma1:.2f} + y²/{sigma2:.2f} = k")
        print(f"1σ ellipse: semi-axes a = {np.sqrt(sigma1):.2f}, b = {np.sqrt(sigma2):.2f}")
        
        # Redraw
        fig_interactive.canvas.draw_idle()
    
    # Connect the sliders to the update function
    slider_sigma1.on_changed(update)
    slider_sigma2.on_changed(update)
    
    plt.tight_layout()
    
    # Return all figures in a dictionary
    return {
        "fig_3d": fig_3d,
        "fig_contour": fig_contour,
        "fig_comparison": fig_comparison,
        "fig_interactive": fig_interactive
    }

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 9: SKETCHING CONTOURS OF A BIVARIATE NORMAL DISTRIBUTION")
    print("*"*80)
    
    # Run sketch contour problem example with detailed step-by-step printing
    print("\nRunning Example 9: Sketching Contours of a Bivariate Normal Distribution")
    figures_contour = sketch_contour_problem()
    
    # Save the sketch contour figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save 3D PDF surface
        save_path = os.path.join(images_dir, "ex9_contour_3d_surface.png")
        figures_contour["fig_3d"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
        
        # Save contour plot
        save_path = os.path.join(images_dir, "ex9_contour_plot.png")
        figures_contour["fig_contour"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save variance comparison
        save_path = os.path.join(images_dir, "ex9_contour_variance_comparison.png")
        figures_contour["fig_comparison"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save interactive plot
        save_path = os.path.join(images_dir, "ex9_contour_interactive.png")
        figures_contour["fig_interactive"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    
    print("\nExample 9 completed successfully!")
        