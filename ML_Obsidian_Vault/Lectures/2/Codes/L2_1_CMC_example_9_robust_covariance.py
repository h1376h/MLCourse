import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from sklearn.covariance import EmpiricalCovariance, MinCovDet
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
    print("Example: Sketching Contours of a Bivariate Normal Distribution")
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

def robust_covariance_comparison():
    """Compare standard covariance estimation with robust methods in the presence of outliers."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Robust vs Standard Covariance Estimation")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How do outliers affect covariance estimation, and how can robust methods help?")
    
    print("\nStep 1: Create a Dataset with Outliers")
    print("We'll generate a clean dataset and then add some outliers to see their effect.")
    
    # Generate clean data
    np.random.seed(42)
    n_samples = 100
    n_outliers = 10
    n_total_samples = n_samples + n_outliers
    
    # Create correlated data
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    mean = np.array([0, 0])
    clean_data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Create outliers
    outliers = np.random.multivariate_normal(mean, cov * 7, n_outliers)
    outliers[:, 0] += 4  # Shift outliers to make them more extreme
    outliers[:, 1] -= 6
    
    # Combine data
    X = np.vstack([clean_data, outliers])
    
    print("\nStep 2: Compute Covariance Estimates")
    print("We'll compare three methods:")
    print("- Standard empirical covariance (sensitive to outliers)")
    print("- Minimum Covariance Determinant (MCD) method (robust to outliers)")
    print("- True covariance (calculated using only clean data, as a reference)")
    
    # Fit standard empirical covariance
    emp_cov = EmpiricalCovariance().fit(X)
    
    # Fit robust covariance (Minimum Covariance Determinant)
    robust_cov = MinCovDet(random_state=42).fit(X)
    
    # Compute true covariance (from clean data only)
    true_cov = EmpiricalCovariance().fit(clean_data)
    
    print("\nEmpirical Covariance Matrix:")
    print(emp_cov.covariance_)
    print("\nRobust Covariance Matrix (MCD):")
    print(robust_cov.covariance_)
    print("\nTrue Covariance Matrix (clean data only):")
    print(true_cov.covariance_)
    
    # Function to plot covariance ellipse (used across multiple figures)
    def plot_covariance_ellipse(covariance, mean, ax, color, alpha=0.3, label=None):
        """Plot an ellipse representing the covariance matrix."""
        v, w = np.linalg.eigh(covariance)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        
        # For 95% confidence interval
        scaling_factor = 2.447  # For 95% CI: sqrt(chi2.ppf(0.95, 2))
        
        ell = Ellipse(xy=mean,
                     width=scaling_factor*2*np.sqrt(v[0]), 
                     height=scaling_factor*2*np.sqrt(v[1]),
                     angle=180 + angle,
                     facecolor=color,
                     alpha=alpha,
                     edgecolor=color,
                     linewidth=2)
        ax.add_patch(ell)
        
        # Add label point for the ellipse
        if label:
            ax.plot([mean[0]], [mean[1]], 'o', color=color, markersize=8, label=label)
    
    print("\nStep 3: Visualize the Data and Covariance Ellipses")
    print("We'll create several visualizations to understand the impact of outliers:")
    
    # 1. Main data visualization with covariance ellipses
    fig_data = plt.figure(figsize=(10, 8))
    ax_data = fig_data.add_subplot(111)
    
    # Plot data points
    ax_data.scatter(clean_data[:, 0], clean_data[:, 1], color='blue', alpha=0.5, label='Clean data')
    ax_data.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='x', s=100, label='Outliers')
    
    # Plot the covariance ellipses
    plot_covariance_ellipse(emp_cov.covariance_, emp_cov.location_, ax_data, 'blue', 
                          label='Standard Covariance (95% CI)')
    plot_covariance_ellipse(robust_cov.covariance_, robust_cov.location_, ax_data, 'green', 
                          label='Robust Covariance (95% CI)')
    plot_covariance_ellipse(true_cov.covariance_, true_cov.location_, ax_data, 'purple', 
                          label='True Covariance (clean data only)')
    
    # Finalize the data plot
    ax_data.set_xlim(-4, 8)
    ax_data.set_ylim(-8, 4)
    ax_data.set_aspect('equal')
    ax_data.grid(True)
    ax_data.set_xlabel('Feature 1')
    ax_data.set_ylabel('Feature 2')
    ax_data.set_title('Comparison of Covariance Estimation Methods')
    ax_data.legend(loc='upper right')
    
    # 2. Before-after comparison visualization
    fig_comparison = plt.figure(figsize=(10, 8))
    ax_comparison = fig_comparison.add_subplot(111)
    
    # Create a "before" plot (only clean data)
    ax_comparison.scatter(clean_data[:, 0], clean_data[:, 1], color='blue', alpha=0.3, label='Clean data')
    plot_covariance_ellipse(true_cov.covariance_, true_cov.location_, ax_comparison, 'purple', 
                           alpha=0.2, label='True Covariance')
    
    # Create an "after" plot with outliers and distorted covariance
    ax_comparison.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='x', s=80, label='Outliers')
    plot_covariance_ellipse(emp_cov.covariance_, emp_cov.location_, ax_comparison, 'blue', 
                           alpha=0.2, label='Distorted Covariance')
    
    # Finalize the comparison plot
    ax_comparison.set_xlim(-4, 8)
    ax_comparison.set_ylim(-8, 4)
    ax_comparison.set_aspect('equal')
    ax_comparison.grid(True)
    ax_comparison.set_xlabel('Feature 1')
    ax_comparison.set_ylabel('Feature 2')
    ax_comparison.set_title('Effect of Outliers on Covariance Estimation')
    ax_comparison.legend(loc='upper right')
    
    # Print explanation of what's happening in the comparison
    print("\nComparison Visualization Explanation:")
    print("- The clean data forms a clustered pattern with moderate correlation")
    print("- Outliers are positioned away from the main data cluster")
    print("- With outliers present, standard covariance shows:")
    print("  * Shifted center (mean) toward the outliers")
    print("  * Inflated size (increased variance)")
    print("  * Distorted orientation (altered correlation structure)")
    print("- The robust covariance remains close to the true covariance structure")
    
    # 3. Create 3D subplot for standard covariance PDF
    fig_3d_std = plt.figure(figsize=(10, 8))
    ax3d_std = fig_3d_std.add_subplot(111, projection='3d')
    
    # Create a grid of points for the 3D visualization
    x = np.linspace(-4, 8, 50)
    y = np.linspace(-8, 4, 50)
    X_grid, Y_grid = np.meshgrid(x, y)
    pos = np.dstack((X_grid, Y_grid))
    
    # Calculate PDFs using the different covariance estimates
    Z_std = multivariate_gaussian(pos, emp_cov.location_, emp_cov.covariance_)
    Z_robust = multivariate_gaussian(pos, robust_cov.location_, robust_cov.covariance_)
    Z_true = multivariate_gaussian(pos, true_cov.location_, true_cov.covariance_)
    
    # Plot the standard covariance PDF surface
    surf_std = ax3d_std.plot_surface(X_grid, Y_grid, Z_std, cmap=cm.Blues, 
                                   linewidth=0, antialiased=True, alpha=0.7)
    
    # Add contours at the bottom
    offset = Z_std.min()
    ax3d_std.contour(X_grid, Y_grid, Z_std, zdir='z', offset=offset, cmap=cm.Blues, levels=5)
    
    # Plot data points
    ax3d_std.scatter(clean_data[:, 0], clean_data[:, 1], offset, color='blue', alpha=0.5, s=10)
    ax3d_std.scatter(outliers[:, 0], outliers[:, 1], offset, color='red', marker='x', s=50)
    
    # Finalize the standard covariance 3D plot
    ax3d_std.set_title('Standard Covariance PDF')
    ax3d_std.set_xlabel('Feature 1')
    ax3d_std.set_ylabel('Feature 2')
    ax3d_std.set_zlabel('Probability Density')
    ax3d_std.view_init(30, 45)
    
    # 4. Create 3D subplot for robust covariance PDF
    fig_3d_robust = plt.figure(figsize=(10, 8))
    ax3d_robust = fig_3d_robust.add_subplot(111, projection='3d')
    
    # Plot the robust covariance PDF surface
    surf_robust = ax3d_robust.plot_surface(X_grid, Y_grid, Z_robust, cmap=cm.Greens, 
                                         linewidth=0, antialiased=True, alpha=0.7)
    
    # Add contours at the bottom
    offset = Z_robust.min()
    ax3d_robust.contour(X_grid, Y_grid, Z_robust, zdir='z', offset=offset, cmap=cm.Greens, levels=5)
    
    # Plot data points
    ax3d_robust.scatter(clean_data[:, 0], clean_data[:, 1], offset, color='blue', alpha=0.5, s=10)
    ax3d_robust.scatter(outliers[:, 0], outliers[:, 1], offset, color='red', marker='x', s=50)
    
    # Finalize the robust covariance 3D plot
    ax3d_robust.set_title('Robust Covariance PDF')
    ax3d_robust.set_xlabel('Feature 1')
    ax3d_robust.set_ylabel('Feature 2')
    ax3d_robust.set_zlabel('Probability Density')
    ax3d_robust.view_init(30, 45)
    
    # 5. Create 3D subplot showing true distribution
    fig_3d_true = plt.figure(figsize=(10, 8))
    ax3d_true = fig_3d_true.add_subplot(111, projection='3d')
    
    # Plot the true PDF surface
    surf_true = ax3d_true.plot_surface(X_grid, Y_grid, Z_true, cmap=cm.Purples, 
                                     linewidth=0, antialiased=True, alpha=0.7)
    
    # Add contours at the bottom
    offset = Z_true.min()
    ax3d_true.contour(X_grid, Y_grid, Z_true, zdir='z', offset=offset, cmap=cm.Purples, levels=5)
    
    # Plot only clean data points
    ax3d_true.scatter(clean_data[:, 0], clean_data[:, 1], offset, color='blue', alpha=0.5, s=10)
    
    # Finalize the true distribution 3D plot
    ax3d_true.set_title('True Distribution PDF (Clean Data Only)')
    ax3d_true.set_xlabel('Feature 1')
    ax3d_true.set_ylabel('Feature 2')
    ax3d_true.set_zlabel('Probability Density')
    ax3d_true.view_init(30, 45)
    
    # 6. Create a methods comparison visualization
    fig_methods = plt.figure(figsize=(10, 8))
    ax_methods = fig_methods.add_subplot(111)
    ax_methods.axis('off')
    
    # Print comparison information instead of displaying it on the plot
    print("\nComparison of Standard vs Robust Methods:")
    print("Standard Method:")
    print("- Principle: Use all data points equally")
    print("- Estimator: Sample Covariance Matrix")
    print("- Complexity: O(n)")
    print("- Breakdown point: 0% (a single extreme outlier can arbitrarily distort the estimate)")
    print("- Best for: Clean data with no outliers")
    
    print("\nRobust Method (MCD):")
    print("- Principle: Identify and downweight outliers")
    print("- Estimator: Minimum Covariance Determinant")
    print("- Complexity: O(n²)")
    print("- Breakdown point: Up to 50% (can handle up to half of the data being outliers)")
    print("- Best for: Data with potential outliers")
    
    print("\nStep 4: 3D Visualization Explanation")
    print("The 3D visualizations show how the probability density functions differ:")
    print("- Standard method's PDF is distorted by outliers:")
    print("  * Flatter, more spread out")
    print("  * Shifted center")
    print("  * Distorted orientation")
    print("- Robust method's PDF is closer to the true distribution:")
    print("  * Maintains appropriate peak height")
    print("  * Centered correctly")
    print("  * Proper orientation")
    print("- True distribution (without outliers) shows how the data actually distributed")
    
    print("\nStep 5: Key Observations and Implications")
    print("Key observations from our visualizations:")
    print("1. The standard covariance is heavily influenced by outliers, resulting in:")
    print("   - An inflated ellipse size")
    print("   - Distorted orientation")
    print("   - Shifted center location")
    print("   - A flatter, more spread-out probability density function")
    
    print("\n2. The robust covariance (MCD) is resistant to outliers, providing:")
    print("   - A more accurate representation of the main data structure")
    print("   - Better preservation of the true covariance shape")
    print("   - Higher reliability for downstream tasks")
    print("   - A probability density function that more closely matches the true distribution")
    
    print("\n3. Implications for machine learning applications:")
    print("   - When outliers are present, robust methods can prevent model distortion")
    print("   - Principal Component Analysis (PCA) benefits from robust covariance")
    print("   - Classification algorithms like Mahalanobis distance-based methods work better")
    print("   - The 3D visualization shows how outliers drastically change the probability model")
    print("   - Anomaly detection is more accurate with robust covariance estimation")
    
    plt.tight_layout()
    
    # Return all figures in a dictionary
    return {
        "fig_data": fig_data,
        "fig_comparison": fig_comparison,
        "fig_3d_std": fig_3d_std,
        "fig_3d_robust": fig_3d_robust,
        "fig_3d_true": fig_3d_true,
        "fig_methods": fig_methods
    }

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 9: ROBUST COVARIANCE ESTIMATION")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    figures_robust = robust_covariance_comparison()
    
    # Save the figures if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    # Save each figure separately
    try:
        # Save data visualization with covariance ellipses
        save_path = os.path.join(images_dir, "ex9_robust_covariance_data.png")
        figures_robust["fig_data"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
        
        # Save comparison visualization
        save_path = os.path.join(images_dir, "ex9_robust_covariance_comparison.png")
        figures_robust["fig_comparison"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D standard covariance PDF
        save_path = os.path.join(images_dir, "ex9_robust_covariance_3d_standard.png")
        figures_robust["fig_3d_std"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D robust covariance PDF
        save_path = os.path.join(images_dir, "ex9_robust_covariance_3d_robust.png")
        figures_robust["fig_3d_robust"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D true covariance PDF
        save_path = os.path.join(images_dir, "ex9_robust_covariance_3d_true.png")
        figures_robust["fig_3d_true"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    
    print("\n\n" + "*"*80)
    print("RUNNING SKETCH CONTOUR PROBLEM WITH DETAILED STEP-BY-STEP SOLUTION")
    print("*"*80)
    
    # Run sketch contour problem example with detailed step-by-step printing
    print("\nRunning Example: Sketch Contour Problem")
    figures_contour = sketch_contour_problem()
        
    # Save the sketch contour figures
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
    
    print("\nExample completed successfully!")
        