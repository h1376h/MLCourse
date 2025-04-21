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
    # Create figure with a grid layout
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 2, 1], height_ratios=[3, 1])
    
    # Main plot area for contours
    ax_contour = fig.add_subplot(gs[0, 0])
    # Visual explanation area
    ax_vis = fig.add_subplot(gs[0, 1])
    # Sliders area
    ax_sigma1 = fig.add_subplot(gs[1, 0])
    ax_sigma2 = fig.add_subplot(gs[1, 1])
    
    # Setup the initial plot data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Initial covariance matrix parameters
    sigma1_init = 1.0
    sigma2_init = 1.0
    
    # Print step-by-step derivation
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
    
    # Create initial covariance matrix and mean
    mu = np.array([0., 0.])
    Sigma = np.array([[sigma1_init, 0], [0, sigma2_init]])
    
    # Calculate initial PDF
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    # Create contour plot
    contour_levels = np.linspace(0.01, 0.15, 5)
    contour = ax_contour.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax_contour.clabel(contour, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma)
    lambda_ = np.sqrt(lambda_)
    
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
    ax_contour.set_title('Contour Lines for Bivariate Normal Distribution\nwith Diagonal Covariance Matrix')
    ax_contour.set_xlabel('x')
    ax_contour.set_ylabel('y')
    ax_contour.grid(True)
    ax_contour.set_xlim(-3, 3)
    ax_contour.set_ylim(-3, 3)
    ax_contour.set_aspect('equal')
    
    # Create simplified visual explanation in the right panel
    ax_vis.axis('off')
    
    # Draw coordinate axes
    ax_vis.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax_vis.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Create visual explanation with simple diagram
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
        ax_vis.add_patch(ell)
    
    # Add legend
    ax_vis.legend([Line2D([0], [0], color=c, lw=2) for c in colors], labels, 
                  loc='upper center', bbox_to_anchor=(0.5, 1.1))
    
    ax_vis.set_xlim(-3, 3)
    ax_vis.set_ylim(-3, 3)
    ax_vis.set_aspect('equal')
    ax_vis.text(0, -2.5, "Visual comparison of different\nvariance combinations", 
                ha='center', fontsize=10)
    
    # Create sliders
    slider_sigma1 = Slider(ax_sigma1, 'σ₁² (x-variance)', 0.1, 3.0, valinit=sigma1_init)
    slider_sigma2 = Slider(ax_sigma2, 'σ₂² (y-variance)', 0.1, 3.0, valinit=sigma2_init)
    
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
        for c in ax_contour.collections:
            c.remove()
        
        # Redraw contours
        contour = ax_contour.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax_contour.clabel(contour, inline=True, fontsize=10)
        
        # Update ellipses
        lambda_, v = np.linalg.eig(Sigma)
        lambda_ = np.sqrt(lambda_)
        
        # Remove old ellipses
        for ell in ellipses:
            ell.remove()
        
        # Create new ellipses
        ellipses.clear()
        for j in range(1, 4):
            ell = Ellipse(xy=(0, 0),
                         width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                         angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            ax_contour.add_patch(ell)
            ellipses.append(ell)
        
        # Print current values and equation
        print(f"\nCurrent settings: σ₁² = {sigma1:.2f}, σ₂² = {sigma2:.2f}")
        print(f"Contour equation: x²/{sigma1:.2f} + y²/{sigma2:.2f} = k")
        print(f"1σ ellipse: semi-axes a = {np.sqrt(sigma1):.2f}, b = {np.sqrt(sigma2):.2f}")
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Connect the sliders to the update function
    slider_sigma1.on_changed(update)
    slider_sigma2.on_changed(update)
    
    plt.tight_layout()
    return fig

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
    print("We'll compare two methods:")
    print("- Standard empirical covariance (sensitive to outliers)")
    print("- Minimum Covariance Determinant (MCD) method (robust to outliers)")
    
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
    
    print("\nStep 3: Visualize the Data and Covariance Ellipses")
    
    # Create a multi-panel figure: 2D view, 3D standard view, and 3D robust view
    fig = plt.figure(figsize=(20, 10))
    
    # 2D plot (first panel)
    ax = fig.add_subplot(131)
    
    # Plot data points
    ax.scatter(clean_data[:, 0], clean_data[:, 1], color='blue', alpha=0.5, label='Clean data')
    ax.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='x', s=100, label='Outliers')
    
    # Function to plot covariance ellipse
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
    
    # Plot the covariance ellipses
    plot_covariance_ellipse(emp_cov.covariance_, emp_cov.location_, ax, 'blue', 
                          label='Standard Covariance (95% CI)')
    plot_covariance_ellipse(robust_cov.covariance_, robust_cov.location_, ax, 'green', 
                          label='Robust Covariance (95% CI)')
    plot_covariance_ellipse(true_cov.covariance_, true_cov.location_, ax, 'purple', 
                          label='True Covariance (clean data only)')
    
    # Finalize the 2D plot
    ax.set_xlim(-4, 8)
    ax.set_ylim(-8, 4)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Comparison of Covariance Estimation Methods')
    ax.legend(loc='upper right')
    
    print("\nStep 4: 3D Visualization of Probability Density Functions")
    print("Now we'll visualize how the probability density functions differ:")
    print("- Standard method's PDF will be distorted by outliers")
    print("- Robust method's PDF will be closer to the true distribution")
    print("- This helps us understand why robust methods work better for classification and anomaly detection")
    
    # Create a grid of points for the 3D visualization
    x = np.linspace(-4, 8, 50)
    y = np.linspace(-8, 4, 50)
    X_grid, Y_grid = np.meshgrid(x, y)
    pos = np.dstack((X_grid, Y_grid))
    
    # Calculate PDFs using the different covariance estimates
    Z_std = multivariate_gaussian(pos, emp_cov.location_, emp_cov.covariance_)
    Z_robust = multivariate_gaussian(pos, robust_cov.location_, robust_cov.covariance_)
    Z_true = multivariate_gaussian(pos, true_cov.location_, true_cov.covariance_)
    
    # Create 3D subplot for standard covariance PDF
    ax3d_std = fig.add_subplot(132, projection='3d')
    
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
    ax3d_std.set_title('Standard Covariance PDF\n(Distorted by Outliers)')
    ax3d_std.set_xlabel('Feature 1')
    ax3d_std.set_ylabel('Feature 2')
    ax3d_std.set_zlabel('Probability Density')
    ax3d_std.view_init(30, 45)
    
    # Create 3D subplot for robust covariance PDF
    ax3d_robust = fig.add_subplot(133, projection='3d')
    
    # Plot the robust covariance PDF surface
    surf_robust = ax3d_robust.plot_surface(X_grid, Y_grid, Z_robust, cmap=cm.Greens, 
                                         linewidth=0, antialiased=True, alpha=0.7)
    
    # Add contours at the bottom
    offset = Z_robust.min()
    ax3d_robust.contour(X_grid, Y_grid, Z_robust, zdir='z', offset=offset, cmap=cm.Greens, levels=5)
    # Also add true distribution contours for comparison
    ax3d_robust.contour(X_grid, Y_grid, Z_true, zdir='z', offset=offset, cmap=cm.Purples, 
                       levels=5, linestyles='dashed')
    
    # Plot data points
    ax3d_robust.scatter(clean_data[:, 0], clean_data[:, 1], offset, color='blue', alpha=0.5, s=10)
    ax3d_robust.scatter(outliers[:, 0], outliers[:, 1], offset, color='red', marker='x', s=50)
    
    # Finalize the robust covariance 3D plot
    ax3d_robust.set_title('Robust Covariance PDF\n(Resistant to Outliers)')
    ax3d_robust.set_xlabel('Feature 1')
    ax3d_robust.set_ylabel('Feature 2')
    ax3d_robust.set_zlabel('Probability Density')
    ax3d_robust.view_init(30, 45)
    
    print("\nStep 5: Analyze the Key Observations")
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
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 9: ROBUST COVARIANCE ESTIMATION")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig_robust = robust_covariance_comparison()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "ex9_robust_covariance_comparison.png")
        fig_robust.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    print("\n\n" + "*"*80)
    print("RUNNING SKETCH CONTOUR PROBLEM WITH DETAILED STEP-BY-STEP SOLUTION")
    print("*"*80)
    
    # Run sketch contour problem example with detailed step-by-step printing
    print("\nRunning Example: Sketch Contour Problem")
    fig_contour = sketch_contour_problem()
        
    # Save the sketch contour figure
    try:
        save_path = os.path.join(images_dir, "ex9_sketch_contour_problem.png")
        fig_contour.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nGenerated and saved sketch_contour_problem.png")
        print(f"Saved to: {save_path}")
    except Exception as e:
        print(f"\nError generating sketch_contour_problem.png: {e}")
    
    print("\nExample completed successfully!")
        