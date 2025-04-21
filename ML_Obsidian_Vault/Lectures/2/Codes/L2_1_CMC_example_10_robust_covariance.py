import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

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

def robust_covariance_comparison():
    """Compare standard covariance estimation with robust methods in the presence of outliers."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example 10: Robust vs Standard Covariance Estimation")
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
    print("EXAMPLE 10: ROBUST COVARIANCE ESTIMATION")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    figures_robust = robust_covariance_comparison()
    
    # Save the figures if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Robust_Covariance")
    ensure_directory_exists(images_dir)
    
    try:
        # Save data visualization with covariance ellipses
        save_path = os.path.join(images_dir, "ex10_robust_covariance_data.png")
        figures_robust["fig_data"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
        
        # Save comparison visualization
        save_path = os.path.join(images_dir, "ex10_robust_covariance_comparison.png")
        figures_robust["fig_comparison"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D standard covariance PDF
        save_path = os.path.join(images_dir, "ex10_robust_covariance_3d_standard.png")
        figures_robust["fig_3d_std"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D robust covariance PDF
        save_path = os.path.join(images_dir, "ex10_robust_covariance_3d_robust.png")
        figures_robust["fig_3d_robust"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
        # Save 3D true covariance PDF
        save_path = os.path.join(images_dir, "ex10_robust_covariance_3d_true.png")
        figures_robust["fig_3d_true"].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    
    print("\nExample 10 completed successfully!")