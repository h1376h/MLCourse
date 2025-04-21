import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from sklearn.covariance import EmpiricalCovariance, MinCovDet
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
    
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

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
    fig = robust_covariance_comparison()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "robust_covariance_comparison.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
        