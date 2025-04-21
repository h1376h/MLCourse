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

def detailed_calculation_example(point, mean, cov_matrix, cov_inv):
    """Show the detailed calculation for one point"""
    diff = point - mean
    
    print(f"\nDetailed calculation for point {point}:")
    print(f"1. Compute the difference vector: x - μ = {point} - {mean} = {diff}")
    
    print(f"2. Matrix multiplication with inverse covariance:")
    print(f"   (x - μ)ᵀ Σ⁻¹ = [{diff[0]:.2f}, {diff[1]:.2f}] × [[{cov_inv[0,0]:.4f}, {cov_inv[0,1]:.4f}],")
    print(f"                                      [{cov_inv[1,0]:.4f}, {cov_inv[1,1]:.4f}]]")
    
    temp_result = diff.dot(cov_inv)
    print(f"   = [{temp_result[0]:.4f}, {temp_result[1]:.4f}]")
    
    print(f"3. Complete the quadratic form:")
    print(f"   (x - μ)ᵀ Σ⁻¹ (x - μ) = [{temp_result[0]:.4f}, {temp_result[1]:.4f}] × [{diff[0]:.2f}, {diff[1]:.2f}]ᵀ")
    
    squared_dist = temp_result.dot(diff)
    print(f"   = {temp_result[0]:.4f} × {diff[0]:.2f} + {temp_result[1]:.4f} × {diff[1]:.2f}")
    print(f"   = {temp_result[0] * diff[0]:.4f} + {temp_result[1] * diff[1]:.4f}")
    print(f"   = {squared_dist:.4f}")
    
    mahalanobis_dist = np.sqrt(squared_dist)
    print(f"4. Take the square root: √{squared_dist:.4f} = {mahalanobis_dist:.4f}")
    
    # Calculate the Euclidean distance for comparison
    euclidean_dist = np.sqrt(diff.dot(diff))
    print(f"\nFor comparison, Euclidean distance = √({diff[0]:.2f}² + {diff[1]:.2f}²) = √{diff[0]**2 + diff[1]**2:.4f} = {euclidean_dist:.4f}")
    
    return mahalanobis_dist, euclidean_dist

def create_3d_pdf_visualization(mean, cov_matrix):
    """Create a 3D visualization of the probability density function"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid to plot the pdf
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate the multivariate normal pdf at each point
    pos = np.dstack((X, Y))
    rv = np.zeros_like(X)
    
    # Calculate determinant of covariance matrix
    det = np.linalg.det(cov_matrix)
    
    # Calculate inverse of covariance matrix
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Calculate the pdf for each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = pos[i, j, :] - mean
            rv[i, j] = (1.0 / (2 * np.pi * np.sqrt(det))) * np.exp(-0.5 * x.dot(inv_cov).dot(x))
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, rv, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    
    # Plot contour lines on the bottom
    ax.contour(X, Y, rv, zdir='z', offset=0, cmap=cm.viridis, levels=10)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Visualization of the Probability Density Function')
    
    # Add text about Mahalanobis distance
    ax.text2D(0.05, 0.95, "Mahalanobis distance contours\nform elliptical slices of the PDF", 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_comparison_visualization(mean, cov_matrix, data):
    """Create a visualization comparing different covariance structures"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original data with correlation
    axs[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Draw mahalanobis distance ellipse
    for m_dist in [1, 2]:
        ell = Ellipse(xy=(0, 0),
                     width=2*m_dist*np.sqrt(eigenvalues[0]), 
                     height=2*m_dist*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='purple', facecolor='none', linestyle='-', alpha=0.7)
        axs[0].add_patch(ell)
    
    # Draw euclidean distance circles
    for e_dist in [1, 2]:
        circle = plt.Circle((0, 0), e_dist, fill=False, edgecolor='green', linestyle='--')
        axs[0].add_patch(circle)
    
    axs[0].set_title('Original Data\nPositive Correlation')
    axs[0].grid(True)
    axs[0].set_xlim(-4, 4)
    axs[0].set_ylim(-4, 4)
    axs[0].set_aspect('equal')
    
    # Case 2: Negative correlation
    np.random.seed(42)
    cov_neg = np.array([[2.0, -1.5], [-1.5, 2.0]])  # Negative correlation
    data_neg = np.random.multivariate_normal(mean, cov_neg, 300)
    
    axs[1].scatter(data_neg[:, 0], data_neg[:, 1], alpha=0.5, s=10)
    
    # Get eigenvalues and eigenvectors
    eigenvalues_neg, eigenvectors_neg = np.linalg.eig(cov_neg)
    
    # Draw mahalanobis distance ellipses
    for m_dist in [1, 2]:
        ell = Ellipse(xy=(0, 0),
                     width=2*m_dist*np.sqrt(eigenvalues_neg[0]), 
                     height=2*m_dist*np.sqrt(eigenvalues_neg[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors_neg[1, 0], eigenvectors_neg[0, 0])),
                     edgecolor='purple', facecolor='none', linestyle='-', alpha=0.7)
        axs[1].add_patch(ell)
    
    # Draw euclidean distance circles
    for e_dist in [1, 2]:
        circle = plt.Circle((0, 0), e_dist, fill=False, edgecolor='green', linestyle='--')
        axs[1].add_patch(circle)
    
    axs[1].set_title('Negative Correlation')
    axs[1].grid(True)
    axs[1].set_xlim(-4, 4)
    axs[1].set_ylim(-4, 4)
    axs[1].set_aspect('equal')
    
    # Case 3: Uncorrelated but different variances
    np.random.seed(42)
    cov_uncorr = np.array([[3.0, 0], [0, 1.0]])  # Uncorrelated, different variances
    data_uncorr = np.random.multivariate_normal(mean, cov_uncorr, 300)
    
    axs[2].scatter(data_uncorr[:, 0], data_uncorr[:, 1], alpha=0.5, s=10)
    
    # Get eigenvalues and eigenvectors
    eigenvalues_uncorr, eigenvectors_uncorr = np.linalg.eig(cov_uncorr)
    
    # Draw mahalanobis distance ellipses
    for m_dist in [1, 2]:
        ell = Ellipse(xy=(0, 0),
                     width=2*m_dist*np.sqrt(eigenvalues_uncorr[0]), 
                     height=2*m_dist*np.sqrt(eigenvalues_uncorr[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors_uncorr[1, 0], eigenvectors_uncorr[0, 0])),
                     edgecolor='purple', facecolor='none', linestyle='-', alpha=0.7)
        axs[2].add_patch(ell)
    
    # Draw euclidean distance circles
    for e_dist in [1, 2]:
        circle = plt.Circle((0, 0), e_dist, fill=False, edgecolor='green', linestyle='--')
        axs[2].add_patch(circle)
    
    axs[2].set_title('Uncorrelated Data\nDifferent Variances')
    axs[2].grid(True)
    axs[2].set_xlim(-4, 4)
    axs[2].set_ylim(-4, 4)
    axs[2].set_aspect('equal')
    
    # Add a common legend
    fig.legend(['Data Points', 'Mahalanobis Dist', 'Euclidean Dist'], 
               loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def create_whitening_visualization(data, cov_matrix):
    """Create a visualization showing data whitening"""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot original data
    axs[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    axs[0].set_title('Original Correlated Data')
    axs[0].grid(True)
    axs[0].set_xlim(-4, 4)
    axs[0].set_ylim(-4, 4)
    axs[0].set_aspect('equal')
    
    # Get eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Create whitening matrix
    whitening_matrix = eigenvectors.dot(np.diag(1.0/np.sqrt(eigenvalues))).dot(eigenvectors.T)
    
    # Whiten the data
    whitened_data = data.dot(whitening_matrix)
    
    # Plot whitened data
    axs[1].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.5, s=10)
    axs[1].set_title('Whitened Data (Unit Variance in All Directions)')
    axs[1].grid(True)
    axs[1].set_xlim(-4, 4)
    axs[1].set_ylim(-4, 4)
    axs[1].set_aspect('equal')
    
    # Draw unit circle representing equal Mahalanobis distances in whitened space
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='red', linestyle='-')
    axs[1].add_patch(circle)
    circle = plt.Circle((0, 0), 2, fill=False, edgecolor='red', linestyle='-')
    axs[1].add_patch(circle)
    
    # Add text explanation
    axs[0].text(0.05, 0.95, "Correlated Data:\nMahalanobis = Ellipses\nEuclidean = Circles", 
               transform=axs[0].transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    axs[1].text(0.05, 0.95, "Whitened Data:\nMahalanobis = Euclidean = Circles\nUnit variance in all directions", 
               transform=axs[1].transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig, whitening_matrix

def simple_mahalanobis_distance():
    """Visualize Mahalanobis distance vs Euclidean distance for correlated data."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Mahalanobis Distance vs Euclidean Distance")
    print("="*80)
    
    print("\nProblem Statement:")
    print("Why is Euclidean distance inadequate for correlated data, and how does Mahalanobis distance address this limitation?")
    
    print("\nStep 1: Understanding Distance Metrics")
    print("Distance metrics provide a way to measure how 'far' points are from each other or from a reference point.")
    print("For multivariate data with correlation structure, standard Euclidean distance can be misleading.")
    print("\nEuclidean distance:")
    print("- Treats all dimensions equally and independently")
    print("- Represented by circles of equal distance from the mean")
    print("- Formula: d_E(x) = √[(x-μ)^T(x-μ)]")
    print("\nMahalanobis distance:")
    print("- Accounts for the covariance structure of the data")
    print("- Represented by ellipses aligned with the data's natural distribution")
    print("- Points at the same Mahalanobis distance have equal probability density under a multivariate normal model")
    print("- Formula: d_M(x) = √[(x-μ)^T Σ^(-1) (x-μ)]")
    print("  where μ is the mean and Σ is the covariance matrix")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    print("\nStep 2: Creating Correlated Data")
    # Create correlated data
    np.random.seed(42)
    cov_matrix = np.array([[2.0, 1.5], [1.5, 2.0]])  # Positive correlation
    mean = np.array([0, 0])
    
    # Calculate the correlation coefficient
    corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    print(f"Using a covariance matrix with correlation coefficient: {corr:.2f}")
    print(f"Covariance matrix:")
    print(f"[[{cov_matrix[0,0]:.1f}, {cov_matrix[0,1]:.1f}],")
    print(f" [{cov_matrix[1,0]:.1f}, {cov_matrix[1,1]:.1f}]]")
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(mean, cov_matrix, 300)
    
    print("\nStep 3: Calculate the Inverse Covariance Matrix")
    # Calculate the inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)
    print(f"Inverse covariance matrix (precision matrix):")
    print(f"[[{cov_inv[0,0]:.4f}, {cov_inv[0,1]:.4f}],")
    print(f" [{cov_inv[1,0]:.4f}, {cov_inv[1,1]:.4f}]]")
    
    print("\nStep 4: Selecting Test Points for Distance Comparison")
    # Test points for distance calculation
    test_points = np.array([
        [2, 0],    # Point along x-axis
        [0, 2],    # Point along y-axis
        [2, 2],    # Point in first quadrant
        [-1.5, 1.5]  # Point in second quadrant
    ])
    
    print("Selected test points:")
    for i, point in enumerate(test_points):
        print(f"P{i+1}: ({point[0]}, {point[1]})")
    
    print("\nStep 5: Computing Mahalanobis Distances")
    # Compute Mahalanobis distances
    mahalanobis_distances = []
    euclidean_distances = []
    
    # Show detailed calculation for the first test point
    m_dist, e_dist = detailed_calculation_example(test_points[0], mean, cov_matrix, cov_inv)
    mahalanobis_distances.append(m_dist)
    euclidean_distances.append(e_dist)
    
    # Calculate distances for the rest of test points
    for point in test_points[1:]:
        diff = point - mean
        mahalanobis_distance = np.sqrt(diff.dot(cov_inv).dot(diff))
        euclidean_distance = np.sqrt(diff.dot(diff))
        mahalanobis_distances.append(mahalanobis_distance)
        euclidean_distances.append(euclidean_distance)
    
    print("\nDistances for each test point:")
    print(f"{'Point':<10} {'Euclidean':<15} {'Mahalanobis':<15}")
    print(f"{'-'*40}")
    for i, (point, e_dist, m_dist) in enumerate(zip(test_points, euclidean_distances, mahalanobis_distances)):
        print(f"P{i+1}:{'':<5} {e_dist:.2f}{'':<10} {m_dist:.2f}")
    
    print("\nStep 6: Visualizing the Distances")
    # Plot the data points
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, label='Data Points')
    
    # Plot test points
    ax.scatter(test_points[:, 0], test_points[:, 1], color='red', s=100, marker='*', label='Test Points')
    
    # Get eigenvalues and eigenvectors for contour ellipses
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Draw multiple contour ellipses representing equal Mahalanobis distances
    for m_dist in [1, 2, 3]:
        ell = Ellipse(xy=(0, 0),
                     width=2*m_dist*np.sqrt(eigenvalues[0]), 
                     height=2*m_dist*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='purple', facecolor='none', linestyle='-', alpha=0.7)
        ax.add_patch(ell)
        ax.text(0, m_dist*np.sqrt(eigenvalues[1]), f'M-dist = {m_dist}', 
               color='purple', ha='center', va='bottom')
    
    # Add text for the test points
    for i, (point, dist) in enumerate(zip(test_points, mahalanobis_distances)):
        ax.text(point[0], point[1] + 0.3, f'P{i+1}: M-dist = {dist:.2f}', ha='center')
    
    # Draw Euclidean distance circles for comparison
    for e_dist in [1, 2, 3]:
        circle = plt.Circle((0, 0), e_dist, fill=False, edgecolor='green', linestyle='--')
        ax.add_patch(circle)
        ax.text(e_dist, 0, f'E-dist = {e_dist}', color='green', ha='left', va='center')
    
    ax.set_title('Mahalanobis Distance vs Euclidean Distance\nfor Correlated Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend()
    
    # Add covariance matrix info
    textstr = f'Covariance Matrix:\n[[{cov_matrix[0,0]:.1f}, {cov_matrix[0,1]:.1f}],\n [{cov_matrix[1,0]:.1f}, {cov_matrix[1,1]:.1f}]]\n\nCorrelation: {corr:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    print("\nStep 7: Creating Additional Visualizations")
    # Create additional visualizations
    fig_3d = create_3d_pdf_visualization(mean, cov_matrix)
    fig_comparison = create_comparison_visualization(mean, cov_matrix, data)
    fig_whitening, whitening_matrix = create_whitening_visualization(data, cov_matrix)
    
    print("\nStep 8: Whitening Transformation Details")
    print("The whitening transformation converts correlated data to uncorrelated data with unit variance.")
    print("Whitening matrix:")
    print(f"[[{whitening_matrix[0,0]:.4f}, {whitening_matrix[0,1]:.4f}],")
    print(f" [{whitening_matrix[1,0]:.4f}, {whitening_matrix[1,1]:.4f}]]")
    
    print("\nStep 9: Testing Mahalanobis Distance after Whitening")
    whitened_point = test_points[0].dot(whitening_matrix)
    print(f"Original test point P1: {test_points[0]}")
    print(f"Whitened test point P1: [{whitened_point[0]:.4f}, {whitened_point[1]:.4f}]")
    
    # After whitening, the covariance matrix becomes identity
    identity_matrix = np.eye(2)
    whitened_euclidean_dist = np.linalg.norm(whitened_point)
    print(f"Euclidean distance of whitened point = {whitened_euclidean_dist:.4f}")
    print(f"Original Mahalanobis distance = {mahalanobis_distances[0]:.4f}")
    print("Notice how the Euclidean distance in whitened space equals the Mahalanobis distance in original space!")
    
    print("\nStep 10: Key Observations from the Visualizations")
    print("- Points at the same Euclidean distance can have very different Mahalanobis distances:")
    print("  * Points along the major axis of correlation have smaller Mahalanobis distances")
    print("  * Points perpendicular to the correlation direction have larger Mahalanobis distances")
    print("- The Mahalanobis distance effectively 'scales' the space according to the data variance")
    print("- Points with the same Mahalanobis distance from the mean form ellipses aligned with the data")
    print("- For uncorrelated data with equal variances, Mahalanobis distance = Euclidean distance")
    print("- Whitening transformation makes Euclidean and Mahalanobis distances equivalent")
    
    print("\nStep 11: Practical Applications")
    print("- Anomaly detection: identifying outliers that account for correlation structure")
    print("- Classification: creating decision boundaries that respect data covariance")
    print("- Clustering: defining distance metrics that capture the natural data structure")
    print("- Quality control: monitoring multivariate processes and detecting unusual states")
    print("- Feature normalization: removing correlations in preprocessing steps")
    
    plt.tight_layout()
    return fig, fig_3d, fig_comparison, fig_whitening

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 7: MAHALANOBIS DISTANCE VS EUCLIDEAN DISTANCE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig_main, fig_3d, fig_comparison, fig_whitening = simple_mahalanobis_distance()
    
    # Save the figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save main visualization
        save_path = os.path.join(images_dir, "ex7_simple_mahalanobis_distance.png")
        fig_main.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nMain figure saved to: {save_path}")
        
        # Save 3D visualization
        save_path_3d = os.path.join(images_dir, "ex7_3d_pdf_visualization.png")
        fig_3d.savefig(save_path_3d, bbox_inches='tight', dpi=300)
        print(f"3D PDF visualization saved to: {save_path_3d}")
        
        # Save comparison visualization
        save_path_comparison = os.path.join(images_dir, "ex7_covariance_comparison.png")
        fig_comparison.savefig(save_path_comparison, bbox_inches='tight', dpi=300)
        print(f"Covariance comparison saved to: {save_path_comparison}")
        
        # Save whitening visualization
        save_path_whitening = os.path.join(images_dir, "ex7_whitening_transformation.png")
        fig_whitening.savefig(save_path_whitening, bbox_inches='tight', dpi=300)
        print(f"Whitening transformation visualization saved to: {save_path_whitening}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    