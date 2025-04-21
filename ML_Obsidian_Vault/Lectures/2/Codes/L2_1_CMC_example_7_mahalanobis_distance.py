import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    for point in test_points:
        diff = point - mean
        mahalanobis_distance = np.sqrt(diff.dot(cov_inv).dot(diff))
        euclidean_distance = np.sqrt(diff.dot(diff))
        mahalanobis_distances.append(mahalanobis_distance)
        euclidean_distances.append(euclidean_distance)
    
    print("Distances for each test point:")
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
    
    print("\nStep 7: Key Observations from the Visualization")
    print("- Points at the same Euclidean distance can have very different Mahalanobis distances:")
    print("  * Points along the major axis of correlation have smaller Mahalanobis distances")
    print("  * Points perpendicular to the correlation direction have larger Mahalanobis distances")
    print("- The Mahalanobis distance effectively 'scales' the space according to the data variance")
    print("- Points with the same Mahalanobis distance from the mean form ellipses aligned with the data")
    print("- For uncorrelated data with equal variances, Mahalanobis distance = Euclidean distance")
    
    print("\nStep 8: Practical Applications")
    print("- Anomaly detection: identifying outliers that account for correlation structure")
    print("- Classification: creating decision boundaries that respect data covariance")
    print("- Clustering: defining distance metrics that capture the natural data structure")
    print("- Quality control: monitoring multivariate processes and detecting unusual states")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 7: MAHALANOBIS DISTANCE VS EUCLIDEAN DISTANCE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = simple_mahalanobis_distance()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "simple_mahalanobis_distance.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    plt.show() 