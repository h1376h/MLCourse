import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def toy_data_covariance_change():
    """Visualize how a dataset's covariance changes with rotation."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: How Rotation Affects Covariance Structure")
    print("="*80)
    
    print("\nProblem Statement:")
    print("What happens to the covariance matrix when we rotate a dataset, and why is this important?")
    
    print("\nStep 1: Mathematical Foundation")
    print("When we rotate a dataset using a rotation matrix R, the covariance matrix transforms according to:")
    print("Σ' = R·Σ·R^T")
    print("Where:")
    print("- Σ is the original covariance matrix")
    print("- Σ' is the transformed covariance matrix")
    print("- R is the rotation matrix")
    print("\nFor a 2D rotation by angle θ, the rotation matrix is:")
    print("R = [[cos θ, -sin θ], [sin θ, cos θ]]")
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    print("\nStep 2: Creating Original Uncorrelated Data")
    print("We start with a dataset where variables are uncorrelated:")
    print("- Mean vector: μ = [0, 0]")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]] (identity matrix)")
    print("- This represents independent variables with equal variances")
    print("- The contours form circles centered at the origin")
    print("- Zero correlation: ρ = 0")
    
    # Create a simple 2D dataset
    np.random.seed(42)
    n_points = 300
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    data_original = np.vstack([x, y]).T
    
    # Calculate original covariance
    cov_original = np.cov(data_original, rowvar=False)
    corr_original = cov_original[0, 1] / np.sqrt(cov_original[0, 0] * cov_original[1, 1])
    
    print(f"Original covariance matrix:")
    print(f"[[{cov_original[0,0]:.2f}, {cov_original[0,1]:.2f}],")
    print(f" [{cov_original[1,0]:.2f}, {cov_original[1,1]:.2f}]]")
    print(f"Original correlation: {corr_original:.2f}")
    
    # Rotation matrices for different angles
    angles = [0, 30, 60]
    titles = ['Original Data', '30° Rotation', '60° Rotation']
    
    for i, (angle, title) in enumerate(zip(angles, titles)):
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        print(f"\nStep {i+3}: Applying {angle}° Rotation")
        print(f"Rotation matrix for {angle}°:")
        print(f"[[{rot_matrix[0,0]:.3f}, {rot_matrix[0,1]:.3f}],")
        print(f" [{rot_matrix[1,0]:.3f}, {rot_matrix[1,1]:.3f}]]")
        
        # Rotate the data
        data_rotated = np.dot(data_original, rot_matrix)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(data_rotated, rowvar=False)
        corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        
        print(f"Covariance matrix after {angle}° rotation:")
        print(f"[[{cov_matrix[0,0]:.2f}, {cov_matrix[0,1]:.2f}],")
        print(f" [{cov_matrix[1,0]:.2f}, {cov_matrix[1,1]:.2f}]]")
        print(f"Correlation coefficient: {corr:.2f}")
        
        # Expected correlation for identity covariance matrix with equal variances:
        expected_corr = np.sin(2 * theta) / 2
        print(f"Expected correlation (sin(2θ)/2): {expected_corr:.2f}")
        
        # Plot the data
        axs[i].scatter(data_rotated[:, 0], data_rotated[:, 1], alpha=0.5, s=10)
        
        # Get eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Draw 2σ ellipse
        ell = Ellipse(xy=(0, 0),
                     width=4*np.sqrt(eigenvalues[0]), 
                     height=4*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        axs[i].add_patch(ell)
        
        # Add covariance info
        axs[i].text(0.05, 0.95, f'Cov(x,y) = {cov_matrix[0,1]:.2f}\nCorr = {corr:.2f}', 
                   transform=axs[i].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axs[i].set_title(title)
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_xlim(-4, 4)
        axs[i].set_ylim(-4, 4)
        axs[i].grid(True)
        axs[i].set_aspect('equal')
    
    print("\nStep 6: General Pattern for Rotation Effects")
    print("For initially uncorrelated data with equal variances (Σ = σ²I), rotation by angle θ produces:")
    print("Σ' = σ² [[1, sin(2θ)/2], [sin(2θ)/2, 1]]")
    print("\nThe correlation coefficient follows the pattern:")
    print("ρ = sin(2θ)/2")
    print("\nKey observations:")
    print("- At θ = 0°: ρ = 0 (no correlation)")
    print("- At θ = 45°: ρ = 0.5 (maximum correlation)")
    print("- At θ = 90°: ρ = 0 (variables effectively swap positions)")
    print("- The correlation oscillates as rotation angle increases")
    
    print("\nStep 7: Properties Preserved Under Rotation")
    print("Despite the changes in correlation, certain properties remain invariant:")
    print("- Total variance (trace of covariance matrix): tr(Σ') = tr(Σ)")
    print("- Determinant of covariance matrix: |Σ'| = |Σ|")
    print("- Eigenvalues of the covariance matrix (though eigenvectors rotate)")
    
    # Check invariants
    print(f"\nDemonstrating invariants:")
    print(f"Original trace: {np.trace(cov_original):.4f}, Rotated trace (60°): {np.trace(cov_matrix):.4f}")
    print(f"Original determinant: {np.linalg.det(cov_original):.4f}, Rotated determinant (60°): {np.linalg.det(cov_matrix):.4f}")
    
    print("\nStep 8: Practical Significance")
    print("Understanding rotation effects on covariance has important applications:")
    print("1. Coordinate system choice affects the observed correlation structure")
    print("2. Feature engineering: rotation can introduce or remove correlations")
    print("3. Principal Component Analysis (PCA) exploits this by finding a rotation that diagonalizes the covariance matrix")
    print("4. Feature independence is coordinate-dependent; what looks uncorrelated in one coordinate system may be correlated in another")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 6: ROTATION EFFECTS ON COVARIANCE STRUCTURE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = toy_data_covariance_change()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "toy_data_covariance_change.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    