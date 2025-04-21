import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_concept_visualization():
    """Create a conceptual visualization to explain rotation and covariance change."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original data (uncorrelated)
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    data_original = np.vstack([x, y]).T
    
    # Calculate original covariance
    cov_original = np.cov(data_original, rowvar=False)
    
    # Create rotated versions
    angles = [0, 30, 60]
    titles = ['Original Data\nNo Correlation', 
              'Rotated by 30°\nCorrelation Introduced', 
              'Rotated by 60°\nMaximum Correlation']
    
    for i, (angle, title) in enumerate(zip(angles, titles)):
        ax = axs[i]
        
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate the data
        if angle == 0:
            data_rotated = data_original.copy()
        else:
            data_rotated = np.dot(data_original, rot_matrix)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(data_rotated, rowvar=False)
        corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        
        # Plot the data
        ax.scatter(data_rotated[:, 0], data_rotated[:, 1], alpha=0.5, s=15)
        
        # Calculate and plot covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 2σ ellipse
        ell = Ellipse(xy=(0, 0),
                     width=4*np.sqrt(eigenvalues[0]), 
                     height=4*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(ell)
        
        # Show coordinate axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Show rotation lines for non-zero rotations
        if angle > 0:
            # Draw the rotation angle
            arc = np.linspace(0, theta, 50)
            ax.plot(1.5 * np.cos(arc), 1.5 * np.sin(arc), 'g-', alpha=0.6)
            
            # Draw the original x-axis and its rotated version
            ax.plot([0, 2], [0, 0], 'b-', alpha=0.4)  # Original x-axis
            ax.plot([0, 2*np.cos(theta)], [0, 2*np.sin(theta)], 'r-', alpha=0.4)  # Rotated x-axis
        
        # Add covariance info as print output instead of on the plot
        print(f"Covariance and correlation for {title}:")
        print(f"Cov(x,y) = {cov_matrix[0,1]:.2f}")
        print(f"Corr = {corr:.2f}\n")
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Add overall explanation as a print statement instead of on the figure
    print("Rotation transforms the coordinate system, creating artificial correlation between originally uncorrelated variables.")
    print("This shows why correlation is coordinate-dependent and can be introduced or removed through rotation.\n")
    
    plt.tight_layout()
    return fig

def plot_correlation_vs_angle():
    """Plot how correlation changes with rotation angle."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Generate angle values
    theta_deg = np.linspace(0, 180, 181)
    theta_rad = np.radians(theta_deg)
    
    # Calculate correlation coefficient for each angle
    corr_values = np.sin(2 * theta_rad) / 2
    
    # Plot the correlation vs angle curve
    ax.plot(theta_deg, corr_values, 'b-', linewidth=2)
    
    # Highlight key angles
    key_angles = [0, 30, 45, 60, 90, 135, 180]
    key_corrs = np.sin(2 * np.radians(key_angles)) / 2
    ax.plot(key_angles, key_corrs, 'ro', markersize=8)
    
    # Print key point explanations instead of annotations
    print("Key points on the correlation vs angle curve:")
    print("θ = 0°: ρ = 0 (No correlation)")
    print("θ = 45°: ρ = 0.5 (Maximum positive correlation)")
    print("θ = 90°: ρ = 0 (Zero correlation again)")
    print("θ = 135°: ρ = -0.5 (Maximum negative correlation)")
    print("θ = 180°: ρ = 0 (No correlation)\n")
    
    # Print explanation of the pattern
    print("Correlation follows ρ = sin(2θ)/2")
    print("This mathematical relationship explains why correlation oscillates as rotation angle increases.\n")
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Correlation Coefficient (ρ)')
    ax.set_title('How Correlation Changes with Rotation Angle')
    
    # Set limits
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.8, 0.8)
    ax.grid(True, alpha=0.3)
    
    # Add tick marks at key angles
    ax.set_xticks(key_angles)
    
    return fig

def toy_data_covariance_change():
    """Visualize how a dataset's covariance changes with rotation."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: How Rotation Affects Covariance Structure")
    print("="*80)
    
    print("\nProblem Statement:")
    print("What happens to the covariance matrix when we rotate a dataset, and why is this important?")
    print("How does a change in coordinate system affect the correlation structure of data?")
    
    # Create conceptual visualization
    concept_fig = create_concept_visualization()
    
    # Create correlation vs angle plot
    corr_angle_fig = plot_correlation_vs_angle()
    
    print("\nStep 1: Mathematical Foundation")
    print("When we rotate a dataset using a rotation matrix R, the covariance matrix transforms according to:")
    print("Σ' = R·Σ·R^T")
    print("Where:")
    print("- Σ is the original covariance matrix")
    print("- Σ' is the transformed covariance matrix")
    print("- R is the rotation matrix")
    print("\nFor a 2D rotation by angle θ, the rotation matrix is:")
    print("R = [[cos θ, -sin θ], [sin θ, cos θ]]")
    
    # Create main figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 3, height_ratios=[1, 1])
    
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
    
    # Create a subplot for the formula visualization
    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis('off')
    
    # Print the mathematical explanation instead of showing it on the plot
    print("\nMathematical Transformation of Covariance Under Rotation:")
    print("Σ' = R · Σ · Rᵀ")
    print("R = rotation matrix = [[cos θ, -sin θ], [sin θ, cos θ]]")
    print("Σ = original covariance matrix")
    print("Σ' = transformed covariance matrix")
    
    print("\nFor initially uncorrelated data with equal variances (σ²ᵢ = σ²):")
    print("• Original covariance matrix: Σ = σ² · I (identity matrix)")
    print("• After rotation by angle θ: Σ' = σ² · [[1, sin(2θ)/2], [sin(2θ)/2, 1]]")
    print("• Correlation coefficient after rotation: ρ = sin(2θ)/2")
    print("• Maximum correlation occurs at θ = 45° with ρ = 0.5")
    
    # Create subplots for each rotation angle
    axs = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    for i, (angle, title, ax) in enumerate(zip(angles, titles, axs)):
        print(f"\nStep {i+3}: Applying {angle}° Rotation")
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
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
        sc = ax.scatter(data_rotated[:, 0], data_rotated[:, 1], alpha=0.5, s=10, c=data_original[:, 0], 
                      cmap='coolwarm')
        
        # Get eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Draw 2σ ellipse
        ell = Ellipse(xy=(0, 0),
                     width=4*np.sqrt(eigenvalues[0]), 
                     height=4*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(ell)
        
        # Print covariance info instead of showing on plot
        print(f"Covariance Matrix:")
        print(f"[[{cov_matrix[0,0]:.2f}, {cov_matrix[0,1]:.2f}],")
        print(f"[{cov_matrix[1,0]:.2f}, {cov_matrix[1,1]:.2f}]]")
        print(f"Correlation = {corr:.2f}")
        
        # Show coordinate axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Show rotation angle for non-zero rotations
        if angle > 0:
            # Draw the rotation angle
            arc = np.linspace(0, theta, 50)
            ax.plot(1.5 * np.cos(arc), 1.5 * np.sin(arc), 'g-', alpha=0.6)
            
            # Draw the original x-axis and its rotated version
            ax.plot([0, 2], [0, 0], 'b-', alpha=0.4)  # Original x-axis
            ax.plot([0, 2*np.cos(theta)], [0, 2*np.sin(theta)], 'r-', alpha=0.4)  # Rotated x-axis
        
        ax.set_title(f'{title}\nCorrelation: ρ = {corr:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Add a colorbar to show that points are colored by their original x-coordinate
    cbar = plt.colorbar(sc, ax=axs, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Original x-coordinate (before rotation)')
    
    print("\nStep 6: General Pattern for Rotation Effects")
    print("For initially uncorrelated data with equal variances (Σ = σ²I), rotation by angle θ produces:")
    print("Σ' = σ² [[1, sin(2θ)/2], [sin(2θ)/2, 1]]")
    print("\nThe correlation coefficient follows the pattern:")
    print("ρ = sin(2θ)/2")
    print("\nKey observations:")
    print("- At θ = 0°: ρ = 0 (no correlation)")
    print("- At θ = 45°: ρ = 0.5 (maximum correlation)")
    print("- At θ = 90°: ρ = 0 (variables effectively swap positions)")
    print("- At θ = 135°: ρ = -0.5 (maximum negative correlation)")
    print("- At θ = 180°: ρ = 0 (back to uncorrelated)")
    print("- The correlation oscillates as rotation angle increases")
    
    print("\nStep 7: Properties Preserved Under Rotation")
    print("Despite the changes in correlation, certain properties remain invariant under rotation:")
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
    
    # Save the supplementary figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        concept_save_path = os.path.join(images_dir, "ex6_concept_visualization.png")
        concept_fig.savefig(concept_save_path, bbox_inches='tight', dpi=300)
        print(f"\nConcept visualization saved to: {concept_save_path}")
        
        corr_angle_save_path = os.path.join(images_dir, "ex6_correlation_angle_curve.png")
        corr_angle_fig.savefig(corr_angle_save_path, bbox_inches='tight', dpi=300)
        print(f"Correlation vs angle curve saved to: {corr_angle_save_path}")
    except Exception as e:
        print(f"\nError saving supplementary figures: {e}")
    
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
        save_path = os.path.join(images_dir, "ex6_toy_data_covariance_change.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    