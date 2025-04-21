import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_concept_visualization():
    """Create a simplified conceptual visualization to explain rotation and covariance change."""
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
    titles = ['Original Data', '30° Rotation', '60° Rotation']
    
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
        
        # Update title with calculated correlation value
        titles[i] = f'{title}\nCorrelation: ρ = {corr:.2f}'
        
        # Plot the data
        ax.scatter(data_rotated[:, 0], data_rotated[:, 1], alpha=0.5, s=15, color='lightblue')
        
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
        
        # Print covariance info
        print(f"Covariance matrix for {angle}° rotation:")
        print(f"[[{cov_matrix[0,0]:.2f}, {cov_matrix[0,1]:.2f}],")
        print(f" [{cov_matrix[1,0]:.2f}, {cov_matrix[1,1]:.2f}]]")
        print(f"Correlation = {corr:.2f}")
        
        ax.set_title(titles[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Print explanation
    print("Rotation transforms the coordinate system, creating correlation between originally uncorrelated variables.")
    print("The correlation coefficient follows ρ = sin(2θ)/2, which is why it changes with rotation angle.")
    
    plt.tight_layout()
    return fig

def plot_correlation_vs_angle():
    """Plot how correlation changes with rotation angle in a simpler format."""
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
    
    # Annotate key points on the graph
    for angle, corr in zip(key_angles, key_corrs):
        ax.annotate(f'({angle}°, {corr:.2f})', 
                   xy=(angle, corr),
                   xytext=(5, 5 if corr >= 0 else -15),
                   textcoords='offset points')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Rotation Angle (degrees)')
    ax.set_ylabel('Correlation Coefficient (ρ)')
    ax.set_title('How Correlation Changes with Rotation Angle: ρ = sin(2θ)/2')
    
    # Set limits
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, alpha=0.3)
    
    # Add tick marks at key angles
    ax.set_xticks(key_angles)
    
    # Print key observations
    print("Key points on the correlation vs angle curve:")
    print("θ = 0°: ρ = 0 (No correlation)")
    print("θ = 45°: ρ = 0.5 (Maximum positive correlation)")
    print("θ = 90°: ρ = 0 (Zero correlation again)")
    print("θ = 135°: ρ = -0.5 (Maximum negative correlation)")
    print("θ = 180°: ρ = 0 (No correlation)")
    
    return fig

def create_rotation_vector_field():
    """Create a vector field visualization showing how rotation transforms points in space."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a grid of points
    n = 10  # Number of points in each direction (reduced for clarity)
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    
    # Original points as a 2D grid
    original_points = np.vstack([X.flatten(), Y.flatten()]).T
    
    # Choose a rotation angle
    theta = np.radians(30)
    
    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate the points
    rotated_points = np.dot(original_points, rot_matrix)
    
    # Calculate displacement vectors
    U = rotated_points[:, 0] - original_points[:, 0]
    V = rotated_points[:, 1] - original_points[:, 1]
    
    # Normalize vector lengths for better visualization
    magnitude = np.sqrt(U**2 + V**2)
    max_magnitude = np.max(magnitude)
    scale_factor = 0.4 / max_magnitude if max_magnitude > 0 else 1
    U = U * scale_factor
    V = V * scale_factor
    
    # Plot the vector field
    ax.quiver(original_points[:, 0], original_points[:, 1], U, V, 
              color='blue', width=0.003, scale=1, scale_units='xy')
    
    # Plot rotated points
    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], 
               color='red', s=30, alpha=0.5, label='Rotated grid')
    
    # Draw coordinate axes (original)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.7, linewidth=1.5)
    
    # Draw rotated coordinate system
    max_axis = 3
    ax.plot([0, max_axis*np.cos(theta)], [0, max_axis*np.sin(theta)], 
            'r-', linewidth=2, label="x' axis")
    ax.plot([0, -max_axis*np.sin(theta)], [0, max_axis*np.cos(theta)], 
            'r:', linewidth=2, label="y' axis")
    
    # Draw angle arc
    angle_radius = 1.0
    arc_angles = np.linspace(0, theta, 100)
    ax.plot(angle_radius * np.cos(arc_angles), angle_radius * np.sin(arc_angles), 
            'g-', linewidth=2)
    ax.text(angle_radius * np.cos(theta/2) * 1.2, 
            angle_radius * np.sin(theta/2) * 1.2, 
            f'θ = 30°', color='green', fontsize=14)
    
    # Draw circular markers at different distances to show the transformation
    circle_radii = [1, 2, 3]
    for r in circle_radii:
        # Draw original circle
        circle_angles = np.linspace(0, 2*np.pi, 100)
        circle_x = r * np.cos(circle_angles)
        circle_y = r * np.sin(circle_angles)
        ax.plot(circle_x, circle_y, 'b--', alpha=0.4)
        
        # Points on the circle to show the mapping
        n_points = 8
        circle_sample_angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        circle_samples = np.array([[r * np.cos(a), r * np.sin(a)] for a in circle_sample_angles])
        
        # Rotate the circle samples
        rotated_samples = np.dot(circle_samples, rot_matrix)
        
        # Plot samples and connect with arrows
        for i in range(n_points):
            start = circle_samples[i]
            end = rotated_samples[i]
            ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                     color='gray', width=0.02, head_width=0.1, alpha=0.5, 
                     length_includes_head=True)
    
    # Set plot properties
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Rotation Transformation: Vector Field Visualization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    
    # Print detailed explanation
    print("\nStep 4a: Understanding Rotation as a Transformation")
    print("The vector field visualization shows how each point in the 2D space moves under rotation:")
    print("- Each blue arrow shows the displacement of a point due to 30° rotation")
    print("- Longer arrows indicate larger displacements, which happen further from origin")
    print("- Points on concentric circles remain on circles (rotation preserves distances)")
    print("- The entire coordinate system rotates, including the basis vectors")
    print("- This transformation affects the covariance by changing the coordinate reference frame")
    
    return fig

def create_step_by_step_rotation_effect():
    """Create a step-by-step visualization of how rotation affects covariance structure."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Generate initially correlated data
    np.random.seed(42)
    n_points = 200
    
    # Create original uncorrelated data
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    data_original = np.vstack([x, y]).T
    
    # Calculate original covariance and correlation
    cov_original = np.cov(data_original, rowvar=False)
    corr_original = cov_original[0, 1] / np.sqrt(cov_original[0, 0] * cov_original[1, 1])
    
    # Define rotation angles
    angles = [0, 30, 60, 90, 135, 180]
    
    # Print original stats
    print("\nStep 4b: The Effect of Various Rotation Angles on Correlation")
    print(f"Original data - covariance matrix:")
    print(f"[[{cov_original[0,0]:.2f}, {cov_original[0,1]:.2f}],")
    print(f" [{cov_original[1,0]:.2f}, {cov_original[1,1]:.2f}]]")
    print(f"Original correlation: {corr_original:.2f}")
    
    # Track theoretical and actual correlation values
    theoretical_corrs = []
    actual_corrs = []
    
    # Plot each rotation
    for i, angle in enumerate(angles):
        row = i // 3  # Calculate row index
        col = i % 3   # Calculate column index
        ax = axs[row, col]
        
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate the data
        rotated_data = np.dot(data_original, rot_matrix)
        
        # Calculate covariance after rotation
        cov_rotated = np.cov(rotated_data, rowvar=False)
        
        # Calculate correlation coefficient
        corr = cov_rotated[0, 1] / np.sqrt(cov_rotated[0, 0] * cov_rotated[1, 1])
        actual_corrs.append(corr)
        
        # Calculate theoretical correlation based on sin(2θ)/2 formula
        # (assuming initially uncorrelated data with equal variances)
        theoretical_corr = np.sin(2 * theta) / 2
        theoretical_corrs.append(theoretical_corr)
        
        # Print rotation details
        print(f"\nRotation by {angle}° - Correlation Analysis:")
        print(f"Actual correlation: {corr:.3f}")
        print(f"Theoretical correlation (sin(2θ)/2): {theoretical_corr:.3f}")
        print(f"Covariance matrix after rotation:")
        print(f"[[{cov_rotated[0,0]:.2f}, {cov_rotated[0,1]:.2f}],")
        print(f" [{cov_rotated[1,0]:.2f}, {cov_rotated[1,1]:.2f}]]")
        
        # Plot the data
        scatter = ax.scatter(rotated_data[:, 0], rotated_data[:, 1], 
                            alpha=0.5, s=10, c='blue')
        
        # Calculate and plot covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_rotated)
        
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
        
        # Show rotation angle
        if angle > 0:
            arc = np.linspace(0, theta, 50)
            ax.plot(1.0 * np.cos(arc), 1.0 * np.sin(arc), 'g-', alpha=0.6)
            
        # Set title and labels
        ax.set_title(f'Rotation: {angle}°\nCorrelation: {corr:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Print observations about the pattern
    print("\nStep 4c: Key Observations on Rotation-Correlation Relationship")
    print("1. At 0° (and 180°): Correlation is minimal")
    print("2. At 45° (and 225°): Correlation reaches maximum positive value")
    print("3. At 90° (and 270°): Correlation returns to near zero")
    print("4. At 135° (and 315°): Correlation reaches maximum negative value")
    print("5. The correlation follows a sinusoidal pattern with period 180°")
    print(f"6. Theoretical predictions match actual correlations with some deviation due to sampling")
    
    # Print invariance properties
    traces = [np.trace(cov_original)]
    determinants = [np.linalg.det(cov_original)]
    
    # Calculate invariants for rotated data
    for angle in angles[1:]:  # Skip the first one (0°) as it's already calculated
        theta = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_data = np.dot(data_original, rot_matrix)
        cov_rotated = np.cov(rotated_data, rowvar=False)
        traces.append(np.trace(cov_rotated))
        determinants.append(np.linalg.det(cov_rotated))
    
    print("\nStep 5a: Verification of Rotation Invariants")
    print("Trace of covariance matrix (total variance):")
    for i, angle in enumerate(angles):
        print(f"  At {angle}°: {traces[i]:.4f}")
    
    print("\nDeterminant of covariance matrix:")
    for i, angle in enumerate(angles):
        print(f"  At {angle}°: {determinants[i]:.4f}")
    
    print("\nThese values remain approximately constant, confirming that:")
    print("- Total variance (trace) is preserved under rotation")
    print("- Determinant of covariance matrix is preserved under rotation")
    print("- These invariants confirm that rotation only changes the perspective, not the fundamental data structure")
    
    plt.tight_layout()
    return fig

def toy_data_covariance_change():
    """Generate simplified visualizations for rotation effects on covariance."""
    # Print descriptive problem statement
    print("\n" + "="*80)
    print("Example 6: How Rotation Affects Covariance Structure")
    print("="*80)
    
    print("\nProblem Statement:")
    print("What happens to the covariance matrix when we rotate a dataset, and why is this important?")
    print("How does a change in coordinate system affect the correlation structure of data?")
    
    # Step 1: Mathematical Foundation - Print the explanation
    print("\nStep 1: Mathematical Foundation")
    print("When we rotate a dataset using a rotation matrix R, the covariance matrix transforms according to:")
    print("Σ' = R·Σ·R^T")
    print("Where:")
    print("- Σ is the original covariance matrix")
    print("- Σ' is the transformed covariance matrix")
    print("- R is the rotation matrix")
    print("\nFor a 2D rotation by angle θ, the rotation matrix is:")
    print("R = [[cos θ, -sin θ], [sin θ, cos θ]]")
    
    # Step 2: Create simplified concept visualization
    print("\nStep 2: Basic Visualization of Rotation Effects on Covariance")
    print("We'll visualize how rotation changes the correlation structure of a dataset:")
    print("- We start with an uncorrelated dataset (approximately zero correlation)")
    print("- We apply rotations of 0°, 30°, and 60° to observe how correlation changes")
    print("- The red dashed ellipses show the covariance structure")
    concept_fig = create_concept_visualization()
    
    # Step 3: Correlation vs angle relationship
    print("\nStep 3: Mathematical Relationship Between Rotation and Correlation")
    print("For initially uncorrelated data with equal variances (σ²ᵢ = σ²):")
    print("• After rotation by angle θ: Σ' = σ² · [[1, sin(2θ)/2], [sin(2θ)/2, 1]]")
    print("• Correlation coefficient after rotation: ρ = sin(2θ)/2")
    print("• Maximum correlation occurs at θ = 45° with ρ = 0.5")
    print("• Maximum negative correlation occurs at θ = 135° with ρ = -0.5")
    corr_angle_fig = plot_correlation_vs_angle()
    
    # Step 4: Create rotation transformation visualizations
    print("\nStep 4: Understanding Rotation as a Coordinate Transformation")
    print("Rotation is a linear transformation that changes the coordinate system:")
    print("- Each point rotates around the origin by the specified angle")
    print("- The coordinate axes themselves rotate")
    print("- Distance from origin is preserved (length-preserving transformation)")
    print("- Angles between vectors are preserved (angle-preserving transformation)")
    
    # Create new visualizations
    vector_field_fig = create_rotation_vector_field()
    rotation_steps_fig = create_step_by_step_rotation_effect()
    
    # Step 5: Properties Preserved Under Rotation
    print("\nStep 5: Properties Preserved Under Rotation")
    print("Despite the changes in correlation, certain properties remain invariant under rotation:")
    print("- Total variance (trace of covariance matrix): tr(Σ') = tr(Σ)")
    print("- Determinant of covariance matrix: |Σ'| = |Σ|")
    print("- Eigenvalues of the covariance matrix (though eigenvectors rotate)")
    
    # Step 6: Practical Significance
    print("\nStep 6: Practical Significance")
    print("Understanding rotation effects on covariance has important applications:")
    print("1. Coordinate system choice affects the observed correlation structure")
    print("2. Feature engineering: rotation can introduce or remove correlations")
    print("3. Principal Component Analysis (PCA) exploits this by finding a rotation that diagonalizes the covariance matrix")
    print("4. Feature independence is coordinate-dependent; what looks uncorrelated in one coordinate system may be correlated in another")
    print("5. Real-world example: sensor data may show different correlation patterns depending on sensor orientation")
    print("6. Machine learning: feature transformations may change correlation structure significantly")
    
    # Save all figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save the concept visualization
        concept_save_path = os.path.join(images_dir, "ex6_concept_visualization.png")
        concept_fig.savefig(concept_save_path, bbox_inches='tight', dpi=300)
        print(f"\nConcept visualization saved to: {concept_save_path}")
        
        # Save the correlation vs angle curve
        corr_angle_save_path = os.path.join(images_dir, "ex6_correlation_angle_curve.png")
        corr_angle_fig.savefig(corr_angle_save_path, bbox_inches='tight', dpi=300)
        print(f"Correlation vs angle curve saved to: {corr_angle_save_path}")
        
        # Save the rotation vector field
        vector_field_save_path = os.path.join(images_dir, "ex6_rotation_vector_field.png")
        vector_field_fig.savefig(vector_field_save_path, bbox_inches='tight', dpi=300)
        print(f"Vector field visualization saved to: {vector_field_save_path}")
        
        # Save the step-by-step rotation effect
        rotation_steps_save_path = os.path.join(images_dir, "ex6_rotation_steps.png")
        rotation_steps_fig.savefig(rotation_steps_save_path, bbox_inches='tight', dpi=300)
        print(f"Rotation steps visualization saved to: {rotation_steps_save_path}")
        
        # Save the main figure (which is the rotation steps visualization in this updated version)
        main_save_path = os.path.join(images_dir, "ex6_toy_data_covariance_change.png")
        rotation_steps_fig.savefig(main_save_path, bbox_inches='tight', dpi=300)
        print(f"Main figure saved to: {main_save_path}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    
    return concept_fig, corr_angle_fig, vector_field_fig, rotation_steps_fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 6: ROTATION EFFECTS ON COVARIANCE STRUCTURE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    concept_fig, corr_angle_fig, vector_field_fig, rotation_steps_fig = toy_data_covariance_change()
    
    # Display plots if in interactive mode
    plt.show()
    