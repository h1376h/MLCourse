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
    """Create a simple conceptual visualization to introduce height-weight relationships."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw stylized human figures with different heights and weights
    heights = [160, 170, 180]
    weights = [60, 70, 80]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    
    for i, (height, weight, color) in enumerate(zip(heights, weights, colors)):
        # Scale the figures to represent different heights and weights
        h_scale = height / 170  # Scale based on height
        w_scale = weight / 70   # Scale based on weight
        
        # Calculate position with some spacing
        x_pos = i * 3
        
        # Draw a stylized human figure
        # Head
        head_radius = 0.3 * w_scale
        ax.add_patch(plt.Circle((x_pos, h_scale * 5), head_radius, color=color))
        
        # Body
        body_height = h_scale * 3
        body_width = 0.7 * w_scale
        ax.add_patch(plt.Rectangle((x_pos - body_width/2, h_scale * 2), body_width, body_height, 
                                 color=color))
        
        # Legs
        leg_height = h_scale * 2
        leg_width = 0.25 * w_scale
        ax.add_patch(plt.Rectangle((x_pos - body_width/2, 0), leg_width, leg_height, color=color))
        ax.add_patch(plt.Rectangle((x_pos + body_width/2 - leg_width, 0), leg_width, leg_height, 
                                 color=color))
        
        # Print labels instead of adding them to the figure
        print(f"Figure {i+1}: Height: {height}cm, Weight: {weight}kg")
    
    # Add grid showing the growth trend
    x_grid = np.linspace(155, 185, 4)
    y_grid = 0.5 * (x_grid - 170) + 70  # Simple linear relationship
    
    # Draw grid lines
    for x in x_grid:
        ax.axvline(x=6 + (x-170)/5, color='gray', alpha=0.2, linestyle='--')
    for y in y_grid:
        ax.axhline(y=y/10 - 3, color='gray', alpha=0.2, linestyle='--')
    
    # Draw correlation arrow
    ax.arrow(6, 1, 2, 1.5, head_width=0.3, head_length=0.3, 
             fc='red', ec='red', linewidth=2)
    
    # Print annotation instead of adding it to the figure
    print("\nPositive Correlation: Taller → Heavier")
    
    # Set axis properties
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title only
    ax.set_title("Conceptual Visualization of Height-Weight Relationship")
    
    # Print explanation
    print("Height and weight typically show positive correlation in human populations.")
    print("As height increases, weight tends to increase (though with individual variation).")
    
    plt.tight_layout()
    return fig

def simple_covariance_example_real_world():
    """Simple real-world example of covariance using height and weight data."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Height and Weight - A Real-World Covariance Example")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How does natural covariance appear in the real world, and how can it be visualized using height and weight data?")
    
    # Create concept visualization for problem statement
    concept_fig = create_concept_visualization()
    
    print("\nStep 1: Understanding the Natural Relationship")
    print("Height and weight are naturally correlated variables in human populations:")
    print("- Taller people tend to weigh more (positive correlation)")
    print("- This relationship is not deterministic but statistical")
    print("- The covariance structure can be visualized as an elliptical pattern in a scatter plot")
    print("- The direction of maximum variance typically aligns with the 'growth trajectory'")
    
    print("\nMathematical model:")
    print("- Height (cm): h ~ N(170, 7²) (mean 170cm, standard deviation 7cm)")
    print("- Weight (kg): w = 0.5h + ε, where ε ~ N(0, 5²)")
    print("- This creates a positive correlation between height and weight")
    
    # Simulated height (cm) and weight (kg) data with positive correlation
    np.random.seed(42)  # For reproducibility
    heights = 170 + np.random.normal(0, 7, 100)  # Mean 170cm, std 7cm
    weights = heights * 0.5 + np.random.normal(0, 5, 100)  # Positively correlated with heights
    
    print("\nStep 2: Calculating the Covariance Matrix")
    data = np.vstack([heights, weights]).T
    cov_matrix = np.cov(data, rowvar=False)
    
    print(f"Covariance matrix:")
    print(f"[[{cov_matrix[0,0]:.2f}, {cov_matrix[0,1]:.2f}],")
    print(f" [{cov_matrix[1,0]:.2f}, {cov_matrix[1,1]:.2f}]]")
    
    # Calculate correlation coefficient
    corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    print(f"Correlation coefficient: {corr:.2f}")
    
    print("\nCovariance Matrix Explanation:")
    print(f"• Var(Height) = {cov_matrix[0,0]:.2f} cm²")
    print(f"• Var(Weight) = {cov_matrix[1,1]:.2f} kg²")
    print(f"• Cov(Height,Weight) = {cov_matrix[0,1]:.2f} cm·kg")
    print(f"• Correlation(ρ) = {corr:.2f}")
    print(f"• SD(Height) = {np.sqrt(cov_matrix[0,0]):.2f} cm")
    print(f"• SD(Weight) = {np.sqrt(cov_matrix[1,1]):.2f} kg")
    
    print("\nStep 3: Eigendecomposition of the Covariance Matrix")
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(f"Eigenvalues: λ₁ = {eigenvalues[0]:.2f}, λ₂ = {eigenvalues[1]:.2f}")
    print(f"Eigenvectors (as columns): [[{eigenvectors[0,0]:.2f}, {eigenvectors[0,1]:.2f}],")
    print(f"                             [{eigenvectors[1,0]:.2f}, {eigenvectors[1,1]:.2f}]]")
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate mean
    mean_height, mean_weight = np.mean(heights), np.mean(weights)
    
    # Create the scatter plot with 1σ and 2σ ellipse
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    
    # Plot the data points
    ax_scatter.scatter(heights, weights, alpha=0.7, label='Height-Weight Data', c='#4575b4')
    
    # Draw the 1σ and 2σ covariance ellipses
    sigma_values = [1, 2]
    ellipse_colors = ['red', 'orange']
    
    for i, (sigma, color) in enumerate(zip(sigma_values, ellipse_colors)):
        ell = Ellipse(xy=(mean_height, mean_weight),
                     width=2*sigma*np.sqrt(eigenvalues[0]), 
                     height=2*sigma*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor=color, facecolor='none', linestyle='-', linewidth=2)
        ax_scatter.add_patch(ell)
        ax_scatter.text(mean_height, mean_weight + sigma*np.sqrt(eigenvalues[1]) + 2, 
                f'{sigma}σ confidence region', color=color, ha='center', va='bottom')
    
    # Plot the eigenvectors (principal components)
    for i in range(2):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
        ax_scatter.arrow(mean_height, mean_weight, vec[0], vec[1], 
                 head_width=1, head_length=1.5, fc='blue', ec='blue', linewidth=2)
        ax_scatter.text(mean_height + vec[0]*1.1, mean_weight + vec[1]*1.1, 
                f'PC{i+1}', color='blue', ha='center', va='center')
    
    # Calculate the best-fit line
    slope = cov_matrix[0,1] / cov_matrix[0,0]
    intercept = mean_weight - slope * mean_height
    
    # Plot the best-fit line
    x_line = np.array([np.min(heights), np.max(heights)])
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, 'g--', label=f'Best-fit line: w = {slope:.2f}h + {intercept:.2f}')
    
    # Add labels and title
    ax_scatter.set_xlabel('Height (cm)')
    ax_scatter.set_ylabel('Weight (kg)')
    ax_scatter.set_title('Height vs Weight: A Natural Example of Positive Covariance')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc='upper left')
    
    # Create covariance matrix visualization
    fig_matrix, ax_matrix = plt.subplots(figsize=(6, 5))
    ax_matrix.matshow(cov_matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax_matrix.text(j, i, f'{cov_matrix[i, j]:.2f}', 
                         ha='center', va='center', color='black', fontsize=12)
    
    ax_matrix.set_xticklabels(['', 'Height', 'Weight'])
    ax_matrix.set_yticklabels(['', 'Height', 'Weight'])
    ax_matrix.set_title('Covariance Matrix')
    
    # Create 3D visualization of the bivariate normal distribution
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create a 2D grid for 3D plot
    h_grid = np.linspace(min(heights)-5, max(heights)+5, 50)
    w_grid = np.linspace(min(weights)-5, max(weights)+5, 50)
    H, W = np.meshgrid(h_grid, w_grid)
    
    # Calculate PDF values for the grid
    # We need to evaluate the bivariate normal PDF using the covariance matrix
    from scipy.stats import multivariate_normal
    
    # Create the multivariate normal object with our mean and covariance
    rv = multivariate_normal([mean_height, mean_weight], cov_matrix)
    
    # Evaluate the PDF on the grid
    pos = np.dstack((H, W))
    Z = rv.pdf(pos)
    
    # Plot the 3D surface
    surf = ax_3d.plot_surface(H, W, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
    
    # Add contour lines at the base of the 3D plot
    ax_3d.contour(H, W, Z, zdir='z', offset=0, cmap='viridis', levels=10, alpha=0.5)
    
    # Add the original data points at z=0
    ax_3d.scatter(heights, weights, 0, c='red', marker='.', alpha=0.5)
    
    # Label the axes
    ax_3d.set_xlabel('Height (cm)')
    ax_3d.set_ylabel('Weight (kg)')
    ax_3d.set_zlabel('Probability Density')
    ax_3d.set_title('3D Probability Density Function')
    
    # Create a better viewpoint
    ax_3d.view_init(elev=30, azim=45)
    
    # Create marginal distributions plot
    fig_marginal, (ax_height, ax_weight) = plt.subplots(1, 2, figsize=(12, 5))
    
    from scipy.stats import norm
    
    # Height distribution
    bins_h = np.linspace(min(heights)-5, max(heights)+5, 20)
    ax_height.hist(heights, bins=bins_h, density=True, alpha=0.6, color='#4575b4')
    
    # Plot the theoretical normal distribution for height
    h_range = np.linspace(min(heights)-10, max(heights)+10, 100)
    h_norm = norm.pdf(h_range, mean_height, np.sqrt(cov_matrix[0,0]))
    ax_height.plot(h_range, h_norm, 'r-', linewidth=2)
    
    ax_height.set_xlabel('Height (cm)')
    ax_height.set_ylabel('Density')
    ax_height.set_title('Height Distribution')
    
    # Weight distribution
    bins_w = np.linspace(min(weights)-5, max(weights)+5, 20)
    ax_weight.hist(weights, bins=bins_w, density=True, alpha=0.6, color='#4575b4')
    
    # Plot the theoretical normal distribution for weight
    w_range = np.linspace(min(weights)-10, max(weights)+10, 100)
    w_norm = norm.pdf(w_range, mean_weight, np.sqrt(cov_matrix[1,1]))
    ax_weight.plot(w_range, w_norm, 'r-', linewidth=2)
    
    ax_weight.set_xlabel('Weight (kg)')
    ax_weight.set_ylabel('Density')
    ax_weight.set_title('Weight Distribution')
    
    plt.tight_layout()
    
    print("\nStep 5: Interpreting the Results")
    print("The visualization reveals key insights:")
    print("- The data cloud forms an elongated elliptical pattern")
    print("- The first principal component points along the 'growth direction'")
    print("  where both height and weight increase together")
    print("- The second principal component represents variations in body type")
    print("  (more weight relative to height or vice versa)")
    print("- The angle of the first principal component indicates the rate")
    print("  of weight change relative to height")
    print("- The eccentricity of the ellipse reflects the strength of the correlation")
    
    print("\nReal-World Applications:")
    print("- Medical research and anthropometry: establishing normal ranges and relationships")
    print("- Clothing industry: designing size systems based on correlated body measurements")
    print("- Sports science: analyzing performance metrics and their relationships")
    print("- Public health: monitoring population trends in body metrics")
    
    # Save all the figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save the concept figure
        concept_save_path = os.path.join(images_dir, "ex5_concept_visualization.png")
        concept_fig.savefig(concept_save_path, bbox_inches='tight', dpi=300)
        print(f"\nConcept visualization saved to: {concept_save_path}")
        
        # Save the scatter plot with ellipses
        scatter_save_path = os.path.join(images_dir, "ex5_simple_covariance_real_world.png")
        fig_scatter.savefig(scatter_save_path, bbox_inches='tight', dpi=300)
        print(f"\nScatter plot with covariance ellipses saved to: {scatter_save_path}")
        
        # Save the covariance matrix visualization
        matrix_save_path = os.path.join(images_dir, "ex5_covariance_matrix.png")
        fig_matrix.savefig(matrix_save_path, bbox_inches='tight', dpi=300)
        print(f"\nCovariance matrix visualization saved to: {matrix_save_path}")
        
        # Save the 3D visualization
        plot3d_save_path = os.path.join(images_dir, "ex5_3d_visualization.png")
        fig_3d.savefig(plot3d_save_path, bbox_inches='tight', dpi=300)
        print(f"\n3D visualization saved to: {plot3d_save_path}")
        
        # Save the marginal distributions
        marginal_save_path = os.path.join(images_dir, "ex5_marginal_distributions.png")
        fig_marginal.savefig(marginal_save_path, bbox_inches='tight', dpi=300)
        print(f"\nMarginal distributions saved to: {marginal_save_path}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    
    return fig_scatter, fig_matrix, fig_3d, fig_marginal

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 5: HEIGHT-WEIGHT REAL-WORLD COVARIANCE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    figures = simple_covariance_example_real_world()
    