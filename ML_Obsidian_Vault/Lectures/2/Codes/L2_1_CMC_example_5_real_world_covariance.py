import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def simple_covariance_example_real_world():
    """Simple real-world example of covariance using height and weight data."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Height and Weight - A Real-World Covariance Example")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How does natural covariance appear in the real world, and how can it be visualized using height and weight data?")
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    print("\nStep 3: Eigendecomposition of the Covariance Matrix")
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(f"Eigenvalues: λ₁ = {eigenvalues[0]:.2f}, λ₂ = {eigenvalues[1]:.2f}")
    print(f"Eigenvectors (as columns): [[{eigenvectors[0,0]:.2f}, {eigenvectors[0,1]:.2f}],")
    print(f"                             [{eigenvectors[1,0]:.2f}, {eigenvectors[1,1]:.2f}]]")
    
    print("\nStep 4: Visualizing with Confidence Ellipses")
    # Plot the data points
    ax.scatter(heights, weights, alpha=0.7, label='Height-Weight Data')
    
    # Calculate mean
    mean_height, mean_weight = np.mean(heights), np.mean(weights)
    
    # Draw the covariance ellipse (1σ and 2σ)
    for j in [1, 2]:
        ell = Ellipse(xy=(mean_height, mean_weight),
                     width=2*j*np.sqrt(eigenvalues[0]), 
                     height=2*j*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(ell)
        if j == 2:
            ax.text(mean_height, mean_weight + j*np.sqrt(eigenvalues[1]), 
                    f'{j}σ confidence region', color='red', ha='center', va='bottom')
    
    # Plot the eigenvectors (principal components)
    for i in range(2):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.arrow(mean_height, mean_weight, vec[0], vec[1], 
                 head_width=1, head_length=1.5, fc='blue', ec='blue')
        ax.text(mean_height + vec[0]*1.1, mean_weight + vec[1]*1.1, 
                f'PC{i+1}', color='blue', ha='center', va='center')
    
    # Add labels and title
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Height vs Weight: A Natural Example of Positive Covariance')
    ax.grid(True)
    ax.axis('equal')
    
    # Add text explaining the covariance
    textstr = f'Covariance Matrix:\n[[{cov_matrix[0,0]:.1f}, {cov_matrix[0,1]:.1f}],\n [{cov_matrix[1,0]:.1f}, {cov_matrix[1,1]:.1f}]]\n\nCorrelation: {corr:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
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
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 5: HEIGHT-WEIGHT REAL-WORLD COVARIANCE")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = simple_covariance_example_real_world()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "simple_covariance_real_world.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    