import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def generate_geometric_interpretation():
    """
    Generate a visualization of the geometric interpretation of linear regression as projection.
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sample data in 3D space
    # We'll create points in 3D where the column space of X is a plane
    
    # Define the column space (a plane in 3D)
    # Column vectors of X
    v1 = np.array([1, 0, 0.5])  # First column of X (including intercept)
    v2 = np.array([0, 1, 0.5])  # Second column of X
    
    # Normalize vectors for cleaner visualization
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Create the grid for the plane (column space of X)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    z = np.zeros_like(xx)
    
    # Calculate z values for the plane
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = xx[i, j] * v1 + yy[i, j] * v2
            z[i, j] = point[2]
    
    # Plot the column space (plane)
    ax.plot_surface(xx, yy, z, alpha=0.3, color='blue')
    
    # Create a target point y
    y = np.array([0.5, 0.5, 1.2])
    
    # Calculate the projection of y onto the plane
    # Using the projection formula: y_hat = X(X^TX)^{-1}X^Ty
    # For simplicity, we'll use a direct geometric approach
    
    # Create a basis for the column space
    basis = np.vstack((v1, v2)).T
    
    # Calculate projection matrix
    proj_matrix = basis @ np.linalg.inv(basis.T @ basis) @ basis.T
    
    # Calculate the projected point
    y_hat = proj_matrix @ y
    
    # Plot the original point y
    ax.scatter([y[0]], [y[1]], [y[2]], color='red', s=100, label='y (target)')
    
    # Plot the projected point y_hat
    ax.scatter([y_hat[0]], [y_hat[1]], [y_hat[2]], color='green', s=100, label='ŷ (prediction)')
    
    # Plot the residual vector (y - y_hat)
    ax.quiver(y_hat[0], y_hat[1], y_hat[2], 
              y[0]-y_hat[0], y[1]-y_hat[1], y[2]-y_hat[2], 
              color='purple', arrow_length_ratio=0.1, label='residual')
    
    # Plot basis vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='blue', arrow_length_ratio=0.1)
    
    # Add text annotations
    ax.text(y[0]+0.1, y[1], y[2], "y", color='red', fontsize=12)
    ax.text(y_hat[0]+0.1, y_hat[1], y_hat[2], "ŷ = Xw", color='green', fontsize=12)
    ax.text((y[0]+y_hat[0])/2, (y[1]+y_hat[1])/2, (y[2]+y_hat[2])/2, 
            "residual\n(y - ŷ)", color='purple', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_zlabel('y', fontsize=12)
    ax.set_title('Geometric Interpretation: Linear Regression as Projection', fontsize=14)
    
    # Add legend and annotation text
    ax.legend(loc='upper right')
    
    # Add explanatory text
    plt.figtext(0.1, 0.02, 
                r"Linear regression finds $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ that is the projection of $\mathbf{y}$ " + 
                r"onto the column space of $\mathbf{X}$.", 
                fontsize=12)
    plt.figtext(0.1, 0.06, 
                r"The residual vector $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to the column space of $\mathbf{X}$.", 
                fontsize=12)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/geometric_interpretation.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_geometric_interpretation()
    print("Geometric interpretation visualization generated successfully.") 