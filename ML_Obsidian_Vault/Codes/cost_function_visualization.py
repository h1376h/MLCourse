import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def generate_cost_function_visualization():
    """
    Generate a 3D visualization of the cost function for univariate linear regression.
    """
    # Create sample data
    np.random.seed(42)
    x = np.linspace(-5, 5, 20)
    y = 2 * x + 1 + np.random.normal(0, 2, size=len(x))
    
    # Create a meshgrid of w0 and w1 values
    w0_vals = np.linspace(-1, 3, 50)
    w1_vals = np.linspace(0, 4, 50)
    w0_grid, w1_grid = np.meshgrid(w0_vals, w1_vals)
    
    # Calculate the cost function for each combination of w0 and w1
    J_vals = np.zeros_like(w0_grid)
    for i in range(len(w0_vals)):
        for j in range(len(w1_vals)):
            w0 = w0_vals[i]
            w1 = w1_vals[j]
            predictions = w0 + w1 * x
            errors = y - predictions
            J_vals[j, i] = np.mean(errors**2)  # Using mean squared error
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(w0_grid, w1_grid, J_vals, cmap=cm.coolwarm,
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Mark the minimum point
    # Find the minimum cost
    min_idx = np.unravel_index(np.argmin(J_vals), J_vals.shape)
    w0_min = w0_vals[min_idx[1]]
    w1_min = w1_vals[min_idx[0]]
    J_min = J_vals[min_idx]
    
    # Add a marker at the minimum
    ax.scatter([w0_min], [w1_min], [J_min], color='black', s=100, marker='*')
    ax.text(w0_min, w1_min, J_min + 0.5, "Minimum", color='black', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('w0 (intercept)', fontsize=12)
    ax.set_ylabel('w1 (slope)', fontsize=12)
    ax.set_zlabel('J(w0, w1) - Cost', fontsize=12)
    ax.set_title('Cost Function for Univariate Linear Regression', fontsize=14)
    
    # Add explanatory text
    plt.figtext(0.15, 0.05, 
                r"Cost Function: $J(w_0, w_1) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - w_0 - w_1 x^{(i)})^2$", 
                fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/cost_function_3d.png', dpi=300)
    plt.close()
    
    # Create a contour plot
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(w0_grid, w1_grid, J_vals, 50, cmap='coolwarm')
    fig.colorbar(contour, ax=ax)
    
    # Add contour lines
    contour_lines = ax.contour(w0_grid, w1_grid, J_vals, 10, colors='black', alpha=0.3)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    # Mark the minimum point
    ax.scatter([w0_min], [w1_min], color='black', s=100, marker='*')
    ax.text(w0_min + 0.1, w1_min + 0.1, "Minimum", color='black', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('w0 (intercept)', fontsize=12)
    ax.set_ylabel('w1 (slope)', fontsize=12)
    ax.set_title('Contour Plot of Cost Function J(w0, w1)', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/cost_function_contour.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_cost_function_visualization()
    print("Cost function visualizations generated successfully.") 