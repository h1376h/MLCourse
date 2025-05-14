import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def generate_gradient_descent_visualization():
    """
    Generate a visualization of gradient descent optimization for linear regression.
    """
    # Create sample data
    np.random.seed(42)
    x = np.linspace(-5, 5, 20)
    y = 2 * x + 1 + np.random.normal(0, 2, size=len(x))
    
    # Create a meshgrid of w0 and w1 values
    w0_vals = np.linspace(-4, 6, 100)
    w1_vals = np.linspace(-1, 5, 100)
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
    
    # Helper function for gradient descent
    def compute_gradient(w0, w1, X, y):
        m = len(y)
        predictions = w0 + w1 * X
        errors = predictions - y
        
        # Gradients
        dw0 = (1/m) * np.sum(errors)
        dw1 = (1/m) * np.sum(errors * X)
        
        return dw0, dw1
    
    # Initial parameters
    w0_init = 5.0
    w1_init = -0.5
    
    # Learning rate
    alpha = 0.05
    
    # Run gradient descent for several iterations
    iterations = 20
    w0_history = [w0_init]
    w1_history = [w1_init]
    cost_history = []
    
    w0 = w0_init
    w1 = w1_init
    
    for _ in range(iterations):
        predictions = w0 + w1 * x
        errors = y - predictions
        cost = np.mean(errors**2)
        cost_history.append(cost)
        
        # Compute gradients
        dw0, dw1 = compute_gradient(w0, w1, x, y)
        
        # Update parameters
        w0 = w0 - alpha * dw0
        w1 = w1 - alpha * dw1
        
        # Store parameters
        w0_history.append(w0)
        w1_history.append(w1)
    
    # Create main figure with contour plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot contour with gradient descent path
    contour = ax1.contourf(w0_grid, w1_grid, J_vals, 50, cmap='coolwarm')
    contour_lines = ax1.contour(w0_grid, w1_grid, J_vals, 20, colors='black', alpha=0.3)
    fig.colorbar(contour, ax=ax1)
    
    # Plot gradient descent path
    ax1.plot(w0_history, w1_history, 'o-', color='lime', markersize=6, linewidth=2)
    ax1.plot(w0_history[0], w1_history[0], 'o', color='lime', markersize=10, label='Initial Parameters')
    ax1.plot(w0_history[-1], w1_history[-1], '*', color='red', markersize=10, label='Final Parameters')
    
    # Find the optimal parameters (minimum cost)
    min_idx = np.unravel_index(np.argmin(J_vals), J_vals.shape)
    w0_min = w0_vals[min_idx[1]]
    w1_min = w1_vals[min_idx[0]]
    ax1.plot(w0_min, w1_min, 'x', color='black', markersize=10, label='True Minimum')
    
    ax1.set_xlabel('w0 (intercept)', fontsize=12)
    ax1.set_ylabel('w1 (slope)', fontsize=12)
    ax1.set_title('Gradient Descent Path in Parameter Space', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cost history
    ax2.plot(range(iterations), cost_history, 'b-o', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost J(w)', fontsize=12)
    ax2.set_title('Cost vs. Iteration', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add explanatory text
    plt.figtext(0.1, 0.01, 
                r"Gradient Descent: $w_j := w_j - \alpha \frac{\partial J(w)}{\partial w_j}$ " + 
                r"where $\alpha$ is the learning rate.",
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/gradient_descent.png', dpi=300)
    
    # Create a separate figure to show the data fitting process
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.scatter(x, y, color='blue', label='Training data')
    
    # Plot initial line
    x_line = np.linspace(-6, 6, 100)
    plt.plot(x_line, w0_init + w1_init * x_line, 'r--', 
             linewidth=2, label=f'Initial: y = {w0_init:.2f} + {w1_init:.2f}x')
    
    # Plot final line
    plt.plot(x_line, w0_history[-1] + w1_history[-1] * x_line, 'g-', 
             linewidth=2, label=f'Final: y = {w0_history[-1]:.2f} + {w1_history[-1]:.2f}x')
    
    # Plot optimal line
    plt.plot(x_line, w0_min + w1_min * x_line, 'k:', 
             linewidth=2, label=f'Optimal: y = {w0_min:.2f} + {w1_min:.2f}x')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Data Fitting with Gradient Descent', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/gradient_descent_fitting.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_gradient_descent_visualization()
    print("Gradient descent visualization generated successfully.") 