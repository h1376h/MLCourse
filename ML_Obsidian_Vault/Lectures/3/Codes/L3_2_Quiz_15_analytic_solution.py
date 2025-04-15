import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data
ages = np.array([43, 21, 25, 42, 57, 59])
glucose_levels = np.array([99, 65, 79, 75, 87, 81])

# Function to calculate cost function J(w)
def cost_function(w0, w1, x, y):
    """Calculate the sum of squared residuals (cost function)."""
    n = len(x)
    return np.sum([(y[i] - w0 - w1 * x[i])**2 for i in range(n)])

# Calculate derivatives of J(w) with respect to w0 and w1 analytically
def calc_derivatives(w0, w1, x, y):
    """Calculate the partial derivatives of J(w) with respect to w0 and w1."""
    n = len(x)
    dj_dw0 = -2 * np.sum([y[i] - w0 - w1 * x[i] for i in range(n)])
    dj_dw1 = -2 * np.sum([(y[i] - w0 - w1 * x[i]) * x[i] for i in range(n)])
    return dj_dw0, dj_dw1

# Analytical solution using normal equations (derived from setting derivatives to zero)
def analytic_solution(x, y):
    """
    Solve for w0 and w1 analytically by setting derivatives to zero
    and solving the resulting system of equations.
    """
    n = len(x)
    
    # From setting derivative with respect to w0 to zero:
    # sum(y[i] - w0 - w1*x[i]) = 0
    # n*w0 + w1*sum(x) = sum(y)
    
    # From setting derivative with respect to w1 to zero:
    # sum((y[i] - w0 - w1*x[i])*x[i]) = 0
    # w0*sum(x) + w1*sum(x[i]^2) = sum(y[i]*x[i])
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    # Solve the system of equations for w0 and w1
    # [ n      sum_x    ] [ w0 ] = [ sum_y    ]
    # [ sum_x  sum_x_sq ] [ w1 ] = [ sum_xy   ]
    
    A = np.array([[n, sum_x], [sum_x, sum_x_squared]])
    b = np.array([sum_y, sum_xy])
    
    w0, w1 = np.linalg.solve(A, b)
    
    print("Analytical Solution Process:")
    print(f"Step 1: Set up system of equations from derivatives:")
    print(f"  n*w0 + sum(x)*w1 = sum(y)")
    print(f"  sum(x)*w0 + sum(x^2)*w1 = sum(x*y)")
    print()
    print(f"Step 2: Substitute the values:")
    print(f"  {n}*w0 + {sum_x:.2f}*w1 = {sum_y:.2f}")
    print(f"  {sum_x:.2f}*w0 + {sum_x_squared:.2f}*w1 = {sum_xy:.2f}")
    print()
    print(f"Step 3: Solve the system of equations:")
    print(f"  w0 = {w0:.4f}")
    print(f"  w1 = {w1:.4f}")
    print()
    
    return w0, w1

# Execute the analytical solution
w0, w1 = analytic_solution(ages, glucose_levels)

# Print the regression equation
print(f"Linear Regression Equation: Glucose Level = {w0:.4f} + {w1:.4f} · Age")
print()

# Calculate R-squared
def calculate_r_squared(x, y, w0, w1):
    """Calculate coefficient of determination (R²)."""
    y_pred = w0 + w1 * x
    y_mean = np.mean(y)
    
    # Total sum of squares
    ss_total = np.sum((y - y_mean)**2)
    
    # Residual sum of squares
    ss_residual = np.sum((y - y_pred)**2)
    
    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total)
    
    print(f"R-squared calculation:")
    print(f"  Total sum of squares (TSS): {ss_total:.4f}")
    print(f"  Residual sum of squares (RSS): {ss_residual:.4f}")
    print(f"  R² = 1 - RSS/TSS = {r_squared:.4f}")
    print(f"  This means {r_squared*100:.2f}% of the variation in glucose levels can be explained by age.")
    print()
    
    return r_squared, y_pred

# Calculate R-squared and predicted values
r_squared, y_pred = calculate_r_squared(ages, glucose_levels, w0, w1)

# Predict glucose level for a 55-year-old
pred_age = 55
pred_glucose = w0 + w1 * pred_age
print(f"Predicted glucose level for a 55-year-old: {pred_glucose:.2f}")
print()

# Visualize the cost function in 3D
def visualize_cost_function(x, y, w0, w1, save_dir=None):
    """Visualize the cost function J(w) in 3D."""
    # Create a grid of w0 and w1 values
    w0_values = np.linspace(w0 - 20, w0 + 20, 100)
    w1_values = np.linspace(w1 - 0.5, w1 + 0.5, 100)
    
    # Create a mesh grid
    W0, W1 = np.meshgrid(w0_values, w1_values)
    
    # Calculate cost function values for each point in the grid
    J = np.zeros(W0.shape)
    for i in range(len(w0_values)):
        for j in range(len(w1_values)):
            J[j, i] = cost_function(W0[j, i], W1[j, i], x, y)
    
    # Create 3D surface plot of cost function
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surface = ax.plot_surface(W0, W1, J, cmap='viridis', alpha=0.8)
    
    # Mark the minimum point
    min_cost = cost_function(w0, w1, x, y)
    ax.scatter([w0], [w1], [min_cost], color='r', s=100, label='Minimum')
    
    # Add labels
    ax.set_xlabel('w0 (Intercept)')
    ax.set_ylabel('w1 (Slope)')
    ax.set_zlabel('J(w) (Cost)')
    ax.set_title('Cost Function J(w) Surface')
    
    # Add a colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    ax.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "cost_function_3d.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create contour plot (2D view from above)
    plt.figure(figsize=(10, 8))
    contour = plt.contour(W0, W1, J, 30, cmap='viridis')
    plt.colorbar(contour)
    
    # Mark the minimum point
    plt.scatter(w0, w1, color='r', s=100, label='Minimum')
    
    # Add labels
    plt.xlabel('w0 (Intercept)')
    plt.ylabel('w1 (Slope)')
    plt.title('Cost Function J(w) Contour Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "cost_function_contour.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create visualization of data with regression line
def visualize_regression(x, y, w0, w1, pred_x, pred_y, r_squared, save_dir=None):
    """Visualize the regression line and data points."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of original data
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 5, max(x) + 5, 100)
    y_line = w0 + w1 * x_line
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {w0:.2f} + {w1:.2f}x')
    
    # Add prediction point
    plt.scatter(pred_x, pred_y, color='green', s=100, 
                label=f'Prediction ({pred_x}, {pred_y:.2f})')
    plt.plot([pred_x, pred_x], [0, pred_y], 'g--', alpha=0.5)
    
    # Add title, labels, and legend
    plt.title(f'Glucose Level vs Age (R² = {r_squared:.4f})', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Glucose Level', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create residuals plot
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    residuals = y - (w0 + w1 * x)
    
    # Scatter plot of residuals
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Add vertical lines to represent residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    # Add title, labels
    plt.title('Residuals Plot', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "residuals_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

# Execute visualizations
visualize_cost_function(ages, glucose_levels, w0, w1, save_dir)
visualize_regression(ages, glucose_levels, w0, w1, pred_age, pred_glucose, r_squared, save_dir)

print(f"Analytical Solution Results Summary:")
print(f"1. Linear regression equation: Glucose Level = {w0:.4f} + {w1:.4f} * Age")
print(f"2. Predicted glucose level for a 55-year-old: {pred_glucose:.2f}")
print(f"3. Coefficient of determination (R²): {r_squared:.4f}")
print(f"   {r_squared*100:.2f}% of the variation in glucose levels can be explained by age.")
print()
print(f"Visualizations saved to: {save_dir}") 