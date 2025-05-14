import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_33")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_section_header(title):
    """Print a section header with a title."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80 + "\n")

# Part 1: Derive the partial derivative of the cost function with respect to w_0
def derive_partial_w0():
    print_section_header("Part 1: Partial Derivative with respect to w_0")
    
    # Using sympy for symbolic math
    w0, w1, x, y = sp.symbols('w_0 w_1 x y')
    i = sp.symbols('i', integer=True)
    n = sp.symbols('n', integer=True)
    
    # Define the cost function
    J_element = (y - w0 - w1 * x)**2
    J = sp.Sum(J_element, (i, 1, n))
    
    print("The cost function is:")
    print(f"J(w_0, w_1) = ∑(y^(i) - w_0 - w_1 * x^(i))^2")
    print()
    
    # Take the partial derivative with respect to w0
    dJ_dw0 = sp.diff(J_element, w0)
    
    print("To find the partial derivative with respect to w_0, we apply the chain rule:")
    print("∂/∂w_0 [(y - w_0 - w_1*x)^2]")
    print("= 2(y - w_0 - w_1*x) * ∂/∂w_0 [y - w_0 - w_1*x]")
    print("= 2(y - w_0 - w_1*x) * (-1)")
    print("= -2(y - w_0 - w_1*x)")
    print()
    
    print("Therefore, the partial derivative of the cost function with respect to w_0 is:")
    print("∂J/∂w_0 = ∑ -2(y^(i) - w_0 - w_1*x^(i))")
    print("        = -2∑(y^(i) - w_0 - w_1*x^(i))")
    print()
    
    # Create visualization for the partial derivative
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple linear regression example data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_data = 2 + 3 * x_data + np.random.normal(0, 2, 20)
    
    # Plot data points
    ax.scatter(x_data, y_data, color='blue', label='Data points')
    
    # Plot regression line and residuals for two different w0 values
    w1_fixed = 3  # Fixed slope
    w0_values = [1, 3]  # Two different intercepts
    colors = ['red', 'green']
    
    for idx, w0_val in enumerate(w0_values):
        y_pred = w0_val + w1_fixed * x_data
        residuals = y_data - y_pred
        
        # Plot regression line
        ax.plot(x_data, y_pred, color=colors[idx], 
                label=f'w_0={w0_val}, w_1={w1_fixed}')
        
        # Plot residuals
        for i, (x_i, y_i, res_i) in enumerate(zip(x_data, y_data, residuals)):
            alpha = 0.3 if i % 3 != 0 else 0.8  # Make only some residuals more visible
            ax.plot([x_i, x_i], [y_i, y_pred[i]], '--', color=colors[idx], alpha=alpha)
    
    # Compute total SSE for each line
    sse_1 = np.sum((y_data - (w0_values[0] + w1_fixed * x_data))**2)
    sse_2 = np.sum((y_data - (w0_values[1] + w1_fixed * x_data))**2)
    
    # Add annotations
    ax.annotate(f'SSE = {sse_1:.2f}', xy=(8, w0_values[0] + w1_fixed * 8 + 0.5), 
                color=colors[0], fontsize=12)
    ax.annotate(f'SSE = {sse_2:.2f}', xy=(8, w0_values[1] + w1_fixed * 8 + 0.5), 
                color=colors[1], fontsize=12)
    
    ax.set_title('Effect of Changing w_0 on Residuals and SSE', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, "part1_partial_w0.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {file_path}")
    
    return dJ_dw0

# Part 2: Derive the partial derivative of the cost function with respect to w_1
def derive_partial_w1():
    print_section_header("Part 2: Partial Derivative with respect to w_1")
    
    # Using sympy for symbolic math
    w0, w1, x, y = sp.symbols('w_0 w_1 x y')
    i = sp.symbols('i', integer=True)
    n = sp.symbols('n', integer=True)
    
    # Define the cost function
    J_element = (y - w0 - w1 * x)**2
    J = sp.Sum(J_element, (i, 1, n))
    
    # Take the partial derivative with respect to w1
    dJ_dw1 = sp.diff(J_element, w1)
    
    print("To find the partial derivative with respect to w_1, we apply the chain rule:")
    print("∂/∂w_1 [(y - w_0 - w_1*x)^2]")
    print("= 2(y - w_0 - w_1*x) * ∂/∂w_1 [y - w_0 - w_1*x]")
    print("= 2(y - w_0 - w_1*x) * (-x)")
    print("= -2x(y - w_0 - w_1*x)")
    print()
    
    print("Therefore, the partial derivative of the cost function with respect to w_1 is:")
    print("∂J/∂w_1 = ∑ -2x^(i)(y^(i) - w_0 - w_1*x^(i))")
    print("        = -2∑ x^(i)(y^(i) - w_0 - w_1*x^(i))")
    print()
    
    # Create visualization for the partial derivative
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple linear regression example data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_data = 2 + 3 * x_data + np.random.normal(0, 2, 20)
    
    # Plot data points
    ax.scatter(x_data, y_data, color='blue', label='Data points')
    
    # Plot regression line and residuals for two different w1 values
    w0_fixed = 2  # Fixed intercept
    w1_values = [2, 4]  # Two different slopes
    colors = ['red', 'green']
    
    for idx, w1_val in enumerate(w1_values):
        y_pred = w0_fixed + w1_val * x_data
        residuals = y_data - y_pred
        
        # Plot regression line
        ax.plot(x_data, y_pred, color=colors[idx], 
                label=f'w_0={w0_fixed}, w_1={w1_val}')
        
        # Plot residuals
        for i, (x_i, y_i, res_i) in enumerate(zip(x_data, y_data, residuals)):
            alpha = 0.3 if i % 3 != 0 else 0.8  # Make only some residuals more visible
            ax.plot([x_i, x_i], [y_i, y_pred[i]], '--', color=colors[idx], alpha=alpha)
    
    # Compute total SSE for each line
    sse_1 = np.sum((y_data - (w0_fixed + w1_values[0] * x_data))**2)
    sse_2 = np.sum((y_data - (w0_fixed + w1_values[1] * x_data))**2)
    
    # Add annotations
    ax.annotate(f'SSE = {sse_1:.2f}', xy=(7, w0_fixed + w1_values[0] * 7 + 1), 
                color=colors[0], fontsize=12)
    ax.annotate(f'SSE = {sse_2:.2f}', xy=(7, w0_fixed + w1_values[1] * 7 + 1), 
                color=colors[1], fontsize=12)
    
    ax.set_title('Effect of Changing w_1 on Residuals and SSE', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, "part2_partial_w1.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {file_path}")
    
    return dJ_dw1

# Part 3: Derive the normal equations
def derive_normal_equations():
    print_section_header("Part 3: Deriving the Normal Equations")
    
    print("To derive the normal equations, we set the partial derivatives equal to zero and solve:")
    print()
    
    print("Setting the derivative with respect to w_0 equal to zero:")
    print("∂J/∂w_0 = -2∑(y^(i) - w_0 - w_1*x^(i)) = 0")
    print()
    
    print("Dividing by -2:")
    print("∑(y^(i) - w_0 - w_1*x^(i)) = 0")
    print()
    
    print("Expanding the summation:")
    print("∑y^(i) - ∑w_0 - ∑w_1*x^(i) = 0")
    print()
    
    print("Since w_0 and w_1 are constants with respect to the summation:")
    print("∑y^(i) - n*w_0 - w_1*∑x^(i) = 0")
    print()
    
    print("Solving for w_0:")
    print("n*w_0 = ∑y^(i) - w_1*∑x^(i)")
    print("w_0 = (∑y^(i) - w_1*∑x^(i))/n")
    print("w_0 = ȳ - w_1*x̄")
    print()
    
    print("Setting the derivative with respect to w_1 equal to zero:")
    print("∂J/∂w_1 = -2∑x^(i)(y^(i) - w_0 - w_1*x^(i)) = 0")
    print()
    
    print("Dividing by -2:")
    print("∑x^(i)(y^(i) - w_0 - w_1*x^(i)) = 0")
    print()
    
    print("Expanding the summation:")
    print("∑x^(i)y^(i) - w_0∑x^(i) - w_1∑(x^(i))^2 = 0")
    print()
    
    print("Substituting our expression for w_0:")
    print("∑x^(i)y^(i) - (ȳ - w_1*x̄)∑x^(i) - w_1∑(x^(i))^2 = 0")
    print("∑x^(i)y^(i) - ȳ∑x^(i) + w_1*x̄∑x^(i) - w_1∑(x^(i))^2 = 0")
    print()
    
    print("Noting that ∑x^(i) = n*x̄, we get:")
    print("∑x^(i)y^(i) - ȳ*n*x̄ + w_1*x̄*n*x̄ - w_1∑(x^(i))^2 = 0")
    print("∑x^(i)y^(i) - n*x̄*ȳ + w_1*n*(x̄)^2 - w_1∑(x^(i))^2 = 0")
    print()
    
    print("Isolating terms with w_1:")
    print("∑x^(i)y^(i) - n*x̄*ȳ = w_1∑(x^(i))^2 - w_1*n*(x̄)^2")
    print("∑x^(i)y^(i) - n*x̄*ȳ = w_1(∑(x^(i))^2 - n*(x̄)^2)")
    print()
    
    print("Solving for w_1:")
    print("w_1 = (∑x^(i)y^(i) - n*x̄*ȳ)/(∑(x^(i))^2 - n*(x̄)^2)")
    print()
    
    print("We can simplify the numerator and denominator:")
    print("∑x^(i)y^(i) - n*x̄*ȳ = ∑x^(i)y^(i) - ∑x^(i)*∑y^(i)/n = cov(x,y)*n")
    print("∑(x^(i))^2 - n*(x̄)^2 = ∑(x^(i))^2 - (∑x^(i))^2/n = var(x)*n")
    print()
    
    print("Therefore, the simplified form for w_1 is:")
    print("w_1 = cov(x,y)/var(x)")
    print()
    
    print("And the formula for w_0 is:")
    print("w_0 = ȳ - w_1*x̄")
    print()
    
    print("These two equations are known as the normal equations for simple linear regression.")
    
    # Create visualization for the normal equations solution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple linear regression example data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_data = 2 + 3 * x_data + np.random.normal(0, 2, 20)
    
    # Plot data points
    ax.scatter(x_data, y_data, color='blue', label='Data points')
    
    # Calculate the optimal parameters using the normal equations
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    
    # Calculate the slope (w_1)
    numerator = np.sum((x_data - x_mean) * (y_data - y_mean))
    denominator = np.sum((x_data - x_mean) ** 2)
    w1_opt = numerator / denominator
    
    # Calculate the intercept (w_0)
    w0_opt = y_mean - w1_opt * x_mean
    
    # Plot the optimal regression line
    x_range = np.linspace(0, 10, 100)
    y_pred_opt = w0_opt + w1_opt * x_range
    ax.plot(x_range, y_pred_opt, color='red', linewidth=2, 
            label=f'Optimal: w_0={w0_opt:.2f}, w_1={w1_opt:.2f}')
    
    # Plot some non-optimal lines
    w0_values = [w0_opt - 1, w0_opt + 1]
    w1_values = [w1_opt - 0.5, w1_opt + 0.5]
    
    colors = ['green', 'purple', 'orange', 'brown']
    sse_values = []
    
    for i, w0 in enumerate(w0_values):
        for j, w1 in enumerate(w1_values):
            y_pred = w0 + w1 * x_range
            sse = np.sum((y_data - (w0 + w1 * x_data))**2)
            sse_values.append(sse)
            
            color_idx = i*2 + j
            ax.plot(x_range, y_pred, color=colors[color_idx], linestyle='--', alpha=0.7,
                    label=f'w_0={w0:.2f}, w_1={w1:.2f}, SSE={sse:.2f}')
    
    # Calculate and plot the mean point
    ax.scatter([x_mean], [y_mean], color='black', s=100, marker='X', 
               label=f'Mean point (x̄={x_mean:.2f}, ȳ={y_mean:.2f})')
    
    # Calculate SSE for optimal line
    sse_opt = np.sum((y_data - (w0_opt + w1_opt * x_data))**2)
    ax.annotate(f'Optimal SSE = {sse_opt:.2f}', xy=(8, w0_opt + w1_opt * 8 - 2),
                color='red', fontsize=12)
    
    ax.set_title('Optimal vs. Non-optimal Regression Lines', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, "part3_normal_equations.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {file_path}")

# Part 4: Show that the regression line passes through the point (x̄, ȳ)
def show_regression_through_mean():
    print_section_header("Part 4: Showing the Regression Line Passes Through (x̄, ȳ)")
    
    print("We want to show that the regression line passes through the point (x̄, ȳ).")
    print("The equation of the regression line is: y = w_0 + w_1*x")
    print()
    
    print("We know that w_0 = ȳ - w_1*x̄")
    print()
    
    print("Let's substitute the point (x̄, ȳ) into the regression line equation:")
    print("y = w_0 + w_1*x")
    print("ȳ = w_0 + w_1*x̄")
    print("ȳ = (ȳ - w_1*x̄) + w_1*x̄")
    print("ȳ = ȳ")
    print()
    
    print("This proves that the regression line always passes through the point (x̄, ȳ).")
    print("This is a direct consequence of the normal equation for w_0, which ensures this property.")
    print()
    
    # Create visualization showing the regression line through the mean point
    # Generate multiple datasets to show this property holds in general
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate multiple datasets with different parameters
    np.random.seed(42)
    num_datasets = 4
    
    # Generate some example data
    n_points = 20
    x_ranges = [np.linspace(0, 10, n_points), np.linspace(-5, 5, n_points), 
                np.linspace(2, 8, n_points), np.linspace(-10, 0, n_points)]
    slopes = [3, -2, 1, 0.5]
    intercepts = [2, 5, -1, -3]
    noise_scales = [2, 1.5, 1, 3]
    
    colors = ['blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i in range(num_datasets):
        x_data = x_ranges[i]
        y_data = intercepts[i] + slopes[i] * x_data + np.random.normal(0, noise_scales[i], n_points)
        
        # Calculate the mean point
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)
        
        # Calculate optimal parameters
        numerator = np.sum((x_data - x_mean) * (y_data - y_mean))
        denominator = np.sum((x_data - x_mean) ** 2)
        w1_opt = numerator / denominator
        w0_opt = y_mean - w1_opt * x_mean
        
        # Plot data points
        ax.scatter(x_data, y_data, color=colors[i], label=f'Dataset {i+1}', 
                  marker=markers[i], alpha=0.7)
        
        # Plot mean point
        ax.scatter([x_mean], [y_mean], color=colors[i], s=150, marker='X', 
                  edgecolor='black', linewidth=1.5)
        
        # Plot regression line
        x_line = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
        y_line = w0_opt + w1_opt * x_line
        ax.plot(x_line, y_line, color=colors[i], linewidth=2)
        
        # Add annotation for the mean point
        ax.annotate(f'(x̄={x_mean:.2f}, ȳ={y_mean:.2f})', 
                   xy=(x_mean, y_mean), xytext=(x_mean + 0.5, y_mean + 0.5),
                   arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    # Add explanatory text
    ax.text(0.02, 0.98, "All regression lines pass through the mean point (x̄, ȳ) of their respective datasets.\nThis is guaranteed by the normal equation: w_0 = ȳ - w_1*x̄", 
            transform=ax.transAxes, fontsize=12, va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Regression Lines Always Pass Through the Mean Point (x̄, ȳ)', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, "part4_regression_through_mean.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {file_path}")
    
    # Visual proof with 3D SSE surface
    print("\nCreating 3D visualization of SSE surface to illustrate the normal equations solution...\n")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[2, 1])
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_contour = fig.add_subplot(gs[0, 1])
    ax_w0 = fig.add_subplot(gs[1, 0])
    ax_w1 = fig.add_subplot(gs[1, 1])
    
    # Use simple dataset
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_data = 2 + 3 * x_data + np.random.normal(0, 2, 20)
    
    # Calculate optimal parameters
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    numerator = np.sum((x_data - x_mean) * (y_data - y_mean))
    denominator = np.sum((x_data - x_mean) ** 2)
    w1_opt = numerator / denominator
    w0_opt = y_mean - w1_opt * x_mean
    
    # Define the SSE function
    def sse_function(w0, w1):
        return np.sum((y_data - (w0 + w1 * x_data))**2)
    
    # Create a grid of w0 and w1 values
    w0_range = np.linspace(w0_opt - 3, w0_opt + 3, 50)
    w1_range = np.linspace(w1_opt - 2, w1_opt + 2, 50)
    w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
    
    # Calculate SSE for each combination of w0 and w1
    sse_grid = np.zeros_like(w0_grid)
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            sse_grid[j, i] = sse_function(w0_range[i], w1_range[j])
    
    # Plot the 3D surface
    surf = ax_3d.plot_surface(w0_grid, w1_grid, sse_grid, cmap='viridis', alpha=0.8)
    ax_3d.set_xlabel('w_0 (Intercept)', fontsize=10)
    ax_3d.set_ylabel('w_1 (Slope)', fontsize=10)
    ax_3d.set_zlabel('SSE', fontsize=10)
    ax_3d.set_title('SSE Surface', fontsize=12)
    
    # Mark the optimal point
    sse_opt = sse_function(w0_opt, w1_opt)
    ax_3d.scatter([w0_opt], [w1_opt], [sse_opt], color='red', s=50, marker='o')
    
    # Plot the path of w0 when setting partial derivative to zero
    w0_curve = np.array([y_mean - w1 * x_mean for w1 in w1_range])
    sse_curve = np.array([sse_function(w0, w1) for w0, w1 in zip(w0_curve, w1_range)])
    ax_3d.plot(w0_curve, w1_range, sse_curve, 'r-', linewidth=2, 
               label='∂J/∂w_0 = 0 path')
    
    # Contour plot
    contour = ax_contour.contour(w0_grid, w1_grid, sse_grid, 20, cmap='viridis')
    ax_contour.set_xlabel('w_0', fontsize=10)
    ax_contour.set_ylabel('w_1', fontsize=10)
    ax_contour.set_title('SSE Contour Plot', fontsize=12)
    fig.colorbar(contour, ax=ax_contour)
    
    # Mark optimal point on contour
    ax_contour.scatter([w0_opt], [w1_opt], color='red', s=50, marker='o')
    
    # Plot the path where partial w.r.t w0 is zero
    ax_contour.plot(w0_curve, w1_range, 'r-', linewidth=2, label='∂J/∂w_0 = 0')
    
    # Mark the mean point
    w1_at_mean = np.linspace(w1_opt - 2, w1_opt + 2, 50)
    w0_at_mean = np.array([y_mean - w1 * x_mean for w1 in w1_at_mean])
    ax_contour.plot(w0_at_mean, w1_at_mean, 'b--', linewidth=1.5, 
                   label='w_0 = ȳ - w_1*x̄')
    
    ax_contour.legend(fontsize=8)
    
    # Plot SSE vs w0 (fixing w1 at optimal)
    w0_line = np.linspace(w0_opt - 3, w0_opt + 3, 100)
    sse_w0 = np.array([sse_function(w0, w1_opt) for w0 in w0_line])
    ax_w0.plot(w0_line, sse_w0, 'b-', linewidth=2)
    ax_w0.axvline(x=w0_opt, color='red', linestyle='--')
    ax_w0.axvline(x=y_mean - w1_opt * x_mean, color='green', linestyle=':')
    ax_w0.set_xlabel('w_0 (with w_1 fixed at optimal)', fontsize=10)
    ax_w0.set_ylabel('SSE', fontsize=10)
    ax_w0.set_title('SSE vs. w_0', fontsize=12)
    
    # Plot SSE vs w1 (fixing w0 at optimal)
    w1_line = np.linspace(w1_opt - 2, w1_opt + 2, 100)
    sse_w1 = np.array([sse_function(w0_opt, w1) for w1 in w1_line])
    ax_w1.plot(w1_line, sse_w1, 'b-', linewidth=2)
    ax_w1.axvline(x=w1_opt, color='red', linestyle='--')
    ax_w1.set_xlabel('w_1 (with w_0 fixed at optimal)', fontsize=10)
    ax_w1.set_ylabel('SSE', fontsize=10)
    ax_w1.set_title('SSE vs. w_1', fontsize=12)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, "part4_sse_surface.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {file_path}")

# Main execution
if __name__ == "__main__":
    # Part 1: Derive the partial derivative with respect to w_0
    dJ_dw0 = derive_partial_w0()
    
    # Part 2: Derive the partial derivative with respect to w_1
    dJ_dw1 = derive_partial_w1()
    
    # Part 3: Derive the normal equations
    derive_normal_equations()
    
    # Part 4: Show that the regression line passes through the point (x̄, ȳ)
    show_regression_through_mean()
    
    print_section_header("Final Results Summary")
    
    print("1. Partial derivative with respect to w_0:")
    print("   ∂J/∂w_0 = -2∑(y^(i) - w_0 - w_1*x^(i))")
    print()
    
    print("2. Partial derivative with respect to w_1:")
    print("   ∂J/∂w_1 = -2∑x^(i)(y^(i) - w_0 - w_1*x^(i))")
    print()
    
    print("3. Normal equations:")
    print("   w_0 = ȳ - w_1*x̄")
    print("   w_1 = cov(x,y)/var(x) = ∑(x^(i) - x̄)(y^(i) - ȳ)/∑(x^(i) - x̄)^2")
    print()
    
    print("4. The normal equation w_0 = ȳ - w_1*x̄ guarantees that the regression line")
    print("   always passes through the mean point (x̄, ȳ) of the data.")
    print()
    
    print("All figures have been saved to:", save_dir) 