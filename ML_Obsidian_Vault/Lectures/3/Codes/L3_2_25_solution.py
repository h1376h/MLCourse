import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set nice style for the plots
plt.style.use('default')

def explain_solutions():
    """Explain the solution steps using direct mathematical notation."""
    print("Step 1: Derive the partial derivatives of the cost function")
    print("=========================================================")
    
    print("The cost function for linear regression is:")
    print("J(w_0, w_1) = (1/2n)∑_{i=1}^{n}(w_0 + w_1x^(i) - y^(i))^2")
    print()
    
    # Task 1: Partial derivative with respect to w0
    print("Task 1: Partial derivative with respect to w0")
    print("--------------------------------------------")
    print("For a single data point (x, y), the derivative of (w_0 + w_1x - y)^2 with respect to w_0 is:")
    print("∂/∂w_0 (w_0 + w_1x - y)^2 = 2(w_0 + w_1x - y) × 1")
    print("                           = 2(w_0 + w_1x - y)")
    print()
    
    print("For the entire cost function, using the linearity of differentiation and summation:")
    print("∂J/∂w_0 = (1/n)∑_{i=1}^{n}(w_0 + w_1x^(i) - y^(i))")
    print()
    
    print("We can simplify this to:")
    print("∂J/∂w_0 = (1/n)[nw_0 + w_1∑_{i=1}^{n}x^(i) - ∑_{i=1}^{n}y^(i)]")
    print("        = w_0 + w_1(1/n)∑_{i=1}^{n}x^(i) - (1/n)∑_{i=1}^{n}y^(i)")
    print("        = w_0 + w_1x̄ - ȳ")
    print()
    
    # Task 2: Partial derivative with respect to w1
    print("Task 2: Partial derivative with respect to w1")
    print("--------------------------------------------")
    print("For a single data point (x, y), the derivative of (w_0 + w_1x - y)^2 with respect to w_1 is:")
    print("∂/∂w_1 (w_0 + w_1x - y)^2 = 2(w_0 + w_1x - y) × x")
    print("                           = 2(w_0 + w_1x - y)x")
    print()
    
    print("For the entire cost function:")
    print("∂J/∂w_1 = (1/n)∑_{i=1}^{n}(w_0 + w_1x^(i) - y^(i))x^(i)")
    print()
    
    print("Expanding this expression:")
    print("∂J/∂w_1 = (1/n)[w_0∑_{i=1}^{n}x^(i) + w_1∑_{i=1}^{n}(x^(i))^2 - ∑_{i=1}^{n}y^(i)x^(i)]")
    print()
    
    print("Using mean notation for clarity:")
    print("Let's define:")
    print("- x̄ = (1/n)∑x^(i) (mean of x values)")
    print("- ȳ = (1/n)∑y^(i) (mean of y values)")
    print("- SS_xx = ∑(x^(i))² (sum of squared x values)")
    print("- SS_xy = ∑x^(i)y^(i) (sum of products of x and y)")
    print()
    
    print("Then we can rewrite as:")
    print("∂J/∂w_1 = w_0x̄ + w_1(1/n)SS_xx - (1/n)SS_xy")
    print()
    
    # Task 3: Set derivatives to zero and solve
    print("Task 3: Set partial derivatives to zero and solve for w0 and w1")
    print("------------------------------------------------------------")
    print("Setting ∂J/∂w_0 = 0:")
    print("w_0 + w_1x̄ - ȳ = 0")
    print("w_0 = ȳ - w_1x̄  ... (1)")
    print()
    
    print("Setting ∂J/∂w_1 = 0:")
    print("w_0x̄ + w_1(1/n)SS_xx - (1/n)SS_xy = 0")
    print()
    
    print("Substituting equation (1) for w_0:")
    print("(ȳ - w_1x̄)x̄ + w_1(1/n)SS_xx - (1/n)SS_xy = 0")
    print("ȳx̄ - w_1x̄² + w_1(1/n)SS_xx - (1/n)SS_xy = 0")
    print()
    
    print("Solving for w_1:")
    print("w_1[(1/n)SS_xx - x̄²] = (1/n)SS_xy - ȳx̄")
    print()
    
    print("Note that (1/n)SS_xx - x̄² = (1/n)∑(x^(i))² - (1/n)²(∑x^(i))² = (1/n)∑(x^(i) - x̄)² = Var(x)")
    print("And (1/n)SS_xy - ȳx̄ = (1/n)∑x^(i)y^(i) - (1/n)²(∑x^(i))(∑y^(i)) = (1/n)∑(x^(i) - x̄)(y^(i) - ȳ) = Cov(x, y)")
    print()
    
    print("Therefore:")
    print("w_1 = Cov(x, y) / Var(x)")
    print("w_1 = ∑(x^(i) - x̄)(y^(i) - ȳ) / ∑(x^(i) - x̄)²")
    print()
    
    print("And substituting back into equation (1):")
    print("w_0 = ȳ - w_1x̄")
    print()
    
    # Task 4: Show these are the standard formulas
    print("Task 4: Confirm these are the standard formulas for simple linear regression")
    print("--------------------------------------------------------------------")
    print("The standard formulas for simple linear regression are:")
    print("Slope (β₁): β₁ = ∑(x_i - x̄)(y_i - ȳ) / ∑(x_i - x̄)²")
    print("Intercept (β₀): β₀ = ȳ - β₁x̄")
    print()
    
    print("Our derived formulas are identical:")
    print("w_1 = ∑(x^(i) - x̄)(y^(i) - ȳ) / ∑(x^(i) - x̄)²  (which is the formula for the slope)")
    print("w_0 = ȳ - w_1x̄  (which is the formula for the intercept)")
    print()
    
    print("Therefore, by minimizing the squared error cost function through differentiation,")
    print("we arrive at the standard formulas for simple linear regression parameters:")
    print("w_0 = β₀ (intercept)")
    print("w_1 = β₁ (slope)")
    
    return {
        'w0_formula': "ȳ - w_1x̄",
        'w1_formula': "∑(x^(i) - x̄)(y^(i) - ȳ) / ∑(x^(i) - x̄)²"
    }

def create_visualizations(formulas, save_dir=None):
    """Create visualizations to illustrate the concepts."""
    saved_files = []
    
    # Plot 1: Visualization of the cost function and critical points
    print("\nCreating visualizations to illustrate the concepts...")
    
    # Generate sample data
    np.random.seed(42)
    n = 20
    x = np.linspace(0, 10, n)
    true_w0, true_w1 = 2, 3
    y = true_w0 + true_w1 * x + np.random.normal(0, 1.5, n)
    
    # Calculate means for annotation
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate OLS estimates
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    w1_estimated = numerator / denominator
    w0_estimated = y_mean - w1_estimated * x_mean
    
    # Calculate cost function
    def cost_function(w0, w1):
        return 0.5 / n * np.sum((w0 + w1 * x - y) ** 2)
    
    # Create a grid of w0, w1 values
    w0_range = np.linspace(w0_estimated - 5, w0_estimated + 5, 100)
    w1_range = np.linspace(w1_estimated - 5, w1_estimated + 5, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    J = np.zeros_like(W0)
    
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            J[j, i] = cost_function(w0_range[i], w1_range[j])
    
    # Plot 3D cost function
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # 3D surface plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(W0, W1, J, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # Mark the minimum point
    ax1.scatter([w0_estimated], [w1_estimated], [cost_function(w0_estimated, w1_estimated)], 
                color='green', s=100, label='Minimum')
    
    ax1.set_xlabel('$w_0$ (Intercept)')
    ax1.set_ylabel('$w_1$ (Slope)')
    ax1.set_zlabel('Cost J($w_0$, $w_1$)')
    ax1.set_title('Cost Function Surface', fontsize=14)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Add text annotation explaining the minimum
    min_cost = cost_function(w0_estimated, w1_estimated)
    ax1.text(w0_estimated, w1_estimated, min_cost + 1,
             f'Minimum at:\n$w_0$ = {w0_estimated:.2f}\n$w_1$ = {w1_estimated:.2f}\nCost = {min_cost:.2f}',
             color='black', fontsize=10, backgroundcolor='white')
    
    # Contour plot
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contour(W0, W1, J, 20, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter([w0_estimated], [w1_estimated], color='green', s=100, label='Minimum')
    ax2.set_xlabel('$w_0$ (Intercept)')
    ax2.set_ylabel('$w_1$ (Slope)')
    ax2.set_title('Contour Plot of Cost Function', fontsize=14)
    ax2.grid(True)
    ax2.legend()
    
    # Visualization of the data and fitted line
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(x, y, color='blue', label='Data points')
    
    # Plot fitted line
    x_line = np.linspace(0, 10, 100)
    y_line = w0_estimated + w1_estimated * x_line
    ax3.plot(x_line, y_line, color='red', label=f'Fitted line: y = {w0_estimated:.2f} + {w1_estimated:.2f}x')
    
    # Plot mean point (x̄, ȳ)
    ax3.scatter([x_mean], [y_mean], color='green', s=100, label='Mean point (x̄, ȳ)')
    
    # Add explanatory annotations
    ax3.annotate(f'Mean point: ({x_mean:.2f}, {y_mean:.2f})',
                xy=(x_mean, y_mean), xytext=(x_mean+1, y_mean+1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Data and Fitted Regression Line', fontsize=14)
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_cost_function_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Partial derivatives visualization
    fig = plt.figure(figsize=(14, 7))
    
    # Derivative with respect to w0 (holding w1 constant)
    ax1 = fig.add_subplot(121)
    w0_values = np.linspace(w0_estimated - 5, w0_estimated + 5, 100)
    costs_w0 = np.array([cost_function(w0, w1_estimated) for w0 in w0_values])
    
    ax1.plot(w0_values, costs_w0, 'b-', linewidth=2)
    ax1.axvline(x=w0_estimated, color='r', linestyle='--', label=f'Minimum at $w_0$ = {w0_estimated:.2f}')
    
    # Mark the optimal point
    ax1.scatter([w0_estimated], [cost_function(w0_estimated, w1_estimated)], 
                color='red', s=100)
    
    # Calculate derivative at a specific point
    w0_test = w0_estimated - 3
    # Numerical approximation of derivative (for illustration)
    h = 0.0001
    deriv_w0 = (cost_function(w0_test + h, w1_estimated) - cost_function(w0_test, w1_estimated)) / h
    
    # Tangent line
    if deriv_w0 != 0:  # Avoid horizontal line
        x_tangent = np.linspace(w0_test - 1, w0_test + 1, 10)
        y_tangent = deriv_w0 * (x_tangent - w0_test) + cost_function(w0_test, w1_estimated)
        ax1.plot(x_tangent, y_tangent, 'g-', linewidth=2, 
                 label=f'Tangent at $w_0$ = {w0_test:.2f}')
        ax1.scatter([w0_test], [cost_function(w0_test, w1_estimated)], color='green', s=100)
    
    ax1.set_xlabel('$w_0$ (Intercept)', fontsize=12)
    ax1.set_ylabel('Cost J($w_0$, $w_1$)', fontsize=12)
    ax1.set_title('Cost Function vs $w_0$ (with $w_1$ fixed)', fontsize=14)
    ax1.grid(True)
    ax1.legend()
    
    # Derivative with respect to w1 (holding w0 constant)
    ax2 = fig.add_subplot(122)
    w1_values = np.linspace(w1_estimated - 5, w1_estimated + 5, 100)
    costs_w1 = np.array([cost_function(w0_estimated, w1) for w1 in w1_values])
    
    ax2.plot(w1_values, costs_w1, 'b-', linewidth=2)
    ax2.axvline(x=w1_estimated, color='r', linestyle='--', label=f'Minimum at $w_1$ = {w1_estimated:.2f}')
    
    # Mark the optimal point
    ax2.scatter([w1_estimated], [cost_function(w0_estimated, w1_estimated)], 
                color='red', s=100)
    
    # Calculate derivative at a specific point
    w1_test = w1_estimated + 2
    # Numerical approximation of derivative (for illustration)
    h = 0.0001
    deriv_w1 = (cost_function(w0_estimated, w1_test + h) - cost_function(w0_estimated, w1_test)) / h
    
    # Tangent line
    if deriv_w1 != 0:  # Avoid horizontal line
        x_tangent = np.linspace(w1_test - 1, w1_test + 1, 10)
        y_tangent = deriv_w1 * (x_tangent - w1_test) + cost_function(w0_estimated, w1_test)
        ax2.plot(x_tangent, y_tangent, 'g-', linewidth=2,
                 label=f'Tangent at $w_1$ = {w1_test:.2f}')
        ax2.scatter([w1_test], [cost_function(w0_estimated, w1_test)], color='green', s=100)
    
    ax2.set_xlabel('$w_1$ (Slope)', fontsize=12)
    ax2.set_ylabel('Cost J($w_0$, $w_1$)', fontsize=12)
    ax2.set_title('Cost Function vs $w_1$ (with $w_0$ fixed)', fontsize=14)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_partial_derivatives.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Geometric interpretation of the formulas
    fig = plt.figure(figsize=(10, 8))
    
    # Scatter plot of data
    plt.scatter(x, y, color='blue', label='Data points')
    
    # Plot fitted line
    plt.plot(x_line, y_line, color='red', label=f'Fitted line: y = {w0_estimated:.2f} + {w1_estimated:.2f}x')
    
    # Plot mean point
    plt.scatter([x_mean], [y_mean], color='green', s=100, label='Mean point (x̄, ȳ)')
    
    # Draw vertical lines from points to fitted line (residuals)
    for i in range(n):
        y_pred = w0_estimated + w1_estimated * x[i]
        plt.plot([x[i], x[i]], [y[i], y_pred], 'k--', alpha=0.3)
    
    # Illustrate the formula components
    # Mark a point and show the deviations from means
    i_sample = 15  # Choose a specific data point to annotate
    
    # Draw lines from the point to the means
    plt.plot([x[i_sample], x_mean], [y[i_sample], y[i_sample]], 'g-', linewidth=2)
    plt.plot([x[i_sample], x[i_sample]], [y[i_sample], y_mean], 'g-', linewidth=2)
    
    # Add annotations for formula components
    plt.annotate(f'$(x^{{({i_sample})}} - \\bar{{x}})$',
                xy=((x[i_sample] + x_mean)/2, y[i_sample]),
                xytext=((x[i_sample] + x_mean)/2, y[i_sample] - 0.5),
                fontsize=12, ha='center')
    
    plt.annotate(f'$(y^{{({i_sample})}} - \\bar{{y}})$',
                xy=(x[i_sample], (y[i_sample] + y_mean)/2),
                xytext=(x[i_sample] + 0.5, (y[i_sample] + y_mean)/2),
                fontsize=12, va='center')
    
    # Add a point for the interaction term (x-mean)(y-mean)
    plt.annotate('Product: $(x^{(i)} - \\bar{x})(y^{(i)} - \\bar{y})$',
                xy=(x[i_sample], y[i_sample]),
                xytext=(x[i_sample] + 1, y[i_sample] + 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center', va='center')
    
    # Add formula for w1 (slope)
    plt.text(0.05, 0.05, f"$w_1 = \\frac{{\\sum_{{i=1}}^{{n}}(x^{{(i)}} - \\bar{{x}})(y^{{(i)}} - \\bar{{y}})}}{{\\sum_{{i=1}}^{{n}}(x^{{(i)}} - \\bar{{x}})^2}}$",
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add formula for w0 (intercept)
    plt.text(0.05, 0.15, f"$w_0 = \\bar{{y}} - w_1\\bar{{x}}$",
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add the regression line going through the mean point
    plt.annotate('Regression line passes through (x̄, ȳ)',
                xy=(x_mean, y_mean),
                xytext=(x_mean - 3, y_mean - 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Geometric Interpretation of Regression Formulas', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_geometric_interpretation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute the derivation and visualization
print("# Question 25: Deriving Linear Regression Parameters\n")
formulas = explain_solutions()
saved_files = create_visualizations(formulas, save_dir)

print("\nVisualizations saved to: " + ", ".join(saved_files))
print("\nSummary:")
print("1. We derived the partial derivatives of the cost function with respect to w0 and w1.")
print("2. By setting these derivatives to zero, we found the formulas for w0 and w1.")
print("3. We verified that these formulas match the standard formulas for linear regression.")
print(f"4. Slope (w1): {formulas['w1_formula']}")
print(f"5. Intercept (w0): {formulas['w0_formula']}") 