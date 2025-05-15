import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Generate some example data
np.random.seed(42)
n = 20
x = np.linspace(0, 10, n)
y_true = 2 + 1.5 * x
noise = np.random.normal(0, 2, n)
y = y_true + noise

# Fit linear regression using least squares
X_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - X_mean) * (y - y_mean))
denominator = np.sum((x - X_mean) ** 2)
beta1 = numerator / denominator
beta0 = y_mean - beta1 * X_mean

# Generate predictions
y_pred = beta0 + beta1 * x

# Calculate errors
vertical_errors = y - y_pred
horizontal_errors = ((y - beta0) / beta1) - x if beta1 != 0 else np.zeros_like(x)
# Calculate perpendicular errors
# The perpendicular distance formula for a point (x0, y0) to line y = mx + b
# d = |y0 - mx0 - b| / sqrt(1 + m^2)
perpendicular_errors = np.abs(y - beta1 * x - beta0) / np.sqrt(1 + beta1**2)

# Helper function to calculate the distance from a point to a line
def point_to_line_distance(x0, y0, m, b, distance_type='vertical'):
    """
    Calculate the distance from point (x0, y0) to line y = mx + b
    distance_type can be 'vertical', 'horizontal', or 'perpendicular'
    """
    if distance_type == 'vertical':
        return np.abs(y0 - (m * x0 + b))
    elif distance_type == 'horizontal':
        # For horizontal distance, we need to find x where y0 = mx + b
        # So x = (y0 - b) / m
        if m == 0:  # Horizontal line
            return float('inf')  # Horizontal distance is undefined
        x_on_line = (y0 - b) / m
        return np.abs(x0 - x_on_line)
    elif distance_type == 'perpendicular':
        # Perpendicular distance formula
        return np.abs(y0 - m * x0 - b) / np.sqrt(1 + m**2)
    
# Function to draw a line connecting a point to the regression line
def draw_connection(ax, x0, y0, m, b, distance_type='vertical', color='red', linestyle='-', alpha=0.7):
    """
    Draw a line connecting point (x0, y0) to the regression line y = mx + b
    according to the specified distance_type
    """
    if distance_type == 'vertical':
        # Vertical connection: same x, different y
        y_on_line = m * x0 + b
        ax.plot([x0, x0], [y0, y_on_line], color=color, linestyle=linestyle, alpha=alpha)
        return np.abs(y0 - y_on_line)
    elif distance_type == 'horizontal':
        # Horizontal connection: same y, different x
        if m == 0:  # Horizontal line
            return None  # Horizontal distance is undefined for horizontal lines
        x_on_line = (y0 - b) / m
        ax.plot([x0, x_on_line], [y0, y0], color=color, linestyle=linestyle, alpha=alpha)
        return np.abs(x0 - x_on_line)
    elif distance_type == 'perpendicular':
        # Perpendicular connection: find the point on the line closest to (x0, y0)
        # The closest point is the intersection of the regression line and the perpendicular line through (x0, y0)
        # The slope of the perpendicular line is -1/m
        if m == 0:  # Horizontal line
            # The closest point is directly below/above (x0, y0)
            x_closest = x0
            y_closest = b
        elif np.isinf(m):  # Vertical line
            # The closest point is directly left/right of (x0, y0)
            x_closest = b  # For vertical lines, b represents the x-intercept
            y_closest = y0
        else:
            # Calculate the perpendicular slope
            m_perp = -1 / m
            # Calculate the y-intercept of the perpendicular line
            b_perp = y0 - m_perp * x0
            # Calculate the intersection point
            x_closest = (b_perp - b) / (m - m_perp)
            y_closest = m * x_closest + b
        
        ax.plot([x0, x_closest], [y0, y_closest], color=color, linestyle=linestyle, alpha=alpha)
        return np.sqrt((x0 - x_closest)**2 + (y0 - y_closest)**2)

def plot_regression_with_distances(save_dir):
    """
    Plot the data points, regression line, and different types of distances.
    """
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # Common setup for all subplots
    def setup_subplot(ax, title):
        ax.scatter(x, y, color='blue', alpha=0.7, label='Data points')
        ax.plot(x, y_pred, color='black', label='Least squares line')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
    
    # Subplot 1: Vertical distances (Option A)
    ax1 = plt.subplot(gs[0, 0])
    setup_subplot(ax1, 'A) Vertical Distances')
    
    # Draw vertical distance lines
    vertical_distances_sum = 0
    for i in range(n):
        distance = draw_connection(ax1, x[i], y[i], beta1, beta0, 'vertical', 'red')
        vertical_distances_sum += distance**2
    
    ax1.text(0.05, 0.95, f'Sum of squared distances: {vertical_distances_sum:.2f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    ax1.legend()
    
    # Subplot 2: Horizontal distances (Option B)
    ax2 = plt.subplot(gs[0, 1])
    setup_subplot(ax2, 'B) Horizontal Distances')
    
    # Draw horizontal distance lines
    horizontal_distances_sum = 0
    for i in range(n):
        distance = draw_connection(ax2, x[i], y[i], beta1, beta0, 'horizontal', 'green')
        if distance is not None:
            horizontal_distances_sum += distance**2
    
    ax2.text(0.05, 0.95, f'Sum of squared distances: {horizontal_distances_sum:.2f}',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    ax2.legend()
    
    # Subplot 3: Perpendicular distances (Option C)
    ax3 = plt.subplot(gs[0, 2])
    setup_subplot(ax3, 'C) Perpendicular Distances')
    
    # Draw perpendicular distance lines
    perpendicular_distances_sum = 0
    for i in range(n):
        distance = draw_connection(ax3, x[i], y[i], beta1, beta0, 'perpendicular', 'purple')
        perpendicular_distances_sum += distance**2
    
    ax3.text(0.05, 0.95, f'Sum of squared distances: {perpendicular_distances_sum:.2f}',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    ax3.legend()
    
    # Subplot 4: Vector projection visualization
    ax4 = plt.subplot(gs[1, 0:2])
    
    # Create the original vectors
    y_vector = y - np.mean(y)
    X = np.column_stack((np.ones(n), x))
    X_centered = X - np.mean(X, axis=0)
    
    # Get the fitted values in centered form
    y_hat_centered = y_pred - np.mean(y)
    
    # For simplicity, we'll create a 2D visualization of the projection
    # We'll project y onto the space spanned by x
    projection_length = np.dot(y_vector, y_hat_centered) / np.linalg.norm(y_hat_centered)
    
    # Create a simplified 2D plot
    origin = np.zeros(2)
    ax4.quiver(*origin, 0, 1, scale=1, scale_units='xy', angles='xy', color='blue', label='y vector', width=0.01)
    ax4.quiver(*origin, 1, 0, scale=1, scale_units='xy', angles='xy', color='red', label='x vector', width=0.01)
    
    # Calculate the projection angle
    angle = np.arccos(projection_length / np.linalg.norm(y_vector))
    angle_deg = np.degrees(angle)
    
    # Draw an arc to show the angle
    theta = np.linspace(0, angle, 100)
    radius = 0.2
    ax4.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-')
    ax4.text(radius * np.cos(angle/2) * 1.2, radius * np.sin(angle/2) * 1.2, f'{angle_deg:.1f}°', fontsize=12)
    
    ax4.quiver(*origin, 1, beta1, scale=3, scale_units='xy', angles='xy', color='green', label='Projection', width=0.01)
    
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_xlabel('X space', fontsize=12)
    ax4.set_ylabel('Y space', fontsize=12)
    ax4.set_title('Vector Projection Interpretation', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4.set_aspect('equal')
    
    # Add text to explain vector projection
    ax4.text(0.05, 0.05, 
             "In the geometric interpretation, y is projected onto\n"
             "the space spanned by X. The least squares solution\n"
             "minimizes the squared length of the residual vector,\n"
             "which is equivalent to minimizing the vertical distances.",
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Subplot 5: 3D Visualization of squared error surface
    ax5 = plt.subplot(gs[1, 2], projection='3d')
    
    # Create a grid of beta0 and beta1 values
    beta0_range = np.linspace(beta0 - 4, beta0 + 4, 50)
    beta1_range = np.linspace(beta1 - 1, beta1 + 1, 50)
    beta0_grid, beta1_grid = np.meshgrid(beta0_range, beta1_range)
    
    # Calculate the sum of squared errors for each combination
    sse_grid = np.zeros_like(beta0_grid)
    for i in range(len(beta0_range)):
        for j in range(len(beta1_range)):
            b0 = beta0_grid[i, j]
            b1 = beta1_grid[i, j]
            pred = b0 + b1 * x
            sse_grid[i, j] = np.sum((y - pred) ** 2)
    
    # Plot the error surface
    surf = ax5.plot_surface(beta0_grid, beta1_grid, sse_grid, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    
    # Mark the minimum point
    ax5.scatter([beta0], [beta1], [np.sum((y - y_pred) ** 2)], color='green', s=100, marker='*')
    
    ax5.set_xlabel('Intercept (β₀)', fontsize=10)
    ax5.set_ylabel('Slope (β₁)', fontsize=10)
    ax5.set_zlabel('Sum of Squared Errors', fontsize=10)
    ax5.set_title('Error Surface with Minimum at OLS Estimates', fontsize=14)
    
    # Add a colorbar
    cbar = plt.colorbar(surf, ax=ax5, shrink=0.6, aspect=10)
    cbar.set_label('Sum of Squared Errors', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_interpretation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return os.path.join(save_dir, 'geometric_interpretation.png')

def plot_optimization_process(save_dir):
    """
    Visualize the optimization process for finding the least squares solution.
    """
    plt.figure(figsize=(12, 10))
    
    # Generate a range of possible slope values
    slopes = np.linspace(0, 3, 100)
    
    # Calculate the sum of squared errors for each slope
    # (fixing the intercept to maintain the constraint that the line passes through the mean point)
    sse_values = []
    for slope in slopes:
        intercept = y_mean - slope * X_mean
        y_hat = intercept + slope * x
        sse = np.sum((y - y_hat) ** 2)
        sse_values.append(sse)
    
    # Find the minimum SSE
    min_idx = np.argmin(sse_values)
    min_slope = slopes[min_idx]
    min_intercept = y_mean - min_slope * X_mean
    min_sse = sse_values[min_idx]
    
    # Plot the SSE curve
    plt.subplot(2, 1, 1)
    plt.plot(slopes, sse_values, 'b-', linewidth=2)
    plt.axvline(x=beta1, color='r', linestyle='--', alpha=0.7, label='OLS Estimate')
    plt.scatter([beta1], [np.sum((y - y_pred) ** 2)], color='r', s=100)
    
    plt.xlabel('Slope (β₁)', fontsize=12)
    plt.ylabel('Sum of Squared Errors', fontsize=12)
    plt.title('Sum of Squared Errors as a Function of Slope', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotation explaining the minimum
    plt.annotate(f'Minimum at β₁ = {beta1:.2f}\nSSE = {np.sum((y - y_pred) ** 2):.2f}',
                xy=(beta1, np.sum((y - y_pred) ** 2)),
                xytext=(beta1+0.5, np.sum((y - y_pred) ** 2)+20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot data and regression lines for different slopes
    plt.subplot(2, 1, 2)
    
    # Data points
    plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')
    
    # OLS regression line
    plt.plot(x, beta0 + beta1 * x, 'r-', linewidth=2, label='OLS line')
    
    # Add a few other lines with different slopes
    other_slopes = [0.5, 1.0, 2.0, 2.5]
    colors = ['green', 'purple', 'orange', 'brown']
    
    for i, slope in enumerate(other_slopes):
        intercept = y_mean - slope * X_mean
        plt.plot(x, intercept + slope * x, color=colors[i], linestyle='--', alpha=0.7,
                label=f'Line with β₁ = {slope:.1f}')
        
        # Calculate SSE for this line
        y_hat = intercept + slope * x
        sse = np.sum((y - y_hat) ** 2)
        
        # Display SSE
        plt.text(x[-1]+0.2, intercept + slope * x[-1], f'SSE = {sse:.1f}', color=colors[i], fontsize=8)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Data with OLS and Alternative Regression Lines', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_process.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return os.path.join(save_dir, 'optimization_process.png')

def plot_matrix_interpretation(save_dir):
    """
    Visualize the matrix algebra interpretation of least squares.
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Create column space visualization
    plt.subplot(2, 1, 1)
    
    # Create the y vector and X matrix
    X = np.column_stack((np.ones(n), x))
    
    # For visualization purposes, we'll project to 2D
    # Let's use PCA for this (simplified version)
    X_centered = X - np.mean(X, axis=0)
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
    X_2d = U[:, :2] * S[:2]
    
    # Project y into same space
    y_2d = np.zeros((n, 2))
    y_2d[:, 0] = y - np.mean(y)
    
    # Calculate fitted values and residuals in this space
    y_hat = y_pred - np.mean(y_pred)
    y_hat_2d = np.zeros((n, 2))
    y_hat_2d[:, 0] = y_hat
    
    residuals_2d = np.zeros((n, 2))
    residuals_2d[:, 1] = residuals
    
    # Plot vectors
    origin = np.zeros(2)
    
    # Set up plot limits
    max_range = max(np.max(np.abs(X_2d)), np.max(np.abs(y_2d)))
    plt.xlim(-max_range*1.2, max_range*1.2)
    plt.ylim(-max_range*1.2, max_range*1.2)
    
    # Plot column space and orthogonal residuals
    for i in range(n):
        plt.arrow(0, 0, y_2d[i, 0], y_2d[i, 1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.3)
        plt.arrow(0, 0, y_hat_2d[i, 0], y_hat_2d[i, 1], head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.3)
        plt.arrow(y_hat_2d[i, 0], y_hat_2d[i, 1], residuals_2d[i, 0], residuals_2d[i, 1], 
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.3)
    
    # Add column space vectors
    plt.arrow(0, 0, X_2d[0, 0], X_2d[0, 1], head_width=0.2, head_length=0.2, fc='purple', ec='purple', width=0.05, label='Column space of X')
    
    plt.scatter(0, 0, color='black', s=50)  # Origin
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Vector Space Interpretation: y is Projected onto Column Space of X', fontsize=14)
    
    # Add legend with colored dots instead of arrows
    plt.scatter([], [], color='blue', alpha=0.7, label='y vectors')
    plt.scatter([], [], color='green', alpha=0.7, label='ŷ projection (fitted values)')
    plt.scatter([], [], color='red', alpha=0.7, label='Residual vectors (y - ŷ)')
    plt.scatter([], [], color='purple', alpha=0.7, label='Column space of X')
    plt.legend(loc='upper right')
    
    # Add explanatory text
    plt.text(-max_range*1.1, max_range*0.8, 
             "The least squares solution projects y onto the column space of X.\n"
             "The residual vector is orthogonal to the column space of X,\n"
             "which is a fundamental property of the OLS solution.\n"
             "This orthogonality condition leads to the normal equations: X'(y - Xβ) = 0",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot normal equations visualization
    plt.subplot(2, 1, 2)
    
    # Generate new data for better visualization
    normal_x = np.linspace(0, 10, 100)
    normal_y_pred = beta0 + beta1 * normal_x
    
    # Plot the data and regression line
    plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')
    plt.plot(normal_x, normal_y_pred, 'r-', linewidth=2, label='OLS line')
    
    # Draw residual vectors
    for i in range(n):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g-', alpha=0.5)
    
    # Highlight the orthogonality condition
    # Create weighted residuals
    weighted_residuals_x = x * residuals
    weighted_residuals_constant = residuals
    
    # Show that sum of weighted residuals is zero
    sum_weighted_x = np.sum(weighted_residuals_x)
    sum_weighted_constant = np.sum(weighted_residuals_constant)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Normal Equations: X\'(y - Xβ) = 0', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text explaining normal equations
    plt.text(0.05, 0.05,
             f"The normal equations ensure that residuals are uncorrelated with predictors:\n"
             f"Sum of residuals × 1: {sum_weighted_constant:.6f} ≈ 0\n"
             f"Sum of residuals × x: {sum_weighted_x:.6f} ≈ 0\n\n"
             f"This means that the residual vector is orthogonal to each column of X,\n"
             f"which is equivalent to minimizing the squared vertical distances.",
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'matrix_interpretation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return os.path.join(save_dir, 'matrix_interpretation.png')

def main():
    """
    Main function to generate all visualizations and print out the solution.
    """
    print("\n" + "="*80)
    print("Question 17: Geometric Interpretation of Linear Regression".center(80))
    print("="*80 + "\n")
    
    print("Step 1: Understanding the Least Squares Method Geometrically")
    print("-" * 60)
    print("In linear regression, we aim to find the line that best fits our data.")
    print("The least squares method finds the line that minimizes the sum of squared errors.")
    print(f"For our example data, we found the regression line: y = {beta0:.4f} + {beta1:.4f}x")
    print()
    
    # Calculate the sum of squared errors for each distance type
    sse_vertical = np.sum(vertical_errors**2)
    sse_horizontal = np.sum(horizontal_errors**2)
    sse_perpendicular = np.sum(perpendicular_errors**2)
    
    print("Step 2: Comparing Different Distance Metrics")
    print("-" * 60)
    print(f"Sum of squared vertical distances: {sse_vertical:.4f}")
    print(f"Sum of squared horizontal distances: {sse_horizontal:.4f}")
    print(f"Sum of squared perpendicular distances: {sse_perpendicular:.4f}")
    print()
    
    print("Step 3: Analyzing the Least Squares Solution")
    print("-" * 60)
    print("The least squares method specifically minimizes the sum of squared vertical distances.")
    print("This corresponds to option A in the multiple-choice question.")
    print()
    
    print("Key insights about the geometric interpretation of least squares:")
    print("1. The vertical distances represent the residuals (y - ŷ) in the model.")
    print("2. Minimizing squared vertical distances is equivalent to:")
    print("   - Projecting the response vector onto the column space of the design matrix")
    print("   - Making the residual vector orthogonal to the column space of the design matrix")
    print("3. The orthogonality condition leads to the normal equations: X'(y - Xβ) = 0")
    print("4. While other distance metrics (horizontal or perpendicular) could be minimized,")
    print("   standard linear regression specifically minimizes vertical distances.")
    print()
    
    # Generate all visualizations
    geometric_fig = plot_regression_with_distances(save_dir)
    optimization_fig = plot_optimization_process(save_dir)
    matrix_fig = plot_matrix_interpretation(save_dir)
    
    print("Visualizations generated:")
    print(f"1. {geometric_fig}")
    print(f"2. {optimization_fig}")
    print(f"3. {matrix_fig}")
    print()
    
    print("Conclusion:")
    print("The correct answer is A) It minimizes the sum of vertical distances between points and the regression line.")
    print("This is a fundamental property of ordinary least squares regression.")

if __name__ == "__main__":
    main() 