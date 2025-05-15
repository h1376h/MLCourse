import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots and use a font that supports subscripts
plt.style.use('seaborn-v0_8-whitegrid')
# Use a font that's widely available
plt.rcParams['font.family'] = 'DejaVu Sans'

def step1_explain_normal_equations():
    """Explain the normal equations in matrix form."""
    print("Step 1: Normal Equations in Matrix Form")
    print("---------------------------------------")
    print("For a simple linear regression model y = β₀ + β₁x with n data points:")
    print()
    print("1. First, we can write this in matrix form as:")
    print("   y = Xβ + ε")
    print("   where:")
    print("   - y is an n×1 vector of response values")
    print("   - X is an n×2 design matrix where the first column is all 1s and the second column contains the x values")
    print("   - β is a 2×1 vector [β₀, β₁]ᵀ of parameters")
    print("   - ε is an n×1 vector of error terms")
    print()
    print("2. The objective is to minimize the sum of squared errors:")
    print("   min S(β) = (y - Xβ)ᵀ(y - Xβ)")
    print()
    print("3. Taking the derivative with respect to β and setting to zero:")
    print("   ∂S/∂β = -2Xᵀ(y - Xβ) = 0")
    print()
    print("4. Simplifying, we get the normal equations:")
    print("   Xᵀy = XᵀXβ")
    print()
    print("5. Therefore, the solution is:")
    print("   β = (XᵀX)⁻¹Xᵀy")
    print("   which is the formula for the normal equations in matrix form.")
    print()

def step2_matrix_property_for_unique_solution():
    """Explain the matrix property that ensures a unique solution exists."""
    print("Step 2: Matrix Property for Unique Solution")
    print("------------------------------------------")
    print("The key matrix property that ensures a unique solution exists is that XᵀX must be invertible (non-singular).")
    print()
    print("For XᵀX to be invertible, the following conditions must be met:")
    print("1. The matrix X must have full column rank.")
    print("2. There must be at least as many observations as parameters (n ≥ p, where p is the number of parameters).")
    print("3. The columns of X must be linearly independent.")
    print()
    print("In our simple linear regression case with y = β₀ + β₁x:")
    print("- We have 2 parameters (β₀ and β₁)")
    print("- We need at least 2 data points (n ≥ 2)")
    print("- The x values must not all be identical (otherwise the second column would be a scalar multiple of the first)")
    print()
    print("When XᵀX is invertible, the normal equations have a unique solution β = (XᵀX)⁻¹Xᵀy.")
    print("When XᵀX is not invertible, the system is said to be singular or rank-deficient, and either:")
    print("- No solution exists, or")
    print("- Infinitely many solutions exist")
    print()
    print("The condition that XᵀX is invertible is equivalent to saying that the determinant of XᵀX is non-zero:")
    print("det(XᵀX) ≠ 0")
    print()

def demonstrate_normal_equations():
    """Demonstrate the normal equations with a simple example."""
    print("Step 3: Demonstration with a Simple Example")
    print("------------------------------------------")
    
    # Generate some example data
    np.random.seed(42)
    n = 10  # number of data points
    x = np.linspace(0, 10, n)
    beta_true = np.array([3, 2])  # true parameters: intercept=3, slope=2
    epsilon = np.random.normal(0, 1, n)  # random noise
    y = beta_true[0] + beta_true[1] * x + epsilon
    
    print(f"Generated {n} data points with true parameters: β₀ = {beta_true[0]}, β₁ = {beta_true[1]}")
    
    # Create the design matrix X
    X = np.column_stack((np.ones(n), x))
    
    print("\nDesign matrix X (first few rows):")
    for i in range(min(5, n)):
        print(f"  X[{i}] = [{X[i, 0]}, {X[i, 1]}]")
    print("  ...")
    
    # Calculate X^T X
    XtX = X.T @ X
    print("\nX^T X matrix:")
    print(f"  [{XtX[0, 0]:.2f}, {XtX[0, 1]:.2f}]")
    print(f"  [{XtX[1, 0]:.2f}, {XtX[1, 1]:.2f}]")
    
    # Calculate the determinant of X^T X
    det_XtX = np.linalg.det(XtX)
    print(f"\nDeterminant of X^T X: {det_XtX:.2f}")
    
    if det_XtX != 0:
        print("  The determinant is non-zero, so X^T X is invertible and a unique solution exists.")
    else:
        print("  The determinant is zero, so X^T X is not invertible and a unique solution does not exist.")
    
    # Calculate X^T y
    Xty = X.T @ y
    print("\nX^T y vector:")
    print(f"  [{Xty[0]:.2f}, {Xty[1]:.2f}]")
    
    # Calculate the solution using the normal equations
    beta_hat = np.linalg.inv(XtX) @ Xty
    print("\nSolution using normal equations:")
    print(f"  β₀ = {beta_hat[0]:.4f}")
    print(f"  β₁ = {beta_hat[1]:.4f}")
    
    # Calculate using numpy's built-in least squares function for verification
    beta_verify = np.linalg.lstsq(X, y, rcond=None)[0]
    print("\nVerification using numpy's lstsq function:")
    print(f"  β₀ = {beta_verify[0]:.4f}")
    print(f"  β₁ = {beta_verify[1]:.4f}")
    
    # Calculate residuals and SSE
    y_pred = X @ beta_hat
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    print(f"\nSum of squared errors: {sse:.4f}")
    
    return x, y, beta_hat, beta_true, X, XtX

def demonstrate_singular_case():
    """Demonstrate a case where X^T X is singular."""
    print("\nStep 4: Demonstration of a Singular Case")
    print("---------------------------------------")
    
    # Create a case where all x values are the same (perfect collinearity)
    n = 10
    x_singular = np.ones(n) * 5  # all x values are 5
    y_singular = np.array([1, 2, 3, 2, 1, 3, 4, 2, 3, 2])
    
    # Create design matrix
    X_singular = np.column_stack((np.ones(n), x_singular))
    
    print("In this example, all x values are identical (x = 5):")
    print(f"x = {x_singular}")
    
    # Calculate X^T X
    XtX_singular = X_singular.T @ X_singular
    print("\nX^T X matrix for singular case:")
    print(f"  [{XtX_singular[0, 0]:.2f}, {XtX_singular[0, 1]:.2f}]")
    print(f"  [{XtX_singular[1, 0]:.2f}, {XtX_singular[1, 1]:.2f}]")
    
    # Calculate determinant
    det_singular = np.linalg.det(XtX_singular)
    print(f"\nDeterminant of X^T X: {det_singular:.10f}")
    
    print("  The determinant is effectively zero, so X^T X is not invertible.")
    print("  This means there is no unique solution to the normal equations.")
    print("  In this case, the columns of X are linearly dependent because the second column")
    print("  is a scalar multiple of the first (all x values are identical).")
    
    # Try to solve and handle the error
    try:
        beta_singular = np.linalg.inv(XtX_singular) @ (X_singular.T @ y_singular)
        print("\nSolution (this should not execute if truly singular):")
        print(f"  β₀ = {beta_singular[0]:.4f}")
        print(f"  β₁ = {beta_singular[1]:.4f}")
    except np.linalg.LinAlgError:
        print("\nLinear algebra error: Singular matrix cannot be inverted.")
        print("This confirms that no unique solution exists when X^T X is singular.")
    
    # Use the pseudoinverse (Moore-Penrose) for a least-squares solution
    beta_pinv = np.linalg.pinv(X_singular) @ y_singular
    print("\nUsing pseudoinverse to find a solution:")
    print(f"  β₀ = {beta_pinv[0]:.4f}")
    print(f"  β₁ = {beta_pinv[1]:.4f}")
    print("  Note: This is one of infinitely many solutions that minimize the sum of squared errors.")
    
    return x_singular, y_singular, X_singular, XtX_singular

def visualize_normal_equations(x, y, beta_hat, beta_true, save_dir=None):
    """Create visualizations to help understand the normal equations."""
    saved_files = []
    
    # Plot 1: Data points and fitted line
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')
    
    # Plot true line
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_true = beta_true[0] + beta_true[1] * x_line
    plt.plot(x_line, y_true, 'r--', linewidth=2, label=f'True: y = {beta_true[0]} + {beta_true[1]}x')
    
    # Plot fitted line
    y_fit = beta_hat[0] + beta_hat[1] * x_line
    plt.plot(x_line, y_fit, 'g-', linewidth=2, label=f'Fitted: y = {beta_hat[0]:.2f} + {beta_hat[1]:.2f}x')
    
    # Add residuals as vertical lines
    y_pred = beta_hat[0] + beta_hat[1] * x
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Linear Regression: Fitted Line and Residuals', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Least Squares Objective Function Surface
    beta0_range = np.linspace(beta_hat[0] - 3, beta_hat[0] + 3, 50)
    beta1_range = np.linspace(beta_hat[1] - 2, beta_hat[1] + 2, 50)
    
    beta0_grid, beta1_grid = np.meshgrid(beta0_range, beta1_range)
    sse_grid = np.zeros_like(beta0_grid)
    
    X = np.column_stack((np.ones(len(x)), x))
    
    for i in range(len(beta0_range)):
        for j in range(len(beta1_range)):
            beta = np.array([beta0_grid[j, i], beta1_grid[j, i]])
            y_pred = X @ beta
            sse_grid[j, i] = np.sum((y - y_pred) ** 2)
    
    # 3D Surface plot
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1])
    
    # 3D surface
    ax1 = fig.add_subplot(gs[0], projection='3d')
    surf = ax1.plot_surface(beta0_grid, beta1_grid, sse_grid, cmap='viridis', alpha=0.8, linewidth=0)
    ax1.scatter([beta_hat[0]], [beta_hat[1]], [np.min(sse_grid)], color='r', s=100, marker='*')
    
    ax1.set_xlabel('Intercept', fontsize=12)
    ax1.set_ylabel('Slope', fontsize=12)
    ax1.set_zlabel('Sum of Squared Errors', fontsize=12)
    ax1.set_title('SSE as a Function of Intercept and Slope', fontsize=14)
    ax1.view_init(30, 45)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
    cbar.set_label('Sum of Squared Errors')
    
    # Contour plot
    ax2 = fig.add_subplot(gs[1])
    contour = ax2.contour(beta0_grid, beta1_grid, sse_grid, 20, cmap='viridis')
    ax2.scatter([beta_hat[0]], [beta_hat[1]], color='r', s=100, marker='*')
    ax2.set_xlabel('Intercept', fontsize=12)
    ax2.set_ylabel('Slope', fontsize=12)
    ax2.set_title('Contour Plot of SSE', fontsize=14)
    
    # Add colorbar
    cbar2 = fig.colorbar(contour, ax=ax2)
    cbar2.set_label('Sum of Squared Errors')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_sse_surface.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Visualization of the collinearity vs non-collinearity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Non-collinear data
    np.random.seed(42)
    n_points = 50
    x1 = np.linspace(0, 10, n_points)
    y1 = 3 + 2*x1 + np.random.normal(0, 1, n_points)
    
    axes[0].scatter(x1, y1, color='blue', alpha=0.7)
    axes[0].plot(x1, 3 + 2*x1, 'r--', linewidth=2)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Non-Collinear Data: Unique Solution Exists', fontsize=12)
    
    # Create design matrix visualization
    X_noncollinear = np.column_stack((np.ones_like(x1), x1))
    axes[0].text(0.05, 0.95, 'det(X^T X) > 0', transform=axes[0].transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Collinear data
    x2 = np.ones(n_points) * 5  # all x values are 5
    y2 = np.random.normal(3, 1, n_points)
    
    axes[1].scatter(x2, y2, color='red', alpha=0.7)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Collinear Data: No Unique Solution', fontsize=12)
    axes[1].set_xlim([0, 10])
    
    # Create design matrix visualization
    X_collinear = np.column_stack((np.ones_like(x2), x2))
    axes[1].text(0.05, 0.95, 'det(X^T X) = 0', transform=axes[1].transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_collinearity.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Eigenvalues of X^T X matrix
    def plot_eigenvalues(ax, matrix, title):
        eigenvalues = np.linalg.eigvals(matrix)
        bars = ax.bar(range(len(eigenvalues)), np.sort(eigenvalues)[::-1], color='blue', alpha=0.7)
        
        # Add eigenvalue labels
        for i, v in enumerate(np.sort(eigenvalues)[::-1]):
            ax.text(i, v + 0.05*max(eigenvalues), f"{v:.2f}", ha='center')
        
        ax.set_xticks(range(len(eigenvalues)))
        ax.set_xticklabels(['λ1', 'λ2'])
        ax.set_ylabel('Eigenvalue Magnitude')
        ax.set_title(title)
        return eigenvalues
    
    # Create matrices for comparison
    np.random.seed(42)
    n_examples = 30
    
    # Well-conditioned case: independent variables
    x_good = np.linspace(0, 10, n_examples)
    X_good = np.column_stack((np.ones(n_examples), x_good))
    XtX_good = X_good.T @ X_good
    
    # Ill-conditioned case: nearly collinear variables
    x1_bad = np.linspace(0, 10, n_examples)
    x2_bad = 2 * x1_bad + np.random.normal(0, 0.01, n_examples)  # Almost perfect collinearity
    X_bad = np.column_stack((x1_bad, x2_bad))
    XtX_bad = X_bad.T @ X_bad
    
    # Singular case: perfectly collinear variables
    x1_sing = np.linspace(0, 10, n_examples)
    x2_sing = 2 * x1_sing  # Perfect collinearity
    X_sing = np.column_stack((x1_sing, x2_sing))
    XtX_sing = X_sing.T @ X_sing
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    eigenvalues_good = plot_eigenvalues(axs[0], XtX_good, 'Well-Conditioned: All Eigenvalues > 0')
    eigenvalues_bad = plot_eigenvalues(axs[1], XtX_bad, 'Ill-Conditioned: Smallest Eigenvalue Close to 0')
    eigenvalues_sing = plot_eigenvalues(axs[2], XtX_sing, 'Singular: One Eigenvalue = 0')
    
    # Calculate condition numbers
    cond_good = max(eigenvalues_good) / min(eigenvalues_good)
    cond_bad = max(eigenvalues_bad) / min(eigenvalues_bad)
    
    # Add condition numbers as annotations
    axs[0].annotate(f'Condition Number: {cond_good:.2f}', xy=(0.95, 0.85), 
                  xycoords='axes fraction', ha='right', va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axs[1].annotate(f'Condition Number: {cond_bad:.2f}', xy=(0.95, 0.85), 
                  xycoords='axes fraction', ha='right', va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axs[2].annotate('Condition Number: ∞', xy=(0.95, 0.85), 
                  xycoords='axes fraction', ha='right', va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_eigenvalues.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # NEW: Plot 5 - Matrix Representation of Normal Equations
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visualize the X matrix
    n_points = 6  # smaller number for clearer visualization
    x_small = np.linspace(1, 5, n_points)
    X_small = np.column_stack((np.ones(n_points), x_small))
    
    # Plot the design matrix X
    axes[0].imshow(X_small, cmap='Blues', aspect='auto')
    axes[0].set_title('Design Matrix X', fontsize=14)
    axes[0].set_xlabel('Columns', fontsize=12)
    axes[0].set_ylabel('Observations', fontsize=12)
    
    # Add cell values
    for i in range(n_points):
        for j in range(2):
            axes[0].text(j, i, f"{X_small[i, j]:.2f}", ha="center", va="center", color="black")
    
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['1 (Intercept)', 'x (Feature)'])
    
    # Visualize X^T X
    XtX_small = X_small.T @ X_small
    im = axes[1].imshow(XtX_small, cmap='Reds', aspect='equal')
    axes[1].set_title('X^T X Matrix', fontsize=14)
    axes[1].set_xlabel('Columns', fontsize=12)
    axes[1].set_ylabel('Rows', fontsize=12)
    
    # Add cell values for X^T X
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{XtX_small[i, j]:.2f}", ha="center", va="center", color="black")
    
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Intercept', 'Feature'])
    axes[1].set_yticklabels(['Intercept', 'Feature'])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label('Value')
    
    # Add det(X^T X) in the title
    det_XtX_small = np.linalg.det(XtX_small)
    fig.suptitle(f'Matrix Representation of Normal Equations\ndet(X^T X) = {det_XtX_small:.2f}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_matrix_representation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # NEW: Plot 6 - Geometric Interpretation of Normal Equations
    fig = plt.figure(figsize=(12, 8))
    
    # Generate some data for visualization
    np.random.seed(42)
    n_vis = 20
    x_vis = np.linspace(0, 10, n_vis)
    beta_true_vis = np.array([2, 1.5])
    epsilon_vis = np.random.normal(0, 1.5, n_vis)
    y_vis = beta_true_vis[0] + beta_true_vis[1] * x_vis + epsilon_vis
    
    # Create design matrix
    X_vis = np.column_stack((np.ones(n_vis), x_vis))
    
    # Solve normal equations
    beta_hat_vis = np.linalg.inv(X_vis.T @ X_vis) @ (X_vis.T @ y_vis)
    
    # Calculate projections
    y_hat = X_vis @ beta_hat_vis  # Projection of y onto column space of X
    residuals = y_vis - y_hat     # Orthogonal component (residuals)
    
    # Create a 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for the column space of X
    x1_range = np.linspace(0, 10, 20)
    intercept_range = np.linspace(0, 5, 20)
    X1, Intercept = np.meshgrid(x1_range, intercept_range)
    
    # Calculate the points on the plane
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = beta_hat_vis[0] + beta_hat_vis[1] * X1[i, j]
    
    # Plot the column space of X (a plane)
    ax.plot_surface(X1, Z, Intercept, alpha=0.3, color='blue', label='Column Space of X')
    
    # Plot the data points
    ax.scatter(x_vis, y_vis, np.zeros_like(x_vis), color='red', s=50, label='Data Points')
    
    # Plot the fitted points
    ax.scatter(x_vis, y_hat, np.zeros_like(x_vis), color='green', s=50, label='Fitted Points (Projections)')
    
    # Connect each data point to its projection with a line (showing orthogonality)
    for i in range(n_vis):
        ax.plot([x_vis[i], x_vis[i]], [y_vis[i], y_hat[i]], [0, 0], 'k-', alpha=0.3)
    
    # Set labels
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('Intercept', fontsize=12)
    ax.set_title('Geometric Interpretation: y = Projection + Residuals', fontsize=14)
    
    # Adjust the view
    ax.view_init(elev=20, azim=-45)
    
    # Add a legend
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot6_geometric_interpretation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Run the steps
step1_explain_normal_equations()
step2_matrix_property_for_unique_solution()
x, y, beta_hat, beta_true, X, XtX = demonstrate_normal_equations()
x_singular, y_singular, X_singular, XtX_singular = demonstrate_singular_case()

# Create visualizations
saved_files = visualize_normal_equations(x, y, beta_hat, beta_true, save_dir)

print(f"\nVisualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 18 Solution Summary:")
print("1. Normal Equations in Matrix Form: X^T X β = X^T y")
print("2. Matrix property for unique solution: X^T X must be invertible (non-singular)")