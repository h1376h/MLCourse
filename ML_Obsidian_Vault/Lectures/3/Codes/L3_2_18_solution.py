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
    """Explain the normal equations in matrix form with detailed derivation."""
    print("\n" + "="*80)
    print("STEP 1: DERIVING THE NORMAL EQUATIONS IN MATRIX FORM")
    print("="*80)
    print("For a simple linear regression model y = β₀ + β₁x with n data points:")
    print()
    print("1. First, we express the model in matrix form:")
    print("   y = Xβ + ε")
    print()
    print("   where:")
    print("   - y is an n×1 vector of response values [y₁, y₂, ..., yₙ]ᵀ")
    print("   - X is an n×2 design matrix where the first column is all 1s (for the intercept)")
    print("     and the second column contains the x values:")
    print()
    print("     X = [1  x₁]")
    print("         [1  x₂]")
    print("         [⋮  ⋮ ]")
    print("         [1  xₙ]")
    print()
    print("   - β is a 2×1 vector [β₀, β₁]ᵀ of parameters")
    print("   - ε is an n×1 vector of error terms [ε₁, ε₂, ..., εₙ]ᵀ")
    print()
    print("2. The objective in linear regression is to minimize the sum of squared errors (SSE):")
    print("   min S(β) = εᵀε = (y - Xβ)ᵀ(y - Xβ)")
    print()
    print("   Expanding this expression:")
    print("   S(β) = (y - Xβ)ᵀ(y - Xβ)")
    print("        = yᵀy - yᵀXβ - βᵀXᵀy + βᵀXᵀXβ")
    print("        = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ")
    print("   Note that yᵀXβ = βᵀXᵀy because both are scalar (1×1) values")
    print()
    print("3. To find the values of β that minimize S(β), we take the derivative with respect to β:")
    print("   ∂S/∂β = ∂/∂β(yᵀy - 2βᵀXᵀy + βᵀXᵀXβ)")
    print("         = 0 - 2Xᵀy + 2XᵀXβ")
    print("         = -2Xᵀy + 2XᵀXβ")
    print("         = -2Xᵀ(y - Xβ)")
    print()
    print("4. Setting the derivative equal to zero (necessary condition for minimum):")
    print("   ∂S/∂β = -2Xᵀ(y - Xβ) = 0")
    print()
    print("   Simplifying:")
    print("   Xᵀ(y - Xβ) = 0")
    print("   Xᵀy - XᵀXβ = 0")
    print("   XᵀXβ = Xᵀy")
    print()
    print("5. This final equation, XᵀXβ = Xᵀy, is the formula for the normal equations in matrix form.")
    print()
    print("6. If XᵀX is invertible, the solution is:")
    print("   β = (XᵀX)⁻¹Xᵀy")
    print()
    print("   This is the least squares estimator for β.")
    print()
    print("7. For our simple linear regression model with two parameters, the normal equations")
    print("   can be written explicitly as:")
    print()
    print("   [∑(1)    ∑(xᵢ)   ] [β₀] = [∑(yᵢ)      ]")
    print("   [∑(xᵢ)   ∑(xᵢ²)  ] [β₁]   [∑(xᵢyᵢ)    ]")
    print()
    print("   which gives the familiar formulas for simple linear regression:")
    print("   β₁ = [∑(xᵢyᵢ) - n⁻¹(∑xᵢ)(∑yᵢ)] / [∑(xᵢ²) - n⁻¹(∑xᵢ)²]")
    print("   β₀ = ȳ - β₁x̄")
    print()

def step2_matrix_property_for_unique_solution():
    """Explain the matrix property that ensures a unique solution exists with detailed analysis."""
    print("\n" + "="*80)
    print("STEP 2: MATRIX PROPERTY FOR UNIQUE SOLUTION")
    print("="*80)
    print("The key matrix property that ensures a unique solution exists is that XᵀX must be invertible (non-singular).")
    print()
    print("For XᵀX to be invertible, the following conditions must be met:")
    print()
    print("1. The matrix X must have full column rank.")
    print("   This means that the columns of X must be linearly independent vectors.")
    print("   In mathematical terms: rank(X) = p, where p is the number of columns in X.")
    print()
    print("2. There must be at least as many observations as parameters (n ≥ p).")
    print("   This is because a matrix with more columns than rows cannot have full column rank.")
    print()
    print("3. No exact linear relationship can exist among the predictor variables.")
    print("   This means no column can be expressed as a linear combination of other columns.")
    print()
    print("Mathematical Properties:")
    print()
    print("1. XᵀX is always a square matrix with dimensions p×p.")
    print("2. XᵀX is always symmetric: (XᵀX)ᵀ = XᵀX")
    print("3. XᵀX is positive semi-definite, meaning:")
    print("   a. All eigenvalues are non-negative")
    print("   b. For any vector v, vᵀ(XᵀX)v ≥ 0")
    print("4. XᵀX is positive definite (and thus invertible) if and only if X has full column rank.")
    print("   When positive definite:")
    print("   a. All eigenvalues are strictly positive")
    print("   b. For any non-zero vector v, vᵀ(XᵀX)v > 0")
    print()
    print("In our simple linear regression case with y = β₀ + β₁x, the design matrix X has 2 columns:")
    print("- First column: all 1s (for the intercept β₀)")
    print("- Second column: the x values (for the slope β₁)")
    print()
    print("For XᵀX to be invertible in this case:")
    print("1. We need at least 2 data points (n ≥ 2)")
    print("2. The x values must not all be identical")
    print()
    print("If all x values were identical (e.g., all equal to 5), then the second column would be")
    print("a scalar multiple of the first: [x₁, x₂, ..., xₙ]ᵀ = 5[1, 1, ..., 1]ᵀ")
    print("This would make the columns linearly dependent, causing XᵀX to be singular.")
    print()
    print("Testing for invertibility:")
    print("1. Check the determinant: det(XᵀX) ≠ 0")
    print("2. Check the eigenvalues: all eigenvalues of XᵀX > 0")
    print("3. Check the rank: rank(X) = p")
    print()
    print("When XᵀX is invertible, the normal equations have a unique solution β = (XᵀX)⁻¹Xᵀy.")
    print("When XᵀX is not invertible, the system is rank-deficient, and either:")
    print("- No solution exists, or")
    print("- Infinitely many solutions exist that minimize the sum of squared errors")
    print()

def demonstrate_normal_equations():
    """Demonstrate the normal equations with a detailed numerical example."""
    print("\n" + "="*80)
    print("STEP 3: NUMERICAL EXAMPLE - WELL-CONDITIONED CASE")
    print("="*80)
    
    # Generate some example data
    np.random.seed(42)
    n = 10  # number of data points
    x = np.linspace(0, 10, n)
    beta_true = np.array([3, 2])  # true parameters: intercept=3, slope=2
    epsilon = np.random.normal(0, 1, n)  # random noise
    y = beta_true[0] + beta_true[1] * x + epsilon
    
    print(f"Generated {n} data points with true parameters: β₀ = {beta_true[0]}, β₁ = {beta_true[1]}")
    print(f"Added random noise with mean 0 and standard deviation 1")
    print()
    
    # Display the data
    print("Data points (x, y):")
    for i in range(n):
        print(f"  ({x[i]:.2f}, {y[i]:.2f})")
    print()
    
    # Create the design matrix X
    X = np.column_stack((np.ones(n), x))
    
    print("Design matrix X:")
    for i in range(n):
        print(f"  X[{i}] = [{X[i, 0]:.1f}, {X[i, 1]:.2f}]")
    print()
    
    # Calculate X^T (transpose of X)
    Xt = X.T
    print("X^T (transpose of X):")
    print(f"  X^T[0] = {Xt[0]}")
    print(f"  X^T[1] = {Xt[1]}")
    print()
    
    # Calculate X^T X
    XtX = X.T @ X
    print("X^T X matrix:")
    print(f"  [{XtX[0, 0]:.2f}, {XtX[0, 1]:.2f}]")
    print(f"  [{XtX[1, 0]:.2f}, {XtX[1, 1]:.2f}]")
    print()
    print("Interpretation of X^T X elements:")
    print(f"  X^T X[0,0] = {XtX[0, 0]:.2f} = sum of squared elements in column 1 = n (number of observations)")
    print(f"  X^T X[0,1] = X^T X[1,0] = {XtX[0, 1]:.2f} = sum of x values")
    print(f"  X^T X[1,1] = {XtX[1, 1]:.2f} = sum of squared x values")
    print()
    
    # Calculate the determinant of X^T X
    det_XtX = np.linalg.det(XtX)
    print(f"Determinant of X^T X: det(X^T X) = {det_XtX:.2f}")
    
    if det_XtX != 0:
        print("  ✓ The determinant is non-zero, so X^T X is invertible (non-singular).")
        print("  ✓ A unique solution exists for the normal equations.")
    else:
        print("  ✗ The determinant is zero, so X^T X is not invertible (singular).")
        print("  ✗ A unique solution does not exist.")
    print()
    
    # Calculate eigenvalues of X^T X
    eigenvalues = np.linalg.eigvals(XtX)
    print(f"Eigenvalues of X^T X: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}")
    if np.all(eigenvalues > 0):
        print("  ✓ All eigenvalues are positive, confirming X^T X is positive definite.")
    else:
        print("  ✗ Not all eigenvalues are positive. X^T X is not positive definite.")
    print()
    
    # Calculate the condition number
    cond_num = np.linalg.cond(XtX)
    print(f"Condition number of X^T X: {cond_num:.2f}")
    if cond_num < 1000:
        print("  ✓ Condition number is relatively small, suggesting good numerical stability.")
    else:
        print("  ! Large condition number indicates potential numerical issues.")
    print()
    
    # Calculate X^T y
    Xty = X.T @ y
    print("X^T y vector:")
    print(f"  [{Xty[0]:.2f}, {Xty[1]:.2f}]")
    print()
    print("Interpretation of X^T y elements:")
    print(f"  X^T y[0] = {Xty[0]:.2f} = sum of y values")
    print(f"  X^T y[1] = {Xty[1]:.2f} = sum of x*y products")
    print()
    
    # Calculate the inverse of X^T X
    XtX_inv = np.linalg.inv(XtX)
    print("(X^T X)^-1 matrix:")
    print(f"  [{XtX_inv[0, 0]:.6f}, {XtX_inv[0, 1]:.6f}]")
    print(f"  [{XtX_inv[1, 0]:.6f}, {XtX_inv[1, 1]:.6f}]")
    print()
    
    # Verify that (X^T X)^-1 * (X^T X) = I
    I_check = XtX_inv @ XtX
    print("Verification: (X^T X)^-1 * (X^T X) should equal the identity matrix:")
    print(f"  [{I_check[0, 0]:.6f}, {I_check[0, 1]:.6f}]")
    print(f"  [{I_check[1, 0]:.6f}, {I_check[1, 1]:.6f}]")
    print("  ✓ Values are very close to the identity matrix [1, 0; 0, 1]")
    print()
    
    # Calculate the solution using the normal equations
    beta_hat = XtX_inv @ Xty
    print("Step-by-step solution using normal equations:")
    print("1. XᵀX = \n", XtX)
    print("2. Xᵀy = \n", Xty)
    print("3. (X^T X)⁻¹ = \n", XtX_inv)
    print("4. β = (X^T X)⁻¹X^T y = \n", XtX_inv @ Xty)
    print()
    print("Final solution:")
    print(f"  β₀ = {beta_hat[0]:.4f}  (True value: {beta_true[0]})")
    print(f"  β₁ = {beta_hat[1]:.4f}  (True value: {beta_true[1]})")
    print()
    
    # Calculate using numpy's built-in least squares function for verification
    beta_verify = np.linalg.lstsq(X, y, rcond=None)[0]
    print("Verification using numpy's built-in least squares function:")
    print(f"  β₀ = {beta_verify[0]:.4f}")
    print(f"  β₁ = {beta_verify[1]:.4f}")
    print("  ✓ Results match our manual calculation")
    print()
    
    # Calculate residuals and SSE
    y_pred = X @ beta_hat
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    print("Model Evaluation:")
    print(f"  Sum of Squared Errors (SSE): {sse:.4f}")
    
    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (sse / ss_total)
    print(f"  R-squared: {r_squared:.4f}")
    print()
    
    return x, y, beta_hat, beta_true, X, XtX

def demonstrate_singular_case():
    """Demonstrate a case where X^T X is singular with detailed analysis."""
    print("\n" + "="*80)
    print("STEP 4: NUMERICAL EXAMPLE - SINGULAR CASE")
    print("="*80)
    
    # Create a case where all x values are the same (perfect collinearity)
    n = 10
    x_singular = np.ones(n) * 5  # all x values are 5
    y_singular = np.array([1, 2, 3, 2, 1, 3, 4, 2, 3, 2])
    
    print(f"Created a special case with {n} data points where all x values are identical (x = 5).")
    print("In this case, the columns of the design matrix X will be linearly dependent.")
    print()
    
    # Display the data
    print("Data points (x, y):")
    for i in range(n):
        print(f"  ({x_singular[i]:.1f}, {y_singular[i]:.1f})")
    print()
    
    # Create design matrix
    X_singular = np.column_stack((np.ones(n), x_singular))
    
    print("Design matrix X:")
    for i in range(min(5, n)):
        print(f"  X[{i}] = [{X_singular[i, 0]:.1f}, {X_singular[i, 1]:.1f}]")
    print("  ...")
    print()
    
    print("Observation: The second column is exactly 5 times the first column.")
    print("  This means the two columns are linearly dependent: col2 = 5 * col1")
    print("  Linear dependence causes the matrix X to not have full column rank.")
    print()
    
    # Calculate X^T X
    XtX_singular = X_singular.T @ X_singular
    print("X^T X matrix:")
    print(f"  [{XtX_singular[0, 0]:.2f}, {XtX_singular[0, 1]:.2f}]")
    print(f"  [{XtX_singular[1, 0]:.2f}, {XtX_singular[1, 1]:.2f}]")
    print()
    
    # Verify linear dependence in X^T X
    ratio1 = XtX_singular[0, 1] / XtX_singular[0, 0] if XtX_singular[0, 0] != 0 else float('inf')
    ratio2 = XtX_singular[1, 1] / XtX_singular[1, 0] if XtX_singular[1, 0] != 0 else float('inf')
    
    print(f"Ratio check for linear dependence in X^T X:")
    print(f"  Row 1 ratio: {XtX_singular[0, 1]:.1f} / {XtX_singular[0, 0]:.1f} = {ratio1:.1f}")
    print(f"  Row 2 ratio: {XtX_singular[1, 1]:.1f} / {XtX_singular[1, 0]:.1f} = {ratio2:.1f}")
    print(f"  Both rows have the same ratio = {ratio1:.1f}, confirming linear dependence.")
    print()
    
    # Calculate determinant
    det_singular = np.linalg.det(XtX_singular)
    print(f"Determinant of X^T X: det(X^T X) = {det_singular:.10f}")
    
    if abs(det_singular) < 1e-10:
        print("  ✗ The determinant is effectively zero, so X^T X is not invertible (singular).")
        print("  ✗ This means a unique solution to the normal equations does not exist.")
    else:
        print("  ✓ The determinant is non-zero, so X^T X is invertible.")
    print()
    
    # Calculate eigenvalues
    try:
        eigenvalues_singular = np.linalg.eigvals(XtX_singular)
        print(f"Eigenvalues of X^T X: {eigenvalues_singular[0]:.6f}, {eigenvalues_singular[1]:.6f}")
        print(f"  One eigenvalue is effectively zero, confirming that X^T X is singular.")
        print()
    except np.linalg.LinAlgError:
        print("  Could not calculate eigenvalues due to numerical issues.")
        print()
    
    # Try to solve and handle the error
    print("Attempting to solve normal equations using direct matrix inversion:")
    try:
        beta_singular = np.linalg.inv(XtX_singular) @ (X_singular.T @ y_singular)
        print("  Solution found (this should not happen for a truly singular matrix):")
        print(f"  β₀ = {beta_singular[0]:.4f}")
        print(f"  β₁ = {beta_singular[1]:.4f}")
    except np.linalg.LinAlgError as e:
        print(f"  Error: {e}")
        print("  ✓ As expected, matrix inversion failed because X^T X is singular.")
    print()
    
    # Use the pseudoinverse (Moore-Penrose) for a least-squares solution
    print("Alternative approach: Using the pseudoinverse (Moore-Penrose):")
    X_pinv = np.linalg.pinv(X_singular)
    beta_pinv = X_pinv @ y_singular
    
    print("Step-by-step pseudoinverse solution:")
    print("1. Calculate the pseudoinverse X⁺ using SVD decomposition")
    print("2. β = X⁺y")
    print()
    print("Result:")
    print(f"  β₀ = {beta_pinv[0]:.4f}")
    print(f"  β₁ = {beta_pinv[1]:.4f}")
    print()
    
    # Verify that this is a valid solution (minimizes SSE)
    y_pred_pinv = X_singular @ beta_pinv
    sse_pinv = np.sum((y_singular - y_pred_pinv)**2)
    print(f"Sum of squared errors for pseudoinverse solution: {sse_pinv:.4f}")
    print()
    
    # Find another solution that also minimizes SSE
    beta_alt = np.array([beta_pinv[0] + 1, beta_pinv[1] - 0.2])
    y_pred_alt = X_singular @ beta_alt
    sse_alt = np.sum((y_singular - y_pred_alt)**2)
    print("Testing an alternative solution:")
    print(f"  β₀ = {beta_alt[0]:.4f}")
    print(f"  β₁ = {beta_alt[1]:.4f}")
    print(f"  SSE = {sse_alt:.4f}")
    print()
    
    if abs(sse_alt - sse_pinv) < 1e-10:
        print("  ✓ The alternative solution gives the same SSE.")
        print("  ✓ This confirms there are infinitely many solutions that minimize the SSE.")
    else:
        print("  ✗ The alternative solution does not give the same SSE.")
    print()
    print("Conclusion: When X^T X is singular, there are infinitely many solutions")
    print("that minimize the sum of squared errors.")
    print()
    
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