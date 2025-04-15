import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Class for 3D arrows - Updated to be compatible with newer matplotlib
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
        
    def draw(self, renderer):
        super().draw(renderer)

# Step 1: Explain column space and fitted values
def explain_column_space():
    """Explain what it means for the column space of X to contain fitted values y_hat."""
    print("Step 1: Explaining the column space of X and fitted values")
    print("In linear regression, we have the model y = Xβ + ε, where:")
    print("- y is the vector of observed responses")
    print("- X is the design matrix")
    print("- β is the vector of coefficients")
    print("- ε is the vector of errors")
    print()
    print("The column space of X, denoted C(X), is the vector space spanned by the columns of X.")
    print("It consists of all possible linear combinations of the columns of X.")
    print()
    print("The fitted values ŷ = Xβ̂ are the predictions from our model.")
    print("Since ŷ is a linear combination of the columns of X, it lies in the column space of X.")
    print()
    print("This means:")
    print("1. The fitted values are restricted to the column space of X")
    print("2. The model can only predict values that are linear combinations of the columns of X")
    print("3. The flexibility of our model is constrained by the dimension of the column space")
    print()
    
    # Create a simple example to illustrate
    print("Let's illustrate with a simple example:")
    
    # Generate some data
    np.random.seed(42)
    n = 5  # number of observations
    
    # Create a design matrix with an intercept and one predictor
    X = np.column_stack((np.ones(n), np.random.rand(n)))
    
    # True coefficients
    beta_true = np.array([1, 2])
    
    # Generate responses with some noise
    epsilon = np.random.normal(0, 0.1, n)
    y = X @ beta_true + epsilon
    
    # Compute OLS estimates
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute fitted values
    y_hat = X @ beta_hat
    
    print(f"Design matrix X (with intercept):")
    for i in range(n):
        print(f"  [{X[i,0]:.2f}, {X[i,1]:.2f}]")
    
    print(f"\nResponse vector y:")
    for i in range(n):
        print(f"  {y[i]:.2f}")
    
    print(f"\nEstimated coefficients β̂: [{beta_hat[0]:.2f}, {beta_hat[1]:.2f}]")
    
    print(f"\nFitted values ŷ = Xβ̂:")
    for i in range(n):
        print(f"  {y_hat[i]:.2f} = {X[i,0]:.2f} × {beta_hat[0]:.2f} + {X[i,1]:.2f} × {beta_hat[1]:.2f}")
    
    print("\nThe fitted values ŷ are indeed linear combinations of the columns of X,")
    print("confirming they lie in the column space of X.")
    
    return X, y, beta_hat, y_hat

X, y, beta_hat, y_hat = explain_column_space()

# Step 2: Examine the dimension of the column space
def explain_dimension():
    """Explain the dimension of the column space and its implications."""
    print("\nStep 2: Understanding the dimension of the column space")
    
    print("For a design matrix X with dimensions n × p (n observations, p parameters),")
    print("the dimension of the column space is at most min(n, p).")
    print()
    print("In the given scenario, X is an n × 2 matrix (one column for the intercept, one for a single predictor).")
    print("If the columns are linearly independent (which is typically the case), the dimension is 2.")
    print()
    print("This dimension has several important implications:")
    print("1. The model can only fit patterns in a 2-dimensional subspace of the n-dimensional space")
    print("2. We can only estimate 2 parameters (intercept and slope)")
    print("3. The model can only represent linear relationships between the predictor and response")
    print()
    print("Geometrically, this means our fitted values must lie on a line (or plane) in the n-dimensional space.")
    print("This constrains the flexibility of our model - it can only capture linear patterns in the data.")
    print()
    
    # Demonstrate with an example
    np.random.seed(123)
    n = 100
    x = np.linspace(0, 10, n)
    
    # Linear vs. nonlinear relationship
    y_linear = 2 + 3 * x + np.random.normal(0, 2, n)
    y_nonlinear = 2 + 3 * x + 2 * np.sin(x) + np.random.normal(0, 1, n)
    
    # Fit linear model
    X_linear = np.column_stack((np.ones(n), x))
    beta_hat_linear = np.linalg.inv(X_linear.T @ X_linear) @ X_linear.T @ y_linear
    y_hat_linear = X_linear @ beta_hat_linear
    
    beta_hat_nonlinear = np.linalg.inv(X_linear.T @ X_linear) @ X_linear.T @ y_nonlinear
    y_hat_nonlinear = X_linear @ beta_hat_nonlinear
    
    print("Example: Linear vs. Nonlinear Data")
    print(f"Linear model fitted to linear data: intercept = {beta_hat_linear[0]:.2f}, slope = {beta_hat_linear[1]:.2f}")
    print(f"Linear model fitted to nonlinear data: intercept = {beta_hat_nonlinear[0]:.2f}, slope = {beta_hat_nonlinear[1]:.2f}")
    print()
    print("For the nonlinear data, the linear model cannot capture the sinusoidal pattern")
    print("because the fitted values are constrained to the 2-dimensional column space.")
    print("To capture nonlinear patterns, we would need to expand the column space by adding")
    print("additional predictors or transformations of the original predictor.")
    
    return x, y_linear, y_nonlinear, y_hat_linear, y_hat_nonlinear, X_linear

x, y_linear, y_nonlinear, y_hat_linear, y_hat_nonlinear, X_linear = explain_dimension()

# Step 3: Geometric interpretation of projection
def explain_projection():
    """Explain the geometric interpretation of projection in regression."""
    print("\nStep 3: Geometric interpretation of projection in regression")
    
    print("In linear regression, we seek the vector of coefficients β̂ that minimizes:")
    print("||y - Xβ||² (the sum of squared residuals)")
    print()
    print("The solution is the projection of y onto the column space of X.")
    print("Geometrically, this means:")
    print("1. We project the response vector y orthogonally onto the column space of X")
    print("2. The projected vector is the fitted values ŷ = Xβ̂")
    print("3. The residual vector e = y - ŷ is orthogonal to the column space of X")
    print()
    print("This orthogonality principle is fundamental to least squares regression.")
    print("It ensures that the residuals are uncorrelated with the predictors:")
    print("X'e = 0, meaning the predictors cannot explain any more of the variation in y.")
    print()
    
    # Create an example to illustrate projection in 3D
    np.random.seed(42)
    
    # Create data for 3D example
    n = 20
    X_3d = np.column_stack((np.ones(n), np.random.rand(n), np.random.rand(n)))
    beta_true_3d = np.array([1, 2, 3])
    
    epsilon_3d = np.random.normal(0, 0.5, n)
    y_3d = X_3d @ beta_true_3d + epsilon_3d
    
    # Compute OLS estimates
    beta_hat_3d = np.linalg.inv(X_3d.T @ X_3d) @ X_3d.T @ y_3d
    
    # Compute fitted values
    y_hat_3d = X_3d @ beta_hat_3d
    
    # Compute residuals
    residuals_3d = y_3d - y_hat_3d
    
    # Verify orthogonality
    orth_check = X_3d.T @ residuals_3d
    
    print("3D Example to Illustrate Projection:")
    print(f"Estimated coefficients β̂: [{beta_hat_3d[0]:.2f}, {beta_hat_3d[1]:.2f}, {beta_hat_3d[2]:.2f}]")
    print("Checking orthogonality (X'e should be close to zero):")
    print(f"X'e = [{orth_check[0]:.2e}, {orth_check[1]:.2e}, {orth_check[2]:.2e}]")
    print()
    print("The small values confirm that the residual vector is indeed orthogonal to the column space of X.")
    print("This geometric interpretation helps us understand why the least squares solution")
    print("provides the best linear unbiased estimator (BLUE) under the Gauss-Markov assumptions.")
    
    # Create simpler 2D data for visualization
    n_2d = 20
    X_2d = np.column_stack((np.ones(n_2d), np.random.rand(n_2d)))
    beta_true_2d = np.array([1, 2])
    
    epsilon_2d = np.random.normal(0, 0.5, n_2d)
    y_2d = X_2d @ beta_true_2d + epsilon_2d
    
    # Compute OLS estimates
    beta_hat_2d = np.linalg.inv(X_2d.T @ X_2d) @ X_2d.T @ y_2d
    
    # Compute fitted values
    y_hat_2d = X_2d @ beta_hat_2d
    
    # Compute residuals
    residuals_2d = y_2d - y_hat_2d
    
    return X_3d, y_3d, y_hat_3d, residuals_3d, X_2d, y_2d, y_hat_2d, residuals_2d, beta_hat_2d

X_3d, y_3d, y_hat_3d, residuals_3d, X_2d, y_2d, y_hat_2d, residuals_2d, beta_hat_2d = explain_projection()

# Create visualizations
def create_visualizations(X, y, beta_hat, y_hat, x, y_linear, y_nonlinear, y_hat_linear,
                         y_hat_nonlinear, X_linear, X_3d, y_3d, y_hat_3d, residuals_3d,
                         X_2d, y_2d, y_hat_2d, residuals_2d, save_dir=None):
    """Create visualizations to help understand the vector space concepts in regression."""
    saved_files = []
    
    # Plot 1: Column Space Visualization in 2D
    plt.figure(figsize=(10, 6))
    
    # Extract x values and intercept from design matrix
    x_values = X[:, 1]
    
    # Plot the data points
    plt.scatter(x_values, y, color='blue', s=80, label='Observed Data')
    
    # Plot the fitted values
    plt.scatter(x_values, y_hat, color='red', s=80, label='Fitted Values (ŷ)')
    
    # Plot the column space (which is a line in this case)
    x_line = np.linspace(0, 1, 100)
    y_line = beta_hat[0] + beta_hat[1] * x_line
    plt.plot(x_line, y_line, 'g-', linewidth=2, 
             label='Column Space of X (Span of Intercept and x)')
    
    # Connect actual data to fitted values with lines (residuals)
    for i in range(len(y)):
        plt.plot([x_values[i], x_values[i]], [y[i], y_hat[i]], 'k--', alpha=0.5)
    
    plt.title('Column Space of X and Fitted Values', fontsize=14)
    plt.xlabel('Predictor (x)', fontsize=12)
    plt.ylabel('Response/Fitted Values', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_column_space_2d.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Dimension of Column Space and Model Flexibility
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear data
    axes[0].scatter(x, y_linear, color='blue', alpha=0.6, label='Data')
    axes[0].plot(x, y_hat_linear, 'r-', linewidth=2, label='Linear Model Fit')
    axes[0].set_title('Linear Data: Good Fit', fontsize=14)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)
    
    # Nonlinear data
    axes[1].scatter(x, y_nonlinear, color='blue', alpha=0.6, label='Data')
    axes[1].plot(x, y_hat_nonlinear, 'r-', linewidth=2, label='Linear Model Fit')
    
    # Add true nonlinear function
    y_true_nonlinear = 2 + 3 * x + 2 * np.sin(x)
    axes[1].plot(x, y_true_nonlinear, 'g-', linewidth=2, label='True Nonlinear Function')
    
    axes[1].set_title('Nonlinear Data: Limited Fit', fontsize=14)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle('Model Flexibility Limited by Dimension of Column Space', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_dimension_flexibility.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Column Space Expansion with Additional Features
    plt.figure(figsize=(10, 6))
    
    # Nonlinear data
    plt.scatter(x, y_nonlinear, color='blue', alpha=0.6, label='Nonlinear Data')
    
    # Linear fit
    plt.plot(x, y_hat_nonlinear, 'r-', linewidth=2, label='Linear Model (2D Column Space)')
    
    # Expanded model with sine term
    X_expanded = np.column_stack((X_linear, np.sin(x)))
    beta_hat_expanded = np.linalg.inv(X_expanded.T @ X_expanded) @ X_expanded.T @ y_nonlinear
    y_hat_expanded = X_expanded @ beta_hat_expanded
    
    plt.plot(x, y_hat_expanded, 'g-', linewidth=2, label='Expanded Model (3D Column Space)')
    plt.plot(x, y_true_nonlinear, 'k--', linewidth=1.5, label='True Nonlinear Function')
    
    plt.title('Expanding Column Space for Better Fit', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Add annotation about dimension
    plt.text(0.5, 0.1, 'Adding the sin(x) feature expands\nthe column space to 3 dimensions,\nallowing the model to capture\nnonlinear patterns.',
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_column_space_expansion.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Skip the 3D plot that was causing issues
    # We'll create a simpler version without Arrow3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a simpler 3D visualization of projection
    # Define a plane (column space) and a point (data vector)
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    
    # Define a simple plane z = ax + by + c
    a, b, c = 0.5, 1, 1
    z = a*xx + b*yy + c
    
    # Create a point (data vector) outside the plane
    point = np.array([1, 1, 4])
    
    # Calculate projection onto the plane
    normal = np.array([a, b, -1])
    normal = normal / np.linalg.norm(normal)
    d = -c  # plane equation: ax + by + c = 0 -> ax + by - z + c = 0
    
    # Project the point onto the plane
    t = (np.dot(normal, point) + d) / np.dot(normal, normal)
    projection = point - t * normal
    
    # Plot the plane (column space)
    surf = ax.plot_surface(xx, yy, z, alpha=0.3, color='blue')
    
    # Plot the point (data vector) and its projection
    ax.scatter([point[0]], [point[1]], [point[2]], color='red', s=100, label='Data Vector')
    ax.scatter([projection[0]], [projection[1]], [projection[2]], color='green', s=100, label='Projection (Fitted Values)')
    
    # Connect point to its projection with a line (residual vector)
    ax.plot([point[0], projection[0]], [point[1], projection[1]], [point[2], projection[2]], 
            'purple', linestyle='-', linewidth=2, label='Residual Vector')
    
    # Label axes
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Geometric Interpretation of Projection in Regression', fontsize=14)
    
    # Add text explanation
    ax.text2D(0.05, 0.95, "Projection of data vector onto column space\nResidual vector is orthogonal to column space",
             transform=ax.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Create a custom legend
    ax.legend()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_projection_3d.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Orthogonality of Residuals
    plt.figure(figsize=(10, 6))
    
    # Calculate product of residuals with each column of X_2d
    orth_products = []
    for j in range(X_2d.shape[1]):
        prod = np.dot(X_2d[:, j], residuals_2d)
        orth_products.append(prod)
    
    # Plot correlation between residuals and X columns
    bar_labels = ['Intercept', 'Predictor']
    plt.bar(bar_labels, orth_products, color='purple')
    plt.axhline(y=0, color='red', linestyle='-')
    
    plt.title('Orthogonality of Residuals to Column Space', fontsize=14)
    plt.ylabel('Inner Product with Residuals', fontsize=12)
    plt.grid(True, axis='y')
    
    # Add text explanation
    plt.text(0.5, 0.8, "The inner products are close to zero,\ndemonstrating orthogonality between\nresiduals and the column space of X.",
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_orthogonality_check.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(X, y, beta_hat, y_hat, x, y_linear, y_nonlinear, 
                                   y_hat_linear, y_hat_nonlinear, X_linear, X_3d, 
                                   y_3d, y_hat_3d, residuals_3d, X_2d, y_2d, 
                                   y_hat_2d, residuals_2d, save_dir)

print(f"Visualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 7 Solution Summary:")
print("1. The column space of X contains the fitted values ŷ because ŷ = Xβ̂, which is a linear combination of the columns of X.")
print("2. For an n × 2 design matrix (intercept and one predictor), the dimension of the column space is 2, limiting the model to linear relationships.")
print("3. Geometrically, the projection of y onto the column space of X represents the fitted values ŷ, with the residual vector being orthogonal to the column space.") 