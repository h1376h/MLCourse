import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def plot_design_matrix():
    """Demonstrate the dimensions of a design matrix in multiple linear regression."""
    print("\n1. Design Matrix Dimensions")
    print("----------------------------")
    
    # Create a small example dataset
    n_samples = 5  # Number of samples
    d_features = 3  # Number of features
    
    # Generate random data
    X = np.random.rand(n_samples, d_features)
    
    # Add a column of ones for the intercept
    X_with_intercept = np.column_stack((np.ones(n_samples), X))
    
    # Display the matrices
    print(f"Original features (X) with shape {X.shape}:")
    print(X)
    print("\nDesign matrix (X with intercept) with shape {X_with_intercept.shape}:")
    print(X_with_intercept)
    
    # Create a visual representation of the design matrix
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create heatmap of design matrix
    im = ax.imshow(X_with_intercept, cmap='viridis')
    
    # Add grid lines
    for i in range(X_with_intercept.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(X_with_intercept.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)
    
    # Add labels
    ax.set_xticks(np.arange(X_with_intercept.shape[1]))
    ax.set_yticks(np.arange(X_with_intercept.shape[0]))
    ax.set_xticklabels(['Intercept', 'Feature 1', 'Feature 2', 'Feature 3'])
    ax.set_yticklabels([f'Sample {i+1}' for i in range(n_samples)])
    
    # Add text annotations
    for i in range(X_with_intercept.shape[0]):
        for j in range(X_with_intercept.shape[1]):
            ax.text(j, i, f'{X_with_intercept[i, j]:.2f}', 
                    ha='center', va='center', color='white')
    
    # Add a colorbar
    plt.colorbar(im)
    
    # Set title and labels
    ax.set_title('Design Matrix Dimensions', fontsize=14)
    
    # Instead of figtext, print the explanation
    print(f"The design matrix has dimensions n × (d+1): {n_samples} × {d_features+1}")
    print(f"where n = {n_samples} is the number of samples and d = {d_features} is the number of features")
    print(f"The +1 is for the intercept column of ones")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'design_matrix.png'), dpi=300)
    
    print(f"Answer: n × (d+1) or {n_samples} × {d_features+1} in this example")
    print(f"where n is the number of samples and d is the number of features\n")
    
    return fig

def demonstrate_closed_form_solution():
    """Demonstrate the closed-form solution for linear regression."""
    print("\n2. Closed-Form Solution to Least Squares")
    print("---------------------------------------")
    
    # Generate some sample data
    np.random.seed(42)
    n_samples = 50
    X = np.random.rand(n_samples, 1) * 10  # Single feature for simplicity
    
    # True coefficients
    beta_true = np.array([2.5, 1.5])  # [intercept, slope]
    
    # Generate target values with some noise
    y = beta_true[0] + beta_true[1] * X.squeeze() + np.random.randn(n_samples) * 2
    
    # Create design matrix with intercept
    X_design = np.column_stack((np.ones(n_samples), X))
    
    # Calculate matrices for closed-form solution
    X_T_X = X_design.T @ X_design
    X_T_y = X_design.T @ y
    
    # Calculate the closed-form solution
    beta_hat = np.linalg.inv(X_T_X) @ X_T_y
    
    print(f"Design matrix X shape: {X_design.shape}")
    print(f"Target vector y shape: {y.shape}")
    print("\nStep 1: Calculate X^T X")
    print(f"X^T X shape: {X_T_X.shape}")
    print(X_T_X)
    
    print("\nStep 2: Calculate X^T y")
    print(f"X^T y shape: {X_T_y.shape}")
    print(X_T_y)
    
    print("\nStep 3: Calculate (X^T X)^(-1)")
    print(f"(X^T X)^(-1) shape: {np.linalg.inv(X_T_X).shape}")
    print(np.linalg.inv(X_T_X))
    
    print("\nStep 4: Calculate (X^T X)^(-1) X^T y")
    print(f"Beta hat: {beta_hat}")
    
    # Create a figure to visualize the solution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the data and regression line
    axes[0].scatter(X, y, alpha=0.7, label='Data points')
    x_range = np.linspace(0, 10, 100)
    y_pred = beta_hat[0] + beta_hat[1] * x_range
    axes[0].plot(x_range, y_pred, 'r-', linewidth=2, label='Regression line')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Least Squares Regression', fontsize=14)
    axes[0].legend()
    
    # Visualize the equation
    axes[1].axis('off')
    equation_text = r"""
    Closed-form solution:
    $\hat{\boldsymbol{\beta}} = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$
    
    In our example:
    $\hat{\boldsymbol{\beta}} = 
    \begin{bmatrix}
    %.2f \\
    %.2f
    \end{bmatrix}$
    """ % (beta_hat[0], beta_hat[1])
    
    axes[1].text(0.1, 0.5, equation_text, fontsize=14, va='center')
    
    # Print explanation instead of including in the figure
    print("\nClosed-form solution explanation:")
    print("Where:")
    print("- X is the design matrix (n × (d+1))")
    print("- y is the target vector (n × 1)")
    print("- β̂ is the vector of estimated coefficients ((d+1) × 1)")
    print(f"In our example, β̂ = [{beta_hat[0]:.2f}, {beta_hat[1]:.2f}]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'closed_form_solution.png'), dpi=300)
    
    print(f"\nAnswer: (X^T X)^(-1) X^T y\n")
    
    return fig

def demonstrate_multicollinearity():
    """Demonstrate the effect of multicollinearity on the X^T X matrix."""
    print("\n3. Effect of Perfect Multicollinearity on X^T X")
    print("---------------------------------------------")
    
    # Create a design matrix with perfect multicollinearity
    n_samples = 5
    
    # Create features where x2 = 2*x1 (perfect multicollinearity)
    x1 = np.random.rand(n_samples)
    x2 = 2 * x1  # Perfect multicollinearity
    x3 = np.random.rand(n_samples)  # Independent feature
    
    # Create design matrix with intercept
    X = np.column_stack((np.ones(n_samples), x1, x2, x3))
    
    # Calculate X^T X
    X_T_X = X.T @ X
    
    print("Design matrix X with multicollinearity:")
    print(X)
    
    print("\nX^T X matrix:")
    print(X_T_X)
    
    try:
        # Try to calculate the inverse
        X_T_X_inv = np.linalg.inv(X_T_X)
        print("\nInverse of X^T X (should fail or give incorrect results):")
        print(X_T_X_inv)
    except np.linalg.LinAlgError as e:
        print(f"\nError computing inverse: {e}")
    
    # Calculate determinant to show it's singular (or close to zero)
    det_X_T_X = np.linalg.det(X_T_X)
    print(f"\nDeterminant of X^T X: {det_X_T_X}")
    
    # Calculate eigenvalues to show the matrix is singular
    eigenvalues = np.linalg.eigvals(X_T_X)
    print(f"\nEigenvalues of X^T X: {eigenvalues}")
    
    # Create a visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the relationship between features
    axes[0].scatter(x1, x2, s=80, alpha=0.7)
    axes[0].plot([0, 1], [0, 2], 'r--', linewidth=2, label='x2 = 2*x1')
    axes[0].set_xlabel('Feature x1', fontsize=12)
    axes[0].set_ylabel('Feature x2', fontsize=12)
    axes[0].set_title('Perfect Multicollinearity Between Features', fontsize=14)
    axes[0].legend()
    
    # Visualize the X^T X matrix
    im = axes[1].imshow(X_T_X, cmap='viridis')
    plt.colorbar(im, ax=axes[1])
    
    # Add grid lines and labels
    for i in range(X_T_X.shape[0] + 1):
        axes[1].axhline(i - 0.5, color='black', linewidth=1)
        axes[1].axvline(i - 0.5, color='black', linewidth=1)
    
    axes[1].set_xticks(np.arange(X_T_X.shape[1]))
    axes[1].set_yticks(np.arange(X_T_X.shape[0]))
    axes[1].set_xticklabels(['Intercept', 'x1', 'x2', 'x3'])
    axes[1].set_yticklabels(['Intercept', 'x1', 'x2', 'x3'])
    
    # Add text annotations
    for i in range(X_T_X.shape[0]):
        for j in range(X_T_X.shape[1]):
            axes[1].text(j, i, f'{X_T_X[i, j]:.2f}', 
                       ha='center', va='center', color='white')
    
    axes[1].set_title('X^T X Matrix with Multicollinearity', fontsize=14)
    
    # Print explanation instead of figtext
    print("\nMulticollinearity explanation:")
    print(f"With perfect multicollinearity, the X^T X matrix becomes singular.")
    print(f"Determinant ≈ {det_X_T_X:.2e}, which is effectively zero.")
    print(f"This means the matrix has no inverse, making least squares estimation impossible.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multicollinearity.png'), dpi=300)
    
    print(f"\nAnswer: singular\n")
    
    return fig

def demonstrate_dummy_variables():
    """Demonstrate how many dummy variables are needed for a categorical variable."""
    print("\n4. Number of Dummy Variables for Categorical Variables")
    print("--------------------------------------------------")
    
    # Create a sample categorical variable with k levels
    k = 4  # Number of levels
    categories = [f'Category {i+1}' for i in range(k)]
    
    # Create 10 random samples from these categories
    np.random.seed(42)
    n_samples = 10
    category_indices = np.random.randint(0, k, size=n_samples)
    category_values = [categories[i] for i in category_indices]
    
    # Create dummy variables (one-hot encoding)
    dummy_matrix = np.zeros((n_samples, k))
    for i, idx in enumerate(category_indices):
        dummy_matrix[i, idx] = 1
    
    # Remove one column to avoid multicollinearity (k-1 dummies)
    dummy_matrix_k_minus_1 = dummy_matrix[:, 1:]
    
    print(f"Original categorical variable with {k} levels (categories):")
    for i, cat in enumerate(category_values):
        print(f"Sample {i+1}: {cat}")
    
    print("\nFull one-hot encoding (k dummies, leading to multicollinearity):")
    print(dummy_matrix)
    
    print("\nProper dummy encoding (k-1 dummies, avoiding multicollinearity):")
    print(dummy_matrix_k_minus_1)
    print(f"Note: Category 1 is now the reference category")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Visualize the full one-hot encoding
    im1 = axes[0].imshow(dummy_matrix, cmap='Blues')
    plt.colorbar(im1, ax=axes[0])
    
    # Add grid lines
    for i in range(dummy_matrix.shape[0] + 1):
        axes[0].axhline(i - 0.5, color='black', linewidth=1)
    for j in range(dummy_matrix.shape[1] + 1):
        axes[0].axvline(j - 0.5, color='black', linewidth=1)
    
    # Add labels
    axes[0].set_xticks(np.arange(dummy_matrix.shape[1]))
    axes[0].set_yticks(np.arange(dummy_matrix.shape[0]))
    axes[0].set_xticklabels(categories)
    axes[0].set_yticklabels([f'Sample {i+1}' for i in range(n_samples)])
    
    axes[0].set_title('Full One-Hot Encoding (k dummies)', fontsize=14)
    
    # Visualize the k-1 dummy encoding
    im2 = axes[1].imshow(dummy_matrix_k_minus_1, cmap='Blues')
    plt.colorbar(im2, ax=axes[1])
    
    # Add grid lines
    for i in range(dummy_matrix_k_minus_1.shape[0] + 1):
        axes[1].axhline(i - 0.5, color='black', linewidth=1)
    for j in range(dummy_matrix_k_minus_1.shape[1] + 1):
        axes[1].axvline(j - 0.5, color='black', linewidth=1)
    
    # Add labels
    axes[1].set_xticks(np.arange(dummy_matrix_k_minus_1.shape[1]))
    axes[1].set_yticks(np.arange(dummy_matrix_k_minus_1.shape[0]))
    axes[1].set_xticklabels(categories[1:])
    axes[1].set_yticklabels([f'Sample {i+1}' for i in range(n_samples)])
    
    axes[1].set_title('Proper Dummy Encoding (k-1 dummies)', fontsize=14)
    
    # Print explanation instead of figtext
    print("\nDummy variables explanation:")
    print(f"For a categorical variable with k={k} levels, we typically create k-1={k-1} dummy variables.")
    print(f"Using all k dummies would cause perfect multicollinearity with the intercept term.")
    print(f"The omitted category becomes the reference/baseline category.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dummy_variables.png'), dpi=300)
    
    print(f"\nAnswer: k-1\n")
    
    return fig

def demonstrate_polynomial_regression():
    """Demonstrate polynomial regression of degree 3."""
    print("\n5. Polynomial Regression of Degree 3")
    print("---------------------------------")
    
    # Generate some sample data with non-linear relationship
    np.random.seed(42)
    n_samples = 50
    x = np.linspace(-3, 3, n_samples)
    
    # True coefficients for a cubic polynomial
    beta_true = [1, -0.5, 2, -0.3]  # [constant, x, x^2, x^3]
    
    # Generate y using the cubic polynomial with some noise
    y = beta_true[0] + beta_true[1] * x + beta_true[2] * x**2 + beta_true[3] * x**3 + np.random.randn(n_samples) * 1.5
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=True)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    
    # Fit the model using linear regression
    model = LinearRegression()
    model.fit(x_poly, y)
    
    # Get the coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Print the polynomial equation
    print(f"Polynomial regression of degree 3:")
    print(f"y = {intercept:.4f} + ({coefficients[1]:.4f})x + ({coefficients[2]:.4f})x^2 + ({coefficients[3]:.4f})x^3")
    
    # Calculate predictions
    x_plot = np.linspace(-3.5, 3.5, 100)
    x_plot_poly = poly.transform(x_plot.reshape(-1, 1))
    y_plot = model.predict(x_plot_poly)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot the data points
    ax.scatter(x, y, alpha=0.7, label='Data points')
    
    # Plot the polynomial regression curve
    ax.plot(x_plot, y_plot, 'r-', linewidth=2, label='Cubic polynomial fit')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Polynomial Regression of Degree 3', fontsize=14)
    ax.legend(fontsize=12)
    
    # Print equation instead of adding to the plot
    print(f"\nEquation: y = {intercept:.2f} + ({coefficients[1]:.2f})x + ({coefficients[2]:.2f})x^2 + ({coefficients[3]:.2f})x^3 + ε")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'polynomial_regression.png'), dpi=300)
    
    print(f"\nAnswer: β₀ + β₁x + β₂x² + β₃x³ + ε\n")
    
    return fig

def demonstrate_rbf():
    """Demonstrate a Gaussian radial basis function."""
    print("\n6. Gaussian Radial Basis Function")
    print("-------------------------------")
    
    # Define a Gaussian RBF
    def gaussian_rbf(x, center, gamma):
        return np.exp(-gamma * np.linalg.norm(x - center)**2)
    
    # Create a 2D grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Stack X and Y to create input points
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # RBF parameters
    center = np.array([0, 0])  # Center of the RBF
    gamma = 1.0  # Width parameter
    
    # Calculate RBF values
    Z = np.array([gaussian_rbf(point, center, gamma) for point in points])
    Z = Z.reshape(X.shape)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig)
    
    # 3D surface plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_zlabel('φ(x)', fontsize=12)
    ax1.set_title('Gaussian Radial Basis Function (3D)', fontsize=14)
    
    # 2D contour plot
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='φ(x) value')
    
    # Mark the center
    ax2.scatter([center[0]], [center[1]], color='red', s=100, marker='*', 
                label='Center (μ)')
    
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('Gaussian RBF Contour Plot', fontsize=14)
    ax2.legend(fontsize=10)
    
    # Print explanation instead of figtext
    print("\nGaussian RBF explanation:")
    print(f"φ(x) = exp(-γ||x-μ||²)")
    print(f"where γ={gamma} controls the width, and μ={center} is the center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gaussian_rbf.png'), dpi=300)
    
    print(f"Gaussian RBF: φ(x) = exp(-γ||x-μ||²)")
    print(f"where:")
    print(f"- γ is a parameter controlling the width of the RBF")
    print(f"- μ is the center of the RBF")
    print(f"- ||x-μ|| is the Euclidean distance between x and μ")
    
    print(f"\nAnswer: exp(-γ||x-μ||²)\n")
    
    return fig

def demonstrate_curse_of_dimensionality():
    """Demonstrate the curse of dimensionality in regression."""
    print("\n7. Curse of Dimensionality")
    print("------------------------")
    
    # Create a demonstration of how the number of samples needed grows exponentially
    dimensions = np.arange(1, 11)
    # Assuming we want a certain number of samples per unit length in each dimension
    samples_per_dim = 10
    total_samples = samples_per_dim ** dimensions
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the exponential growth of samples needed
    axes[0].semilogy(dimensions, total_samples, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Dimensions (d)', fontsize=12)
    axes[0].set_ylabel('Number of Samples Needed (log scale)', fontsize=12)
    axes[0].set_title('Exponential Growth of Required Samples', fontsize=14)
    axes[0].grid(True)
    
    # Annotate some points
    for i, dim in enumerate([1, 2, 3, 5, 10]):
        idx = dim - 1
        axes[0].annotate(f'd={dim}: {total_samples[idx]:,} samples', 
                     xy=(dim, total_samples[idx]), 
                     xytext=(dim+0.2, total_samples[idx]*1.2),
                     arrowprops=dict(arrowstyle='->'))
    
    # Create a visualization of the sparsity problem
    # For 1D, 2D, and 3D cases with fixed number of samples
    fixed_samples = 100
    
    # Plot for 1D case
    x1 = np.random.uniform(0, 1, fixed_samples)
    axes[1].plot(x1, np.zeros_like(x1), 'ro', alpha=0.7, label='1D (d=1)')
    
    # Plot for 2D case
    x2 = np.random.uniform(0, 1, (fixed_samples, 2))
    axes[1].plot(x2[:, 0], x2[:, 1], 'bx', alpha=0.7, label='2D (d=2)')
    
    # Create a rectangle to represent the space
    axes[1].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='k', linestyle='-'))
    
    axes[1].set_xlim(-0.1, 1.1)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel('Dimension 1', fontsize=12)
    axes[1].set_ylabel('Dimension 2', fontsize=12)
    axes[1].set_title('Data Sparsity with Fixed Sample Size', fontsize=14)
    axes[1].legend(fontsize=10)
    
    # Print the explanation instead of figtext
    print("\nCurse of dimensionality explanation:")
    print("The \"curse of dimensionality\" refers to problems that arise when the number of features (dimensions) increases:")
    print("1. The number of samples needed grows exponentially with dimensions")
    print("2. Data becomes increasingly sparse in the feature space")
    print("3. Distance metrics become less meaningful as dimensions increase")
    print("4. Risk of overfitting increases due to the increased model complexity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'curse_of_dimensionality.png'), dpi=300)
    
    print("The \"curse of dimensionality\" in regression refers to problems that arise when the number of features/dimensions becomes large relative to the number of samples.")
    
    print("\nAnswer: the number of features/dimensions becomes large relative to the number of samples\n")
    
    return fig

def demonstrate_matrix_prediction():
    """Demonstrate the matrix form of linear regression predictions."""
    print("\n8. Matrix Form for Linear Regression Predictions")
    print("---------------------------------------------")
    
    # Generate some example data
    np.random.seed(42)
    n_samples = 5
    n_features = 2
    
    # Create X and beta
    X = np.random.rand(n_samples, n_features)
    beta_true = np.array([3, 1.5, 2])  # [intercept, beta_1, beta_2]
    
    # Create design matrix with intercept
    X_design = np.column_stack((np.ones(n_samples), X))
    
    # Calculate predictions
    y_hat = X_design @ beta_true
    
    print("Design matrix X with intercept:")
    print(X_design)
    
    print("\nCoefficient vector beta:")
    print(beta_true)
    
    print("\nPredictions y_hat = X * beta:")
    print(y_hat)
    
    # Print LaTeX code for matrices that can be used in the markdown file
    print("\nLaTeX representation of the matrices for use in markdown:")
    
    # Design matrix X in LaTeX format
    print("\nDesign matrix X:")
    print("\\begin{bmatrix}")
    for i in range(X_design.shape[0]):
        row = " & ".join([f"{X_design[i, j]:.2f}" for j in range(X_design.shape[1])])
        if i < X_design.shape[0] - 1:
            print(row + " \\\\")
        else:
            print(row)
    print("\\end{bmatrix}")
    
    # Beta vector in LaTeX format
    print("\nCoefficient vector beta:")
    print("\\begin{bmatrix}")
    for i, val in enumerate(beta_true):
        if i < len(beta_true) - 1:
            print(f"{val:.2f} \\\\")
        else:
            print(f"{val:.2f}")
    print("\\end{bmatrix}")
    
    # Predictions vector in LaTeX format
    print("\nPredictions vector y_hat:")
    print("\\begin{bmatrix}")
    for i, val in enumerate(y_hat):
        if i < len(y_hat) - 1:
            print(f"{val:.2f} \\\\")
        else:
            print(f"{val:.2f}")
    print("\\end{bmatrix}")
    
    print("\nMatrix form equation:")
    print("ŷ = X β")
    print("Where:")
    print("- ŷ is the vector of predicted values")
    print("- X is the design matrix")
    print("- β is the vector of coefficients")
    
    print(f"\nAnswer: X β\n")
    
    return None

def demonstrate_multicollinearity_effects():
    """Visualize the effects of multicollinearity on coefficient estimates."""
    print("\n9. Effects of Multicollinearity on Coefficient Estimates")
    print("----------------------------------------------------")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data for the experiment
    n_samples = 100
    
    # Create a range of correlation values
    correlations = np.linspace(0, 0.99, 10)
    
    # Storage for coefficient variances
    coefficient_variances = []
    condition_numbers = []
    
    # For each correlation level
    for corr in correlations:
        # Create covariance matrix for 2 correlated predictors
        cov_matrix = np.array([[1, corr], [corr, 1]])
        
        # Generate correlated features
        X = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_samples)
        
        # Add a third uncorrelated feature
        X = np.column_stack((X, np.random.randn(n_samples)))
        
        # True coefficients
        beta_true = np.array([1.0, 1.0, 1.0])
        
        # Generate target with noise
        y = X @ beta_true + np.random.randn(n_samples) * 0.5
        
        # Add intercept
        X_design = np.column_stack((np.ones(n_samples), X))
        
        # Compute X^T X and its condition number
        X_T_X = X_design.T @ X_design
        eigvals = np.linalg.eigvals(X_T_X)
        condition_number = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))
        condition_numbers.append(condition_number)
        
        # Fit multiple models with bootstrap to assess coefficient variance
        n_bootstraps = 100
        coefficients = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_design[indices]
            y_boot = y[indices]
            
            # Fit model
            try:
                beta_hat = np.linalg.inv(X_boot.T @ X_boot) @ X_boot.T @ y_boot
                coefficients.append(beta_hat)
            except np.linalg.LinAlgError:
                # In case of singular matrix, skip this bootstrap
                continue
        
        # Convert to array and compute variance of coefficient for the first feature
        coefficients = np.array(coefficients)
        var_beta1 = np.var(coefficients[:, 1])
        coefficient_variances.append(var_beta1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Coefficient variance vs correlation
    axes[0].plot(correlations, coefficient_variances, 'bo-', linewidth=2)
    axes[0].set_xlabel('Correlation between predictors', fontsize=12)
    axes[0].set_ylabel('Variance of coefficient estimate', fontsize=12)
    axes[0].set_title('Effect of Multicollinearity on Coefficient Variance', fontsize=14)
    axes[0].grid(True)
    
    # Plot 2: Condition number vs correlation
    axes[1].semilogy(correlations, condition_numbers, 'ro-', linewidth=2)
    axes[1].set_xlabel('Correlation between predictors', fontsize=12)
    axes[1].set_ylabel('Condition number (log scale)', fontsize=12)
    axes[1].set_title('Condition Number as Indicator of Multicollinearity', fontsize=14)
    axes[1].grid(True)
    
    # Print explanation
    print("\nVisualization of multicollinearity effects:")
    print("1. As correlation between predictors increases, the variance of coefficient estimates increases")
    print("2. This makes coefficient estimates unstable and sensitive to small changes in the data")
    print("3. The condition number of X^T X rises dramatically as correlation approaches 1")
    print("4. A high condition number indicates numerical instability in solving the linear system")
    print("5. When perfect multicollinearity occurs, the condition number becomes infinite")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multicollinearity_effects.png'), dpi=300)
    
    return fig

def demonstrate_polynomial_bias_variance():
    """Visualize the bias-variance tradeoff in polynomial regression."""
    print("\n10. Bias-Variance Tradeoff in Polynomial Regression")
    print("-----------------------------------------------")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate true function: f(x) = sin(x)
    x_true = np.linspace(0, 10, 1000)
    y_true = np.sin(x_true)
    
    # Generate training data with noise
    n_samples = 20
    x_train = np.random.uniform(0, 10, n_samples)
    x_train = np.sort(x_train)  # Sort for easier visualization
    y_train = np.sin(x_train) + np.random.normal(0, 0.2, n_samples)
    
    # Generate test data for evaluation
    x_test = np.linspace(0, 10, 100)
    y_test = np.sin(x_test)
    
    # Fit polynomial models of different degrees
    max_degree = 15
    degrees = range(1, max_degree + 1)
    train_errors = []
    test_errors = []
    
    # Storage for different model predictions
    predictions = {}
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
        x_test_poly = poly.transform(x_test.reshape(-1, 1))
        
        # Fit model
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(x_train_poly)
        y_test_pred = model.predict(x_test_poly)
        
        # Store predictions for selected degrees
        if degree in [1, 3, 10, 15]:
            predictions[degree] = y_test_pred
        
        # Calculate errors
        train_error = np.mean((y_train - y_train_pred) ** 2)
        test_error = np.mean((y_test - y_test_pred) ** 2)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Selected polynomial fits
    highlight_degrees = [1, 3, 10, 15]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, degree in enumerate(highlight_degrees):
        axes[0, 0].plot(x_test, predictions[degree], color=colors[i], 
                      linewidth=2, label=f'Degree {degree}')
    
    # Plot the true function and training data
    axes[0, 0].plot(x_true, y_true, 'k--', linewidth=1.5, label='True function')
    axes[0, 0].scatter(x_train, y_train, color='black', s=30, alpha=0.7, label='Training data')
    
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('y', fontsize=12)
    axes[0, 0].set_title('Polynomial Regression Models of Different Degrees', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training and test error vs. polynomial degree
    axes[0, 1].plot(degrees, train_errors, 'b-', linewidth=2, label='Training error')
    axes[0, 1].plot(degrees, test_errors, 'r-', linewidth=2, label='Test error')
    axes[0, 1].set_xlabel('Polynomial Degree', fontsize=12)
    axes[0, 1].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0, 1].set_title('Training and Test Error vs. Model Complexity', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Detailed visualization of underfitting
    axes[1, 0].plot(x_true, y_true, 'k--', linewidth=1.5, label='True function')
    axes[1, 0].scatter(x_train, y_train, color='black', s=30, alpha=0.7, label='Training data')
    axes[1, 0].plot(x_test, predictions[1], color='blue', linewidth=2, label='Degree 1 (Linear)')
    
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('y', fontsize=12)
    axes[1, 0].set_title('Underfitting with Linear Model (High Bias)', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Detailed visualization of overfitting
    axes[1, 1].plot(x_true, y_true, 'k--', linewidth=1.5, label='True function')
    axes[1, 1].scatter(x_train, y_train, color='black', s=30, alpha=0.7, label='Training data')
    axes[1, 1].plot(x_test, predictions[15], color='red', linewidth=2, label='Degree 15 (Complex)')
    
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('y', fontsize=12)
    axes[1, 1].set_title('Overfitting with High-Degree Polynomial (High Variance)', fontsize=14)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Print explanation
    print("\nBias-Variance Tradeoff in Polynomial Regression:")
    print("1. Low-degree models (like linear) show high bias (underfitting)")
    print("   - They fail to capture the complexity of the underlying function")
    print("   - Both training and test errors are high")
    print("\n2. High-degree models show high variance (overfitting)")
    print("   - They capture noise in the training data")
    print("   - Training error is very low but test error increases")
    print("\n3. The optimal model complexity (polynomial degree) balances bias and variance")
    print("   - It minimizes test error while maintaining reasonable training error")
    print("\n4. This illustrates the fundamental bias-variance tradeoff in machine learning:")
    print("   - Simple models: high bias, low variance")
    print("   - Complex models: low bias, high variance")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'polynomial_bias_variance.png'), dpi=300)
    
    return fig

def main():
    """Run all demonstrations and generate a complete solution."""
    print("# Question 15: Matrix Form and Properties of Linear Regression")
    print("## Step-by-Step Solution\n")
    
    # Run all demonstrations
    plot_design_matrix()
    demonstrate_closed_form_solution()
    demonstrate_multicollinearity()
    demonstrate_dummy_variables()
    demonstrate_polynomial_regression()
    demonstrate_rbf()
    demonstrate_curse_of_dimensionality()
    demonstrate_matrix_prediction()
    demonstrate_multicollinearity_effects()
    demonstrate_polynomial_bias_variance()
    
    print("\n## Summary of Answers:")
    print("1. In a multiple linear regression model with d features, the design matrix X has dimensions n × (d+1).")
    print("2. The closed-form solution to the least squares problem in matrix form is given by (X^T X)^(-1) X^T y.")
    print("3. When there is perfect multicollinearity among predictors, the matrix X^T X becomes singular.")
    print("4. If a categorical variable has k levels, we typically create k-1 dummy variables to represent it.")
    print("5. A polynomial regression model of degree 3 with a single input variable x can be written as y = β₀ + β₁x + β₂x² + β₃x³ + ε.")
    print("6. A Gaussian radial basis function can be expressed as φ(x) = exp(-γ||x-μ||²).")
    print("7. The \"curse of dimensionality\" in regression refers to problems that arise when the number of features/dimensions becomes large relative to the number of samples.")
    print("8. In matrix form, the predictions of a linear regression model can be written as ŷ = X β.")
    print("9. The effects of multicollinearity on coefficient estimates are visualized in the multicollinearity_effects function.")
    print("10. The bias-variance tradeoff in polynomial regression is visualized in the polynomial_bias_variance function.")

if __name__ == "__main__":
    main() 