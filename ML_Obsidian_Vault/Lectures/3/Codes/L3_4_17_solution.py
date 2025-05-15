import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import os
from scipy.special import binom
import time
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

# Part 1: Define basis functions
print_section_header("Part 1: Defining Basis Functions")

def polynomial_basis(x, degree):
    """
    Polynomial basis functions.
    
    Parameters:
    -----------
    x : array-like
        Input data.
    degree : int
        Degree of the polynomial.
        
    Returns:
    --------
    array-like
        Transformed features.
    """
    if np.isscalar(x) or len(x.shape) == 1:
        x = np.array(x).reshape(-1, 1)
    
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(x)

def gaussian_rbf(x, centers, width=1.0):
    """
    Gaussian Radial Basis Functions.
    
    Parameters:
    -----------
    x : array-like
        Input data.
    centers : array-like
        Centers of the Gaussian functions.
    width : float
        Width parameter (sigma) of the Gaussian functions.
        
    Returns:
    --------
    array-like
        Transformed features.
    """
    if np.isscalar(x):
        x = np.array([x])
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    # Add bias term
    n_samples = x.shape[0]
    phi = np.ones((n_samples, len(centers) + 1))
    
    for i, center in enumerate(centers):
        dist = np.sum((x - center) ** 2, axis=1)
        phi[:, i+1] = np.exp(-dist / (2 * width ** 2))
    
    return phi

def sigmoid_basis(x, centers, scaling=1.0):
    """
    Sigmoid basis functions.
    
    Parameters:
    -----------
    x : array-like
        Input data.
    centers : array-like
        Centers (thresholds) of the sigmoid functions.
    scaling : float
        Scaling factor affecting the steepness of the sigmoid.
        
    Returns:
    --------
    array-like
        Transformed features.
    """
    if np.isscalar(x):
        x = np.array([x])
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    # Add bias term
    n_samples = x.shape[0]
    phi = np.ones((n_samples, len(centers) + 1))
    
    for i, center in enumerate(centers):
        phi[:, i+1] = 1 / (1 + np.exp(-scaling * (x[:, 0] - center)))
    
    return phi

# New: Define analytical gradient functions for each basis type
def poly_gradient(x, degree):
    """Calculate gradients of polynomial basis functions."""
    gradients = []
    for d in range(degree + 1):
        if d == 0:
            # Gradient of constant term is 0
            gradients.append(0)
        else:
            # Gradient of x^d is d*x^(d-1)
            gradients.append(d * x**(d-1))
    return gradients

def rbf_gradient(x, center, width):
    """Calculate gradient of a Gaussian RBF w.r.t. x."""
    # ∂φ(x)/∂x = φ(x) * (-(x-center)/(width^2))
    phi_x = np.exp(-(x - center)**2 / (2 * width**2))
    grad = phi_x * (-(x - center) / width**2)
    return grad

def sigmoid_gradient(x, center, scaling):
    """Calculate gradient of a sigmoid basis function w.r.t. x."""
    # ∂φ(x)/∂x = φ(x) * (1 - φ(x)) * scaling
    phi_x = 1 / (1 + np.exp(-scaling * (x - center)))
    grad = phi_x * (1 - phi_x) * scaling
    return grad

print("Basis functions are non-linear transformations of input features that allow linear models")
print("to capture non-linear relationships in the data. By projecting the original features into")
print("a higher-dimensional space using basis functions, we can apply linear methods to solve")
print("non-linear problems.")
print("\nWe've defined three types of basis functions:")
print("1. Polynomial basis functions")
print("2. Gaussian radial basis functions")
print("3. Sigmoid basis functions")

# Part 1.5: Analytical properties of basis functions
print_section_header("Part 1.5: Analytical Properties of Basis Functions")

# Define a point to evaluate gradients
x_point = 1.0
print(f"Evaluating basis function gradients at x = {x_point}")

# Calculate and print polynomial gradients
degree = 3
poly_grads = poly_gradient(x_point, degree)
print(f"\nPolynomial basis function gradients (degree {degree}):")
for d, grad in enumerate(poly_grads):
    print(f"  ∂φ_{d}(x)/∂x = ∂(x^{d})/∂x = {grad}")

# Calculate and print RBF gradients
rbf_center = 2.0
rbf_width = 1.5
rbf_grad = rbf_gradient(x_point, rbf_center, rbf_width)
phi_rbf = np.exp(-(x_point - rbf_center)**2 / (2 * rbf_width**2))
print(f"\nGaussian RBF gradient at center = {rbf_center}, width = {rbf_width}:")
print(f"  φ(x) = exp(-(x-{rbf_center})²/(2*{rbf_width}²)) = {phi_rbf:.6f}")
print(f"  ∂φ(x)/∂x = φ(x) * (-(x-center)/width²) = {rbf_grad:.6f}")

# Calculate and print sigmoid gradients
sigmoid_center = 0.5
sigmoid_scaling = 2.0
sigmoid_grad = sigmoid_gradient(x_point, sigmoid_center, sigmoid_scaling)
phi_sigmoid = 1 / (1 + np.exp(-sigmoid_scaling * (x_point - sigmoid_center)))
print(f"\nSigmoid gradient at center = {sigmoid_center}, scaling = {sigmoid_scaling}:")
print(f"  φ(x) = 1/(1+exp(-{sigmoid_scaling}*(x-{sigmoid_center}))) = {phi_sigmoid:.6f}")
print(f"  ∂φ(x)/∂x = φ(x) * (1 - φ(x)) * scaling = {sigmoid_grad:.6f}")

# Calculate number of basis functions for polynomial model
print("\nNumber of basis functions in polynomial models of different degrees:")
input_dims = [1, 2, 3, 5, 10]
degrees = [1, 2, 3, 5, 10]

table_data = []
for n in input_dims:
    row = [n]
    for d in degrees:
        # Number of basis functions (including constant term)
        # Formula: binom(n+d, d) = (n+d)! / (n! * d!)
        num_basis = binom(n+d, d)
        row.append(int(num_basis))
    table_data.append(row)

df = pd.DataFrame(table_data, 
                  columns=['Input Dim'] + [f'Degree {d}' for d in degrees])
print(df.to_string(index=False))

# Part 2: Visualize different basis functions
print_section_header("Part 2: Visualizing Basis Functions")

# Generate data
x = np.linspace(-5, 5, 500)

# Define centers and parameters
rbf_centers = np.array([-3, -1, 0, 2, 4]).reshape(-1, 1)
sigmoid_centers = np.array([-3, -1, 0, 2, 4])

# Calculate transformed features
poly_features = polynomial_basis(x, degree=3)
rbf_features = gaussian_rbf(x, rbf_centers, width=1.0)
sigmoid_features = sigmoid_basis(x, sigmoid_centers, scaling=2.0)

# Create visualizations
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 2, figure=fig)

# Plot 1: Polynomial basis functions
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, poly_features[:, 1], label=r'$\phi_1(x) = x$')
ax1.plot(x, poly_features[:, 2], label=r'$\phi_2(x) = x^2$')
ax1.plot(x, poly_features[:, 3], label=r'$\phi_3(x) = x^3$')
ax1.set_title('Polynomial Basis Functions', fontsize=14)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel(r'$\phi_i(x)$', fontsize=12)
ax1.legend()
ax1.grid(True)

# Plot 2: Example of polynomial regression
ax2 = fig.add_subplot(gs[0, 1])
# Generate data with non-linear pattern
np.random.seed(42)
x_data = np.random.uniform(-4, 4, 40)
y_data = 0.5 + 1.5 * x_data - 0.8 * x_data**2 + 0.2 * x_data**3 + np.random.normal(0, 1, 40)

# Fit polynomial regression models of different degrees
x_plot = np.linspace(-4.5, 4.5, 100).reshape(-1, 1)
degrees = [1, 3, 10]
for degree in degrees:
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x_data.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y_data)
    
    x_plot_poly = poly.transform(x_plot)
    y_plot = model.predict(x_plot_poly)
    ax2.plot(x_plot, y_plot, label=f'Degree {degree}')

ax2.scatter(x_data, y_data, color='black', alpha=0.5, label='Data')
ax2.set_title('Polynomial Regression with Different Degrees', fontsize=14)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.legend()
ax2.grid(True)
ax2.set_ylim(-15, 15)

# Plot 3: Gaussian RBF basis functions
ax3 = fig.add_subplot(gs[1, 0])
for i in range(1, rbf_features.shape[1]):
    ax3.plot(x, rbf_features[:, i], label=f'RBF at {rbf_centers[i-1][0]}')

# Add vertical lines for centers
for center in rbf_centers:
    ax3.axvline(x=center, color='gray', linestyle='--', alpha=0.5)

ax3.set_title('Gaussian Radial Basis Functions', fontsize=14)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel(r'$\phi_i(x)$', fontsize=12)
ax3.legend()
ax3.grid(True)

# Plot 4: Example of RBF regression
ax4 = fig.add_subplot(gs[1, 1])
# Generate RBF centers for regression
rbf_reg_centers = np.linspace(-4, 4, 8).reshape(-1, 1)

# Transform data
x_rbf = gaussian_rbf(x_data.reshape(-1, 1), rbf_reg_centers, width=1.5)
x_plot_rbf = gaussian_rbf(x_plot, rbf_reg_centers, width=1.5)

# Fit model
model = LinearRegression()
model.fit(x_rbf, y_data)
y_plot = model.predict(x_plot_rbf)

ax4.scatter(x_data, y_data, color='black', alpha=0.5, label='Data')
ax4.plot(x_plot, y_plot, color='red', label='RBF Regression')

# Add RBF components
for i in range(1, x_plot_rbf.shape[1]):
    component = model.coef_[i-1] * x_plot_rbf[:, i]
    ax4.plot(x_plot, component, linestyle='--', alpha=0.5, label=f'Component {i}' if i <= 3 else None)

ax4.set_title('Regression with Gaussian RBF Basis Functions', fontsize=14)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.legend(loc='upper left')
ax4.grid(True)

# Plot 5: Sigmoid basis functions
ax5 = fig.add_subplot(gs[2, 0])
for i in range(1, sigmoid_features.shape[1]):
    ax5.plot(x, sigmoid_features[:, i], label=f'Sigmoid at {sigmoid_centers[i-1]}')

# Add vertical lines for centers
for center in sigmoid_centers:
    ax5.axvline(x=center, color='gray', linestyle='--', alpha=0.5)

ax5.set_title('Sigmoid Basis Functions', fontsize=14)
ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel(r'$\phi_i(x)$', fontsize=12)
ax5.legend()
ax5.grid(True)

# Plot 6: Example of Sigmoid regression
ax6 = fig.add_subplot(gs[2, 1])
# Generate sigmoid centers for regression
sigmoid_reg_centers = np.linspace(-4, 4, 8)

# Transform data
x_sigmoid = sigmoid_basis(x_data.reshape(-1, 1), sigmoid_reg_centers, scaling=2.0)
x_plot_sigmoid = sigmoid_basis(x_plot, sigmoid_reg_centers, scaling=2.0)

# Fit model
model = LinearRegression()
model.fit(x_sigmoid, y_data)
y_plot = model.predict(x_plot_sigmoid)

ax6.scatter(x_data, y_data, color='black', alpha=0.5, label='Data')
ax6.plot(x_plot, y_plot, color='green', label='Sigmoid Regression')

ax6.set_title('Regression with Sigmoid Basis Functions', fontsize=14)
ax6.set_xlabel('x', fontsize=12)
ax6.set_ylabel('y', fontsize=12)
ax6.legend()
ax6.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'basis_functions_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations of different basis functions and their use in regression have been created.")
print("Figure saved as 'basis_functions_visualization.png'")

# Part 3: Quadratic model with 2D input
print_section_header("Part 3: Quadratic Model with 2D Input")

print("For a dataset with input features x ∈ ℝ² and a quadratic model, we need the following basis functions:")
print("1. Constant term: φ₀(x) = 1")
print("2. Linear terms: φ₁(x) = x₁, φ₂(x) = x₂")
print("3. Quadratic terms: φ₃(x) = x₁², φ₄(x) = x₁x₂, φ₅(x) = x₂²")
print("\nLet's demonstrate this with a 2D example:")

# Generate 2D data
np.random.seed(42)
n_samples = 100
X = np.random.uniform(-3, 3, (n_samples, 2))
# True function: f(x) = 1 + 2*x₁ - x₂ + 0.5*x₁² + 2*x₁*x₂ - 1.5*x₂²
true_w0 = 1.0
true_w1 = 2.0
true_w2 = -1.0
true_w3 = 0.5
true_w4 = 2.0
true_w5 = -1.5

y_true = true_w0 + true_w1*X[:, 0] + true_w2*X[:, 1] + true_w3*X[:, 0]**2 + true_w4*X[:, 0]*X[:, 1] + true_w5*X[:, 1]**2
y = y_true + np.random.normal(0, 1, n_samples)

# Transform features and fit model
quad = PolynomialFeatures(degree=2)
X_quad = quad.fit_transform(X)
feature_names = quad.get_feature_names_out(['x₁', 'x₂'])

print("\nThe transformed features are:")
for i, name in enumerate(feature_names):
    print(f"φ_{i}(x) = {name}")

# Fit model
model = LinearRegression()
model.fit(X_quad, y)

print("\nThe fitted model weights are:")
for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
    print(f"w_{i} (for {name}) = {coef:.4f}")
print(f"Intercept (w_0) = {model.intercept_:.4f}")

# Print true vs. fitted coefficients
print("\nComparison of true vs. fitted coefficients:")
true_coefs = [true_w0, true_w1, true_w2, true_w3, true_w4, true_w5]
fitted_coefs = [model.intercept_] + list(model.coef_[1:6])  # Skip the constant term from coef_
for i, (true_coef, fitted_coef) in enumerate(zip(true_coefs, fitted_coefs)):
    if i == 0:
        name = "Intercept"
    else:
        name = feature_names[i]
    print(f"{name}: True = {true_coef:.4f}, Fitted = {fitted_coef:.4f}, Difference = {fitted_coef - true_coef:.4f}")

# Create visualization of the model surface
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: 3D surface of true function
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
z_true = true_w0 + true_w1*X_grid[:, 0] + true_w2*X_grid[:, 1] + true_w3*X_grid[:, 0]**2 + true_w4*X_grid[:, 0]*X_grid[:, 1] + true_w5*X_grid[:, 1]**2
z_true = z_true.reshape(x1_grid.shape)

surf1 = ax1.plot_surface(x1_grid, x2_grid, z_true, cmap='viridis', alpha=0.8)
ax1.scatter(X[:, 0], X[:, 1], y, color='red', alpha=0.5)
ax1.set_title('True Quadratic Function', fontsize=14)
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_zlabel('y', fontsize=12)

# Plot 2: 3D surface of predicted function
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
X_grid_quad = quad.transform(X_grid)
z_pred = model.predict(X_grid_quad).reshape(x1_grid.shape)

surf2 = ax2.plot_surface(x1_grid, x2_grid, z_pred, cmap='plasma', alpha=0.8)
ax2.scatter(X[:, 0], X[:, 1], y, color='red', alpha=0.5)
ax2.set_title('Fitted Quadratic Model', fontsize=14)
ax2.set_xlabel('x₁', fontsize=12)
ax2.set_ylabel('x₂', fontsize=12)
ax2.set_zlabel('y', fontsize=12)

# Plot 3: Contour plot of true function
ax3 = fig.add_subplot(gs[1, 0])
contour1 = ax3.contourf(x1_grid, x2_grid, z_true, 50, cmap='viridis')
plt.colorbar(contour1, ax=ax3)
ax3.scatter(X[:, 0], X[:, 1], color='red', alpha=0.5)
ax3.set_title('Contour Plot of True Function', fontsize=14)
ax3.set_xlabel('x₁', fontsize=12)
ax3.set_ylabel('x₂', fontsize=12)

# Plot 4: Contour plot of predicted function
ax4 = fig.add_subplot(gs[1, 1])
contour2 = ax4.contourf(x1_grid, x2_grid, z_pred, 50, cmap='plasma')
plt.colorbar(contour2, ax=ax4)
ax4.scatter(X[:, 0], X[:, 1], color='red', alpha=0.5)
ax4.set_title('Contour Plot of Fitted Model', fontsize=14)
ax4.set_xlabel('x₁', fontsize=12)
ax4.set_ylabel('x₂', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'quadratic_model_2d.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization of the quadratic model with 2D input has been created.")
print("Figure saved as 'quadratic_model_2d.png'")

# Part 3.5: Computational Efficiency Comparison
print_section_header("Part 3.5: Computational Efficiency Comparison")

# Generate larger dataset for timing comparison
np.random.seed(42)
n_timing_samples = 10000
X_timing = np.random.uniform(-5, 5, (n_timing_samples, 1))
y_timing = np.sin(X_timing.ravel()) + 0.1 * np.random.randn(n_timing_samples)

# Define parameter ranges
poly_degrees = [1, 2, 3, 5, 10]
rbf_n_centers = [5, 10, 20, 50, 100]
sigmoid_n_centers = [5, 10, 20, 50, 100]

# Function to time the transformation and fitting
def time_model(transform_func, fit_func, name):
    transform_time = time.time()
    X_transformed = transform_func()
    transform_time = time.time() - transform_time
    
    fit_time = time.time()
    model = fit_func(X_transformed)
    fit_time = time.time() - fit_time
    
    # Calculate memory usage (approximate)
    memory_mb = X_transformed.nbytes / (1024 * 1024)
    
    return {
        'name': name,
        'transform_time': transform_time,
        'fit_time': fit_time,
        'total_time': transform_time + fit_time,
        'memory_mb': memory_mb,
        'n_features': X_transformed.shape[1]
    }

# Time polynomial transformations
poly_results = []
for degree in poly_degrees:
    result = time_model(
        lambda: PolynomialFeatures(degree).fit_transform(X_timing),
        lambda X: LinearRegression().fit(X, y_timing),
        f'Polynomial (degree={degree})'
    )
    poly_results.append(result)

# Time RBF transformations 
rbf_results = []
for n_centers in rbf_n_centers:
    centers = np.linspace(-5, 5, n_centers).reshape(-1, 1)
    result = time_model(
        lambda: gaussian_rbf(X_timing, centers, width=1.0),
        lambda X: LinearRegression().fit(X, y_timing),
        f'RBF (centers={n_centers})'
    )
    rbf_results.append(result)

# Time sigmoid transformations
sigmoid_results = []
for n_centers in sigmoid_n_centers:
    centers = np.linspace(-5, 5, n_centers)
    result = time_model(
        lambda: sigmoid_basis(X_timing, centers, scaling=2.0),
        lambda X: LinearRegression().fit(X, y_timing),
        f'Sigmoid (centers={n_centers})'
    )
    sigmoid_results.append(result)

# Combine results
all_results = poly_results + rbf_results + sigmoid_results
results_df = pd.DataFrame(all_results)

# Print results as a table
print(f"Computational efficiency comparison (dataset size: {n_timing_samples} samples)")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Visualize computation time vs. number of features
plt.figure(figsize=(12, 6))
plt.plot([r['n_features'] for r in poly_results], [r['total_time'] for r in poly_results], 'bo-', label='Polynomial')
plt.plot([r['n_features'] for r in rbf_results], [r['total_time'] for r in rbf_results], 'ro-', label='RBF')
plt.plot([r['n_features'] for r in sigmoid_results], [r['total_time'] for r in sigmoid_results], 'go-', label='Sigmoid')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Computation Time (seconds)', fontsize=12)
plt.title('Computation Time vs. Number of Features', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'computation_time.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualize memory usage vs. number of features
plt.figure(figsize=(12, 6))
plt.plot([r['n_features'] for r in poly_results], [r['memory_mb'] for r in poly_results], 'bo-', label='Polynomial')
plt.plot([r['n_features'] for r in rbf_results], [r['memory_mb'] for r in rbf_results], 'ro-', label='RBF')
plt.plot([r['n_features'] for r in sigmoid_results], [r['memory_mb'] for r in sigmoid_results], 'go-', label='Sigmoid')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Memory Usage (MB)', fontsize=12)
plt.title('Memory Usage vs. Number of Features', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nComputational efficiency visualizations have been created.")
print("Figures saved as 'computation_time.png' and 'memory_usage.png'")

# Part 4: Bias-Variance Tradeoff
print_section_header("Part 4: Bias-Variance Tradeoff with Different Basis Functions")

print("The choice of basis functions directly affects the bias-variance tradeoff in the model:")
print("- Too few basis functions or too simple basis functions lead to high bias (underfitting)")
print("- Too many basis functions or too flexible basis functions lead to high variance (overfitting)")
print("\nLet's demonstrate this with an example using polynomial basis functions of different degrees:")

# Generate data
np.random.seed(42)
x = np.linspace(0, 1, 30)
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, size=len(x))

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Define degrees to test
degrees = [1, 3, 5, 9, 15]
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Create figure
plt.figure(figsize=(15, 10))

# Plot data
plt.scatter(X_train, y_train, color='black', alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, color='gray', alpha=0.6, label='Test data')
plt.plot(x, y_true, 'k--', label='True function')

# Fit and plot models with different degrees
train_errors = []
test_errors = []

for i, degree in enumerate(degrees):
    # Transform data
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    # Plot predictions
    x_plot = np.linspace(0, 1, 100)
    x_plot_poly = poly.transform(x_plot.reshape(-1, 1))
    y_plot = model.predict(x_plot_poly)
    
    plt.plot(x_plot, y_plot, color=colors[i], label=f'Degree {degree}')

plt.title('Polynomial Regression with Different Degrees', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'bias_variance_fits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot train vs. test error
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'bo-', label='Training Error')
plt.plot(degrees, test_errors, 'ro-', label='Test Error')
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Bias-Variance Tradeoff: Error vs. Model Complexity', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a table of results
bias_variance_table = pd.DataFrame({
    'Degree': degrees,
    'Training Error': train_errors,
    'Test Error': test_errors,
    'Difference': np.array(test_errors) - np.array(train_errors)
})
print("\nBias-Variance Tradeoff Results:")
print(bias_variance_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nAs the degree of polynomial basis functions increases:")
print("- Training error consistently decreases")
print("- Test error initially decreases (reducing bias)")
print("- But beyond a certain complexity, test error increases (due to increasing variance)")
print("\nThis demonstrates the classic bias-variance tradeoff in machine learning")
print("\nFigures saved as 'bias_variance_fits.png' and 'bias_variance_tradeoff.png'")

# Part 4.5: Cross-Validation for Optimal Model Selection
print_section_header("Part 4.5: Cross-Validation for Optimal Model Selection")

print("We'll use cross-validation to find the optimal model complexity:")

# Generate new dataset
np.random.seed(456)
n_samples = 100
x_cv = np.linspace(-3, 3, n_samples)
y_cv_true = 0.5 * np.sin(np.pi * x_cv) + 0.5 * x_cv + 0.2 * x_cv**2
y_cv = y_cv_true + np.random.normal(0, 0.3, size=n_samples)
X_cv = x_cv.reshape(-1, 1)

# Define models to test
cv_degrees = list(range(1, 21))  # Polynomial degrees 1 to 20
cv_scores = []
cv_scores_std = []

# Perform cross-validation for each degree
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for degree in cv_degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_cv)
    
    # Cross-validate
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y_cv, cv=kf, scoring='neg_mean_squared_error')
    
    # Store results (convert to positive MSE)
    cv_scores.append(-scores.mean())
    cv_scores_std.append(scores.std())

# Find the best degree
best_degree_idx = np.argmin(cv_scores)
best_degree = cv_degrees[best_degree_idx]
print(f"Cross-validation results for polynomial degrees 1-20:")
for i, (degree, score, std) in enumerate(zip(cv_degrees, cv_scores, cv_scores_std)):
    if i == best_degree_idx:
        print(f"Degree {degree}: MSE = {score:.6f} ± {std:.6f} (BEST)")
    else:
        print(f"Degree {degree}: MSE = {score:.6f} ± {std:.6f}")

# Plot cross-validation results
plt.figure(figsize=(12, 6))
plt.errorbar(cv_degrees, cv_scores, yerr=cv_scores_std, fmt='o-')
plt.axvline(x=best_degree, color='r', linestyle='--', label=f'Best degree: {best_degree}')
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Mean Squared Error (CV)', fontsize=12)
plt.title('Cross-Validation Results: Error vs. Model Complexity', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'cross_validation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Train final model with best degree
best_poly = PolynomialFeatures(best_degree)
X_best_poly = best_poly.fit_transform(X_cv)
best_model = LinearRegression().fit(X_best_poly, y_cv)

# Generate predictions for plotting
x_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
x_plot_poly = best_poly.transform(x_plot)
y_plot_pred = best_model.predict(x_plot_poly)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(x_cv, y_cv, alpha=0.6, label='Data')
plt.plot(x_plot, y_plot_pred, 'r-', linewidth=2, label=f'Best model (degree {best_degree})')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Optimal Model Selected via Cross-Validation (Degree {best_degree})', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'optimal_model.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nThe optimal polynomial degree determined by cross-validation is {best_degree}")
print("Figures saved as 'cross_validation.png' and 'optimal_model.png'")

# Part 5: Compare different basis functions on the same problem
print_section_header("Part 5: Comparing Different Basis Functions")

print("Let's compare how different basis functions perform on the same problem:")

# Generate data
np.random.seed(45)
x = np.linspace(-3, 3, 40)
y_true = 0.5 * np.sin(np.pi * x) + 0.5 * x + 0.2 * x**2
y = y_true + np.random.normal(0, 0.2, size=len(x))

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)

# Define models
models = {
    'Linear': PolynomialFeatures(degree=1),
    'Quadratic': PolynomialFeatures(degree=2),
    'Cubic': PolynomialFeatures(degree=3),
    'Gaussian RBF': {
        'centers': np.linspace(-3, 3, 7).reshape(-1, 1),
        'width': 1.0
    },
    'Sigmoid': {
        'centers': np.linspace(-3, 3, 7),
        'scaling': 2.0
    }
}

# Create figure
plt.figure(figsize=(15, 10))

# Plot data
plt.scatter(X_train, y_train, color='black', alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, color='gray', alpha=0.6, label='Test data')
plt.plot(np.sort(x), y_true[np.argsort(x)], 'k--', label='True function')

# Fit and plot models
results = {}
x_plot = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)

for name, model_config in models.items():
    # Transform data based on model type
    if name in ['Linear', 'Quadratic', 'Cubic']:
        X_train_transformed = model_config.fit_transform(X_train_reshaped)
        X_test_transformed = model_config.transform(X_test_reshaped)
        X_plot_transformed = model_config.transform(x_plot)
    elif name == 'Gaussian RBF':
        X_train_transformed = gaussian_rbf(X_train_reshaped, model_config['centers'], model_config['width'])
        X_test_transformed = gaussian_rbf(X_test_reshaped, model_config['centers'], model_config['width'])
        X_plot_transformed = gaussian_rbf(x_plot, model_config['centers'], model_config['width'])
    elif name == 'Sigmoid':
        X_train_transformed = sigmoid_basis(X_train_reshaped, model_config['centers'], model_config['scaling'])
        X_test_transformed = sigmoid_basis(X_test_reshaped, model_config['centers'], model_config['scaling'])
        X_plot_transformed = sigmoid_basis(x_plot, model_config['centers'], model_config['scaling'])
    
    # Fit linear model on transformed features
    lr = LinearRegression()
    lr.fit(X_train_transformed, y_train)
    
    # Make predictions
    y_train_pred = lr.predict(X_train_transformed)
    y_test_pred = lr.predict(X_test_transformed)
    y_plot = lr.predict(X_plot_transformed)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Store results
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'n_features': X_train_transformed.shape[1]
    }
    
    # Plot predictions
    plt.plot(x_plot, y_plot, label=f'{name} (Test MSE: {test_mse:.4f})')

plt.title('Comparison of Different Basis Functions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'basis_functions_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create table of results
results_table = pd.DataFrame([
    {
        'Model': name,
        'Features': info['n_features'],
        'Train MSE': info['train_mse'],
        'Test MSE': info['test_mse'],
        'Gap': info['test_mse'] - info['train_mse']
    }
    for name, info in results.items()
])
print("\nComparison of different basis function types:")
print(results_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Create bar chart of errors
plt.figure(figsize=(12, 6))
models_list = list(results.keys())
train_errors = [results[model]['train_mse'] for model in models_list]
test_errors = [results[model]['test_mse'] for model in models_list]

x_pos = np.arange(len(models_list))
width = 0.35

plt.bar(x_pos - width/2, train_errors, width, label='Training Error')
plt.bar(x_pos + width/2, test_errors, width, label='Test Error')

plt.xlabel('Basis Function Type', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Error Comparison Across Different Basis Functions', fontsize=14)
plt.xticks(x_pos, models_list)
plt.legend()
plt.grid(True, axis='y')
plt.savefig(os.path.join(save_dir, 'basis_functions_errors.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nWe've compared different basis functions on the same regression problem:")
for name, result in results.items():
    print(f"- {name}: Training MSE = {result['train_mse']:.4f}, Test MSE = {result['test_mse']:.4f}")

print("\nFigures saved as 'basis_functions_comparison.png' and 'basis_functions_errors.png'")

# Part 6: Regularization with Basis Functions
print_section_header("Part 6: Regularization with Basis Functions")

print("When using many basis functions, regularization helps prevent overfitting.")
print("Let's demonstrate this with polynomial basis functions and ridge regression.")

# Generate data with noise
np.random.seed(123)
n_reg_samples = 50
x_reg = np.random.uniform(-3, 3, n_reg_samples)
X_reg = x_reg.reshape(-1, 1)
y_reg_true = 0.5 * np.sin(np.pi * x_reg) + 0.5 * x_reg
y_reg = y_reg_true + np.random.normal(0, 0.5, size=n_reg_samples)

# Split data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Use high-degree polynomial basis functions
reg_degree = 15
poly_reg = PolynomialFeatures(reg_degree)
X_reg_train_poly = poly_reg.fit_transform(X_reg_train)
X_reg_test_poly = poly_reg.transform(X_reg_test)

# Try different regularization strengths
alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
reg_results = []

plt.figure(figsize=(15, 10))
plt.scatter(X_reg_train, y_reg_train, color='black', alpha=0.6, label='Training data')
plt.scatter(X_reg_test, y_reg_test, color='gray', alpha=0.6, label='Test data')

x_reg_plot = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)
x_reg_plot_poly = poly_reg.transform(x_reg_plot)

for alpha in alphas:
    # Fit model with ridge regression
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)
    
    model.fit(X_reg_train_poly, y_reg_train)
    
    # Make predictions
    y_reg_train_pred = model.predict(X_reg_train_poly)
    y_reg_test_pred = model.predict(X_reg_test_poly)
    y_reg_plot_pred = model.predict(x_reg_plot_poly)
    
    # Calculate errors
    train_mse = mean_squared_error(y_reg_train, y_reg_train_pred)
    test_mse = mean_squared_error(y_reg_test, y_reg_test_pred)
    
    # Store results
    reg_results.append({
        'alpha': alpha,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'coef_norm': np.linalg.norm(model.coef_)
    })
    
    # Plot predictions
    label = 'OLS' if alpha == 0 else f'Ridge (α={alpha})'
    plt.plot(x_reg_plot, y_reg_plot_pred, label=f'{label}, Test MSE: {test_mse:.4f}')

plt.title(f'Ridge Regression with Polynomial Basis Functions (Degree {reg_degree})', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'regularization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot regularization path
reg_results_df = pd.DataFrame(reg_results)
print("\nRegularization results with different alpha values:")
print(reg_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

plt.figure(figsize=(12, 6))
plt.plot(reg_results_df['alpha'], reg_results_df['train_mse'], 'bo-', label='Training Error')
plt.plot(reg_results_df['alpha'], reg_results_df['test_mse'], 'ro-', label='Test Error')
plt.xscale('log')
plt.xlabel('Regularization Parameter (α)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Effect of Regularization on Training and Test Error', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'regularization_path.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot coefficient norm vs. regularization
plt.figure(figsize=(12, 6))
plt.plot(reg_results_df['alpha'], reg_results_df['coef_norm'], 'go-')
plt.xscale('log')
plt.xlabel('Regularization Parameter (α)', fontsize=12)
plt.ylabel('Coefficient Norm', fontsize=12)
plt.title('Effect of Regularization on Model Complexity', fontsize=14)
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'coefficient_norm.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nRegularization helps control model complexity when using many basis functions:")
print("- Without regularization (α=0), the high-degree polynomial model overfits")
print("- With optimal regularization, the model generalizes better")
print("- As regularization increases, model becomes simpler but may underfit")
print("\nFigures saved as 'regularization.png', 'regularization_path.png', and 'coefficient_norm.png'")

print_section_header("Summary")
print("1. Basis functions allow linear models to capture non-linear relationships by transforming input features.")
print("2. We've explored three types of basis functions:")
print("   - Polynomial: powers of input features (x, x², x³, etc.)")
print("   - Gaussian RBF: radial basis functions centered at key points")
print("   - Sigmoid: logistic functions with different thresholds")
print("3. For a 2D quadratic model, we need 6 basis functions: 1, x₁, x₂, x₁², x₁x₂, x₂²")
print("4. The choice of basis functions greatly affects the bias-variance tradeoff:")
print("   - Simpler basis functions may underfit (high bias)")
print("   - More complex basis functions may overfit (high variance)")
print("5. Different basis functions are suitable for different types of relationships in the data")
print("6. Regularization helps control model complexity when using many basis functions")

print("\nImages have been saved to the directory:", save_dir) 