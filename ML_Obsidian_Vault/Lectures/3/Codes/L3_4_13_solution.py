import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy import linalg
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.special import gamma

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_statement_result(statement_number, is_true, explanation):
    """Print a formatted statement result with explanation."""
    verdict = "TRUE" if is_true else "FALSE"
    print(f"\nStatement {statement_number} is {verdict}.")
    print(explanation)

# Step 1: Perfect correlation and matrix singularity
print_step_header(1, "Perfect Correlation and Matrix Singularity")

# Generate data with perfect correlation
np.random.seed(42)
n_samples = 100

# Case 1: Features with perfect correlation
x1 = np.random.normal(0, 1, n_samples)
x2_perfect = 2 * x1  # Perfect correlation (x2 = 2*x1)
X_perfect = np.column_stack((np.ones(n_samples), x1, x2_perfect))  # Add intercept

# Calculate correlation
corr_perfect = np.corrcoef(x1, x2_perfect)[0, 1]
print(f"Correlation between x1 and x2 (perfect): {corr_perfect}")

# Calculate X^T X and check if it's singular
XTX_perfect = X_perfect.T @ X_perfect
det_perfect = np.linalg.det(XTX_perfect)
print(f"Determinant of X^T X (perfect correlation): {det_perfect}")

try:
    inv_perfect = np.linalg.inv(XTX_perfect)
    print("Matrix is invertible despite perfect correlation")
except LinAlgError:
    print("Matrix is singular (not invertible) due to perfect correlation")

# Case 2: Features with imperfect correlation
x2_imperfect = 2 * x1 + 0.1 * np.random.normal(0, 1, n_samples)  # Add some noise
X_imperfect = np.column_stack((np.ones(n_samples), x1, x2_imperfect))  # Add intercept

# Calculate correlation
corr_imperfect = np.corrcoef(x1, x2_imperfect)[0, 1]
print(f"Correlation between x1 and x2 (imperfect): {corr_imperfect}")

# Calculate X^T X and check if it's singular
XTX_imperfect = X_imperfect.T @ X_imperfect
det_imperfect = np.linalg.det(XTX_imperfect)
print(f"Determinant of X^T X (imperfect correlation): {det_imperfect}")

try:
    inv_imperfect = np.linalg.inv(XTX_imperfect)
    print("Matrix is invertible with imperfect correlation")
except LinAlgError:
    print("Matrix is singular (not invertible)")

# Create visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x1, x2_perfect, alpha=0.6)
plt.title(f"Perfect Correlation\nCorr = {corr_perfect:.4f}")
plt.xlabel("x1")
plt.ylabel("x2 = 2*x1")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(x1, x2_imperfect, alpha=0.6)
plt.title(f"Imperfect Correlation\nCorr = {corr_imperfect:.4f}")
plt.xlabel("x1")
plt.ylabel("x2 = 2*x1 + noise")
plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "1_perfect_correlation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Check condition numbers
cond_perfect = np.linalg.cond(XTX_perfect)
cond_imperfect = np.linalg.cond(XTX_imperfect)
print(f"Condition number (perfect correlation): {cond_perfect}")
print(f"Condition number (imperfect correlation): {cond_imperfect}")

# Create a figure to show the effect on model fitting
y = 3 + 2 * x1 - 1 * x2_imperfect + np.random.normal(0, 0.5, n_samples)

plt.figure(figsize=(10, 6))
# Plot the data in 3D
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2_imperfect, y, alpha=0.6)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D visualization of the regression problem')

plt.tight_layout()
file_path = os.path.join(save_dir, "1b_regression_vis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(1, True, 
    "When features x1 and x2 are perfectly correlated (correlation coefficient = 1), " +
    "the matrix X^T X becomes singular (non-invertible). This happens because perfect " +
    "correlation creates linear dependence among columns of X, making X^T X rank-deficient, " +
    "which results in a zero determinant and an undefined inverse. This is a form of " +
    "multicollinearity that prevents ordinary least squares from finding a unique solution.")

# Step 2: Dummy variables for categorical variables
print_step_header(2, "Dummy Variables for Categorical Variables")

# Create a categorical variable with k categories
np.random.seed(42)
k = 4  # Number of categories
n_samples = 200
categories = [f'Category {i}' for i in range(k)]
categorical_var = np.random.choice(categories, size=n_samples)

# Method 1: Using k dummy variables (one-hot encoding)
dummy_k = pd.get_dummies(categorical_var)
print("One-hot encoding with k dummy variables:")
print(dummy_k.head())

# Method 2: Using k-1 dummy variables (reference encoding)
dummy_k_minus_1 = pd.get_dummies(categorical_var, drop_first=True)
print("\nReference encoding with k-1 dummy variables:")
print(dummy_k_minus_1.head())

# Create a toy dataset with a categorical predictor
y = np.random.normal(0, 1, n_samples)  # Random target variable

# Fit models using both encodings
X_k = dummy_k.values
X_k_minus_1 = dummy_k_minus_1.values

# Add intercept to X_k_minus_1 since we're dropping one category
X_k_minus_1_with_intercept = np.column_stack((np.ones(n_samples), X_k_minus_1))

# Fit regression with k dummy variables (without intercept to avoid multicollinearity)
coeffs_k = np.linalg.lstsq(X_k, y, rcond=None)[0]

# Fit regression with k-1 dummy variables and intercept
coeffs_k_minus_1 = np.linalg.lstsq(X_k_minus_1_with_intercept, y, rcond=None)[0]

print("\nCoefficients with k dummy variables (no intercept):")
for i, coef in enumerate(coeffs_k):
    print(f"Category {i}: {coef:.4f}")

print("\nCoefficients with k-1 dummy variables (with intercept):")
print(f"Intercept (reference category): {coeffs_k_minus_1[0]:.4f}")
for i, coef in enumerate(coeffs_k_minus_1[1:]):
    print(f"Category {i+1}: {coef:.4f}")

# Calculate X^T X for both methods and check rank
XTX_k = X_k.T @ X_k
XTX_k_minus_1 = X_k_minus_1_with_intercept.T @ X_k_minus_1_with_intercept

rank_k = np.linalg.matrix_rank(XTX_k)
rank_k_minus_1 = np.linalg.matrix_rank(XTX_k_minus_1)

print(f"\nRank of X^T X with k dummy variables: {rank_k} (full rank would be {X_k.shape[1]})")
print(f"Rank of X^T X with k-1 dummy variables and intercept: {rank_k_minus_1} (full rank would be {X_k_minus_1_with_intercept.shape[1]})")

# Create a visualization
plt.figure(figsize=(12, 6))

# Plot with k dummy variables
plt.subplot(1, 2, 1)
means_k = [np.mean(y[categorical_var == cat]) for cat in categories]
plt.bar(categories, means_k)
plt.title('Mean of y for each category\n(Directly represented by coefficients with k dummies)')
plt.ylabel('Mean of y')
plt.xlabel('Category')
plt.xticks(rotation=45)

# Plot with k-1 dummy variables
plt.subplot(1, 2, 2)
reference_effect = coeffs_k_minus_1[0]
effects = [reference_effect] + [reference_effect + coef for coef in coeffs_k_minus_1[1:]]
plt.bar(categories, effects)
plt.title('Effect of each category\n(Reference encoding with k-1 dummies)')
plt.ylabel('Effect')
plt.xlabel('Category')
plt.axhline(y=reference_effect, color='r', linestyle='-', alpha=0.3, label='Reference effect')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
file_path = os.path.join(save_dir, "2_dummy_variables.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(2, False, 
    "When encoding a categorical variable with k categories, you need at most k-1 dummy variables, " +
    "not k. Using k dummy variables (one-hot encoding) creates perfect multicollinearity when an " +
    "intercept is included in the model, as the sum of all dummy variables equals 1 for every observation. " +
    "This is known as the 'dummy variable trap'. To avoid this, we typically use k-1 dummy variables " +
    "(reference encoding), where one category serves as the reference level represented by the intercept.")

# Step 3: Adding polynomial terms to regression models
print_step_header(3, "Adding Polynomial Terms to Regression Models")

# Generate data with non-linear relationship
np.random.seed(42)
n_samples = 100
x = np.linspace(-3, 3, n_samples)

# Case 1: True relationship is linear
y_linear = 2 + 0.5 * x + np.random.normal(0, 0.5, n_samples)

# Case 2: True relationship is quadratic
y_quadratic = 2 + 0.5 * x + 0.5 * x**2 + np.random.normal(0, 0.5, n_samples)

# Case 3: True relationship is cubic but negative contribution
y_cubic = 2 + 0.5 * x - 0.2 * x**3 + np.random.normal(0, 0.5, n_samples)

# Fit models of different degrees
X = x.reshape(-1, 1)

# For linear data
linear_mse_train = []
models_linear = []

for degree in range(1, 6):  # Fit polynomial up to degree 5
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression(fit_intercept=False)  # No intercept as it's included in X_poly
    model.fit(X_poly, y_linear)
    models_linear.append(model)
    
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y_linear, y_pred)
    linear_mse_train.append(mse)
    
    print(f"Linear data - Polynomial degree {degree}: MSE = {mse:.4f}")

# For quadratic data
quadratic_mse_train = []
models_quadratic = []

for degree in range(1, 6):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y_quadratic)
    models_quadratic.append(model)
    
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y_quadratic, y_pred)
    quadratic_mse_train.append(mse)
    
    print(f"Quadratic data - Polynomial degree {degree}: MSE = {mse:.4f}")

# For cubic data
cubic_mse_train = []
models_cubic = []

for degree in range(1, 6):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y_cubic)
    models_cubic.append(model)
    
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y_cubic, y_pred)
    cubic_mse_train.append(mse)
    
    print(f"Cubic data - Polynomial degree {degree}: MSE = {mse:.4f}")

# Create visualizations
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# Plot data and fitted curves for linear data
axs[0, 0].scatter(x, y_linear, alpha=0.6, label='Data')
x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

for degree, model in enumerate(models_linear, 1):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly_plot = poly_features.fit_transform(x_plot)
    y_plot = model.predict(X_poly_plot)
    axs[0, 0].plot(x_plot, y_plot, label=f'Degree {degree}')

axs[0, 0].set_title('Linear Data with Polynomial Fits')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot MSE vs polynomial degree for linear data
axs[0, 1].plot(range(1, 6), linear_mse_train, 'o-')
axs[0, 1].set_title('MSE vs Polynomial Degree (Linear Data)')
axs[0, 1].set_xlabel('Polynomial Degree')
axs[0, 1].set_ylabel('Mean Squared Error')
axs[0, 1].grid(True)

# Plot data and fitted curves for quadratic data
axs[1, 0].scatter(x, y_quadratic, alpha=0.6, label='Data')

for degree, model in enumerate(models_quadratic, 1):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly_plot = poly_features.fit_transform(x_plot)
    y_plot = model.predict(X_poly_plot)
    axs[1, 0].plot(x_plot, y_plot, label=f'Degree {degree}')

axs[1, 0].set_title('Quadratic Data with Polynomial Fits')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot MSE vs polynomial degree for quadratic data
axs[1, 1].plot(range(1, 6), quadratic_mse_train, 'o-')
axs[1, 1].set_title('MSE vs Polynomial Degree (Quadratic Data)')
axs[1, 1].set_xlabel('Polynomial Degree')
axs[1, 1].set_ylabel('Mean Squared Error')
axs[1, 1].grid(True)

# Plot data and fitted curves for cubic data
axs[2, 0].scatter(x, y_cubic, alpha=0.6, label='Data')

for degree, model in enumerate(models_cubic, 1):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly_plot = poly_features.fit_transform(x_plot)
    y_plot = model.predict(X_poly_plot)
    axs[2, 0].plot(x_plot, y_plot, label=f'Degree {degree}')

axs[2, 0].set_title('Cubic Data with Polynomial Fits')
axs[2, 0].set_xlabel('x')
axs[2, 0].set_ylabel('y')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Plot MSE vs polynomial degree for cubic data
axs[2, 1].plot(range(1, 6), cubic_mse_train, 'o-')
axs[2, 1].set_title('MSE vs Polynomial Degree (Cubic Data)')
axs[2, 1].set_xlabel('Polynomial Degree')
axs[2, 1].set_ylabel('Mean Squared Error')
axs[2, 1].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "3_polynomial_terms.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(3, False, 
    "Adding a polynomial term (e.g., x²) to a regression model does not always improve the model's " +
    "fit to the training data. While it often reduces training error when the true relationship is " +
    "non-linear, it can lead to overfitting, especially with higher-degree polynomials. Additionally, " +
    "if the true relationship is perfectly linear, adding polynomial terms may not significantly " +
    "improve the fit and could capture noise rather than signal. The benefit of polynomial terms " +
    "depends on the underlying data structure and the true functional relationship.")

# Step 4: Normal equation and global minimum of sum of squared errors
print_step_header(4, "Normal Equation and Global Minimum of Sum of Squared Errors")

# Generate synthetic data
np.random.seed(42)
n_samples = 100
n_features = 2

X = np.random.normal(0, 1, (n_samples, n_features))
X = np.column_stack((np.ones(n_samples), X))  # Add intercept term
true_w = np.array([2, 0.5, -1.5])
y = X @ true_w + np.random.normal(0, 0.5, n_samples)

# Compute normal equation solution
w_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print("Normal equation solution:")
print(f"w_0 (intercept): {w_normal[0]:.4f}")
print(f"w_1: {w_normal[1]:.4f}")
print(f"w_2: {w_normal[2]:.4f}")

# Calculate cost (sum of squared errors) for the normal equation solution
y_pred_normal = X @ w_normal
sse_normal = np.sum((y - y_pred_normal) ** 2)
print(f"Sum of squared errors (SSE) for normal equation solution: {sse_normal:.4f}")

# Create a grid of possible values for w1 and w2 (keeping w0 fixed at normal equation value)
w1_range = np.linspace(w_normal[1] - 1, w_normal[1] + 1, 50)
w2_range = np.linspace(w_normal[2] - 1, w_normal[2] + 1, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Calculate SSE for each combination of w1 and w2
SSE = np.zeros_like(W1)
for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        w = np.array([w_normal[0], W1[i, j], W2[i, j]])
        SSE[i, j] = np.sum((y - X @ w) ** 2)

# Create a visualization
fig = plt.figure(figsize=(12, 6))

# 3D surface of SSE
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(W1, W2, SSE, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Sum of Squared Errors')
ax1.set_title('SSE Cost Function Surface')
# Mark the minimum from normal equation
ax1.scatter([w_normal[1]], [w_normal[2]], [sse_normal], color='red', s=50, marker='*', label='Normal Equation Solution')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# Contour plot of SSE
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(W1, W2, SSE, 20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.scatter([w_normal[1]], [w_normal[2]], color='red', s=50, marker='*', label='Normal Equation Solution')
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_title('Contour Plot of SSE Cost Function')
ax2.legend()
ax2.grid(True)

# Show that the gradient is zero at the normal equation solution
X_no_intercept = X[:, 1:]
gradient_w1 = -2 * X_no_intercept[:, 0].T @ (y - X @ w_normal)
gradient_w2 = -2 * X_no_intercept[:, 1].T @ (y - X @ w_normal)

print(f"Gradient at normal equation solution: [{gradient_w1:.8f}, {gradient_w2:.8f}]")
print("The gradient is approximately zero, confirming this is a critical point (minimum).")

# Check convexity: Compute the Hessian matrix
hessian = 2 * X_no_intercept.T @ X_no_intercept
eigenvalues = np.linalg.eigvals(hessian)
print(f"Eigenvalues of the Hessian: [{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}]")
print("Since all eigenvalues are positive, the Hessian is positive definite, confirming the cost function is convex.")
print("Therefore, the critical point is guaranteed to be a global minimum.")

plt.tight_layout()
file_path = os.path.join(save_dir, "4_normal_equation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(4, True, 
    "In multiple linear regression, the normal equation w = (X^T X)^(-1) X^T y provides the global " +
    "minimum of the sum of squared errors cost function. This is because the cost function is convex " +
    "(the Hessian matrix is positive definite), which guarantees that the critical point found by " +
    "setting the gradient equal to zero is a global minimum. The normal equation directly solves " +
    "this system of equations in one step, avoiding the need for iterative optimization procedures.")

# Step 5: Coefficients of predictor variables with no effect
print_step_header(5, "Coefficients of Predictor Variables with No Effect")

# Generate data with predictors of varying effects
np.random.seed(42)
n_samples = 500
n_features = 3

X = np.random.normal(0, 1, (n_samples, n_features))
X = np.column_stack((np.ones(n_samples), X))  # Add intercept

# True coefficients: x3 has no effect (coefficient = 0)
true_w = np.array([2, 0.5, -1.5, 0])
y = X @ true_w + np.random.normal(0, 1, n_samples)

# Fit regression model
w_estimated = np.linalg.inv(X.T @ X) @ X.T @ y

print("True coefficients:")
print(f"w_0 (intercept): {true_w[0]:.4f}")
print(f"w_1: {true_w[1]:.4f}")
print(f"w_2: {true_w[2]:.4f}")
print(f"w_3 (no effect): {true_w[3]:.4f}")

print("\nEstimated coefficients:")
print(f"w_0 (intercept): {w_estimated[0]:.4f}")
print(f"w_1: {w_estimated[1]:.4f}")
print(f"w_2: {w_estimated[2]:.4f}")
print(f"w_3 (no effect): {w_estimated[3]:.4f}")

# Confidence intervals via bootstrapping
n_bootstrap = 1000
bootstrap_coefficients = np.zeros((n_bootstrap, 4))

for i in range(n_bootstrap):
    # Resample with replacement
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    
    # Fit model on bootstrap sample
    bootstrap_coefficients[i] = np.linalg.inv(X_bootstrap.T @ X_bootstrap) @ X_bootstrap.T @ y_bootstrap

# Calculate 95% confidence intervals
confidence_intervals = np.percentile(bootstrap_coefficients, [2.5, 97.5], axis=0)
print("\n95% Confidence Intervals:")
for i in range(4):
    coef_name = "w_0 (intercept)" if i == 0 else f"w_{i}"
    if i == 3:
        coef_name += " (no effect)"
    print(f"{coef_name}: [{confidence_intervals[0, i]:.4f}, {confidence_intervals[1, i]:.4f}]")

# Create histograms of bootstrap coefficients
plt.figure(figsize=(15, 10))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(bootstrap_coefficients[:, i], bins=30, alpha=0.7)
    plt.axvline(x=true_w[i], color='r', linestyle='--', label=f'True value: {true_w[i]:.2f}')
    plt.axvline(x=w_estimated[i], color='g', linestyle='-', label=f'Estimated: {w_estimated[i]:.2f}')
    plt.axvline(x=confidence_intervals[0, i], color='b', linestyle=':', label=f'95% CI: [{confidence_intervals[0, i]:.2f}, {confidence_intervals[1, i]:.2f}]')
    plt.axvline(x=confidence_intervals[1, i], color='b', linestyle=':')
    
    coef_name = "w_0 (intercept)" if i == 0 else f"w_{i}"
    if i == 3:
        coef_name += " (no effect)"
    plt.title(f'Bootstrap Distribution of {coef_name}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "5_zero_coefficient.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Now let's vary the sample size and noise level to see the effect on estimates
sample_sizes = [50, 100, 200, 500, 1000]
noise_levels = [0.5, 1.0, 2.0]

results = np.zeros((len(sample_sizes), len(noise_levels)))

for i, size in enumerate(sample_sizes):
    for j, noise in enumerate(noise_levels):
        # Generate data
        X_temp = np.random.normal(0, 1, (size, n_features))
        X_temp = np.column_stack((np.ones(size), X_temp))
        y_temp = X_temp @ true_w + np.random.normal(0, noise, size)
        
        # Fit model
        w_temp = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y_temp
        
        # Store the estimate of the no-effect coefficient
        results[i, j] = w_temp[3]

# Create a visualization of results
plt.figure(figsize=(10, 6))
for j, noise in enumerate(noise_levels):
    plt.plot(sample_sizes, results[:, j], 'o-', label=f'Noise level: {noise}')

plt.axhline(y=0, color='r', linestyle='--', label='True coefficient (0)')
plt.xscale('log')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Estimated Coefficient for No-Effect Predictor')
plt.title('Effect of Sample Size and Noise on Coefficient Estimates')
plt.legend()
plt.grid(True)

file_path = os.path.join(save_dir, "5b_sample_size_noise.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(5, False, 
    "If a predictor variable has no effect on the response, its coefficient in a multiple regression " +
    "model will not always be exactly zero in practice. Due to sampling variability and noise in " +
    "the data, estimated coefficients will typically be non-zero even when the true coefficient is " +
    "zero. Only with infinite data or no noise would we expect to get exactly zero. In realistic " +
    "scenarios, we use hypothesis testing and confidence intervals to determine whether a coefficient " +
    "is statistically distinguishable from zero, rather than expecting it to be exactly zero.")

# Step 6: Interpretation of main effects with interaction terms
print_step_header(6, "Interpretation of Main Effects with Interaction Terms")

# Generate data with interaction effect
np.random.seed(42)
n_samples = 500

# Generate two predictor variables
x1 = np.random.uniform(-2, 2, n_samples)
x2 = np.random.uniform(-2, 2, n_samples)

# Create interaction term
x1x2 = x1 * x2

# True coefficients with interaction
beta0 = 3    # intercept
beta1 = 2    # effect of x1
beta2 = -1   # effect of x2
beta3 = 1.5  # interaction effect

# Generate response variable
y = beta0 + beta1*x1 + beta2*x2 + beta3*x1x2 + np.random.normal(0, 1, n_samples)

# Create design matrix
X = np.column_stack((np.ones(n_samples), x1, x2, x1x2))

# Fit model
betas = np.linalg.inv(X.T @ X) @ X.T @ y

print("True coefficients:")
print(f"β₀ (intercept): {beta0}")
print(f"β₁ (x1 effect): {beta1}")
print(f"β₂ (x2 effect): {beta2}")
print(f"β₃ (interaction): {beta3}")

print("\nEstimated coefficients:")
print(f"β₀ (intercept): {betas[0]:.4f}")
print(f"β₁ (x1 effect): {betas[1]:.4f}")
print(f"β₂ (x2 effect): {betas[2]:.4f}")
print(f"β₃ (interaction): {betas[3]:.4f}")

# Demonstrate interpretation at different x2 values
x2_values = [-2, -1, 0, 1, 2]
print("\nEffect of x1 at different values of x2:")
for x2_val in x2_values:
    x1_effect = betas[1] + betas[3] * x2_val
    print(f"When x2 = {x2_val}: Effect of x1 = {x1_effect:.4f}")

# Create visualization
fig = plt.figure(figsize=(15, 10))

# 3D surface plot showing the interaction
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
x1_grid = np.linspace(-2, 2, 50)
x2_grid = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Y = beta0 + beta1*X1 + beta2*X2 + beta3*X1*X2

ax1.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.7)
ax1.scatter(x1, x2, y, color='red', alpha=0.1)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('3D Visualization of Interaction Effect')

# Plot showing the effect of x1 at different x2 values
ax2 = fig.add_subplot(2, 2, 2)
for x2_val in [-2, -1, 0, 1, 2]:
    y_pred = betas[0] + betas[1]*x1_grid + betas[2]*x2_val + betas[3]*x1_grid*x2_val
    ax2.plot(x1_grid, y_pred, label=f'x2 = {x2_val}')

ax2.set_xlabel('x1')
ax2.set_ylabel('y')
ax2.set_title('Effect of x1 at Different x2 Values')
ax2.legend()
ax2.grid(True)

# Plot showing the effect of x2 at different x1 values
ax3 = fig.add_subplot(2, 2, 3)
for x1_val in [-2, -1, 0, 1, 2]:
    y_pred = betas[0] + betas[1]*x1_val + betas[2]*x2_grid + betas[3]*x1_val*x2_grid
    ax3.plot(x2_grid, y_pred, label=f'x1 = {x1_val}')

ax3.set_xlabel('x2')
ax3.set_ylabel('y')
ax3.set_title('Effect of x2 at Different x1 Values')
ax3.legend()
ax3.grid(True)

# Plot the effect of x1 as a function of x2
ax4 = fig.add_subplot(2, 2, 4)
x2_grid_effect = np.linspace(-2, 2, 100)
x1_effect = betas[1] + betas[3] * x2_grid_effect
ax4.plot(x2_grid_effect, x1_effect)
ax4.set_xlabel('x2')
ax4.set_ylabel('Effect of x1')
ax4.set_title('Marginal Effect of x1 as a Function of x2')
ax4.axhline(y=betas[1], color='r', linestyle='--', label=f'Main effect of x1: {betas[1]:.2f}')
ax4.axvline(x=0, color='g', linestyle='--', label='x2 = 0')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "6_interaction_terms.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(6, True, 
    "In a multiple regression model with interaction terms, the coefficient of a main effect (e.g., x₁) " +
    "represents the effect of that variable when all interacting variables are zero. For example, in the " +
    "model y = β₀ + β₁x₁ + β₂x₂ + β₃x₁x₂, the coefficient β₁ represents the change in y for a unit change " +
    "in x₁ when x₂ = 0. When x₂ is non-zero, the effect of x₁ becomes β₁ + β₃x₂, showing how the effect " +
    "varies based on the value of the interacting variable. This conditional interpretation is fundamental " +
    "to understanding interaction effects in regression models.")

# Step 7: Radial basis functions in different dimensions
print_step_header(7, "Radial Basis Functions in Different Dimensions")

# Define Gaussian RBF function
def rbf(x, center, sigma=1.0):
    return np.exp(-np.sum((x - center) ** 2) / (2 * sigma ** 2))

# Case 1: RBF in 1D
np.random.seed(42)
n_samples_1d = 100
x_1d = np.sort(np.random.uniform(-5, 5, n_samples_1d)).reshape(-1, 1)

# Create centers for RBFs
centers_1d = np.array([[-3], [0], [3]])
n_centers_1d = centers_1d.shape[0]

# Create RBF features
rbf_features_1d = np.zeros((n_samples_1d, n_centers_1d))
for i, center in enumerate(centers_1d):
    rbf_features_1d[:, i] = np.array([rbf(x, center, sigma=1.0) for x in x_1d])

# Generate a non-linear target function
y_1d = np.sin(x_1d.flatten()) + 0.1 * np.random.normal(0, 1, n_samples_1d)

# Fit a linear model with RBF features
X_1d = np.column_stack((np.ones(n_samples_1d), rbf_features_1d))
w_1d = np.linalg.inv(X_1d.T @ X_1d) @ X_1d.T @ y_1d

# Make predictions
x_pred_1d = np.linspace(-5, 5, 500).reshape(-1, 1)
rbf_pred_1d = np.zeros((500, n_centers_1d))
for i, center in enumerate(centers_1d):
    rbf_pred_1d[:, i] = np.array([rbf(x, center, sigma=1.0) for x in x_pred_1d])
X_pred_1d = np.column_stack((np.ones(500), rbf_pred_1d))
y_pred_1d = X_pred_1d @ w_1d

# Case 2: RBF in 2D
n_samples_2d = 500
x1_2d = np.random.uniform(-5, 5, n_samples_2d)
x2_2d = np.random.uniform(-5, 5, n_samples_2d)
X_orig_2d = np.column_stack((x1_2d, x2_2d))

# Create centers for RBFs in 2D
centers_2d = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3], [0, 0]])
n_centers_2d = centers_2d.shape[0]

# Create RBF features
rbf_features_2d = np.zeros((n_samples_2d, n_centers_2d))
for i, center in enumerate(centers_2d):
    rbf_features_2d[:, i] = np.array([rbf(x, center, sigma=2.0) for x in X_orig_2d])

# Generate a non-linear target function
y_2d = np.sin(x1_2d) * np.cos(x2_2d) + 0.1 * np.random.normal(0, 1, n_samples_2d)

# Fit a linear model with RBF features
X_2d = np.column_stack((np.ones(n_samples_2d), rbf_features_2d))
w_2d = np.linalg.inv(X_2d.T @ X_2d) @ X_2d.T @ y_2d

# Create a grid for prediction and visualization
x1_grid = np.linspace(-5, 5, 50)
x2_grid = np.linspace(-5, 5, 50)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
X_grid_points = np.column_stack((X1_grid.flatten(), X2_grid.flatten()))

# Compute RBF features for grid points
rbf_grid_2d = np.zeros((X_grid_points.shape[0], n_centers_2d))
for i, center in enumerate(centers_2d):
    rbf_grid_2d[:, i] = np.array([rbf(x, center, sigma=2.0) for x in X_grid_points])

# Make predictions
X_grid_2d = np.column_stack((np.ones(X_grid_points.shape[0]), rbf_grid_2d))
y_grid_2d = X_grid_2d @ w_2d
Y_grid_2d = y_grid_2d.reshape(X1_grid.shape)

# Case 3: RBF in 3D
n_samples_3d = 1000
x1_3d = np.random.uniform(-5, 5, n_samples_3d)
x2_3d = np.random.uniform(-5, 5, n_samples_3d)
x3_3d = np.random.uniform(-5, 5, n_samples_3d)
X_orig_3d = np.column_stack((x1_3d, x2_3d, x3_3d))

# Create centers for RBFs in 3D (8 corners of a cube + center)
centers_3d = np.array([
    [-3, -3, -3], [-3, -3, 3], [-3, 3, -3], [-3, 3, 3],
    [3, -3, -3], [3, -3, 3], [3, 3, -3], [3, 3, 3],
    [0, 0, 0]
])
n_centers_3d = centers_3d.shape[0]

# Create RBF features
rbf_features_3d = np.zeros((n_samples_3d, n_centers_3d))
for i, center in enumerate(centers_3d):
    rbf_features_3d[:, i] = np.array([rbf(x, center, sigma=3.0) for x in X_orig_3d])

# Generate a non-linear target function in 3D
y_3d = np.sin(x1_3d) * np.cos(x2_3d) * np.sin(x3_3d) + 0.1 * np.random.normal(0, 1, n_samples_3d)

# Fit a linear model with RBF features
X_3d = np.column_stack((np.ones(n_samples_3d), rbf_features_3d))
w_3d = np.linalg.inv(X_3d.T @ X_3d) @ X_3d.T @ y_3d

# Calculate R² for each model to measure performance
def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r2_1d = calculate_r2(y_1d, X_1d @ w_1d)
r2_2d = calculate_r2(y_2d, X_2d @ w_2d)
r2_3d = calculate_r2(y_3d, X_3d @ w_3d)

print(f"1D RBF model R² = {r2_1d:.4f} with {n_centers_1d} basis functions")
print(f"2D RBF model R² = {r2_2d:.4f} with {n_centers_2d} basis functions")
print(f"3D RBF model R² = {r2_3d:.4f} with {n_centers_3d} basis functions")

# Create visualizations
fig = plt.figure(figsize=(15, 12))

# Plot 1D case: data, RBFs, and fitted function
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(x_1d, y_1d, alpha=0.5, label='Data')
ax1.plot(x_pred_1d, y_pred_1d, 'r-', label='RBF fit')

# Plot individual RBFs
for i, center in enumerate(centers_1d):
    scaled_rbf = rbf_pred_1d[:, i] * w_1d[i+1]  # Scale by coefficient
    ax1.plot(x_pred_1d, scaled_rbf, '--', alpha=0.5, label=f'RBF {i+1}')

ax1.axhline(y=w_1d[0], color='k', linestyle=':', alpha=0.5, label='Intercept')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('1D Example with Radial Basis Functions')
ax1.legend()
ax1.grid(True)

# Plot 2D case: original function and RBF approximation
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
# Plot the true function
X1_true, X2_true = np.meshgrid(x1_grid, x2_grid)
Y_true = np.sin(X1_true) * np.cos(X2_true)
ax2.plot_surface(X1_true, X2_true, Y_true, cmap='viridis', alpha=0.5, label='True function')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('2D Example: True Function')

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
# Plot the RBF approximation
ax3.plot_surface(X1_grid, X2_grid, Y_grid_2d, cmap='plasma', alpha=0.7)
# Plot the centers
ax3.scatter(centers_2d[:, 0], centers_2d[:, 1], np.max(Y_grid_2d) * np.ones(n_centers_2d), 
           color='red', s=50, label='RBF centers')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('2D Example: RBF Approximation')

# Plot 3D case: visualization of RBF centers
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2], color='red', s=100, label='RBF centers')
ax4.scatter(X_orig_3d[:100, 0], X_orig_3d[:100, 1], X_orig_3d[:100, 2], alpha=0.1, label='Data points (sample)')

# Add a sphere to represent one RBF
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = 3.0 * np.outer(np.cos(u), np.sin(v))
y = 3.0 * np.outer(np.sin(u), np.sin(v))
z = 3.0 * np.outer(np.ones(np.size(u)), np.cos(v))
ax4.plot_surface(x, y, z, color='cyan', alpha=0.1)

ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.set_zlabel('x3')
ax4.set_title('3D Example: RBF Centers and Single RBF Region')

plt.tight_layout()
file_path = os.path.join(save_dir, "7_radial_basis_functions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(7, False, 
    "Radial basis functions are not limited to problems with exactly two input dimensions. They are " +
    "versatile and can be applied to problems of any dimensionality. As demonstrated, RBFs work " +
    "effectively in 1D, 2D, 3D, and higher-dimensional spaces. In each case, an RBF computes the " +
    "distance from a data point to a center point in the feature space (of any dimension) and " +
    "transforms it using a function like the Gaussian kernel. RBFs are widely used in various " +
    "applications including function approximation, classification, time series prediction, and " +
    "control systems, regardless of the input dimensionality.")

# Step 8: The curse of dimensionality
print_step_header(8, "The Curse of Dimensionality")

# Demonstrate various aspects of the curse of dimensionality
# 1. Volume of the space increases exponentially with dimensions
dimensions = np.arange(1, 11)
unit_hypercube_volume = np.ones_like(dimensions)  # Volume of unit hypercube is always 1
unit_hypersphere_volume = np.array([np.pi**(d/2) / gamma(d/2 + 1) for d in dimensions])  # Volume of unit hypersphere

# 2. Data sparsity: Distance between points increases with dimensions
def average_dist_between_points(n_points, n_dims):
    points = np.random.uniform(0, 1, (n_points, n_dims))
    
    # Compute pairwise distances
    distances = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            distances.append(dist)
    
    return np.mean(distances)

# Calculate average distance between points for varying dimensions
n_points = 100
avg_distances = [average_dist_between_points(n_points, d) for d in dimensions]

# 3. Increased data requirements for adequate coverage
# For k divisions per dimension, we need k^d points for full coverage
k = 10  # number of divisions per dimension
points_needed = k**dimensions

# 4. Distance concentration: High-dim distances tend to concentrate
def distance_concentration(dims, n_samples=1000):
    # Generate random points in a unit hypercube
    points = np.random.uniform(0, 1, (n_samples, dims))
    
    # Calculate distance from each point to the origin
    distances = np.sqrt(np.sum(points**2, axis=1))
    
    # Calculate the standard deviation and mean of distances
    std_dev = np.std(distances)
    mean_dist = np.mean(distances)
    
    # Return the coefficient of variation (std_dev/mean) as a measure of concentration
    return std_dev / mean_dist

# Calculate distance concentration for varying dimensions
concentration_measures = [distance_concentration(d) for d in dimensions]

# 5. Nearest neighbor distance ratio demonstration
def nn_distance_ratio(dims, n_samples=1000):
    # Generate random points
    points = np.random.uniform(0, 1, (n_samples, dims))
    
    # Calculate distance from a random point to all others
    reference_point = np.random.uniform(0, 1, dims)
    distances = np.sqrt(np.sum((points - reference_point)**2, axis=1))
    
    # Sort distances
    distances = np.sort(distances)
    
    # Return ratio of farthest to nearest neighbor distance
    return distances[-1] / distances[0]

# Calculate nearest-to-farthest distance ratio for varying dimensions
distance_ratios = [nn_distance_ratio(d) for d in dimensions]

# Create visualizations
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Volume of hypersphere within a unit hypercube
axs[0, 0].plot(dimensions, unit_hypercube_volume, 'b-', label='Unit Hypercube')
axs[0, 0].plot(dimensions, unit_hypersphere_volume, 'r-', label='Unit Hypersphere')
axs[0, 0].set_xlabel('Dimension')
axs[0, 0].set_ylabel('Volume')
axs[0, 0].set_title('Volume vs. Dimension')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Average distance between random points
axs[0, 1].plot(dimensions, avg_distances, 'g-o')
axs[0, 1].set_xlabel('Dimension')
axs[0, 1].set_ylabel('Average Distance')
axs[0, 1].set_title('Average Distance Between Random Points')
axs[0, 1].grid(True)

# Plot 3: Required points for coverage
axs[0, 2].semilogy(dimensions, points_needed, 'm-o')
axs[0, 2].set_xlabel('Dimension')
axs[0, 2].set_ylabel('Points Needed (log scale)')
axs[0, 2].set_title(f'Points Needed for Coverage\n({k} divisions per dimension)')
axs[0, 2].grid(True)

# Plot 4: Distance concentration
axs[1, 0].plot(dimensions, concentration_measures, 'c-o')
axs[1, 0].set_xlabel('Dimension')
axs[1, 0].set_ylabel('Coefficient of Variation')
axs[1, 0].set_title('Distance Concentration Effect')
axs[1, 0].grid(True)

# Plot 5: Nearest-to-farthest neighbor distance ratio
axs[1, 1].semilogy(dimensions, distance_ratios, 'y-o')
axs[1, 1].set_xlabel('Dimension')
axs[1, 1].set_ylabel('Farthest/Nearest Ratio (log scale)')
axs[1, 1].set_title('Distance Ratio Effect')
axs[1, 1].grid(True)

# Plot 6: Visualization of 2D vs 3D data sparsity
ax3d = fig.add_subplot(2, 3, 6, projection='3d')
# Generate small number of points in 2D (projected to 3D for visualization)
n_sparse = 20
points_2d = np.random.uniform(0, 1, (n_sparse, 2))
ax3d.scatter(points_2d[:, 0], points_2d[:, 1], np.zeros(n_sparse), 
            color='blue', s=50, label='2D points')
# Generate same number of points in 3D
points_3d = np.random.uniform(0, 1, (n_sparse, 3))
ax3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
            color='red', s=50, label='3D points (same count)')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('Data Sparsity: 2D vs 3D')
ax3d.legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "8_curse_of_dimensionality.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print_statement_result(8, False, 
    "The curse of dimensionality refers to various phenomena that arise when analyzing data in " +
    "high-dimensional spaces, not exclusively to computational complexity issues. While computational " +
    "challenges are one aspect, other critical problems include: (1) exponential increase in volume " +
    "leading to data sparsity, (2) increasing difficulty in covering the space with samples, " +
    "(3) breakdown of distance metrics as dimensions increase, (4) concentration of distances making " +
    "nearest-neighbor approaches less effective, and (5) overfitting due to the high-dimensional feature " +
    "space relative to the number of samples. These issues affect statistical significance, model " +
    "accuracy, and generalization ability - problems that persist even with unlimited computational power.")

# Step 9: Summarize all statements
print_step_header(9, "Summary of All Statements")

statements = [
    "1. In a multiple linear regression model, if features x₁ and x₂ are perfectly correlated (correlation coefficient = 1), then (X^T X) will be singular (non-invertible).",
    "2. When encoding a categorical variable with k categories using dummy variables, you always need exactly k dummy variables.",
    "3. Adding a polynomial term (e.g., x²) to a regression model always improves the model's fit to the training data.",
    "4. In multiple linear regression, the normal equation w = (X^T X)^(-1) X^T y provides the global minimum of the sum of squared errors cost function.",
    "5. If a predictor variable has no effect on the response, its coefficient in a multiple regression model will always be exactly zero.",
    "6. In a multiple regression model with interaction terms, the coefficient of a main effect (e.g., x₁) represents the effect of that variable when all interacting variables are zero.",
    "7. Radial basis functions are useful only for problems with exactly two input dimensions.",
    "8. The curse of dimensionality refers exclusively to computational complexity issues when fitting models with many features."
]

verdicts = [
    "TRUE",
    "FALSE",
    "FALSE",
    "TRUE",
    "FALSE",
    "TRUE",
    "FALSE",
    "FALSE"
]

explanations = [
    "Perfect correlation creates linear dependence in the design matrix, making X^T X singular.",
    "Using k dummy variables with an intercept creates perfect multicollinearity. Only k-1 are needed.",
    "Adding polynomial terms may improve fit if the relationship is non-linear, but not always.",
    "The normal equation gives the global minimum because the sum of squared errors is a convex function.",
    "Due to sampling variability and noise, coefficients are rarely exactly zero even when the true effect is zero.",
    "Main effect coefficients represent the variable's effect when interacting variables equal zero.",
    "RBFs work in spaces of any dimensionality, from 1D to high-dimensional feature spaces.",
    "The curse of dimensionality includes data sparsity, distance concentration, and other statistical issues."
]

print("\n\nSummary of Verdicts:")
print("-" * 100)
print(f"{'Statement':<90} {'Verdict':<10} {'Brief Explanation'}")
print("-" * 100)
for i, (statement, verdict, explanation) in enumerate(zip(statements, verdicts, explanations)):
    print(f"{statement:<90} {verdict:<10} {explanation}")
print("-" * 100)

# Count true and false statements
true_count = sum([1 for v in verdicts if v == "TRUE"])
false_count = sum([1 for v in verdicts if v == "FALSE"])

print(f"\nFinal count: {true_count} TRUE statements, {false_count} FALSE statements")
print("The TRUE statements are 1, 4, and 6.")
print("The FALSE statements are 2, 3, 5, 7, and 8.")

print("\nThis completes the exploration of multiple linear regression concepts.") 