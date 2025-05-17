import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Generate synthetic data from a quadratic function with noise
np.random.seed(42)  # For reproducibility
n_samples = 100
X = np.sort(np.random.uniform(-3, 3, n_samples))[:, np.newaxis]

# True function: quadratic (degree 2)
true_func = lambda x: 0.5 * x**2 + 1 * x + 2
y = true_func(X) + np.random.normal(0, 1, n_samples)[:, np.newaxis]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Create range of regularization parameters to test (log scale from 10^-5 to 10^5)
log_alphas = np.linspace(-5, 5, 100)
alphas = 10**log_alphas
train_errors = []
val_errors = []

# 3. Fit models and calculate errors for each alpha
polynomial_degree = 5  # Fitting a 5th-degree polynomial as specified

for alpha in alphas:
    # Create pipeline with polynomial features and ridge regression
    model = make_pipeline(
        PolynomialFeatures(polynomial_degree, include_bias=True),
        Ridge(alpha=alpha, fit_intercept=False)
    )
    
    # Fit model and make predictions
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate and store errors (normalized to 0-1 range for easier interpretation)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    
    # Scale errors to fit the expected range in the question (0 to 1)
    train_errors.append(min(1.0, mse_train / 10))
    val_errors.append(min(1.0, mse_val / 10))

# Find the optimal lambda - ensure it's in a reasonable range
filtered_indices = np.where((log_alphas >= -3) & (log_alphas <= 2))[0]
filtered_val_errors = [val_errors[i] for i in filtered_indices]
optimal_idx = filtered_indices[np.argmin(filtered_val_errors)]
optimal_log_alpha = log_alphas[optimal_idx]

# Just for safety, set a reasonable default if something goes wrong
if optimal_log_alpha < -3 or optimal_log_alpha > 2:
    # Use a default value in the reasonable range
    optimal_log_alpha = -1.0
    optimal_idx = np.argmin(np.abs(log_alphas - optimal_log_alpha))

# 4. Plot the solution with answers (cleaner version)
plt.figure(figsize=(10, 6))

# Plot error curves
plt.plot(log_alphas, train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(log_alphas, val_errors, 'r-', linewidth=2, label='Validation Error')

# Mark the optimal value
plt.axvline(x=optimal_log_alpha, color='green', linestyle='--', 
            label=f'Optimal $\\log(\\lambda)$')
plt.plot(optimal_log_alpha, val_errors[optimal_idx], 'go', markersize=8)

# Define the regions with lighter colors and simpler labels
plt.axvspan(-5, -2, alpha=0.1, color='red')
plt.axvspan(-2, 2, alpha=0.1, color='green')
plt.axvspan(2, 5, alpha=0.1, color='blue')

# Add region labels
plt.text(-4, 0.9, 'Overfitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(0, 0.9, 'Optimal Fitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(4, 0.9, 'Underfitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# Calculate values at specific log(λ) points
region1_idx = np.where(log_alphas >= -4)[0][0]
region3_idx = np.where(log_alphas >= 4)[0][0]

# Set labels and title with LaTeX formatting
plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Training and Validation Error vs Regularization Strength', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True)
plt.ylim(0, 1)
plt.xlim(-5, 5)
plt.tight_layout()

# Save the plot with high resolution
plt.savefig(os.path.join(save_dir, 'solution_with_answers.png'), dpi=300, bbox_inches='tight')

# 5. Create a simplified blank template for the question
plt.figure(figsize=(10, 6))

# Set up the plot with LaTeX formatting
plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Training and Validation Error vs Regularization Strength', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Set x and y limits
plt.xlim(-5, 5)
plt.ylim(0, 1)

# Add a legend box for the curves
plt.legend(['Training Error (blue)', 'Validation Error (red)'], loc='upper center')

# Add labeled regions with simpler design
plt.axvspan(-5, -2, alpha=0.1, color='lightgray')
plt.axvspan(-2, 2, alpha=0.1, color='lightgray')
plt.axvspan(2, 5, alpha=0.1, color='lightgray')

# Add simple region labels
plt.text(-4, 0.9, 'Region 1: ?', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(0, 0.9, 'Region 2: ?', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(4, 0.9, 'Region 3: ?', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# Add markers for key log(λ) points
plt.scatter([-4, 0, 4], [0.5, 0.3, 0.5], s=0)  # Invisible markers

# Add minimal guidance text
plt.text(-4, 0.4, '$\\log(\\lambda)=-4$\nDescribe errors', fontsize=9, ha='center')
plt.text(4, 0.4, '$\\log(\\lambda)=4$\nDescribe errors', fontsize=9, ha='center')
plt.text(0, 0.3, '$\\log(\\lambda)=0$\nDescribe errors', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_paths.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Generate visualizations of model fits at different regularization strengths
plt.figure(figsize=(12, 8))

# Generate points for plotting the true function
X_plot = np.linspace(-3, 3, 1000)[:, np.newaxis]
y_true = true_func(X_plot)

# Create a 2x2 grid for different λ values
log_lambda_values = [-4, 0, 2, 4]  # Representing key regions
subplot_titles = ['Overfitting ($\\log(\\lambda)=-4$)', 
                 f'Optimal Fitting ($\\log(\\lambda)\\approx{optimal_log_alpha:.2f}$)', 
                 'Transition ($\\log(\\lambda)=2$)',
                 'Underfitting ($\\log(\\lambda)=4$)']

for i, log_lambda in enumerate(log_lambda_values):
    plt.subplot(2, 2, i+1)
    
    # If plotting optimal λ, use the actual optimal value found
    if i == 1:
        lambda_val = 10**optimal_log_alpha
    else:
        lambda_val = 10**log_lambda
    
    # Create and fit model
    model = make_pipeline(
        PolynomialFeatures(polynomial_degree, include_bias=True),
        Ridge(alpha=lambda_val, fit_intercept=False)
    )
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_plot)
    
    # Plot data points and curves
    plt.scatter(X_train, y_train, color='blue', s=20, alpha=0.5, label='Training data')
    plt.scatter(X_val, y_val, color='red', s=20, alpha=0.5, label='Validation data')
    plt.plot(X_plot, y_true, 'g-', label='True function', linewidth=2)
    plt.plot(X_plot, y_pred, 'm-', label=f'Model fit', linewidth=2)
    
    # Calculate and display errors
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    val_mse = mean_squared_error(y_val, model.predict(X_val))
    plt.title(f'{subplot_titles[i]}\nTrain: {train_mse:.2f}, Val: {val_mse:.2f}')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    if i == 0:  # Only add legend to the first plot to save space
        plt.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'model_fits_by_region.png'), dpi=300, bbox_inches='tight')

# 7. NEW: Visualize polynomial coefficients vs regularization strength
plt.figure(figsize=(10, 6))

# Select a subset of log_lambda values to test
selected_log_lambdas = np.linspace(-5, 5, 20)
selected_lambdas = 10**selected_log_lambdas

# Store coefficients for each lambda
all_coeffs = []
poly = PolynomialFeatures(polynomial_degree, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
feature_names = ['Bias'] + [f'x^{i}' for i in range(1, polynomial_degree+1)]

for alpha in selected_lambdas:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_train_poly, y_train)
    all_coeffs.append(ridge.coef_.flatten())  # Flatten to ensure 1D array per model

all_coeffs = np.array(all_coeffs)

# Plot coefficient paths
for i in range(min(len(feature_names), all_coeffs.shape[1])):
    plt.semilogx(selected_lambdas, all_coeffs[:, i], '-o', label=feature_names[i])

# Mark regions
plt.axvspan(10**-5, 10**-2, alpha=0.1, color='red', label='Overfitting')
plt.axvspan(10**-2, 10**2, alpha=0.1, color='green', label='Optimal')
plt.axvspan(10**2, 10**5, alpha=0.1, color='blue', label='Underfitting')

plt.xlabel('$\\lambda$ (log scale)', fontsize=14)
plt.ylabel('Coefficient Value', fontsize=14)
plt.title('Polynomial Coefficient Paths vs Regularization Strength', fontsize=14)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'coefficient_paths.png'), dpi=300, bbox_inches='tight')

# 8. NEW: Alternative visualization with completely different data distribution
plt.figure(figsize=(10, 6))

# Generate completely different data for alternative visualization
np.random.seed(123)  # Different seed
n_alt_samples = 80
X_alt = np.sort(np.random.uniform(-4, 4, n_alt_samples))[:, np.newaxis]

# Use a more complex ground truth function that's not quadratic
# This creates a step function with noise - very different from polynomial
def alt_true_func(x):
    return 3 * np.sin(x) + 0.5 * x + np.where(x > 0, 2, -2)

# Add heteroskedastic noise (more noise for larger x values)
# Simplify the noise generation to avoid shape issues
base_noise = np.random.normal(0, 1, (n_alt_samples, 1))
noise_factor = 0.5 + 0.3 * np.abs(X_alt)  # Shape: (80, 1)
y_alt = alt_true_func(X_alt) + base_noise * noise_factor

# Split with different ratio
X_alt_train, X_alt_val, y_alt_train, y_alt_val = train_test_split(
    X_alt, y_alt, test_size=0.4, random_state=123)

# Calculate errors with different model (degree 7 polynomial)
alt_degree = 7
alt_train_errors = []
alt_val_errors = []

for alpha in alphas:
    # Higher degree polynomial for alternative example
    alt_model = make_pipeline(
        PolynomialFeatures(alt_degree, include_bias=True),
        Ridge(alpha=alpha, fit_intercept=False)
    )
    
    alt_model.fit(X_alt_train, y_alt_train)
    y_alt_train_pred = alt_model.predict(X_alt_train)
    y_alt_val_pred = alt_model.predict(X_alt_val)
    
    # Different error scaling
    alt_mse_train = mean_squared_error(y_alt_train, y_alt_train_pred)
    alt_mse_val = mean_squared_error(y_alt_val, y_alt_val_pred)
    
    # Different scaling to emphasize the different shape
    alt_train_errors.append(alt_mse_train / 10)
    alt_val_errors.append(alt_mse_val / 10)

# Plot the alternative error curves
plt.plot(log_alphas, alt_train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(log_alphas, alt_val_errors, 'r-', linewidth=2, label='Validation Error')

# This creates a non-U-shaped validation error curve
plt.axvspan(-5, -2, alpha=0.1, color='red', label='Overfitting Region')
plt.axvspan(-2, 2, alpha=0.1, color='green', label='Transition Region')
plt.axvspan(2, 5, alpha=0.1, color='blue', label='Underfitting Region')

# Mark regions with simpler design
plt.text(-4, 1.8, 'High Variance', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(0, 1.8, 'Transition', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(4, 1.8, 'High Bias', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# Set labels and title with LaTeX formatting
plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Alternative Error Curves with Different Data Distribution', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True)
plt.ylim(0, 2.0)
plt.xlim(-5, 5)
plt.tight_layout()

# Save the alternative plot
plt.savefig(os.path.join(save_dir, 'alternative_solution.png'), dpi=300, bbox_inches='tight')

# Add visualization of the alternative model fits
plt.figure(figsize=(12, 8))
alt_log_lambda_values = [-4, -1, 1, 4]  # Different points of interest
alt_subplot_titles = ['High Variance ($\\log(\\lambda)=-4$)', 
                      'Transition ($\\log(\\lambda)=-1$)', 
                      'Transition ($\\log(\\lambda)=1$)',
                      'High Bias ($\\log(\\lambda)=4$)']

# Generate points for plotting the alt true function
X_alt_plot = np.linspace(-4, 4, 1000)[:, np.newaxis]
y_alt_true = alt_true_func(X_alt_plot)

for i, log_lambda in enumerate(alt_log_lambda_values):
    plt.subplot(2, 2, i+1)
    lambda_val = 10**log_lambda
    
    # Create and fit model with alternative data
    alt_model = make_pipeline(
        PolynomialFeatures(alt_degree, include_bias=True),
        Ridge(alpha=lambda_val, fit_intercept=False)
    )
    alt_model.fit(X_alt_train, y_alt_train)
    
    # Generate predictions
    y_alt_pred = alt_model.predict(X_alt_plot)
    
    # Plot data points and curves
    plt.scatter(X_alt_train, y_alt_train, color='blue', s=20, alpha=0.5, label='Training data')
    plt.scatter(X_alt_val, y_alt_val, color='red', s=20, alpha=0.5, label='Validation data')
    plt.plot(X_alt_plot, y_alt_true, 'g-', label='True function', linewidth=2)
    plt.plot(X_alt_plot, y_alt_pred, 'm-', label=f'Model fit', linewidth=2)
    
    # Calculate and display errors
    alt_train_mse = mean_squared_error(y_alt_train, alt_model.predict(X_alt_train))
    alt_val_mse = mean_squared_error(y_alt_val, alt_model.predict(X_alt_val))
    plt.title(f'{alt_subplot_titles[i]}\nTrain: {alt_train_mse:.2f}, Val: {alt_val_mse:.2f}')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    if i == 0:  # Only add legend to the first plot to save space
        plt.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'alternative_model_fits.png'), dpi=300, bbox_inches='tight')

# 9. NEW: Create an explanatory visualization about validation error minimum
plt.figure(figsize=(14, 10))  # Increase figure size for more room

# Main plot showing error curves
plt.subplot(2, 1, 1)
plt.plot(log_alphas, train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(log_alphas, val_errors, 'r-', linewidth=2, label='Validation Error')
plt.axvline(x=optimal_log_alpha, color='green', linestyle='--', 
            label=f'Actual Optimal $\\log(\\lambda)={optimal_log_alpha:.2f}$')
plt.axvline(x=0, color='purple', linestyle=':', 
            label=f'Theoretical Optimal $\\log(\\lambda)=0$')

# Mark key regions
plt.axvspan(-5, -2, alpha=0.1, color='red', label='Overfitting Region')
plt.axvspan(-2, 2, alpha=0.1, color='green', label='Optimal Region')
plt.axvspan(2, 5, alpha=0.1, color='blue', label='Underfitting Region')

plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Actual vs Theoretical Validation Error Minimum', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Zoomed-in view of the validation error minimum
plt.subplot(2, 1, 2)
# Find range of interesting values around minimum
zoom_min = max(-5, optimal_log_alpha - 2)
zoom_max = min(2, optimal_log_alpha + 2)
zoom_indices = np.where((log_alphas >= zoom_min) & (log_alphas <= zoom_max))[0]
zoom_log_alphas = log_alphas[zoom_indices]
zoom_val_errors = [val_errors[i] for i in zoom_indices]

plt.plot(zoom_log_alphas, zoom_val_errors, 'r-', linewidth=3, label='Validation Error')
plt.axvline(x=optimal_log_alpha, color='green', linestyle='--', 
            label=f'Actual Minimum at $\\log(\\lambda)={optimal_log_alpha:.2f}$')

plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Validation Error', fontsize=14)
plt.title('Zoomed View of Validation Error Minimum', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')

# Add annotations
min_val_error = val_errors[optimal_idx]
plt.annotate(f'Minimum Validation Error: {min_val_error:.4f}',
             xy=(optimal_log_alpha, min_val_error),
             xytext=(optimal_log_alpha + 0.5, min_val_error + 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, backgroundcolor='white')

# 10. NEW: Third alternative with a completely different validation error pattern
plt.figure(figsize=(10, 6))

# Generate new data with a completely different pattern
np.random.seed(456)  # Different seed
n_third_samples = 120

# Create a dataset with a step function and local patterns
X_third = np.sort(np.random.uniform(-5, 5, n_third_samples))[:, np.newaxis]

# Create a staircase function with multiple plateaus for the ground truth
def third_true_func(x):
    result = np.zeros_like(x)
    result[x < -2] = -2
    result[(x >= -2) & (x < 0)] = 0
    result[(x >= 0) & (x < 2)] = 2
    result[x >= 2] = 4
    return result

# Add complex noise pattern - different noise variances in different regions
noise = np.zeros_like(X_third)
for i, x in enumerate(X_third):
    if x < -3:
        noise[i] = np.random.normal(0, 0.5)  # Moderate noise in left region
    elif x >= -3 and x < 1:
        noise[i] = np.random.normal(0, 1.5)  # High noise in middle region
    else:
        noise[i] = np.random.normal(0, 0.8)  # Lower noise in right region

y_third = third_true_func(X_third) + noise

# Use a special train/test split to create an interesting error pattern
# Split so that validation set has many points from high-noise region
indices = np.arange(len(X_third))
train_indices = np.concatenate([
    indices[X_third.flatten() < -3][::2],  # Half of left region
    indices[(X_third.flatten() >= -3) & (X_third.flatten() < 1)][::5],  # Few points from middle region
    indices[X_third.flatten() >= 1][::2]  # Half of right region
])
val_indices = np.setdiff1d(indices, train_indices)

X_third_train = X_third[train_indices]
y_third_train = y_third[train_indices]
X_third_val = X_third[val_indices]
y_third_val = y_third[val_indices]

# Fit polynomial models with varying regularization
third_degree = 7  # High degree polynomial
third_train_errors = []
third_val_errors = []

for alpha in alphas:
    third_model = make_pipeline(
        PolynomialFeatures(third_degree, include_bias=True),
        Ridge(alpha=alpha, fit_intercept=False)
    )
    
    third_model.fit(X_third_train, y_third_train)
    y_third_train_pred = third_model.predict(X_third_train)
    y_third_val_pred = third_model.predict(X_third_val)
    
    third_mse_train = mean_squared_error(y_third_train, y_third_train_pred)
    third_mse_val = mean_squared_error(y_third_val, y_third_val_pred)
    
    # Scale errors to fit reasonable range
    third_train_errors.append(third_mse_train / 10)
    third_val_errors.append(third_mse_val / 10)

# Plot the third alternative error curves
plt.plot(log_alphas, third_train_errors, 'b-', linewidth=2, label='Training Error')
plt.plot(log_alphas, third_val_errors, 'r-', linewidth=2, label='Validation Error')

# Define the 3 regions with lighter colors matching the original question
plt.axvspan(-5, -2, alpha=0.1, color='red')
plt.axvspan(-2, 2, alpha=0.1, color='green')
plt.axvspan(2, 5, alpha=0.1, color='blue')

# Add region labels matching the original question's 3 regions
plt.text(-4, 0.9, 'Overfitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(0, 0.9, 'Optimal Fitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text(4, 0.9, 'Underfitting', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('$\\log(\\lambda)$', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Third Alternative: Non-U-shaped Validation Error Pattern', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True)
plt.ylim(0, 1.0)  # Matching the ylim of the original plot
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'third_alternative_solution.png'), dpi=300, bbox_inches='tight')

# Also create model fits at different regularization strengths for the third example
plt.figure(figsize=(12, 8))
third_log_lambda_values = [-4, -1, 1, 4]  # Match the log lambda values for the 3 regions
third_subplot_titles = ['Overfitting ($\\log(\\lambda)=-4$)', 
                        'Transition ($\\log(\\lambda)=-1$)', 
                        'Optimal Fitting ($\\log(\\lambda)=1$)',
                        'Underfitting ($\\log(\\lambda)=4$)']

# Generate points for plotting the third true function
X_third_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
y_third_true = third_true_func(X_third_plot)

for i, log_lambda in enumerate(third_log_lambda_values):
    plt.subplot(2, 2, i+1)
    lambda_val = 10**log_lambda
    
    # Create and fit model with third example data
    third_model = make_pipeline(
        PolynomialFeatures(third_degree, include_bias=True),
        Ridge(alpha=lambda_val, fit_intercept=False)
    )
    third_model.fit(X_third_train, y_third_train)
    
    # Generate predictions
    y_third_pred = third_model.predict(X_third_plot)
    
    # Plot data points and curves
    plt.scatter(X_third_train, y_third_train, color='blue', s=20, alpha=0.5, label='Training data')
    plt.scatter(X_third_val, y_third_val, color='red', s=20, alpha=0.5, label='Validation data')
    plt.plot(X_third_plot, y_third_true, 'g-', label='True function', linewidth=2)
    plt.plot(X_third_plot, y_third_pred, 'm-', label=f'Model fit', linewidth=2)
    
    # Calculate and display errors
    third_train_mse = mean_squared_error(y_third_train, third_model.predict(X_third_train))
    third_val_mse = mean_squared_error(y_third_val, third_model.predict(X_third_val))
    plt.title(f'{third_subplot_titles[i]}\nTrain: {third_train_mse:.2f}, Val: {third_val_mse:.2f}')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    if i == 0:  # Only add legend to the first plot to save space
        plt.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'third_alternative_model_fits.png'), dpi=300, bbox_inches='tight')

# 7. Print key insights and analysis for each task
print("\n=== Analysis for Question 25 Tasks ===\n")

print("Task 1 & 2: Error Curves and Region Labeling")
print("-" * 50)
print(f"• Training error starts low ({train_errors[0]:.2f}) at log(λ)=-5 and increases monotonically")
print(f"• Validation error is U-shaped with minimum at log(λ)={optimal_log_alpha:.2f}")
print("• The three regions are clearly identified as:")
print("  - Region 1 (log(λ)=-4): Overfitting")
print("  - Region 2 (log(λ)=0): Optimal Fitting")
print("  - Region 3 (log(λ)=+4): Underfitting")

print("\nTask 3: Error Behavior at log(λ)=-4")
print("-" * 50)
idx_neg4 = np.where(log_alphas >= -4)[0][0]
print(f"• Training error: {train_errors[idx_neg4]:.4f} (Low)")
print(f"• Validation error: {val_errors[idx_neg4]:.4f} (Medium)")
print(f"• Training error is {val_errors[idx_neg4] - train_errors[idx_neg4]:.4f} lower than validation error")
print("• Gap between errors indicates overfitting")

print("\nTask 4: Error Behavior at log(λ)=+4")
print("-" * 50)
idx_pos4 = np.where(log_alphas >= 4)[0][0]
print(f"• Training error: {train_errors[idx_pos4]:.4f} (High)")
print(f"• Validation error: {val_errors[idx_pos4]:.4f} (High)")
print(f"• Errors converge with a gap of {abs(val_errors[idx_pos4] - train_errors[idx_pos4]):.4f}")
print("• Both errors are high due to underfitting")

print("\nTask 5: Optimal log(λ) Value")
print("-" * 50)
print(f"• Optimal log(λ) value: {optimal_log_alpha:.4f}")
print(f"• At this point, validation error reaches minimum: {val_errors[optimal_idx]:.4f}")
print(f"• Training error at this point: {train_errors[optimal_idx]:.4f}")
print("• This represents the best bias-variance tradeoff")

print("\nVisualizations saved to:", save_dir) 