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

# Find the optimal lambda (minimum validation error)
optimal_idx = np.argmin(val_errors)
optimal_log_alpha = log_alphas[optimal_idx]

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