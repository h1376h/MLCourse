import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with noise
def generate_data(n_samples=200, noise=0.5):
    """Generate synthetic data for regression with polynomial relationship"""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    # True function: y = x^3 - 6x^2 + 4x + 10 + noise
    y_true = X**3 - 6*X**2 + 4*X + 10
    y = y_true + noise * np.random.randn(n_samples, 1)
    return X, y, y_true

X, y, y_true = generate_data(n_samples=100, noise=2.0)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Ensure y is flattened for compatibility with sklearn functions
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Create a high-degree polynomial model to demonstrate overfitting
degree = 15
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
X_test_poly = poly.transform(X_test)

# Print explanation
print("\nEarly Stopping as Implicit Regularization")
print("=========================================")
print("This code demonstrates early stopping as an implicit regularization technique")
print("by comparing it with explicit regularization methods like Ridge and Lasso.")
print("\nSetup:")
print(f"- Generated synthetic data with polynomial relationship and noise")
print(f"- Split data into training ({len(X_train)} samples), validation ({len(X_val)} samples), and test ({len(X_test)} samples) sets")
print(f"- Created a high-degree polynomial model (degree={degree}) to demonstrate overfitting")

# Implement a custom gradient descent with early stopping
def gradient_descent(X, y, X_val, y_val, learning_rate=0.01, max_iterations=1000, 
                      tol=1e-6, display_step=100, return_path=False):
    """
    Implement gradient descent for linear regression with early stopping
    Returns: Coefficients, training errors, validation errors, and weight paths
    """
    n_samples, n_features = X.shape
    
    # Initialize weights randomly
    weights = np.random.randn(n_features) * 0.1
    
    # Lists to store errors during training
    train_errors = []
    val_errors = []
    best_val_error = float('inf')
    best_weights = None
    no_improvement_count = 0
    patience = 20  # Early stopping patience
    
    # Store the path of weights during training
    weight_path = []
    
    # Gradient descent iterations
    for iteration in range(max_iterations):
        # Compute predictions and error
        y_pred = X @ weights
        error = y_pred - y
        
        # Compute gradients and update weights
        gradients = (1/n_samples) * (X.T @ error)
        weights = weights - learning_rate * gradients
        
        # Compute training and validation errors
        train_mse = mean_squared_error(y, X @ weights)
        val_mse = mean_squared_error(y_val, X_val @ weights)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
        
        if return_path:
            weight_path.append(weights.copy())
        
        # Early stopping logic
        if val_mse < best_val_error:
            best_val_error = val_mse
            best_weights = weights.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If no improvement for 'patience' iterations, stop
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {iteration}")
            break
        
        # Print progress
        if (iteration+1) % display_step == 0:
            print(f"Iteration {iteration+1}/{max_iterations}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
    
    # If we didn't stop early, use the best weights found
    if iteration == max_iterations - 1:
        print(f"Reached maximum iterations: {max_iterations}")
        weights = best_weights
    
    early_stop_iteration = iteration - patience
    return weights, train_errors, val_errors, weight_path, early_stop_iteration

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)
X_test_scaled = scaler.transform(X_test_poly)

print("\nRunning gradient descent with early stopping...")
start_time = time.time()
weights, train_errors, val_errors, weight_path, early_stop_iter = gradient_descent(
    X_train_scaled, y_train, X_val_scaled, y_val, 
    learning_rate=0.01, max_iterations=1000, return_path=True
)
elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")

# Visualization 1: Training and validation error curves with early stopping point
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error', color='blue')
plt.plot(val_errors, label='Validation Error', color='red')
plt.axvline(x=early_stop_iter, linestyle='--', color='green', 
            label=f'Early Stopping (iter={early_stop_iter})')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.yscale('log')
plt.title('Error Curves During Training with Early Stopping', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'early_stopping_errors.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Model complexity (measured by L2 norm of weights)
plt.figure(figsize=(10, 6))
weight_norms = [np.linalg.norm(w) for w in weight_path]
plt.plot(weight_norms, label='L2 Norm of Weights', color='purple')
plt.axvline(x=early_stop_iter, linestyle='--', color='green', 
            label=f'Early Stopping (iter={early_stop_iter})')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('L2 Norm of Weights', fontsize=12)
plt.title('Model Complexity vs. Iterations', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'model_complexity.png'), dpi=300, bbox_inches='tight')

# Function to make predictions using weights
def predict(X, weights):
    return X @ weights

# Visualization 3: Fitted models at different iterations
plt.figure(figsize=(12, 8))

# Generate a grid of x values for smooth curves
X_grid = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
X_grid_poly = poly.transform(X_grid)
X_grid_scaled = scaler.transform(X_grid_poly)

# Generate true function values for the grid points
y_true_grid = X_grid**3 - 6*X_grid**2 + 4*X_grid + 10

iter_to_plot = [0, 10, 50, early_stop_iter, len(weight_path)-1]
colors = ['purple', 'blue', 'green', 'orange', 'red']
labels = ['Initial', 'Early Iterations', 'Mid Training', 'Early Stop Point', 'Final']

plt.scatter(X_train, y_train, color='black', alpha=0.5, label='Training Data')
plt.plot(X_grid, y_true_grid, '--', color='black', label='True Function', linewidth=2)

for i, iter_idx in enumerate(iter_to_plot):
    if iter_idx < len(weight_path):
        y_pred = predict(X_grid_scaled, weight_path[iter_idx])
        plt.plot(X_grid, y_pred, '-', color=colors[i], label=f'{labels[i]} (iter={iter_idx})', linewidth=2)

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Model Fit at Different Iterations', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'model_iterations.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Comparison with explicit regularization methods
print("\nComparing with explicit regularization methods...")

# Compute test error for model with early stopping
early_stop_weights = weight_path[early_stop_iter]
early_stop_test_error = mean_squared_error(y_test, X_test_scaled @ early_stop_weights)

# Train models with different regularization methods
models = {
    "No Regularization": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
}

test_errors = {"Early Stopping": early_stop_test_error}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_errors[name] = mean_squared_error(y_test, y_pred)
    print(f"{name} - Test MSE: {test_errors[name]:.4f}")

plt.figure(figsize=(10, 6))
bar_colors = ['green', 'blue', 'red', 'purple']
plt.bar(list(test_errors.keys()), list(test_errors.values()), color=bar_colors)
plt.ylabel('Test Mean Squared Error', fontsize=12)
plt.title('Comparison of Regularization Methods', fontsize=14)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Determine optimal stopping point with k-fold cross-validation
from sklearn.model_selection import KFold

print("\nDemonstrating cross-validation to find optimal stopping point...")

# Simulate k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Combine train and validation sets for cross-validation
X_train_val = np.vstack((X_train, X_val))
y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).ravel()

X_train_val_poly = poly.transform(X_train_val)
X_train_val_scaled = scaler.transform(X_train_val_poly)

# Store cross-validation results
max_iter_cv = 300  # Reduce for speed
cv_val_errors = np.zeros((k_folds, max_iter_cv))
optimal_iters = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val_scaled)):
    X_train_fold, X_val_fold = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
    y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
    
    print(f"Processing fold {fold+1}/{k_folds}...")
    
    _, train_fold_errors, val_fold_errors, _, _ = gradient_descent(
        X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
        learning_rate=0.01, max_iterations=max_iter_cv, 
        return_path=False, display_step=max_iter_cv//2
    )
    
    # Store validation errors for this fold up to max_iter_cv
    n_iters = min(len(val_fold_errors), max_iter_cv)
    cv_val_errors[fold, :n_iters] = val_fold_errors[:n_iters]
    
    # Find optimal iteration for this fold
    optimal_iter = np.argmin(val_fold_errors[:n_iters])
    optimal_iters.append(optimal_iter)
    print(f"Optimal iteration for fold {fold+1}: {optimal_iter}")

# Calculate mean validation error across folds
mean_cv_errors = np.mean(cv_val_errors, axis=0)
optimal_iter_cv = np.argmin(mean_cv_errors)
print(f"Overall optimal iteration from cross-validation: {optimal_iter_cv}")

plt.figure(figsize=(10, 6))
for fold in range(k_folds):
    plt.plot(cv_val_errors[fold], alpha=0.3, label=f'Fold {fold+1}' if fold == 0 else None)
    plt.axvline(x=optimal_iters[fold], linestyle=':', color='gray', alpha=0.3)

plt.plot(mean_cv_errors, linewidth=2, color='red', label='Mean Validation Error')
plt.axvline(x=optimal_iter_cv, linestyle='--', color='green', 
            label=f'Optimal Iteration ({optimal_iter_cv})')

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Validation MSE', fontsize=12)
plt.yscale('log')
plt.title('Cross-Validation for Optimal Stopping Point', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cross_validation.png'), dpi=300, bbox_inches='tight')

# Visualization 6: Animated GIF of the model evolution through iterations
print("\nGenerating animation of model evolution...")

fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    ax.scatter(X_train, y_train, color='black', alpha=0.5, label='Training Data')
    ax.plot(X_grid, y_true_grid, '--', color='black', label='True Function', linewidth=2)
    
    # Get predictions for current iteration
    if frame < len(weight_path):
        y_pred = predict(X_grid_scaled, weight_path[frame])
        ax.plot(X_grid, y_pred, '-', color='red', label=f'Iteration {frame}', linewidth=2)
    
    # Mark early stopping point
    if frame == early_stop_iter:
        ax.set_title(f'Iteration {frame} - Early Stopping Point', fontsize=14, color='green')
    else:
        ax.set_title(f'Iteration {frame}', fontsize=14)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True)
    ax.legend()
    
    # Focus on the interesting region
    ax.set_ylim(-10, 30)
    return ax,

num_frames = min(100, len(weight_path))  # Limit frames for file size
step = max(1, len(weight_path) // num_frames)
frames = list(range(0, len(weight_path), step))
if early_stop_iter not in frames:  # Make sure early stopping point is included
    frames.append(early_stop_iter)
    frames.sort()

anim = FuncAnimation(fig, update, frames=frames, blit=False)
anim.save(os.path.join(save_dir, 'model_evolution.gif'), writer='pillow', fps=5, dpi=100)

# Visualization 7: Regularization path comparison
print("\nComparing regularization paths...")

# Set up alphas for Ridge and Lasso
alphas = np.logspace(-3, 3, 7)

# Train models with different alphas
ridge_weights = []
lasso_weights = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_weights.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_weights.append(lasso.coef_)

# Plot weight paths for Ridge
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(min(10, X_train_scaled.shape[1])):  # Limit to first 10 features for visibility
    coef_path = [weights[i] for weights in ridge_weights]
    plt.semilogx(alphas, coef_path, label=f'Feature {i}' if i < 5 else None)

plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Ridge Regularization Path', fontsize=14)
if X_train_scaled.shape[1] > 5:
    plt.legend(loc='upper right')
plt.grid(True)

# Plot weight paths for Lasso
plt.subplot(1, 2, 2)
for i in range(min(10, X_train_scaled.shape[1])):  # Limit to first 10 features for visibility
    coef_path = [weights[i] for weights in lasso_weights]
    plt.semilogx(alphas, coef_path, label=f'Feature {i}' if i < 5 else None)

plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Lasso Regularization Path', fontsize=14)
if X_train_scaled.shape[1] > 5:
    plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_paths.png'), dpi=300, bbox_inches='tight')

# Final plot: Early stopping effect on different complexity models
print("\nDemonstrating early stopping effect on different model complexities...")

# Train models of different complexities
degrees = [1, 3, 5, 10, 15]
early_stop_iters = []
test_errors_by_degree = []

for deg in degrees:
    print(f"Training polynomial model with degree {deg}...")
    poly_deg = PolynomialFeatures(degree=deg)
    X_train_poly_deg = poly_deg.fit_transform(X_train)
    X_val_poly_deg = poly_deg.transform(X_val)
    X_test_poly_deg = poly_deg.transform(X_test)
    
    # Scale the data
    scaler_deg = StandardScaler()
    X_train_scaled_deg = scaler_deg.fit_transform(X_train_poly_deg)
    X_val_scaled_deg = scaler_deg.transform(X_val_poly_deg)
    X_test_scaled_deg = scaler_deg.transform(X_test_poly_deg)
    
    # Train with gradient descent and early stopping
    weights_deg, _, _, _, early_stop_iter_deg = gradient_descent(
        X_train_scaled_deg, y_train, X_val_scaled_deg, y_val, 
        learning_rate=0.01, max_iterations=500, return_path=False
    )
    
    # Compute test error
    y_pred_test = X_test_scaled_deg @ weights_deg
    test_error = mean_squared_error(y_test, y_pred_test)
    
    early_stop_iters.append(early_stop_iter_deg)
    test_errors_by_degree.append(test_error)

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(degrees)), early_stop_iters, tick_label=[f"Deg {d}" for d in degrees])
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Early Stopping Iteration', fontsize=12)
plt.title('Early Stopping Iteration vs. Model Complexity', fontsize=14)
plt.grid(True, axis='y')

plt.subplot(1, 2, 2)
plt.bar(range(len(degrees)), test_errors_by_degree, tick_label=[f"Deg {d}" for d in degrees])
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Test MSE', fontsize=12)
plt.title('Test Error vs. Model Complexity', fontsize=14)
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary
print("\nSummary of Early Stopping as Regularization:")
print("---------------------------------------------")
print("1. Early stopping prevents overfitting by stopping training when validation error starts increasing")
print("2. It acts as implicit regularization by controlling model complexity (weight magnitudes)")
print("3. Compared to explicit methods:")
print(f"   - Early Stopping Test MSE: {test_errors['Early Stopping']:.4f}")
print(f"   - Ridge (L2) Test MSE: {test_errors['Ridge (L2)']:.4f}")
print(f"   - Lasso (L1) Test MSE: {test_errors['Lasso (L1)']:.4f}")
print("4. Cross-validation can determine the optimal stopping point")
print(f"   - Optimal iteration from CV: {optimal_iter_cv}")
print("5. Early stopping point varies with model complexity")
print(f"   - More complex models typically require earlier stopping") 