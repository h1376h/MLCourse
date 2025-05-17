import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import os
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
images_dir = os.path.join(parent_dir, "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data from a degree 3 polynomial: y = 1 + 2x - x^2 + 0.5x^3 + noise
def true_function(x):
    return 1 + 2*x - x**2 + 0.5*x**3

# Generate data
n_samples = 100
X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
y_true = true_function(X.flatten())
noise = np.random.normal(0, 1, n_samples)  # Add Gaussian noise
y = y_true + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create models with different complexities
degrees = [1, 3, 10]  # Linear, Correct complexity, Overfit
colors = ['blue', 'green', 'red']
model_names = ['Linear (degree 1)', 'Polynomial (degree 3)', 'Polynomial (degree 10)']
models = []

for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
    linear_regression = LinearRegression()
    model = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression)
    ])
    model.fit(X_train, y_train)
    models.append(model)

# Calculate performance metrics for each model
train_errors = []
test_errors = []
bias_values = []
variance_values = []

for model in models:
    # Training error
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_errors.append(train_mse)
    
    # Test error
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_errors.append(test_mse)
    
    # Approximate bias (error between true function and model prediction)
    y_true_test = true_function(X_test.flatten())
    bias_squared = np.mean((y_true_test - np.mean(y_test_pred))**2)
    bias_values.append(bias_squared)
    
    # Approximate variance (variance of predictions)
    # In reality, we would need multiple training sets, but we'll simulate it with cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    model_variance = np.var(-cv_scores)
    variance_values.append(model_variance)

# Visualization 1: Data and fitted models
plt.figure(figsize=(14, 8))

# Plot the training data
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training data')

# Plot the true function
X_plot = np.linspace(-3.5, 3.5, 1000).reshape(-1, 1)
y_plot_true = true_function(X_plot.flatten())
plt.plot(X_plot, y_plot_true, color='black', linestyle='--', label='True function (degree 3)')

# Plot the fitted models
for i, (model, color, name) in enumerate(zip(models, colors, model_names)):
    y_plot_pred = model.predict(X_plot)
    plt.plot(X_plot, y_plot_pred, color=color, label=f'{name}, Train MSE: {train_errors[i]:.2f}, Test MSE: {test_errors[i]:.2f}')

plt.title('Polynomial Regression Models with Different Degrees')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'model_fits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Training and test errors
plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(degrees)) - 0.2, train_errors, width=0.4, color='blue', label='Training MSE')
plt.bar(np.arange(len(degrees)) + 0.2, test_errors, width=0.4, color='red', label='Test MSE')
plt.xticks(np.arange(len(degrees)), [f'Degree {d}' for d in degrees])
plt.ylabel('Mean Squared Error')
plt.title('Training vs Test Error for Models with Different Complexity')
plt.legend()
plt.grid(True, axis='y')
plt.savefig(os.path.join(save_dir, 'errors_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Bias-Variance decomposition
plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(degrees)) - 0.2, bias_values, width=0.4, color='green', label='Bias²')
plt.bar(np.arange(len(degrees)) + 0.2, variance_values, width=0.4, color='purple', label='Variance')
plt.xticks(np.arange(len(degrees)), [f'Degree {d}' for d in degrees])
plt.ylabel('Error Component')
plt.title('Bias-Variance Decomposition for Models with Different Complexity')
plt.legend()
plt.grid(True, axis='y')
plt.savefig(os.path.join(save_dir, 'bias_variance_decomposition.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Model complexity vs Error components
plt.figure(figsize=(12, 6))
complexity = np.array(degrees)
x_smooth = np.linspace(1, 10, 100)

plt.plot(x_smooth, 1/x_smooth + 0.5, 'g-', linewidth=2, label='Typical Bias Trend')
plt.plot(x_smooth, x_smooth/10, 'purple', linewidth=2, label='Typical Variance Trend')
plt.plot(x_smooth, 1/x_smooth + 0.5 + x_smooth/10, 'r-', linewidth=2, label='Total Error Trend')

# Mark actual points
plt.scatter(degrees, bias_values, color='green', s=100, zorder=10)
plt.scatter(degrees, variance_values, color='purple', s=100, zorder=10)
plt.scatter(degrees, np.array(bias_values) + np.array(variance_values), color='red', s=100, zorder=10)

plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Error Component')
plt.title('Bias-Variance Tradeoff with Increasing Model Complexity')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 5: Individual predictions and their variance
plt.figure(figsize=(16, 12))

for i, (model, degree, color, name) in enumerate(zip(models, degrees, colors, model_names)):
    plt.subplot(3, 1, i+1)
    
    # Get multiple predictions using bootstrapping
    n_bootstraps = 100
    y_pred_bootstraps = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sampling with replacement
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        
        # Train model on bootstrap sample
        bootstrap_model = Pipeline([
            ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=True)),
            ("linear_regression", LinearRegression())
        ])
        bootstrap_model.fit(X_bootstrap, y_bootstrap)
        
        # Predict on test grid
        y_pred = bootstrap_model.predict(X_plot)
        y_pred_bootstraps.append(y_pred)
    
    # Convert to numpy array for easier calculations
    y_pred_bootstraps = np.array(y_pred_bootstraps)
    
    # Calculate mean and standard deviation of predictions
    y_pred_mean = np.mean(y_pred_bootstraps, axis=0)
    y_pred_std = np.std(y_pred_bootstraps, axis=0)
    
    # Plot the true function
    plt.plot(X_plot, y_plot_true, color='black', linestyle='--', label='True function')
    
    # Plot mean prediction
    plt.plot(X_plot, y_pred_mean, color=color, label=f'Mean prediction')
    
    # Plot confidence interval (± 2 std)
    plt.fill_between(X_plot.flatten(), 
                     y_pred_mean - 2*y_pred_std, 
                     y_pred_mean + 2*y_pred_std, 
                     color=color, alpha=0.3, label='Prediction variance')
    
    # Plot training data
    plt.scatter(X_train, y_train, color='gray', alpha=0.2, label='Training data')
    
    # Calculate metrics for this degree
    bias_squared = np.mean((y_plot_true - y_pred_mean)**2)
    variance = np.mean(y_pred_std**2)
    
    plt.title(f'{name} - Bias²: {bias_squared:.2f}, Variance: {variance:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prediction_intervals.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print summary of results
print("\nBias-Variance Tradeoff in Polynomial Regression")
print("=" * 50)
print(f"{'Model':<25} {'Train MSE':<12} {'Test MSE':<12} {'Bias²':<12} {'Variance':<12}")
print("-" * 70)
for i, degree in enumerate(degrees):
    print(f"Polynomial (degree {degree}):{'':<5} {train_errors[i]:8.4f}{'':<4} {test_errors[i]:8.4f}{'':<4} {bias_values[i]:8.4f}{'':<4} {variance_values[i]:8.4f}")

print("\nAnalysis:")
print("1. Linear regression (degree 1): High bias, low variance")
print("2. Polynomial regression (degree 3): Low bias, moderate variance")
print("3. Polynomial regression (degree 10): Low bias, high variance")
print("\nThe underlying data was generated from a polynomial of degree 3.")
print("Therefore, the degree 3 model has the best balance of bias and variance.")
print(f"\nVisualizations saved to: {save_dir}")