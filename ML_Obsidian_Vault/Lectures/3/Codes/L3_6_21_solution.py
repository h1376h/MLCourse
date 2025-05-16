import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate synthetic data
def true_function(x):
    """The true function that generates the data (without noise)"""
    return 0.5 + 3 * x - 2 * x**2 + 0.5 * x**3

def generate_data(n_samples=200, noise_level=0.5):
    """Generate synthetic data around the true function"""
    x = np.random.uniform(-2, 2, n_samples)
    noise = np.random.normal(0, noise_level, n_samples)
    y = true_function(x) + noise
    return x.reshape(-1, 1), y

# Generate a large dataset
X, y = generate_data(n_samples=500, noise_level=1.0)

# Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to compute learning curves
def compute_learning_curve(degree, X_train_full, y_train_full, X_test, y_test, train_sizes):
    """Compute learning curve for a polynomial model of the given degree"""
    train_errors = []
    test_errors = []
    
    for size in train_sizes:
        # Take a subset of the training data
        indices = np.random.choice(len(X_train_full), size=size, replace=False)
        X_subset = X_train_full[indices]
        y_subset = y_train_full[indices]
        
        # Create and fit the model
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=True),
            LinearRegression()
        )
        model.fit(X_subset, y_subset)
        
        # Compute training and test errors
        y_train_pred = model.predict(X_subset)
        y_test_pred = model.predict(X_test)
        
        train_error = mean_squared_error(y_subset, y_train_pred)
        test_error = mean_squared_error(y_test, y_test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return train_errors, test_errors

# Define training set sizes
train_sizes = [10, 20, 30, 50, 70, 100, 150, 200, 300, 400]

# Compute learning curves for different model complexities
# Linear model (degree=1) - likely to underfit
train_errors_linear, test_errors_linear = compute_learning_curve(1, X_train_full, y_train_full, X_test, y_test, train_sizes)

# Quadratic model (degree=2)
train_errors_quadratic, test_errors_quadratic = compute_learning_curve(2, X_train_full, y_train_full, X_test, y_test, train_sizes)

# Cubic model (degree=3) - about right for our true function
train_errors_cubic, test_errors_cubic = compute_learning_curve(3, X_train_full, y_train_full, X_test, y_test, train_sizes)

# High-degree polynomial (degree=10) - likely to overfit with small datasets
train_errors_complex, test_errors_complex = compute_learning_curve(10, X_train_full, y_train_full, X_test, y_test, train_sizes)

# Plot all learning curves on a single graph
plt.figure(figsize=(10, 6))

plt.plot(train_sizes, train_errors_linear, '-o', color='blue', alpha=0.7, label='Linear Model - Train')
plt.plot(train_sizes, test_errors_linear, '-o', color='blue', alpha=0.3, label='Linear Model - Test')

plt.plot(train_sizes, train_errors_quadratic, '-s', color='green', alpha=0.7, label='Quadratic Model - Train')
plt.plot(train_sizes, test_errors_quadratic, '-s', color='green', alpha=0.3, label='Quadratic Model - Test')

plt.plot(train_sizes, train_errors_cubic, '-^', color='orange', alpha=0.7, label='Cubic Model - Train')
plt.plot(train_sizes, test_errors_cubic, '-^', color='orange', alpha=0.3, label='Cubic Model - Test')

plt.plot(train_sizes, train_errors_complex, '-d', color='red', alpha=0.7, label='10th-degree Model - Train')
plt.plot(train_sizes, test_errors_complex, '-d', color='red', alpha=0.3, label='10th-degree Model - Test')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Learning Curves for Models with Different Complexities', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curves_combined.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot learning curves for underfitting (high bias) - Linear Model
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors_linear, 'o-', color='blue', label='Training Error')
plt.plot(train_sizes, test_errors_linear, 'o-', color='red', label='Test Error')
plt.fill_between(train_sizes, train_errors_linear, test_errors_linear, 
                 color='gray', alpha=0.2, label='Generalization Gap')

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Learning Curves: Linear Model (High Bias/Underfitting)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curve_underfitting.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot learning curves for overfitting (high variance) - 10th degree polynomial
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors_complex, 'o-', color='blue', label='Training Error')
plt.plot(train_sizes, test_errors_complex, 'o-', color='red', label='Test Error')
plt.fill_between(train_sizes, train_errors_complex, test_errors_complex, 
                 color='gray', alpha=0.2, label='Generalization Gap')

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Learning Curves: 10th-degree Polynomial (High Variance/Overfitting)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curve_overfitting.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot learning curves for well-balanced model (cubic)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors_cubic, 'o-', color='blue', label='Training Error')
plt.plot(train_sizes, test_errors_cubic, 'o-', color='red', label='Test Error')
plt.fill_between(train_sizes, train_errors_cubic, test_errors_cubic, 
                 color='gray', alpha=0.2, label='Generalization Gap')

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Learning Curves: Cubic Model (Well-balanced)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curve_balanced.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot the gap between training and test errors
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.array(test_errors_linear) - np.array(train_errors_linear), 
         'o-', color='blue', label='Linear Model')
plt.plot(train_sizes, np.array(test_errors_quadratic) - np.array(train_errors_quadratic), 
         's-', color='green', label='Quadratic Model')
plt.plot(train_sizes, np.array(test_errors_cubic) - np.array(train_errors_cubic), 
         '^-', color='orange', label='Cubic Model')
plt.plot(train_sizes, np.array(test_errors_complex) - np.array(train_errors_complex), 
         'd-', color='red', label='10th-degree Model')

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Generalization Gap (Test Error - Train Error)', fontsize=12)
plt.title('Generalization Gap vs. Training Set Size', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "generalization_gap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualize model fits for different training set sizes
def plot_model_fits(degree, sizes_to_plot=[10, 50, 400]):
    fig, axs = plt.subplots(1, len(sizes_to_plot), figsize=(15, 5))
    
    # Create a grid for smooth curve plotting
    X_grid = np.linspace(-2, 2, 1000).reshape(-1, 1)
    y_true_grid = true_function(X_grid.flatten())
    
    for i, size in enumerate(sizes_to_plot):
        # Take a subset of the training data
        indices = np.random.choice(len(X_train_full), size=size, replace=False)
        X_subset = X_train_full[indices]
        y_subset = y_train_full[indices]
        
        # Create and fit the model
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=True),
            LinearRegression()
        )
        model.fit(X_subset, y_subset)
        
        # Predict on the grid
        y_pred_grid = model.predict(X_grid)
        
        # Plot
        axs[i].scatter(X_subset, y_subset, color='blue', alpha=0.6, label='Training Data')
        axs[i].plot(X_grid, y_true_grid, 'k--', label='True Function')
        axs[i].plot(X_grid, y_pred_grid, 'r-', label='Model Fit')
        axs[i].set_xlim(-2, 2)
        axs[i].set_ylim(np.min(y_true_grid) - 5, np.max(y_true_grid) + 5)
        axs[i].set_title(f'n={size} examples', fontsize=12)
        if i == 0:
            axs[i].legend(fontsize=8)
    
    plt.suptitle(f'Model Fits: Polynomial Degree {degree}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"model_fits_degree_{degree}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Plot fits for different models
plot_model_fits(1)  # Linear model
plot_model_fits(10)  # Complex model

# Create visualization showing bias-variance tradeoff
train_sizes_all = [10, 50, 100, 200, 400]
degrees = list(range(1, 16))

bias_variance_data = []

for size in train_sizes_all:
    bias_squared = []
    variance = []
    total_error = []
    
    for degree in degrees:
        # Repeat multiple times to get average behavior
        n_repeats = 20
        predictions = []
        
        for _ in range(n_repeats):
            # Take a subset of the training data
            indices = np.random.choice(len(X_train_full), size=size, replace=True)
            X_subset = X_train_full[indices]
            y_subset = y_train_full[indices]
            
            # Create and fit the model
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=True),
                LinearRegression()
            )
            model.fit(X_subset, y_subset)
            
            # Predict on test data
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
        
        # Calculate bias and variance
        predictions = np.array(predictions)
        expected_pred = np.mean(predictions, axis=0)
        
        model_bias_squared = np.mean((expected_pred - y_test) ** 2)
        model_variance = np.mean(np.var(predictions, axis=0))
        
        bias_squared.append(model_bias_squared)
        variance.append(model_variance)
        total_error.append(model_bias_squared + model_variance)
    
    bias_variance_data.append((size, bias_squared, variance, total_error))

# Plot bias-variance tradeoff curves
plt.figure(figsize=(12, 8))

for size, bias_squared, variance, total_error in bias_variance_data:
    plt.plot(degrees, bias_squared, '--', label=f'Bias² (n={size})')
    plt.plot(degrees, variance, '-.', label=f'Variance (n={size})')
    plt.plot(degrees, total_error, '-', label=f'Total Error (n={size})')

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Bias-Variance Tradeoff for Different Training Set Sizes', fontsize=14)
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "bias_variance_tradeoff.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create a more focused plot for the first question
plt.figure(figsize=(12, 6))

# Underfitting (high bias)
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_errors_linear, 'o-', color='blue', label='Training Error')
plt.plot(train_sizes, test_errors_linear, 'o-', color='red', label='Test Error')
plt.fill_between(train_sizes, train_errors_linear, test_errors_linear, 
                 color='gray', alpha=0.2)

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('High Bias (Underfitting)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Overfitting (high variance)
plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_errors_complex, 'o-', color='blue', label='Training Error')
plt.plot(train_sizes, test_errors_complex, 'o-', color='red', label='Test Error')
plt.fill_between(train_sizes, train_errors_complex, test_errors_complex, 
                 color='gray', alpha=0.2)

plt.xscale('log')
plt.xlabel('Number of Training Examples', fontsize=12)
plt.title('High Variance (Overfitting)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curves_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Asymptotic performance comparison
plt.figure(figsize=(10, 6))
last_idx = -1  # Last training size

plt.bar(['Linear (degree=1)', 'Quadratic (degree=2)', 'Cubic (degree=3)', '10th-degree Polynomial'],
        [test_errors_linear[last_idx], test_errors_quadratic[last_idx], 
         test_errors_cubic[last_idx], test_errors_complex[last_idx]],
        color=['blue', 'green', 'orange', 'red'])

plt.ylabel('Test Error (MSE) with Maximum Training Data', fontsize=12)
plt.title('Asymptotic Performance Comparison', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "asymptotic_performance.png"), dpi=300, bbox_inches='tight')

print(f"All visualizations have been saved to: {save_dir}")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Linear Model (degree=1): Final train error = {train_errors_linear[-1]:.4f}, test error = {test_errors_linear[-1]:.4f}")
print(f"Quadratic Model (degree=2): Final train error = {train_errors_quadratic[-1]:.4f}, test error = {test_errors_quadratic[-1]:.4f}")
print(f"Cubic Model (degree=3): Final train error = {train_errors_cubic[-1]:.4f}, test error = {test_errors_cubic[-1]:.4f}")
print(f"10th-degree Model: Final train error = {train_errors_complex[-1]:.4f}, test error = {test_errors_complex[-1]:.4f}")

print("\nGeneralization gap (test error - train error):")
print(f"Linear Model (degree=1): Initial gap = {(test_errors_linear[0]-train_errors_linear[0]):.4f}, Final gap = {(test_errors_linear[-1]-train_errors_linear[-1]):.4f}")
print(f"Quadratic Model (degree=2): Initial gap = {(test_errors_quadratic[0]-train_errors_quadratic[0]):.4f}, Final gap = {(test_errors_quadratic[-1]-train_errors_quadratic[-1]):.4f}")
print(f"Cubic Model (degree=3): Initial gap = {(test_errors_cubic[0]-train_errors_cubic[0]):.4f}, Final gap = {(test_errors_cubic[-1]-train_errors_cubic[-1]):.4f}")
print(f"10th-degree Model: Initial gap = {(test_errors_complex[0]-train_errors_complex[0]):.4f}, Final gap = {(test_errors_complex[-1]-train_errors_complex[-1]):.4f}")

# Mathematical explanation of learning curve behavior
print("\nMathematical explanation of learning curve behavior:")
print("Training error typically increases with more examples because:")
print("  - With few examples, complex models can memorize the data (especially for high-variance models)")
print("  - As more examples are added, it becomes harder to fit all points perfectly")
print("  - Mathematically, with n examples and p parameters, if n < p, perfect interpolation is possible")
print("  - When n > p, the model must balance errors across all examples")

print("\nTest error typically decreases with more examples because:")
print("  - More training examples lead to better generalization")
print("  - The model learns the true underlying pattern instead of noise")
print("  - Statistically, variance in parameter estimation decreases with more data as O(1/n)")
print("  - For well-specified models, test error converges to irreducible error (noise level)")

# Diagnostic information
print("\nDiagnostic information based on learning curves:")
print("High bias indicators:")
print("  - High training error (even with many examples)")
print("  - Small gap between training and test error")
print("  - Both errors plateau at similar, high values")

print("\nHigh variance indicators:")
print("  - Low training error (especially with few examples)")
print("  - Large gap between training and test error")
print("  - Test error decreases steadily with more training examples")

print("\nCombination of high bias and high variance:")
print("  - Moderate to high training error")
print("  - Significant gap between training and test error")
print("  - Test error far from optimal, even with many examples") 