import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Define the true function (sine curve)
def true_function(x):
    return np.sin(x)

# Generate data points
x_range = np.linspace(0, 2*np.pi, 1000)
y_true = true_function(x_range)

# Function to generate noisy data points
def generate_data(n_samples, noise_level=0.3):
    x = np.random.uniform(0, 2*np.pi, n_samples)
    y = true_function(x) + np.random.normal(0, noise_level, n_samples)
    return x, y

# Function to train constant model (Model A)
def train_constant_model(x_train, y_train):
    return np.mean(y_train)

# Function to train linear model (Model B)
def train_linear_model(x_train, y_train):
    X = x_train.reshape(-1, 1)
    model = LinearRegression().fit(X, y_train)
    return model

# Function to train regularized linear model
def train_regularized_model(x_train, y_train, alpha):
    X = x_train.reshape(-1, 1)
    model = Ridge(alpha=alpha).fit(X, y_train)
    return model

# Function to evaluate models
def evaluate_models(x_train, y_train, x_test, y_test, const_model, linear_model):
    # Constant model predictions
    y_pred_const_train = np.full_like(y_train, const_model)
    y_pred_const_test = np.full_like(y_test, const_model)
    
    # Linear model predictions
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)
    y_pred_linear_train = linear_model.predict(X_train)
    y_pred_linear_test = linear_model.predict(X_test)
    
    # Calculate errors
    const_train_error = mean_squared_error(y_train, y_pred_const_train)
    const_test_error = mean_squared_error(y_test, y_pred_const_test)
    linear_train_error = mean_squared_error(y_train, y_pred_linear_train)
    linear_test_error = mean_squared_error(y_test, y_pred_linear_test)
    
    return {
        'const_train_error': const_train_error,
        'const_test_error': const_test_error,
        'linear_train_error': linear_train_error,
        'linear_test_error': linear_test_error,
        'y_pred_const_train': y_pred_const_train,
        'y_pred_const_test': y_pred_const_test,
        'y_pred_linear_train': y_pred_linear_train,
        'y_pred_linear_test': y_pred_linear_test
    }

# Print explanations and insights
print("\nBias-Variance Tradeoff in Model Selection")
print("=========================================")
print("\n1. Model Families:")
print("   - Model A: Constant model (horizontal line)")
print("   - Model B: Linear model (sloped line)")
print("   - True function: Sine curve")

print("\n2. Key Concepts:")
print("   - Bias: Error from incorrect assumptions in the learning algorithm")
print("   - Variance: Error from sensitivity to small fluctuations in the training set")
print("   - Bias-variance tradeoff: Balancing the model's ability to fit the training data vs. generalize to new data")

# Visualization 1: Few training examples (n=2)
n_samples_small = 2
x_train_small, y_train_small = generate_data(n_samples_small)
x_test, y_test = generate_data(100)  # Large test set

# Train models
const_model_small = train_constant_model(x_train_small, y_train_small)
linear_model_small = train_linear_model(x_train_small, y_train_small)

# Evaluate models
results_small = evaluate_models(
    x_train_small, y_train_small, 
    x_test, y_test, 
    const_model_small, linear_model_small
)

# Print results for n=2
print(f"\n3. Results with {n_samples_small} training examples:")
print(f"   - Constant model training error: {results_small['const_train_error']:.4f}")
print(f"   - Constant model test error: {results_small['const_test_error']:.4f}")
print(f"   - Linear model training error: {results_small['linear_train_error']:.4f}")
print(f"   - Linear model test error: {results_small['linear_test_error']:.4f}")

# Plot the results for n=2
plt.figure(figsize=(12, 8))
plt.plot(x_range, y_true, 'g-', alpha=0.7, label='True function (sine curve)')
plt.scatter(x_train_small, y_train_small, color='blue', s=100, label='Training data')
plt.scatter(x_test, y_test, color='gray', s=20, alpha=0.3, label='Test data')

# Plot constant model prediction
plt.axhline(y=const_model_small, color='red', linestyle='-', label=f'Constant model (b={const_model_small:.2f})')

# Plot linear model prediction
x_plot = np.linspace(0, 2*np.pi, 100)
X_plot = x_plot.reshape(-1, 1)
y_plot_linear = linear_model_small.predict(X_plot)
plt.plot(x_plot, y_plot_linear, 'orange', label=f'Linear model (a={linear_model_small.coef_[0]:.2f}, b={linear_model_small.intercept_:.2f})')

plt.title(f'Model Performance with {n_samples_small} Training Examples', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig(os.path.join(save_dir, f"bias_variance_n{n_samples_small}.png"), dpi=300, bbox_inches='tight')

# Visualization 2: More training examples (n=20)
n_samples_large = 20
x_train_large, y_train_large = generate_data(n_samples_large)

# Train models
const_model_large = train_constant_model(x_train_large, y_train_large)
linear_model_large = train_linear_model(x_train_large, y_train_large)

# Evaluate models
results_large = evaluate_models(
    x_train_large, y_train_large, 
    x_test, y_test, 
    const_model_large, linear_model_large
)

# Print results for n=20
print(f"\n4. Results with {n_samples_large} training examples:")
print(f"   - Constant model training error: {results_large['const_train_error']:.4f}")
print(f"   - Constant model test error: {results_large['const_test_error']:.4f}")
print(f"   - Linear model training error: {results_large['linear_train_error']:.4f}")
print(f"   - Linear model test error: {results_large['linear_test_error']:.4f}")

# Plot the results for n=20
plt.figure(figsize=(12, 8))
plt.plot(x_range, y_true, 'g-', alpha=0.7, label='True function (sine curve)')
plt.scatter(x_train_large, y_train_large, color='blue', s=100, label='Training data')
plt.scatter(x_test, y_test, color='gray', s=20, alpha=0.3, label='Test data')

# Plot constant model prediction
plt.axhline(y=const_model_large, color='red', linestyle='-', label=f'Constant model (b={const_model_large:.2f})')

# Plot linear model prediction
x_plot = np.linspace(0, 2*np.pi, 100)
X_plot = x_plot.reshape(-1, 1)
y_plot_linear = linear_model_large.predict(X_plot)
plt.plot(x_plot, y_plot_linear, 'orange', label=f'Linear model (a={linear_model_large.coef_[0]:.2f}, b={linear_model_large.intercept_:.2f})')

plt.title(f'Model Performance with {n_samples_large} Training Examples', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig(os.path.join(save_dir, f"bias_variance_n{n_samples_large}.png"), dpi=300, bbox_inches='tight')

# Visualization 3: Training and test error as a function of sample size
sample_sizes = np.arange(2, 101, 5)
const_train_errors = []
const_test_errors = []
linear_train_errors = []
linear_test_errors = []

for n in sample_sizes:
    # Generate training data
    x_train, y_train = generate_data(n)
    
    # Train models
    const_model = train_constant_model(x_train, y_train)
    linear_model = train_linear_model(x_train, y_train)
    
    # Evaluate models
    results = evaluate_models(
        x_train, y_train, 
        x_test, y_test, 
        const_model, linear_model
    )
    
    # Store errors
    const_train_errors.append(results['const_train_error'])
    const_test_errors.append(results['const_test_error'])
    linear_train_errors.append(results['linear_train_error'])
    linear_test_errors.append(results['linear_test_error'])

# Plot training and test error curves
plt.figure(figsize=(12, 8))
plt.plot(sample_sizes, const_train_errors, 'r--', label='Constant model - Training error')
plt.plot(sample_sizes, const_test_errors, 'r-', label='Constant model - Test error')
plt.plot(sample_sizes, linear_train_errors, 'b--', label='Linear model - Training error')
plt.plot(sample_sizes, linear_test_errors, 'b-', label='Linear model - Test error')

plt.title('Training and Test Error vs. Number of Training Examples', fontsize=14)
plt.xlabel('Number of Training Examples', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "error_vs_sample_size.png"), dpi=300, bbox_inches='tight')

# Visualization 4: Regularization effect
alphas = [0, 0.1, 1, 10, 100]
n_samples_reg = 10
x_train_reg, y_train_reg = generate_data(n_samples_reg, noise_level=0.5)

plt.figure(figsize=(12, 8))
plt.plot(x_range, y_true, 'g-', alpha=0.7, label='True function (sine curve)')
plt.scatter(x_train_reg, y_train_reg, color='blue', s=100, label='Training data')

# Train and plot regularized models
for alpha in alphas:
    reg_model = train_regularized_model(x_train_reg, y_train_reg, alpha)
    y_plot_reg = reg_model.predict(X_plot)
    plt.plot(x_plot, y_plot_reg, label=f'Ridge (α={alpha})')
    
    # Calculate training and test errors
    X_train_reg = x_train_reg.reshape(-1, 1)
    y_pred_reg_train = reg_model.predict(X_train_reg)
    y_pred_reg_test = reg_model.predict(x_test.reshape(-1, 1))
    
    reg_train_error = mean_squared_error(y_train_reg, y_pred_reg_train)
    reg_test_error = mean_squared_error(y_test, y_pred_reg_test)
    
    print(f"\n5. Regularized model with alpha={alpha}:")
    print(f"   - Training error: {reg_train_error:.4f}")
    print(f"   - Test error: {reg_test_error:.4f}")
    print(f"   - Coefficient: {reg_model.coef_[0]:.4f}")
    print(f"   - Intercept: {reg_model.intercept_:.4f}")

plt.title('Effect of Regularization on Linear Model', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "regularization_effect.png"), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}") 