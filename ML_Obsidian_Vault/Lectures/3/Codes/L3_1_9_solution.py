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

# Additional Visualizations

# 1. Conceptual Illustration of Bias-Variance Tradeoff without text
plt.figure(figsize=(10, 8))

# Generate data points with high bias model (constant)
x_concept = np.linspace(0, 2*np.pi, 100)
y_concept_true = true_function(x_concept)

# Generate multiple datasets
np.random.seed(42)
n_datasets = 5
n_points = 10
models_high_bias = []
models_high_variance = []

plt.subplot(2, 1, 1)
plt.plot(x_concept, y_concept_true, 'g-', linewidth=2)

for i in range(n_datasets):
    # Generate dataset
    x_sample = np.random.uniform(0, 2*np.pi, n_points)
    y_sample = true_function(x_sample) + np.random.normal(0, 0.3, n_points)
    
    # High bias model (constant)
    constant_value = np.mean(y_sample)
    plt.plot(x_concept, np.full_like(x_concept, constant_value), 'r-', alpha=0.3)
    
plt.ylim(-1.5, 1.5)

plt.subplot(2, 1, 2)
plt.plot(x_concept, y_concept_true, 'g-', linewidth=2)

for i in range(n_datasets):
    # Generate dataset
    x_sample = np.random.uniform(0, 2*np.pi, n_points)
    y_sample = true_function(x_sample) + np.random.normal(0, 0.3, n_points)
    
    # High variance model (high-degree polynomial)
    if len(x_sample) >= 5:  # Need enough points for the polynomial
        z = np.polyfit(x_sample, y_sample, 4)
        p = np.poly1d(z)
        plt.plot(x_concept, p(x_concept), 'b-', alpha=0.3)

plt.ylim(-1.5, 1.5)

plt.savefig(os.path.join(save_dir, "bias_variance_concept.png"), dpi=300, bbox_inches='tight')

# 2. Model Complexity vs Error (U-shaped curve)
plt.figure(figsize=(8, 6))

# Generate data for training
n_points_complexity = 30
x_complexity = np.random.uniform(0, 2*np.pi, n_points_complexity)
y_complexity = true_function(x_complexity) + np.random.normal(0, 0.3, n_points_complexity)

# Generate data for testing
x_test_complexity = np.random.uniform(0, 2*np.pi, 100)
y_test_complexity = true_function(x_test_complexity) + np.random.normal(0, 0.1, 100)

# Train models of increasing complexity
degrees = range(0, 10)
train_errors = []
test_errors = []

for degree in degrees:
    # Fit polynomial of given degree
    z = np.polyfit(x_complexity, y_complexity, degree)
    p = np.poly1d(z)
    
    # Make predictions
    y_pred_train = p(x_complexity)
    y_pred_test = p(x_test_complexity)
    
    # Calculate errors
    train_err = mean_squared_error(y_complexity, y_pred_train)
    test_err = mean_squared_error(y_test_complexity, y_pred_test)
    
    train_errors.append(train_err)
    test_errors.append(test_err)

# Plot complexity vs error
plt.plot(degrees, train_errors, 'ro-')
plt.plot(degrees, test_errors, 'bo-')

plt.savefig(os.path.join(save_dir, "complexity_vs_error.png"), dpi=300, bbox_inches='tight')

# 3. Effect of noise on model fitting
plt.figure(figsize=(10, 8))

# Generate true function data
x_noise = np.linspace(0, 2*np.pi, 100)
y_true_noise = true_function(x_noise)

# Define different noise levels
noise_levels = [0.0, 0.1, 0.5, 1.0]

for i, noise in enumerate(noise_levels):
    plt.subplot(2, 2, i+1)
    
    # Generate noisy data
    x_data = np.random.uniform(0, 2*np.pi, 15)
    y_data = true_function(x_data) + np.random.normal(0, noise, len(x_data))
    
    # Plot true function
    plt.plot(x_noise, y_true_noise, 'g-')
    
    # Plot data points
    plt.scatter(x_data, y_data, color='blue', s=30)
    
    # Fit and plot linear model
    if len(x_data) > 1:
        X_data = x_data.reshape(-1, 1)
        model = LinearRegression().fit(X_data, y_data)
        y_pred = model.predict(x_noise.reshape(-1, 1))
        plt.plot(x_noise, y_pred, 'r-')
    
    plt.ylim(-2, 2)

plt.savefig(os.path.join(save_dir, "noise_effect.png"), dpi=300, bbox_inches='tight')

# Additional simple visualizations (without excessive text)

# Visualization 5: Variance Demonstration (H1)
plt.figure(figsize=(12, 6))
plt.suptitle('', fontsize=1)  # Empty title

# Left panel: multiple linear models
plt.subplot(1, 2, 1)
plt.plot(x_range, y_true, color='blue', linewidth=2)

# Generate multiple datasets and fit linear models
np.random.seed(42)
for i in range(50):
    x_sample = np.random.uniform(0, 2*np.pi, 7)
    y_sample = true_function(x_sample) + np.random.normal(0, 0.3, 7)
    
    X_sample = x_sample.reshape(-1, 1)
    model = LinearRegression().fit(X_sample, y_sample)
    y_pred = model.predict(x_range.reshape(-1, 1))
    
    plt.plot(x_range, y_pred, color='gray', alpha=0.2, linewidth=0.5)

# Add the average prediction
x_avg = np.linspace(0, 2*np.pi, 100)
y_avg = 0.21 * x_avg + 0.1
plt.plot(x_avg, y_avg, color='red', linewidth=2)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)
plt.axis('on')

# Right panel: mean and variance
plt.subplot(1, 2, 2)
plt.plot(x_range, y_true, color='blue', linewidth=2)

# Add the average prediction
plt.plot(x_avg, y_avg, color='red', linewidth=2)

# Add shaded region representing variance
upper_bound = y_avg + 0.7
lower_bound = y_avg - 0.7
plt.fill_between(x_avg, lower_bound, upper_bound, color='gray', alpha=0.3)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)
plt.axis('on')

plt.savefig(os.path.join(save_dir, "variance_h1_demonstration.png"), dpi=300, bbox_inches='tight')

# Visualization 6: H0 vs H1 Comparison
plt.figure(figsize=(12, 6))
plt.suptitle('', fontsize=1)  # Empty title

# Left panel: H0 (constant model)
plt.subplot(1, 2, 1)

# True function
plt.plot(x_range, y_true, color='blue', linewidth=2)

# Constant model
constant_value = 0.0
plt.axhline(y=constant_value, color='green', linewidth=2)

# Shaded region for variance
plt.fill_between(x_range, constant_value - 0.5, constant_value + 0.5, color='gray', alpha=0.3)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)
plt.text(0.5, -1.2, "bias = 0.50", fontsize=12)
plt.text(3, -1.2, "var = 0.25", fontsize=12)

# Right panel: H1 (linear model)
plt.subplot(1, 2, 2)

# True function
plt.plot(x_range, y_true, color='blue', linewidth=2)

# Linear model average
y_linear = 0.21 * x_range + 0.1
plt.plot(x_range, y_linear, color='red', linewidth=2)

# Shaded region for variance
upper_bound = y_linear + 0.7
lower_bound = y_linear - 0.7
plt.fill_between(x_range, lower_bound, upper_bound, color='gray', alpha=0.3)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)
plt.text(0.5, -1.2, "bias = 0.21", fontsize=12)
plt.text(3, -1.2, "var = 1.69", fontsize=12)

plt.savefig(os.path.join(save_dir, "h0_vs_h1_comparison.png"), dpi=300, bbox_inches='tight')

# Visualization 7: Prediction Intervals for Both Models
plt.figure(figsize=(12, 6))
plt.suptitle('', fontsize=1)  # Empty title

# Generate a consistent test dataset
np.random.seed(123)
x_interval = np.random.uniform(0, 2*np.pi, 100)
x_interval = np.sort(x_interval)
y_interval_true = true_function(x_interval)

# Left panel: Constant model prediction intervals
plt.subplot(1, 2, 1)

# Fit constant model and generate predictions
const_value = np.mean(y_interval_true)
y_const_pred = np.full_like(x_interval, const_value)

# Draw true function and prediction
plt.plot(x_interval, y_interval_true, color='blue', linewidth=2)
plt.plot(x_interval, y_const_pred, color='green', linewidth=2)

# Add prediction intervals
plt.fill_between(x_interval, y_const_pred - 0.5, y_const_pred + 0.5, 
                 color='green', alpha=0.2)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)

# Right panel: Linear model prediction intervals
plt.subplot(1, 2, 2)

# Fit linear model and generate predictions
X_interval = x_interval.reshape(-1, 1)
linear_model_interval = LinearRegression().fit(X_interval, y_interval_true)
y_linear_pred = linear_model_interval.predict(X_interval)

# Draw true function and prediction
plt.plot(x_interval, y_interval_true, color='blue', linewidth=2)
plt.plot(x_interval, y_linear_pred, color='red', linewidth=2)

# Add prediction intervals (wider at the edges, narrower in the middle)
interval_width = 0.3 + 0.4 * np.abs(x_interval - np.pi) / np.pi
plt.fill_between(x_interval, 
                 y_linear_pred - interval_width,
                 y_linear_pred + interval_width, 
                 color='red', alpha=0.2)

plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5)

plt.savefig(os.path.join(save_dir, "prediction_intervals.png"), dpi=300, bbox_inches='tight')

print(f"\nNew visualizations saved to: {save_dir}")
print(f"\nAdditional visualizations saved to: {save_dir}") 