import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots and use a font that supports Unicode subscripts
plt.style.use('seaborn-v0_8-whitegrid')
# Set a font that supports Unicode subscripts (DejaVu Sans is usually available)
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Print explanations and formulas
print("\nHypothesis Spaces in Linear Modeling")
print("===================================")
print("\nHypothesis spaces definition:")
print("A hypothesis space is the set of all possible functions that an algorithm can select as the solution.")
print("\nThree common hypothesis spaces:")
print("1. H₀: Constant functions f(x) = b")
print("2. H₁: Linear functions f(x) = ax + b")
print("3. H₂: Quadratic functions f(x) = ax² + bx + c")

print("\nApproximation-Generalization Trade-off:")
print("- Larger hypothesis spaces can better approximate complex target functions")
print("- Smaller hypothesis spaces often generalize better with limited data")
print("- Optimal choice depends on the amount of training data and the complexity of the true function")

# Generate data for visualizations
np.random.seed(42)
x_range = np.linspace(-1, 1, 1000)

# Define hypothesis spaces
def h0(x, params):  # Constant function
    return np.ones_like(x) * params[0]

def h1(x, params):  # Linear function
    return params[0] * x + params[1]

def h2(x, params):  # Quadratic function
    return params[0] * x**2 + params[1] * x + params[2]

# Target function: sine curve
def target_function(x):
    return np.sin(3 * np.pi * x)

# Visualization 1: Different hypothesis spaces
plt.figure(figsize=(12, 6))

# Plot examples from each hypothesis space
plt.plot(x_range, h0(x_range, [0.5]), 'b-', label='H₀: f(x) = 0.5', alpha=0.7)
plt.plot(x_range, h1(x_range, [1, 0]), 'g-', label='H₁: f(x) = x', alpha=0.7)
plt.plot(x_range, h1(x_range, [-0.5, 0.7]), 'g--', label='H₁: f(x) = -0.5x + 0.7', alpha=0.7)
plt.plot(x_range, h2(x_range, [1, 0, 0]), 'r-', label='H₂: f(x) = x²', alpha=0.7)
plt.plot(x_range, h2(x_range, [-1, 2, 0.5]), 'r--', label='H₂: f(x) = -x² + 2x + 0.5', alpha=0.7)

plt.title('Examples of Different Hypothesis Spaces', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "hypothesis_spaces_examples.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Model complexity demonstration
plt.figure(figsize=(16, 8))

# Target function
y_true = target_function(x_range)
plt.plot(x_range, y_true, 'k-', label='Target function: sin(3πx)', linewidth=2)

# Generate limited training data (2 points)
x_train = np.array([-0.7, 0.5])
y_train = target_function(x_train)
plt.scatter(x_train, y_train, color='black', s=100, zorder=5, 
           label='Training points (2 samples)')

# Fit models from different hypothesis spaces
# Constant model (H₀)
constant_model = np.mean(y_train)
y_h0 = np.ones_like(x_range) * constant_model

# Linear model (H₁)
linear_model = LinearRegression()
linear_model.fit(x_train.reshape(-1, 1), y_train)
y_h1 = linear_model.predict(x_range.reshape(-1, 1))

# Quadratic model (H₂)
quad_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
quad_model.fit(x_train.reshape(-1, 1), y_train)
y_h2 = quad_model.predict(x_range.reshape(-1, 1))

# Higher order polynomial model (H₅)
poly5_model = make_pipeline(PolynomialFeatures(5), LinearRegression())
poly5_model.fit(x_train.reshape(-1, 1), y_train)
y_h5 = poly5_model.predict(x_range.reshape(-1, 1))

# Plot the fitted models
plt.plot(x_range, y_h0, 'b-', label=f'H₀: f(x) = {constant_model:.2f}', linewidth=2)
plt.plot(x_range, y_h1, 'g-', 
        label=f'H₁: f(x) = {linear_model.coef_[0]:.2f}x + {linear_model.intercept_:.2f}', 
        linewidth=2)
plt.plot(x_range, y_h2, 'r-', label='H₂: Quadratic function', linewidth=2)
plt.plot(x_range, y_h5, 'm-', label='H₅: 5th degree polynomial', linewidth=2)

plt.title('Hypothesis Spaces and Fitting with Limited Data (2 points)', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "limited_data_fitting.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Approximation-Generalization Trade-off
plt.figure(figsize=(14, 10))

# Target function and training points
plt.subplot(2, 2, 1)
plt.plot(x_range, y_true, 'k-', label='Target: sin(3πx)', linewidth=2)
plt.scatter(x_train, y_train, color='black', s=80, zorder=5, label='Training points')
plt.title('True Target Function', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Generate more training points for comparison
x_train_more = np.array([-0.9, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8])
y_train_more = target_function(x_train_more)

# H₁ with 2 points vs with more points
plt.subplot(2, 2, 2)
plt.plot(x_range, y_true, 'k-', label='Target: sin(3πx)', linewidth=2, alpha=0.7)
plt.scatter(x_train, y_train, color='black', s=80, zorder=5, label='2 training points')
plt.scatter(x_train_more, y_train_more, color='blue', s=80, zorder=5, label='7 training points')

# Linear model with 2 points
linear_model.fit(x_train.reshape(-1, 1), y_train)
y_h1_2pts = linear_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h1_2pts, 'g-', label=f'H₁ with 2 points', linewidth=2)

# Linear model with more points
linear_model.fit(x_train_more.reshape(-1, 1), y_train_more)
y_h1_more = linear_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h1_more, 'b-', label=f'H₁ with 7 points', linewidth=2)

plt.title('Linear Models (H₁) with Different Training Set Sizes', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# H₂ with 2 points vs with more points
plt.subplot(2, 2, 3)
plt.plot(x_range, y_true, 'k-', label='Target: sin(3πx)', linewidth=2, alpha=0.7)
plt.scatter(x_train, y_train, color='black', s=80, zorder=5, label='2 training points')
plt.scatter(x_train_more, y_train_more, color='blue', s=80, zorder=5, label='7 training points')

# Quadratic model with 2 points
quad_model.fit(x_train.reshape(-1, 1), y_train)
y_h2_2pts = quad_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h2_2pts, 'r-', label='H₂ with 2 points', linewidth=2)

# Quadratic model with more points
quad_model.fit(x_train_more.reshape(-1, 1), y_train_more)
y_h2_more = quad_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h2_more, 'm-', label='H₂ with 7 points', linewidth=2)

plt.title('Quadratic Models (H₂) with Different Training Set Sizes', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Higher order polynomial with 2 points vs with more points
plt.subplot(2, 2, 4)
plt.plot(x_range, y_true, 'k-', label='Target: sin(3πx)', linewidth=2, alpha=0.7)
plt.scatter(x_train, y_train, color='black', s=80, zorder=5, label='2 training points')
plt.scatter(x_train_more, y_train_more, color='blue', s=80, zorder=5, label='7 training points')

# 5th degree polynomial with 2 points
poly5_model.fit(x_train.reshape(-1, 1), y_train)
y_h5_2pts = poly5_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h5_2pts, 'c-', label='H₅ with 2 points', linewidth=2)

# 5th degree polynomial with more points
poly5_model.fit(x_train_more.reshape(-1, 1), y_train_more)
y_h5_more = poly5_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_h5_more, 'y-', label='H₅ with 7 points', linewidth=2)

plt.title('5th Degree Polynomial Models (H₅) with Different Training Set Sizes', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "approximation_generalization_tradeoff.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Error comparison for different hypothesis spaces
plt.figure(figsize=(12, 8))

# Calculate errors for different models with 2 points
# Test points for evaluation (more dense than training points)
x_test = np.linspace(-1, 1, 50)
y_test = target_function(x_test)

# Models with 2 training points
constant_model = np.mean(y_train)
y_h0_pred = np.ones_like(x_test) * constant_model

linear_model.fit(x_train.reshape(-1, 1), y_train)
y_h1_pred = linear_model.predict(x_test.reshape(-1, 1))

quad_model.fit(x_train.reshape(-1, 1), y_train)
y_h2_pred = quad_model.predict(x_test.reshape(-1, 1))

poly5_model.fit(x_train.reshape(-1, 1), y_train)
y_h5_pred = poly5_model.predict(x_test.reshape(-1, 1))

# Calculate MSE
mse_h0 = np.mean((y_test - y_h0_pred) ** 2)
mse_h1 = np.mean((y_test - y_h1_pred) ** 2)
mse_h2 = np.mean((y_test - y_h2_pred) ** 2)
mse_h5 = np.mean((y_test - y_h5_pred) ** 2)

# Calculate training errors
train_mse_h0 = np.mean((y_train - np.ones_like(y_train) * constant_model) ** 2)
train_mse_h1 = np.mean((y_train - linear_model.predict(x_train.reshape(-1, 1))) ** 2)
train_mse_h2 = np.mean((y_train - quad_model.predict(x_train.reshape(-1, 1))) ** 2)
train_mse_h5 = np.mean((y_train - poly5_model.predict(x_train.reshape(-1, 1))) ** 2)

# Bar plots for training and test errors
hypothesis_spaces = ['H₀ (Constant)', 'H₁ (Linear)', 'H₂ (Quadratic)', 'H₅ (Polynomial)']
train_errors = [train_mse_h0, train_mse_h1, train_mse_h2, train_mse_h5]
test_errors = [mse_h0, mse_h1, mse_h2, mse_h5]

x = np.arange(len(hypothesis_spaces))
width = 0.35

plt.bar(x - width/2, train_errors, width, label='Training Error (2 points)')
plt.bar(x + width/2, test_errors, width, label='Test Error')

plt.title('Errors for Different Hypothesis Spaces with 2 Training Points', fontsize=16)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.xticks(x, hypothesis_spaces, fontsize=12)
plt.legend(fontsize=12)

# Print numerical results
print("\nError Analysis with 2 Training Points:")
for i, space in enumerate(hypothesis_spaces):
    print(f"{space}: Training MSE = {train_errors[i]:.4f}, Test MSE = {test_errors[i]:.4f}")

# Add text annotations on the bars
for i, v in enumerate(train_errors):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
for i, v in enumerate(test_errors):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "error_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}") 