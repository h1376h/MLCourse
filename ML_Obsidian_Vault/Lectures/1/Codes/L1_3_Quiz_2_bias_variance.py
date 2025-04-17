import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_3_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Define the true function and data generation
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

def generate_data(n_samples=30):
    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.2 # Add noise
    return X, y

def plot_bias_illustration(degree, all_predictions, X_test, y_test_true, filename):
    """Plot the average prediction vs true function to illustrate bias."""
    avg_prediction = np.mean(all_predictions, axis=0)
    bias_squared = np.mean((avg_prediction - y_test_true)**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_test_true, label='True Function', color='black', linestyle='--', linewidth=2)
    plt.plot(X_test, avg_prediction, label=f'Average Model Prediction (Degree {degree})', color='blue', linewidth=2)
    plt.fill_between(X_test, avg_prediction, y_test_true, color='red', alpha=0.2, label=f'Bias² ≈ {bias_squared:.3f}')
    
    plt.title(f'Bias Illustration (Polynomial Degree {degree})', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim((-1.5, 1.5))
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bias illustration plot for degree {degree} saved to: {file_path}")

def plot_variance_illustration(degree, all_predictions, X_test, filename, n_fits_to_show=10):
    """Plot multiple individual fits to illustrate variance."""
    avg_prediction = np.mean(all_predictions, axis=0)
    variance = np.mean(np.var(all_predictions, axis=0))
    
    plt.figure(figsize=(10, 6))
    # Plot some individual fits
    indices_to_show = np.random.choice(all_predictions.shape[0], n_fits_to_show, replace=False)
    for i, idx in enumerate(indices_to_show):
        plt.plot(X_test, all_predictions[idx, :], color='lightgray', linewidth=1, 
                 label='Individual Fits' if i == 0 else None)
        
    # Plot the average prediction
    plt.plot(X_test, avg_prediction, label='Average Model Prediction', color='blue', linewidth=2)
    
    plt.title(f'Variance Illustration (Polynomial Degree {degree}, Var ≈ {variance:.3f})', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim((-1.5, 1.5))
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Variance illustration plot for degree {degree} saved to: {file_path}")

# --- Step 1: Define Bias and Variance ---
print_step_header(1, "Definitions of Bias and Variance")

print("Bias:")
print("- Bias measures the difference between the average prediction of our model and the correct value we are trying to predict.")
print("- High bias means the model makes strong assumptions about the data (e.g., assumes linearity) and fails to capture the true underlying patterns.")
print("- Models with high bias are generally too simple and tend to underfit.")
print()
print("Variance:")
print("- Variance measures the variability of model prediction for a given data point if we were to retrain the model multiple times on different subsets of the training data.")
print("- High variance means the model is highly sensitive to the specific training data, including noise.")
print("- Models with high variance are generally too complex and tend to overfit.")
print()
print("Bias-Variance Tradeoff:")
print("- Goal: Find a model that minimizes total error, which decomposes into Bias^2 + Variance + Irreducible Error.")
print("- Simple models: High Bias, Low Variance.")
print("- Complex models: Low Bias, High Variance.")
print("- There\'s a tradeoff: Decreasing bias often increases variance, and vice versa.")

# --- Step 2: Demonstrate with Varying Model Complexity ---
print_step_header(2, "Demonstrating Tradeoff with Polynomial Regression")

# Generate sample data for visualization
X_vis, y_vis = generate_data(n_samples=30)

# Plot the true function and sample data
plt.figure(figsize=(10, 6))
X_true = np.linspace(0, 1, 100)
plt.plot(X_true, true_fun(X_true), label="True Function (cos(1.5*pi*X))", color='black', linestyle='--')
plt.scatter(X_vis, y_vis, edgecolor='k', facecolor='none', s=50, label='Sample Data Points')

# Fit and plot models with different polynomial degrees
degrees = [1, 4, 15]
colors = ['teal', 'gold', 'red']
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_vis.reshape(-1, 1), y_vis)
    y_plot = model.predict(X_true.reshape(-1, 1))
    plt.plot(X_true, y_plot, color=colors[i], linewidth=2, label=f"Degree {degree}")

plt.title('Models with Different Complexity', fontsize=14)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(loc='best')
plt.ylim((-1.5, 1.5))
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "model_complexity_fits.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot showing fits for degrees {degrees} saved to: {file_path}")
print("- Degree 1 (Linear): Simple model, likely high bias (underfits).")
print("- Degree 4: Moderate complexity, potentially good balance.")
print("- Degree 15: Complex model, likely high variance (overfits).")

# --- Step 3: Estimate Bias and Variance vs. Complexity ---
print_step_header(3, "Estimating Bias^2, Variance, and Error")

# Parameters for simulation
n_simulations = 100
n_samples_train = 30
n_samples_test = 100
max_degree = 12
noise_std = 0.2

# Generate a fixed test set
X_test = np.linspace(0, 1, n_samples_test)
y_test_true = true_fun(X_test)

degrees = np.arange(1, max_degree + 1)
bias_squared_list = []
variance_list = []
mse_list = []

# Store predictions for specific degrees to visualize bias/variance
predictions_deg1 = None
predictions_deg10 = None

print(f"Running {n_simulations} simulations for degrees 1 to {max_degree}...")

for degree in degrees:
    all_predictions = np.zeros((n_simulations, n_samples_test))
    
    for i in range(n_simulations):
        # Generate a new training set for each simulation
        X_train, y_train = generate_data(n_samples=n_samples_train)
        
        # Train the model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train.reshape(-1, 1), y_train)
        
        # Make predictions on the fixed test set
        y_pred = model.predict(X_test.reshape(-1, 1))
        all_predictions[i, :] = y_pred
        
    # Calculate average prediction across simulations
    avg_prediction = np.mean(all_predictions, axis=0)
    
    # Calculate Bias^2: (Average Prediction - True Value)^2, averaged over test points
    bias_squared = np.mean((avg_prediction - y_test_true)**2)
    bias_squared_list.append(bias_squared)
    
    # Calculate Variance: Average variance of predictions for each test point
    variance = np.mean(np.var(all_predictions, axis=0))
    variance_list.append(variance)
    
    # Calculate average MSE across simulations
    # Note: Avg MSE = Bias^2 + Variance + Irreducible Error (noise variance)
    mse = np.mean((all_predictions - y_test_true[np.newaxis, :])**2)
    mse_list.append(mse)
    
    # Store predictions for plotting bias/variance examples
    if degree == 1:
        predictions_deg1 = all_predictions.copy()
    elif degree == 10: # Choose a higher degree for variance example
        predictions_deg10 = all_predictions.copy()

print("Simulations complete.")

# Plot Bias-Variance Tradeoff
plt.figure(figsize=(10, 6))
plt.plot(degrees, bias_squared_list, label='Bias²', color='blue', marker='o')
plt.plot(degrees, variance_list, label='Variance', color='red', marker='s')
plt.plot(degrees, mse_list, label='Total Error (MSE)', color='black', marker='^')
plt.axhline(y=noise_std**2, color='gray', linestyle='--', label=f'Irreducible Error ({noise_std**2:.2f})')

# Add the sum of bias^2 + variance + irreducible error for verification
plt.plot(degrees, np.array(bias_squared_list) + np.array(variance_list) + noise_std**2, label='Bias² + Var + NoiseVar', color='purple', linestyle=':')

plt.title('Bias-Variance Tradeoff vs. Model Complexity (Polynomial Degree)', fontsize=14)
plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.xticks(degrees)
plt.legend(loc='upper center')
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

file_path = os.path.join(save_dir, "bias_variance_tradeoff_plot.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Bias-Variance tradeoff plot saved to: {file_path}")
print("- Low degrees (simple models): High Bias, Low Variance.")
print("- High degrees (complex models): Low Bias, High Variance.")
print("- Optimal complexity minimizes total error (MSE).")

# Add new plots for bias and variance illustration
if predictions_deg1 is not None:
    plot_bias_illustration(1, predictions_deg1, X_test, y_test_true, "bias_illustration_deg1.png")
    plot_variance_illustration(1, predictions_deg1, X_test, "variance_illustration_deg1.png")
    
if predictions_deg10 is not None:
    plot_bias_illustration(10, predictions_deg10, X_test, y_test_true, "bias_illustration_deg10.png")
    plot_variance_illustration(10, predictions_deg10, X_test, "variance_illustration_deg10.png")

# --- Step 4: Classify Models ---
print_step_header(4, "Classifying Models by Bias/Variance")

print("a. Linear Regression:")
print("   - Typically HIGH BIAS (especially if the true relationship is non-linear) and LOW VARIANCE.")
print("   - It makes strong assumptions (linearity).")

print("b. Decision Tree (no max depth):")
print("   - Typically LOW BIAS (can capture complex patterns) and HIGH VARIANCE (very sensitive to training data, prone to overfitting).")

print("c. k-Nearest Neighbors (k=1):")
print("   - Typically LOW BIAS (makes predictions based on the single closest point) and VERY HIGH VARIANCE (extremely sensitive to noise and individual data points).")

print("d. Support Vector Machine (Linear Kernel):")
print("   - Similar to Linear Regression: Typically HIGH BIAS (if data isn\'t linearly separable) and LOW VARIANCE.")

print("\nScript finished. Plots saved in:", save_dir) 