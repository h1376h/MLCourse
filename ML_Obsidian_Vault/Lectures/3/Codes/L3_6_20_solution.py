import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Print explanations and formulas
print("\nError Decomposition in Linear Regression")
print("=======================================")
print("\nDefinitions:")
print("1. Structural Error: The error due to the model's inability to capture the true relationship")
print("   (Even with infinite data and optimal parameters)")
print("   Mathematical expression: E_x,y[(y - w*^T x)^2]")
print("\n2. Approximation Error: The error due to estimating parameters from finite training data")
print("   Mathematical expression: E_x[(w*^T x - ŵ^T x)^2]")
print("\n3. Total Expected Error: The sum of structural and approximation errors")
print("   E_x,y[(y - ŵ^T x)^2] = E_x,y[(y - w*^T x)^2] + E_x[(w*^T x - ŵ^T x)^2]")

# Generate synthetic data
np.random.seed(42)

def true_function(x):
    """The true underlying function (cubic)"""
    return 0.5 * x**3 - 0.8 * x**2 + 0.2 * x + 2 + np.sin(x*3)

def sample_data(n_samples, noise_level=0.5, x_range=(-3, 3)):
    """Generate n_samples data points with given noise level"""
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y_true = true_function(x)
    y = y_true + np.random.normal(0, noise_level, n_samples)
    return x, y, y_true

# Functions to calculate errors
def calculate_structural_error(model, degree, x_test, y_test_true):
    """Calculate structural error - error of the best possible model of this complexity"""
    # Create a very large dataset to approximate "infinite data"
    x_large = np.linspace(min(x_test), max(x_test), 1000).reshape(-1, 1)
    y_large = true_function(x_large.ravel())
    
    # Fit the model on this large dataset to approximate "optimal parameters"
    optimal_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    optimal_model.fit(x_large, y_large)
    
    # Predict on test data using this "optimal" model
    y_pred_optimal = optimal_model.predict(x_test.reshape(-1, 1))
    
    # Calculate structural error (error with optimal parameters)
    structural_error = mean_squared_error(y_test_true, y_pred_optimal)
    
    return structural_error, optimal_model

def calculate_approximation_error(optimal_model, trained_model, x_test):
    """Calculate approximation error - error due to parameter estimation from finite data"""
    # Predictions using optimal and trained models
    y_pred_optimal = optimal_model.predict(x_test.reshape(-1, 1))
    y_pred_trained = trained_model.predict(x_test.reshape(-1, 1))
    
    # Calculate approximation error (difference between optimal and trained predictions)
    approximation_error = mean_squared_error(y_pred_optimal, y_pred_trained)
    
    return approximation_error

# Experiment 1: Effect of training sample size
print("\nExperiment 1: Effect of Training Sample Size")
print("-------------------------------------------")

# Generate a fixed test set
x_test_large, y_test_large, y_test_true_large = sample_data(1000, noise_level=0.0)

# Try different training sample sizes
sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]
structural_errors = []
approximation_errors = []
total_errors = []

degree = 3  # Fixed model complexity (cubic polynomial)

for n_samples in sample_sizes:
    print(f"\nTraining with {n_samples} samples...")
    
    # Generate training data
    x_train, y_train, _ = sample_data(n_samples)
    
    # Train model on the training data
    trained_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    trained_model.fit(x_train.reshape(-1, 1), y_train)
    
    # Calculate structural error (with optimal parameters)
    structural_error, optimal_model = calculate_structural_error(trained_model, degree, 
                                                              x_test_large.reshape(-1, 1), 
                                                              y_test_true_large)
    
    # Calculate approximation error (due to finite training data)
    approximation_error = calculate_approximation_error(optimal_model, trained_model, 
                                                     x_test_large.reshape(-1, 1))
    
    # Calculate total error
    y_pred_trained = trained_model.predict(x_test_large.reshape(-1, 1))
    total_error = mean_squared_error(y_test_true_large, y_pred_trained)
    
    # Store errors
    structural_errors.append(structural_error)
    approximation_errors.append(approximation_error)
    total_errors.append(total_error)
    
    print(f"  Structural Error: {structural_error:.4f}")
    print(f"  Approximation Error: {approximation_error:.4f}")
    print(f"  Total Error: {total_error:.4f}")
    print(f"  Sum of Components: {structural_error + approximation_error:.4f}")

# Plot errors vs. sample size
plt.figure(figsize=(10, 6))
plt.semilogx(sample_sizes, structural_errors, 'o-', label='Structural Error', linewidth=2)
plt.semilogx(sample_sizes, approximation_errors, 's-', label='Approximation Error', linewidth=2)
plt.semilogx(sample_sizes, total_errors, '^-', label='Total Error', linewidth=2)
plt.grid(True)
plt.xlabel('Training Sample Size (log scale)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Error Decomposition vs. Training Sample Size', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, "error_vs_sample_size.png"), dpi=300, bbox_inches='tight')
plt.close()

# Experiment 2: Effect of model complexity
print("\nExperiment 2: Effect of Model Complexity")
print("---------------------------------------")

# Fixed training sample size
n_samples = 100
x_train, y_train, _ = sample_data(n_samples)
x_test, y_test, y_test_true = sample_data(1000, noise_level=0.0)

# Try different polynomial degrees
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
structural_errors_complexity = []
approximation_errors_complexity = []
total_errors_complexity = []

for degree in degrees:
    print(f"\nTraining with polynomial degree {degree}...")
    
    # Train model on the training data
    trained_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    trained_model.fit(x_train.reshape(-1, 1), y_train)
    
    # Calculate structural error (with optimal parameters)
    structural_error, optimal_model = calculate_structural_error(trained_model, degree, 
                                                              x_test.reshape(-1, 1), 
                                                              y_test_true)
    
    # Calculate approximation error (due to finite training data)
    approximation_error = calculate_approximation_error(optimal_model, trained_model, 
                                                     x_test.reshape(-1, 1))
    
    # Calculate total error
    y_pred_trained = trained_model.predict(x_test.reshape(-1, 1))
    total_error = mean_squared_error(y_test_true, y_pred_trained)
    
    # Store errors
    structural_errors_complexity.append(structural_error)
    approximation_errors_complexity.append(approximation_error)
    total_errors_complexity.append(total_error)
    
    print(f"  Structural Error: {structural_error:.4f}")
    print(f"  Approximation Error: {approximation_error:.4f}")
    print(f"  Total Error: {total_error:.4f}")
    print(f"  Sum of Components: {structural_error + approximation_error:.4f}")

# Plot errors vs. model complexity
plt.figure(figsize=(10, 6))
plt.plot(degrees, structural_errors_complexity, 'o-', label='Structural Error', linewidth=2)
plt.plot(degrees, approximation_errors_complexity, 's-', label='Approximation Error', linewidth=2)
plt.plot(degrees, total_errors_complexity, '^-', label='Total Error', linewidth=2)
plt.grid(True)
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Error Decomposition vs. Model Complexity', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, "error_vs_complexity.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization: Illustrate error decomposition
# Choose a specific case for visualization
degree = 3
n_samples = 50
print(f"\nVisualizing error decomposition (degree={degree}, n_samples={n_samples})...")

# Generate data
x_train, y_train, _ = sample_data(n_samples)
x_viz = np.linspace(-3, 3, 500).reshape(-1, 1)
y_viz_true = true_function(x_viz.ravel())

# Train model on the training data
trained_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
trained_model.fit(x_train.reshape(-1, 1), y_train)
y_viz_trained = trained_model.predict(x_viz)

# Optimal model (approximating infinite data)
optimal_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
x_large = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_large = true_function(x_large.ravel())
optimal_model.fit(x_large, y_large)
y_viz_optimal = optimal_model.predict(x_viz)

# Create visualization
plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

# Plot 1: Original data, true function, and models
ax1 = plt.subplot(gs[0, :])
ax1.scatter(x_train, y_train, color='blue', alpha=0.6, label='Training Data')
ax1.plot(x_viz, y_viz_true, 'k-', linewidth=2, label='True Function')
ax1.plot(x_viz, y_viz_optimal, 'g-', linewidth=2, label='Optimal Model (w*)')
ax1.plot(x_viz, y_viz_trained, 'r-', linewidth=2, label='Trained Model (ŵ)')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title(f'Error Decomposition in Linear Regression (degree={degree}, n_samples={n_samples})', fontsize=14)
ax1.legend()
ax1.grid(True)

# Plot 2: Structural Error
ax2 = plt.subplot(gs[1, 0])
ax2.plot(x_viz, y_viz_true, 'k-', linewidth=2, label='True Function')
ax2.plot(x_viz, y_viz_optimal, 'g-', linewidth=2, label='Optimal Model (w*)')
ax2.fill_between(x_viz.ravel(), y_viz_true, y_viz_optimal, color='blue', alpha=0.3, label='Structural Error')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Structural Error: E[(y - w*ᵀx)²]', fontsize=12)
ax2.legend()
ax2.grid(True)

# Plot 3: Approximation Error
ax3 = plt.subplot(gs[1, 1])
ax3.plot(x_viz, y_viz_optimal, 'g-', linewidth=2, label='Optimal Model (w*)')
ax3.plot(x_viz, y_viz_trained, 'r-', linewidth=2, label='Trained Model (ŵ)')
ax3.fill_between(x_viz.ravel(), y_viz_optimal, y_viz_trained, color='red', alpha=0.3, label='Approximation Error')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Approximation Error: E[(w*ᵀx - ŵᵀx)²]', fontsize=12)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "error_decomposition_visualization.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization: 3D surface showing the error components
# Create meshgrid for training sample size and model complexity
x_mesh, y_mesh = np.meshgrid(np.array(degrees), np.array(sample_sizes))
z_structural = np.zeros(x_mesh.shape)
z_approximation = np.zeros(x_mesh.shape)
z_total = np.zeros(x_mesh.shape)

# Generate data for the surfaces
print("\nGenerating 3D visualization data...")
for i, n_samples in enumerate(sample_sizes):
    for j, degree in enumerate(degrees):
        # Generate training data
        x_train, y_train, _ = sample_data(n_samples)
        
        # Train model on the training data
        trained_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        trained_model.fit(x_train.reshape(-1, 1), y_train)
        
        # Calculate errors
        structural_error, optimal_model = calculate_structural_error(trained_model, degree, 
                                                                  x_test.reshape(-1, 1), 
                                                                  y_test_true)
        
        approximation_error = calculate_approximation_error(optimal_model, trained_model, 
                                                         x_test.reshape(-1, 1))
        
        y_pred_trained = trained_model.predict(x_test.reshape(-1, 1))
        total_error = mean_squared_error(y_test_true, y_pred_trained)
        
        # Store errors
        z_structural[i, j] = structural_error
        z_approximation[i, j] = approximation_error
        z_total[i, j] = total_error

# Plot 3D surfaces
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(np.log10(x_mesh), y_mesh, z_structural, cmap='viridis', alpha=0.8)
ax1.set_xlabel('log10(Training Sample Size)', fontsize=10)
ax1.set_ylabel('Polynomial Degree', fontsize=10)
ax1.set_zlabel('MSE', fontsize=10)
ax1.set_title('Structural Error', fontsize=12)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(np.log10(x_mesh), y_mesh, z_approximation, cmap='plasma', alpha=0.8)
ax2.set_xlabel('log10(Training Sample Size)', fontsize=10)
ax2.set_ylabel('Polynomial Degree', fontsize=10)
ax2.set_zlabel('MSE', fontsize=10)
ax2.set_title('Approximation Error', fontsize=12)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(np.log10(x_mesh), y_mesh, z_total, cmap='magma', alpha=0.8)
ax3.set_xlabel('log10(Training Sample Size)', fontsize=10)
ax3.set_ylabel('Polynomial Degree', fontsize=10)
ax3.set_zlabel('MSE', fontsize=10)
ax3.set_title('Total Error', fontsize=12)
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "error_decomposition_3d.png"), dpi=300, bbox_inches='tight')
plt.close()

# Proof illustration: Visual demonstration that total error = structural + approximation
print("\nProof Illustration: Total Error = Structural Error + Approximation Error")
print("---------------------------------------------------------------------")

n_samples_proof = 50
degree_proof = 3

# Generate large test dataset
x_test_proof, y_test_proof, y_test_true_proof = sample_data(200, noise_level=0.0)

# Generate training data
x_train_proof, y_train_proof, _ = sample_data(n_samples_proof)

# Train model on the training data
trained_model_proof = make_pipeline(PolynomialFeatures(degree_proof), LinearRegression())
trained_model_proof.fit(x_train_proof.reshape(-1, 1), y_train_proof)

# Calculate structural error (with optimal parameters)
structural_error_proof, optimal_model_proof = calculate_structural_error(
    trained_model_proof, degree_proof, x_test_proof.reshape(-1, 1), y_test_true_proof)

# Calculate approximation error (due to finite training data)
approximation_error_proof = calculate_approximation_error(
    optimal_model_proof, trained_model_proof, x_test_proof.reshape(-1, 1))

# Calculate total error
y_pred_trained_proof = trained_model_proof.predict(x_test_proof.reshape(-1, 1))
total_error_proof = mean_squared_error(y_test_true_proof, y_pred_trained_proof)

# Select a sample point for detailed illustration
idx = np.random.randint(0, len(x_test_proof))
x_point = x_test_proof[idx]
y_true_point = y_test_true_proof[idx]
y_optimal_point = optimal_model_proof.predict(np.array([[x_point]]))[0]
y_trained_point = trained_model_proof.predict(np.array([[x_point]]))[0]

# Calculate individual errors at the sample point
structural_error_point = (y_true_point - y_optimal_point)**2
approximation_error_point = (y_optimal_point - y_trained_point)**2
total_error_point = (y_true_point - y_trained_point)**2

print(f"\nAt sample point x = {x_point:.4f}:")
print(f"  True y = {y_true_point:.4f}")
print(f"  Optimal model prediction y* = {y_optimal_point:.4f}")
print(f"  Trained model prediction ŷ = {y_trained_point:.4f}")
print(f"  Structural Error: (y - y*)^2 = {structural_error_point:.4f}")
print(f"  Approximation Error: (y* - ŷ)^2 = {approximation_error_point:.4f}")
print(f"  Total Error: (y - ŷ)^2 = {total_error_point:.4f}")
print(f"  Sum of Error Components: {structural_error_point + approximation_error_point:.4f}")

print(f"\nOverall Test Dataset:")
print(f"  Average Structural Error: {structural_error_proof:.4f}")
print(f"  Average Approximation Error: {approximation_error_proof:.4f}")
print(f"  Average Total Error: {total_error_proof:.4f}")
print(f"  Sum of Average Error Components: {structural_error_proof + approximation_error_proof:.4f}")
print(f"  Difference: {total_error_proof - (structural_error_proof + approximation_error_proof):.8f}")

# Visualize the error decomposition proof
plt.figure(figsize=(10, 6))
x_viz_proof = np.linspace(-3, 3, 500)
y_viz_true_proof = true_function(x_viz_proof)
y_viz_optimal_proof = optimal_model_proof.predict(x_viz_proof.reshape(-1, 1))
y_viz_trained_proof = trained_model_proof.predict(x_viz_proof.reshape(-1, 1))

plt.plot(x_viz_proof, y_viz_true_proof, 'k-', linewidth=2, label='True Function')
plt.plot(x_viz_proof, y_viz_optimal_proof, 'g-', linewidth=2, label='Optimal Model (w*)')
plt.plot(x_viz_proof, y_viz_trained_proof, 'r-', linewidth=2, label='Trained Model (ŵ)')
plt.scatter(x_train_proof, y_train_proof, color='blue', alpha=0.6, label='Training Data')

# Mark the sample point
plt.scatter([x_point], [y_true_point], color='purple', s=100, marker='o', label='Sample Point (y)')
plt.scatter([x_point], [y_optimal_point], color='green', s=100, marker='s', label='Optimal Prediction (y*)')
plt.scatter([x_point], [y_trained_point], color='red', s=100, marker='^', label='Trained Prediction (ŷ)')

# Draw lines to show errors
plt.plot([x_point, x_point], [y_true_point, y_optimal_point], 'b--', linewidth=2, label='Structural Error')
plt.plot([x_point, x_point], [y_optimal_point, y_trained_point], 'r--', linewidth=2, label='Approximation Error')
plt.plot([x_point, x_point], [y_true_point, y_trained_point], 'k--', linewidth=2, label='Total Error')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Proof: Total Error = Structural Error + Approximation Error', fontsize=14)
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(save_dir, "error_decomposition_proof.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}") 