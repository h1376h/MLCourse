import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_points_large = 1000  # For approximating "infinite" data
min_x, max_x = -5, 5
noise_std = 0.5

# Define the true function (non-linear)
def true_function(x):
    return np.sin(x)

def generate_data(n_points, noise_std=noise_std):
    """Generate data from the true function with noise"""
    x = np.random.uniform(min_x, max_x, n_points)
    epsilon = np.random.normal(0, noise_std, n_points)
    y = true_function(x) + epsilon
    return x.reshape(-1, 1), y

def get_optimal_linear_params(n_points=n_points_large):
    """Get the optimal linear parameters using a very large dataset"""
    x_large, y_large = generate_data(n_points)
    model = LinearRegression()
    model.fit(x_large, y_large)
    return model.coef_[0], model.intercept_

def demonstration_part1():
    """Demonstrate structural error vs. approximation error"""
    print("\nPart 1: Structural Error vs. Approximation Error")
    
    # Get the optimal linear parameters (w* - approximation of infinite data)
    optimal_slope, optimal_intercept = get_optimal_linear_params()
    
    print(f"1. Optimal Linear Parameters (w*) with very large dataset:")
    print(f"   Slope: {optimal_slope:.6f}, Intercept: {optimal_intercept:.6f}")
    
    # Generate a grid of x values for plotting the true function
    x_grid = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
    y_true = true_function(x_grid.flatten())
    
    # Predict using the optimal linear model
    y_optimal_linear = optimal_slope * x_grid.flatten() + optimal_intercept
    
    # Calculate structural error (error between true function and optimal linear model)
    structural_error = mean_squared_error(y_true, y_optimal_linear)
    
    print(f"2. Structural Error:")
    print(f"   MSE between true function and optimal linear model: {structural_error:.6f}")
    print(f"   This error exists because a linear model cannot perfectly fit the non-linear function.")
    
    # Now demonstrate approximation error with a small dataset
    n_samples_small = 20
    x_small, y_small = generate_data(n_samples_small)
    
    # Fit a linear model to the small dataset
    model_small = LinearRegression()
    model_small.fit(x_small, y_small)
    slope_small = model_small.coef_[0]
    intercept_small = model_small.intercept_
    
    print(f"3. Linear Parameters (ŵ) with small dataset (n={n_samples_small}):")
    print(f"   Slope: {slope_small:.6f}, Intercept: {intercept_small:.6f}")
    
    # Predict using the model trained on small dataset
    y_small_linear = slope_small * x_grid.flatten() + intercept_small
    
    # Calculate total error (error between true function and model from small dataset)
    total_error = mean_squared_error(y_true, y_small_linear)
    
    # Approximation error is the difference between total error and structural error
    approximation_error = mean_squared_error(y_optimal_linear, y_small_linear)
    
    print(f"4. Total Error (with small dataset):")
    print(f"   MSE between true function and small-dataset model: {total_error:.6f}")
    print(f"5. Approximation Error:")
    print(f"   MSE between optimal linear model and small-dataset model: {approximation_error:.6f}")
    print(f"   This error exists because we estimated parameters from limited data.")
    
    # Verify the decomposition: total_error ≈ structural_error + approximation_error
    print(f"6. Error Decomposition:")
    print(f"   Total Error: {total_error:.6f}")
    print(f"   Structural Error + Approximation Error: {structural_error + approximation_error:.6f}")
    print(f"   Difference: {total_error - (structural_error + approximation_error):.6f}")
    
    # Create visualization
    print("Creating first visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot true function
    plt.plot(x_grid, y_true, 'g-', linewidth=2, label='True Function f(x) = sin(x)')
    
    # Plot optimal linear model
    plt.plot(x_grid, y_optimal_linear, 'b-', linewidth=2, 
             label=f'Optimal Linear Model: y = {optimal_slope:.4f}x + {optimal_intercept:.4f}')
    
    # Plot small dataset model
    plt.plot(x_grid, y_small_linear, 'r-', linewidth=2, 
             label=f'Small Dataset Model: y = {slope_small:.4f}x + {intercept_small:.4f}')
    
    # Plot the small dataset points
    plt.scatter(x_small, y_small, color='black', s=30, label=f'Small Dataset (n={n_samples_small})')
    
    # Shade the structural error region
    plt.fill_between(x_grid.flatten(), y_true, y_optimal_linear, color='blue', alpha=0.2, 
                    label='Structural Error Area')
    
    # Shade the approximation error region
    plt.fill_between(x_grid.flatten(), y_optimal_linear, y_small_linear, color='red', alpha=0.2, 
                     label='Approximation Error Area')
    
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Structural Error vs. Approximation Error', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, "structural_vs_approximation_error.png")
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {file_path}")
    
    return optimal_slope, optimal_intercept

def demonstration_part2(optimal_slope, optimal_intercept):
    """Demonstrate the effect of sample size on approximation error"""
    print("\nPart 2: Effect of Sample Size on Approximation Error")
    
    # Generate a grid of x values for plotting the true function
    x_grid = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
    y_true = true_function(x_grid.flatten())
    
    # Predict using the optimal linear model
    y_optimal_linear = optimal_slope * x_grid.flatten() + optimal_intercept
    
    # Calculate structural error
    structural_error = mean_squared_error(y_true, y_optimal_linear)
    
    # Sample sizes to demonstrate
    sample_sizes = [10, 30, 100, 300, 1000]
    
    # Store results
    slopes = []
    intercepts = []
    total_errors = []
    approximation_errors = []
    
    for n in sample_sizes:
        # Generate dataset
        x_train, y_train = generate_data(n)
        
        # Fit model
        model = LinearRegression()
        model.fit(x_train, y_train)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Store parameters
        slopes.append(slope)
        intercepts.append(intercept)
        
        # Predict
        y_pred = slope * x_grid.flatten() + intercept
        
        # Calculate errors
        total_error = mean_squared_error(y_true, y_pred)
        approx_error = mean_squared_error(y_optimal_linear, y_pred)
        
        # Store errors
        total_errors.append(total_error)
        approximation_errors.append(approx_error)
        
        print(f"Sample size n = {n}:")
        print(f"  Slope: {slope:.6f}, Intercept: {intercept:.6f}")
        print(f"  Total Error: {total_error:.6f}")
        print(f"  Structural Error: {structural_error:.6f}")
        print(f"  Approximation Error: {approx_error:.6f}")
        print(f"  Structural Error + Approximation Error: {structural_error + approx_error:.6f}")
        print(f"  Difference: {total_error - (structural_error + approx_error):.6f}")
        print()
    
    # Create visualization for error vs. sample size
    print("Creating second visualization...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(sample_sizes, total_errors, 'bo-', linewidth=2, markersize=8, label='Total Error')
    plt.plot(sample_sizes, [structural_error] * len(sample_sizes), 'g--', linewidth=2, 
             label='Structural Error (constant)')
    plt.plot(sample_sizes, approximation_errors, 'ro-', linewidth=2, markersize=8, label='Approximation Error')
    
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title('Error Components vs. Sample Size', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    
    # Add theoretical curve for approximation error (proportional to 1/n)
    approx_theory = approximation_errors[0] * (sample_sizes[0] / np.array(sample_sizes))
    plt.plot(sample_sizes, approx_theory, 'r--', linewidth=1, label='Theory: ∝ 1/n')
    
    # Add annotations
    plt.text(0.5, 0.9, 'As n increases:\n- Structural error remains constant\n- Approximation error decreases ∝ 1/n\n- Total error approaches structural error',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, "error_vs_sample_size.png")
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {file_path}")
    
    # Create visualization for multiple models with different sample sizes
    print("Creating third visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot true function
    plt.plot(x_grid, y_true, 'g-', linewidth=2, label='True Function f(x) = sin(x)')
    
    # Plot optimal linear model
    plt.plot(x_grid, y_optimal_linear, 'k-', linewidth=2, 
             label=f'Optimal Linear Model (w*)')
    
    # Colors for different sample sizes
    colors = ['r', 'orange', 'b', 'purple', 'brown']
    
    # Plot models from different sample sizes
    for i, n in enumerate(sample_sizes):
        y_pred = slopes[i] * x_grid.flatten() + intercepts[i]
        plt.plot(x_grid, y_pred, color=colors[i], linestyle='-', linewidth=1, alpha=0.7, 
                 label=f'Model with n={n}')
    
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Linear Models with Different Sample Sizes', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, "models_with_different_sample_sizes.png")
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {file_path}")
    
    return sample_sizes, total_errors, structural_error, approximation_errors

def bias_variance_decomposition():
    """Demonstrate the relationship between errors and bias-variance decomposition"""
    print("\nPart 3: Bias-Variance Decomposition")
    
    # Number of different datasets/models to generate
    n_models = 100
    n_samples = 30  # Sample size for each dataset
    
    # Generate a grid of x values for plotting the true function
    x_grid = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
    y_true = true_function(x_grid.flatten())
    
    # Get the optimal linear parameters
    optimal_slope, optimal_intercept = get_optimal_linear_params()
    y_optimal_linear = optimal_slope * x_grid.flatten() + optimal_intercept
    
    # Calculate structural error (squared bias)
    structural_error = mean_squared_error(y_true, y_optimal_linear)
    
    # Lists to store models and predictions
    all_slopes = []
    all_intercepts = []
    all_predictions = []
    
    print(f"Generating {n_models} different models...")
    # Generate multiple datasets and fit models
    for i in range(n_models):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_models} models")
        # Generate dataset
        x_train, y_train = generate_data(n_samples)
        
        # Fit model
        model = LinearRegression()
        model.fit(x_train, y_train)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Store parameters
        all_slopes.append(slope)
        all_intercepts.append(intercept)
        
        # Predict
        y_pred = slope * x_grid.flatten() + intercept
        all_predictions.append(y_pred)
    
    # Convert to numpy array for easier calculations
    all_predictions = np.array(all_predictions)
    
    # Calculate the average prediction at each x
    average_prediction = np.mean(all_predictions, axis=0)
    
    # Calculate bias^2 at each x (between average prediction and true function)
    bias_squared = (average_prediction - y_true) ** 2
    
    # Calculate variance at each x (variance of predictions across models)
    variance = np.var(all_predictions, axis=0)
    
    # Calculate MSE at each x (average squared error across models)
    mse_x = np.mean((all_predictions - y_true.reshape(1, -1)) ** 2, axis=0)
    
    # Calculate average values across all x
    avg_bias_squared = np.mean(bias_squared)
    avg_variance = np.mean(variance)
    avg_mse = np.mean(mse_x)
    
    print(f"Average Bias^2: {avg_bias_squared:.6f}")
    print(f"Average Variance: {avg_variance:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Bias^2 + Variance: {avg_bias_squared + avg_variance:.6f}")
    print(f"Difference: {avg_mse - (avg_bias_squared + avg_variance):.6f}")
    
    # Create visualization
    print("Creating bias-variance decomposition visualization...")
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[3, 1])
    
    # Plot 1: Multiple models and their average
    ax1 = plt.subplot(gs[0, :])
    
    # Plot true function
    ax1.plot(x_grid, y_true, 'g-', linewidth=3, label='True Function f(x) = sin(x)')
    
    # Plot a subset of models
    for i in range(min(20, n_models)):
        ax1.plot(x_grid, all_predictions[i], 'r-', linewidth=0.5, alpha=0.3)
    
    # Plot average prediction
    ax1.plot(x_grid, average_prediction, 'b-', linewidth=2, label='Average Prediction (over all models)')
    
    # Plot optimal linear model
    ax1.plot(x_grid, y_optimal_linear, 'k--', linewidth=2, label='Optimal Linear Model (w*)')
    
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax1.set_title('Multiple Models with Different Training Datasets', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    
    # Plot 2: Bias and Variance across x
    ax2 = plt.subplot(gs[1, 0])
    
    ax2.plot(x_grid, bias_squared, 'r-', linewidth=2, label=f'Bias² (avg: {avg_bias_squared:.4f})')
    ax2.plot(x_grid, variance, 'b-', linewidth=2, label=f'Variance (avg: {avg_variance:.4f})')
    ax2.plot(x_grid, mse_x, 'k-', linewidth=2, label=f'MSE (avg: {avg_mse:.4f})')
    
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Bias² and Variance Components', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    
    # Plot 3: Pie chart of error decomposition
    ax3 = plt.subplot(gs[1, 1])
    
    labels = ['Bias² (Structural)', 'Variance (Approximation)']
    sizes = [avg_bias_squared, avg_variance]
    explode = (0.1, 0)  # explode the first slice
    
    ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=['red', 'blue'])
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.set_title('MSE Decomposition', fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, "bias_variance_decomposition.png")
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {file_path}")
    
    # Create a diagram showing the error decomposition
    print("Creating error decomposition diagram...")
    plt.figure(figsize=(12, 8))
    
    # Create boxes for the error components
    box_width = 0.8
    box_height = 0.6
    
    # Main box - Total Error (MSE)
    plt.gca().add_patch(plt.Rectangle((0, 0), box_width*3, box_height*3, 
                                      facecolor='lightgray', edgecolor='black', 
                                      alpha=0.5))
    plt.text(box_width*1.5, box_height*3.1, 'Total Error (MSE)', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Left box - Bias²
    plt.gca().add_patch(plt.Rectangle((0.1, 0.1), box_width*1.4, box_height*2.8, 
                                      facecolor='red', edgecolor='black', 
                                      alpha=0.3))
    plt.text(box_width*0.8, box_height*1.5, 'Bias²\n(Structural Error)', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Right box - Variance
    plt.gca().add_patch(plt.Rectangle((0.1+box_width*1.5, 0.1), box_width*1.4, box_height*2.8, 
                                      facecolor='blue', edgecolor='black', 
                                      alpha=0.3))
    plt.text(box_width*2.2, box_height*1.5, 'Variance\n(Approximation Error)', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows and labels
    # From True Function to Linear Model Family
    plt.annotate('Structural Error\n(Model Misspecification)', 
                xy=(box_width*0.8, box_height*2.5), xytext=(box_width*0.8, box_height*4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', va='center', fontsize=10)
    
    # From Infinite Data to Finite Data
    plt.annotate('Approximation Error\n(Finite Sample Estimation)', 
                xy=(box_width*2.2, box_height*2.5), xytext=(box_width*2.2, box_height*4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center', va='center', fontsize=10)
    
    # Additional labels at the bottom
    plt.text(box_width*0.8, -0.5*box_height, 'Cannot be reduced\nby more data', 
             ha='center', va='center', fontsize=10)
    plt.text(box_width*2.2, -0.5*box_height, 'Decreases as sample\nsize increases (∝ 1/n)', 
             ha='center', va='center', fontsize=10)
    
    # Set axis limits and remove ticks
    plt.xlim(-0.5, box_width*3.5)
    plt.ylim(-box_height, box_height*4)
    plt.axis('off')
    
    # Title
    plt.suptitle('Decomposition of Mean Squared Error (MSE)\nin Linear Regression', fontsize=16, y=0.98)
    
    # Save the figure
    plt.tight_layout()
    file_path = os.path.join(save_dir, "error_decomposition_diagram.png")
    print(f"Saving figure to: {file_path}")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagram saved to: {file_path}")

def main():
    print("Linear Regression Error Decomposition: Structural vs. Approximation Error")
    print("=====================================================================")
    
    # Part 1: Demonstrate structural error vs. approximation error
    optimal_slope, optimal_intercept = demonstration_part1()
    
    # Part 2: Demonstrate the effect of sample size on approximation error
    sample_sizes, total_errors, structural_error, approximation_errors = demonstration_part2(optimal_slope, optimal_intercept)
    
    # Part 3: Bias-variance decomposition
    bias_variance_decomposition()
    
    print("\nSummary:")
    print("========")
    print("1. Structural Error (Bias²):")
    print("   - Represents the error due to model misspecification")
    print("   - Linear model cannot capture the true non-linear function")
    print("   - Does not decrease with more data")
    print(f"   - Value: {structural_error:.6f}")
    print()
    
    print("2. Approximation Error (Variance component):")
    print("   - Represents the error due to estimating parameters from finite data")
    print("   - Decreases as sample size increases (approximately proportional to 1/n)")
    print("   - Values for different sample sizes:")
    for i, n in enumerate(sample_sizes):
        print(f"     n = {n}: {approximation_errors[i]:.6f}")
    print()
    
    print("3. Total Error:")
    print("   - Sum of structural and approximation errors")
    print("   - Approaches the structural error as sample size increases")
    print("   - Values for different sample sizes:")
    for i, n in enumerate(sample_sizes):
        print(f"     n = {n}: {total_errors[i]:.6f}")
    print()
    
    print("All visualizations saved to:", save_dir)

if __name__ == "__main__":
    main()