import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

# Function to create and fit a simple linear regression model
def fit_simple_linear_regression(x, y):
    """
    Fit a simple linear regression model: y = β₀ + β₁x.
    Returns the intercept (β₀), slope (β₁), predicted values, and residuals.
    """
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope (β₁)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    beta1 = numerator / denominator
    
    # Calculate intercept (β₀)
    beta0 = y_mean - beta1 * x_mean
    
    # Calculate predicted values
    y_pred = beta0 + beta1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    return beta0, beta1, y_pred, residuals

# Function to prove and demonstrate that sum of residuals equals zero
def prove_sum_residuals_zero(x, y, beta0, beta1, residuals):
    print_section_header("PROOF 1: Demonstrating Σe_i = 0 when model includes intercept")
    
    # Step 1: Show residual definition
    print("Step 1: Define residuals")
    print("e_i = y_i - ŷ_i = y_i - (β₀ + β₁x_i)")
    print()
    
    # Step 2: Sum the residuals
    print("Step 2: Sum the residuals")
    print("Σe_i = Σ(y_i - (β₀ + β₁x_i))")
    print("     = Σy_i - Σ(β₀ + β₁x_i)")
    print("     = Σy_i - β₀Σ(1) - β₁Σx_i")
    print("     = Σy_i - nβ₀ - β₁Σx_i  (where n is the number of observations)")
    print()
    
    # Step 3: Substitute the formula for β₀
    print("Step 3: Substitute the formula for β₀")
    print("We know that β₀ = ȳ - β₁x̄, where ȳ and x̄ are the means of y and x")
    print("Σe_i = Σy_i - n(ȳ - β₁x̄) - β₁Σx_i")
    print("     = Σy_i - nȳ + nβ₁x̄ - β₁Σx_i")
    print()
    
    # Step 4: Simplify using properties of means
    print("Step 4: Simplify using properties of means")
    print("We know that Σy_i = nȳ and Σx_i = nx̄")
    print("Σe_i = nȳ - nȳ + nβ₁x̄ - β₁nx̄")
    print("     = nβ₁x̄ - nβ₁x̄")
    print("     = 0")
    print()
    
    # Calculate and display numerical results
    sum_residuals = np.sum(residuals)
    theoretical_sum = 0
    
    print(f"Numerical verification:")
    print(f"Sum of residuals: {sum_residuals:.10f}")
    print(f"Theoretical value: {theoretical_sum}")
    print(f"Absolute difference: {abs(sum_residuals - theoretical_sum):.10f}")
    print(f"This confirms that Σe_i = 0 (up to numerical precision).")
    
    return sum_residuals

# Function to prove and demonstrate that sum of x_i * residuals equals zero
def prove_sum_x_residuals_zero(x, y, beta0, beta1, residuals):
    print_section_header("PROOF 2: Demonstrating Σx_i·e_i = 0")
    
    # Step 1: Start with the definition
    print("Step 1: Define the sum of x_i multiplied by residuals")
    print("Σx_i·e_i = Σx_i·(y_i - ŷ_i)")
    print("         = Σx_i·(y_i - (β₀ + β₁x_i))")
    print("         = Σx_i·y_i - Σx_i·(β₀ + β₁x_i)")
    print("         = Σx_i·y_i - β₀Σx_i - β₁Σx_i²")
    print()
    
    # Step 2: Use the normal equations
    print("Step 2: Use the normal equations for OLS")
    print("The OLS estimator β₁ is derived from the normal equations:")
    print("β₁ = (Σx_i·y_i - nx̄ȳ) / (Σx_i² - nx̄²)")
    print("Rearranging: Σx_i·y_i - nx̄ȳ = β₁(Σx_i² - nx̄²)")
    print("Therefore: Σx_i·y_i = β₁(Σx_i² - nx̄²) + nx̄ȳ")
    print()
    
    # Step 3: Substitute into our expression
    print("Step 3: Substitute this into our expression for Σx_i·e_i")
    print("Σx_i·e_i = β₁(Σx_i² - nx̄²) + nx̄ȳ - β₀Σx_i - β₁Σx_i²")
    print("         = β₁(Σx_i² - nx̄²) + nx̄ȳ - (ȳ - β₁x̄)Σx_i - β₁Σx_i²")
    print("         = β₁(Σx_i² - nx̄²) + nx̄ȳ - ȳΣx_i + β₁x̄Σx_i - β₁Σx_i²")
    print()
    
    # Step 4: Simplify using Σx_i = nx̄
    print("Step 4: Simplify using Σx_i = nx̄")
    print("Σx_i·e_i = β₁(Σx_i² - nx̄²) + nx̄ȳ - ȳnx̄ + β₁x̄nx̄ - β₁Σx_i²")
    print("         = β₁(Σx_i² - nx̄²) + nx̄ȳ - nx̄ȳ + β₁nx̄² - β₁Σx_i²")
    print("         = β₁(Σx_i² - nx̄²) + β₁nx̄² - β₁Σx_i²")
    print("         = β₁(Σx_i² - nx̄² + nx̄² - Σx_i²)")
    print("         = β₁(0)")
    print("         = 0")
    print()
    
    # Calculate and display numerical results
    sum_x_residuals = np.sum(x * residuals)
    theoretical_sum = 0
    
    print(f"Numerical verification:")
    print(f"Sum of x_i·e_i: {sum_x_residuals:.10f}")
    print(f"Theoretical value: {theoretical_sum}")
    print(f"Absolute difference: {abs(sum_x_residuals - theoretical_sum):.10f}")
    print(f"This confirms that Σx_i·e_i = 0 (up to numerical precision).")
    
    return sum_x_residuals

# Function to explain what these properties tell us about residuals
def explain_residual_properties():
    print_section_header("INTERPRETATION: What these properties tell us")
    
    print("1. Property: Σe_i = 0")
    print("   - This means the residuals sum to zero")
    print("   - The positive residuals exactly balance out the negative residuals")
    print("   - Geometrically, the fitted line passes through the centroid (x̄, ȳ) of the data")
    print("   - The average of the residuals is exactly zero")
    print()
    
    print("2. Property: Σx_i·e_i = 0")
    print("   - This means the residuals are uncorrelated with the predictor variable x")
    print("   - There is no systematic pattern in the residuals with respect to x")
    print("   - This is a fundamental assumption of linear regression (no patterns in residuals)")
    print("   - This property ensures that we've extracted all the linear information from x")
    print()
    
    print("3. Combined interpretation:")
    print("   - These properties are direct consequences of OLS (Ordinary Least Squares) estimation")
    print("   - They are mathematical properties of the minimization problem")
    print("   - Together, they ensure that our model has 'used up' all linear information")
    print("   - If either property is violated, then the model is not optimally fit")
    print()

# Function to explain how to use these properties for checking calculations
def explain_checking_calculations():
    print_section_header("PRACTICAL USE: How to check regression calculations")
    
    print("These properties provide simple checks for the correctness of regression calculations:")
    print()
    
    print("1. Check that Σe_i ≈ 0:")
    print("   - After fitting a regression model, compute the residuals e_i = y_i - ŷ_i")
    print("   - Calculate the sum of all residuals")
    print("   - If the sum is not very close to zero (accounting for floating-point precision),")
    print("     there may be an error in your calculations")
    print()
    
    print("2. Check that Σx_i·e_i ≈ 0:")
    print("   - Multiply each residual by its corresponding x-value")
    print("   - Calculate the sum of these products")
    print("   - If not very close to zero, there may be an error in your slope calculation")
    print()
    
    print("3. Additional checks you could make:")
    print("   - Check that residuals have zero correlation with fitted values")
    print("   - Verify that Σŷ_i·e_i = 0, another property of OLS residuals")
    print("   - For multiple regression, check that each predictor is uncorrelated with residuals")
    print()
    
    print("4. Practical implementation:")
    print("   - Include these checks as automated tests in regression code")
    print("   - Use absolute tolerances (e.g., |Σe_i| < 1e-10) rather than exact equality")
    print("   - For large datasets or extreme values, consider relative tolerances or normalize first")
    print()

# Function to visualize the results
def visualize_residual_properties(x, y, beta0, beta1, residuals, sum_residuals, sum_x_residuals, save_dir):
    saved_files = []
    
    # Figure 1: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    y_pred = beta0 + beta1 * x
    
    plt.scatter(x, y, color='blue', alpha=0.6, label='Data points')
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'Fitted line: y = {beta0:.2f} + {beta1:.2f}x')
    
    # Highlight the centroid
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    plt.scatter([x_mean], [y_mean], color='green', s=100, label='Centroid (x̄, ȳ)', zorder=5)
    
    # Add annotations
    plt.title('Linear Regression Fit with Centroid', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot1_regression_fit.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Figure 2: Residual plot
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x, residuals, color='blue', alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1, label='Zero line')
    
    # Add text for sum of residuals
    plt.text(0.02, 0.95, f'Sum of residuals: {sum_residuals:.10f}', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Residuals vs. Predictor Variable', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Residuals (e_i)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot2_residuals_vs_x.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Figure 3: x_i * e_i plot
    plt.figure(figsize=(10, 6))
    
    x_times_residuals = x * residuals
    plt.stem(x, x_times_residuals, linefmt='b-', markerfmt='bo', basefmt='r-', 
           label='x_i · e_i values')
    
    # Add text for sum of x_i * residuals
    plt.text(0.02, 0.95, f'Sum of x_i · e_i: {sum_x_residuals:.10f}', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('x_i · e_i Values (Should Sum to Zero)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('x_i · e_i', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot3_x_times_residuals.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Figure 4: Illustration of orthogonality
    plt.figure(figsize=(10, 6))
    
    # Create vectors to illustrate orthogonality
    x_vec = x - np.mean(x)  # Center x (not necessary for the proof but helps visualization)
    res_vec = residuals
    
    # Normalize for better visualization
    x_vec_norm = x_vec / np.linalg.norm(x_vec)
    res_vec_norm = res_vec / np.linalg.norm(res_vec)
    
    # Create a scatter plot showing vectors
    plt.scatter(x_vec_norm, res_vec_norm, alpha=0.3, label='Data points')
    
    # Add vector representations at the origin
    plt.quiver(0, 0, np.mean(x_vec_norm), 0, angles='xy', scale_units='xy', scale=1, 
             color='red', width=0.008, label='x direction')
    plt.quiver(0, 0, 0, np.mean(res_vec_norm), angles='xy', scale_units='xy', scale=1, 
             color='blue', width=0.008, label='residual direction')
    
    # Add text for sum
    plt.text(0.02, 0.95, f'x · e (dot product): {np.dot(x_vec, res_vec):.10f}', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Orthogonality of Predictor and Residuals', fontsize=14)
    plt.xlabel('x direction', fontsize=12)
    plt.ylabel('Residual direction', fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Set equal aspect ratio to properly show orthogonality
    plt.axis('equal')
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot4_orthogonality.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Figure 5: Geometric interpretation
    plt.figure(figsize=(12, 6))
    
    # Create a grid of points representing a 2D vector space
    x_grid = np.linspace(-4, 4, 20)
    y_grid = np.linspace(-4, 4, 20)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Plot the subspaces
    plt.figure(figsize=(12, 6))
    
    # Plot the 'x-space' (column space of X)
    # Scale for visualization
    x_direction = np.array([1, 0])
    constant_direction = np.array([0, 1])
    
    # Plot data points (centered)
    x_centered = x_vec
    y_centered = y - np.mean(y)
    y_pred_centered = beta1 * x_centered
    residual_vectors = y_centered - y_pred_centered
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_centered, y_centered, alpha=0.7, label='Data (centered)')
    
    # Plot the projection
    for i in range(len(x_centered)):
        plt.plot([x_centered[i], x_centered[i]], [0, y_pred_centered[i]], 'gray', alpha=0.4)
        plt.plot([x_centered[i], x_centered[i]], [y_pred_centered[i], y_centered[i]], 'r-', alpha=0.4)
    
    plt.plot(x_centered, y_pred_centered, 'g-', label='Projection onto x-space')
    plt.scatter(x_centered, y_pred_centered, c='g', s=30)
    
    plt.title('Projection Interpretation', fontsize=14)
    plt.xlabel('x (centered)', fontsize=12)
    plt.ylabel('y (centered)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Plot the orthogonality relation
    plt.subplot(1, 2, 2)
    
    # Create scatter plot of (x, residual)
    plt.scatter(x, residuals, alpha=0.7, label='(x_i, e_i) pairs')
    
    # Add annotations
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add a covariance line (should be flat)
    z = np.polyfit(x, residuals, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', label=f'Best fit: slope={z[0]:.10f}')
    
    plt.text(0.02, 0.95, f'Correlation(x, e): {np.corrcoef(x, residuals)[0,1]:.10f}', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Orthogonality: x vs Residuals', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Residuals (e_i)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot5_geometric_interpretation.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create data for demonstration
def generate_data():
    print_section_header("GENERATING SAMPLE DATA")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate x values
    n = 50  # Sample size
    x = np.linspace(0, 10, n) + np.random.normal(0, 0.5, n)
    
    # True regression line parameters
    true_beta0 = 2.5
    true_beta1 = 1.8
    
    # Generate y values with some noise
    epsilon = np.random.normal(0, 2, n)
    y = true_beta0 + true_beta1 * x + epsilon
    
    print(f"Generated a dataset with {n} points")
    print(f"True intercept (β₀): {true_beta0}")
    print(f"True slope (β₁): {true_beta1}")
    print(f"Error standard deviation: {2.0}")
    
    return x, y

# Main function
def main():
    # Set nicer formatting for printing
    np.set_printoptions(precision=6, suppress=True)
    
    # Generate data
    x, y = generate_data()
    
    # Fit linear regression model
    beta0, beta1, y_pred, residuals = fit_simple_linear_regression(x, y)
    
    print_section_header("FITTED MODEL")
    print(f"Fitted intercept (β₀): {beta0:.6f}")
    print(f"Fitted slope (β₁): {beta1:.6f}")
    print(f"Number of data points: {len(x)}")
    print(f"Mean of x: {np.mean(x):.6f}")
    print(f"Mean of y: {np.mean(y):.6f}")
    print(f"Mean of residuals: {np.mean(residuals):.10f}")
    
    # Prove sum of residuals equals zero
    sum_residuals = prove_sum_residuals_zero(x, y, beta0, beta1, residuals)
    
    # Prove sum of x_i * residuals equals zero
    sum_x_residuals = prove_sum_x_residuals_zero(x, y, beta0, beta1, residuals)
    
    # Explain what these properties tell us
    explain_residual_properties()
    
    # Explain how to use these properties for checking calculations
    explain_checking_calculations()
    
    # Visualize results
    saved_files = visualize_residual_properties(x, y, beta0, beta1, residuals, 
                                           sum_residuals, sum_x_residuals, save_dir)
    
    print_section_header("VISUALIZATIONS GENERATED")
    for i, file_path in enumerate(saved_files, 1):
        print(f"{i}. {os.path.basename(file_path)}")
    
    print("\nSUMMARY OF FINDINGS:")
    print(f"1. Sum of residuals: {sum_residuals:.10f} (should be 0)")
    print(f"2. Sum of x_i · residuals: {sum_x_residuals:.10f} (should be 0)")
    print("3. These properties are direct consequences of OLS estimation")
    print("4. They provide a way to check regression calculations for accuracy")

if __name__ == "__main__":
    main() 