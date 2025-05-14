import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 1
house_sizes = np.array([1000, 1500, 2000, 2500, 3000])
house_prices = np.array([150, 200, 250, 300, 350])

# Step 1: Calculate the least squares estimates for slope and intercept using matrix approach
def calculate_least_squares_matrix(x, y):
    """Calculate the least squares estimates using the matrix formula: β̂ = (X^T X)^{-1} X^T y"""
    n = len(x)
    
    # Calculate means (for explanation purposes)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1.1: Calculate means (for reference)")
    print(f"Mean of house sizes (x̄): {x_mean} sq ft")
    print(f"Mean of house prices (ȳ): ${y_mean}k")
    print()
    
    # Step 1.2: Create the design matrix X (adding a column of 1s for the intercept)
    # Initialize an empty matrix of size n×2
    X = np.zeros((n, 2))
    
    # Fill the first column with 1s for the intercept
    X[:, 0] = 1
    
    # Fill the second column with the x values
    X[:, 1] = x
    
    print(f"Step 1.2: Create the design matrix X (with column of 1s for intercept)")
    print("X = ")
    for i in range(n):
        print(f"    [{X[i,0]:.1f}, {X[i,1]:.1f}]  # For data point {i+1}: (size={x[i]}, price={y[i]})")
    print()
    
    # Step 1.3: Calculate X^T (transpose of X)
    X_T = np.zeros((2, n))
    for i in range(2):
        for j in range(n):
            X_T[i, j] = X[j, i]
    
    print(f"Step 1.3: Calculate X^T (transpose of X)")
    print("X^T = ")
    for i in range(2):
        row = "    ["
        for j in range(n):
            row += f"{X_T[i,j]:.1f}, "
        row = row[:-2] + "]"  # remove last comma and space, add closing bracket
        print(row)
    print()
    
    # Step 1.4: Calculate X^T X manually
    X_T_X = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            sum_val = 0
            for k in range(n):
                sum_val += X_T[i, k] * X[k, j]
            X_T_X[i, j] = sum_val
            
    print(f"Step 1.4: Calculate X^T X manually")
    print("X^T X = ")
    print(f"    [{X_T_X[0,0]:.1f}, {X_T_X[0,1]:.1f}]")
    print(f"    [{X_T_X[1,0]:.1f}, {X_T_X[1,1]:.1f}]")
    
    # Explaining the calculation of X^T X elements
    print("\nDetailed calculation of X^T X elements:")
    
    # Top-left element (sum of 1^2, n times)
    print(f"X^T_X[0,0] = sum of X_T[0,k] * X[k,0] for k=0 to {n-1}")
    detail = " + ".join([f"{X_T[0,k]:.1f} × {X[k,0]:.1f}" for k in range(n)])
    print(f"X^T_X[0,0] = {detail} = {X_T_X[0,0]:.1f}")
    
    # Top-right element (sum of 1 × each x value)
    print(f"X^T_X[0,1] = sum of X_T[0,k] * X[k,1] for k=0 to {n-1}")
    detail = " + ".join([f"{X_T[0,k]:.1f} × {X[k,1]:.1f}" for k in range(n)])
    print(f"X^T_X[0,1] = {detail} = {X_T_X[0,1]:.1f}")
    
    # Bottom-left element (same as top-right due to symmetry)
    print(f"X^T_X[1,0] = sum of X_T[1,k] * X[k,0] for k=0 to {n-1}")
    detail = " + ".join([f"{X_T[1,k]:.1f} × {X[k,0]:.1f}" for k in range(n)])
    print(f"X^T_X[1,0] = {detail} = {X_T_X[1,0]:.1f}")
    
    # Bottom-right element (sum of x^2 values)
    print(f"X^T_X[1,1] = sum of X_T[1,k] * X[k,1] for k=0 to {n-1}")
    detail = " + ".join([f"{X_T[1,k]:.1f} × {X[k,1]:.1f}" for k in range(n)])
    print(f"X^T_X[1,1] = {detail} = {X_T_X[1,1]:.1f}")
    print()
    
    # Step 1.5: Calculate (X^T X)^(-1) using the formula for 2x2 matrix inversion
    # For a 2x2 matrix M = [[a, b], [c, d]], M^(-1) = (1/(ad-bc)) * [[d, -b], [-c, a]]
    a = X_T_X[0, 0]
    b = X_T_X[0, 1]
    c = X_T_X[1, 0]
    d = X_T_X[1, 1]
    
    # Calculate determinant
    det = a * d - b * c
    
    # Calculate inverse elements
    X_T_X_inv = np.zeros((2, 2))
    X_T_X_inv[0, 0] = d / det
    X_T_X_inv[0, 1] = -b / det
    X_T_X_inv[1, 0] = -c / det
    X_T_X_inv[1, 1] = a / det
    
    print(f"Step 1.5: Calculate (X^T X)^(-1) using the formula for 2x2 matrix inversion")
    print("For a 2x2 matrix M = [[a, b], [c, d]], M^(-1) = (1/(ad-bc)) * [[d, -b], [-c, a]]")
    print(f"a = {a:.1f}, b = {b:.1f}, c = {c:.1f}, d = {d:.1f}")
    print(f"det = a*d - b*c = {a:.1f}×{d:.1f} - {b:.1f}×{c:.1f} = {det:.1f}")
    print(f"X^T_X_inv[0,0] = d/det = {d:.1f}/{det:.1f} = {X_T_X_inv[0,0]:.8f}")
    print(f"X^T_X_inv[0,1] = -b/det = -{b:.1f}/{det:.1f} = {X_T_X_inv[0,1]:.8f}")
    print(f"X^T_X_inv[1,0] = -c/det = -{c:.1f}/{det:.1f} = {X_T_X_inv[1,0]:.8f}")
    print(f"X^T_X_inv[1,1] = a/det = {a:.1f}/{det:.1f} = {X_T_X_inv[1,1]:.8f}")
    
    print("\n(X^T X)^(-1) = ")
    print(f"    [{X_T_X_inv[0,0]:.8f}, {X_T_X_inv[0,1]:.8f}]")
    print(f"    [{X_T_X_inv[1,0]:.8f}, {X_T_X_inv[1,1]:.8f}]")
    print()
    
    # Step 1.6: Calculate X^T y manually
    X_T_y = np.zeros(2)
    for i in range(2):
        sum_val = 0
        for j in range(n):
            sum_val += X_T[i, j] * y[j]
        X_T_y[i] = sum_val
    
    print(f"Step 1.6: Calculate X^T y manually")
    print("X^T y = ")
    
    # First element calculation (sum of all y values)
    print(f"X^T_y[0] = sum of X_T[0,j] * y[j] for j=0 to {n-1}")
    detail = " + ".join([f"{X_T[0,j]:.1f} × {y[j]:.1f}" for j in range(n)])
    print(f"X^T_y[0] = {detail} = {X_T_y[0]:.1f}")
    
    # Second element calculation (sum of x*y products)
    print(f"X^T_y[1] = sum of X_T[1,j] * y[j] for j=0 to {n-1}")
    detail = " + ".join([f"{X_T[1,j]:.1f} × {y[j]:.1f}" for j in range(n)])
    print(f"X^T_y[1] = {detail} = {X_T_y[1]:.1f}")
    
    print(f"\nX^T y = [{X_T_y[0]:.1f}, {X_T_y[1]:.1f}]")
    print()
    
    # Step 1.7: Calculate β̂ = (X^T X)^(-1) X^T y manually
    beta = np.zeros(2)
    for i in range(2):
        sum_val = 0
        for j in range(2):
            sum_val += X_T_X_inv[i, j] * X_T_y[j]
        beta[i] = sum_val
    
    print(f"Step 1.7: Calculate β̂ = (X^T X)^(-1) X^T y manually")
    
    # Intercept calculation
    print(f"β₀ = X^T_X_inv[0,0] * X^T_y[0] + X^T_X_inv[0,1] * X^T_y[1]")
    print(f"β₀ = {X_T_X_inv[0,0]:.8f} × {X_T_y[0]:.1f} + {X_T_X_inv[0,1]:.8f} × {X_T_y[1]:.1f}")
    print(f"β₀ = {X_T_X_inv[0,0] * X_T_y[0]:.8f} + {X_T_X_inv[0,1] * X_T_y[1]:.8f}")
    print(f"β₀ = {beta[0]:.8f}")
    
    # Slope calculation
    print(f"β₁ = X^T_X_inv[1,0] * X^T_y[0] + X^T_X_inv[1,1] * X^T_y[1]")
    print(f"β₁ = {X_T_X_inv[1,0]:.8f} × {X_T_y[0]:.1f} + {X_T_X_inv[1,1]:.8f} × {X_T_y[1]:.1f}")
    print(f"β₁ = {X_T_X_inv[1,0] * X_T_y[0]:.8f} + {X_T_X_inv[1,1] * X_T_y[1]:.8f}")
    print(f"β₁ = {beta[1]:.8f}")
    
    print(f"\nβ = [{beta[0]:.8f}, {beta[1]:.8f}]")
    print(f"β₀ (intercept) = {beta[0]:.8f}")
    print(f"β₁ (slope) = {beta[1]:.8f}")
    print()
    
    # Double-check with the traditional formula 
    print(f"Verification: Calculate using traditional formulas")
    numerator = sum((x - x_mean) * (y - y_mean))
    denominator = sum((x - x_mean) ** 2)
    beta_1_check = numerator / denominator
    beta_0_check = y_mean - beta_1_check * x_mean
    
    # Show detailed calculation
    print("Traditional formula:")
    print(f"Numerator = sum((x_i - x̄)(y_i - ȳ))")
    numerator_terms = [f"({x[i]:.1f} - {x_mean:.1f})({y[i]:.1f} - {y_mean:.1f}) = {(x[i] - x_mean):.1f}×{(y[i] - y_mean):.1f} = {(x[i] - x_mean)*(y[i] - y_mean):.1f}" for i in range(n)]
    print("Numerator = " + " + ".join(numerator_terms))
    print(f"Numerator = {numerator:.1f}")
    
    print(f"Denominator = sum((x_i - x̄)²)")
    denominator_terms = [f"({x[i]:.1f} - {x_mean:.1f})² = {(x[i] - x_mean):.1f}² = {(x[i] - x_mean)**2:.1f}" for i in range(n)]
    print("Denominator = " + " + ".join(denominator_terms))
    print(f"Denominator = {denominator:.1f}")
    
    print(f"β₁ = Numerator / Denominator = {numerator:.1f} / {denominator:.1f} = {beta_1_check:.8f}")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean:.1f} - {beta_1_check:.8f}×{x_mean:.1f} = {y_mean:.1f} - {beta_1_check * x_mean:.8f} = {beta_0_check:.8f}")
    
    print(f"\nMatrix formula: β₁ = {beta[1]:.8f}, β₀ = {beta[0]:.8f}")
    print(f"Traditional formula: β₁ = {beta_1_check:.8f}, β₀ = {beta_0_check:.8f}")
    print(f"The results are {'identical' if np.allclose(np.array([beta_0_check, beta_1_check]), beta) else 'different'}")
    print()
    
    return beta[0], beta[1]

# Execute Step 1
beta_0, beta_1 = calculate_least_squares_matrix(house_sizes, house_prices)

# Print the regression equation
print(f"Linear Regression Equation: Price = {beta_0:.2f} + {beta_1:.4f} · Size")
print()

# Step 2: Interpret the slope coefficient
print("Step 2: Interpret the slope coefficient")
print(f"The slope coefficient (β₁ = {beta_1:.4f}) represents the change in house price (in $1000s)")
print(f"for each additional square foot of house size.")
print(f"In other words, for every additional square foot, the house price increases by ${beta_1*1000:.2f}.")
print()

# Step 3: Calculate prediction for a house with 1800 square feet
new_size = 1800
predicted_price = beta_0 + beta_1 * new_size

print("Step 3: Calculate prediction for a house with 1800 square feet")
print(f"Predicted price = {beta_0:.2f} + {beta_1:.4f} · 1800 = {beta_0:.2f} + {beta_1 * new_size:.2f} = ${predicted_price:.2f}k")
print()

# Step 4: Calculate residuals and residual sum of squares (RSS)
def calculate_residuals(x, y, beta_0, beta_1):
    """Calculate residuals and residual sum of squares using matrix notation."""
    n = len(x)
    
    # Step 4.1: Create the design matrix X again
    X = np.column_stack((np.ones(n), x))
    
    # Step 4.2: Create the parameter vector β
    beta = np.array([beta_0, beta_1])
    
    # Step 4.3: Calculate predicted values using matrix multiplication
    y_pred = np.zeros(n)
    for i in range(n):
        y_pred[i] = 0
        for j in range(2):  # 2 columns in X
            y_pred[i] += X[i, j] * beta[j]
    
    print("Step 4.1: Calculate predicted values (ŷ = Xβ)")
    for i in range(n):
        print(f"ŷ_{i+1} = {X[i,0]:.1f}×{beta[0]:.2f} + {X[i,1]:.1f}×{beta[1]:.4f} = {X[i,0]*beta[0]:.2f} + {X[i,1]*beta[1]:.2f} = {y_pred[i]:.2f}")
    print()
    
    # Step 4.4: Calculate residuals
    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = y[i] - y_pred[i]
    
    print("Step 4.2: Calculate residuals (e = y - ŷ)")
    for i in range(n):
        print(f"e_{i+1} = {y[i]:.1f} - {y_pred[i]:.2f} = {residuals[i]:.2f}")
    print()
    
    # Step 4.5: Calculate squared residuals
    squared_residuals = np.zeros(n)
    for i in range(n):
        squared_residuals[i] = residuals[i] ** 2
    
    print("Step 4.3: Calculate squared residuals (e²)")
    for i in range(n):
        print(f"e²_{i+1} = ({residuals[i]:.2f})² = {squared_residuals[i]:.4f}")
    print()
    
    # Step 4.6: Calculate RSS (sum of squared residuals)
    rss = np.sum(squared_residuals)
    
    print("Step 4.4: Calculate Residual Sum of Squares (RSS = sum of e²)")
    rss_terms = [f"{squared_residuals[i]:.4f}" for i in range(n)]
    print(f"RSS = {' + '.join(rss_terms)} = {rss:.4f}")
    print()
    
    # Alternative: Calculate RSS using matrix notation
    # Convert to column vectors
    y_vector = y.reshape(-1, 1)
    y_pred_vector = y_pred.reshape(-1, 1)
    residuals_vector = residuals.reshape(-1, 1)
    
    # RSS = e^T e (residuals-transpose * residuals)
    rss_matrix = residuals_vector.T @ residuals_vector
    rss_from_matrix = rss_matrix[0, 0]
    
    print("Step 4.5: Verify RSS using matrix notation (RSS = e^T e)")
    print(f"RSS using matrix notation: {rss_from_matrix:.4f}")
    print()
    
    # Print residuals table
    print("Summary table:")
    print("House Size (x) | Price (y) | Predicted Price (ŷ) | Residual (y - ŷ) | Squared Residual (y - ŷ)²")
    print("---------------------------------------------------------------------------------")
    
    for i in range(n):
        print(f"{x[i]:^13} | ${y[i]:^8}k | ${y_pred[i]:^17.2f}k | ${residuals[i]:^14.2f}k | ${squared_residuals[i]:^20.4f}k²")
    
    print(f"\nResidual Sum of Squares (RSS) = sum(residuals²) = {rss:.4f}")
    
    return y_pred, residuals, rss

# Execute Step 4
y_pred, residuals, rss = calculate_residuals(house_sizes, house_prices, beta_0, beta_1)

# Create visualizations
def create_visualizations(x, y, y_pred, residuals, beta_0, beta_1, new_x=None, new_y=None, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 200, max(x) + 200, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.4f}x')
    
    # If we have a new prediction point, add it
    if new_x is not None and new_y is not None:
        plt.scatter(new_x, new_y, color='green', s=100, label=f'Prediction: ${new_y:.2f}k')
        plt.plot([new_x, new_x], [0, new_y], 'g--', alpha=0.5)
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('House Price ($1000s)', fontsize=12)
    plt.title('Linear Regression: House Price vs Size (Matrix Approach)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_regression_line_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residuals plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('Residuals ($1000s)', fontsize=12)
    plt.title('Residuals Plot (Matrix Approach)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_residuals_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Visual representation of the least squares principle
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Regression Line')
    
    # Draw lines for the squared residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.7)
        
        # Create rectangle to represent squared error
        width = 100  # Width of the squared error illustration
        rect_x = x[i] - width/2
        
        if residuals[i] >= 0:
            rect_y = y_pred[i]
            rect_height = residuals[i]
        else:
            rect_y = y[i]
            rect_height = -residuals[i]
            
        plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), width, rect_height, 
                                    color='orange', alpha=0.4))
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('House Price ($1000s)', fontsize=12)
    plt.title('Visualization of Squared Residuals (Matrix Approach)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_squared_residuals_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Actual vs. Predicted Values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='green', s=100)
    
    # Add reference line (perfect predictions)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add data point labels
    for i in range(len(x)):
        plt.annotate(f"{x[i]} sq ft", (y[i], y_pred[i]), 
                    xytext=(7, 3), textcoords='offset points')
    
    plt.xlabel('Actual Price ($1000s)', fontsize=12)
    plt.ylabel('Predicted Price ($1000s)', fontsize=12)
    plt.title('Actual vs. Predicted Prices (Matrix Approach)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_actual_vs_predicted_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Histogram of Residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=5, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.7)
    plt.xlabel('Residuals ($1000s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Residuals (Matrix Approach)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_residuals_histogram_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 6: Matrix operations visualization
    plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2)
    
    # Define colors for matrix elements
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    
    # Design matrix X
    ax1 = plt.subplot(gs[0, 0])
    X = np.column_stack((np.ones(len(x)), x))
    ax1.imshow(X, cmap='viridis', aspect='auto')
    for i in range(len(x)):
        for j in range(2):
            ax1.text(j, i, f"{X[i, j]}", ha="center", va="center", color="white")
    ax1.set_title("Design Matrix X")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["1", "Size"])
    ax1.set_yticks(np.arange(len(x)))
    ax1.set_yticklabels([f"House {i+1}" for i in range(len(x))])
    
    # Parameter matrix β
    ax2 = plt.subplot(gs[0, 1])
    beta_matrix = np.array([[beta_0], [beta_1]])
    ax2.imshow(beta_matrix, cmap='viridis', aspect='auto')
    for i in range(2):
        ax2.text(0, i, f"{beta_matrix[i, 0]:.4f}", ha="center", va="center", color="white")
    ax2.set_title("Parameter Vector β")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Value"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["β₀", "β₁"])
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "matrix_operations_alt.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(house_sizes, house_prices, y_pred, residuals, beta_0, beta_1, 
                                   new_size, predicted_price, save_dir)

print(f"\nVisualizations saved to: {save_dir}")
for file in saved_files:
    print(f"  - {os.path.basename(file)}")

print("\nQuestion 1 Solution Summary:")
print(f"1. Least squares estimates: β₀ = {beta_0:.2f}, β₁ = {beta_1:.4f}")
print(f"2. Slope interpretation: For each additional square foot, house price increases by ${beta_1*1000:.2f}")
print(f"3. Predicted price for a 1800 sq ft house: ${predicted_price:.2f}k")
print(f"4. Residual Sum of Squares (RSS): {rss:.4f}") 