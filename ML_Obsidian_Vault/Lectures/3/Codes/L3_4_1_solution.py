import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Given data from the problem
x1 = np.array([2, 3, 1, 4])
x2 = np.array([5, 2, 4, 3])
x3 = np.array([1, 0, 2, 1])
y = np.array([12, 7, 11, 13])

# Print the raw data in tabular format
print("Raw data from the problem:")
data_table = pd.DataFrame({
    'x₁': x1,
    'x₂': x2,
    'x₃': x3,
    'y': y
})
print(data_table)
print()

# Step 1: Create the design matrix X and target vector y
def create_design_matrix():
    """Create the design matrix X with a column of ones for the intercept."""
    print("Step 1: Creating the design matrix X and target vector y")
    
    # Create the design matrix X with a column of ones for the intercept
    X = np.column_stack((np.ones(len(x1)), x1, x2, x3))
    
    print("Design matrix X:")
    print(X)
    print("\nTarget vector y:")
    print(y)
    print("\nDimensions of X:", X.shape)
    print("Dimensions of y:", y.shape)
    print()
    
    # Visualize the design matrix and target vector 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    
    # Design matrix as heatmap
    X_df = pd.DataFrame(X, columns=['Intercept', 'x₁', 'x₂', 'x₃'])
    sns.heatmap(X_df, annot=True, fmt=".1f", cmap="viridis", ax=ax1)
    ax1.set_title("Design Matrix X")
    
    # Target vector
    ax2.bar(range(len(y)), y, color='teal')
    ax2.set_xticks(range(len(y)))
    ax2.set_xticklabels([f'Obs {i+1}' for i in range(len(y))])
    ax2.set_ylabel('Target Value (y)')
    ax2.set_title('Target Vector y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix.png"), dpi=300)
    plt.close()
    
    # Explain design matrix conceptually
    print("Explanation of the design matrix:")
    print("- Each row represents one observation/data point")
    print("- The first column is all 1's for the intercept term")
    print("- The remaining columns contain the feature values for each observation")
    print("- For our problem, we have 4 observations and 3 features, plus the intercept")
    print()
    
    return X

X = create_design_matrix()

# Step 2: Express the multiple linear regression model in matrix form
def express_matrix_form():
    """Express the multiple linear regression model in matrix form."""
    print("Step 2: Expressing the multiple linear regression model in matrix form")
    print("The multiple linear regression model is given by:")
    print("y = Xw + ε")
    print("\nWhere:")
    print("- y is the target vector of shape (4,)")
    print("- X is the design matrix of shape (4, 4)")
    print("- w is the weight vector of shape (4,)")
    print("- ε is the vector of errors of shape (4,)")
    print()
    
    # Create a visualization of matrix dimensions for the model
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define sizes for the boxes (number of rows and columns)
    X_rows, X_cols = X.shape
    y_rows = len(y)
    w_rows = X_cols
    
    # Set positions for the boxes
    pos_X = [0.2, 0.5]  # [x, y] position for X
    pos_w = [0.5, 0.5]  # [x, y] position for w
    pos_y = [0.8, 0.5]  # [x, y] position for y
    
    # Draw X matrix
    ax.add_patch(plt.Rectangle((pos_X[0] - 0.1, pos_X[1] - 0.3), 0.2, 0.6, 
                               fill=True, alpha=0.3, color='blue'))
    ax.text(pos_X[0], pos_X[1], "X", ha='center', va='center', fontsize=14)
    ax.text(pos_X[0], pos_X[1] - 0.4, f"({X_rows}×{X_cols})", ha='center', va='center', fontsize=10)
    
    # Draw w vector
    ax.add_patch(plt.Rectangle((pos_w[0] - 0.05, pos_w[1] - 0.2), 0.1, 0.4, 
                               fill=True, alpha=0.3, color='green'))
    ax.text(pos_w[0], pos_w[1], "w", ha='center', va='center', fontsize=14)
    ax.text(pos_w[0], pos_w[1] - 0.3, f"({w_rows}×1)", ha='center', va='center', fontsize=10)
    
    # Draw equals sign
    ax.text(0.65, 0.5, "=", ha='center', va='center', fontsize=16)
    
    # Draw y vector
    ax.add_patch(plt.Rectangle((pos_y[0] - 0.05, pos_y[1] - 0.2), 0.1, 0.4, 
                               fill=True, alpha=0.3, color='red'))
    ax.text(pos_y[0], pos_y[1], "y", ha='center', va='center', fontsize=14)
    ax.text(pos_y[0], pos_y[1] - 0.3, f"({y_rows}×1)", ha='center', va='center', fontsize=10)
    
    # Add operation symbol
    ax.text(0.35, 0.5, "×", ha='center', va='center', fontsize=16)
    
    # Set plot limits and remove axis ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title("Matrix Form of Multiple Linear Regression: y = Xw", fontsize=14)
    
    plt.savefig(os.path.join(save_dir, "matrix_form.png"), dpi=300)
    plt.close()
    
    # Write out the expanded form of the equation
    print("In expanded form, the model is:")
    print("y_i = w_0 + w_1*x_i1 + w_2*x_i2 + w_3*x_i3 + ε_i for each observation i")
    print()

express_matrix_form()

# Step 3: Calculate X^T X and X^T y
def calculate_normal_equation_components():
    """Calculate X^T X and X^T y components of the normal equation."""
    print("Step 3: Calculating X^T X and X^T y components of the normal equation")
    
    # Calculate X^T (transpose of X)
    X_T = X.T
    print("X^T (transpose of X):")
    print(X_T)
    print(f"Dimensions of X^T: {X_T.shape}")
    print()
    
    # Calculate X^T X
    X_T_X = X_T @ X
    print("X^T X:")
    print(X_T_X)
    print(f"Dimensions of X^T X: {X_T_X.shape}")
    print()
    
    # Calculate X^T y
    X_T_y = X_T @ y
    print("X^T y:")
    print(X_T_y)
    print(f"Dimensions of X^T y: {X_T_y.shape}")
    print()
    
    # Visualize X^T X as a heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(X_T_X, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("X^T X Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "X_transpose_X.png"), dpi=300)
    plt.close()
    
    # Visualize X^T y
    plt.figure(figsize=(4, 6))
    plt.barh(range(len(X_T_y)), X_T_y, color='coral')
    plt.yticks(range(len(X_T_y)), ['w₀', 'w₁', 'w₂', 'w₃'])
    plt.xlabel('Value')
    plt.title('X^T y Vector')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "X_transpose_y.png"), dpi=300)
    plt.close()
    
    # Detailed explanation of the X^T X calculation (for the first element)
    print("Detailed calculation of X^T X[0,0] (first element):")
    print(f"X^T X[0,0] = X^T[0,:] · X[:,0] = ", end="")
    for i in range(len(X)):
        print(f"{X_T[0,i]} × {X[i,0]}", end="")
        if i < len(X) - 1:
            print(" + ", end="")
    print(f" = {X_T_X[0,0]}")
    print()
    
    # Detailed explanation of the X^T y calculation (for the first element)
    print("Detailed calculation of X^T y[0] (first element):")
    print(f"X^T y[0] = X^T[0,:] · y = ", end="")
    for i in range(len(y)):
        print(f"{X_T[0,i]} × {y[i]}", end="")
        if i < len(y) - 1:
            print(" + ", end="")
    print(f" = {X_T_y[0]}")
    print()
    
    return X_T, X_T_X, X_T_y

X_T, X_T_X, X_T_y = calculate_normal_equation_components()

# Step 4: Solve the normal equation to find optimal weights
def solve_normal_equation():
    """Solve the normal equation to find the optimal weights w."""
    print("Step 4: Solving the normal equation to find optimal weights w")
    print("The normal equation: w = (X^T X)^(-1) X^T y")
    
    # Calculate (X^T X)^(-1)
    X_T_X_inv = np.linalg.inv(X_T_X)
    print("(X^T X)^(-1):")
    print(X_T_X_inv)
    print(f"Dimensions of (X^T X)^(-1): {X_T_X_inv.shape}")
    print()
    
    # Calculate w = (X^T X)^(-1) X^T y
    w = X_T_X_inv @ X_T_y
    print("Optimal weights w = (X^T X)^(-1) X^T y:")
    print(w)
    print()
    
    # Verify the calculation by checking if X^T X w = X^T y
    XTX_w = X_T_X @ w
    print("Verification - X^T X w:")
    print(XTX_w)
    print("Should be approximately equal to X^T y:")
    print(X_T_y)
    print("Difference (should be very close to zero):")
    print(XTX_w - X_T_y)
    print()
    
    # Visualize optimal weights
    plt.figure(figsize=(6, 5))
    bars = plt.bar(range(len(w)), w, color='purple')
    plt.xticks(range(len(w)), ['w₀', 'w₁', 'w₂', 'w₃'])
    plt.ylabel('Weight Value')
    plt.title('Optimal Weights (w)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimal_weights.png"), dpi=300)
    plt.close()
    
    # Write the regression equation
    print("Final regression equation:")
    print(f"y = {w[0]:.1f} + {w[1]:.1f}*x₁ + {w[2]:.1f}*x₂ + {w[3]:.1f}*x₃")
    print()
    
    return w

w = solve_normal_equation()

# Step 5: Evaluate the model and visualize predictions
def evaluate_model(w):
    """Evaluate the model by comparing predictions with actual values."""
    print("Step 5: Evaluating the model")
    
    # Calculate predictions
    y_pred = X @ w
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate Mean Squared Error
    mse = np.mean(residuals**2)
    
    # Create a table with actual values, predictions, and residuals
    results = pd.DataFrame({
        'Observation': [f'Obs {i+1}' for i in range(len(y))],
        'Actual y': y,
        'Predicted y': y_pred,
        'Residual': residuals
    })
    
    print("Model evaluation:")
    print(results)
    print(f"\nMean Squared Error: {mse:.6f}")
    print()
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(8, 6))
    
    # Bar chart comparing actual and predicted values
    bar_width = 0.35
    index = np.arange(len(y))
    
    bar1 = plt.bar(index, y, bar_width, label='Actual y', color='blue', alpha=0.7)
    bar2 = plt.bar(index + bar_width, y_pred, bar_width, label='Predicted y', color='red', alpha=0.7)
    
    # Add value labels on top of bars
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
    
    plt.xlabel('Observation')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.xticks(index + bar_width/2, [f'Obs {i+1}' for i in range(len(y))])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # Visualize residuals
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(residuals)), residuals, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(range(len(residuals)), [f'Obs {i+1}' for i in range(len(residuals))])
    plt.ylabel('Residual (y - ŷ)')
    plt.title('Residuals')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals.png"), dpi=300)
    plt.close()
    
    # Create a 3D visualization for multiple regression (using only x1 and x2 for visualization)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for the visualization
    x1_range = np.linspace(min(x1) - 0.5, max(x1) + 0.5, 20)
    x2_range = np.linspace(min(x2) - 0.5, max(x2) + 0.5, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # For simplicity, set x3 to its mean value
    x3_mean = np.mean(x3)
    
    # Calculate predicted values for the grid (fixing x3 at its mean)
    Z = w[0] + w[1] * X1 + w[2] * X2 + w[3] * x3_mean
    
    # Plot the regression plane
    surf = ax.plot_surface(X1, X2, Z, alpha=0.5, cmap='viridis', edgecolor='none')
    
    # Plot the original data points
    scatter = ax.scatter(x1, x2, y, c='red', s=100, marker='o', label='Data Points')
    
    # Draw lines from points to plane to show residuals
    for i in range(len(x1)):
        pred_z = w[0] + w[1] * x1[i] + w[2] * x2[i] + w[3] * x3_mean
        ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [y[i], pred_z], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('y')
    ax.set_title('Multiple Linear Regression Visualization\n(x₃ fixed at mean value)')
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3D_visualization.png"), dpi=300)
    plt.close()

evaluate_model(w)

# Step 6: Discuss the matrix dimensions and provide a comprehensive visual
def visualize_matrix_operations():
    """Create visualizations explaining matrix dimensions and operations."""
    print("Step 6: Visualizing matrix dimensions and operations")
    
    n, d = X.shape  # n observations, d features (including intercept)
    
    # Matrix dimensions explanation
    print(f"The design matrix X has dimensions {n} × {d} (rows × columns).")
    print(f"X^T has dimensions {d} × {n}.")
    print(f"X^T X has dimensions {d} × {d}.")
    print(f"X^T y has dimensions {d} × 1.")
    print(f"The weight vector w has dimensions {d} × 1.")
    print()
    
    # Create a comprehensive visualization of all matrix operations
    plt.figure(figsize=(12, 8))
    
    # Create a grid layout
    gs = GridSpec(2, 4, width_ratios=[3, 1, 3, 2])
    
    # Top row: X^T * X = X^T X
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(X_T, cmap='Blues', aspect='auto')
    ax1.set_title('X^T ({} × {})'.format(d, n))
    for i in range(d):
        for j in range(n):
            ax1.text(j, i, f"{X_T[i, j]:.1f}", ha="center", va="center", color="black")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(d))
    ax1.set_xticklabels([f'Obs {i+1}' for i in range(n)])
    ax1.set_yticklabels(['w₀', 'w₁', 'w₂', 'w₃'])
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.text(0.5, 0.5, '×', fontsize=24, ha='center', va='center')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(X, cmap='Blues', aspect='auto')
    ax3.set_title('X ({} × {})'.format(n, d))
    for i in range(n):
        for j in range(d):
            ax3.text(j, i, f"{X[i, j]:.1f}", ha="center", va="center", color="black")
    ax3.set_xticks(range(d))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels(['w₀', 'w₁', 'w₂', 'w₃'])
    ax3.set_yticklabels([f'Obs {i+1}' for i in range(n)])
    
    ax4 = plt.subplot(gs[0, 3])
    ax4.text(0.5, 0.5, '=', fontsize=24, ha='center', va='center')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_frame_on(False)
    
    # Split the matrix equation across two rows
    ax5 = plt.subplot(gs[1, 0:2])
    ax5.imshow(X_T_X, cmap='Reds', aspect='auto')
    ax5.set_title('X^T X ({} × {})'.format(d, d))
    for i in range(d):
        for j in range(d):
            ax5.text(j, i, f"{X_T_X[i, j]:.1f}", ha="center", va="center", color="black")
    ax5.set_xticks(range(d))
    ax5.set_yticks(range(d))
    ax5.set_xticklabels(['w₀', 'w₁', 'w₂', 'w₃'])
    ax5.set_yticklabels(['w₀', 'w₁', 'w₂', 'w₃'])
    
    # Show the X^T y calculation
    ax6 = plt.subplot(gs[1, 2])
    X_T_y_reshaped = X_T_y.reshape(-1, 1)
    ax6.imshow(X_T_y_reshaped, cmap='Greens', aspect='auto')
    ax6.set_title('X^T y ({} × 1)'.format(d))
    for i in range(d):
        ax6.text(0, i, f"{X_T_y[i]:.1f}", ha="center", va="center", color="black")
    ax6.set_xticks([])
    ax6.set_yticks(range(d))
    ax6.set_yticklabels(['w₀', 'w₁', 'w₂', 'w₃'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "matrix_operations.png"), dpi=300)
    plt.close()
    
    # Create a visual for the complete normal equation
    plt.figure(figsize=(14, 4))
    
    # Define positions
    pos_inv = [0.2, 0.5]  # Position for (X^T X)^-1
    pos_XTy = [0.4, 0.5]  # Position for X^T y
    pos_w = [0.7, 0.5]    # Position for w
    
    ax = plt.gca()
    
    # Draw (X^T X)^-1 matrix
    ax.add_patch(plt.Rectangle((pos_inv[0] - 0.15, pos_inv[1] - 0.2), 0.3, 0.4, 
                               fill=True, alpha=0.3, color='orange'))
    ax.text(pos_inv[0], pos_inv[1], "(X^T X)^-1", ha='center', va='center', fontsize=12)
    ax.text(pos_inv[0], pos_inv[1] - 0.3, f"({d}×{d})", ha='center', va='center', fontsize=10)
    
    # Draw operation symbol
    ax.text(0.3, 0.5, "×", ha='center', va='center', fontsize=16)
    
    # Draw X^T y vector
    ax.add_patch(plt.Rectangle((pos_XTy[0] - 0.05, pos_XTy[1] - 0.2), 0.1, 0.4, 
                               fill=True, alpha=0.3, color='green'))
    ax.text(pos_XTy[0], pos_XTy[1], "X^T y", ha='center', va='center', fontsize=12)
    ax.text(pos_XTy[0], pos_XTy[1] - 0.3, f"({d}×1)", ha='center', va='center', fontsize=10)
    
    # Draw equals sign
    ax.text(0.55, 0.5, "=", ha='center', va='center', fontsize=16)
    
    # Draw w vector
    ax.add_patch(plt.Rectangle((pos_w[0] - 0.05, pos_w[1] - 0.2), 0.1, 0.4, 
                               fill=True, alpha=0.3, color='purple'))
    ax.text(pos_w[0], pos_w[1], "w", ha='center', va='center', fontsize=12)
    ax.text(pos_w[0], pos_w[1] - 0.3, f"({d}×1)", ha='center', va='center', fontsize=10)
    
    # Add formula at the bottom
    ax.text(0.5, 0.1, "Normal Equation: w = (X^T X)^-1 X^T y", ha='center', va='center', fontsize=14)
    
    # Add values
    w_str = ", ".join([f"{val:.1f}" for val in w])
    ax.text(pos_w[0], pos_w[1] + 0.3, f"[{w_str}]", ha='center', va='center', fontsize=10)
    
    # Set plot limits and remove axis ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title("The Normal Equation for Multiple Linear Regression", fontsize=14)
    
    plt.savefig(os.path.join(save_dir, "normal_equation.png"), dpi=300)
    plt.close()

visualize_matrix_operations()

# Summary of the solution
print("\nQuestion 1 Solution Summary:")
print("1. Design matrix X is a 4×4 matrix with a column of ones for the intercept and columns for each feature.")
print("2. Multiple linear regression model in matrix form: y = Xw + ε")
print("3. Normal equation for optimal weights: w = (X^T X)^(-1) X^T y")
print("4. X^T X is a 4×4 matrix and X^T y is a vector of length 4 (4×1)")
print("5. The optimal weights are:", w)
print("6. The final regression equation is:")
print(f"   y = {w[0]:.1f} + {w[1]:.1f}*x₁ + {w[2]:.1f}*x₂ + {w[3]:.1f}*x₃")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- design_matrix.png: Visualization of the design matrix and target vector")
print("- matrix_form.png: Visual representation of matrix form of regression")
print("- X_transpose_X.png: Heatmap visualization of X^T X")
print("- X_transpose_y.png: Bar chart of X^T y")
print("- optimal_weights.png: Bar chart of optimal weight values")
print("- actual_vs_predicted.png: Comparison of actual vs predicted values")
print("- residuals.png: Bar chart of residuals")
print("- 3D_visualization.png: 3D visualization of the regression model")
print("- matrix_operations.png: Comprehensive visualization of matrix operations")
print("- normal_equation.png: Visual representation of the normal equation") 