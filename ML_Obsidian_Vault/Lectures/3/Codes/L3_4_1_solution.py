import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
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
print("- 3D_visualization.png: 3D visualization of the regression model")