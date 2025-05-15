import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import pandas as pd
import seaborn as sns

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
    
    # Visualize the design matrix as a heatmap
    plt.figure(figsize=(8, 6))
    X_df = pd.DataFrame(X, columns=['Intercept', 'x₁', 'x₂', 'x₃'])
    sns.heatmap(X_df, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Design Matrix X")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix_heatmap.png"), dpi=300)
    plt.close()
    
    # Visualize target vector
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(y)), y, color='teal')
    plt.xticks(range(len(y)), [f'Obs {i+1}' for i in range(len(y))])
    plt.ylabel('Target Value (y)')
    plt.title('Target Vector y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_vector.png"), dpi=300)
    plt.close()
    
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

express_matrix_form()

# Step 3: Write the normal equation for finding the optimal weights w
def write_normal_equation():
    """Write the normal equation for finding the optimal weights w."""
    print("Step 3: Writing the normal equation for finding the optimal weights w")
    print("The normal equation for finding the optimal weights w is:")
    print("X^T X w = X^T y")
    print("\nSolving for w:")
    print("w = (X^T X)^(-1) X^T y")
    print()
    
    # Actually calculating X^T X and X^T y to demonstrate
    X_T = X.T
    X_T_X = X_T @ X
    X_T_y = X_T @ y
    
    # Let's actually solve for w to demonstrate
    w = np.linalg.inv(X_T_X) @ X_T_y
    
    print("For our specific problem:")
    print("X^T X =")
    print(X_T_X)
    print("\nX^T y =")
    print(X_T_y)
    print("\nOptimal weights w =")
    print(w)
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
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "X_transpose_y.png"), dpi=300)
    plt.close()
    
    # Visualize optimal weights
    plt.figure(figsize=(6, 5))
    plt.bar(range(len(w)), w, color='purple')
    plt.xticks(range(len(w)), ['w₀', 'w₁', 'w₂', 'w₃'])
    plt.ylabel('Coefficient Value')
    plt.title('Optimal Weights (w)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimal_weights.png"), dpi=300)
    plt.close()
    
    # Compare predicted vs actual values
    y_pred = X @ w
    plt.figure(figsize=(7, 5))
    bar_width = 0.35
    index = np.arange(len(y))
    plt.bar(index, y, bar_width, label='Actual y', color='blue', alpha=0.7)
    plt.bar(index + bar_width, y_pred, bar_width, label='Predicted y', color='red', alpha=0.7)
    plt.xticks(index + bar_width/2, [f'Obs {i+1}' for i in range(len(y))])
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    return X_T_X, X_T_y, w

X_T_X, X_T_y, w = write_normal_equation()

# Step 4: Describe the dimensions of X^T X and X^T y
def describe_dimensions():
    """Describe the dimensions of X^T X and X^T y."""
    print("Step 4: Describing the dimensions of X^T X and X^T y")
    
    n, p = X.shape  # n = number of observations, p = number of features including intercept
    
    print(f"The design matrix X has dimensions {n} × {p} ({n} observations, {p} features including the intercept).")
    print(f"X^T has dimensions {p} × {n}.")
    print(f"X^T X has dimensions {p} × {p}. In this case, it's a {X_T_X.shape[0]} × {X_T_X.shape[1]} matrix.")
    print(f"X^T y has dimensions {p} × 1. In this case, it's a vector of length {len(X_T_y)}.")
    print()
    
    # Visualize matrix dimensions with a diagram
    plt.figure(figsize=(10, 6))
    
    # Create a grid layout
    gs = GridSpec(2, 4, width_ratios=[3, 1, 3, 1])
    
    # Define matrices for visualization
    X_vis = np.zeros((n, p))
    X_vis.fill(1)
    XT_vis = np.zeros((p, n))
    XT_vis.fill(2)
    XTX_vis = np.zeros((p, p))
    XTX_vis.fill(3)
    XTy_vis = np.zeros((p, 1))
    XTy_vis.fill(4)
    y_vis = np.zeros((n, 1))
    y_vis.fill(5)
    
    # Top row: X^T * X = X^T X
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(XT_vis, cmap='Blues', aspect='auto')
    ax1.set_title('X^T ({} × {})'.format(p, n))
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.text(0.5, 0.5, '×', fontsize=20, ha='center', va='center')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)
    
    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(X_vis, cmap='Blues', aspect='auto')
    ax3.set_title('X ({} × {})'.format(n, p))
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    ax4 = plt.subplot(gs[0, 3])
    ax4.text(0.5, 0.5, '=', fontsize=20, ha='center', va='center')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_frame_on(False)
    
    # Bottom row: result = X^T X
    ax5 = plt.subplot(gs[1, 0:2])
    ax5.imshow(XTX_vis, cmap='Reds', aspect='auto')
    ax5.set_title('X^T X ({} × {})'.format(p, p))
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    # Add X^T * y = X^T y below
    plt.figtext(0.5, 0.05, 'Similarly, X^T ({0} × {1}) × y ({1} × 1) = X^T y ({0} × 1)'.format(p, n), 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_dir, "matrix_dimensions_visualization.png"), dpi=300)
    plt.close()

describe_dimensions()

# Summary of the solution
print("\nQuestion 1 Solution Summary:")
print("1. Design matrix X is a 4×4 matrix with a column of ones for the intercept and columns for each feature.")
print("2. Multiple linear regression model in matrix form: y = Xw + ε")
print("3. Normal equation for optimal weights: w = (X^T X)^(-1) X^T y")
print("4. X^T X is a 4×4 matrix and X^T y is a vector of length 4 (4×1)")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- design_matrix_heatmap.png: Heatmap visualization of the design matrix")
print("- target_vector.png: Bar chart of the target vector values")
print("- X_transpose_X.png: Heatmap visualization of X^T X")
print("- X_transpose_y.png: Bar chart of X^T y")
print("- optimal_weights.png: Bar chart of optimal weight values")
print("- actual_vs_predicted.png: Comparison of actual vs predicted values")
print("- matrix_dimensions_visualization.png: Visual representation of matrix dimensions and operations") 