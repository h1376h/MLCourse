import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import scipy.linalg as linalg

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Section 1: Design Matrix and Intercept Term
def explain_design_matrix():
    """
    Explain the design matrix and how to incorporate the intercept term.
    """
    print("Step 1: Design Matrix and Intercept Term")
    print("----------------------------------------")
    
    # Create example data
    np.random.seed(42)
    n = 5  # number of observations
    d = 2  # number of features (excluding intercept)
    
    # Generate random features
    X_features = np.random.rand(n, d) * 10
    
    print(f"Example data with {n} observations and {d} features (excluding intercept):")
    for i in range(n):
        print(f"Observation {i+1}: x₁ = {X_features[i, 0]:.2f}, x₂ = {X_features[i, 1]:.2f}")
    
    # Show the feature matrix without intercept
    print("\nFeature matrix (without intercept):")
    print(X_features)
    
    # Add column of ones for intercept
    X = np.column_stack((np.ones(n), X_features))
    
    print("\nDesign matrix (with intercept column):")
    print(X)
    
    print("\nExplanation:")
    print("The design matrix X has dimensions n×(d+1), where:")
    print("- n is the number of observations")
    print("- d is the number of features (excluding intercept)")
    print("- The first column is all ones to account for the intercept w₀")
    print("- The remaining columns contain the feature values")
    
    print("\nMathematically, the design matrix is:")
    print("       ⎡ 1  x₁₁  x₁₂  ...  x₁ₚ ⎤")
    print("       ⎢ 1  x₂₁  x₂₂  ...  x₂ₚ ⎥")
    print("X =    ⎢ 1  x₃₁  x₃₂  ...  x₃ₚ ⎥")
    print("       ⎢ ⋮   ⋮    ⋮    ⋱    ⋮  ⎥")
    print("       ⎣ 1  xₙ₁  xₙ₂  ...  xₙₚ ⎦")
    
    print("\nWhere xᵢⱼ represents the value of feature j for observation i.")
    
    return X

# Section 2: Prediction in Matrix Form
def explain_matrix_prediction(X):
    """
    Explain the prediction of the model in matrix form.
    """
    print("\nStep 2: Prediction in Matrix Form")
    print("--------------------------------")
    
    n, d_plus_1 = X.shape
    d = d_plus_1 - 1
    
    # Generate random weights for demonstration
    np.random.seed(42)
    w = np.random.randn(d_plus_1)
    
    print(f"Weight vector (w) with {d_plus_1} elements (including intercept):")
    print(f"w₀ (intercept) = {w[0]:.4f}")
    for i in range(1, d_plus_1):
        print(f"w{i} = {w[i]:.4f}")
    
    # Calculate predictions
    y_pred = X @ w  # matrix multiplication
    
    print("\nPredictions (ŷ) using matrix multiplication X @ w:")
    for i in range(n):
        print(f"ŷ{i+1} = {y_pred[i]:.4f}")
    
    # Calculate predictions manually to show the process
    print("\nCalculating predictions step by step:")
    for i in range(n):
        pred = 0
        print(f"ŷ{i+1} = ", end="")
        for j in range(d_plus_1):
            if j == 0:
                term = f"{w[j]:.4f}"
            else:
                term = f"{w[j]:.4f} × {X[i, j]:.4f}"
            
            if j < d_plus_1 - 1:
                print(term + " + ", end="")
            else:
                print(term + f" = {y_pred[i]:.4f}")
            
            pred += w[j] * X[i, j]
    
    print("\nExplanation:")
    print("The prediction ŷ for all observations can be written in matrix form as:")
    print("ŷ = X × w")
    
    print("\nWhere:")
    print("- ŷ is an n×1 vector of predictions")
    print("- X is the n×(d+1) design matrix")
    print("- w is a (d+1)×1 vector of weights")
    
    print("\nMathematically:")
    print("       ⎡ 1  x₁₁  x₁₂  ...  x₁ₚ ⎤   ⎡ w₀ ⎤   ⎡ ŷ₁ ⎤")
    print("       ⎢ 1  x₂₁  x₂₂  ...  x₂ₚ ⎥   ⎢ w₁ ⎥   ⎢ ŷ₂ ⎥")
    print("ŷ =    ⎢ 1  x₃₁  x₃₂  ...  x₃ₚ ⎥ × ⎢ w₂ ⎥ = ⎢ ŷ₃ ⎥")
    print("       ⎢ ⋮   ⋮    ⋮    ⋱    ⋮  ⎥   ⎢ ⋮  ⎥   ⎢ ⋮  ⎥")
    print("       ⎣ 1  xₙ₁  xₙ₂  ...  xₙₚ ⎦   ⎣ wₚ ⎦   ⎣ ŷₙ ⎦")
    
    print("\nFor each observation i, the prediction is:")
    print("ŷᵢ = w₀ + w₁x_{i1} + w₂x_{i2} + ... + wₚx_{ip}")
    
    return w, y_pred

# Section 3: Cost Function in Matrix Form
def explain_cost_function(X, w, y_pred):
    """
    Explain the cost function in matrix form.
    """
    print("\nStep 3: Cost Function in Matrix Form")
    print("----------------------------------")
    
    n = X.shape[0]
    
    # Generate some target values y
    np.random.seed(123)
    y = y_pred + np.random.normal(0, 1, n) # add some noise to y_pred
    
    print(f"Target values (y) for {n} observations:")
    for i in range(n):
        print(f"y{i+1} = {y[i]:.4f}")
    
    # Calculate residuals
    residuals = y - y_pred
    
    print("\nResiduals (y - ŷ):")
    for i in range(n):
        print(f"y{i+1} - ŷ{i+1} = {y[i]:.4f} - {y_pred[i]:.4f} = {residuals[i]:.4f}")
    
    # Calculate cost function
    cost = np.sum(residuals**2) / n
    
    print(f"\nCost function (average squared error): {cost:.4f}")
    print("Calculated as (1/n) * sum((y - ŷ)²)")
    
    # Calculate cost function in matrix form
    residuals_matrix = residuals.reshape(-1, 1)
    cost_matrix = (residuals_matrix.T @ residuals_matrix)[0, 0] / n
    
    print(f"\nCost function in matrix form: {cost_matrix:.4f}")
    print("Calculated as (1/n) * (y - Xw)ᵀ(y - Xw)")
    
    print("\nExplanation:")
    print("The cost function (sum of squared errors) in matrix form is:")
    print("J(w) = (1/n) * (y - Xw)ᵀ(y - Xw)")
    
    print("\nWhere:")
    print("- y is the n×1 vector of target values")
    print("- X is the n×(d+1) design matrix")
    print("- w is the (d+1)×1 vector of weights")
    print("- (y - Xw) represents the residuals")
    
    # Create visualization for the cost function
    fig = plt.figure(figsize=(10, 6))
    
    # Create 3D surface for a simple 2D case (one feature + intercept)
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a range of weight values
    w0_range = np.linspace(w[0] - 2, w[0] + 2, 30)
    w1_range = np.linspace(w[1] - 2, w[1] + 2, 30)
    w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
    
    # Calculate cost for each weight combination
    cost_grid = np.zeros_like(w0_grid)
    X_simple = X[:, :2]  # Use just intercept and first feature
    
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            w_temp = np.array([w0_grid[i, j], w1_grid[i, j]])
            y_pred_temp = X_simple @ w_temp
            residuals_temp = y - y_pred_temp
            cost_grid[i, j] = np.mean(residuals_temp**2)
    
    # Plot the surface
    surf = ax.plot_surface(w0_grid, w1_grid, cost_grid, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True, alpha=0.7)
    
    # Mark the minimum point
    min_cost_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    min_w0 = w0_grid[min_cost_idx]
    min_w1 = w1_grid[min_cost_idx]
    min_cost = cost_grid[min_cost_idx]
    
    ax.scatter([min_w0], [min_w1], [min_cost], color='black', s=100, label='Minimum')
    
    # Add a title and labels
    ax.set_title("Cost Function Surface (Simplified to 2D)", fontsize=14)
    ax.set_xlabel('w₀ (Intercept)')
    ax.set_ylabel('w₁')
    ax.set_zlabel('Cost J(w)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cost_function.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return y, residuals, cost

# Section 4: Gradient of Cost Function
def explain_gradient(X, w, y):
    """
    Explain the gradient of the cost function.
    """
    print("\nStep 4: Gradient of the Cost Function")
    print("------------------------------------")
    
    n, d_plus_1 = X.shape
    
    # Calculate predictions
    y_pred = X @ w
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate gradient
    gradient = (-2/n) * (X.T @ residuals)
    
    print("The gradient of the cost function with respect to w is:")
    print("∇J(w) = (-2/n) * X^T (y - Xw)")
    
    print("\nFor our example:")
    for i in range(len(gradient)):
        if i == 0:
            print(f"∂J/∂w₀ = {gradient[i]:.4f}")
        else:
            print(f"∂J/∂w{i} = {gradient[i]:.4f}")
    
    print("\nExplanation of the derivation:")
    print("Starting with the cost function: J(w) = (1/n) * (y - Xw)ᵀ(y - Xw)")
    print("Expanding: J(w) = (1/n) * (yᵀy - yᵀXw - wᵀXᵀy + wᵀXᵀXw)")
    print("Simplifying (note that yᵀXw is a scalar, so yᵀXw = (yᵀXw)ᵀ = wᵀXᵀy):")
    print("J(w) = (1/n) * (yᵀy - 2wᵀXᵀy + wᵀXᵀXw)")
    
    print("\nTaking the gradient with respect to w:")
    print("∇J(w) = (1/n) * (-2Xᵀy + 2XᵀXw) = (2/n) * (XᵀXw - Xᵀy) = (-2/n) * Xᵀ(y - Xw)")
    
    # Create a visualization of the gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate a range of weight values (for a simplified 2D case)
    w0_vals = np.linspace(w[0] - 2, w[0] + 2, 20)
    w1_vals = np.linspace(w[1] - 2, w[1] + 2, 20)
    w0_grid, w1_grid = np.meshgrid(w0_vals, w1_vals)
    
    # Calculate cost for each weight combination
    cost_grid = np.zeros_like(w0_grid)
    grad_w0 = np.zeros_like(w0_grid)
    grad_w1 = np.zeros_like(w1_grid)
    
    X_simple = X[:, :2]  # Use just intercept and first feature
    
    for i in range(len(w0_vals)):
        for j in range(len(w1_vals)):
            w_temp = np.array([w0_grid[i, j], w1_grid[i, j]])
            y_pred_temp = X_simple @ w_temp
            residuals_temp = y - y_pred_temp
            cost_grid[i, j] = np.mean(residuals_temp**2)
            
            # Calculate gradient
            grad_temp = (-2/n) * (X_simple.T @ residuals_temp)
            grad_w0[i, j] = grad_temp[0]
            grad_w1[i, j] = grad_temp[1]
    
    # Create a contour plot of the cost function
    contour = ax.contourf(w0_grid, w1_grid, cost_grid, levels=50, cmap='viridis', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Cost J(w)')
    
    # Plot the gradient vectors
    # Subsample for clarity
    step = 4
    ax.quiver(w0_grid[::step, ::step], w1_grid[::step, ::step], 
              -grad_w0[::step, ::step], -grad_w1[::step, ::step], 
              color='white', scale=50, alpha=0.8)
    
    # Mark the optimal point
    min_cost_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    min_w0 = w0_grid[min_cost_idx]
    min_w1 = w1_grid[min_cost_idx]
    ax.scatter([min_w0], [min_w1], color='red', s=100, label='Minimum')
    
    # Mark the current point
    ax.scatter([w[0]], [w[1]], color='black', s=100, label='Current w')
    
    ax.set_xlabel('w₀ (Intercept)')
    ax.set_ylabel('w₁')
    ax.set_title('Gradient of Cost Function', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return gradient

# Section 5: Normal Equations and Closed-Form Solution
def explain_normal_equations(X, y):
    """
    Explain the normal equations and the closed-form solution.
    """
    print("\nStep 5: Normal Equations and Closed-Form Solution")
    print("----------------------------------------------")
    
    print("To find the optimal weights that minimize the cost function,")
    print("we set the gradient equal to zero and solve for w:")
    
    print("\n∇J(w) = (-2/n) * Xᵀ(y - Xw) = 0")
    print("Xᵀ(y - Xw) = 0")
    print("Xᵀy - XᵀXw = 0")
    print("XᵀXw = Xᵀy")
    
    print("\nThese are the normal equations. Solving for w:")
    print("w = (XᵀX)⁻¹Xᵀy")
    
    # Calculate the closed-form solution
    XTX = X.T @ X
    XTy = X.T @ y
    
    print("\nIn our example:")
    print("XᵀX =")
    print(XTX)
    
    print("\nXᵀy =")
    print(XTy)
    
    # Check if XᵀX is invertible
    try:
        XTX_inv = np.linalg.inv(XTX)
        print("\n(XᵀX)⁻¹ =")
        print(XTX_inv)
        
        w_optimal = XTX_inv @ XTy
        
        print("\nOptimal weights:")
        for i, wi in enumerate(w_optimal):
            if i == 0:
                print(f"w₀ = {wi:.4f}")
            else:
                print(f"w{i} = {wi:.4f}")
        
    except np.linalg.LinAlgError:
        print("\nXᵀX is not invertible. This can happen when:")
        print("1. There are more features than observations (d+1 > n)")
        print("2. The features are linearly dependent")
        print("3. There's perfect multicollinearity")
        
        print("\nIn such cases, we can use:")
        print("1. Regularization (like Ridge Regression)")
        print("2. Pseudo-inverse using SVD")
        print("3. Dimensionality reduction")
        
        # Use pseudoinverse as an alternative
        w_optimal = np.linalg.pinv(X) @ y
        
        print("\nOptimal weights using pseudoinverse:")
        for i, wi in enumerate(w_optimal):
            if i == 0:
                print(f"w₀ = {wi:.4f}")
            else:
                print(f"w{i} = {wi:.4f}")
    
    # Verify the solution
    y_pred_optimal = X @ w_optimal
    residuals_optimal = y - y_pred_optimal
    cost_optimal = np.mean(residuals_optimal**2)
    
    print(f"\nOptimal cost: {cost_optimal:.4f}")
    
    # Calculate gradient at the optimal point
    gradient_optimal = (-2/len(y)) * (X.T @ residuals_optimal)
    
    print("\nGradient at the optimal point:")
    for i, grad in enumerate(gradient_optimal):
        if i == 0:
            print(f"∂J/∂w₀ = {grad:.8f}")
        else:
            print(f"∂J/∂w{i} = {grad:.8f}")
    
    print("\nNote that the gradient is very close to zero (within numerical precision),")
    print("confirming that we've found the minimum of the cost function.")
    
    # Comparison of solutions with different numbers of observations and features
    print("\nComparison of solutions with different data dimensions:")
    print("------------------------------------------------------")
    
    scenarios = [
        {"n": 10, "d": 2, "name": "More observations than features (n > d+1)"},
        {"n": 3, "d": 2, "name": "Equal observations and parameters (n = d+1)"},
        {"n": 2, "d": 2, "name": "Fewer observations than features (n < d+1)"}
    ]
    
    for scenario in scenarios:
        n = scenario["n"]
        d = scenario["d"]
        name = scenario["name"]
        
        # Generate data
        np.random.seed(42)
        X_features = np.random.rand(n, d) * 10
        X_scenario = np.column_stack((np.ones(n), X_features))
        
        # Generate target values
        true_w = np.random.randn(d+1)
        y_true = X_scenario @ true_w
        y_scenario = y_true + np.random.normal(0, 1, n)
        
        # Calculate closed-form solution
        try:
            XTX = X_scenario.T @ X_scenario
            XTy = X_scenario.T @ y_scenario
            w_solution = np.linalg.inv(XTX) @ XTy
            method = "Standard inverse"
        except np.linalg.LinAlgError:
            w_solution = np.linalg.pinv(X_scenario) @ y_scenario
            method = "Pseudoinverse"
        
        # Calculate cost
        y_pred = X_scenario @ w_solution
        cost = np.mean((y_scenario - y_pred)**2)
        
        print(f"\n{name}:")
        print(f"- n = {n}, d = {d} (d+1 = {d+1})")
        print(f"- Solution method: {method}")
        print(f"- Cost: {cost:.4f}")
    
    # Create a visualization of the normal equations
    fig = plt.figure(figsize=(12, 8))
    
    # Generate simple data for visualization (2D case)
    np.random.seed(42)
    n_viz = 30
    X_viz1 = np.random.rand(n_viz) * 10
    X_viz = np.column_stack((np.ones(n_viz), X_viz1))
    
    # True relationship with some noise
    true_w = np.array([2, 0.5])
    y_viz = X_viz @ true_w + np.random.normal(0, 1, n_viz)
    
    # Calculate the optimal weights
    w_viz = np.linalg.inv(X_viz.T @ X_viz) @ (X_viz.T @ y_viz)
    
    # Create the plot
    plt.scatter(X_viz[:, 1], y_viz, color='blue', label='Data points')
    
    # Plot the true relationship
    x_line = np.linspace(0, 10, 100)
    y_true_line = true_w[0] + true_w[1] * x_line
    plt.plot(x_line, y_true_line, 'g-', label=f'True relationship: y = {true_w[0]:.2f} + {true_w[1]:.2f}x')
    
    # Plot the estimated relationship
    y_est_line = w_viz[0] + w_viz[1] * x_line
    plt.plot(x_line, y_est_line, 'r-', label=f'Estimated: y = {w_viz[0]:.2f} + {w_viz[1]:.2f}x')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression: Closed-Form Solution', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normal_equations.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visualization showing multiple features
    if X.shape[1] >= 3:  # We need at least 2 features + intercept
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract the features and target
        x1 = X[:, 1]
        x2 = X[:, 2]
        
        # Create a mesh grid for the plane
        x1_range = np.linspace(min(x1) - 1, max(x1) + 1, 20)
        x2_range = np.linspace(min(x2) - 1, max(x2) + 1, 20)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        # Calculate predicted values for the grid
        X_grid = np.column_stack((np.ones(len(x1_grid.flatten())), 
                                 x1_grid.flatten(), 
                                 x2_grid.flatten()))
        y_grid = X_grid @ w_optimal
        y_grid = y_grid.reshape(x1_grid.shape)
        
        # Plot the 3D scatter points
        ax.scatter(x1, x2, y, color='blue', label='Data points')
        
        # Plot the fitted plane
        surf = ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, cmap=cm.coolwarm)
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('y')
        ax.set_title('Multiple Linear Regression: Fitted Plane', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "multiple_regression.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return w_optimal, cost_optimal

# Main function to run all steps
def main():
    print("==================================================")
    print("  Multiple Linear Regression with Matrix Operations")
    print("==================================================\n")
    
    # Step 1: Design Matrix and Intercept Term
    X = explain_design_matrix()
    
    # Step 2: Prediction in Matrix Form
    w, y_pred = explain_matrix_prediction(X)
    
    # Step 3: Cost Function in Matrix Form
    y, residuals, cost = explain_cost_function(X, w, y_pred)
    
    # Step 4: Gradient of Cost Function
    gradient = explain_gradient(X, w, y)
    
    # Step 5: Normal Equations and Closed-Form Solution
    w_optimal, cost_optimal = explain_normal_equations(X, y)
    
    print("\n==================================================")
    print("  Summary of Results")
    print("==================================================")
    
    print(f"\nInitial cost with random weights: {cost:.4f}")
    print(f"Optimal cost after solving normal equations: {cost_optimal:.4f}")
    
    print("\nImprovement: {:.2f}%".format((cost - cost_optimal) / cost * 100))
    
    print("\nImages have been saved to:", save_dir)
    
    return {
        "design_matrix": os.path.join(save_dir, "design_matrix.png"),
        "matrix_prediction": os.path.join(save_dir, "matrix_prediction.png"),
        "cost_function": os.path.join(save_dir, "cost_function.png"),
        "gradient": os.path.join(save_dir, "gradient.png"),
        "normal_equations": os.path.join(save_dir, "normal_equations.png"),
        "multiple_regression": os.path.join(save_dir, "multiple_regression.png")
    }

if __name__ == "__main__":
    image_paths = main() 