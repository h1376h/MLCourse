import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Use LaTeX rendering for matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Generate some example data for the problem
np.random.seed(42)  # For reproducibility

# Define the problem as stated
# X is a 5x3 matrix (including a column of ones for the bias term)
# We'll generate random data but ensure it's consistent with the problem description
X = np.ones((5, 3))
X[:, 1:] = np.random.randn(5, 2)  # Random features

# True parameters (unknown in practice, but we'll use them to generate y)
w_true = np.array([2, -1, 0.5])

# Generate target values with some noise
y = X @ w_true + np.random.randn(5) * 0.5

# Initial parameter vector
w_init = np.zeros(3)

# Learning rate
alpha = 0.1

print("Problem Setup:")
print("==============")
print("Feature matrix X (5x3):")
print(X)
print("\nTarget vector y:")
print(y)
print("\nInitial parameter vector w^(0):")
print(w_init)
print("\nLearning rate α:", alpha)
print("\n")

print("Mathematical Derivation of the Gradient:")
print("======================================")
print("For linear regression with cost function J(w) = ||y - Xw||^2:")
print("1. Expand the squared L2 norm: J(w) = (y - Xw)^T(y - Xw)")
print("2. Expand the product: J(w) = y^Ty - y^TXw - w^TX^Ty + w^TX^TXw")
print("3. Note that y^TXw is a scalar, so y^TXw = (y^TXw)^T = w^TX^Ty")
print("4. Therefore, J(w) = y^Ty - 2w^TX^Ty + w^TX^TXw")
print("5. Now compute the gradient by taking the partial derivative with respect to w:")
print("   ∇J(w) = -2X^Ty + 2X^TXw = 2X^T(Xw - y)")
print("6. Including the scaling factor for m examples: ∇J(w) = (2/m)X^T(Xw - y)")
print("7. This gives us our final gradient formula.\n")

# Define the cost function
def compute_cost(X, y, w):
    """
    Compute the cost function J(w) = ||y - Xw||^2
    
    Parameters:
    X (numpy.ndarray): Feature matrix (m x n)
    y (numpy.ndarray): Target vector (m,)
    w (numpy.ndarray): Parameter vector (n,)
    
    Returns:
    float: The value of the cost function
    """
    m = len(y)
    predictions = X @ w
    errors = predictions - y
    cost = np.sum(errors**2) / m
    
    # Print detailed calculations for the cost
    if m <= 10:  # Only show detailed calculations for small datasets
        print("\nDetailed cost calculation:")
        print(f"Predictions Xw = X @ w = {predictions}")
        print(f"Errors = Xw - y = {errors}")
        print(f"Squared errors = {errors**2}")
        print(f"Sum of squared errors = {np.sum(errors**2)}")
        print(f"Cost = (1/{m}) * sum(errors^2) = {cost}")
    
    return cost

# Define the gradient function
def compute_gradient(X, y, w):
    """
    Compute the gradient of the cost function ∇_w J(w) = 2X^T(Xw - y)
    
    Parameters:
    X (numpy.ndarray): Feature matrix (m x n)
    y (numpy.ndarray): Target vector (m,)
    w (numpy.ndarray): Parameter vector (n,)
    
    Returns:
    numpy.ndarray: Gradient vector (n,)
    """
    m = len(y)
    predictions = X @ w
    errors = predictions - y
    grad = (2/m) * (X.T @ errors)
    
    # Print detailed calculations for the gradient
    if m <= 10 and len(w) <= 5:  # Only show detailed calculations for small problems
        print("\nDetailed gradient calculation:")
        print(f"Predictions Xw = X @ w = {predictions}")
        print(f"Errors = Xw - y = {errors}")
        print(f"X^T = \n{X.T}")
        print(f"X^T @ errors = \n{X.T @ errors}")
        print(f"Gradient = (2/{m}) * X^T @ errors = {grad}")
    
    return grad

# Implement batch gradient descent
def batch_gradient_descent(X, y, w_init, alpha, num_iterations=100, tol=1e-6):
    """
    Implement batch gradient descent algorithm for linear regression
    
    Parameters:
    X (numpy.ndarray): Feature matrix (m x n)
    y (numpy.ndarray): Target vector (m,)
    w_init (numpy.ndarray): Initial parameter vector (n,)
    alpha (float): Learning rate
    num_iterations (int): Maximum number of iterations
    tol (float): Tolerance for convergence
    
    Returns:
    numpy.ndarray: Optimal parameter vector
    list: History of parameter vectors
    list: History of cost values
    list: History of gradient norms
    """
    m = len(y)
    n = X.shape[1]
    w = w_init.copy()
    w_history = [w.copy()]
    cost_history = [compute_cost(X, y, w)]
    grad_history = []
    
    print("\nBatch Gradient Descent Algorithm:")
    print("================================")
    print("Each iteration uses all 5 training examples to compute the gradient.")
    print(f"Iteration 0: Cost = {cost_history[0]:.6f}, w = {w}")
    
    for i in range(1, num_iterations + 1):
        # Compute gradient
        grad = compute_gradient(X, y, w)
        grad_norm = np.linalg.norm(grad)
        grad_history.append(grad_norm)
        
        # Update parameters
        w_old = w.copy()
        w = w - alpha * grad
        
        # Record history
        w_history.append(w.copy())
        current_cost = compute_cost(X, y, w)
        cost_history.append(current_cost)
        
        # Calculate change in parameters and cost
        param_change = np.linalg.norm(w - w_old)
        cost_change = abs(cost_history[i] - cost_history[i-1])
        
        # Print progress with more detailed information
        if i % 10 == 0 or i < 10:
            print(f"\nIteration {i}:")
            print(f"  Gradient = {grad}")
            print(f"  Gradient Norm = {grad_norm:.6f}")
            print(f"  Parameter Update: w^({i-1}) - α∇J(w^({i-1})) = {w_old} - {alpha} * {grad} = {w}")
            print(f"  New Cost = {current_cost:.6f}")
            print(f"  Change in parameters = {param_change:.6f}")
            print(f"  Change in cost = {cost_change:.6f}")
        
        # Check convergence
        if cost_change < tol:
            print(f"\nConverged at iteration {i}")
            print(f"Final parameters: w = {w}")
            print(f"Final cost: J(w) = {current_cost:.6f}")
            break
    
    if i == num_iterations:
        print("\nReached maximum iterations")
        print(f"Final parameters: w = {w}")
        print(f"Final cost: J(w) = {cost_history[-1]:.6f}")
    
    return w, w_history, cost_history, grad_history

# Run batch gradient descent
print("\nRunning Batch Gradient Descent:")
print("===============================")
w_final, w_history, cost_history, grad_history = batch_gradient_descent(X, y, w_init, alpha, num_iterations=100)

# Calculate true cost for comparison (if we used the true parameters)
true_cost = compute_cost(X, y, w_true)
print(f"\nFor reference, true parameters: w_true = {w_true}")
print(f"Cost with true parameters: J(w_true) = {true_cost:.6f}")

# Create a summary table of the results
results_df = pd.DataFrame({
    'True Parameters': w_true,
    'Initial Parameters': w_init,
    'Final Parameters': w_final
})
results_df.index = ['w_0', 'w_1', 'w_2']
print("\nSummary of Results:")
print(results_df)

# Calculate the distance to the true parameters
param_distance = np.linalg.norm(w_final - w_true)
print(f"\nEuclidean distance between final and true parameters: {param_distance:.6f}")

# Visualize the results
print("\nGenerating Visualizations:")
print("=========================")

# 1. Plot cost function vs. iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history)), cost_history, 'b-', linewidth=2)
plt.axhline(y=true_cost, color='r', linestyle='--', label='Cost with true parameters')
plt.xlabel('Iteration')
plt.ylabel('Cost Function $J(\\mathbf{w})$')
plt.title('Cost Function vs. Iterations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_vs_iterations.png'), dpi=300)
plt.close()

# 2. Plot gradient norm vs. iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(grad_history) + 1), grad_history, 'g-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm $\\|\\nabla J(\\mathbf{w})\\|$')
plt.title('Gradient Norm vs. Iterations')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gradient_norm_vs_iterations.png'), dpi=300)
plt.close()

# 3. Visualize the parameter convergence
plt.figure(figsize=(10, 6))
w_history_array = np.array(w_history)
for i in range(3):
    param_name = '$w_0$' if i == 0 else '$w_1$' if i == 1 else '$w_2$'
    plt.plot(range(len(w_history)), w_history_array[:, i], 
             label=f'{param_name}', linewidth=2)
    plt.axhline(y=w_true[i], color=f'C{i}', linestyle='--', 
                label=f'True {param_name}')

plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter Convergence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'parameter_convergence.png'), dpi=300)
plt.close()

# 4. Create a 3D visualization of the cost function (simplified to 2 parameters)
if X.shape[1] >= 3:
    # We'll create a grid by varying w0 and w1, keeping w2 fixed at its final value
    w0_range = np.linspace(w_final[0] - 2, w_final[0] + 2, 50)
    w1_range = np.linspace(w_final[1] - 2, w_final[1] + 2, 50)
    w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
    cost_grid = np.zeros_like(w0_grid)
    
    # Calculate cost for each point in the grid
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            w_temp = np.array([w0_grid[j, i], w1_grid[j, i], w_final[2]])
            cost_grid[j, i] = compute_cost(X, y, w_temp)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(w0_grid, w1_grid, cost_grid, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    # Plot the trajectory of gradient descent
    ax.plot(w_history_array[:, 0], w_history_array[:, 1], 
            [compute_cost(X, y, np.array([w_history_array[i, 0], w_history_array[i, 1], w_history_array[i, 2]])) 
             for i in range(len(w_history))], 
            'r-', linewidth=2, label='Gradient Descent Path')
    
    # Mark the starting and ending points
    ax.scatter(w_init[0], w_init[1], compute_cost(X, y, w_init), color='r', s=100, label='Initial Point')
    ax.scatter(w_final[0], w_final[1], compute_cost(X, y, w_final), color='g', s=100, label='Final Point')
    
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.set_zlabel('Cost Function $J(\\mathbf{w})$')
    ax.set_title('Cost Function Surface and Gradient Descent Path')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cost_surface_3d.png'), dpi=300)
    plt.close()

# 5. Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contour(w0_grid, w1_grid, cost_grid, 20, cmap='viridis')
plt.clabel(contour, inline=1, fontsize=8)
plt.plot(w_history_array[:, 0], w_history_array[:, 1], 'r.-', linewidth=1, markersize=8, label='Gradient Descent Path')
plt.scatter(w_init[0], w_init[1], color='r', s=100, marker='o', label='Initial Point')
plt.scatter(w_final[0], w_final[1], color='g', s=100, marker='*', label='Final Point')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.title('Contour Plot of Cost Function and Gradient Descent Path')
plt.colorbar(contour)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'contour_plot.png'), dpi=300)
plt.close()

# 6. Create a convergence analysis plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(range(len(cost_history)), [abs(c - true_cost) for c in cost_history], 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('$|J(\\mathbf{w}^{(t)}) - J(\\mathbf{w}^{*})|$')
plt.title('Cost Difference from Optimum (log scale)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(range(1, len(grad_history) + 1), grad_history, 'g-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('$\\|\\nabla J(\\mathbf{w}^{(t)})\\|$')
plt.title('Gradient Norm (log scale)')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300)
plt.close()

# Explain the relationship to the problem tasks
print("\n\nTask Explanations:")
print("==================")

print("\n1. Gradient Descent Update Rule:")
print("The update rule for batch gradient descent in linear regression is:")
print("w^(t+1) = w^(t) - α∇J(w^(t))")
print("Where:")
print("- w^(t) is the parameter vector at iteration t")
print("- α is the learning rate (0.1 in our case)")
print("- ∇J(w^(t)) is the gradient of the cost function at w^(t)")

print("\n2. Gradient Formula for Linear Regression:")
print("The formula for computing the gradient of J(w) = ||y - Xw||^2 is:")
print("∇J(w) = 2X^T(Xw - y)")
print("Or equivalently:")
print("∇J(w) = 2/m * X^T(Xw - y)")
print("Where m is the number of training examples")

print("\n3. Number of Training Examples in Batch Gradient Descent:")
print(f"In batch gradient descent, all {len(y)} training examples are used in each iteration.")
print("This is why it's called 'batch' gradient descent - we use the entire batch of training data")
print("to compute the gradient at each step.")

print("\n4. Convergence Criteria:")
print("Common convergence criteria for gradient descent include:")
print("a) Small change in the cost function: |J(w^(t+1)) - J(w^(t))| < ε")
print("b) Small magnitude of the gradient: ||∇J(w^(t))|| < ε")
print("c) Maximum number of iterations reached")
print(f"In our implementation, we used criterion (a) with ε = {1e-6}")
print("and stopped when the change in cost between consecutive iterations was less than this threshold.")
print("We also set a maximum number of iterations as a safeguard.")

print("\nImages generated and saved to:", save_dir)
print("Generated images:")
print("- cost_vs_iterations.png: Plot of cost function vs. iterations")
print("- gradient_norm_vs_iterations.png: Plot of gradient norm vs. iterations")
print("- parameter_convergence.png: Visualization of parameter convergence")
print("- cost_surface_3d.png: 3D visualization of the cost function surface")
print("- contour_plot.png: Contour plot of the cost function with gradient descent path")
print("- convergence_analysis.png: Analysis of convergence in log scale") 