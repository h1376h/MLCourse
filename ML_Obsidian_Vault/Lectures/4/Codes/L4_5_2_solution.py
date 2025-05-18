import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Step 1: Define the logistic regression objective function and sigmoid function
def sigmoid(z):
    """
    Computes the sigmoid function.
    
    Args:
        z: Input value or array
        
    Returns:
        Sigmoid of the input
    """
    return 1 / (1 + np.exp(-z))

def objective_function(w, X, y, lambda_val):
    """
    Computes the logistic regression objective function with L2 regularization.
    
    Args:
        w: Weight vector
        X: Feature matrix (each row is a sample)
        y: Target vector
        lambda_val: Regularization parameter
        
    Returns:
        Value of the objective function
    """
    m = X.shape[0]
    h = sigmoid(X @ w)  # @ is the matrix multiplication operator
    
    # Ensure numerical stability by avoiding log(0)
    epsilon = 1e-10
    h = np.clip(h, epsilon, 1 - epsilon)
    
    # Cross-entropy loss term
    cross_entropy = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    # Regularization term (we exclude the bias term w[0])
    regularization = lambda_val * np.sum(w[1:]**2)
    
    return cross_entropy + regularization

# Step 2: Define the gradient of the objective function
def gradient(w, X, y, lambda_val):
    """
    Computes the gradient of the logistic regression objective function.
    
    Args:
        w: Weight vector
        X: Feature matrix (each row is a sample)
        y: Target vector
        lambda_val: Regularization parameter
        
    Returns:
        Gradient vector of the objective function
    """
    m = X.shape[0]
    h = sigmoid(X @ w)
    
    # Compute the gradient of cross-entropy loss
    grad_cross_entropy = (1/m) * X.T @ (h - y)
    
    # Compute the gradient of regularization term (don't regularize bias)
    grad_reg = np.zeros_like(w)
    grad_reg[1:] = 2 * lambda_val * w[1:]
    
    return grad_cross_entropy + grad_reg

# Step 3: Implement batch gradient descent
def batch_gradient_descent(X, y, lambda_val, learning_rate, num_iters):
    """
    Implements batch gradient descent for logistic regression.
    
    Args:
        X: Feature matrix
        y: Target vector
        lambda_val: Regularization parameter
        learning_rate: Learning rate for gradient descent
        num_iters: Number of iterations
        
    Returns:
        w: Optimized weights
        history: Dictionary containing training history
    """
    m, n = X.shape
    w = np.zeros(n)
    
    # For tracking progress
    history = {
        'weights': [w.copy()],
        'objective': [objective_function(w, X, y, lambda_val)]
    }
    
    for i in range(num_iters):
        # Compute gradient
        grad = gradient(w, X, y, lambda_val)
        
        # Update weights using the gradient
        w = w - learning_rate * grad
        
        # Track progress
        history['weights'].append(w.copy())
        history['objective'].append(objective_function(w, X, y, lambda_val))
        
    return w, history

# Step 4: Implement stochastic gradient descent
def stochastic_gradient_descent(X, y, lambda_val, learning_rate, num_epochs):
    """
    Implements stochastic gradient descent for logistic regression.
    
    Args:
        X: Feature matrix
        y: Target vector
        lambda_val: Regularization parameter
        learning_rate: Learning rate
        num_epochs: Number of passes through the whole dataset
        
    Returns:
        w: Optimized weights
        history: Dictionary containing training history
    """
    m, n = X.shape
    w = np.zeros(n)
    
    # For tracking progress
    history = {
        'weights': [w.copy()],
        'objective': [objective_function(w, X, y, lambda_val)]
    }
    
    for epoch in range(num_epochs):
        # Shuffle the data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            xi = X_shuffled[i:i+1]  # Single sample features
            yi = y_shuffled[i:i+1]  # Single sample target
            
            # Compute gradient for the single sample
            h_i = sigmoid(xi @ w)
            grad_i = xi.T @ (h_i - yi)
            
            # Add regularization gradient (don't regularize bias)
            grad_reg = np.zeros_like(w)
            grad_reg[1:] = 2 * lambda_val * w[1:]
            
            grad = grad_i + grad_reg
            
            # Update weights
            w = w - learning_rate * grad
            
        # Track progress after each epoch
        history['weights'].append(w.copy())
        history['objective'].append(objective_function(w, X, y, lambda_val))
        
    return w, history

# Step 5: Generate synthetic data for visualization
def generate_data(n_samples=100, noise=0.5):
    """
    Generate synthetic binary classification data.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        
    Returns:
        X: Feature matrix with added bias term
        y: Binary target vector
    """
    # Generate 2D data around different centers
    X_class0 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add a column of ones for the bias term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    return X, y

# Step 6: Create a function to visualize decision boundaries
def plot_decision_boundary(X, y, w, title):
    """
    Plot the decision boundary for a logistic regression model.
    
    Args:
        X: Feature matrix with bias term
        y: Target vector
        w: Weight vector
        title: Title for the plot
    """
    # Create a mesh grid
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Add bias term to mesh grid points
    mesh_points = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    
    # Predict class for each mesh point
    Z = sigmoid(mesh_points @ w)
    Z = Z.reshape(xx.shape)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.contour(xx, yy, Z, [0.5], linewidths=2, colors='k')
    
    # Plot the data points
    plt.scatter(X[y == 0, 1], X[y == 0, 2], c='red', marker='o', label='Class 0')
    plt.scatter(X[y == 1, 1], X[y == 1, 2], c='blue', marker='x', label='Class 1')
    
    # Add labels
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    return plt

# Step 7: Plot the convergence of the objective function
def plot_convergence(hist_bgd, hist_sgd):
    """
    Plot the convergence of BGD and SGD.
    
    Args:
        hist_bgd: History of batch gradient descent
        hist_sgd: History of stochastic gradient descent
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(hist_bgd['objective'], 'r-', linewidth=2, label='Batch Gradient Descent')
    plt.plot(hist_sgd['objective'], 'b--', linewidth=2, label='Stochastic Gradient Descent')
    
    plt.xlabel('Iterations/Epochs')
    plt.ylabel('$J(w)$')
    plt.title('Convergence of Objective Function')
    plt.grid(True)
    plt.legend()
    
    return plt

# Step 8: Visualize the effect of regularization parameter
def plot_regularization_effect(X, y, lambda_values):
    """
    Plot the effect of different regularization parameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        lambda_values: List of regularization parameters to try
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different lambda values
    for i, lambda_val in enumerate(lambda_values):
        plt.subplot(2, 3, i+1)
        
        # Train model
        w, _ = batch_gradient_descent(X, y, lambda_val, 0.1, 1000)
        
        # Create a mesh grid
        x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                           np.arange(y_min, y_max, 0.02))
        
        # Add bias term to mesh grid points
        mesh_points = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        
        # Predict class for each mesh point
        Z = sigmoid(mesh_points @ w)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
        plt.contour(xx, yy, Z, [0.5], linewidths=2, colors='k')
        
        # Plot the data points
        plt.scatter(X[y == 0, 1], X[y == 0, 2], c='red', marker='o', edgecolor='k', s=40, label='Class 0')
        plt.scatter(X[y == 1, 1], X[y == 1, 2], c='blue', marker='x', edgecolor='k', s=40, label='Class 1')
        
        # Display the weights
        plt.title(f'$\\lambda = {lambda_val}$\n$w = [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]$')
        plt.grid(True)
        
        # Only add legend to the first subplot
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    
    return plt

# Step 9: Show the analytical derivation of the gradient
def print_derivation():
    """
    Print the mathematical derivation of the gradient.
    """
    derivation = """
    Analytical Derivation of the Gradient:
    
    Given the objective function:
    J(w) = -1/N ∑[yi log(hw(xi)) + (1-yi) log(1-hw(xi))] + λ||w||²
    where hw(xi) = 1/(1 + e^(-w·xi))
    
    Step 1: Compute partial derivative of J(w) with respect to wj
    
    First, consider the cross-entropy term: 
    L(w) = -1/N ∑[yi log(hw(xi)) + (1-yi) log(1-hw(xi))]
    
    Using the chain rule:
    ∂L/∂wj = -1/N ∑[yi · 1/hw(xi) · ∂hw(xi)/∂wj - (1-yi) · 1/(1-hw(xi)) · ∂hw(xi)/∂wj]
    
    For the sigmoid function hw(xi) = 1/(1 + e^(-w·xi)), the derivative is:
    ∂hw(xi)/∂wj = hw(xi) * (1 - hw(xi)) * xij
    
    Where xij is the j-th feature of the i-th sample.
    
    Substituting:
    ∂L/∂wj = -1/N ∑[yi · 1/hw(xi) · hw(xi) * (1 - hw(xi)) * xij - (1-yi) · 1/(1-hw(xi)) · hw(xi) * (1 - hw(xi)) * xij]
             = -1/N ∑[yi · (1 - hw(xi)) * xij - (1-yi) · hw(xi) * xij]
             = -1/N ∑[yi · xij - yi · hw(xi) · xij - xij · hw(xi) + yi · hw(xi) · xij]
             = -1/N ∑[(yi - hw(xi)) · xij]
             = 1/N ∑[(hw(xi) - yi) · xij]
    
    Next, consider the regularization term:
    R(w) = λ||w||² = λ∑(wj²)
    
    The derivative is:
    ∂R/∂wj = 2λwj
    
    Combining both terms, the full gradient is:
    
    ∂J/∂wj = 1/N ∑[(hw(xi) - yi) · xij] + 2λwj
    
    In vector form:
    ∇J(w) = 1/N X^T(hw(X) - y) + 2λw
    
    Note: Typically, we don't regularize the bias term w0, so:
    ∇J(w) = 1/N X^T(hw(X) - y) + 2λ[0, w1, w2, ..., wn]^T
    """
    
    return derivation

# Step 10: Implement and demonstrate the methods
def main():
    """
    Main function to demonstrate the solutions.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    X, y = generate_data(n_samples=200, noise=0.5)
    
    print("Step 1: Analytical Derivation of the Gradient")
    print(print_derivation())
    
    print("\nStep 2: Batch Gradient Descent Implementation")
    # Train with batch gradient descent
    lambda_val = 0.1
    learning_rate = 0.1
    num_iters = 1000
    
    w_bgd, hist_bgd = batch_gradient_descent(X, y, lambda_val, learning_rate, num_iters)
    
    print(f"Final weights (BGD): {w_bgd}")
    print(f"Final objective value (BGD): {hist_bgd['objective'][-1]:.4f}")
    
    print("\nStep 3: Stochastic Gradient Descent Implementation")
    # Train with stochastic gradient descent
    num_epochs = 50
    
    w_sgd, hist_sgd = stochastic_gradient_descent(X, y, lambda_val, learning_rate, num_epochs)
    
    print(f"Final weights (SGD): {w_sgd}")
    print(f"Final objective value (SGD): {hist_sgd['objective'][-1]:.4f}")
    
    print("\nStep 4: Visualizing Decision Boundaries")
    
    # Plot BGD decision boundary
    plt_bgd = plot_decision_boundary(X, y, w_bgd, f"Decision Boundary - Batch Gradient Descent ($\\lambda={lambda_val}$)")
    plt_bgd.savefig(os.path.join(save_dir, 'decision_boundary_bgd.png'), dpi=300, bbox_inches='tight')
    
    # Plot SGD decision boundary
    plt_sgd = plot_decision_boundary(X, y, w_sgd, f"Decision Boundary - Stochastic Gradient Descent ($\\lambda={lambda_val}$)")
    plt_sgd.savefig(os.path.join(save_dir, 'decision_boundary_sgd.png'), dpi=300, bbox_inches='tight')
    
    print("\nStep 5: Visualizing Convergence")
    # Plot convergence
    plt_conv = plot_convergence(hist_bgd, hist_sgd)
    plt_conv.savefig(os.path.join(save_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
    
    print("\nStep 6: Visualizing Regularization Effect")
    # Plot regularization effect
    lambda_values = [0, 0.01, 0.1, 1, 10, 100]
    plt_reg = plot_regularization_effect(X, y, lambda_values)
    plt_reg.savefig(os.path.join(save_dir, 'regularization_effect.png'), dpi=300, bbox_inches='tight')
    
    print("\nStep 7: Batch Gradient Descent Update Rule")
    print("w := w - α * ∇J(w)")
    print("w := w - α * [1/N X^T(hw(X) - y) + 2λw]")
    
    print("\nStep 8: Stochastic Gradient Descent Update Rule")
    print("For each sample (xi, yi):")
    print("w := w - α * [(hw(xi) - yi) * xi + 2λw]")
    
    print("\nStep 9: Role of λ in the Objective Function")
    print("The parameter λ controls the strength of L2 regularization:")
    print("1. λ = 0: No regularization, model may overfit")
    print("2. Small λ: Slight regularization, helps reduce overfitting while maintaining flexibility")
    print("3. Large λ: Strong regularization, may lead to underfitting")
    print("The primary role of λ is to prevent overfitting by penalizing large weight values.")
    
    print("\nAll visualizations saved to:", save_dir)

if __name__ == "__main__":
    main() 