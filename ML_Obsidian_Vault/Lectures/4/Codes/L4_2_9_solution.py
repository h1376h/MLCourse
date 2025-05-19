import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

# Create a non-linearly separable dataset
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# Convert labels from 0/1 to -1/1 for perceptron
y = 2 * y - 1

# Function to generate XOR dataset
def generate_xor_data(n_samples_per_quadrant=25, noise=0.1):
    # Generate data in four quadrants
    np.random.seed(42)
    
    # Positive class: points in Q1 and Q3
    X1_q1 = np.random.randn(n_samples_per_quadrant, 2) * noise + [1, 1]
    X1_q3 = np.random.randn(n_samples_per_quadrant, 2) * noise + [-1, -1]
    
    # Negative class: points in Q2 and Q4
    X2_q2 = np.random.randn(n_samples_per_quadrant, 2) * noise + [-1, 1]
    X2_q4 = np.random.randn(n_samples_per_quadrant, 2) * noise + [1, -1]
    
    # Combine all points
    X = np.vstack([X1_q1, X1_q3, X2_q2, X2_q4])
    y = np.array([1] * (2 * n_samples_per_quadrant) + [-1] * (2 * n_samples_per_quadrant))
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# Define kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2):
    return (np.dot(x1, x2) + 1) ** degree

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Define sigmoid kernel function (additional kernel example)
def sigmoid_kernel(x1, x2, scale=0.01, c=1.0):
    return np.tanh(scale * np.dot(x1, x2) + c)

# Define a function to compute the kernel matrix
def compute_kernel_matrix(X, kernel_func, **kernel_params):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j], **kernel_params)
    
    return K

# Plot the dataset
def plot_dataset(X, y, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Class -1')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Function to visualize data in 3D with a specific transformation
def plot_3d_transformation(X, y, transform_func, title, filename, elev=30, azim=30):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply transformation to get the 3rd dimension
    X_transformed = np.zeros((X.shape[0], 3))
    X_transformed[:, 0:2] = X
    
    for i in range(X.shape[0]):
        X_transformed[i, 2] = transform_func(X[i])
    
    # Plot the transformed points
    ax.scatter(X_transformed[y == 1, 0], X_transformed[y == 1, 1], X_transformed[y == 1, 2], 
               c='blue', marker='o', label='Class 1', s=50)
    ax.scatter(X_transformed[y == -1, 0], X_transformed[y == -1, 1], X_transformed[y == -1, 2], 
               c='red', marker='x', label='Class -1', s=50)
    
    # Add a plane at z=0 to show the decision boundary in the transformed space
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                        np.linspace(ylim[0], ylim[1], 10))
    zz = np.zeros(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\phi(x)$')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Kernel Perceptron Algorithm
def kernel_perceptron(X, y, kernel_func, max_iterations=100, **kernel_params):
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    K = compute_kernel_matrix(X, kernel_func, **kernel_params)
    
    iteration_history = []
    misclassified_history = []
    errors_history = []
    
    for iteration in range(1, max_iterations + 1):
        errors = 0
        misclassified = []
        
        for i in range(n_samples):
            # Compute the prediction using the kernel trick
            prediction = 0
            for j in range(n_samples):
                prediction += alpha[j] * y[j] * K[j, i]
            
            prediction = np.sign(prediction)
            
            # Check if misclassified
            if prediction != y[i]:
                alpha[i] += 1
                errors += 1
                misclassified.append(i)
        
        print(f"Iteration {iteration}: {errors} misclassifications")
        iteration_history.append({
            'iteration': iteration,
            'alpha': alpha.copy(),
            'errors': errors,
            'misclassified': misclassified.copy()
        })
        
        misclassified_history.append(misclassified.copy())
        errors_history.append(errors)
        
        # Check for convergence
        if errors == 0:
            print(f"Converged after {iteration} iterations!")
            break
    
    return alpha, iteration_history, misclassified_history, errors_history

# Plot decision boundaries
def plot_decision_boundary(X, y, alpha, kernel_func, title, filename, resolution=100, **kernel_params):
    # Define the mesh grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    # Compute predictions for each point in the mesh
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([xx[i, j], yy[i, j]])
            prediction = 0
            for k in range(len(X)):
                if alpha[k] > 0:  # Only support vectors contribute
                    prediction += alpha[k] * y[k] * kernel_func(X[k], point, **kernel_params)
            Z[i, j] = prediction
    
    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, 
                 colors=['red', 'white', 'blue'])
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
    
    # Plot the original points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='Class -1')
    
    # Highlight support vectors
    support_vectors = np.where(alpha > 0)[0]
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=200,
               facecolors='none', edgecolors='green', linewidth=2,
               label='Support Vectors')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot learning curves
def plot_learning_curves(errors_history, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors_history) + 1), errors_history, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Misclassifications')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Define transformation for visualization of the kernel trick
def quadratic_transform(x):
    return x[0]**2 + x[1]**2

# Main script
if __name__ == "__main__":
    print("Kernelized Perceptron Demonstration")
    print("=" * 50)
    
    # PART 1: Circle dataset demonstration
    print("\n--- PART 1: Circle Dataset ---")
    
    # Plot the original dataset
    plot_dataset(X, y, "Non-linearly separable dataset (circular pattern)", 
                os.path.join(save_dir, "dataset.png"))
    
    print("\n1. Linear Kernel (Standard Perceptron)")
    print("-" * 50)
    alpha_linear, iterations_linear, misclassified_linear, errors_linear = kernel_perceptron(
        X, y, linear_kernel, max_iterations=50)
    
    plot_decision_boundary(X, y, alpha_linear, linear_kernel, 
                          "Linear Kernel Decision Boundary", 
                          os.path.join(save_dir, "linear_kernel_boundary.png"))
    
    plot_learning_curves(errors_linear, "Learning Curve - Linear Kernel", 
                        os.path.join(save_dir, "linear_kernel_curve.png"))
    
    print("\n2. Polynomial Kernel (Degree 2)")
    print("-" * 50)
    alpha_poly, iterations_poly, misclassified_poly, errors_poly = kernel_perceptron(
        X, y, polynomial_kernel, max_iterations=50, degree=2)
    
    plot_decision_boundary(X, y, alpha_poly, polynomial_kernel, 
                          "Polynomial Kernel (Degree 2) Decision Boundary", 
                          os.path.join(save_dir, "poly_kernel_boundary.png"),
                          degree=2)
    
    plot_learning_curves(errors_poly, "Learning Curve - Polynomial Kernel", 
                        os.path.join(save_dir, "poly_kernel_curve.png"))
    
    print("\n3. RBF Kernel")
    print("-" * 50)
    alpha_rbf, iterations_rbf, misclassified_rbf, errors_rbf = kernel_perceptron(
        X, y, rbf_kernel, max_iterations=50, gamma=1.0)
    
    plot_decision_boundary(X, y, alpha_rbf, rbf_kernel, 
                          "RBF Kernel Decision Boundary", 
                          os.path.join(save_dir, "rbf_kernel_boundary.png"),
                          gamma=1.0)
    
    plot_learning_curves(errors_rbf, "Learning Curve - RBF Kernel", 
                        os.path.join(save_dir, "rbf_kernel_curve.png"))
    
    # Visualize the kernel trick with 3D transformation
    plot_3d_transformation(X, y, quadratic_transform, 
                          "Feature Space Transformation for Circular Data", 
                          os.path.join(save_dir, "kernel_trick_visualization.png"))
    
    # Find optimal kernel and parameters
    print("\n4. Summary of Results (Circle Dataset)")
    print("-" * 50)
    print(f"Linear Kernel: {len(iterations_linear)} iterations, {np.sum(alpha_linear > 0)} support vectors")
    print(f"Polynomial Kernel: {len(iterations_poly)} iterations, {np.sum(alpha_poly > 0)} support vectors")
    print(f"RBF Kernel: {len(iterations_rbf)} iterations, {np.sum(alpha_rbf > 0)} support vectors")
    
    # Generate a side-by-side comparison of decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    kernels = [
        {'title': 'Linear Kernel', 'alpha': alpha_linear, 'func': linear_kernel, 'params': {}},
        {'title': 'Polynomial Kernel', 'alpha': alpha_poly, 'func': polynomial_kernel, 'params': {'degree': 2}},
        {'title': 'RBF Kernel', 'alpha': alpha_rbf, 'func': rbf_kernel, 'params': {'gamma': 1.0}}
    ]
    
    # Define the mesh grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    for i, kernel in enumerate(kernels):
        # Compute predictions for each point in the mesh
        Z = np.zeros((resolution, resolution))
        for r in range(resolution):
            for c in range(resolution):
                point = np.array([xx[r, c], yy[r, c]])
                prediction = 0
                for k in range(len(X)):
                    if kernel['alpha'][k] > 0:  # Only support vectors contribute
                        prediction += kernel['alpha'][k] * y[k] * kernel['func'](X[k], point, **kernel['params'])
                Z[r, c] = prediction
        
        # Plot decision boundary and data points
        axes[i].contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'white', 'blue'])
        axes[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
        axes[i].scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', s=25)
        axes[i].scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', s=25)
        
        # Highlight support vectors
        support_vectors = np.where(kernel['alpha'] > 0)[0]
        axes[i].scatter(X[support_vectors, 0], X[support_vectors, 1], s=100,
                      facecolors='none', edgecolors='green', linewidth=1.5)
        
        axes[i].set_xlabel('$x_1$')
        axes[i].set_ylabel('$x_2$')
        axes[i].set_title(kernel['title'])
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PART 2: XOR dataset demonstration
    print("\n\n--- PART 2: XOR Dataset ---")
    
    # Generate XOR data
    X_xor, y_xor = generate_xor_data(n_samples_per_quadrant=25, noise=0.2)
    
    # Plot the XOR dataset
    plot_dataset(X_xor, y_xor, "XOR Dataset (non-linearly separable)", 
                os.path.join(save_dir, "xor_dataset.png"))
    
    # Define transformation for XOR visualization
    def xor_transform(x):
        return x[0] * x[1]  # Simple product transformation for XOR
    
    # Visualize the kernel trick with 3D transformation for XOR
    plot_3d_transformation(X_xor, y_xor, xor_transform, 
                          "Feature Space Transformation for XOR Data", 
                          os.path.join(save_dir, "xor_kernel_trick.png"),
                          elev=20, azim=45)
    
    print("\n1. Linear Kernel on XOR")
    print("-" * 50)
    alpha_linear_xor, iter_linear_xor, misclassified_linear_xor, errors_linear_xor = kernel_perceptron(
        X_xor, y_xor, linear_kernel, max_iterations=50)
    
    plot_decision_boundary(X_xor, y_xor, alpha_linear_xor, linear_kernel, 
                          "Linear Kernel Decision Boundary (XOR)", 
                          os.path.join(save_dir, "xor_linear_kernel_boundary.png"))
    
    print("\n2. Polynomial Kernel on XOR")
    print("-" * 50)
    alpha_poly_xor, iter_poly_xor, misclassified_poly_xor, errors_poly_xor = kernel_perceptron(
        X_xor, y_xor, polynomial_kernel, max_iterations=50, degree=2)
    
    plot_decision_boundary(X_xor, y_xor, alpha_poly_xor, polynomial_kernel, 
                          "Polynomial Kernel (Degree 2) Decision Boundary (XOR)", 
                          os.path.join(save_dir, "xor_poly_kernel_boundary.png"),
                          degree=2)
    
    print("\n3. Sigmoid Kernel on XOR")
    print("-" * 50)
    alpha_sigmoid_xor, iter_sigmoid_xor, misclassified_sigmoid_xor, errors_sigmoid_xor = kernel_perceptron(
        X_xor, y_xor, sigmoid_kernel, max_iterations=50, scale=1.0, c=0)
    
    plot_decision_boundary(X_xor, y_xor, alpha_sigmoid_xor, sigmoid_kernel, 
                          "Sigmoid Kernel Decision Boundary (XOR)", 
                          os.path.join(save_dir, "xor_sigmoid_kernel_boundary.png"),
                          scale=1.0, c=0)
    
    # Side by side comparison for XOR dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    kernels_xor = [
        {'title': 'Linear Kernel', 'alpha': alpha_linear_xor, 'func': linear_kernel, 'params': {}},
        {'title': 'Polynomial Kernel', 'alpha': alpha_poly_xor, 'func': polynomial_kernel, 'params': {'degree': 2}},
        {'title': 'Sigmoid Kernel', 'alpha': alpha_sigmoid_xor, 'func': sigmoid_kernel, 'params': {'scale': 1.0, 'c': 0}}
    ]
    
    # Define the mesh grid for XOR
    x_min, x_max = X_xor[:, 0].min() - 0.1, X_xor[:, 0].max() + 0.1
    y_min, y_max = X_xor[:, 1].min() - 0.1, X_xor[:, 1].max() + 0.1
    resolution = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    for i, kernel in enumerate(kernels_xor):
        # Compute predictions for each point in the mesh
        Z = np.zeros((resolution, resolution))
        for r in range(resolution):
            for c in range(resolution):
                point = np.array([xx[r, c], yy[r, c]])
                prediction = 0
                for k in range(len(X_xor)):
                    if kernel['alpha'][k] > 0:  # Only support vectors contribute
                        prediction += kernel['alpha'][k] * y_xor[k] * kernel['func'](X_xor[k], point, **kernel['params'])
                Z[r, c] = prediction
        
        # Plot decision boundary and data points
        axes[i].contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'white', 'blue'])
        axes[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
        axes[i].scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], color='blue', marker='o', s=25)
        axes[i].scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], color='red', marker='x', s=25)
        
        # Highlight support vectors
        support_vectors = np.where(kernel['alpha'] > 0)[0]
        axes[i].scatter(X_xor[support_vectors, 0], X_xor[support_vectors, 1], s=100,
                      facecolors='none', edgecolors='green', linewidth=1.5)
        
        axes[i].set_xlabel('$x_1$')
        axes[i].set_ylabel('$x_2$')
        axes[i].set_title(kernel['title'])
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xor_kernel_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n4. Summary of Results (XOR Dataset)")
    print("-" * 50)
    print(f"Linear Kernel: {len(iter_linear_xor)} iterations, {np.sum(alpha_linear_xor > 0)} support vectors")
    print(f"Polynomial Kernel: {len(iter_poly_xor)} iterations, {np.sum(alpha_poly_xor > 0)} support vectors")
    print(f"Sigmoid Kernel: {len(iter_sigmoid_xor)} iterations, {np.sum(alpha_sigmoid_xor > 0)} support vectors")
    
    print(f"\nPlots saved to: {save_dir}")
    
    # Theoretical explanations for the quiz questions
    print("\n--- Quiz Question Answers ---")
    print("\n1. The kernel trick in one or two sentences:")
    print("The kernel trick allows us to implicitly map data into a high-dimensional feature space without actually computing the transformed coordinates, by using kernel functions that directly compute inner products in that space.")
    
    print("\n2. Advantage of kernelized perceptron over standard perceptron:")
    print("Kernelized perceptron can learn non-linear decision boundaries, allowing it to solve classification problems that are not linearly separable in the original feature space.")
    
    print("\n3. Example of a kernel function and resulting decision boundary:")
    print("The Radial Basis Function (RBF) kernel, K(x,y) = exp(-γ||x-y||²), can produce complex, closed decision boundaries like circles and ellipses by implicitly mapping points to an infinite-dimensional feature space.")
    
    print("\n4. Kernelized version of the perceptron decision function:")
    print("The kernelized perceptron decision function is f(x) = sign(Σ αᵢyᵢK(xᵢ,x)), where αᵢ are the coefficients for each training example, yᵢ are their labels, and K is the kernel function.") 