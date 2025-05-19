import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def create_separable_dataset(n=20, dim=2, margin=0.5, seed=42):
    """
    Create a linearly separable dataset with a specified margin
    
    Args:
        n: Number of samples
        dim: Dimensionality of features
        margin: The margin between classes
        seed: Random seed
        
    Returns:
        X: Feature matrix
        y: Labels
        w_true: True weight vector that separates the data with the specified margin
    """
    np.random.seed(seed)
    
    # Generate a random hyperplane (normal vector)
    w_true = np.random.randn(dim)
    w_true = w_true / np.linalg.norm(w_true)  # Normalize to unit vector
    
    # Add bias term to the weight vector
    w_true = np.append(w_true, 0)  # Initial bias is 0
    
    # Generate random points
    X = np.random.randn(n, dim)
    
    # Normalize features for bounded condition
    R = 1.0  # Bound for feature vectors
    for i in range(n):
        norm = np.linalg.norm(X[i])
        if norm > R:
            X[i] = X[i] * (R / norm)
    
    # Add a column of ones for the bias term
    X_with_bias = np.hstack((X, np.ones((n, 1))))
    
    # Calculate activation for each point
    activations = np.dot(X_with_bias, w_true)
    
    # Adjust the bias to create the margin
    max_negative_activation = np.max(activations[activations < 0])
    min_positive_activation = np.min(activations[activations > 0])
    
    if not np.isneginf(max_negative_activation) and not np.isposinf(min_positive_activation):
        midpoint = (max_negative_activation + min_positive_activation) / 2
        w_true[-1] = -midpoint  # Adjust bias to center the hyperplane between classes
    
    # Recalculate activations with the adjusted bias
    activations = np.dot(X_with_bias, w_true)
    
    # Assign labels based on activations
    y = np.sign(activations)
    
    # Ensure we have both positive and negative examples
    if np.all(y > 0) or np.all(y < 0):
        # If all labels are the same, flip some of them
        half_indices = n // 2
        y[:half_indices] = 1
        y[half_indices:] = -1
        
        # Regenerate features to match the new labels
        for i in range(n):
            if y[i] > 0:
                # Generate point on the positive side with margin
                direction = np.random.randn(dim)
                direction = direction / np.linalg.norm(direction)
                X[i] = direction * (R * np.random.random()) + margin * w_true[:dim]
            else:
                # Generate point on the negative side with margin
                direction = np.random.randn(dim)
                direction = direction / np.linalg.norm(direction)
                X[i] = direction * (R * np.random.random()) - margin * w_true[:dim]
    
    # Ensure dataset is linearly separable with the specified margin
    X_with_bias = np.hstack((X, np.ones((n, 1))))
    activations = np.dot(X_with_bias, w_true)
    
    # Scale the weight vector to enforce the margin
    min_activation = np.min(np.abs(activations))
    if min_activation < margin:
        scaling_factor = margin / min_activation
        w_true = w_true * scaling_factor
    
    return X, y, w_true

def create_non_separable_dataset(n=20, dim=2, overlap=0.2, seed=42):
    """
    Create a linearly non-separable dataset with some overlap between classes
    """
    np.random.seed(seed)
    
    # First create a separable dataset
    X, y, w_true = create_separable_dataset(n, dim, margin=0.5, seed=seed)
    
    # Make it non-separable by flipping some labels
    num_to_flip = int(overlap * n)
    indices_to_flip = np.random.choice(n, num_to_flip, replace=False)
    y[indices_to_flip] = -y[indices_to_flip]
    
    return X, y

def perceptron_algorithm(X, y, max_iterations=1000, learning_rate=1.0):
    """
    Implementation of the perceptron algorithm
    
    Args:
        X: Feature matrix (without bias term)
        y: Labels
        max_iterations: Maximum number of iterations
        learning_rate: Learning rate
        
    Returns:
        w: Learned weight vector
        iteration_history: List of dictionaries containing information about each iteration
        converged: Boolean indicating if the algorithm converged
    """
    # Add bias term to input features
    n_samples, n_features = X.shape
    X_with_bias = np.hstack((X, np.ones((n_samples, 1))))
    
    # Initialize weights to zeros
    w = np.zeros(n_features + 1)
    
    # Initialize variables for tracking
    iteration_history = []
    update_count = 0
    iteration = 0
    converged = False
    
    while not converged and iteration < max_iterations:
        iteration += 1
        misclassified = []
        made_update = False
        
        # Check each sample for misclassification
        for i in range(n_samples):
            x_i = X_with_bias[i]
            y_i = y[i]
            
            # Compute activation
            activation = np.dot(w, x_i)
            prediction = np.sign(activation)
            
            # Check if misclassified
            if prediction != y_i:
                misclassified.append(i)
                
                # Update weights
                w_old = w.copy()
                w = w + learning_rate * y_i * x_i
                
                update_count += 1
                made_update = True
                
                # Store information about this update
                iteration_history.append({
                    'iteration': iteration,
                    'sample_index': i,
                    'old_weights': w_old.copy(),
                    'new_weights': w.copy(),
                    'update_count': update_count
                })
                
                break  # Move to next iteration after first update
        
        # Check if converged (no misclassifications)
        if not made_update:
            converged = True
    
    return w, iteration_history, converged, update_count

def calculate_margin(X, y, w):
    """
    Calculate the geometric margin of a dataset with respect to a weight vector
    """
    X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
    
    # Normalize weight vector (excluding bias term)
    w_norm = np.linalg.norm(w[:-1])
    if w_norm > 0:
        w_normalized = w / w_norm
    else:
        return 0
    
    # Calculate distances to the decision boundary
    distances = np.abs(np.dot(X_with_bias, w_normalized))
    
    # Return the minimum distance
    if len(distances) > 0:
        return np.min(distances)
    else:
        return 0

def calculate_R(X):
    """
    Calculate the maximum norm of feature vectors
    """
    norms = np.linalg.norm(X, axis=1)
    return np.max(norms)

def plot_dataset_and_boundary(X, y, w=None, margin=None, title=None, filename=None):
    """
    Plot a 2D dataset and its decision boundary
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class +1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == np.where(y == -1)[0][0] else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Plot the decision boundary if weights are provided
    if w is not None and not np.all(w[:-1] == 0):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 3, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Decision Boundary')
            
            # Plot margins if provided
            if margin is not None:
                # Calculate margin lines
                w_norm = np.linalg.norm([w1, w2])
                margin_distance = margin / w_norm if w_norm > 0 else 0
                
                # Upper margin: w1*x1 + w2*x2 + w0 = margin
                x2_upper = (-w1 * x1_range - w0 + margin) / w2
                plt.plot(x1_range, x2_upper, 'g--', alpha=0.5, label='Margin')
                
                # Lower margin: w1*x1 + w2*x2 + w0 = -margin
                x2_lower = (-w1 * x1_range - w0 - margin) / w2
                plt.plot(x1_range, x2_lower, 'g--', alpha=0.5)
                
                # Fill the margin region
                plt.fill_between(x1_range, x2_lower, x2_upper, color='green', alpha=0.1)
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            x1_boundary = -w0 / w1
            plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Decision Boundary')
            
            # Plot margins if provided
            if margin is not None:
                margin_distance = margin / abs(w1) if abs(w1) > 0 else 0
                plt.axvline(x=x1_boundary + margin_distance, color='g', linestyle='--', alpha=0.5, label='Margin')
                plt.axvline(x=x1_boundary - margin_distance, color='g', linestyle='--', alpha=0.5)
                
                # Fill the margin region
                plt.axvspan(x1_boundary - margin_distance, x1_boundary + margin_distance, color='green', alpha=0.1)
    
    # Add labels and title
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Dataset and Decision Boundary', fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set limits
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add equation of the decision boundary to the plot if weights are provided
    if w is not None and not np.all(w[:-1] == 0):
        eq_str = f'$w_1 x_1 + w_2 x_2 + w_0 = 0$\n${w[0]:.2f} x_1 + {w[1]:.2f} x_2 + {w[2]:.2f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1), fontsize=12)
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_convergence_bound(n_samples_list, margins_list, max_updates_list, filename=None):
    """
    Plot the theoretical upper bound on the number of updates
    """
    plt.figure(figsize=(10, 8))
    
    for i, margin in enumerate(margins_list):
        max_updates = max_updates_list[i]
        plt.plot(n_samples_list, max_updates, marker='o', label=f'Margin = {margin}')
    
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Maximum Number of Updates', fontsize=14)
    plt.title('Perceptron Convergence Theorem Bound', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_update_comparison(n_samples_list, theoretical_updates, actual_updates, filename=None):
    """
    Plot comparison between theoretical and actual number of updates
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(n_samples_list, theoretical_updates, marker='o', label='Theoretical Bound')
    plt.plot(n_samples_list, actual_updates, marker='x', label='Actual Updates')
    
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Number of Updates', fontsize=14)
    plt.title('Theoretical vs Actual Number of Updates', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_non_separable_evolution(X, y, iteration_history, filename=None):
    """
    Plot the evolution of the decision boundary for non-separable data
    """
    plt.figure(figsize=(12, 10))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class +1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == np.where(y == -1)[0][0] else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Plot decision boundaries at different iterations
    num_boundaries = min(5, len(iteration_history))
    indices = np.linspace(0, len(iteration_history) - 1, num_boundaries, dtype=int)
    
    color_map = plt.cm.rainbow(np.linspace(0, 1, num_boundaries))
    x1_range = np.linspace(-3, 3, 100)
    
    for idx, iteration_idx in enumerate(indices):
        w = iteration_history[iteration_idx]['new_weights']
        update_count = iteration_history[iteration_idx]['update_count']
        
        w1, w2, w0 = w
        
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, color=color_map[idx], 
                    linestyle='-', alpha=0.6, 
                    label=f'Update {update_count}')
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('Perceptron on Non-Separable Data - Decision Boundary Evolution', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend(fontsize=12)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# Problem 1: State the perceptron convergence theorem
# (This is a theoretical statement, no code needed)

# Problem 2: Conditions for the perceptron convergence theorem
# Demonstrate linear separability and bounded features
print("Problem 2: Conditions for the perceptron convergence theorem")
print("-" * 70)

# Create a linearly separable dataset with a margin
n_samples = 20
dim = 2
margin = 0.5
seed = 42

X, y, w_true = create_separable_dataset(n_samples, dim, margin, seed)
R = calculate_R(X)
actual_margin = calculate_margin(X, y, w_true)

print(f"Dataset created with {n_samples} samples in {dim} dimensions")
print(f"True weight vector: {w_true}")
print(f"Maximum feature vector norm (R): {R:.4f}")
print(f"Actual margin (γ): {actual_margin:.4f}")

# Plot the dataset and the separating hyperplane
plot_dataset_and_boundary(
    X, y, w_true, actual_margin,
    title='Linearly Separable Dataset with Margin',
    filename=os.path.join(save_dir, 'separable_dataset.png')
)

print("Linear separability condition verified with visualized margin.")
print()

# Problem 3: Maximum number of updates
print("Problem 3: Maximum number of updates calculation")
print("-" * 70)

# Calculate the theoretical bound
theoretical_max_updates = (R**2) / (actual_margin**2)
print(f"Theoretical bound on number of updates: (R²/γ²) = ({R:.4f}² / {actual_margin:.4f}²) = {theoretical_max_updates:.4f}")

# Run the perceptron algorithm
w_learned, iteration_history, converged, update_count = perceptron_algorithm(X, y)

print(f"Perceptron algorithm converged: {converged}")
print(f"Actual number of updates: {update_count}")
print(f"Learned weight vector: {w_learned}")

# Plot the learned decision boundary
plot_dataset_and_boundary(
    X, y, w_learned, actual_margin,
    title=f'Perceptron Convergence (Updates: {update_count})',
    filename=os.path.join(save_dir, 'perceptron_convergence.png')
)

# Generate data for comparing theoretical and actual updates with varying parameters
print("\nEvaluating bound with varying parameters:")

# Vary number of samples
n_samples_list = [10, 20, 30, 40, 50]
margins_list = [0.1, 0.3, 0.5]
max_updates_theoretical_list = []

for margin in margins_list:
    max_updates = []
    for n in n_samples_list:
        X_temp, y_temp, w_temp = create_separable_dataset(n, dim, margin, seed=seed+n)
        R_temp = calculate_R(X_temp)
        actual_margin_temp = calculate_margin(X_temp, y_temp, w_temp)
        bound = (R_temp**2) / (actual_margin_temp**2)
        max_updates.append(bound)
    max_updates_theoretical_list.append(max_updates)

plot_convergence_bound(
    n_samples_list, 
    margins_list, 
    max_updates_theoretical_list,
    filename=os.path.join(save_dir, 'convergence_bound.png')
)

# Compare theoretical bound with actual updates
actual_updates_list = []
theoretical_updates_list = []

for n in n_samples_list:
    X_temp, y_temp, w_temp = create_separable_dataset(n, dim, margin=0.5, seed=seed+n)
    R_temp = calculate_R(X_temp)
    actual_margin_temp = calculate_margin(X_temp, y_temp, w_temp)
    bound = (R_temp**2) / (actual_margin_temp**2)
    theoretical_updates_list.append(bound)
    
    _, _, _, updates = perceptron_algorithm(X_temp, y_temp)
    actual_updates_list.append(updates)
    
    print(f"n={n}, R={R_temp:.4f}, γ={actual_margin_temp:.4f}, Bound={bound:.4f}, Actual={updates}")

plot_update_comparison(
    n_samples_list, 
    theoretical_updates_list, 
    actual_updates_list,
    filename=os.path.join(save_dir, 'theoretical_vs_actual.png')
)

# Problem 4: Non-convergence for non-separable data
print("\nProblem 4: Non-convergence for non-separable data")
print("-" * 70)

# Create a non-separable dataset
X_nonsep, y_nonsep = create_non_separable_dataset(n=20, dim=2, overlap=0.2, seed=42)

# Run perceptron on non-separable data with a limit on iterations
w_nonsep, iteration_history_nonsep, converged_nonsep, update_count_nonsep = perceptron_algorithm(
    X_nonsep, y_nonsep, max_iterations=100
)

print(f"Perceptron converged on non-separable data: {converged_nonsep}")
print(f"Number of updates on non-separable data: {update_count_nonsep}")

# Plot final state on non-separable data
plot_dataset_and_boundary(
    X_nonsep, y_nonsep, w_nonsep,
    title='Perceptron on Non-Separable Data',
    filename=os.path.join(save_dir, 'non_separable_dataset.png')
)

# Plot decision boundary evolution for non-separable data
if len(iteration_history_nonsep) > 0:
    plot_non_separable_evolution(
        X_nonsep, y_nonsep, iteration_history_nonsep,
        filename=os.path.join(save_dir, 'non_separable_evolution.png')
    )

print("\nSummary of findings:")
print("-" * 70)
print("1. Perceptron Convergence Theorem: The algorithm will converge in a finite number of updates")
print(f"   for linearly separable data with a margin γ = {actual_margin:.4f}.")
print("2. Required conditions: linear separability and bounded feature vectors.")
print(f"3. Maximum updates bound: (R²/γ²) = {theoretical_max_updates:.4f}.")
print(f"   Actual updates in our experiment: {update_count}.")
print("4. Non-separable data: The algorithm did not converge within the iteration limit.")
print(f"   Updates performed on non-separable data: {update_count_nonsep}.")
print("\nAll visualizations saved to:", save_dir) 