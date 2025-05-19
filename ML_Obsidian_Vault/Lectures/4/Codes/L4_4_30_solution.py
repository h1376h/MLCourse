import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset
X = np.array([
    [-2, 0],   # Class -1
    [-1, -2],  # Class -1
    [0, -1],   # Class -1
    [1, 1],    # Class 1
    [2, 2],    # Class 1
    [0, 3],    # Class 1
    [4, 0]     # Class -1 (outlier)
])

y = np.array([-1, -1, -1, 1, 1, 1, -1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Initialize weights to zeros [w1, w2, w0]
w = np.array([0, 0, 0])

# Initialize pocket weights
pocket_w = w.copy()

# Learning rate
eta = 1

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to count number of misclassifications
def count_misclassifications(X, y, w):
    predictions = predict(X, w)
    return np.sum(predictions != y)

# Function to calculate accuracy
def calculate_accuracy(X, y, w):
    predictions = predict(X, w)
    return np.mean(predictions == y) * 100

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, pocket_w=None, iteration=None, misclassified_idx=None, 
                          updated_idx=None, filename=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 3 else ('Class -1' if y[i] == -1 and i == 0 else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Highlight misclassified points
    if misclassified_idx is not None:
        for idx in misclassified_idx:
            plt.scatter(X[idx, 0], X[idx, 1], s=200, facecolors='none', 
                        edgecolors='green', linewidth=2, zorder=10)
    
    # Highlight point used for update
    if updated_idx is not None:
        plt.scatter(X[updated_idx, 0], X[updated_idx, 1], s=300, facecolors='none', 
                    edgecolors='purple', linewidth=3, zorder=11)
    
    # Plot the perceptron decision boundary if weights are not all zero
    if w is not None and not np.all(w == 0):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 5, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Current Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Current Decision Boundary')
            else:
                # If both w1 and w2 are close to zero, we don't have a well-defined boundary
                pass
    
    # Plot the pocket decision boundary
    if pocket_w is not None and not np.all(pocket_w == 0) and not np.array_equal(pocket_w, w):
        # Extract weights
        w1, w2, w0 = pocket_w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 5, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'b--', label='Pocket Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='b', linestyle='--', label='Pocket Decision Boundary')
            else:
                # If both w1 and w2 are close to zero, we don't have a well-defined boundary
                pass
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if iteration is not None:
        plt.title(f'Pocket Algorithm - Iteration {iteration}')
    else:
        plt.title('Pocket Algorithm - Final Decision Boundaries')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(-3, 5)
    plt.ylim(-3, 4)
    
    # Add legend
    plt.legend()
    
    # Add equation of the decision boundaries to the plot
    if w is not None and not np.all(w == 0):
        eq_str = f'Current weights: $[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]$\nError: {count_misclassifications(X_with_bias, y, w)}/{len(y)}'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    if pocket_w is not None and not np.all(pocket_w == 0):
        eq_str = f'Pocket weights: $[{pocket_w[0]:.2f}, {pocket_w[1]:.2f}, {pocket_w[2]:.2f}]$\nError: {count_misclassifications(X_with_bias, y, pocket_w)}/{len(y)}'
        plt.annotate(eq_str, xy=(0.05, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1))
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.close()

# Initial plot before training
initial_plot_file = os.path.join(save_dir, 'pocket_initial.png')
plot_decision_boundary(X, y, w, pocket_w, iteration=0, filename=initial_plot_file)

# Initialize variables for training
max_iterations = 4  # Run for exactly 4 iterations as required
converged = False
iteration_history = []

# Record initial state
current_misclassifications = count_misclassifications(X_with_bias, y, w)
best_misclassifications = current_misclassifications
print(f"Initial misclassifications: {current_misclassifications}/{len(y)}")
print(f"Initial accuracy: {calculate_accuracy(X_with_bias, y, w):.2f}%")

# Perform pocket algorithm
for iteration in range(1, max_iterations + 1):
    print(f"\nIteration {iteration}")
    print("-" * 50)
    print(f"Current weights: w = {w}")
    print(f"Pocket weights: pocket_w = {pocket_w}")
    
    # Find all misclassified examples
    predictions = predict(X_with_bias, w)
    misclassified_indices = np.where(predictions != y)[0]
    
    # Store misclassified indices for this iteration
    iteration_misclassified = list(misclassified_indices)
    
    print(f"Misclassified examples: {len(misclassified_indices)}/{len(y)}")
    
    if len(misclassified_indices) == 0:
        print("No misclassifications! Perceptron has converged.")
        converged = True
        break
    
    # Randomly select one misclassified example for update
    update_idx = np.random.choice(misclassified_indices)
    x_i = X_with_bias[update_idx]
    y_i = y[update_idx]
    
    print(f"Selected example {update_idx+1} for update: x = {X[update_idx]}, y = {y_i}")
    
    # Compute activation
    activation = np.dot(w, x_i)
    prediction = np.sign(activation)
    
    print(f"  Activation = w · x = {w} · {x_i} = {activation:.2f}")
    print(f"  Prediction = {prediction}, Actual = {y_i}")
    
    # Update weights
    w_old = w.copy()
    w = w + eta * y_i * x_i
    
    print(f"  Updating weights:")
    print(f"  w_new = w_old + η * y * x")
    print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
    print(f"  w_new = {w}")
    
    # Count misclassifications with new weights
    current_misclassifications = count_misclassifications(X_with_bias, y, w)
    print(f"  New misclassifications: {current_misclassifications}/{len(y)}")
    
    # Update pocket weights if better
    if current_misclassifications < best_misclassifications:
        print(f"  New weights are better! Updating pocket weights.")
        pocket_w = w.copy()
        best_misclassifications = current_misclassifications
    else:
        print(f"  Keeping current pocket weights as they're better.")
    
    # Generate plot for this update
    plot_file = os.path.join(save_dir, f'pocket_iteration_{iteration}.png')
    plot_decision_boundary(X, y, w, pocket_w, iteration=iteration, 
                           misclassified_idx=misclassified_indices, 
                           updated_idx=update_idx, filename=plot_file)
    
    # Store iteration information
    iteration_history.append({
        'iteration': iteration,
        'perceptron_weights': w.copy(),
        'pocket_weights': pocket_w.copy(),
        'misclassified': iteration_misclassified.copy(),
        'updated_idx': update_idx,
        'perceptron_error': current_misclassifications,
        'pocket_error': best_misclassifications
    })

# Final decision boundary
final_plot = os.path.join(save_dir, 'pocket_final.png')
plot_decision_boundary(X, y, w, pocket_w, filename=final_plot)

# Calculate and print final results
final_perceptron_accuracy = calculate_accuracy(X_with_bias, y, w)
final_pocket_accuracy = calculate_accuracy(X_with_bias, y, pocket_w)

print("\nPocket Algorithm Results")
print("=" * 50)
print(f"Dataset size: {len(X)} examples")
print(f"Iterations run: {max_iterations}")
print(f"Final perceptron weights: w = {w}")
print(f"Final pocket weights: pocket_w = {pocket_w}")
print(f"Final perceptron misclassifications: {count_misclassifications(X_with_bias, y, w)}/{len(y)}")
print(f"Final pocket misclassifications: {count_misclassifications(X_with_bias, y, pocket_w)}/{len(y)}")
print(f"Final perceptron accuracy: {final_perceptron_accuracy:.2f}%")
print(f"Final pocket accuracy: {final_pocket_accuracy:.2f}%")

# Save a table of the iteration history
with open(os.path.join(save_dir, 'pocket_iteration_history.txt'), 'w') as f:
    f.write("Iteration History\n")
    f.write("=" * 100 + "\n")
    f.write(f"{'Iter':^5} | {'Perceptron Weights':^30} | {'Pocket Weights':^30} | {'Perceptron Error':^15} | {'Pocket Error':^15}\n")
    f.write("-" * 100 + "\n")
    
    for hist in iteration_history:
        i = hist['iteration']
        pw = hist['perceptron_weights']
        pocket = hist['pocket_weights']
        pe = hist['perceptron_error']
        pocket_e = hist['pocket_error']
        
        f.write(f"{i:^5} | [{pw[0]:6.2f}, {pw[1]:6.2f}, {pw[2]:6.2f}] | [{pocket[0]:6.2f}, {pocket[1]:6.2f}, {pocket[2]:6.2f}] | {pe:^15} | {pocket_e:^15}\n")

# Plot evolution of errors across iterations
plt.figure(figsize=(10, 6))

# Prepare data for plotting
iterations = range(1, len(iteration_history) + 1)
perceptron_errors = [hist['perceptron_error'] for hist in iteration_history]
pocket_errors = [hist['pocket_error'] for hist in iteration_history]

# Plot the errors
plt.plot(iterations, perceptron_errors, 'ro-', label='Perceptron Error')
plt.plot(iterations, pocket_errors, 'bo-', label='Pocket Error')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassifications')
plt.title('Error Evolution - Perceptron vs Pocket')
plt.xticks(iterations)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the error evolution figure
plt.savefig(os.path.join(save_dir, 'pocket_error_evolution.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots and results saved to: {save_dir}") 