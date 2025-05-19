import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset
X = np.array([
    [1, 1],    # Class 1
    [3, 1],    # Class 1
    [2, 4],    # Class 1
    [-1, 1],   # Class -1
    [0, -2],   # Class -1
    [-2, -1]   # Class -1
])

y = np.array([1, 1, 1, -1, -1, -1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, eta, iteration=None, misclassified_idx=None, filename=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 3 else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Highlight misclassified points if provided
    if misclassified_idx is not None:
        for idx in misclassified_idx:
            plt.scatter(X[idx, 0], X[idx, 1], s=200, facecolors='none', 
                        edgecolors='green', linewidth=2, zorder=10)
    
    # Plot the decision boundary if weights are not all zero
    if w is not None and not np.all(w == 0):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 5, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Decision Boundary')
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if iteration is not None:
        plt.title(f'Perceptron Learning - $\\eta={eta}$ - Iteration {iteration}')
    else:
        plt.title(f'Perceptron Learning - $\\eta={eta}$ - Final Decision Boundary')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(-3, 4)
    plt.ylim(-3, 5)
    
    # Add legend
    plt.legend()
    
    # Add equation of the decision boundary to the plot
    if w is not None and not np.all(w == 0):
        eq_str = f'$w_1 x_1 + w_2 x_2 + w_0 = 0$\n${w[0]:.2f} x_1 + {w[1]:.2f} x_2 + {w[2]:.2f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Add weight vector information
    w_str = f'$\\mathbf{{w}} = [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]^T$'
    plt.annotate(w_str, xy=(0.05, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt

def run_perceptron(eta, max_iterations=10, early_stop=3):
    """
    Run the perceptron algorithm with the given learning rate
    
    Parameters:
    eta: learning rate
    max_iterations: maximum number of iterations
    early_stop: number of iterations to show before stopping
    
    Returns:
    List of iteration history dictionaries
    """
    # Initialize weights to zeros
    w = np.array([0.0, 0.0, 0.0])
    
    # Initial plot before training
    initial_plot = plot_decision_boundary(X, y, w, eta, iteration=0)
    initial_plot.savefig(os.path.join(save_dir, f'perceptron_eta_{eta}_initial.png'), dpi=300, bbox_inches='tight')
    
    # Initialize variables for training
    iteration = 0
    converged = False
    iteration_history = []
    
    print(f"\n{'='*60}")
    print(f"PERCEPTRON LEARNING WITH η = {eta}")
    print(f"{'='*60}")
    
    # Perform perceptron learning
    while not converged and iteration < max_iterations and iteration < early_stop:
        iteration += 1
        misclassified = []
        made_update = False
        
        print(f"\nIteration {iteration}")
        print("-" * 50)
        print(f"Current weights: w = {w}")
        
        # Check each sample for misclassification
        for i in range(len(X_with_bias)):
            x_i = X_with_bias[i]
            y_i = y[i]
            
            # Compute activation
            activation = np.dot(w, x_i)
            prediction = np.sign(activation) if activation != 0 else 0
            
            print(f"Sample {i+1}: x = {X[i]}, y = {y_i}")
            print(f"  Activation = w · x = {w} · {x_i} = {activation:.2f}")
            print(f"  Prediction = {prediction}, Actual = {y_i}")
            
            # Check if misclassified
            if prediction != y_i:
                misclassified.append(i)
                
                # Update weights
                w_old = w.copy()
                w = w + eta * y_i * x_i
                
                print(f"  Misclassified! Updating weights:")
                print(f"  w_new = w_old + η * y * x")
                print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
                print(f"  w_new = {w}")
                
                # Generate plot for this update
                plot_file = os.path.join(save_dir, f'perceptron_eta_{eta}_iteration_{iteration}_sample_{i+1}.png')
                plot_decision_boundary(X, y, w, eta, iteration=iteration, 
                                      misclassified_idx=[i], filename=plot_file)
                
                made_update = True
                break  # Move to next iteration after first update
            else:
                print("  Correctly classified!")
        
        print(f"End of iteration {iteration}")
        print(f"Updated weights: w = {w}")
        
        # Store iteration information
        iteration_history.append({
            'iteration': iteration,
            'weights': w.copy(),
            'misclassified': misclassified.copy(),
            'eta': eta
        })
        
        # Check if converged (no misclassifications)
        if not made_update:
            converged = True
            print("\nConverged! No misclassifications detected.")
    
    # Generate final plot if not covered by early stopping
    if iteration < max_iterations and converged:
        final_plot = plot_decision_boundary(X, y, w, eta)
        final_plot.savefig(os.path.join(save_dir, f'perceptron_eta_{eta}_final.png'), dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\nPerceptron Learning Summary")
    print("-" * 50)
    print(f"Learning rate: η = {eta}")
    if iteration < max_iterations and converged:
        print(f"Converged in {iteration} iterations")
    else:
        print(f"Stopped after {iteration} iterations (not converged)")
    print(f"Final weights: w = {w}")
    
    return iteration_history

# Run perceptron with different learning rates
history_01 = run_perceptron(eta=0.1)
history_20 = run_perceptron(eta=2.0)

# Extract weight vectors for each learning rate
weights_01 = np.array([hist['weights'] for hist in history_01])
weights_20 = np.array([hist['weights'] for hist in history_20])

# Plot 2D projections of the weight trajectories
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# w1 vs w2
axs[0, 0].plot(weights_01[:, 0], weights_01[:, 1], 'b-o', linewidth=2, markersize=8, label='$\\eta=0.1$')
axs[0, 0].plot(weights_20[:, 0], weights_20[:, 1], 'r-o', linewidth=2, markersize=8, label='$\\eta=2.0$')
axs[0, 0].set_xlabel('$w_1$')
axs[0, 0].set_ylabel('$w_2$')
axs[0, 0].set_title('$w_1$ vs $w_2$')
axs[0, 0].grid(True)
axs[0, 0].legend()
axs[0, 0].plot(0, 0, 'ko', markersize=8)  # Origin

# w1 vs w0
axs[0, 1].plot(weights_01[:, 0], weights_01[:, 2], 'b-o', linewidth=2, markersize=8, label='$\\eta=0.1$')
axs[0, 1].plot(weights_20[:, 0], weights_20[:, 2], 'r-o', linewidth=2, markersize=8, label='$\\eta=2.0$')
axs[0, 1].set_xlabel('$w_1$')
axs[0, 1].set_ylabel('$w_0$')
axs[0, 1].set_title('$w_1$ vs $w_0$')
axs[0, 1].grid(True)
axs[0, 1].legend()
axs[0, 1].plot(0, 0, 'ko', markersize=8)  # Origin

# w2 vs w0
axs[1, 0].plot(weights_01[:, 1], weights_01[:, 2], 'b-o', linewidth=2, markersize=8, label='$\\eta=0.1$')
axs[1, 0].plot(weights_20[:, 1], weights_20[:, 2], 'r-o', linewidth=2, markersize=8, label='$\\eta=2.0$')
axs[1, 0].set_xlabel('$w_2$')
axs[1, 0].set_ylabel('$w_0$')
axs[1, 0].set_title('$w_2$ vs $w_0$')
axs[1, 0].grid(True)
axs[1, 0].legend()
axs[1, 0].plot(0, 0, 'ko', markersize=8)  # Origin

# Show evolution of decision boundaries
axs[1, 1].axis('off')  # Remove the empty subplot
axs[1, 1].text(0.5, 0.5, 'Learning Rate Comparison\n\n' + 
              f'$\\eta=0.1$ weights after 3 iterations:\n{weights_01[-1]}\n\n' + 
              f'$\\eta=2.0$ weights after 3 iterations:\n{weights_20[-1]}', 
              horizontalalignment='center',
              verticalalignment='center',
              fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_trajectories_2d.png'), dpi=300, bbox_inches='tight')

# Save decision boundary evolution
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Plot data points for both
for ax_idx, ax in enumerate(axs):
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 3 else None)
        ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 5)

# Plot decision boundaries for η=0.1
axs[0].set_title('Decision Boundary Evolution, $\\eta=0.1$')
x1_range = np.linspace(-3, 4, 100)

color_list = ['r', 'g', 'b']
for i, hist in enumerate(history_01):
    w = hist['weights']
    w1, w2, w0 = w
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        axs[0].plot(x1_range, x2_boundary, color=color_list[i % len(color_list)], 
                   linestyle='-', alpha=0.7, 
                   label=f'Iteration {i+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')
    elif abs(w1) > 1e-10:
        x1_boundary = -w0 / w1
        axs[0].axvline(x=x1_boundary, color=color_list[i % len(color_list)], 
                      linestyle='-', alpha=0.7, 
                      label=f'Iteration {i+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

# Plot decision boundaries for η=2.0
axs[1].set_title('Decision Boundary Evolution, $\\eta=2.0$')
for i, hist in enumerate(history_20):
    w = hist['weights']
    w1, w2, w0 = w
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        axs[1].plot(x1_range, x2_boundary, color=color_list[i % len(color_list)], 
                   linestyle='-', alpha=0.7, 
                   label=f'Iteration {i+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')
    elif abs(w1) > 1e-10:
        x1_boundary = -w0 / w1
        axs[1].axvline(x=x1_boundary, color=color_list[i % len(color_list)], 
                      linestyle='-', alpha=0.7, 
                      label=f'Iteration {i+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

# Add legends
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundary_evolution.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}")
print("\nComparison of Weight Changes:")
print("-" * 50)
print(f"{'Iteration':>10} | {'Learning Rate η=0.1':^25} | {'Learning Rate η=2.0':^25}")
print("-" * 70)
for i in range(len(history_01)):
    w_01 = history_01[i]['weights']
    w_20 = history_20[i]['weights']
    print(f"{i+1:>10} | [{w_01[0]:6.2f}, {w_01[1]:6.2f}, {w_01[2]:6.2f}] | [{w_20[0]:6.2f}, {w_20[1]:6.2f}, {w_20[2]:6.2f}]")

# Calculate weight change magnitudes
print("\nMagnitude of Weight Changes:")
print("-" * 50)
print(f"{'Iteration':>10} | {'η=0.1':^10} | {'η=2.0':^10} | {'Ratio (η=2.0/η=0.1)':^20}")
print("-" * 60)
for i in range(1, len(history_01)):
    w_01_prev = history_01[i-1]['weights']
    w_01_curr = history_01[i]['weights']
    w_20_prev = history_20[i-1]['weights']
    w_20_curr = history_20[i]['weights']
    
    delta_w_01 = np.linalg.norm(w_01_curr - w_01_prev)
    delta_w_20 = np.linalg.norm(w_20_curr - w_20_prev)
    ratio = delta_w_20 / delta_w_01 if delta_w_01 > 0 else float('inf')
    
    print(f"{i:>10} | {delta_w_01:10.2f} | {delta_w_20:10.2f} | {ratio:20.2f}") 