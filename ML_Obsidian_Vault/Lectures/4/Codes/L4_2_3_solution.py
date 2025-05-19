import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set random seed for reproducibility
np.random.seed(42)

# =======================================
# Part 1: Geometric interpretation of weight updates
# =======================================

def create_linearly_separable_data(n_samples=20):
    """Create a simple linearly separable dataset"""
    # Class 1: points in the first quadrant
    n_class1 = n_samples // 2
    X1 = np.random.rand(n_class1, 2) + 0.5  # Positive class in first quadrant
    
    # Class -1: points in the third quadrant
    n_class2 = n_samples - n_class1
    X2 = np.random.rand(n_class2, 2) * -1 - 0.5  # Negative class in third quadrant
    
    # Combine data
    X = np.vstack((X1, X2))
    y = np.array([1] * n_class1 + [-1] * n_class2)
    
    return X, y

def create_non_linearly_separable_data(n_samples=40):
    """Create a simple non-linearly separable dataset (XOR-like)"""
    n_per_cluster = n_samples // 4
    
    # Class 1: points in first and third quadrant
    X1_q1 = np.random.rand(n_per_cluster, 2) + 0.5  # First quadrant
    X1_q3 = np.random.rand(n_per_cluster, 2) * -1 - 0.5  # Third quadrant
    X1 = np.vstack((X1_q1, X1_q3))
    
    # Class -1: points in second and fourth quadrant
    X2_q2 = np.concatenate([np.random.rand(n_per_cluster, 1) * -1 - 0.5, 
                           np.random.rand(n_per_cluster, 1) + 0.5], axis=1)  # Second quadrant
    X2_q4 = np.concatenate([np.random.rand(n_per_cluster, 1) + 0.5, 
                           np.random.rand(n_per_cluster, 1) * -1 - 0.5], axis=1)  # Fourth quadrant
    X2 = np.vstack((X2_q2, X2_q4))
    
    # Combine data
    X = np.vstack((X1, X2))
    y = np.array([1] * (2 * n_per_cluster) + [-1] * (2 * n_per_cluster))
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    return X, y

# Create a linearly separable dataset
X_linear, y_linear = create_linearly_separable_data(n_samples=20)

# Add bias term to input features for perceptron
X_linear_with_bias = np.hstack((X_linear, np.ones((X_linear.shape[0], 1))))

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to draw decision boundary and show geometric interpretation
def plot_perceptron_geometry(X, y, w, title=None, misclassified_idx=None, 
                            prev_w=None, update_vector=None, filename=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == min(np.where(y == -1)[0]) else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Highlight misclassified points if provided
    if misclassified_idx is not None:
        for idx in misclassified_idx:
            plt.scatter(X[idx, 0], X[idx, 1], s=200, facecolors='none', 
                        edgecolors='green', linewidth=2, zorder=10, 
                        label='Misclassified' if idx == misclassified_idx[0] else None)
    
    # Plot the decision boundary if weights are not all zero
    if w is not None and not np.allclose(w[:2], 0):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-2, 2, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Decision Boundary')
    
    # Draw the weight vector as an arrow from the origin
    if w is not None and not np.allclose(w[:2], 0):
        # Scale weight vector for visualization
        scale_factor = 2.0 / (np.linalg.norm(w[:2]) + 1e-10)
        arrow_end = scale_factor * w[:2]
        plt.arrow(0, 0, arrow_end[0], arrow_end[1], head_width=0.1, 
                 head_length=0.2, fc='green', ec='green', label='Weight Vector')
        
        # Add text annotation for weight vector
        plt.annotate(f'$\\mathbf{{w}} = [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]^T$',
                    xy=(arrow_end[0], arrow_end[1]),
                    xytext=(arrow_end[0] + 0.1, arrow_end[1] + 0.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Show the weight update if provided
    if prev_w is not None and update_vector is not None:
        # Draw the previous weight vector
        scale_factor = 2.0 / (np.linalg.norm(prev_w[:2]) + 1e-10)
        prev_arrow_end = scale_factor * prev_w[:2]
        plt.arrow(0, 0, prev_arrow_end[0], prev_arrow_end[1], head_width=0.1, 
                 head_length=0.2, fc='orange', ec='orange', 
                 label='Previous Weight Vector', alpha=0.6)
        
        # Draw the update vector
        plt.arrow(prev_arrow_end[0], prev_arrow_end[1], 
                 arrow_end[0] - prev_arrow_end[0], 
                 arrow_end[1] - prev_arrow_end[1], 
                 head_width=0.1, head_length=0.2, fc='purple', ec='purple', 
                 label='Update Vector', alpha=0.8)
        
        # Add explanation text about the update
        if misclassified_idx is not None and len(misclassified_idx) > 0:
            idx = misclassified_idx[0]
            plt.annotate(f'Update: $\\eta \\cdot y_i \\cdot \\mathbf{{x}}_i = {y[idx]} \\cdot [{X[idx, 0]:.2f}, {X[idx, 1]:.2f}]$',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if title:
        plt.title(title)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add origin point
    plt.scatter(0, 0, color='black', s=50, label='Origin')
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt

# Train perceptron and visualize geometric interpretation
def train_perceptron_with_visualization(X, y, learning_rate=1.0, max_iterations=10):
    # Add bias term to input features
    X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
    
    # Initialize weights to zeros [w1, w2, w0]
    w = np.zeros(3)
    
    # Initialize variables for training
    iteration = 0
    converged = False
    history = []
    
    # Initial plot before training
    plot_perceptron_geometry(X, y, w, title="Initial State (Weights = 0)", 
                           filename=os.path.join(save_dir, 'perceptron_geometric_initial.png'))
    
    while not converged and iteration < max_iterations:
        iteration += 1
        misclassified = []
        made_update = False
        
        print(f"\nIteration {iteration}")
        print("-" * 50)
        print(f"Current weights: w = {w}")
        
        # Go through all samples
        for i in range(len(X_with_bias)):
            x_i = X_with_bias[i]
            y_i = y[i]
            
            # Compute activation and prediction
            activation = np.dot(w, x_i)
            prediction = np.sign(activation)
            
            # Check if misclassified
            if prediction != y_i:
                misclassified.append(i)
                
                # Keep previous weights for visualization
                prev_w = w.copy()
                
                # Calculate update vector
                update_vector = learning_rate * y_i * x_i
                
                # Update weights
                w = w + update_vector
                
                print(f"Sample {i+1}: x = {X[i]}, y = {y_i}")
                print(f"  Activation = w · x = {activation:.2f}")
                print(f"  Prediction = {prediction}, Actual = {y_i}")
                print(f"  Misclassified! Updating weights:")
                print(f"  w_new = w_old + η * y * x")
                print(f"  w_new = {prev_w} + {learning_rate} * {y_i} * {x_i}")
                print(f"  w_new = {w}")
                
                # Generate plot showing the geometric interpretation of this update
                plot_file = os.path.join(save_dir, f'perceptron_geometric_iter{iteration}_sample{i+1}.png')
                plot_perceptron_geometry(X, y, w, 
                                       title=f"Iteration {iteration}, Sample {i+1}", 
                                       misclassified_idx=[i],
                                       prev_w=prev_w,
                                       update_vector=update_vector,
                                       filename=plot_file)
                
                made_update = True
                break  # Move to next iteration after first update
        
        if not made_update:
            converged = True
            print("\nConverged! No misclassifications.")
            
        # Store history
        history.append({
            'iteration': iteration,
            'weights': w.copy(),
            'misclassified': misclassified.copy(),
            'converged': converged
        })
    
    # Final decision boundary with weight vector
    final_plot = plot_perceptron_geometry(X, y, w, 
                                        title="Final Decision Boundary", 
                                        filename=os.path.join(save_dir, 'perceptron_geometric_final.png'))
    
    return w, history

# Train perceptron on the linearly separable dataset
print("=== Training on linearly separable dataset ===")
final_w_linear, history_linear = train_perceptron_with_visualization(X_linear, y_linear)

# =======================================
# Part 2: Non-linearly separable case
# =======================================

# Create a non-linearly separable dataset
X_nonlinear, y_nonlinear = create_non_linearly_separable_data(n_samples=40)

# Visualize the non-linearly separable dataset
plt.figure(figsize=(10, 8))
for i in range(len(X_nonlinear)):
    marker = 'o' if y_nonlinear[i] == 1 else 'x'
    color = 'blue' if y_nonlinear[i] == 1 else 'red'
    label = 'Class 1' if y_nonlinear[i] == 1 and i == min(np.where(y_nonlinear == 1)[0]) else \
           ('Class -1' if y_nonlinear[i] == -1 and i == min(np.where(y_nonlinear == -1)[0]) else None)
    plt.scatter(X_nonlinear[i, 0], X_nonlinear[i, 1], marker=marker, color=color, s=100, label=label)

plt.title("Non-Linearly Separable Dataset")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.legend()
plt.savefig(os.path.join(save_dir, 'non_linearly_separable_dataset.png'), dpi=300, bbox_inches='tight')

# Train perceptron on non-linearly separable dataset with limited iterations
print("\n=== Training on non-linearly separable dataset ===")
max_iterations_nonlinear = 20  # Limit iterations for non-convergent case
X_nonlinear_with_bias = np.hstack((X_nonlinear, np.ones((X_nonlinear.shape[0], 1))))

# Initialize weights to zeros [w1, w2, w0]
w_nonlinear = np.zeros(3)

# Initialize variables for training
iteration = 0
converged = False
history_nonlinear = []
all_misclassified = []

# Create a plot to show the "cycling" behavior on non-linearly separable data
plot_perceptron_geometry(X_nonlinear, y_nonlinear, w_nonlinear, 
                      title="Initial State - Non-Linear Dataset", 
                      filename=os.path.join(save_dir, 'perceptron_nonlinear_initial.png'))

while not converged and iteration < max_iterations_nonlinear:
    iteration += 1
    misclassified = []
    made_update = False
    
    print(f"\nIteration {iteration}")
    print("-" * 50)
    print(f"Current weights: w = {w_nonlinear}")
    
    # Check each sample
    for i in range(len(X_nonlinear_with_bias)):
        x_i = X_nonlinear_with_bias[i]
        y_i = y_nonlinear[i]
        
        # Compute activation
        activation = np.dot(w_nonlinear, x_i)
        prediction = np.sign(activation)
        
        # Check if misclassified
        if prediction != y_i:
            misclassified.append(i)
            
            # Keep previous weights for visualization
            prev_w = w_nonlinear.copy()
            
            # Update weights
            w_nonlinear = w_nonlinear + y_i * x_i
            
            print(f"Sample {i+1}: x = {X_nonlinear[i]}, y = {y_i}")
            print(f"  Activation = w · x = {activation:.2f}")
            print(f"  Prediction = {prediction}, Actual = {y_i}")
            print(f"  Misclassified! Updating weights:")
            print(f"  w_new = w_old + η * y * x")
            print(f"  w_new = {prev_w} + 1 * {y_i} * {x_i}")
            print(f"  w_new = {w_nonlinear}")
            
            if iteration % 5 == 0 or iteration == 1:  # Save plots every 5 iterations to avoid too many files
                # Generate plot
                plot_file = os.path.join(save_dir, f'perceptron_nonlinear_iter{iteration}_sample{i+1}.png')
                plot_perceptron_geometry(X_nonlinear, y_nonlinear, w_nonlinear, 
                                      title=f"Non-Linear Dataset - Iteration {iteration}, Sample {i+1}", 
                                      misclassified_idx=[i],
                                      prev_w=prev_w,
                                      update_vector=y_i * x_i,
                                      filename=plot_file)
            
            made_update = True
            all_misclassified.append(i)
            break  # Move to next iteration after first update
    
    if not made_update:
        converged = True
        print("\nConverged! This is unexpected for a non-linearly separable dataset.")
    
    # Store history
    history_nonlinear.append({
        'iteration': iteration,
        'weights': w_nonlinear.copy(),
        'misclassified': misclassified.copy(),
        'converged': converged
    })
    
    # Check if reached max iterations
    if iteration == max_iterations_nonlinear:
        print(f"\nReached maximum iterations ({max_iterations_nonlinear}). Algorithm did not converge.")
        
        # Generate final state plot
        plot_file = os.path.join(save_dir, f'perceptron_nonlinear_final.png')
        plot_perceptron_geometry(X_nonlinear, y_nonlinear, w_nonlinear, 
                              title=f"Non-Linear Dataset - Final State (No Convergence)", 
                              misclassified_idx=misclassified,
                              filename=plot_file)

# =======================================
# Part 3: Create Weight Vector Evolution Plots
# =======================================

def plot_weight_vector_evolution(X, y, history, title, filename_prefix):
    """Create a series of plots showing the evolution of the weight vector"""
    # Create a figure showing all iterations on separate plots
    num_plots = min(6, len(history))  # Show at most 6 iterations
    iterations_to_show = [0]  # Always show initial state (iteration 0)
    
    # Add evenly spaced iterations
    if len(history) > 1:
        step = max(1, len(history) // (num_plots - 1))
        iterations_to_show.extend(range(step, len(history), step))
        # Make sure to include the last iteration
        if iterations_to_show[-1] != len(history) - 1:
            iterations_to_show.append(len(history) - 1)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, iter_idx in enumerate(iterations_to_show[:6]):  # Limit to 6 plots
        ax = axes[i]
        
        # For initial state, use zeros
        if iter_idx == 0:
            w = np.zeros(3)
            iter_title = "Initial State"
        else:
            w = history[iter_idx - 1]['weights'].copy()
            iter_title = f"Iteration {iter_idx}"
        
        # Plot data points
        for j in range(len(X)):
            marker = 'o' if y[j] == 1 else 'x'
            color = 'blue' if y[j] == 1 else 'red'
            ax.scatter(X[j, 0], X[j, 1], marker=marker, color=color, s=50)
        
        # Plot the weight vector as an arrow if not all zeros
        if not np.allclose(w[:2], 0):
            # Scale weight vector for visualization
            scale_factor = 2.0 / (np.linalg.norm(w[:2]) + 1e-10)
            arrow_end = scale_factor * w[:2]
            ax.arrow(0, 0, arrow_end[0], arrow_end[1], head_width=0.1, 
                   head_length=0.2, fc='green', ec='green')
        
        # Plot the decision boundary if weights allow
        if not np.allclose(w[:2], 0):
            w1, w2, w0 = w
            x1_range = np.linspace(-2, 2, 100)
            
            if abs(w2) > 1e-10:
                x2_boundary = (-w1 * x1_range - w0) / w2
                ax.plot(x1_range, x2_boundary, 'g-')
        
        # Add text annotation for weights
        ax.text(0.05, 0.95, f"w = [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]", 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
        
        # Add titles and grid
        ax.set_title(iter_title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add x and y axis labels for bottom and left plots
        if i >= 3:  # Bottom row
            ax.set_xlabel('$x_1$')
        if i % 3 == 0:  # Left column
            ax.set_ylabel('$x_2$')
    
    # Add a main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    plt.savefig(f"{filename_prefix}_evolution.png", dpi=300, bbox_inches='tight')
    
    return fig

# Create weight vector evolution plots
plot_weight_vector_evolution(X_linear, y_linear, history_linear, 
                           "Perceptron Weight Vector Evolution - Linear Dataset",
                           os.path.join(save_dir, 'perceptron_linear'))

plot_weight_vector_evolution(X_nonlinear, y_nonlinear, history_nonlinear,
                           "Perceptron Weight Vector Evolution - Non-Linear Dataset",
                           os.path.join(save_dir, 'perceptron_nonlinear'))

print(f"\nPlots and images saved to: {save_dir}")
print("\nSummary:")
print(f"Linearly separable dataset: {'Converged' if history_linear[-1]['converged'] else 'Did not converge'}")
print(f"Non-linearly separable dataset: {'Converged' if history_nonlinear[-1]['converged'] else 'Did not converge'}") 