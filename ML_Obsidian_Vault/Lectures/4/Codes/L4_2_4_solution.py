import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset
X = np.array([
    [1, 1],   # Class 1
    [2, 2],   # Class 1
    [1, -1],  # Class -1
    [-1, 1],  # Class -1
    [-2, -1]  # Class -1
])

y = np.array([1, 1, -1, -1, -1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Initialize weights to zeros [w1, w2, w0]
w = np.array([0, 0, 0])

# Learning rate
eta = 1

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, iteration=None, misclassified_idx=None, filename=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
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
        x1_range = np.linspace(-3, 3, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            x1_boundary = -w0 / w1
            plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Decision Boundary')
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if iteration is not None:
        plt.title(f'Perceptron Learning - Iteration {iteration}')
    else:
        plt.title('Perceptron Learning - Final Decision Boundary')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set limits
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    # Add legend
    plt.legend()
    
    # Add equation of the decision boundary to the plot
    if w is not None and not np.all(w == 0):
        eq_str = f'$w_1 x_1 + w_2 x_2 + w_0 = 0$\n${w[0]:.2f} x_1 + {w[1]:.2f} x_2 + {w[2]:.2f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt

# Initial plot before training
initial_plot = plot_decision_boundary(X, y, w, iteration=0)
initial_plot.savefig(os.path.join(save_dir, 'perceptron_initial.png'), dpi=300, bbox_inches='tight')

# Initialize variables for training
max_iterations = 10
iteration = 0
converged = False
iteration_history = []

# Perform perceptron learning
while not converged and iteration < max_iterations:
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
        prediction = np.sign(activation)
        
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
            plot_file = os.path.join(save_dir, f'perceptron_iteration_{iteration}_sample_{i+1}.png')
            plot_decision_boundary(X, y, w, iteration=iteration, 
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
        'misclassified': misclassified.copy()
    })
    
    # Check if converged (no misclassifications)
    if not made_update:
        converged = True
        print("\nConverged! No misclassifications detected.")

# Final decision boundary
final_plot = plot_decision_boundary(X, y, w)
final_plot.savefig(os.path.join(save_dir, 'perceptron_final.png'), dpi=300, bbox_inches='tight')

# Print summary
print("\nPerceptron Learning Summary")
print("=" * 50)
print(f"Dataset:")
for i in range(len(X)):
    print(f"Sample {i+1}: x = {X[i]}, y = {y[i]}")
print("\nTraining:")
print(f"Initial weights: w = [0, 0, 0]")
print(f"Learning rate: η = {eta}")
print(f"Converged in {iteration} iterations")
print(f"Final weights: w = {w}")

# Decision boundary equation
w1, w2, w0 = w
print("\nFinal Decision Boundary:")
print(f"{w1}*x1 + {w2}*x2 + {w0} = 0")

# Save a figure with all iteration boundaries
plt.figure(figsize=(12, 10))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot all decision boundaries
color_map = plt.cm.rainbow(np.linspace(0, 1, len(iteration_history) + 1))
x1_range = np.linspace(-3, 3, 100)

# Plot initial boundary (horizontal line at y=0 since w=[0,0,0])
plt.axhline(y=0, color=color_map[0], linestyle='--', alpha=0.5, label='Initial (w=[0,0,0])')

# Plot decision boundaries for each iteration
for idx, hist in enumerate(iteration_history):
    w_iter = hist['weights']
    w1, w2, w0 = w_iter
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        plt.plot(x1_range, x2_boundary, color=color_map[idx+1], 
                linestyle='-', alpha=0.6, 
                label=f'Iteration {idx+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Perceptron Learning - Decision Boundary Evolution')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the evolution figure
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'perceptron_evolution.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}") 