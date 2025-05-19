import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset for Question 10
X = np.array([
    [1, 2],   # Class 1
    [2, 1],   # Class 1
    [3, 3],   # Class 1
    [6, 4],   # Class -1
    [5, 6],   # Class -1
    [7, 5]    # Class -1
])

y = np.array([1, 1, 1, -1, -1, -1])

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
        x1_range = np.linspace(0, 8, 100)
        
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
        plt.title(f'Perceptron Learning - Iteration {iteration}')
    else:
        plt.title('Perceptron Learning - Final Decision Boundary')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    
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
    
    plt.close()  # Close the figure to avoid displaying it in notebooks

# Initial plot before training - just plotting the data points
initial_plot_file = os.path.join(save_dir, 'perceptron_initial.png')
plot_decision_boundary(X, y, None, filename=initial_plot_file)
print(f"Saved initial plot to {initial_plot_file}")

# Initialize variables for training
iteration = 0
updates_done = 0
iteration_history = []

# Perform perceptron learning until we have 2 updates
print("\nStarting Perceptron Algorithm")
print("=" * 50)

while updates_done < 2:
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
        prediction = 1 if activation > 0 else -1 if activation < 0 else 0
        
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
            plot_file = os.path.join(save_dir, f'perceptron_update_{updates_done+1}_iter_{iteration}_sample_{i+1}.png')
            plot_decision_boundary(X, y, w, iteration=iteration, 
                                  misclassified_idx=[i], filename=plot_file)
            print(f"Saved update plot to {plot_file}")
            
            made_update = True
            updates_done += 1
            
            # Store iteration information
            iteration_history.append({
                'iteration': iteration,
                'sample': i+1,
                'weights': w.copy(),
                'misclassified_idx': i
            })
            
            if updates_done >= 2:
                break  # Exit the loop after 2 updates
            
            break  # Move to next iteration after first update in current iteration
        else:
            print("  Correctly classified!")
    
    if not made_update:
        print("\nAll samples correctly classified in this iteration.")
        break
    
    if updates_done >= 2:
        break

# Final decision boundary plot (after full convergence)
# Let's run the algorithm until convergence to get the final decision boundary
w_final = w.copy()  # Save the weights after two updates
w = np.array([0, 0, 0])  # Reset weights
converged = False
iteration = 0
max_iterations = 100

print("\nRunning perceptron until convergence for final decision boundary...")
while not converged and iteration < max_iterations:
    iteration += 1
    misclassified = []
    made_update = False
    
    # Check each sample for misclassification
    for i in range(len(X_with_bias)):
        x_i = X_with_bias[i]
        y_i = y[i]
        
        # Compute activation
        activation = np.dot(w, x_i)
        prediction = 1 if activation > 0 else -1 if activation < 0 else 0
        
        # Check if misclassified
        if prediction != y_i:
            # Update weights
            w = w + eta * y_i * x_i
            made_update = True
            break
    
    if not made_update:
        converged = True
        print(f"Converged after {iteration} iterations!")

# Plot final decision boundary
final_plot_file = os.path.join(save_dir, 'perceptron_final.png')
plot_decision_boundary(X, y, w, filename=final_plot_file)
print(f"Saved final decision boundary plot to {final_plot_file}")

# Plot circular decision boundary for part 4
plt.figure(figsize=(10, 8))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 3 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot an arbitrary circular decision boundary for illustration
circle = plt.Circle((4, 3.5), 2.5, fill=False, color='purple', linestyle='--', 
                    linewidth=2, label='Circular Boundary')
plt.gca().add_patch(circle)

plt.title('Perceptron Cannot Learn Circular Decision Boundary')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 8)
plt.ylim(0, 8)
plt.legend()

# Add an explanation text
plt.annotate("A perceptron can only create a linear decision boundary,\n"
             "not a non-linear boundary like a circle.",
             xy=(0.05, 0.05), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

circular_plot_file = os.path.join(save_dir, 'perceptron_circular.png')
plt.savefig(circular_plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved circular boundary illustration to {circular_plot_file}")

# Evolution plot showing all boundaries
plt.figure(figsize=(12, 10))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 3 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot decision boundaries
x1_range = np.linspace(0, 8, 100)

# Plot initial boundary (undefined since w=[0,0,0])
plt.annotate("Initial: w=[0,0,0] (boundary undefined)", xy=(0.05, 0.80), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# Plot boundaries for each update
colors = ['green', 'orange']
for idx, hist in enumerate(iteration_history):
    w_iter = hist['weights']
    w1, w2, w0 = w_iter
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        plt.plot(x1_range, x2_boundary, color=colors[idx], 
                linestyle='-', linewidth=2,
                label=f'Update {idx+1} (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

# Plot final converged boundary
w1, w2, w0 = w
if abs(w2) > 1e-10:
    x2_boundary = (-w1 * x1_range - w0) / w2
    plt.plot(x1_range, x2_boundary, color='purple', 
            linestyle='-', linewidth=2, label=f'Final (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

plt.title('Perceptron Learning - Decision Boundary Evolution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 8)
plt.ylim(0, 8)
plt.legend(loc='upper left')

evolution_plot_file = os.path.join(save_dir, 'perceptron_evolution.png')
plt.savefig(evolution_plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved evolution plot to {evolution_plot_file}")

# Print summary of results for the explanation document
print("\nPerceptron Learning Summary")
print("=" * 50)
print(f"Dataset:")
for i in range(len(X)):
    print(f"Sample {i+1}: x = {X[i]}, y = {y[i]}")
print("\nTraining:")
print(f"Initial weights: w = [0, 0, 0]")
print(f"Learning rate: η = {eta}")

print("\nUpdate History:")
for idx, hist in enumerate(iteration_history):
    print(f"Update {idx+1}:")
    print(f"  Iteration: {hist['iteration']}")
    print(f"  Sample: {hist['sample']} (x={X[hist['misclassified_idx']]}, y={y[hist['misclassified_idx']]})")
    print(f"  Updated weights: w = {hist['weights']}")

print("\nFinal weights after convergence:")
print(f"w = {w}")
print(f"Decision boundary equation: {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w[2]:.2f} = 0")

print(f"\nAll plots saved to: {save_dir}") 