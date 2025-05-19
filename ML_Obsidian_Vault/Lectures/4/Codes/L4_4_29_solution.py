import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset from the problem
X = np.array([
    [1, 1],   # Class 1
    [2, 2],   # Class 1
    [3, 1],   # Class 1
    [2, 3],   # Class -1
    [1, 2],   # Class -1
    [3, 3]    # Class 1
])

y = np.array([1, 1, 1, -1, -1, 1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Initialize weights to zeros [w1, w2, w0]
w = np.array([0, 0, 0])

# Initialize pocket weights
pocket_w = w.copy()
pocket_errors = len(X)  # Start with maximum possible errors

# Learning rate
eta = 1

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to count errors
def count_errors(X, y, w):
    predictions = predict(X, w)
    return np.sum(predictions != y)

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, pocket_w, iteration=None, misclassified_idx=None, filename=None):
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
    
    # Plot the pocket decision boundary (solid line)
    if pocket_w is not None and not np.all(pocket_w == 0):
        # Extract weights
        w1, w2, w0 = pocket_w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(0, 4, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', linewidth=2, label='Pocket Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', linewidth=2, label='Pocket Decision Boundary')
    
    # Plot current decision boundary (dashed line) if different from pocket
    if w is not None and not np.all(w == 0) and not np.array_equal(w, pocket_w):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(0, 4, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'r--', linewidth=1.5, label='Current Decision Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='r', linestyle='--', linewidth=1.5, label='Current Decision Boundary')
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if iteration is not None:
        plt.title(f'Pocket Algorithm - Iteration {iteration}')
    else:
        plt.title('Pocket Algorithm - Final Decision Boundary')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    
    # Add legend
    plt.legend()
    
    # Add equations of the decision boundaries to the plot
    y_pos = 0.95
    if pocket_w is not None and not np.all(pocket_w == 0):
        pocket_eq = f'Pocket: ${pocket_w[0]:.2f} x_1 + {pocket_w[1]:.2f} x_2 + {pocket_w[2]:.2f} = 0$'
        plt.annotate(pocket_eq, xy=(0.05, y_pos), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=1))
        y_pos -= 0.07
    
    if w is not None and not np.all(w == 0) and not np.array_equal(w, pocket_w):
        current_eq = f'Current: ${w[0]:.2f} x_1 + {w[1]:.2f} x_2 + {w[2]:.2f} = 0$'
        plt.annotate(current_eq, xy=(0.05, y_pos), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1))
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt

# Initial plot before training
initial_plot = plot_decision_boundary(X, y, w, pocket_w, iteration=0)
initial_plot.savefig(os.path.join(save_dir, 'pocket_initial.png'), dpi=300, bbox_inches='tight')

# Initialize variables for training
max_iterations = 3  # We are asked to run 3 iterations
iteration = 0
iteration_history = []

# Perform pocket algorithm
for iteration in range(1, max_iterations + 1):
    print(f"\nIteration {iteration}")
    print("-" * 50)
    print(f"Current weights: w = {w}")
    print(f"Pocket weights: pocket_w = {pocket_w}")
    print(f"Pocket errors: {pocket_errors}")
    
    # Keep track of misclassified points
    all_misclassified = []
    
    # Check each sample and potentially update weights
    for i in range(len(X_with_bias)):
        x_i = X_with_bias[i]
        y_i = y[i]
        
        # Compute activation
        activation = np.dot(w, x_i)
        prediction = np.sign(activation)
        
        print(f"\nSample {i+1}: x = {X[i]}, y = {y_i}")
        print(f"  Activation = w · x = {w} · {x_i} = {activation:.2f}")
        print(f"  Prediction = {prediction}, Actual = {y_i}")
        
        # Check if misclassified
        if prediction != y_i:
            all_misclassified.append(i)
            
            # Update weights
            w_old = w.copy()
            w = w + eta * y_i * x_i
            
            print(f"  Misclassified! Updating weights:")
            print(f"  w_new = w_old + η * y * x")
            print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
            print(f"  w_new = {w}")
            
            # Generate plot for this update
            plot_file = os.path.join(save_dir, f'pocket_iteration_{iteration}_sample_{i+1}.png')
            plot_decision_boundary(X, y, w, pocket_w, iteration=iteration, 
                                  misclassified_idx=[i], filename=plot_file)
            
            # Check if better than pocket weights
            current_errors = count_errors(X_with_bias, y, w)
            print(f"  Current errors: {current_errors}, Pocket errors: {pocket_errors}")
            
            if current_errors < pocket_errors:
                pocket_w = w.copy()
                pocket_errors = current_errors
                print(f"  Better solution found! Updating pocket weights:")
                print(f"  pocket_w = {pocket_w}")
                print(f"  pocket_errors = {pocket_errors}")
                
                # Generate plot for pocket update
                plot_file = os.path.join(save_dir, f'pocket_iteration_{iteration}_sample_{i+1}_pocket_update.png')
                plot_decision_boundary(X, y, w, pocket_w, iteration=iteration, 
                                      misclassified_idx=[], filename=plot_file)
        else:
            print("  Correctly classified!")
    
    # Iterate through all points to record all misclassified for visualization
    all_misclassified = [i for i in range(len(X_with_bias)) if predict([X_with_bias[i]], w)[0] != y[i]]
    
    print(f"\nEnd of iteration {iteration}")
    print(f"Current weights: w = {w}")
    print(f"Pocket weights: pocket_w = {pocket_w}")
    print(f"Total misclassified with current weights: {len(all_misclassified)}")
    print(f"Pocket errors: {pocket_errors}")
    
    # Store iteration information
    iteration_history.append({
        'iteration': iteration,
        'weights': w.copy(),
        'pocket_weights': pocket_w.copy(),
        'misclassified': all_misclassified.copy(),
        'pocket_errors': pocket_errors
    })
    
    # Create end-of-iteration plot
    plot_file = os.path.join(save_dir, f'pocket_iteration_{iteration}_end.png')
    plot_decision_boundary(X, y, w, pocket_w, iteration=iteration, 
                          misclassified_idx=all_misclassified, filename=plot_file)

# Final decision boundary
final_misclassified = [i for i in range(len(X_with_bias)) if predict([X_with_bias[i]], pocket_w)[0] != y[i]]
final_plot = plot_decision_boundary(X, y, w, pocket_w, misclassified_idx=final_misclassified)
final_plot.savefig(os.path.join(save_dir, 'pocket_final.png'), dpi=300, bbox_inches='tight')

# Print summary
print("\nPocket Algorithm Summary")
print("=" * 50)
print(f"Dataset:")
for i in range(len(X)):
    print(f"Sample {i+1}: x = {X[i]}, y = {y[i]}")
print("\nTraining:")
print(f"Initial weights: w = [0, 0, 0]")
print(f"Learning rate: η = {eta}")
print(f"Ran for {max_iterations} iterations")
print(f"Final weights: w = {w}")
print(f"Final pocket weights: pocket_w = {pocket_w}")
print(f"Final errors with pocket weights: {pocket_errors}")

# Decision boundary equation
w1, w2, w0 = pocket_w
print("\nFinal Pocket Decision Boundary:")
print(f"{w1}*x1 + {w2}*x2 + {w0} = 0")

# Save a figure with all iteration boundaries
plt.figure(figsize=(12, 10))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 3 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot all decision boundaries
color_map = plt.cm.rainbow(np.linspace(0, 1, len(iteration_history) + 1))
x1_range = np.linspace(0, 4, 100)

# Plot initial boundary (may not be visible since w=[0,0,0])
plt.axhline(y=0, color=color_map[0], linestyle='--', alpha=0.5, label='Initial (w=[0,0,0])')

# Plot decision boundaries for each iteration (pocket only)
for idx, hist in enumerate(iteration_history):
    w_iter = hist['pocket_weights']
    w1, w2, w0 = w_iter
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        plt.plot(x1_range, x2_boundary, color=color_map[idx+1], 
                linestyle='-', alpha=0.7, 
                label=f'Iteration {idx+1} (pocket w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Pocket Algorithm - Pocket Decision Boundary Evolution')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the evolution figure
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pocket_evolution.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}") 