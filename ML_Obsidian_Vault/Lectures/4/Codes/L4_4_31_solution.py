import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset
X = np.array([
    [2, 0],    # Class 1
    [0, 2],    # Class 1
    [-2, 0],   # Class -1
    [0, -2],   # Class -1
    [2, 2],    # Class -1
    [-2, -2]   # Class 1
])

y = np.array([1, 1, -1, -1, -1, 1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Learning rate
eta = 1

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to calculate the number of correctly classified samples
def calculate_accuracy(X, y, w):
    predictions = predict(X, w)
    correct = np.sum(predictions == y)
    return correct, len(y)

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, pocket_w=None, iteration=None, misclassified_idx=None, filename=None, title=None):
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
    
    # Plot the current decision boundary if weights are not all zero
    if w is not None and not np.all(w == 0):
        # Extract weights
        w1, w2, w0 = w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 3, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'g-', label='Current Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Current Boundary')
    
    # Plot the pocket decision boundary if provided
    if pocket_w is not None and not np.all(pocket_w == 0):
        # Extract weights
        w1, w2, w0 = pocket_w
        
        # Decision boundary: w1*x1 + w2*x2 + w0 = 0
        # Solving for x2: x2 = (-w1*x1 - w0) / w2
        x1_range = np.linspace(-3, 3, 100)
        
        # Check if w2 is not close to zero to avoid division by zero
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            plt.plot(x1_range, x2_boundary, 'b--', label='Pocket Boundary')
        else:
            # If w2 is close to zero, the boundary is a vertical line: x1 = -w0/w1
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='b', linestyle='--', label='Pocket Boundary')
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    if title:
        plt.title(title)
    elif iteration is not None:
        plt.title(f'Pocket Algorithm - Iteration {iteration}')
    else:
        plt.title('Pocket Algorithm - Final Decision Boundary')
    
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
        eq_str = f'Current: ${w[0]:.2f} x_1 + {w[1]:.2f} x_2 + {w[2]:.2f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    if pocket_w is not None and not np.all(pocket_w == 0):
        eq_str = f'Pocket: ${pocket_w[0]:.2f} x_1 + {pocket_w[1]:.2f} x_2 + {pocket_w[2]:.2f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt

# Function to run the Pocket Algorithm
def run_pocket_algorithm(X, y, X_with_bias, initial_w, eta, max_iterations, save_dir, init_num):
    # Initialize weights
    w = initial_w.copy()
    pocket_w = initial_w.copy()
    
    # Calculate initial accuracy
    pocket_correct, total_samples = calculate_accuracy(X_with_bias, y, pocket_w)
    pocket_accuracy = pocket_correct / total_samples
    
    # Plot initial state
    initial_plot = plot_decision_boundary(
        X, y, w, pocket_w, iteration=0,
        title=f'Initialization {init_num}: Initial State (w = {w})',
        filename=os.path.join(save_dir, f'pocket_init{init_num}_initial.png')
    )
    
    print(f"\nInitialization {init_num}: w = {w}")
    print(f"Initial accuracy: {pocket_correct}/{total_samples} = {pocket_accuracy:.2f}")
    
    # Initialize variables for training
    iteration_history = []
    
    # Perform pocket algorithm learning
    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}")
        print("-" * 50)
        print(f"Current weights: w = {w}")
        print(f"Pocket weights: pocket_w = {pocket_w}")
        
        misclassified = []
        
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
                
                # Update weights (standard perceptron update)
                w_old = w.copy()
                w = w + eta * y_i * x_i
                
                print(f"  Misclassified! Updating weights:")
                print(f"  w_new = w_old + η * y * x")
                print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
                print(f"  w_new = {w}")
                
                # Check if the updated weight performs better than the pocket weight
                current_correct, _ = calculate_accuracy(X_with_bias, y, w)
                current_accuracy = current_correct / total_samples
                
                print(f"  Current accuracy: {current_correct}/{total_samples} = {current_accuracy:.2f}")
                print(f"  Pocket accuracy: {pocket_correct}/{total_samples} = {pocket_accuracy:.2f}")
                
                if current_correct > pocket_correct:
                    pocket_w = w.copy()
                    pocket_correct = current_correct
                    pocket_accuracy = current_accuracy
                    print(f"  New weights perform better! Updating pocket weights.")
                    print(f"  New pocket weights: pocket_w = {pocket_w}")
                else:
                    print(f"  Current weights do not perform better than pocket weights.")
                
                # Generate plot for this update
                plot_file = os.path.join(save_dir, f'pocket_init{init_num}_iteration_{iteration}_sample_{i+1}.png')
                plot_decision_boundary(
                    X, y, w, pocket_w, iteration=iteration, 
                    misclassified_idx=[i], 
                    title=f'Initialization {init_num}: Iteration {iteration}, Sample {i+1}',
                    filename=plot_file
                )
                
                break  # Move to next iteration after first update
            else:
                print("  Correctly classified!")
        
        print(f"End of iteration {iteration}")
        print(f"Updated weights: w = {w}")
        print(f"Pocket weights: pocket_w = {pocket_w}")
        print(f"Pocket accuracy: {pocket_correct}/{total_samples} = {pocket_accuracy:.2f}")
        
        # Store iteration information
        iteration_history.append({
            'iteration': iteration,
            'weights': w.copy(),
            'pocket_weights': pocket_w.copy(),
            'pocket_accuracy': pocket_accuracy,
            'misclassified': misclassified.copy()
        })
    
    # Final decision boundary
    final_plot = plot_decision_boundary(
        X, y, w, pocket_w,
        title=f'Initialization {init_num}: Final State',
        filename=os.path.join(save_dir, f'pocket_init{init_num}_final.png')
    )
    
    return {
        'initial_w': initial_w,
        'final_w': w,
        'pocket_w': pocket_w,
        'pocket_accuracy': pocket_accuracy,
        'pocket_correct': pocket_correct,
        'total_samples': total_samples,
        'iteration_history': iteration_history
    }

# Define the initializations
initializations = [
    {'num': 1, 'w': np.array([1, 0, 0])},
    {'num': 2, 'w': np.array([0, 1, 0])}
]

# Run Pocket Algorithm for each initialization
max_iterations = 3
results = []

for init in initializations:
    result = run_pocket_algorithm(
        X, y, X_with_bias, init['w'], eta, max_iterations, save_dir, init['num']
    )
    results.append(result)

# Print summary of results
print("\nPocket Algorithm Summary")
print("=" * 50)
print(f"Dataset:")
for i in range(len(X)):
    print(f"Sample {i+1}: x = {X[i]}, y = {y[i]}")
print("\nResults:")
for i, result in enumerate(results):
    print(f"\nInitialization {i+1}: w_initial = {result['initial_w']}")
    print(f"Final weights: w = {result['final_w']}")
    print(f"Pocket weights: pocket_w = {result['pocket_w']}")
    print(f"Pocket accuracy: {result['pocket_correct']}/{result['total_samples']} = {result['pocket_accuracy']:.2f}")

# Create a comparison plot with both final pocket decision boundaries
plt.figure(figsize=(10, 8))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot both pocket decision boundaries
colors = ['green', 'purple']
styles = ['-', '--']
x1_range = np.linspace(-3, 3, 100)

for i, result in enumerate(results):
    pocket_w = result['pocket_w']
    w1, w2, w0 = pocket_w
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        plt.plot(x1_range, x2_boundary, color=colors[i], linestyle=styles[i], 
                label=f'Init {i+1} Pocket: [{w1:.2f}, {w2:.2f}, {w0:.2f}], Acc: {result["pocket_accuracy"]:.2f}')
    elif abs(w1) > 1e-10:
        x1_boundary = -w0 / w1
        plt.axvline(x=x1_boundary, color=colors[i], linestyle=styles[i], 
                   label=f'Init {i+1} Pocket: [{w1:.2f}, {w2:.2f}, {w0:.2f}], Acc: {result["pocket_accuracy"]:.2f}')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Pocket Algorithm - Comparison of Final Decision Boundaries')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()

# Save the comparison figure
plt.savefig(os.path.join(save_dir, 'pocket_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}") 