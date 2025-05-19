import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset (using the same dataset as in Q4)
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

# Different weight initializations to test
initializations = {
    "zeros": np.array([0, 0, 0]),
    "ones": np.array([1, 1, 0]),
    "random_positive": np.array([0.5, 0.8, 0.3]),
    "random_negative": np.array([-0.5, -0.2, -0.7])
}

# Different learning rates to test
learning_rates = [0.1, 1.0, 2.0]

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, title=None, filename=None, highlight_idx=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)
    
    # Highlight specific points if provided
    if highlight_idx is not None:
        for idx in highlight_idx:
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
            if abs(w1) > 1e-10:
                x1_boundary = -w0 / w1
                plt.axvline(x=x1_boundary, color='g', linestyle='-', label='Decision Boundary')
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if title:
        plt.title(title)
    else:
        plt.title('Perceptron Learning - Decision Boundary')
    
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
        plt.close()
    else:
        plt.show()
    
    return plt

# Function to train perceptron with given initializations
def train_perceptron(X, y, X_with_bias, w_init, eta, max_iterations=100):
    w = w_init.copy()
    iteration = 0
    converged = False
    history = []
    
    # Initial state
    history.append({
        'iteration': 0,
        'weights': w.copy(),
        'misclassified': []
    })
    
    # Print initial information
    print(f"\n==== Training with initial weights = {w}, learning rate = {eta} ====")
    
    # Training loop
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
                
                made_update = True
                break  # Move to next iteration after first update
            else:
                print("  Correctly classified!")
        
        print(f"End of iteration {iteration}")
        print(f"Updated weights: w = {w}")
        
        # Store iteration information
        history.append({
            'iteration': iteration,
            'weights': w.copy(),
            'misclassified': misclassified.copy()
        })
        
        # Check if converged (no misclassifications)
        if not made_update:
            converged = True
            print("\nConverged! No misclassifications detected.")
    
    # Check if hit max iterations without converging
    if not converged:
        print(f"Did not converge in {max_iterations} iterations!")
    
    # Return final weights and history
    return w, history, iteration, converged

# Part 1: Effect of initialization on final solution
print("\n===== PART 1: EFFECT OF INITIALIZATION ON FINAL SOLUTION =====")

# Store results
init_results = {}

# Create visualization for decision boundary evolution (only for zeros initialization)
def plot_boundary_evolution(init_name, history, X, y):
    plt.figure(figsize=(14, 8))
    
    # Create a grid of subplots (2x3)
    iterations_to_show = min(6, len(history))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    color_map = plt.cm.viridis(np.linspace(0, 1, iterations_to_show))
    x1_range = np.linspace(-3, 3, 100)
    
    # Add overall title
    fig.suptitle(f'Decision Boundary Evolution - {init_name} initialization', fontsize=16)
    
    for idx in range(iterations_to_show):
        ax = axes[idx]
        
        # Plot the data points
        for i in range(len(X)):
            marker = 'o' if y[i] == 1 else 'x'
            color = 'blue' if y[i] == 1 else 'red'
            ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70)
        
        w_iter = history[idx]['weights']
        w1, w2, w0 = w_iter
        
        # Plot the decision boundary
        if abs(w2) > 1e-10:
            x2_boundary = (-w1 * x1_range - w0) / w2
            ax.plot(x1_range, x2_boundary, color=color_map[idx], 
                    linestyle='-', linewidth=2)
        elif abs(w1) > 1e-10:
            # Vertical line if w2 is close to zero
            x1_boundary = -w0 / w1
            ax.axvline(x=x1_boundary, color=color_map[idx], linestyle='-', linewidth=2)
        
        # Add subplot title
        ax.set_title(f'Iteration {idx}: w=[{w1:.2f}, {w2:.2f}, {w0:.2f}]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        # Add axis labels only to bottom and left subplots
        if idx >= 3:
            ax.set_xlabel('$x_1$')
        if idx % 3 == 0:
            ax.set_ylabel('$x_2$')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the overall title
    plt.savefig(os.path.join(save_dir, f'init_{init_name}_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Train with different initializations (fixed learning rate)
fixed_eta = 1.0
for init_name, init_weights in initializations.items():
    print(f"\n\n=== Testing initialization: {init_name} with weights {init_weights} ===")
    
    # Plot initial decision boundary
    init_plot_file = os.path.join(save_dir, f'init_{init_name}_initial.png')
    plot_decision_boundary(X, y, init_weights, 
                       title=f'Initial Decision Boundary - {init_name} initialization',
                       filename=init_plot_file)
    
    # Train the model
    final_w, history, iterations, converged = train_perceptron(X, y, X_with_bias, init_weights, fixed_eta)
    
    # Store results
    init_results[init_name] = {
        'initial_weights': init_weights.copy(),
        'final_weights': final_w.copy(),
        'iterations': iterations,
        'converged': converged,
        'history': history
    }
    
    # Plot final decision boundary
    final_plot_file = os.path.join(save_dir, f'init_{init_name}_final.png')
    plot_decision_boundary(X, y, final_w, 
                       title=f'Final Decision Boundary - {init_name} initialization',
                       filename=final_plot_file)
                       
    # Plot evolution of decision boundaries only for zeros and ones initializations
    if init_name in ["zeros", "ones"]:
        plot_boundary_evolution(init_name, history, X, y)
        

# Part 2: Compare all initialization results
print("\n===== PART 2: COMPARING RESULTS OF DIFFERENT INITIALIZATIONS =====")

print("\nSummary of results for different initializations:")
print("-" * 90)
print(f"{'Initialization':<15} | {'Initial Weights':<20} | {'Final Weights':<20} | {'Iterations':<10} | {'Converged':<10}")
print("-" * 90)

for init_name, result in init_results.items():
    init_w_str = f"[{result['initial_weights'][0]:.1f}, {result['initial_weights'][1]:.1f}, {result['initial_weights'][2]:.1f}]"
    final_w_str = f"[{result['final_weights'][0]:.1f}, {result['final_weights'][1]:.1f}, {result['final_weights'][2]:.1f}]"
    iterations = result['iterations']
    converged = "Yes" if result['converged'] else "No"
    
    print(f"{init_name:<15} | {init_w_str:<20} | {final_w_str:<20} | {iterations:<10} | {converged:<10}")

# Create a combined visualization of final decision boundaries
plt.figure(figsize=(12, 10))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot decision boundaries for each initialization
colors = ['green', 'orange', 'purple', 'brown']
x1_range = np.linspace(-3, 3, 100)

for idx, (init_name, result) in enumerate(init_results.items()):
    w = result['final_weights']
    w1, w2, w0 = w
    
    if abs(w2) > 1e-10:
        x2_boundary = (-w1 * x1_range - w0) / w2
        plt.plot(x1_range, x2_boundary, color=colors[idx], 
                linestyle='-', linewidth=2,
                label=f'{init_name}: w=[{w1:.2f}, {w2:.2f}, {w0:.2f}]')
    elif abs(w1) > 1e-10:
        x1_boundary = -w0 / w1
        plt.axvline(x=x1_boundary, color=colors[idx], linestyle='-', linewidth=2,
                  label=f'{init_name}: w=[{w1:.2f}, {w2:.2f}, {w0:.2f}]')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Comparison of Final Decision Boundaries for Different Initializations')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()

# Save the comparison figure
plt.savefig(os.path.join(save_dir, 'init_comparison_final.png'), dpi=300, bbox_inches='tight')
plt.close()

# Part 3: Effect of learning rate on convergence
print("\n===== PART 3: EFFECT OF LEARNING RATE ON CONVERGENCE =====")

# Store results for different learning rates
lr_results = {}

# Fixed initialization for learning rate comparison
fixed_init = "zeros"
init_weights = initializations[fixed_init]

# Create combined visualization for different learning rates
plt.figure(figsize=(16, 5))
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Decision Boundaries for Different Learning Rates', fontsize=16)

# Train with different learning rates
for idx, eta in enumerate(learning_rates):
    print(f"\n\n=== Testing learning rate: {eta} with initial weights {init_weights} ===")
    
    # Train the model
    final_w, history, iterations, converged = train_perceptron(X, y, X_with_bias, init_weights, eta)
    
    # Store results
    lr_results[eta] = {
        'final_weights': final_w.copy(),
        'iterations': iterations,
        'converged': converged,
        'history': history
    }
    
    # Plot on the respective subplot
    ax = axes[idx]
    
    # Plot data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70)
    
    # Plot decision boundary
    w = final_w
    w1, w2, w0 = w
    
    if abs(w2) > 1e-10:
        x1_range = np.linspace(-3, 3, 100)
        x2_boundary = (-w1 * x1_range - w0) / w2
        ax.plot(x1_range, x2_boundary, color='green', 
                linestyle='-', linewidth=2)
    
    # Add subtitle
    ax.set_title(f'$\\eta={eta}$: w=[{w1:.2f}, {w2:.2f}, {w0:.2f}]')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('$x_1$')
    
    # Add y-label only to leftmost subplot
    if idx == 0:
        ax.set_ylabel('$x_2$')
        
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the overall title
plt.savefig(os.path.join(save_dir, 'lr_comparison_combined.png'), dpi=300, bbox_inches='tight')
plt.close()

# Summary of learning rate results
print("\nSummary of results for different learning rates:")
print("-" * 80)
print(f"{'Learning Rate':<15} | {'Final Weights':<20} | {'Iterations':<10} | {'Converged':<10}")
print("-" * 80)

for eta, result in lr_results.items():
    final_w_str = f"[{result['final_weights'][0]:.1f}, {result['final_weights'][1]:.1f}, {result['final_weights'][2]:.1f}]"
    iterations = result['iterations']
    converged = "Yes" if result['converged'] else "No"
    
    print(f"{eta:<15} | {final_w_str:<20} | {iterations:<10} | {converged:<10}")

# Create a combined visualization showing both initialization and learning rate effects
plt.figure(figsize=(12, 8))

# Create a 2x2 grid showing key comparisons
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Key Comparisons for Perceptron Initialization and Learning Rate', fontsize=16)

# 1. Top-left: Zero vs. Ones initialization (initial)
ax = axes[0, 0]
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
    ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70, label=label)

# Plot initial boundaries
w_zeros = initializations['zeros']
w_ones = initializations['ones']

# Zero initialization
if abs(w_zeros[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_zeros[0] * x1_range - w_zeros[2]) / w_zeros[1]
    ax.plot(x1_range, x2_boundary, color='green', 
            linestyle='-', linewidth=2, label='Zeros init')

# Ones initialization
if abs(w_ones[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_ones[0] * x1_range - w_ones[2]) / w_ones[1]
    ax.plot(x1_range, x2_boundary, color='orange', 
            linestyle='-', linewidth=2, label='Ones init')

ax.set_title('Initial Decision Boundaries')
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()

# 2. Top-right: Zero vs. Ones initialization (final)
ax = axes[0, 1]
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70)

# Plot final boundaries
w_zeros_final = init_results['zeros']['final_weights']
w_ones_final = init_results['ones']['final_weights']

# Zero initialization final
if abs(w_zeros_final[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_zeros_final[0] * x1_range - w_zeros_final[2]) / w_zeros_final[1]
    ax.plot(x1_range, x2_boundary, color='green', 
            linestyle='-', linewidth=2, 
            label=f'Zeros final: [{w_zeros_final[0]:.1f}, {w_zeros_final[1]:.1f}, {w_zeros_final[2]:.1f}]')

# Ones initialization final
if abs(w_ones_final[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_ones_final[0] * x1_range - w_ones_final[2]) / w_ones_final[1]
    ax.plot(x1_range, x2_boundary, color='orange', 
            linestyle='-', linewidth=2,
            label=f'Ones final: [{w_ones_final[0]:.1f}, {w_ones_final[1]:.1f}, {w_ones_final[2]:.1f}]')

ax.set_title('Final Decision Boundaries')
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()

# 3. Bottom-left: Random positive vs. Random negative initialization (final)
ax = axes[1, 0]
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70)

# Plot final boundaries
w_rp_final = init_results['random_positive']['final_weights']
w_rn_final = init_results['random_negative']['final_weights']

# Random positive final
if abs(w_rp_final[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_rp_final[0] * x1_range - w_rp_final[2]) / w_rp_final[1]
    ax.plot(x1_range, x2_boundary, color='purple', 
            linestyle='-', linewidth=2,
            label=f'Random+ final: [{w_rp_final[0]:.1f}, {w_rp_final[1]:.1f}, {w_rp_final[2]:.1f}]')

# Random negative final
if abs(w_rn_final[1]) > 1e-10:
    x1_range = np.linspace(-3, 3, 100)
    x2_boundary = (-w_rn_final[0] * x1_range - w_rn_final[2]) / w_rn_final[1]
    ax.plot(x1_range, x2_boundary, color='brown', 
            linestyle='-', linewidth=2,
            label=f'Random- final: [{w_rn_final[0]:.1f}, {w_rn_final[1]:.1f}, {w_rn_final[2]:.1f}]')

ax.set_title('Random Initializations - Final Boundaries')
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()

# 4. Bottom-right: Different learning rates (final)
ax = axes[1, 1]
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=70)

# Learning rate comparison
colors = ['green', 'orange', 'purple']
for idx, (eta, result) in enumerate(lr_results.items()):
    w = result['final_weights']
    w1, w2, w0 = w
    
    if abs(w2) > 1e-10:
        x1_range = np.linspace(-3, 3, 100)
        x2_boundary = (-w1 * x1_range - w0) / w2
        ax.plot(x1_range, x2_boundary, color=colors[idx], 
                linestyle='-', linewidth=2,
                label=f'$\\eta={eta}$: [{w1:.1f}, {w2:.1f}, {w0:.1f}]')

ax.set_title('Different Learning Rates - Final Boundaries')
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the overall title
plt.savefig(os.path.join(save_dir, 'perceptron_key_comparisons.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots and results saved to: {save_dir}") 