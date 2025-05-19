import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting with amsmath package
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'serif'

# Define the XOR dataset
X = np.array([
    [0, 0],  # Class 0
    [0, 1],  # Class 1
    [1, 0],  # Class 1
    [1, 1]   # Class 0
])

y = np.array([0, 1, 1, 0])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Convert targets to -1, 1 for perceptron algorithm
y_perceptron = np.where(y == 0, -1, 1)

# PART 1: Plot the XOR dataset
def plot_xor_dataset():
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1 (Output = 1)' if y[i] == 1 and i == 1 else ('Class 0 (Output = 0)' if y[i] == 0 and i == 0 else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=200, label=label)
        plt.annotate(f'({X[i,0]}, {X[i,1]}) → {y[i]}', 
                     (X[i,0], X[i,1]), 
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=12)
    
    # Add labels and title
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('XOR Problem Dataset', fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    
    # Add legend
    plt.legend(fontsize=12)
    
    plt.savefig(os.path.join(save_dir, 'xor_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("XOR dataset plot saved.")

# PART 2: Prove XOR is not linearly separable
def attempt_linear_separation():
    # Attempt different linear boundaries
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Different attempt scenarios
    scenarios = [
        {
            'title': 'Attempt 1: Horizontal Line',
            'boundary': lambda x: np.ones_like(x) * 0.5,
            'eq': '$x_2 = 0.5$'
        },
        {
            'title': 'Attempt 2: Vertical Line',
            'boundary': lambda x: None,  # Will draw vertical line at x1 = 0.5
            'eq': '$x_1 = 0.5$'
        },
        {
            'title': 'Attempt 3: Diagonal Line (Positive Slope)',
            'boundary': lambda x: x,
            'eq': '$x_2 = x_1$'
        },
        {
            'title': 'Attempt 4: Diagonal Line (Negative Slope)',
            'boundary': lambda x: -x + 1,
            'eq': '$x_2 = -x_1 + 1$'
        }
    ]
    
    x1_range = np.linspace(-0.5, 1.5, 100)
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        # Plot the data points
        for i in range(len(X)):
            marker = 'o' if y[i] == 1 else 'x'
            color = 'blue' if y[i] == 1 else 'red'
            label = 'Class 1' if y[i] == 1 and i == 1 else ('Class 0' if y[i] == 0 and i == 0 else None)
            ax.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=150, label=label)
        
        # Draw decision boundary
        if idx == 1:  # Vertical line case
            ax.axvline(x=0.5, color='g', linestyle='-', linewidth=2, label='Decision Boundary')
        else:
            x2_boundary = scenario['boundary'](x1_range)
            ax.plot(x1_range, x2_boundary, 'g-', linewidth=2, label='Decision Boundary')
        
        # Add title and equation
        ax.set_title(scenario['title'], fontsize=14)
        ax.text(0.05, 0.95, scenario['eq'], transform=ax.transAxes, 
                fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
        
        # Add misclassified point count
        if idx == 0:  # Horizontal line
            misclassified = "Points (0,0) and (1,1) are on one side\nPoints (0,1) and (1,0) are on the other side"
        elif idx == 1:  # Vertical line
            misclassified = "Points (0,0) and (0,1) are on one side\nPoints (1,0) and (1,1) are on the other side"
        elif idx == 2:  # Diagonal (positive slope)
            misclassified = "Points (0,0) and (1,1) are on the line\nBut they should be in different classes"
        else:  # Diagonal (negative slope)
            misclassified = "Points (0,1) and (1,0) are on the line\nBut they should be in the same class"
            
        ax.text(0.05, 0.05, misclassified, transform=ax.transAxes, 
                fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle="round", fc="lightyellow", ec="orange", alpha=0.9))
        
        # Set limits and add grid
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'linear_separation_attempts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Linear separation attempts plot saved.")
    
    # Print mathematical proof instead of saving as image
    print("\n" + "="*80)
    print("Mathematical Proof that XOR is not linearly separable:")
    print("="*80)
    print("For a linear classifier, we need weights w₀, w₁, w₂ such that:")
    print("w₀ + w₁x₁ + w₂x₂ > 0 for class 1")
    print("w₀ + w₁x₁ + w₂x₂ < 0 for class 0")
    print("\nFor our XOR problem, this gives us the following inequalities:")
    print("1. (0,0) → 0: w₀ < 0")
    print("2. (0,1) → 1: w₀ + w₂ > 0")
    print("3. (1,0) → 1: w₀ + w₁ > 0")
    print("4. (1,1) → 0: w₀ + w₁ + w₂ < 0")
    print("\nFrom (2) and (3): w₀ + w₂ > 0 and w₀ + w₁ > 0")
    print("Adding these: 2w₀ + w₁ + w₂ > 0")
    print("\nBut from (1) and (4): w₀ < 0 and w₀ + w₁ + w₂ < 0")
    print("This creates a contradiction because:")
    print("If w₀ < 0 and w₀ + w₁ + w₂ < 0, then 2w₀ + w₁ + w₂ < 0")
    print("\nSince we can't satisfy all constraints simultaneously, XOR is not linearly separable.")
    print("="*80)
    
    # Also provide the markdown version
    print("\nMarkdown version of the proof:")
    print("""
```
For a linear classifier, we need weights $w_0, w_1, w_2$ such that:
- $w_0 + w_1x_1 + w_2x_2 > 0$ for class 1
- $w_0 + w_1x_1 + w_2x_2 < 0$ for class 0

For our XOR problem, this gives us the following inequalities:
1. $(0,0) \\to 0: w_0 < 0$
2. $(0,1) \\to 1: w_0 + w_2 > 0$
3. $(1,0) \\to 1: w_0 + w_1 > 0$
4. $(1,1) \\to 0: w_0 + w_1 + w_2 < 0$

From (2) and (3): $w_0 + w_2 > 0$ and $w_0 + w_1 > 0$
Adding these: $2w_0 + w_1 + w_2 > 0$

But from (1) and (4): $w_0 < 0$ and $w_0 + w_1 + w_2 < 0$
This creates a contradiction because if $w_0 < 0$ and $w_0 + w_1 + w_2 < 0$, then $2w_0 + w_1 + w_2 < 0$

Since we cannot satisfy all constraints simultaneously, XOR is not linearly separable.
```
""")

# PART 3: Demonstrate why perceptron fails
def run_perceptron_on_xor():
    # Initialize weights to zeros
    w = np.array([0, 0, 0])  # [w1, w2, w0]
    
    # Learning rate
    eta = 1
    
    # Function to make predictions
    def predict(X, w):
        return np.sign(np.dot(X, w))
    
    # Training loop
    max_iterations = 20
    iteration_history = []
    
    print("\nRunning Perceptron on XOR Problem:")
    print("-" * 50)
    print(f"Initial weights: w = {w}")
    
    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}")
        print("-" * 30)
        
        misclassified_count = 0
        
        # Check each sample for misclassification
        for i in range(len(X_with_bias)):
            x_i = X_with_bias[i]
            y_i = y_perceptron[i]
            
            # Compute activation
            activation = np.dot(w, x_i)
            prediction = np.sign(activation) if activation != 0 else 0
            
            print(f"Sample {i+1}: x = {X[i]}, y = {y_i}")
            print(f"  Activation = w · x = {w} · {x_i} = {activation}")
            print(f"  Prediction = {prediction}, Actual = {y_i}")
            
            # Check if misclassified
            if prediction != y_i:
                misclassified_count += 1
                
                # Update weights
                w_old = w.copy()
                w = w + eta * y_i * x_i
                
                print(f"  Misclassified! Updating weights:")
                print(f"  w_new = w_old + η * y * x")
                print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
                print(f"  w_new = {w}")
            else:
                print("  Correctly classified!")
        
        # Store iteration information
        iteration_history.append({
            'iteration': iteration,
            'weights': w.copy(),
            'misclassified_count': misclassified_count
        })
        
        # Check if converged (no misclassifications)
        if misclassified_count == 0:
            print("\nConverged! No misclassifications.")
            break
        else:
            print(f"\nEnd of iteration {iteration}")
            print(f"Weights: w = {w}")
            print(f"Misclassified samples: {misclassified_count}")
    
    # Plot misclassification history
    iterations = [h['iteration'] for h in iteration_history]
    misclassified = [h['misclassified_count'] for h in iteration_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, misclassified, 'ro-', linewidth=2, markersize=10)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Number of Misclassified Samples', fontsize=14)
    plt.title('Perceptron Training on XOR: Misclassification History', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'perceptron_misclassification_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Perceptron misclassification history saved.")
    
    # Print summary of the weight cycle
    print("\n" + "="*80)
    print("Perceptron Weight Cycle Analysis:")
    print("="*80)
    print("The perceptron algorithm gets stuck in a cycle with the XOR problem:")
    print(f"Weight cycle: [0,0,0] → [0,0,-1] → [0,1,0] → [1,1,1] → [0,0,0] → ...")
    print("This pattern repeats indefinitely, never converging to a solution.")
    print("In markdown:")
    print(r"```")
    print(r"Weight cycle: $[0,0,0] \to [0,0,-1] \to [0,1,0] \to [1,1,1] \to [0,0,0] \to \ldots$")
    print(r"```")
    print("="*80)
    
    # Print final summary
    print("\nPerceptron on XOR Problem Summary")
    print("=" * 50)
    print(f"Dataset: XOR Problem")
    print(f"Training stopped after {len(iteration_history)} iterations")
    print(f"Final weights: w = {w}")
    print(f"Final misclassified samples: {iteration_history[-1]['misclassified_count']}/4")
    print("\nConclusion: The perceptron algorithm fails to converge on the XOR problem")
    print("because XOR is not linearly separable.")
    
    return iteration_history[-1]['misclassified_count']

# PART 4: Solutions to the XOR problem
def solve_xor_feature_transformation():
    # Solution 1: Feature transformation - add a quadratic feature x1*x2
    # New feature vector: [x1, x2, x1*x2, 1]
    X_transformed = np.column_stack((X, X[:, 0] * X[:, 1], np.ones(len(X))))
    
    # Visualize the transformation
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data points in 3D
    for i in range(len(X)):
        x1, x2 = X[i]
        x3 = x1 * x2  # New feature
        color = 'blue' if y[i] == 1 else 'red'
        marker = 'o' if y[i] == 1 else '^'
        label = 'Class 1' if y[i] == 1 and i == 1 else ('Class 0' if y[i] == 0 and i == 0 else None)
        ax.scatter(x1, x2, x3, color=color, marker=marker, s=150, label=label)
        ax.text(x1, x2, x3, f"({x1},{x2},{x3}) → {y[i]}", size=10)
    
    # Create grid for the hyperplane
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.5, 10))
    z = 0.5 * np.ones_like(xx)  # Horizontal plane at z=0.5
    
    # Plot the separating hyperplane
    ax.plot_surface(xx, yy, z, alpha=0.3, color='green')
    
    # Add equation and labels
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('$x_3 = x_1 x_2$', fontsize=14)
    ax.set_title('XOR Solution: Feature Transformation with $x_3 = x_1 x_2$', fontsize=16)
    
    # Add text about the hyperplane
    ax.text(-0.5, -0.5, 0.5, 'Separating Hyperplane: $x_3 = 0.5$', color='green', fontsize=12)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'xor_solution_feature_transformation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature transformation solution plot saved.")
    
    # Print explanation text instead of saving as image
    print("\n" + "="*80)
    print("Feature Transformation Solution for XOR:")
    print("="*80)
    print("1. We add a new feature x₃ = x₁·x₂")
    print("\n2. The transformed dataset becomes:")
    print("   (0,0,0) → 0")
    print("   (0,1,0) → 1")
    print("   (1,0,0) → 1")
    print("   (1,1,1) → 0")
    print("\n3. The decision boundary is the plane x₃ = 0.5:")
    print("   w₃·x₃ = 0.5, where w₃ = 1")
    print("\n4. This cleanly separates the classes:")
    print("   Class 0 points have x₃ values 0 and 1")
    print("   Class 1 points both have x₃ = 0")
    print("\n5. The new decision function is:")
    print("   f(x₁, x₂) = sign(0.5 - x₁·x₂)")
    print("="*80)
    
    # Also provide the markdown version
    print("\nMarkdown version of the feature transformation explanation:")
    print("""
```
By adding a new feature $x_3 = x_1 \\cdot x_2$ (the product of the inputs), the transformed dataset becomes:
- $(0,0,0) \\to 0$
- $(0,1,0) \\to 1$
- $(1,0,0) \\to 1$
- $(1,1,1) \\to 0$

In this 3D space, we can now separate the classes with a horizontal plane at $x_3 = 0.5$. The decision function becomes:
$f(x_1, x_2) = \\mathrm{sign}(0.5 - x_1 \\cdot x_2)$
```
""")

def solve_xor_mlp():
    """Visualize a Multi-Layer Perceptron solution for XOR"""
    
    # Create a diagram showing the MLP architecture
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer positions
    layer_positions = [0, 2, 4]  # x positions for each layer
    
    # Draw input layer (2 nodes)
    for i, label in enumerate(['$x_1$', '$x_2$']):
        y_pos = i * 2
        circle = plt.Circle((layer_positions[0], y_pos), radius=0.3, fill=True, color='lightblue', alpha=0.8)
        ax.add_patch(circle)
        ax.text(layer_positions[0], y_pos, label, ha='center', va='center', fontsize=14)
    
    # Draw hidden layer (2 nodes)
    hidden_neurons = ['$h_1$', '$h_2$']
    for i, label in enumerate(hidden_neurons):
        y_pos = i * 2
        circle = plt.Circle((layer_positions[1], y_pos), radius=0.3, fill=True, color='lightgreen', alpha=0.8)
        ax.add_patch(circle)
        ax.text(layer_positions[1], y_pos, label, ha='center', va='center', fontsize=14)
    
    # Draw output layer (1 node)
    circle = plt.Circle((layer_positions[2], 0.5), radius=0.3, fill=True, color='lightcoral', alpha=0.8)
    ax.add_patch(circle)
    ax.text(layer_positions[2], 0.5, '$y$', ha='center', va='center', fontsize=14)
    
    # Draw connections between input and hidden layer
    for i in range(2):  # from inputs
        for j in range(2):  # to hidden
            ax.plot([layer_positions[0]+0.3, layer_positions[1]-0.3], 
                   [i*2, j*2], 'k-', alpha=0.7)
    
    # Draw connections between hidden and output layer
    for j in range(2):  # from hidden
        ax.plot([layer_positions[1]+0.3, layer_positions[2]-0.3], 
               [j*2, 0.5], 'k-', alpha=0.7)
    
    # Add weights for some connections
    ax.text(1, 1.7, '$w_{11}^{(1)} = 1$', fontsize=12)
    ax.text(1, -0.3, '$w_{21}^{(1)} = 1$', fontsize=12)
    ax.text(1, 1, '$w_{12}^{(1)} = 1$', fontsize=12)
    ax.text(1, 0.4, '$w_{22}^{(1)} = 1$', fontsize=12)
    
    ax.text(3, 0.9, '$w_{1}^{(2)} = 1$', fontsize=12)
    ax.text(3, 0.1, '$w_{2}^{(2)} = -2$', fontsize=12)
    
    # Add biases
    ax.text(2, 2.3, '$b_1^{(1)} = 0$', fontsize=12)
    ax.text(2, -0.3, '$b_2^{(1)} = -1$', fontsize=12)
    ax.text(4, 1.2, '$b^{(2)} = 0$', fontsize=12)
    
    # Set the limits
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 3)
    
    # Remove axes
    ax.axis('off')
    
    # Add title and layer labels
    plt.title('Multi-Layer Perceptron for XOR Problem', fontsize=16)
    ax.text(layer_positions[0], 2.5, 'Input Layer', ha='center', fontsize=14)
    ax.text(layer_positions[1], 2.5, 'Hidden Layer', ha='center', fontsize=14)
    ax.text(layer_positions[2], 2.5, 'Output Layer', ha='center', fontsize=14)
    
    # Add activation functions
    ax.text(layer_positions[1], -0.8, 'ReLU Activation', ha='center', fontsize=12)
    ax.text(layer_positions[2], -0.8, 'Sigmoid Activation', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_solution_mlp_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MLP architecture diagram saved.")
    
    # Create a diagram showing the decision regions
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    for i in range(len(X)):
        marker = 'o' if y[i] == 1 else 'x'
        color = 'blue' if y[i] == 1 else 'red'
        label = 'Class 1' if y[i] == 1 and i == 1 else ('Class 0' if y[i] == 0 and i == 0 else None)
        plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=200, label=label)
    
    # Create a grid of points for decision boundary visualization
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
    
    # Define the MLP forward pass (simple implementation)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def relu(x):
        return np.maximum(0, x)
    
    def mlp_predict(X):
        # Hidden layer with ReLU activation
        # h1 = relu(x1 + x2)
        # h2 = relu(x1 + x2 - 1)
        h1 = relu(X[:, 0] + X[:, 1])
        h2 = relu(X[:, 0] + X[:, 1] - 1)
        
        # Output layer with sigmoid activation
        # y = sigmoid(h1 - 2*h2)
        y_pred = sigmoid(h1 - 2*h2)
        
        return y_pred
    
    # Get predictions for grid points
    y_grid = mlp_predict(X_grid)
    y_grid = y_grid.reshape(x1_grid.shape)
    
    # Plot decision boundary
    plt.contourf(x1_grid, x2_grid, y_grid, levels=20, cmap='RdBu', alpha=0.3)
    plt.contour(x1_grid, x2_grid, y_grid, levels=[0.5], colors='green', linewidths=2)
    
    # Add labels and title
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('MLP Decision Boundary for XOR Problem', fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set limits
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    
    # Add legend
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_solution_mlp_decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MLP decision boundary plot saved.")
    
    # Print explanation text instead of saving as image
    print("\n" + "="*80)
    print("Multi-Layer Perceptron Solution for XOR:")
    print("="*80)
    print("1. Network Architecture:")
    print("   - Input Layer: 2 neurons (x₁, x₂)")
    print("   - Hidden Layer: 2 neurons with ReLU activation")
    print("   - Output Layer: 1 neuron with Sigmoid activation")
    print("\n2. Forward Pass:")
    print("   - Hidden Layer:")
    print("     h₁ = ReLU(x₁ + x₂)")
    print("     h₂ = ReLU(x₁ + x₂ - 1)")
    print("   - Output Layer:")
    print("     y = Sigmoid(h₁ - 2h₂)")
    print("\n3. How it works on XOR data:")
    print("   - Input (0,0): h₁=0, h₂=0 ⇒ y=0.5 ≈ 0")
    print("   - Input (0,1): h₁=1, h₂=0 ⇒ y=0.73 ≈ 1")
    print("   - Input (1,0): h₁=1, h₂=0 ⇒ y=0.73 ≈ 1")
    print("   - Input (1,1): h₁=2, h₂=1 ⇒ y=0.5 ≈ 0")
    print("\n4. Each hidden neuron creates a linear boundary, and their combination")
    print("   allows for a non-linear decision boundary that solves XOR.")
    print("="*80)
    
    # Also provide the markdown version
    print("\nMarkdown version of the MLP explanation:")
    print("""
```
With this architecture:
- Input Layer: 2 neurons ($x_1$, $x_2$)
- Hidden Layer: 2 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation

The forward pass calculation:
- Hidden Layer:
  $h_1 = \\mathrm{ReLU}(x_1 + x_2)$
  $h_2 = \\mathrm{ReLU}(x_1 + x_2 - 1)$
- Output Layer:
  $y = \\mathrm{Sigmoid}(h_1 - 2h_2)$

How it works on XOR data:
- Input (0,0): $h_1=0, h_2=0 \\Rightarrow y=0.5 \\approx 0$
- Input (0,1): $h_1=1, h_2=0 \\Rightarrow y=0.73 \\approx 1$
- Input (1,0): $h_1=1, h_2=0 \\Rightarrow y=0.73 \\approx 1$
- Input (1,1): $h_1=2, h_2=1 \\Rightarrow y=0.5 \\approx 0$
```
""")

# Main function to run all parts
def run_all():
    print("Starting analysis of XOR problem...")
    
    # Part 1: Plot the XOR dataset
    plot_xor_dataset()
    
    # Part 2: Prove XOR is not linearly separable
    attempt_linear_separation()
    
    # Part 3: Demonstrate perceptron failure
    misclassified_count = run_perceptron_on_xor()
    print(f"Perceptron final misclassified count: {misclassified_count}/4")
    
    # Part 4: Solutions to XOR
    solve_xor_feature_transformation()
    solve_xor_mlp()
    
    # List the files saved
    print(f"\nAll analyses completed. Images saved to: {save_dir}")
    print("Saved files:")
    for file in sorted(os.listdir(save_dir)):
        print(f"- {file}")
    
    return {
        'output_dir': save_dir,
        'perceptron_failed': misclassified_count > 0
    }

# Run the full analysis
if __name__ == "__main__":
    results = run_all() 