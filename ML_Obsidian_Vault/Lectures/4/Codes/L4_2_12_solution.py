import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Define the dataset from Question 12
X = np.array([
    [2, 1],    # Class 1
    [0, 3],    # Class 1
    [-1, 0],   # Class -1
    [-2, -2]   # Class -1
])

y = np.array([1, 1, -1, -1])

# Add bias term to input features
X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Function to make predictions
def predict(X, w):
    return np.sign(np.dot(X, w))

# Function to draw decision boundary
def plot_decision_boundary(X, y, w, title=None, highlight_idx=None, filename=None):
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

# Initial plot before training
plot_decision_boundary(X, y, np.array([0, 0, 0]), 
                      title="Initial Dataset with Zero Weights",
                      filename=os.path.join(save_dir, 'initial_data.png'))

# ===========================
# ONLINE (STOCHASTIC) PERCEPTRON
# ===========================
print("\n=== ONLINE (STOCHASTIC) PERCEPTRON ===")
print("=" * 40)

# Initialize weights
w_online = np.array([0, 0, 0])
eta = 1
max_iterations = 2  # Only run 2 iterations as required

# Store weights history for visualization
online_weights_history = [w_online.copy()]
online_misclassified_history = []

# Perform online perceptron learning
for iteration in range(1, max_iterations + 1):
    print(f"\nIteration {iteration}")
    print("-" * 30)
    
    for i in range(len(X_with_bias)):
        x_i = X_with_bias[i]
        y_i = y[i]
        
        # Compute activation and prediction
        activation = np.dot(w_online, x_i)
        prediction = np.sign(activation) if activation != 0 else 0
        
        print(f"\nSample {i+1}: x = {X[i]}, y = {y_i}")
        print(f"  Current weights: w = {w_online}")
        print(f"  Activation = w · x = {w_online} · {x_i} = {activation:.2f}")
        print(f"  Prediction = {prediction}, Actual = {y_i}")
        
        # Check if misclassified
        if prediction != y_i:
            # Update weights immediately for online algorithm
            w_old = w_online.copy()
            w_online = w_online + eta * y_i * x_i
            
            print(f"  Misclassified! Updating weights:")
            print(f"  w_new = w_old + η * y * x")
            print(f"  w_new = {w_old} + {eta} * {y_i} * {x_i}")
            print(f"  w_new = {w_online}")
            
            # Store for visualization
            online_misclassified_history.append((iteration, i))
            
            # Generate plot for this update
            plot_decision_boundary(
                X, y, w_online,
                title=f"Online Perceptron - Iteration {iteration}, Sample {i+1}",
                highlight_idx=[i],
                filename=os.path.join(save_dir, f'online_iter{iteration}_sample{i+1}.png')
            )
        else:
            print("  Correctly classified! No weight update.")
    
    online_weights_history.append(w_online.copy())
    print(f"\nEnd of iteration {iteration}")
    print(f"Current weights: w = {w_online}")

# ===========================
# BATCH PERCEPTRON
# ===========================
print("\n\n=== BATCH PERCEPTRON ===")
print("=" * 40)

# Initialize weights
w_batch = np.array([0, 0, 0])
eta = 1
max_iterations = 2  # Only run 2 iterations as required

# Store weights history for visualization
batch_weights_history = [w_batch.copy()]
batch_misclassified_history = []

# Perform batch perceptron learning
for iteration in range(1, max_iterations + 1):
    print(f"\nIteration {iteration}")
    print("-" * 30)
    
    # Initialize weight update to zero
    weight_update = np.zeros(3)
    misclassified = []
    
    # First, check all samples and accumulate updates
    for i in range(len(X_with_bias)):
        x_i = X_with_bias[i]
        y_i = y[i]
        
        # Compute activation and prediction
        activation = np.dot(w_batch, x_i)
        prediction = np.sign(activation) if activation != 0 else 0
        
        print(f"\nSample {i+1}: x = {X[i]}, y = {y_i}")
        print(f"  Activation = w · x = {w_batch} · {x_i} = {activation:.2f}")
        print(f"  Prediction = {prediction}, Actual = {y_i}")
        
        # Check if misclassified
        if prediction != y_i:
            print(f"  Misclassified! Adding to batch update...")
            # Accumulate updates for batch algorithm
            weight_update += eta * y_i * x_i
            misclassified.append(i)
            
            # Store for visualization
            batch_misclassified_history.append((iteration, i))
        else:
            print("  Correctly classified!")
    
    # Apply accumulated weight updates (batch update)
    if len(misclassified) > 0:
        w_old = w_batch.copy()
        w_batch = w_batch + weight_update
        
        print(f"\nApplying batch update:")
        print(f"  Misclassified samples: {[i+1 for i in misclassified]}")
        print(f"  w_new = w_old + accumulated_updates")
        print(f"  w_new = {w_old} + {weight_update}")
        print(f"  w_new = {w_batch}")
        
        # Generate plot for this batch update
        plot_decision_boundary(
            X, y, w_batch,
            title=f"Batch Perceptron - After Iteration {iteration}",
            highlight_idx=misclassified,
            filename=os.path.join(save_dir, f'batch_iter{iteration}.png')
        )
    else:
        print("\nNo misclassifications in this iteration. No weight update needed.")
    
    batch_weights_history.append(w_batch.copy())
    print(f"\nEnd of iteration {iteration}")
    print(f"Current weights: w = {w_batch}")

# Create a comparison figure showing both algorithms' decision boundaries
plt.figure(figsize=(12, 10))

# Plot the data points
for i in range(len(X)):
    marker = 'o' if y[i] == 1 else 'x'
    color = 'blue' if y[i] == 1 else 'red'
    label = 'Class 1' if y[i] == 1 and i == 0 else ('Class -1' if y[i] == -1 and i == 2 else None)
    plt.scatter(X[i, 0], X[i, 1], marker=marker, color=color, s=100, label=label)

# Plot decision boundaries
x1_range = np.linspace(-3, 3, 100)

# Initial boundary (w=[0,0,0])
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Initial (w=[0,0,0])')

# Final online boundary
w_online_final = online_weights_history[-1]
w1, w2, w0 = w_online_final
if abs(w2) > 1e-10:
    x2_boundary = (-w1 * x1_range - w0) / w2
    plt.plot(x1_range, x2_boundary, color='green', linestyle='-', 
             label=f'Online Final (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')
elif abs(w1) > 1e-10:
    x1_boundary = -w0 / w1
    plt.axvline(x=x1_boundary, color='green', linestyle='-', label='Online Final')

# Final batch boundary
w_batch_final = batch_weights_history[-1]
w1, w2, w0 = w_batch_final
if abs(w2) > 1e-10:
    x2_boundary = (-w1 * x1_range - w0) / w2
    plt.plot(x1_range, x2_boundary, color='orange', linestyle='-', 
             label=f'Batch Final (w=[{w1:.2f}, {w2:.2f}, {w0:.2f}])')
elif abs(w1) > 1e-10:
    x1_boundary = -w0 / w1
    plt.axvline(x=x1_boundary, color='orange', linestyle='-', label='Batch Final')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Comparison of Online vs Batch Perceptron')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()

# Save the comparison figure
plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nFinal Results:")
print("-" * 40)
print(f"Online Perceptron Final Weights: {w_online}")
print(f"Batch Perceptron Final Weights: {w_batch}")
print(f"\nAll images saved to: {save_dir}") 