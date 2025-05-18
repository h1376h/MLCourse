import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters
# 3-class problem with 2 features
w1 = np.array([1, 2])  # Weight vector for class 1
w2 = np.array([3, -1])  # Weight vector for class 2
w3 = np.array([0, 1])   # Weight vector for class 3
W = np.vstack([w1, w2, w3])  # All weight vectors stacked

x = np.array([2, 2])  # The new point to classify

print("\n==== Multi-class Perceptron: Step-by-Step Solution ====")
print("\nQuestion 5.1: How multi-class perceptron differs from binary perceptron")
print("-" * 60)
print("Binary perceptron uses a single weight vector and classifies based on sign.")
print("Multi-class perceptron uses K weight vectors (one per class) and classifies")
print("based on which weight vector gives the highest score for an input.")

print("\nQuestion 5.2: Number of weight vectors and parameters needed")
print("-" * 60)
print(f"For K={3} classes and d={2} features:")
print(f"- Number of weight vectors: {3}")
print(f"- Number of parameters: {3 * 2} = {3 * 2}")
print(f"In general, for K classes and d features:")
print(f"- Number of weight vectors: K")
print(f"- Number of parameters: K * d")

# Step 1: Compute scores for each class for the new point x = [2, 2]
scores = np.dot(W, x)
print("\nQuestion 5.3: Class prediction for x = [2, 2]")
print("-" * 60)
print(f"Weight vectors:")
print(f"w1 = {w1}")
print(f"w2 = {w2}")
print(f"w3 = {w3}")

print(f"\nInput point: x = {x}")

print(f"\nComputing scores (dot products w_k · x):")
print(f"Score for class 1: w1 · x = ({w1[0]} * {x[0]}) + ({w1[1]} * {x[1]}) = {np.dot(w1, x)}")
print(f"Score for class 2: w2 · x = ({w2[0]} * {x[0]}) + ({w2[1]} * {x[1]}) = {np.dot(w2, x)}")
print(f"Score for class 3: w3 · x = ({w3[0]} * {x[0]}) + ({w3[1]} * {x[1]}) = {np.dot(w3, x)}")

predicted_class = np.argmax(scores) + 1  # +1 because classes are 1-indexed
print(f"\nScores for each class: {scores}")
print(f"Predicted class: Class {predicted_class} (highest score)")

# Step 2: Update rule if the true label is class 1
true_class = 1  # As given in the problem
true_class_idx = true_class - 1  # 0-indexed

print("\nQuestion 5.4: Update rule if the true label is class 1")
print("-" * 60)
print(f"True label: Class {true_class}")
print(f"Predicted label: Class {predicted_class}")

if predicted_class != true_class:
    print("\nSince prediction is incorrect, we need to update the weights:")
    print(f"1. Increment w{true_class}: w{true_class} = w{true_class} + x")
    print(f"   w{true_class}_new = {w1} + {x} = {w1 + x}")
    
    print(f"2. Decrement w{predicted_class}: w{predicted_class} = w{predicted_class} - x")
    print(f"   w{predicted_class}_new = {W[predicted_class-1]} - {x} = {W[predicted_class-1] - x}")
    
    # Perform the update
    w1_updated = w1 + x
    w_pred_updated = W[predicted_class-1] - x
    
    # Update the weight matrix for visualization
    W_updated = W.copy()
    W_updated[true_class_idx] = w1_updated
    W_updated[predicted_class-1] = w_pred_updated
else:
    print("\nPrediction is correct, no update needed.")
    W_updated = W.copy()

# Visualization 1: Decision regions
def plot_decision_regions():
    """
    Plot the decision regions for the multi-class perceptron.
    """
    x_min, x_max = -1, 5
    y_min, y_max = -1, 5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Compute scores for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = np.dot(grid_points, W.T)
    Z = np.argmax(scores, axis=1) + 1  # +1 because classes are 1-indexed
    Z = Z.reshape(xx.shape)
    
    # Plot the decision regions
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot the decision boundaries (where scores are equal)
    # For each pair of classes, find points where their scores are equal
    class_pairs = [(0, 1), (0, 2), (1, 2)]  # Indices of classes to compare
    colors = ['red', 'blue', 'green']
    
    for i, (c1, c2) in enumerate(class_pairs):
        # Points where scores for c1 and c2 are equal
        w_diff = W[c1] - W[c2]
        
        # If the weight difference has non-zero components
        if np.any(w_diff):
            # For 2D, we can compute the line explicitly
            if w_diff[1] != 0:
                x_boundary = np.linspace(x_min, x_max, 100)
                y_boundary = -w_diff[0]/w_diff[1] * x_boundary
                valid_idx = (y_boundary >= y_min) & (y_boundary <= y_max)
                plt.plot(x_boundary[valid_idx], y_boundary[valid_idx], 
                         '-', color=colors[i], alpha=0.7, 
                         label=f'Boundary between classes {c1+1} and {c2+1}')
            else:
                # Vertical line
                x_val = 0  # This would need to be computed properly for non-zero cases
                plt.axvline(x=x_val, color=colors[i], alpha=0.7, 
                           label=f'Boundary between classes {c1+1} and {c2+1}')
    
    # Plot the weight vectors as arrows
    for i, w in enumerate(W):
        plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.2, 
                 fc=f'C{i}', ec=f'C{i}', label=f'Weight vector w{i+1}')
    
    # Plot the test point
    plt.scatter(x[0], x[1], c='black', s=100, marker='x', label='Test point x = [2, 2]')
    
    # Add a grid, labels, title, and legend
    plt.grid(True)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Multi-class Perceptron: Decision Regions')
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'decision_regions.png'), dpi=300, bbox_inches='tight')

# Visualization 2: 3D Scores visualization
def plot_3d_scores():
    """
    Create a 3D visualization of the scores.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points
    x1 = np.linspace(-1, 5, 100)
    x2 = np.linspace(-1, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute scores for class 1, 2, and 3
    Z1 = w1[0] * X1 + w1[1] * X2
    Z2 = w2[0] * X1 + w2[1] * X2
    Z3 = w3[0] * X1 + w3[1] * X2
    
    # Plot the score surfaces
    surf1 = ax.plot_surface(X1, X2, Z1, alpha=0.5, color='red', label='Score for class 1')
    surf2 = ax.plot_surface(X1, X2, Z2, alpha=0.5, color='green', label='Score for class 2')
    surf3 = ax.plot_surface(X1, X2, Z3, alpha=0.5, color='blue', label='Score for class 3')
    
    # Plot the test point with a vertical line to its score value for each class
    test_point_scores = [np.dot(w, x) for w in W]
    ax.scatter(x[0], x[1], test_point_scores[0], c='red', s=100, marker='o')
    ax.scatter(x[0], x[1], test_point_scores[1], c='green', s=100, marker='o')
    ax.scatter(x[0], x[1], test_point_scores[2], c='blue', s=100, marker='o')
    
    # Add vertical lines connecting the test point to its scores
    for i, score in enumerate(test_point_scores):
        color = ['red', 'green', 'blue'][i]
        ax.plot([x[0], x[0]], [x[1], x[1]], [0, score], color=color, linestyle='--')
    
    # Highlight the highest score
    highest_idx = np.argmax(test_point_scores)
    highest_score = test_point_scores[highest_idx]
    ax.scatter(x[0], x[1], highest_score, c='black', s=150, marker='*', 
              label=f'Highest score: Class {highest_idx+1}')
    
    # Add a legend
    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', lw=4),
        Line2D([0], [0], color='green', lw=4),
        Line2D([0], [0], color='blue', lw=4),
        Line2D([0], [0], marker='*', color='black', markersize=10, linestyle='None')
    ]
    ax.legend(custom_lines, [f'Score for class 1: {test_point_scores[0]:.1f}', 
                            f'Score for class 2: {test_point_scores[1]:.1f}', 
                            f'Score for class 3: {test_point_scores[2]:.1f}',
                            f'Highest score: Class {highest_idx+1}'],
             loc='upper center', bbox_to_anchor=(0.5, -0.05))
    
    # Add a grid, labels, and title
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('Score')
    ax.set_title('Multi-class Perceptron: Score Functions')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, '3d_scores.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Updated weights after applying the update rule
def plot_updated_weights():
    """
    Plot the decision regions after updating the weights.
    """
    if predicted_class == true_class:
        # No update needed, just return
        return
    
    x_min, x_max = -1, 5
    y_min, y_max = -1, 5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Compute scores for each point in the grid using updated weights
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = np.dot(grid_points, W_updated.T)
    Z = np.argmax(scores, axis=1) + 1  # +1 because classes are 1-indexed
    Z = Z.reshape(xx.shape)
    
    # Plot the decision regions
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot the decision boundaries (where scores are equal)
    # For each pair of classes, find points where their scores are equal
    class_pairs = [(0, 1), (0, 2), (1, 2)]  # Indices of classes to compare
    colors = ['red', 'blue', 'green']
    
    for i, (c1, c2) in enumerate(class_pairs):
        # Points where scores for c1 and c2 are equal
        w_diff = W_updated[c1] - W_updated[c2]
        
        # If the weight difference has non-zero components
        if np.any(w_diff):
            # For 2D, we can compute the line explicitly
            if w_diff[1] != 0:
                x_boundary = np.linspace(x_min, x_max, 100)
                y_boundary = -w_diff[0]/w_diff[1] * x_boundary
                valid_idx = (y_boundary >= y_min) & (y_boundary <= y_max)
                plt.plot(x_boundary[valid_idx], y_boundary[valid_idx], 
                         '-', color=colors[i], alpha=0.7, 
                         label=f'Boundary between classes {c1+1} and {c2+1}')
            else:
                # Vertical line
                x_val = 0  # This would need to be computed properly for non-zero cases
                plt.axvline(x=x_val, color=colors[i], alpha=0.7, 
                           label=f'Boundary between classes {c1+1} and {c2+1}')
    
    # Plot the original weight vectors as dotted lines
    for i, w in enumerate(W):
        plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.2, 
                 fc=f'C{i}', ec=f'C{i}', linestyle='--', alpha=0.5, 
                 label=f'Original weight vector w{i+1}')
    
    # Plot the updated weight vectors as solid lines
    for i, w in enumerate(W_updated):
        plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.2, 
                 fc=f'C{i}', ec=f'C{i}',
                 label=f'Updated weight vector w{i+1}')
    
    # Plot the test point
    plt.scatter(x[0], x[1], c='black', s=100, marker='x', label='Test point x = [2, 2]')
    
    # Add a grid, labels, title, and legend
    plt.grid(True)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Multi-class Perceptron: Updated Decision Regions')
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'updated_decision_regions.png'), dpi=300, bbox_inches='tight')

# Execute visualizations
print("\nGenerating visualizations...")
plot_decision_regions()
plot_3d_scores()
if predicted_class != true_class:
    plot_updated_weights()
print(f"Visualizations saved to: {save_dir}")
print("\nDone!") 