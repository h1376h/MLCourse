import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 10: Pocket Algorithm Applications")
print("======================================")

# Step 1: Explain the goal of the Pocket Algorithm
print("\nStep 1: Explain the goal of the Pocket Algorithm")
print("--------------------------------------------")

print("The Pocket Algorithm is an extension of the standard Perceptron algorithm designed")
print("to handle non-linearly separable data. Its goal is to find the best possible linear")
print("decision boundary that minimizes the number of misclassifications, even when perfect")
print("separation is impossible.")
print("\nUnlike the standard Perceptron, which may never converge for non-separable data,")
print("the Pocket Algorithm keeps track of the best weights found so far (in its 'pocket')")
print("and returns these weights after a fixed number of iterations, ensuring we get the")
print("best possible linear classifier even for non-separable data.")

# Step 2: Generate a non-separable dataset
print("\nStep 2: Generate a non-separable dataset")
print("------------------------------------")

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with some overlapping points
n_samples = 200

# Generate data for class 1 (centered at [2, 2])
mean1 = np.array([2, 2])
cov1 = np.array([[1.5, 0.6], [0.6, 1.0]])
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
y1 = np.ones(n_samples)

# Generate data for class 2 (centered at [4, 4])
mean2 = np.array([4, 4])
cov2 = np.array([[1.0, 0.4], [0.4, 1.5]])
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
y2 = -np.ones(n_samples)

# Combine the data
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Plot the dataset
plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='blue', alpha=0.5, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', alpha=0.5, label='Class 2')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Non-separable Dataset', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the dataset plot
plt.savefig(os.path.join(save_dir, "non_separable_dataset.png"), dpi=300, bbox_inches='tight')

print(f"Generated a dataset with {len(X)} points:")
print(f"- {len(X1)} points for Class 1 (y = +1)")
print(f"- {len(X2)} points for Class 2 (y = -1)")
print("The classes have overlapping distributions, making them non-linearly separable.")

# Step 3: Implement the Pocket Algorithm
print("\nStep 3: Implement the Pocket Algorithm")
print("----------------------------------")

def pocket_algorithm(X, y, max_iterations=1000, learning_rate=0.1):
    """
    Implement the Pocket Algorithm for classification.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values (+1 or -1)
    max_iterations : int
        Maximum number of iterations
    learning_rate : float
        Learning rate for weight updates
        
    Returns:
    --------
    pocket_weights : array, shape (n_features + 1,)
        Best weights found (including bias term as first element)
    final_weights : array, shape (n_features + 1,)
        Final weights after all iterations (including bias term)
    accuracy_history : list
        Accuracy at each iteration
    best_accuracy_history : list
        Best accuracy at each iteration
    """
    # Add a column of ones for the bias term
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Initialize weights randomly
    weights = np.random.randn(X_with_bias.shape[1])
    
    # Initialize pocket weights and best accuracy
    pocket_weights = weights.copy()
    best_accuracy = 0.0
    
    accuracy_history = []
    best_accuracy_history = []
    
    for iteration in range(max_iterations):
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X_shuffled = X_with_bias[indices]
        y_shuffled = y[indices]
        
        # Loop through all samples
        for i in range(len(X_shuffled)):
            # Make prediction
            prediction = np.sign(np.dot(X_shuffled[i], weights))
            
            # Update weights if misclassified
            if prediction != y_shuffled[i]:
                weights += learning_rate * y_shuffled[i] * X_shuffled[i]
        
        # Calculate current accuracy on the whole dataset
        predictions = np.sign(np.dot(X_with_bias, weights))
        accuracy = np.mean(predictions == y)
        accuracy_history.append(accuracy)
        
        # Update pocket weights if accuracy improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            pocket_weights = weights.copy()
        
        best_accuracy_history.append(best_accuracy)
    
    return pocket_weights, weights, accuracy_history, best_accuracy_history

# Train the Pocket Algorithm
print("Training the Pocket Algorithm on the non-separable dataset...")
pocket_weights, final_weights, accuracy_history, best_accuracy_history = pocket_algorithm(
    X, y, max_iterations=100, learning_rate=0.01
)

print(f"Pocket weights (best): [{pocket_weights[0]:.4f}, {pocket_weights[1]:.4f}, {pocket_weights[2]:.4f}]")
print(f"Final weights: [{final_weights[0]:.4f}, {final_weights[1]:.4f}, {final_weights[2]:.4f}]")
print(f"Best accuracy: {best_accuracy_history[-1]:.4f}")
print(f"Final accuracy: {accuracy_history[-1]:.4f}")

# Step 4: Visualize the accuracy history
print("\nStep 4: Visualize the accuracy history")
print("----------------------------------")

plt.figure(figsize=(12, 6))
iterations = range(1, len(accuracy_history) + 1)
plt.plot(iterations, accuracy_history, 'b-', alpha=0.5, label='Current Accuracy')
plt.plot(iterations, best_accuracy_history, 'r-', linewidth=2, label='Best Accuracy (Pocket)')

# Highlight the final best accuracy
plt.axhline(y=best_accuracy_history[-1], color='r', linestyle='--', alpha=0.7,
           label=f'Best Accuracy: {best_accuracy_history[-1]:.4f}')

# Add annotations
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Pocket Algorithm: Accuracy over Iterations', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the accuracy plot
plt.savefig(os.path.join(save_dir, "pocket_accuracy.png"), dpi=300, bbox_inches='tight')

print("Plotted the accuracy history of the Pocket Algorithm:")
print("- The blue line shows the accuracy of the current weights at each iteration")
print("- The red line shows the best accuracy achieved so far (pocket weights)")
print("- The dashed red line indicates the final best accuracy")
print(f"Note how the current accuracy fluctuates, but the pocket (best) accuracy")
print(f"only improves or stays the same, never decreases.")

# Step 5: Visualize the decision boundaries
print("\nStep 5: Visualize the decision boundaries")
print("-------------------------------------")

# Create a mesh grid for visualization
h = 0.05  # Step size in the mesh
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

# Function to plot the decision boundary
def plot_decision_boundary(weights, X, y, title, filename):
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a color map
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    
    # Create the decision boundary mesh
    X_mesh = np.c_[xx1.ravel(), xx2.ravel()]
    X_mesh_with_bias = np.hstack((np.ones((X_mesh.shape[0], 1)), X_mesh))
    Z = np.sign(np.dot(X_mesh_with_bias, weights))
    Z = Z.reshape(xx1.shape)
    
    # Plot the decision boundary and regions
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light, alpha=0.2)
    
    # Plot the data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5, label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', alpha=0.5, label='Class 2')
    
    # Add the decision boundary line
    slope = -weights[1] / weights[2]
    intercept = -weights[0] / weights[2]
    x1_line = np.array([x1_min, x1_max])
    x2_line = slope * x1_line + intercept
    plt.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision Boundary')
    
    # Highlight misclassified points
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = np.sign(np.dot(X_with_bias, weights))
    misclassified = predictions != y
    plt.scatter(X[misclassified, 0], X[misclassified, 1], color='yellow', 
               edgecolors='k', alpha=0.8, s=80, marker='x', label='Misclassified')
    
    # Count misclassified points
    num_misclassified = np.sum(misclassified)
    accuracy = 1 - num_misclassified / len(y)
    
    # Add the equation of the boundary
    eq_text = f"Decision Boundary Equation:\n{weights[1]:.4f}x₁ + {weights[2]:.4f}x₂ + {weights[0]:.4f} = 0"
    acc_text = f"Accuracy: {accuracy:.4f}\nMisclassified: {num_misclassified}/{len(y)}"
    plt.text(0.02, 0.98, eq_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.86, acc_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    
    return accuracy, num_misclassified

# Plot the decision boundaries
pocket_accuracy, pocket_misclassified = plot_decision_boundary(
    pocket_weights, X, y, 
    "Pocket Algorithm: Best Decision Boundary", 
    "pocket_decision_boundary.png"
)

final_accuracy, final_misclassified = plot_decision_boundary(
    final_weights, X, y,
    "Standard Perceptron: Final Decision Boundary",
    "perceptron_decision_boundary.png"
)

print(f"Plotted the decision boundaries for both algorithms:")
print(f"1. Pocket Algorithm (Best Weights):")
print(f"   - Equation: {pocket_weights[1]:.4f}x₁ + {pocket_weights[2]:.4f}x₂ + {pocket_weights[0]:.4f} = 0")
print(f"   - Accuracy: {pocket_accuracy:.4f}")
print(f"   - Misclassified: {pocket_misclassified}/{len(y)} points")
print(f"\n2. Standard Perceptron (Final Weights):")
print(f"   - Equation: {final_weights[1]:.4f}x₁ + {final_weights[2]:.4f}x₂ + {final_weights[0]:.4f} = 0")
print(f"   - Accuracy: {final_accuracy:.4f}")
print(f"   - Misclassified: {final_misclassified}/{len(y)} points")

# Step 6: Compare the two algorithms' boundaries
print("\nStep 6: Compare Pocket and Standard Perceptron boundaries")
print("----------------------------------------------------")

plt.figure(figsize=(12, 8))

# Plot the data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5, label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', alpha=0.5, label='Class 2')

# Plot the pocket decision boundary
slope_pocket = -pocket_weights[1] / pocket_weights[2]
intercept_pocket = -pocket_weights[0] / pocket_weights[2]
x1_line = np.array([x1_min, x1_max])
x2_line_pocket = slope_pocket * x1_line + intercept_pocket
plt.plot(x1_line, x2_line_pocket, 'g-', linewidth=2, label='Pocket Algorithm')

# Plot the standard perceptron decision boundary
slope_standard = -final_weights[1] / final_weights[2]
intercept_standard = -final_weights[0] / final_weights[2]
x2_line_standard = slope_standard * x1_line + intercept_standard
plt.plot(x1_line, x2_line_standard, 'r--', linewidth=2, label='Standard Perceptron')

# Add explanation text
plt.text(0.02, 0.98, 
         f"Pocket Algorithm:\nAccuracy: {pocket_accuracy:.4f}\nMisclassified: {pocket_misclassified}/{len(y)}", 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.text(0.02, 0.82, 
         f"Standard Perceptron:\nAccuracy: {final_accuracy:.4f}\nMisclassified: {final_misclassified}/{len(y)}", 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Add labels and title
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Comparison: Pocket Algorithm vs. Standard Perceptron', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the comparison plot
plt.savefig(os.path.join(save_dir, "boundary_comparison.png"), dpi=300, bbox_inches='tight')

print(f"Created a comparison plot of both decision boundaries:")
print(f"- The solid green line shows the Pocket Algorithm's best decision boundary")
print(f"- The dashed red line shows the Standard Perceptron's final decision boundary")
print(f"\nAs we can see, the Pocket Algorithm provides a better decision boundary")
print(f"with fewer misclassifications ({pocket_misclassified} vs. {final_misclassified}).")
print(f"This demonstrates the advantage of the Pocket Algorithm for non-separable data.")

# Step 7: Classify a new point
print("\nStep 7: Classify a new data point")
print("------------------------------")

# Define a new data point
new_point = np.array([3, 3])

# Create a function to classify a point
def classify_point(weights, point):
    """Classify a point using the given weights"""
    point_with_bias = np.hstack(([1], point))
    prediction = np.sign(np.dot(point_with_bias, weights))
    if prediction > 0:
        return 1, "Class 1"
    else:
        return -1, "Class 2"

# Classify using both models
pocket_class_num, pocket_class_name = classify_point(pocket_weights, new_point)
standard_class_num, standard_class_name = classify_point(final_weights, new_point)

# Plot the classification of the new point
plt.figure(figsize=(10, 8))

# Plot the data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.3, label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', alpha=0.3, label='Class 2')

# Plot the pocket decision boundary
plt.plot(x1_line, x2_line_pocket, 'g-', linewidth=2, label='Pocket Algorithm')

# Highlight the new point
point_color = 'blue' if pocket_class_num > 0 else 'red'
plt.scatter(new_point[0], new_point[1], color=point_color, s=150, marker='*', 
           edgecolors='k', linewidth=2, label=f'New Point: {pocket_class_name}')

# Add labels and title
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of a New Point using Pocket Algorithm', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower right')

# Add text about the classification
plt.text(0.02, 0.98, 
         f"New Point: ({new_point[0]}, {new_point[1]})\nPocket Classification: {pocket_class_name}\nStandard Classification: {standard_class_name}", 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the new point classification plot
plt.savefig(os.path.join(save_dir, "new_point_classification.png"), dpi=300, bbox_inches='tight')

print(f"Classified a new point at coordinates ({new_point[0]}, {new_point[1]}):")
print(f"- Pocket Algorithm classifies it as: {pocket_class_name}")
print(f"- Standard Perceptron classifies it as: {standard_class_name}")
print("\nThis demonstrates how the two models might make different predictions")
print("on new data points, especially in regions near the decision boundary.")

# Step 8: Analyze the bias-variance tradeoff
print("\nStep 8: Analyze the bias-variance tradeoff")
print("--------------------------------------")

print("The Pocket Algorithm vs. Standard Perceptron relationship connects to")
print("the bias-variance tradeoff in the following ways:")
print("\n1. The Standard Perceptron may have higher variance because:")
print("   - It's sensitive to the ordering of the training examples")
print("   - It may not converge for non-separable data, leading to unstable results")
print("   - The final weights depend heavily on the last few examples processed")
print("\n2. The Pocket Algorithm trades some bias for reduced variance:")
print("   - By keeping the best weights seen so far, it's less affected by the")
print("     specific ordering of examples in the later iterations")
print("   - It ensures a more stable solution by optimizing for accuracy over the")
print("     entire dataset, not just the most recent examples")
print("   - It effectively implements a form of early stopping based on validation")
print("     performance, a common technique to prevent overfitting")

# Step 9: Summary of findings
print("\nStep 9: Summary of findings")
print("------------------------")

print("1. The goal of the Pocket Algorithm is to find the best possible linear decision")
print("   boundary for non-separable data by keeping track of the weights that give the")
print("   highest classification accuracy.")
print("\n2. For the given non-separable dataset, the Pocket Algorithm achieves an accuracy")
print(f"   of {pocket_accuracy:.4f}, misclassifying {pocket_misclassified} out of {len(y)} points.")
print("\n3. The Standard Perceptron achieves an accuracy of {final_accuracy:.4f}, misclassifying")
print(f"   {final_misclassified} out of {len(y)} points.")
print(f"\n4. The Pocket Algorithm's decision boundary is given by:")
print(f"   {pocket_weights[1]:.4f}x₁ + {pocket_weights[2]:.4f}x₂ + {pocket_weights[0]:.4f} = 0")
print("\n5. This demonstrates the advantage of the Pocket Algorithm for non-separable data,")
print("   where the standard Perceptron may never converge to a stable solution.") 