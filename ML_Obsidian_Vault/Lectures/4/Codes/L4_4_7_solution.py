import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 7: Pocket Algorithm")
print("============================")

# Step 1: Explain the difference between Perceptron and Pocket Algorithm
print("\nStep 1: Difference between standard Perceptron and Pocket Algorithm")
print("----------------------------------------------------------------")
print("Standard Perceptron:")
print("- Updates weights immediately when a misclassification is found")
print("- Final weights are simply the last weights after all iterations")
print("- Works well for linearly separable data but may not converge for non-separable data")
print("\nPocket Algorithm:")
print("- Also updates weights when a misclassification is found")
print("- BUT keeps track of the best weights seen so far (i.e., weights that correctly classify the most points)")
print("- Returns the best weights rather than the final weights")
print("- Better suited for non-separable data")

# Step 2: Visualize how the algorithm works with the given weight vectors
print("\nStep 2: Tracking weight vectors and their performance")
print("--------------------------------------------------")

# Data from the problem
weights = [
    np.array([1, 2]),    # w1
    np.array([0, 3]),    # w2
    np.array([2, 1]),    # w3
    np.array([-1, 4]),   # w4
    np.array([3, 0])     # w5
]

correct_counts = [6, 7, 5, 8, 7]  # Number of correctly classified points out of 10
total_points = 10

# Create a plot to track the performance over iterations
plt.figure(figsize=(10, 6))
iterations = range(1, len(weights) + 1)
plt.plot(iterations, correct_counts, marker='o', linestyle='-', linewidth=2, markersize=10)
plt.axhline(y=max(correct_counts), color='r', linestyle='--', alpha=0.7, label=f'Best performance: {max(correct_counts)}/{total_points}')

# Mark the iteration with the best performance
best_iteration = correct_counts.index(max(correct_counts)) + 1
plt.scatter([best_iteration], [max(correct_counts)], s=200, c='red', zorder=5, label=f'Best weights: w{best_iteration}')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Correctly Classified Points', fontsize=14)
plt.title('Pocket Algorithm Performance Tracking', fontsize=16)
plt.xticks(iterations)
plt.yticks(range(0, total_points + 1))
plt.ylim(0, total_points + 0.5)
plt.grid(True)
plt.legend(fontsize=12)

# Save the performance plot
plt.savefig(os.path.join(save_dir, "pocket_performance.png"), dpi=300, bbox_inches='tight')

print("Weight vectors and their performance:")
for i, (w, count) in enumerate(zip(weights, correct_counts)):
    print(f"w{i+1} = {w}, correctly classifies {count}/{total_points} points")

best_idx = correct_counts.index(max(correct_counts))
print(f"\nBest performing weights: w{best_idx+1} = {weights[best_idx]}, correctly classifies {correct_counts[best_idx]}/{total_points} points")

# Step 3: Visualize the weight vectors in feature space
print("\nStep 3: Visualizing weight vectors in feature space")
print("------------------------------------------------")

# Create a 2D plot to visualize the weight vectors
plt.figure(figsize=(10, 8))

# Plot weight vectors as arrows from origin
for i, w in enumerate(weights):
    plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.3, fc=f'C{i}', ec=f'C{i}', 
              alpha=0.7, width=0.03, length_includes_head=True, label=f'w{i+1} = {w}')

# Highlight the best weight vector
plt.arrow(0, 0, weights[best_idx][0], weights[best_idx][1], head_width=0.3, head_length=0.4, 
          fc='red', ec='red', alpha=1.0, width=0.05, length_includes_head=True, 
          label=f'Best: w{best_idx+1} = {weights[best_idx]}')

# Set up the plot
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'$w_1$', fontsize=14)
plt.ylabel(r'$w_2$', fontsize=14)
plt.title('Weight Vectors in Feature Space', fontsize=16)
plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 4)
plt.ylim(-1, 5)
plt.legend(fontsize=10, loc='upper left')

# Save the weight vectors plot
plt.savefig(os.path.join(save_dir, "weight_vectors.png"), dpi=300, bbox_inches='tight')

# Step 4: Compare Pocket Algorithm with standard Perceptron
print("\nStep 4: Comparing Pocket Algorithm with standard Perceptron")
print("--------------------------------------------------------")

# Simulating what standard Perceptron would do
def accuracy(weight_vector, iteration):
    """Return the accuracy for visualization purposes only"""
    accuracies = [0.6, 0.7, 0.5, 0.8, 0.7]  # Based on given correct counts
    return accuracies[iteration]

# Visualize Perceptron vs Pocket algorithm performance
plt.figure(figsize=(12, 6))

# Plot Pocket performance
plt.plot(iterations, correct_counts, marker='o', linestyle='-', color='blue', 
         linewidth=2, markersize=10, label='Pocket performance')

# Mark Pocket's best weights
plt.scatter([best_iteration], [max(correct_counts)], s=200, c='blue', marker='*', 
            zorder=5, label=f'Pocket best: w{best_iteration}')

# Plot standard Perceptron performance (which would just have the final weights)
plt.plot(iterations, correct_counts, marker='x', linestyle='--', color='green', 
         linewidth=2, markersize=10, alpha=0.5)

# Mark Perceptron's final weights
plt.scatter([len(weights)], [correct_counts[-1]], s=200, c='green', marker='*', 
            zorder=5, label=f'Perceptron final: w{len(weights)}')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Correctly Classified Points', fontsize=14)
plt.title('Pocket Algorithm vs. Standard Perceptron', fontsize=16)
plt.xticks(iterations)
plt.yticks(range(0, total_points + 1))
plt.ylim(0, total_points + 0.5)
plt.grid(True)
plt.legend(fontsize=12, loc='lower right')

# Save the comparison plot
plt.savefig(os.path.join(save_dir, "pocket_vs_perceptron.png"), dpi=300, bbox_inches='tight')

print("Standard Perceptron would keep the final weights w5 = [3, 0], which correctly classifies 7/10 points.")
print("Pocket Algorithm would keep the best weights w4 = [-1, 4], which correctly classifies 8/10 points.")
print("\nThis illustrates the key difference: Pocket Algorithm returns the weights that performed best during training,")
print("while standard Perceptron simply returns the final weights after all iterations.")

# Step 5: Simulating a simple 2D non-separable dataset to demonstrate
print("\nStep 5: Demonstrating Pocket Algorithm on a 2D non-separable dataset")
print("-------------------------------------------------------------------")

# Create a synthetic dataset with some non-separable points
np.random.seed(42)

# Class 1 centered at (1, 1)
n_samples1 = 30
X1 = np.random.randn(n_samples1, 2) * 0.7 + np.array([1, 1])

# Class 2 centered at (3, 3)
n_samples2 = 30
X2 = np.random.randn(n_samples2, 2) * 0.7 + np.array([3, 3])

# Add some overlapping points to make it non-separable
overlap_samples = 10
X1_overlap = np.random.randn(overlap_samples, 2) * 0.5 + np.array([2.5, 2.5])
X2_overlap = np.random.randn(overlap_samples, 2) * 0.5 + np.array([1.5, 1.5])

X1 = np.vstack([X1, X2_overlap])
X2 = np.vstack([X2, X1_overlap])

# Combine data and create labels
X = np.vstack([X1, X2])
y = np.hstack([np.ones(X1.shape[0]), -np.ones(X2.shape[0])])

# Perceptron algorithm implementation
def perceptron(X, y, max_iter=100, pocket=False):
    """
    Implement the Perceptron algorithm with option for Pocket Algorithm
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    max_iter : int, default=100
        Maximum number of iterations
    pocket : bool, default=False
        Whether to use Pocket Algorithm
        
    Returns:
    --------
    w : array, shape (n_features,)
        Weights
    b : float
        Bias
    pocket_w : array, shape (n_features,)
        Best weights (if pocket=True)
    pocket_b : float
        Best bias (if pocket=True)
    accuracies : list
        List of accuracies for each iteration
    """
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0
    
    # For Pocket Algorithm
    pocket_w = w.copy()
    pocket_b = b
    best_correct = 0
    
    accuracies = []
    
    for _ in range(max_iter):
        misclassified = 0
        correct = 0
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            
            # Predict
            y_pred = np.sign(np.dot(w, xi) + b)
            
            # Update if misclassified
            if yi * y_pred <= 0:
                w = w + yi * xi
                b = b + yi
                misclassified += 1
            else:
                correct += 1
        
        accuracy = correct / n_samples
        accuracies.append(accuracy)
        
        # Update pocket weights if better
        if pocket and correct > best_correct:
            best_correct = correct
            pocket_w = w.copy()
            pocket_b = b
            
        # Early stopping if all points are correctly classified
        if misclassified == 0:
            break
    
    if pocket:
        return w, b, pocket_w, pocket_b, accuracies
    else:
        return w, b, accuracies

# Run standard Perceptron and Pocket Algorithm
w, b, standard_accuracies = perceptron(X, y, max_iter=20, pocket=False)
_, _, pocket_w, pocket_b, pocket_accuracies = perceptron(X, y, max_iter=20, pocket=True)

# Visualize the results
# Make sure both lists have the same length to avoid dimension mismatch
iterations = np.arange(1, min(len(standard_accuracies), len(pocket_accuracies)) + 1)
standard_accuracies = standard_accuracies[:len(iterations)]
pocket_accuracies = pocket_accuracies[:len(iterations)]

plt.figure(figsize=(12, 6))
plt.plot(iterations, standard_accuracies, marker='o', linestyle='-', color='green', 
         linewidth=2, markersize=8, alpha=0.7, label='Standard Perceptron')
plt.plot(iterations, pocket_accuracies, marker='x', linestyle='-', color='blue', 
         linewidth=2, markersize=8, label='Pocket Algorithm')

best_pocket_accuracy = max(pocket_accuracies)
best_iter = pocket_accuracies.index(best_pocket_accuracy) + 1
final_standard_accuracy = standard_accuracies[-1]

plt.axhline(y=best_pocket_accuracy, color='blue', linestyle='--', alpha=0.5, 
           label=f'Best Pocket Accuracy: {best_pocket_accuracy:.2f}')
plt.axhline(y=final_standard_accuracy, color='green', linestyle='--', alpha=0.5, 
           label=f'Final Perceptron Accuracy: {final_standard_accuracy:.2f}')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy Comparison: Standard Perceptron vs. Pocket Algorithm', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)

# Save the accuracy comparison plot
plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')

# Plot the dataset and decision boundaries
plt.figure(figsize=(10, 8))

# Plot the data points
plt.scatter(X1[:, 0], X1[:, 1], c='blue', s=50, marker='o', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='red', s=50, marker='x', label='Class 2')

# Create a mesh grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot the decision boundary for the final standard Perceptron
Z_standard = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b).reshape(xx.shape)
plt.contour(xx, yy, Z_standard, colors=['green'], levels=[0], alpha=0.7, linestyles=['--'], 
            linewidths=2, label='Standard Perceptron')

# Plot the decision boundary for the Pocket Algorithm
Z_pocket = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w) + pocket_b).reshape(xx.shape)
plt.contour(xx, yy, Z_pocket, colors=['blue'], levels=[0], alpha=0.9, linestyles=['-'],
            linewidths=2, label='Pocket Algorithm')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Decision Boundaries: Standard Perceptron vs. Pocket Algorithm', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)

# Add text explaining the boundaries
plt.text(0.05, 0.95, 'Standard Perceptron: green dashed line\nPocket Algorithm: blue solid line',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the decision boundaries plot
plt.savefig(os.path.join(save_dir, "decision_boundaries.png"), dpi=300, bbox_inches='tight')

print("Simulation Results:")
print(f"Standard Perceptron final accuracy: {final_standard_accuracy:.2f}")
print(f"Pocket Algorithm best accuracy: {best_pocket_accuracy:.2f} (at iteration {best_iter})")
print("\nThe visualization shows how the Pocket Algorithm retains the best decision boundary")
print("encountered during training, which often results in better classification performance")
print("compared to the standard Perceptron algorithm, especially for non-separable data.")

# Step 6: Summary of findings
print("\nStep 6: Summary and conclusion")
print("--------------------------")
print("Based on the given sequence of weight vectors:")
print("- w1 = [1, 2], correctly classifies 6/10 points")
print("- w2 = [0, 3], correctly classifies 7/10 points")
print("- w3 = [2, 1], correctly classifies 5/10 points")
print("- w4 = [-1, 4], correctly classifies 8/10 points")
print("- w5 = [3, 0], correctly classifies 7/10 points")
print("\nThe Pocket Algorithm would retain w4 = [-1, 4] as it correctly classifies the most points (8/10).")
print("The standard Perceptron would have w5 = [3, 0] after these iterations, which only correctly classifies 7/10 points.")
print("\nThis demonstrates the key advantage of the Pocket Algorithm: it keeps track of the best weights")
print("seen during training, rather than just using the final weights, which makes it more robust")
print("for datasets that are not linearly separable.")

print("\nConclusion:")
print("-----------")
print("The Pocket Algorithm is an enhancement to the standard Perceptron algorithm that makes it")
print("suitable for datasets that are not linearly separable. It does this by keeping track of the")
print("best-performing weights during training, rather than just returning the final weights.")
print("\nThis is particularly important because for non-separable data, the standard Perceptron")
print("will continue to update weights indefinitely without converging to a stable solution.")
print("The Pocket Algorithm ensures we at least get the best possible linear classifier given the constraints.") 