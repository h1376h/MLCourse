import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

print("Question 16: Perceptron vs Pocket Algorithm")
print("=========================================")

# Define the Perceptron algorithm
def perceptron_algorithm(X, y, max_iterations=1000, learning_rate=1.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    iterations = 0
    weights_history = [(w.copy(), b)]
    
    for _ in range(max_iterations):
        misclassified = 0
        for i in range(n_samples):
            y_pred = np.sign(np.dot(X[i], w) + b)
            if y[i] * y_pred <= 0:  # Misclassified
                w += learning_rate * y[i] * X[i]
                b += learning_rate * y[i]
                weights_history.append((w.copy(), b))
                misclassified += 1
        iterations += 1
        if misclassified == 0:
            break
    
    return w, b, iterations, weights_history

# Define the Pocket algorithm
def pocket_algorithm(X, y, max_iterations=1000, learning_rate=1.0):
    n_samples, n_features = X.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    b = 0
    
    # Initialize best weights (pocket)
    best_w = w.copy()
    best_b = b
    
    # Calculate initial accuracy
    predictions = np.sign(np.dot(X, w) + b)
    best_accuracy = np.mean(predictions == y)
    
    weights_history = [(w.copy(), b)]
    pocket_history = [(best_w.copy(), best_b, best_accuracy)]
    
    for _ in range(max_iterations):
        misclassified_indices = []
        
        # Find all misclassified samples
        for i in range(n_samples):
            y_pred = np.sign(np.dot(X[i], w) + b)
            if y[i] * y_pred <= 0:  # Misclassified
                misclassified_indices.append(i)
        
        # If no misclassifications, we're done
        if len(misclassified_indices) == 0:
            break
        
        # Pick a random misclassified sample
        i = np.random.choice(misclassified_indices)
        
        # Update weights
        w += learning_rate * y[i] * X[i]
        b += learning_rate * y[i]
        
        weights_history.append((w.copy(), b))
        
        # Calculate new accuracy
        predictions = np.sign(np.dot(X, w) + b)
        accuracy = np.mean(predictions == y)
        
        # Update pocket if better
        if accuracy > best_accuracy:
            best_w = w.copy()
            best_b = b
            best_accuracy = accuracy
            pocket_history.append((best_w.copy(), best_b, best_accuracy))
    
    return best_w, best_b, len(weights_history), best_accuracy, weights_history, pocket_history

# Task 1: Explanation of Pocket Algorithm
print("\nTask 1: Explanation of the Pocket Algorithm")
print("------------------------------------------")
print("The Pocket Algorithm is an improvement over the standard Perceptron for non-separable data.")
print("The key difference is that Pocket keeps track of the best performing weights encountered")
print("during training (the weights in the 'pocket'), even as it continues updating weights like")
print("the standard Perceptron. When training is done, it returns the best weights instead of the final weights.")
print("\nThis makes Pocket more robust for datasets where a perfect linear separator doesn't exist.")

# Task 2: Compare on linearly separable data
print("\nTask 2: Perceptron vs Pocket on Linearly Separable Data")
print("-----------------------------------------------------")

# Generate a linearly separable dataset
np.random.seed(42)
X_separable = np.random.randn(100, 2)
w_true = np.array([1, 2])
b_true = -5
y_separable = np.sign(np.dot(X_separable, w_true) + b_true)

# Apply both algorithms
perceptron_w, perceptron_b, perceptron_iter, perceptron_history = perceptron_algorithm(X_separable, y_separable)
pocket_w, pocket_b, pocket_iter, pocket_acc, pocket_all_weights, pocket_best_weights = pocket_algorithm(X_separable, y_separable)

# Calculate accuracies
perceptron_pred = np.sign(np.dot(X_separable, perceptron_w) + perceptron_b)
perceptron_acc = np.mean(perceptron_pred == y_separable)
pocket_pred = np.sign(np.dot(X_separable, pocket_w) + pocket_b)
pocket_acc = np.mean(pocket_pred == y_separable)

print(f"Perceptron:\n  Iterations: {perceptron_iter}\n  Accuracy: {perceptron_acc:.4f}")
print(f"Pocket:\n  Iterations: {pocket_iter}\n  Accuracy: {pocket_acc:.4f}")
print("\nFor linearly separable data, both algorithms achieve perfect accuracy.")

# Task 3: Add noise and compare
print("\nTask 3: Comparing Algorithms with Label Noise")
print("-------------------------------------------")

# Add 5% random noise to the labels
np.random.seed(42)
noise_indices = np.random.choice(len(y_separable), size=int(0.05 * len(y_separable)), replace=False)
y_noisy = y_separable.copy()
y_noisy[noise_indices] *= -1  # Flip the labels for the noise indices

# Apply both algorithms to noisy data
perceptron_w_noisy, perceptron_b_noisy, perceptron_iter_noisy, perceptron_history_noisy = perceptron_algorithm(
    X_separable, y_noisy, max_iterations=100)
pocket_w_noisy, pocket_b_noisy, pocket_iter_noisy, pocket_acc_noisy, pocket_all_weights_noisy, pocket_best_weights_noisy = pocket_algorithm(
    X_separable, y_noisy, max_iterations=100)

# Calculate accuracies on the original clean data
perceptron_pred_noisy = np.sign(np.dot(X_separable, perceptron_w_noisy) + perceptron_b_noisy)
perceptron_acc_noisy_clean = np.mean(perceptron_pred_noisy == y_separable)
pocket_pred_noisy = np.sign(np.dot(X_separable, pocket_w_noisy) + pocket_b_noisy)
pocket_acc_noisy_clean = np.mean(pocket_pred_noisy == y_separable)

# Calculate accuracies on the noisy data
perceptron_acc_noisy = np.mean(perceptron_pred_noisy == y_noisy)
pocket_acc_noisy = np.mean(pocket_pred_noisy == y_noisy)

print(f"Perceptron with noisy data:\n  Accuracy on noisy data: {perceptron_acc_noisy:.4f}\n  Accuracy on clean data: {perceptron_acc_noisy_clean:.4f}")
print(f"Pocket with noisy data:\n  Accuracy on noisy data: {pocket_acc_noisy:.4f}\n  Accuracy on clean data: {pocket_acc_noisy_clean:.4f}")
print("\nPocket typically outperforms the standard Perceptron on noisy data.")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Set up the grid for contour plots
x_min, x_max = X_separable[:, 0].min() - 1, X_separable[:, 0].max() + 1
y_min, y_max = X_separable[:, 1].min() - 1, X_separable[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Perceptron plot
axes[0].scatter(X_separable[y_noisy == 1, 0], X_separable[y_noisy == 1, 1], 
               color='blue', marker='o', label='Class +1')
axes[0].scatter(X_separable[y_noisy == -1, 0], X_separable[y_noisy == -1, 1], 
               color='red', marker='x', label='Class -1')

# Highlight the noisy points
axes[0].scatter(X_separable[noise_indices, 0], X_separable[noise_indices, 1], 
               color='green', marker='s', s=80, alpha=0.5, label='Noisy Labels')

# Plot decision boundary
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], perceptron_w_noisy) + perceptron_b_noisy)
Z = Z.reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
axes[0].contour(xx, yy, Z, colors='k', linewidths=1, levels=[-1, 0, 1])

axes[0].set_title('Perceptron: Data with Label Noise')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()

# Pocket plot
axes[1].scatter(X_separable[y_noisy == 1, 0], X_separable[y_noisy == 1, 1], 
               color='blue', marker='o', label='Class +1')
axes[1].scatter(X_separable[y_noisy == -1, 0], X_separable[y_noisy == -1, 1], 
               color='red', marker='x', label='Class -1')

# Highlight the noisy points
axes[1].scatter(X_separable[noise_indices, 0], X_separable[noise_indices, 1], 
               color='green', marker='s', s=80, alpha=0.5, label='Noisy Labels')

# Plot decision boundary
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w_noisy) + pocket_b_noisy)
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
axes[1].contour(xx, yy, Z, colors='k', linewidths=1, levels=[-1, 0, 1])

axes[1].set_title('Pocket: Data with Label Noise')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "noisy_comparison.png"), dpi=300, bbox_inches='tight')

# Task 4: Demonstrate initialization dependence of Perceptron
print("\nTask 4: Perceptron Initialization Dependence")
print("------------------------------------------")

# Function to run Perceptron with specific initialization
def perceptron_with_init(X, y, w_init, b_init, max_iter=100):
    w = w_init.copy()
    b = b_init
    history = [(w.copy(), b)]
    
    for _ in range(max_iter):
        misclassified = 0
        for i in range(len(X)):
            y_pred = np.sign(np.dot(X[i], w) + b)
            if y[i] * y_pred <= 0:
                w += y[i] * X[i]
                b += y[i]
                history.append((w.copy(), b))
                misclassified += 1
        if misclassified == 0:
            break
    
    return w, b, history

# Create a simple separable dataset with multiple possible boundaries
X_init = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [2, 2]
])
y_init = np.array([1, -1, -1, 1])

# Try two different initializations
init1_w = np.array([0.1, 0.1])
init1_b = 0
init2_w = np.array([1.0, -0.5])
init2_b = 0.5

w1, b1, history1 = perceptron_with_init(X_init, y_init, init1_w, init1_b)
w2, b2, history2 = perceptron_with_init(X_init, y_init, init2_w, init2_b)

# Calculate accuracy for both solutions
pred1 = np.sign(np.dot(X_init, w1) + b1)
acc1 = np.mean(pred1 == y_init)
pred2 = np.sign(np.dot(X_init, w2) + b2)
acc2 = np.mean(pred2 == y_init)

print(f"Initialization 1: w={init1_w}, b={init1_b}")
print(f"  Final solution: w={w1}, b={b1}")
print(f"  Accuracy: {acc1:.4f}")

print(f"\nInitialization 2: w={init2_w}, b={init2_b}")
print(f"  Final solution: w={w2}, b={b2}")
print(f"  Accuracy: {acc2:.4f}")

print("\nObservation: Different initializations can lead to different decision boundaries,")
print("even when both achieve perfect classification. This demonstrates that the Perceptron")
print("algorithm just finds any separating hyperplane, not necessarily a unique or optimal one.")

# Visualize the different solutions
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Set up mesh grid for decision boundaries
x_min, x_max = X_init[:, 0].min() - 0.5, X_init[:, 0].max() + 0.5
y_min, y_max = X_init[:, 1].min() - 0.5, X_init[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Plot first initialization and result
axes[0].scatter(X_init[y_init == 1, 0], X_init[y_init == 1, 1], 
              color='blue', marker='o', s=100, label='Class +1')
axes[0].scatter(X_init[y_init == -1, 0], X_init[y_init == -1, 1], 
              color='red', marker='x', s=100, label='Class -1')

# Plot initial and final decision boundaries
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w1) + b1)
Z = Z.reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
axes[0].contour(xx, yy, Z, colors='blue', linestyles=['-'], levels=[0])

axes[0].set_title('Perceptron Solution with Initialization 1')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()

# Plot second initialization and result
axes[1].scatter(X_init[y_init == 1, 0], X_init[y_init == 1, 1], 
              color='blue', marker='o', s=100, label='Class +1')
axes[1].scatter(X_init[y_init == -1, 0], X_init[y_init == -1, 1], 
              color='red', marker='x', s=100, label='Class -1')

# Plot initial and final decision boundaries
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w2) + b2)
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
axes[1].contour(xx, yy, Z, colors='green', linestyles=['-'], levels=[0])

axes[1].set_title('Perceptron Solution with Initialization 2')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "initialization_dependence.png"), dpi=300, bbox_inches='tight')