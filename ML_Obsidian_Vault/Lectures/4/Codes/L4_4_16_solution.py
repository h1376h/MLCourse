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

# Define color scheme for improved visualizations
class1_color = '#4169E1'  # Royal blue
class2_color = '#FF6347'  # Tomato
noise_color = '#32CD32'   # Lime green
region1_color = '#B3C6FF'  # Light blue
region2_color = '#FFB3B3'  # Light red
custom_cmap = ListedColormap([region2_color, region1_color])

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

# Visualize separable data and both decision boundaries
plt.figure(figsize=(10, 8))

# Set up mesh grid for decision boundaries
x_min, x_max = X_separable[:, 0].min() - 1, X_separable[:, 0].max() + 1
y_min, y_max = X_separable[:, 1].min() - 1, X_separable[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Plot data points
plt.scatter(X_separable[y_separable == 1, 0], X_separable[y_separable == 1, 1], 
           color=class1_color, marker='o', s=70, label='Class +1', alpha=0.7)
plt.scatter(X_separable[y_separable == -1, 0], X_separable[y_separable == -1, 1], 
           color=class2_color, marker='x', s=70, label='Class -1', alpha=0.7)

# Plot decision boundaries
Z_perceptron = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], perceptron_w) + perceptron_b)
Z_perceptron = Z_perceptron.reshape(xx.shape)
plt.contour(xx, yy, Z_perceptron, colors='#e41a1c', linestyles=['-'], linewidths=2, levels=[0], alpha=0.8)

Z_pocket = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w) + pocket_b)
Z_pocket = Z_pocket.reshape(xx.shape)
plt.contour(xx, yy, Z_pocket, colors='#4daf4a', linestyles=['--'], linewidths=2, levels=[0], alpha=0.8)

# Add custom legend for boundaries
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#e41a1c', linestyle='-', linewidth=2),
    Line2D([0], [0], color='#4daf4a', linestyle='--', linewidth=2)
]
first_legend = plt.legend(custom_lines, ['Perceptron Boundary', 'Pocket Boundary'], 
                         loc='lower right', fontsize=12)
plt.gca().add_artist(first_legend)

# Add a second legend for data points
plt.legend(loc='upper left', fontsize=12)

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Linearly Separable Data with Decision Boundaries', fontsize=16)
plt.grid(True, alpha=0.3)

# Add accuracy information
plt.figtext(0.5, 0.01, f"Perceptron Accuracy: {perceptron_acc:.4f}  |  Pocket Accuracy: {pocket_acc:.4f}", 
           ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "separable_comparison.png"), dpi=300, bbox_inches='tight')

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

# Create visualization for noisy data comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Perceptron plot (left)
axes[0].scatter(X_separable[y_noisy == 1, 0], X_separable[y_noisy == 1, 1], 
               color=class1_color, marker='o', s=70, label='Class +1', alpha=0.7)
axes[0].scatter(X_separable[y_noisy == -1, 0], X_separable[y_noisy == -1, 1], 
               color=class2_color, marker='x', s=70, label='Class -1', alpha=0.7)

# Highlight the noisy points
axes[0].scatter(X_separable[noise_indices, 0], X_separable[noise_indices, 1], 
               color=noise_color, marker='s', s=100, alpha=0.7, label='Noisy Labels', edgecolors='k')

# Plot decision boundary
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], perceptron_w_noisy) + perceptron_b_noisy)
Z = Z.reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)
axes[0].contour(xx, yy, Z, colors='k', linewidths=2, levels=[0])

axes[0].set_title('Perceptron: Data with Label Noise', fontsize=16)
axes[0].set_xlabel('Feature 1', fontsize=14)
axes[0].set_ylabel('Feature 2', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=12)

# Add accuracy text
acc_text = f"Accuracy on noisy data: {perceptron_acc_noisy:.4f}\nAccuracy on clean data: {perceptron_acc_noisy_clean:.4f}"
axes[0].text(0.5, -0.1, acc_text, transform=axes[0].transAxes, ha='center', va='center', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Pocket plot (right)
axes[1].scatter(X_separable[y_noisy == 1, 0], X_separable[y_noisy == 1, 1], 
               color=class1_color, marker='o', s=70, label='Class +1', alpha=0.7)
axes[1].scatter(X_separable[y_noisy == -1, 0], X_separable[y_noisy == -1, 1], 
               color=class2_color, marker='x', s=70, label='Class -1', alpha=0.7)

# Highlight the noisy points
axes[1].scatter(X_separable[noise_indices, 0], X_separable[noise_indices, 1], 
               color=noise_color, marker='s', s=100, alpha=0.7, label='Noisy Labels', edgecolors='k')

# Plot decision boundary
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w_noisy) + pocket_b_noisy)
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)
axes[1].contour(xx, yy, Z, colors='k', linewidths=2, levels=[0])

axes[1].set_title('Pocket: Data with Label Noise', fontsize=16)
axes[1].set_xlabel('Feature 1', fontsize=14)
axes[1].set_ylabel('Feature 2', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12)

# Add accuracy text
acc_text = f"Accuracy on noisy data: {pocket_acc_noisy:.4f}\nAccuracy on clean data: {pocket_acc_noisy_clean:.4f}"
axes[1].text(0.5, -0.1, acc_text, transform=axes[1].transAxes, ha='center', va='center', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

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
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Set up mesh grid for decision boundaries
x_min, x_max = X_init[:, 0].min() - 0.5, X_init[:, 0].max() + 0.5
y_min, y_max = X_init[:, 1].min() - 0.5, X_init[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Plot first initialization and result
axes[0].scatter(X_init[y_init == 1, 0], X_init[y_init == 1, 1], 
              color=class1_color, marker='o', s=100, label='Class +1', alpha=0.7)
axes[0].scatter(X_init[y_init == -1, 0], X_init[y_init == -1, 1], 
              color=class2_color, marker='x', s=100, label='Class -1', alpha=0.7)

# Plot initial and final decision boundaries for first initialization
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w1) + b1)
Z = Z.reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)
axes[0].contour(xx, yy, Z, colors='blue', linestyles=['-'], linewidths=2, levels=[0])

# Plot initial decision line (as a reference)
init_Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], init1_w) + init1_b)
init_Z = init_Z.reshape(xx.shape)
axes[0].contour(xx, yy, init_Z, colors='grey', linestyles=['--'], linewidths=1, levels=[0])

# Add custom legend
custom_lines = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2)
]
first_legend = axes[0].legend(custom_lines, ['Initial Boundary', 'Final Boundary'], 
                         loc='lower right', fontsize=12)
axes[0].add_artist(first_legend)
axes[0].legend(loc='upper left', fontsize=12)

axes[0].set_title('Initialization 1: w=[0.1, 0.1], b=0', fontsize=16)
axes[0].set_xlabel('Feature 1', fontsize=14)
axes[0].set_ylabel('Feature 2', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Add accuracy text
axes[0].text(0.5, -0.1, f"Final Solution: w={w1}, b={b1}\nAccuracy: {acc1:.4f}", 
             transform=axes[0].transAxes, ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Plot second initialization and result
axes[1].scatter(X_init[y_init == 1, 0], X_init[y_init == 1, 1], 
              color=class1_color, marker='o', s=100, label='Class +1', alpha=0.7)
axes[1].scatter(X_init[y_init == -1, 0], X_init[y_init == -1, 1], 
              color=class2_color, marker='x', s=100, label='Class -1', alpha=0.7)

# Plot initial and final decision boundaries for second initialization
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w2) + b2)
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap)
axes[1].contour(xx, yy, Z, colors='green', linestyles=['-'], linewidths=2, levels=[0])

# Plot initial decision line (as a reference)
init_Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], init2_w) + init2_b)
init_Z = init_Z.reshape(xx.shape)
axes[1].contour(xx, yy, init_Z, colors='grey', linestyles=['--'], linewidths=1, levels=[0])

# Add custom legend
custom_lines = [
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1),
    Line2D([0], [0], color='green', linestyle='-', linewidth=2)
]
first_legend = axes[1].legend(custom_lines, ['Initial Boundary', 'Final Boundary'], 
                         loc='lower right', fontsize=12)
axes[1].add_artist(first_legend)
axes[1].legend(loc='upper left', fontsize=12)

axes[1].set_title('Initialization 2: w=[1.0, -0.5], b=0.5', fontsize=16)
axes[1].set_xlabel('Feature 1', fontsize=14)
axes[1].set_ylabel('Feature 2', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Add accuracy text
axes[1].text(0.5, -0.1, f"Final Solution: w={w2}, b={b2}\nAccuracy: {acc2:.4f}", 
             transform=axes[1].transAxes, ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "initialization_dependence.png"), dpi=300, bbox_inches='tight')

# Create a third visualization showing both solutions on the same plot for direct comparison
plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(X_init[y_init == 1, 0], X_init[y_init == 1, 1], 
           color=class1_color, marker='o', s=100, label='Class +1', alpha=0.7)
plt.scatter(X_init[y_init == -1, 0], X_init[y_init == -1, 1], 
           color=class2_color, marker='x', s=100, label='Class -1', alpha=0.7)

# Plot both final decision boundaries
plt.contour(xx, yy, np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w1) + b1).reshape(xx.shape), 
           colors='blue', linestyles=['-'], linewidths=2, levels=[0])
plt.contour(xx, yy, np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w2) + b2).reshape(xx.shape), 
           colors='green', linestyles=['--'], linewidths=2, levels=[0])

# Add custom legend
custom_lines = [
    Line2D([0], [0], color='blue', linestyle='-', linewidth=2),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2)
]
plt.legend(custom_lines, 
          ['Boundary from Init 1: w=[0.1, 0.1], b=0', 
           'Boundary from Init 2: w=[1.0, -0.5], b=0.5'], 
          loc='upper center', fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.title('Different Solutions from Different Initializations', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, "initialization_comparison.png"), dpi=300, bbox_inches='tight')

# Task 5: Empirical Risk Minimization visualization for Pocket algorithm
print("\nTask 5: Pocket Algorithm and Empirical Risk Minimization")
print("------------------------------------------------------")

# Generate a dataset with some class overlap
np.random.seed(123)
X_overlap = np.random.randn(100, 2)
w_orig = np.array([1, 1])
b_orig = 0
y_clean = np.sign(np.dot(X_overlap, w_orig) + b_orig + 0.1 * np.random.randn(100))

# Flip some labels to create non-separable data
flip_indices = np.random.choice(range(100), size=10, replace=False)
y_overlap = y_clean.copy()
y_overlap[flip_indices] *= -1

# Apply Pocket algorithm and track errors over iterations
pocket_w_overlap, pocket_b_overlap, _, pocket_acc_overlap, all_weights, best_weights = pocket_algorithm(X_overlap, y_overlap, max_iterations=50)

# Calculate errors for all weights in history
all_errors = []
best_errors = []
iterations = []
best_iter = 0

for i, (w, b) in enumerate(all_weights):
    preds = np.sign(np.dot(X_overlap, w) + b)
    error = 1 - np.mean(preds == y_overlap)
    all_errors.append(error)
    
    # Check if this is a best weight snapshot
    if i > 0 and error < min(all_errors[:-1]):
        best_errors.append(error)
        iterations.append(i)
        best_iter = i

# Visualize the error minimization
plt.figure(figsize=(12, 6))
plt.plot(range(len(all_errors)), all_errors, 'b-', alpha=0.5, label='Error rate after each update')
plt.scatter(iterations, best_errors, color='red', s=80, label='Best weights kept by Pocket', zorder=3)
plt.axhline(min(all_errors), color='green', linestyle='--', label='Minimum error rate achieved')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Error Rate (1 - Accuracy)', fontsize=14)
plt.title('Pocket Algorithm: Empirical Risk Minimization', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add annotation for minimum error
min_error = min(all_errors)
min_idx = all_errors.index(min_error)
plt.annotate(f'Minimum Error: {min_error:.4f}',
            xy=(min_idx, min_error), xytext=(min_idx+5, min_error+0.05),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
            fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "empirical_risk_minimization.png"), dpi=300, bbox_inches='tight')

# Now visualize the final decision boundaries from both algorithms on the overlap data
plt.figure(figsize=(10, 8))

# Apply standard Perceptron for comparison
perceptron_w_overlap, perceptron_b_overlap, _, _ = perceptron_algorithm(X_overlap, y_overlap, max_iterations=50)

# Create mesh grid for decision boundaries
x_min, x_max = X_overlap[:, 0].min() - 1, X_overlap[:, 0].max() + 1
y_min, y_max = X_overlap[:, 1].min() - 1, X_overlap[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Plot data points
plt.scatter(X_overlap[y_overlap == 1, 0], X_overlap[y_overlap == 1, 1], 
           color=class1_color, marker='o', s=70, label='Class +1', alpha=0.7)
plt.scatter(X_overlap[y_overlap == -1, 0], X_overlap[y_overlap == -1, 1], 
           color=class2_color, marker='x', s=70, label='Class -1', alpha=0.7)

# Highlight the flipped points
plt.scatter(X_overlap[flip_indices, 0], X_overlap[flip_indices, 1], 
           color=noise_color, marker='s', s=100, alpha=0.7, label='Flipped Labels', edgecolors='k')

# Plot decision boundaries
Z_perceptron = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], perceptron_w_overlap) + perceptron_b_overlap)
Z_perceptron = Z_perceptron.reshape(xx.shape)
plt.contour(xx, yy, Z_perceptron, colors='#e41a1c', linestyles=['-'], linewidths=2, levels=[0])

Z_pocket = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w_overlap) + pocket_b_overlap)
Z_pocket = Z_pocket.reshape(xx.shape)
plt.contour(xx, yy, Z_pocket, colors='#4daf4a', linestyles=['--'], linewidths=2, levels=[0])

# Add custom legend for boundaries
custom_lines = [
    Line2D([0], [0], color='#e41a1c', linestyle='-', linewidth=2),
    Line2D([0], [0], color='#4daf4a', linestyle='--', linewidth=2)
]
first_legend = plt.legend(custom_lines, ['Perceptron Boundary', 'Pocket Boundary (ERM)'], 
                         loc='lower right', fontsize=12)
plt.gca().add_artist(first_legend)

# Add second legend for points
plt.legend(loc='upper left', fontsize=12)

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Empirical Risk Minimization: Perceptron vs Pocket', fontsize=16)
plt.grid(True, alpha=0.3)

# Add accuracy information
perceptron_acc_overlap = np.mean(np.sign(np.dot(X_overlap, perceptron_w_overlap) + perceptron_b_overlap) == y_overlap)
plt.figtext(0.5, 0.01, f"Perceptron Accuracy: {perceptron_acc_overlap:.4f}  |  Pocket Accuracy: {pocket_acc_overlap:.4f}", 
           ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "erm_comparison.png"), dpi=300, bbox_inches='tight')