import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Part 1: Define the softmax function
def softmax(z):
    """
    Compute softmax values for each set of scores in z.
    
    Args:
        z: A numpy array of shape (n_samples, n_classes) or (n_classes,)
           containing the logits or scores
    
    Returns:
        Softmax probabilities of same shape as z
    """
    # For numerical stability, subtract max value before exponentiating
    # This prevents overflow when taking exponential of large numbers
    if z.ndim > 1:
        # Matrix case: apply softmax to each row
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    else:
        # Vector case: apply softmax to the vector
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z)

# Part 2: Calculate softmax for the given example
z = np.array([2, 0, 1])  # Given scores: z₁ = 2, z₂ = 0, z₃ = 1
probs = softmax(z)

print("\n===== Softmax Calculation Steps =====")
print(f"Step 1: Input scores z = {z}")

# Show the detailed calculation steps
shifted_z = z - np.max(z)
print(f"Step 2: Shift scores by subtracting max value {np.max(z)}: {shifted_z}")

exp_z = np.exp(shifted_z)
print(f"Step 3: Calculate exponentials: exp(z) = {exp_z}")

sum_exp_z = np.sum(exp_z)
print(f"Step 4: Sum of exponentials: {sum_exp_z}")

print(f"Step 5: Normalize by dividing each exp(z_i) by the sum: {exp_z} / {sum_exp_z}")
print(f"Final softmax probabilities: {probs}")
print(f"Sum of probabilities: {np.sum(probs)}")  # Verify sum = 1

# Part 3: Implement logistic function for comparison
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Part 4: Visualizations

# Visualization 1: Plot softmax vs. sigmoid for comparison
plt.figure(figsize=(12, 6))

# Sigmoid curve
x = np.linspace(-10, 10, 1000)
y_sigmoid = sigmoid(x)
plt.subplot(1, 2, 1)
plt.plot(x, y_sigmoid, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Input $z$', fontsize=12)
plt.ylabel(r'Output $\sigma(z)$', fontsize=12)
plt.title('Logistic Regression (Sigmoid Function)', fontsize=14)
plt.annotate('$\\sigma(z) = \\frac{1}{1 + e^{-z}}$', 
             xy=(0, 0.5), xytext=(3, 0.75),
             fontsize=14, arrowprops=dict(arrowstyle='->'))

# Softmax for three classes
z_range = np.linspace(-5, 5, 1000)
z_fixed = np.zeros((1000, 3))

# Set first class input to vary, others fixed
plt.subplot(1, 2, 2)
for i, fixed_val in enumerate([-2, 0, 2]):
    z_fixed[:, 1] = fixed_val  # Fix class 2 score
    z_fixed[:, 2] = 0  # Fix class 3 score
    z_fixed[:, 0] = z_range  # Vary class 1 score
    probs = np.array([softmax(z) for z in z_fixed])
    plt.plot(z_range, probs[:, 0], label=f'$z_2={fixed_val}, z_3=0$', linewidth=2)

plt.grid(True)
plt.xlabel('Input $z_1$', fontsize=12)
plt.ylabel('Probability for class 1', fontsize=12)
plt.title('Softmax Regression (Class 1 Probability)', fontsize=14)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'logistic_vs_softmax.png'), dpi=300, bbox_inches='tight')

# Visualization 2: 3D plot of softmax probabilities
fig = plt.figure(figsize=(15, 5))

# Create a 3D plot for each class probability
z1_range = np.linspace(-4, 4, 50)
z2_range = np.linspace(-4, 4, 50)
Z1, Z2 = np.meshgrid(z1_range, z2_range)
Z3 = np.zeros_like(Z1)  # Fix third class score to 0

# Calculate softmax probabilities for each position in the grid
probs = np.zeros((Z1.shape[0], Z1.shape[1], 3))
for i in range(Z1.shape[0]):
    for j in range(Z1.shape[1]):
        z = np.array([Z1[i, j], Z2[i, j], Z3[i, j]])
        probs[i, j, :] = softmax(z)

# Plot probability surface for each class
class_names = ['Class 1', 'Class 2', 'Class 3']
for idx in range(3):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    surf = ax.plot_surface(Z1, Z2, probs[:, :, idx], cmap=cm.viridis, 
                          linewidth=0, antialiased=True, alpha=0.7)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    
    # Set labels and title
    ax.set_xlabel('$z_1$', fontsize=12)
    ax.set_ylabel('$z_2$', fontsize=12)
    ax.set_zlabel(f'$P({idx+1})$', fontsize=12)
    ax.set_title(f'Softmax Probability for {class_names[idx]}', fontsize=14)
    
    # Mark the example point (z₁=2, z₂=0, z₃=0) on the plot
    if idx == 0:
        example_x, example_y = 2, 0
        example_z = softmax(np.array([example_x, example_y, 0]))[idx]
        ax.scatter([example_x], [example_y], [example_z], color='r', s=100, marker='o')
        ax.text(example_x, example_y, example_z + 0.1, f'Example: P(1)={example_z:.4f}', color='r')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Softmax Probabilities as a Function of $z_1$ and $z_2$ (with $z_3=0$)', fontsize=16)
plt.subplots_adjust(top=0.85)

plt.savefig(os.path.join(save_dir, 'softmax_3d.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Bar chart for our specific example
plt.figure(figsize=(10, 6))
classes = ['Class 1', 'Class 2', 'Class 3']
scores = np.array([2, 0, 1])  # Given scores
probabilities = softmax(scores)

# Plot the scores
plt.subplot(1, 2, 1)
plt.bar(classes, scores, color='skyblue')
plt.title('Raw Scores (z)', fontsize=14)
plt.ylabel('Score Value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot the corresponding probabilities
plt.subplot(1, 2, 2)
bars = plt.bar(classes, probabilities, color='salmon')
plt.title('Softmax Probabilities', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add probability values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=12)

# Add a line showing sum = 1
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
plt.text(1.75, 1.02, 'Sum = 1', color='red', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'example_probabilities.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Animation of how probabilities change with scores
# We'll create a series of plots showing how changing one score affects all probabilities

plt.figure(figsize=(12, 8))

# Initial scores
base_scores = np.array([2, 0, 1])
changing_class = 0  # We'll vary the score of class 1

# Range of score values to try
delta_range = np.linspace(-4, 4, 9)
num_plots = len(delta_range)
rows = 3
cols = 3

for i, delta in enumerate(delta_range):
    # Update scores
    current_scores = base_scores.copy()
    current_scores[changing_class] = base_scores[changing_class] + delta
    
    # Calculate probabilities
    current_probs = softmax(current_scores)
    
    # Create subplot
    plt.subplot(rows, cols, i + 1)
    plt.bar(classes, current_probs, color='salmon')
    
    # Display scores and probabilities
    plt.title(f'z = [{current_scores[0]:.1f}, {current_scores[1]:.1f}, {current_scores[2]:.1f}]', fontsize=12)
    plt.ylim(0, 1)
    
    # Only add y-axis label for leftmost subplots
    if i % cols == 0:
        plt.ylabel('Probability', fontsize=12)
    
    # Only add x-axis labels for bottom row
    if i >= num_plots - cols:
        plt.xticks(rotation=45)
    else:
        plt.xticks([])
    
    # Add sum text
    plt.text(1, 0.9, f'Sum = {np.sum(current_probs):.4f}', fontsize=10)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.suptitle('Effect of Varying $z_1$ on All Class Probabilities', fontsize=16, y=1.02)
plt.subplots_adjust(top=0.9)

plt.savefig(os.path.join(save_dir, 'varying_scores.png'), dpi=300, bbox_inches='tight')

# Add a new simple visualization: Decision boundaries visualization
plt.figure(figsize=(10, 8))
# Create a 2D grid
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# Create weights for 3 classes (simplified linear model)
W = np.array([
    [1.0, 0.5],    # Class 1: strong weight on x1
    [-0.5, 1.0],   # Class 2: strong weight on x2
    [-0.5, -1.5]   # Class 3: negative weights
])
b = np.array([0.0, 0.0, 0.0])  # Bias terms

# Calculate scores for each point
Z = np.zeros((100, 100, 3))
for i in range(100):
    for j in range(100):
        point = np.array([X1[i, j], X2[i, j]])
        for k in range(3):
            Z[i, j, k] = np.dot(W[k], point) + b[k]

# Apply softmax to get probabilities
probs = np.zeros_like(Z)
for i in range(100):
    for j in range(100):
        probs[i, j] = softmax(Z[i, j])

# Find the predicted class (argmax of probabilities)
predicted_class = np.argmax(probs, axis=2)

# Plot decision boundaries (without text labels)
plt.contourf(X1, X2, predicted_class, alpha=0.8, cmap=plt.cm.viridis)
plt.colorbar(ticks=[0, 1, 2])

# Plot some sample points for each class
for c in range(3):
    indices = np.where(predicted_class == c)
    sample_indices = np.random.choice(len(indices[0]), 20, replace=False)
    x1_samples = X1[indices[0][sample_indices], indices[1][sample_indices]]
    x2_samples = X2[indices[0][sample_indices], indices[1][sample_indices]]
    plt.scatter(x1_samples, x2_samples, s=50, c=f'C{c}', edgecolors='k')

plt.grid(True)
plt.title('Softmax Decision Boundaries', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'softmax_decision_boundaries.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations have been saved to: {save_dir}")

# Additional mathematical insights
print("\n===== Mathematical Insights =====")
print("1. Difference between softmax and logistic regression:")
print("   Logistic regression is a binary classifier using the sigmoid function,")
print("   while softmax regression extends to multiple classes using the softmax function.")

print("\n2. Softmax function formula:")
print("   softmax(z_i) = e^{z_i} / ∑_j e^{z_j}")

print("\n3. Why softmax probabilities sum to 1:")
print("   Because we divide each exponential by the sum of all exponentials,")
print("   so ∑_i (e^{z_i} / ∑_j e^{z_j}) = (∑_i e^{z_i}) / (∑_j e^{z_j}) = 1")

print("\n4. Relationship to multinomial logistic regression:")
print("   Softmax regression is also called multinomial logistic regression,")
print("   because it extends logistic regression to multiple classes.")

print("\n5. Cross-entropy loss for softmax regression:")
print("   L(y, y_hat) = -∑_i y_i * log(y_hat_i)")
print("   where y is the one-hot encoded true label and y_hat is softmax output.") 