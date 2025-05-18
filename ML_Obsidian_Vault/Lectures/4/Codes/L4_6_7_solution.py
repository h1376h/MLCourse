import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# ==========================================
# Task 1: Calculate the cross-entropy loss
# ==========================================

# Define the function to calculate cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    """
    Calculate the cross-entropy loss
    
    Args:
        y_true: One-hot encoded true labels
        y_pred: Predicted probabilities
        
    Returns:
        Cross-entropy loss value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross-entropy loss
    ce_loss = -np.sum(y_true * np.log(y_pred))
    return ce_loss

# Given data for Task 1
y_true = np.array([0, 1, 0])  # true class is 2 (one-hot encoding)
y_pred = np.array([0.2, 0.5, 0.3])  # predicted probabilities

# Calculate the cross-entropy loss for the given data
loss_value = cross_entropy_loss(y_true, y_pred)
print("Task 1: Cross-entropy loss calculation")
print("-" * 40)
print(f"True class: 2 (one-hot encoded as {y_true})")
print(f"Predicted probabilities: p1 = {y_pred[0]}, p2 = {y_pred[1]}, p3 = {y_pred[2]}")
print(f"Cross-entropy loss: -{y_true[0]}*log({y_pred[0]}) - {y_true[1]}*log({y_pred[1]}) - {y_true[2]}*log({y_pred[2]})")
print(f"Cross-entropy loss: -0*log(0.2) - 1*log(0.5) - 0*log(0.3)")
print(f"Cross-entropy loss: -log(0.5) = {-np.log(0.5)}")
print(f"Cross-entropy loss: {loss_value}")
print()

# Visualize the example
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35

true_bars = ax.bar(x - width/2, y_true, width, label='True Class (One-hot)', color='blue', alpha=0.7)
pred_bars = ax.bar(x + width/2, y_pred, width, label='Predicted Probabilities', color='red', alpha=0.7)

ax.set_xlabel('Class')
ax.set_ylabel('Probability')
ax.set_title('Cross-entropy Loss Example: True Class vs Predicted Probabilities')
ax.set_xticks(x)
ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
ax.legend()

# Add text showing the loss value
ax.text(1.5, 0.8, f'Cross-entropy Loss = {loss_value:.4f}', 
        bbox=dict(facecolor='white', alpha=0.8),
        ha='center', va='center', fontsize=12)

plt.savefig(os.path.join(save_dir, 'task1_cross_entropy_example.png'), dpi=300, bbox_inches='tight')

# ==========================================
# Task 2: Derive the gradient of cross-entropy loss
# ==========================================

print("Task 2: Derive the gradient of cross-entropy loss")
print("-" * 40)
print("Derivation of the gradient:")

# Instead of using sympy for symbolic calculations, we'll just explain the derivation

print("For a single data point with true class 2 and 3 classes total:")
print("Let's use the softmax activation to compute predicted probabilities:")
print(f"p_k = exp(w_k^T x) / sum_j(exp(w_j^T x))")
print("The cross-entropy loss for this example is:")
print(f"L = -log(p_2)")
print()
print("The partial derivative of the loss with respect to p_2 is:")
print(f"dL/dp_2 = -1/p_2")
print()
print("The partial derivative of p_2 with respect to w_2 is:")
print(f"dp_2/dw_2 = p_2(1-p_2)x")
print()
print("For weights of other classes (k ≠ 2):")
print(f"dp_2/dw_k = -p_2*p_k*x")
print()
print("Using the chain rule, for the true class:")
print(f"dL/dw_2 = dL/dp_2 * dp_2/dw_2 = -1/p_2 * p_2(1-p_2)x = (p_2-1)x")
print()
print("For the other classes (k ≠ 2):")
print(f"dL/dw_k = dL/dp_2 * dp_2/dw_k = -1/p_2 * (-p_2*p_k*x) = p_k*x")
print()
print("In general, for any class k and true class y:")
print(f"dL/dw_k = (p_k - 1[k=y])x")
print("where 1[k=y] is 1 if k equals the true class y, and 0 otherwise.")
print()

# Create a visualization showing the gradient computation
# Visualizing the gradient of softmax with cross-entropy
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

# Create a simple scenario
w = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # weights for 3 classes
x = np.array([1.0, 1.0])  # example feature vector
true_class = 1  # class 2

# Calculate linear scores
z = np.dot(w, x)

# Calculate softmax probabilities
probs = softmax(z)

# Calculate gradient
gradient = np.zeros_like(w)
for k in range(3):
    if k == true_class:
        gradient[k] = (probs[k] - 1) * x
    else:
        gradient[k] = probs[k] * x

# Visualize the gradient calculation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Display weights, scores, and probabilities
ax1.bar(range(3), z, alpha=0.7, label='Linear Scores (z)')
ax1.bar(range(3), probs, alpha=0.7, label='Softmax Probabilities')
ax1.set_xticks(range(3))
ax1.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
ax1.set_title('Scores and Probabilities')
ax1.legend()

# Display gradient values
ax2.bar(range(3), [np.linalg.norm(gradient[i]) for i in range(3)], alpha=0.7)
for i in range(3):
    ax2.text(i, np.linalg.norm(gradient[i])/2, 
             f'{"(p-1)x" if i == true_class else "px"}',
             ha='center', va='center',
             color='black', fontweight='bold')
ax2.set_xticks(range(3))
ax2.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
ax2.set_title('Gradient Magnitudes')
ax2.set_ylabel('$||\\nabla L / \\nabla w_k||$')

plt.suptitle('Cross-entropy Gradient Computation')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task2_gradient_visualization.png'), dpi=300, bbox_inches='tight')

# ==========================================
# Task 3: Compare cross-entropy with squared error
# ==========================================

print("Task 3: Why is cross-entropy more appropriate than squared error?")
print("-" * 40)
print("Comparison of cross-entropy and squared error losses:")

# Generate predicted probabilities ranging from 0 to 1
p_range = np.linspace(0.01, 0.99, 100)

# Calculate cross-entropy and squared error for these probabilities
# Assume true class is 1 (y=1)
ce_loss = -np.log(p_range)  # Cross-entropy: -log(p)
se_loss = (1 - p_range)**2  # Squared error: (y-p)²

# Create a comparison plot
plt.figure(figsize=(10, 6))
plt.plot(p_range, ce_loss, label='Cross-Entropy Loss: -log(p)', color='blue', linewidth=2)
plt.plot(p_range, se_loss, label='Squared Error: (y-p)²', color='red', linewidth=2)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='p=0.5')
plt.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='p=0.1')

# Add text annotations for loss values at p=0.1
plt.text(0.12, -np.log(0.1)+0.2, f'CE({0.1})={-np.log(0.1):.2f}', color='blue')
plt.text(0.12, (1-0.1)**2+0.2, f'SE({0.1})={((1-0.1)**2):.2f}', color='red')

plt.xlabel('Predicted Probability (p) for True Class')
plt.ylabel('Loss Value')
plt.title('Comparison of Cross-Entropy and Squared Error Losses')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'task3_ce_vs_se.png'), dpi=300, bbox_inches='tight')

# Create a more complete visualization showing behavior for wrong predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Loss vs. predicted probability for the true class
ax1.plot(p_range, ce_loss, label='Cross-Entropy', color='blue', linewidth=2)
ax1.plot(p_range, se_loss, label='Squared Error', color='red', linewidth=2)
ax1.set_xlabel('Predicted Probability (p) for True Class')
ax1.set_ylabel('Loss Value')
ax1.set_title('Loss Functions (True Class)')
ax1.legend()
ax1.grid(True)

# Plot 2: Gradient magnitude vs. predicted probability
ce_gradient = -1/p_range
se_gradient = -2*(1-p_range)
ax2.plot(p_range, np.abs(ce_gradient), label='$|\\nabla$ Cross-Entropy$|$', color='blue', linewidth=2)
ax2.plot(p_range, np.abs(se_gradient), label='$|\\nabla$ Squared Error$|$', color='red', linewidth=2)
ax2.set_xlabel('Predicted Probability (p) for True Class')
ax2.set_ylabel('Gradient Magnitude')
ax2.set_title('Gradient Magnitudes')
ax2.legend()
ax2.grid(True)

plt.suptitle('Why Cross-Entropy is Better for Classification')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task3_loss_comparison_detailed.png'), dpi=300, bbox_inches='tight')

print("Cross-entropy loss has several advantages over squared error for classification:")
print("1. Cross-entropy provides much stronger penalties for confident wrong predictions")
print("2. The gradient of cross-entropy approaches infinity as p approaches 0, creating stronger learning signal")
print("3. Cross-entropy is derived from the principle of maximum likelihood estimation for probability distributions")
print("4. Squared error treats classes as ordered numerical values rather than discrete categories")
print()

# ==========================================
# Task 4: Show how multi-class cross-entropy reduces to binary cross-entropy
# ==========================================

print("Task 4: How does multi-class cross-entropy reduce to binary cross-entropy")
print("-" * 40)

# Create a visualization showing the reduction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Visualize multi-class case (K=3)
n_classes = 3
class_labels = [f'Class {i+1}' for i in range(n_classes)]
true_class_mc = np.array([0, 1, 0])  # Class 2 is true
probs_mc = np.array([0.2, 0.5, 0.3])  # Predicted probabilities

ax1.bar(class_labels, true_class_mc, alpha=0.7, label='True Class (One-hot)', color='blue')
ax1.bar(class_labels, probs_mc, alpha=0.7, label='Predicted Probabilities', color='red')
ax1.set_title('Multi-class Classification (K=3)')
ax1.text(0.5, 0.85, f'CE Loss = -{np.sum(true_class_mc * np.log(probs_mc)):.4f}',
        transform=ax1.transAxes, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.8))
ax1.legend()

# Visualize binary case (K=2)
class_labels_bin = ['Class 1', 'Class 2']
true_class_bin = np.array([0, 1])  # Class 2 is true
probs_bin = np.array([0.3, 0.7])  # Predicted probabilities

ax2.bar(class_labels_bin, true_class_bin, alpha=0.7, label='True Class (One-hot)', color='blue')
ax2.bar(class_labels_bin, probs_bin, alpha=0.7, label='Predicted Probabilities', color='red')
ax2.set_title('Binary Classification (K=2)')
ax2.text(0.5, 0.85, f'CE Loss = -{np.sum(true_class_bin * np.log(probs_bin)):.4f}',
        transform=ax2.transAxes, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.8))
ax2.legend()

plt.suptitle('Multi-class to Binary Cross-Entropy Reduction')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task4_mc_to_binary_reduction.png'), dpi=300, bbox_inches='tight')

# Instead of creating an image, print the mathematical transformation
print("Step-by-step reduction of multi-class cross-entropy to binary cross-entropy:")
print("1. Multi-class formula: L(w) = -∑ᵢ∑ₖ yᵢₖ log(pᵢₖ)")
print()
print("2. For K=2 classes, this becomes:")
print("   L(w) = -∑ᵢ [yᵢ₁ log(pᵢ₁) + yᵢ₂ log(pᵢ₂)]")
print()
print("3. For binary classification with K=2 classes:")
print("   • yᵢ₁ = 1-yᵢ (probability of class 1)")
print("   • yᵢ₂ = yᵢ (probability of class 2)")
print("   • pᵢ₁ = 1-pᵢ (predicted probability of class 1)")
print("   • pᵢ₂ = pᵢ (predicted probability of class 2)")
print()
print("4. Substituting these values:")
print("   L(w) = -∑ᵢ [(1-yᵢ)log(1-pᵢ) + yᵢlog(pᵢ)]")
print()
print("This is the standard binary cross-entropy formula used in logistic regression")
print("For a single example, we have L(w) = -(1-y)log(1-p) - ylog(p)")

print("Multi-class cross-entropy reduction to binary cross-entropy:")
print("For a multi-class problem with K=2 classes:")
print("1. Multi-class formula: L(w) = -∑ᵢ∑ₖ yᵢₖ log(pᵢₖ)")
print("2. For K=2, we have yᵢ₁ = 1-yᵢ₂, pᵢ₁ = 1-pᵢ₂")
print("3. Substituting: L(w) = -∑ᵢ [(1-yᵢ)log(1-pᵢ) + yᵢlog(pᵢ)]")
print("4. This is exactly the binary cross-entropy formula")
print()
print("All visualizations saved to:", save_dir) 