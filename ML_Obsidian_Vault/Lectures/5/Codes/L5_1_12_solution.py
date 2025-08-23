import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import expit  # sigmoid function

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 12: Decision Function Analysis")
print("=" * 80)

# Define a sample SVM hyperplane for demonstration
# w = [3, 4], b = -5 (normalized so ||w|| = 5)
w = np.array([3, 4])
b = -5
w_norm = np.linalg.norm(w)

print(f"Sample SVM hyperplane:")
print(f"Weight vector w = {w}")
print(f"Bias term b = {b}")
print(f"||w|| = {w_norm}")
print(f"Decision function: f(x) = {w[0]}*x1 + {w[1]}*x2 + {b}")

# Task 1: What does |f(x)| represent?
print("\n" + "="*50)
print("Task 1: Meaning of |f(x)|")
print("="*50)

def decision_function(x, w, b):
    """Compute the decision function f(x) = w^T x + b"""
    return np.dot(w, x) + b

def distance_to_hyperplane(x, w, b):
    """Compute the distance from point x to hyperplane w^T x + b = 0"""
    return abs(np.dot(w, x) + b) / np.linalg.norm(w)

# Sample points for demonstration
sample_points = np.array([
    [1, 1],    # Point A
    [2, 0],    # Point B  
    [0, 2],    # Point C
    [-1, 0],   # Point D
    [0, -1]    # Point E
])

print("Sample points and their decision values:")
print("Point\t\tf(x)\t\t|f(x)|\t\tDistance")
print("-" * 60)

for i, point in enumerate(sample_points):
    f_val = decision_function(point, w, b)
    f_abs = abs(f_val)
    dist = distance_to_hyperplane(point, w, b)
    print(f"Point {chr(65+i)} {point}\t{f_val:.3f}\t\t{f_abs:.3f}\t\t{dist:.3f}")

print(f"\nKey insight: |f(x)| is proportional to the distance from x to the hyperplane")
print(f"Specifically: distance = |f(x)| / ||w|| = |f(x)| / {w_norm:.3f}")

# Task 2: Derive the relationship between |f(x)| and distance
print("\n" + "="*50)
print("Task 2: Relationship between |f(x)| and distance")
print("="*50)

print("Mathematical derivation:")
print("1. The hyperplane equation is: w^T x + b = 0")
print("2. For any point x0, the distance to the hyperplane is:")
print("   d = |w^T x0 + b| / ||w||")
print("3. Since f(x0) = w^T x0 + b, we have:")
print("   d = |f(x0)| / ||w||")
print("4. Therefore: |f(x)| = ||w|| * distance(x, hyperplane)")

# Verification with our sample points
print(f"\nVerification:")
print(f"For our hyperplane with ||w|| = {w_norm}:")
for i, point in enumerate(sample_points):
    f_val = decision_function(point, w, b)
    calculated_dist = abs(f_val) / w_norm
    actual_dist = distance_to_hyperplane(point, w, b)
    print(f"Point {chr(65+i)}: |f(x)|/||w|| = {calculated_dist:.6f}, actual distance = {actual_dist:.6f}")

# Task 3: Classification confidence comparison
print("\n" + "="*50)
print("Task 3: Classification Confidence")
print("="*50)

# Given test points
x1_val = 0.1
x2_val = 2.5

print(f"Given decision values:")
print(f"f(x1) = {x1_val}")
print(f"f(x2) = {x2_val}")
print(f"|f(x1)| = {abs(x1_val)}")
print(f"|f(x2)| = {abs(x2_val)}")

print(f"\nClassification confidence interpretation:")
print(f"- Both points are classified as positive (f(x) > 0)")
print(f"- Point x2 has higher confidence because |f(x2)| = {abs(x2_val)} > |f(x1)| = {abs(x1_val)}")
print(f"- Point x2 is farther from the decision boundary")
print(f"- Distance ratio: x2 is {abs(x2_val)/abs(x1_val):.1f}x farther from boundary than x1")

# Task 4: Converting to probabilistic outputs
print("\n" + "="*50)
print("Task 4: Converting to Probabilistic Outputs")
print("="*50)

def platt_scaling(f_vals, A=-1.0, B=0.0):
    """
    Convert SVM decision values to probabilities using Platt scaling
    P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
    """
    return 1.0 / (1.0 + np.exp(A * f_vals + B))

def sigmoid_scaling(f_vals, scale=1.0):
    """
    Simple sigmoid scaling: P(y=1|x) = sigmoid(scale * f(x))
    """
    return expit(scale * f_vals)

# Demonstrate different scaling methods
f_values = np.array([x1_val, x2_val])
print("Method 1: Platt Scaling (requires calibration on validation set)")
platt_probs = platt_scaling(f_values)
print(f"f(x1) = {x1_val} → P(y=1|x1) = {platt_probs[0]:.4f}")
print(f"f(x2) = {x2_val} → P(y=1|x2) = {platt_probs[1]:.4f}")

print("\nMethod 2: Simple Sigmoid Scaling")
sigmoid_probs = sigmoid_scaling(f_values)
print(f"f(x1) = {x1_val} → P(y=1|x1) = {sigmoid_probs[0]:.4f}")
print(f"f(x2) = {x2_val} → P(y=1|x2) = {sigmoid_probs[1]:.4f}")

print("\nNote: The exact calibration parameters (A, B) should be learned from validation data")

# Task 5: Effect of scaling weight vector
print("\n" + "="*50)
print("Task 5: Effect of Scaling Weight Vector")
print("="*50)

# Scale the weight vector by different factors
scale_factors = [0.5, 1.0, 2.0, 5.0]
test_point = np.array([1, 1])

print("Effect of scaling w on decision values and distances:")
print("Scale\tw_scaled\t\tf(x)\t\t|f(x)|\t\tDistance")
print("-" * 80)

for scale in scale_factors:
    w_scaled = scale * w
    b_scaled = scale * b  # Scale bias proportionally
    f_val = decision_function(test_point, w_scaled, b_scaled)
    dist = distance_to_hyperplane(test_point, w_scaled, b_scaled)
    print(f"{scale:.1f}\t{w_scaled}\t{f_val:.3f}\t\t{abs(f_val):.3f}\t\t{dist:.3f}")

print(f"\nKey insights:")
print(f"- When we scale w by factor k, f(x) scales by factor k")
print(f"- But distance remains unchanged (both numerator and denominator scale by k)")
print(f"- This means confidence ranking is preserved, but absolute values change")
print(f"- For probability calibration, scaling affects the calibration parameters")

# Create visualizations
print("\n" + "="*50)
print("Creating Visualizations...")
print("="*50)

# Figure 1: Decision function and distance relationship
plt.figure(figsize=(12, 10))

# Create a grid of points
x1_range = np.linspace(-3, 4, 100)
x2_range = np.linspace(-2, 3, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Compute decision function values for the grid
F_vals = w[0] * X1 + w[1] * X2 + b

# Plot the decision function as a contour plot
plt.subplot(2, 2, 1)
contour = plt.contourf(X1, X2, F_vals, levels=20, cmap='RdYlBu', alpha=0.7)
plt.colorbar(contour, label='$f(\\mathbf{x})$')
plt.contour(X1, X2, F_vals, levels=[0], colors='black', linewidths=2, linestyles='-')
plt.contour(X1, X2, F_vals, levels=[-5, 5], colors='gray', linewidths=1, linestyles='--')

# Plot sample points
for i, point in enumerate(sample_points):
    f_val = decision_function(point, w, b)
    color = 'red' if f_val > 0 else 'blue'
    plt.scatter(point[0], point[1], c=color, s=100, edgecolor='black', zorder=5)
    plt.annotate(f'{chr(65+i)}', (point[0], point[1]), xytext=(5, 5),
                textcoords='offset points', fontsize=12, fontweight='bold')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Function $f(\\mathbf{x}) = \\mathbf{w}^T\\mathbf{x} + b$')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot distance visualization
plt.subplot(2, 2, 2)
distances = np.abs(F_vals) / w_norm
contour_dist = plt.contourf(X1, X2, distances, levels=20, cmap='viridis', alpha=0.7)
plt.colorbar(contour_dist, label='Distance to Hyperplane')
plt.contour(X1, X2, F_vals, levels=[0], colors='white', linewidths=2, linestyles='-')

# Plot sample points with distance annotations
for i, point in enumerate(sample_points):
    dist = distance_to_hyperplane(point, w, b)
    plt.scatter(point[0], point[1], c='white', s=100, edgecolor='black', zorder=5)
    plt.annotate(f'{chr(65+i)}\nd={dist:.2f}', (point[0], point[1]), xytext=(5, 5),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Distance to Hyperplane')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot confidence comparison
plt.subplot(2, 2, 3)
f_range = np.linspace(-3, 3, 100)
confidence = np.abs(f_range)

plt.plot(f_range, confidence, 'b-', linewidth=2, label='$|f(\\mathbf{x})|$')
plt.axhline(y=abs(x1_val), color='red', linestyle='--', alpha=0.7, label=f'$|f(\\mathbf{{x}}_1)| = {abs(x1_val)}$')
plt.axhline(y=abs(x2_val), color='green', linestyle='--', alpha=0.7, label=f'$|f(\\mathbf{{x}}_2)| = {abs(x2_val)}$')
plt.axvline(x=x1_val, color='red', linestyle=':', alpha=0.7)
plt.axvline(x=x2_val, color='green', linestyle=':', alpha=0.7)

plt.scatter([x1_val], [abs(x1_val)], color='red', s=100, zorder=5, label='Point $\\mathbf{{x}}_1$')
plt.scatter([x2_val], [abs(x2_val)], color='green', s=100, zorder=5, label='Point $\\mathbf{{x}}_2$')

plt.xlabel('$f(\\mathbf{{x}})$')
plt.ylabel('Confidence $|f(\\mathbf{{x}})|$')
plt.title('Classification Confidence')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot probability calibration
plt.subplot(2, 2, 4)
f_range_prob = np.linspace(-3, 3, 100)
prob_platt = platt_scaling(f_range_prob)
prob_sigmoid = sigmoid_scaling(f_range_prob)

plt.plot(f_range_prob, prob_platt, 'b-', linewidth=2, label='Platt Scaling')
plt.plot(f_range_prob, prob_sigmoid, 'r-', linewidth=2, label='Sigmoid Scaling')

# Mark our test points
plt.scatter([x1_val], [platt_scaling(np.array([x1_val]))[0]], color='blue', s=100, zorder=5)
plt.scatter([x2_val], [platt_scaling(np.array([x2_val]))[0]], color='blue', s=100, zorder=5)
plt.scatter([x1_val], [sigmoid_scaling(np.array([x1_val]))[0]], color='red', s=100, zorder=5)
plt.scatter([x2_val], [sigmoid_scaling(np.array([x2_val]))[0]], color='red', s=100, zorder=5)

plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('$f(\\mathbf{{x}})$')
plt.ylabel('$P(y=1|\\mathbf{{x}})$')
plt.title('Probability Calibration')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_function_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 3: New informative visualization - Confidence Landscape
plt.figure(figsize=(10, 8))

# Create a finer grid for smooth visualization
x1_fine = np.linspace(-3, 4, 200)
x2_fine = np.linspace(-2, 3, 200)
X1_fine, X2_fine = np.meshgrid(x1_fine, x2_fine)

# Compute decision function and confidence
F_fine = w[0] * X1_fine + w[1] * X2_fine + b
confidence_fine = np.abs(F_fine)

# Create confidence landscape
confidence_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
contour_filled = plt.contourf(X1_fine, X2_fine, confidence_fine,
                             levels=20, cmap='viridis', alpha=0.8)
contour_lines = plt.contour(X1_fine, X2_fine, confidence_fine,
                           levels=confidence_levels, colors='white',
                           linewidths=1.5, alpha=0.7)

# Add decision boundary
plt.contour(X1_fine, X2_fine, F_fine, levels=[0], colors='red', linewidths=3)

# Plot sample points with size proportional to confidence
for i, point in enumerate(sample_points):
    f_val = decision_function(point, w, b)
    confidence = abs(f_val)
    color = 'red' if f_val > 0 else 'blue'
    size = 50 + confidence * 100  # Scale size with confidence
    plt.scatter(point[0], point[1], c=color, s=size,
               edgecolor='white', linewidth=2, zorder=5, alpha=0.9)

plt.colorbar(contour_filled, label='Confidence $|f(\\mathbf{x})|$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM Confidence Landscape')
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(save_dir, 'confidence_landscape.png'), dpi=300, bbox_inches='tight')

# Figure 2: Effect of scaling
plt.figure(figsize=(12, 8))

# Plot effect of scaling on decision values
plt.subplot(2, 3, 1)
for i, scale in enumerate(scale_factors):
    w_scaled = scale * w
    b_scaled = scale * b
    F_scaled = w_scaled[0] * X1 + w_scaled[1] * X2 + b_scaled

    plt.contour(X1, X2, F_scaled, levels=[0], colors=plt.cm.tab10(i),
                linewidths=2, linestyles='-', label=f'Scale {scale}')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Hyperplanes with Different Scaling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot decision values vs scaling
plt.subplot(2, 3, 2)
test_f_vals = []
for scale in scale_factors:
    w_scaled = scale * w
    b_scaled = scale * b
    f_val = decision_function(test_point, w_scaled, b_scaled)
    test_f_vals.append(f_val)

plt.plot(scale_factors, test_f_vals, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Scaling Factor')
plt.ylabel('$f(\\mathbf{{x}})$ for test point')
plt.title('Decision Value vs Scaling')
plt.grid(True, alpha=0.3)

# Plot distances vs scaling (should be constant)
plt.subplot(2, 3, 3)
test_distances = []
for scale in scale_factors:
    w_scaled = scale * w
    b_scaled = scale * b
    dist = distance_to_hyperplane(test_point, w_scaled, b_scaled)
    test_distances.append(dist)

plt.plot(scale_factors, test_distances, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Scaling Factor')
plt.ylabel('Distance to Hyperplane')
plt.title('Distance vs Scaling (Invariant)')
plt.grid(True, alpha=0.3)

# Plot probability calibration for different scales
plt.subplot(2, 3, 4)
f_test_range = np.linspace(-2, 2, 100)
for i, scale in enumerate([0.5, 1.0, 2.0]):
    f_scaled = scale * f_test_range
    prob_scaled = sigmoid_scaling(f_scaled)
    plt.plot(f_test_range, prob_scaled, linewidth=2, label=f'Scale {scale}')

plt.xlabel('Original $f(\\mathbf{{x}})$')
plt.ylabel('$P(y=1|\\mathbf{{x}})$')
plt.title('Probability Calibration vs Scaling')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot confidence ranking preservation
plt.subplot(2, 3, 5)
original_f = np.array([0.1, 0.5, 1.0, 2.5])
for i, scale in enumerate([0.5, 1.0, 2.0]):
    scaled_f = scale * original_f
    plt.plot(range(len(original_f)), np.abs(scaled_f), 'o-',
             linewidth=2, markersize=8, label=f'Scale {scale}')

plt.xlabel('Point Index')
plt.ylabel('$|f(\\mathbf{{x}})|$')
plt.title('Confidence Ranking (Preserved)')
plt.legend()
plt.grid(True, alpha=0.3)

# Summary plot - show scaling relationship graphically
plt.subplot(2, 3, 6)
scale_demo = np.array([0.5, 1.0, 2.0, 5.0])
f_demo = scale_demo * 1.0  # Example f(x) = 1.0
distance_demo = np.ones_like(scale_demo) * 0.2  # Constant distance

plt.plot(scale_demo, f_demo, 'bo-', linewidth=2, markersize=8, label='$f(\\mathbf{x})$')
plt.plot(scale_demo, distance_demo, 'ro-', linewidth=2, markersize=8, label='Distance')

plt.xlabel('Scaling Factor')
plt.ylabel('Value')
plt.title('Scaling Effects')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scaling_effects.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")
print("Files created:")
print("- decision_function_analysis.png")
print("- scaling_effects.png")
print("- confidence_landscape.png")
