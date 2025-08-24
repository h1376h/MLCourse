import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("Question 25: Feature Space Scaling and SVM Margins")
print("=" * 60)

# Generate synthetic 1D data that is separable with a quadratic boundary
np.random.seed(42)
n_samples = 100

# Create data that requires a quadratic decision boundary
x_neg = np.random.uniform(-2, -0.5, n_samples//2)
x_pos = np.random.uniform(0.5, 2, n_samples//2)

# Add some noise to make it more realistic
x_neg += np.random.normal(0, 0.1, n_samples//2)
x_pos += np.random.normal(0, 0.1, n_samples//2)

# Combine data
X_1d = np.concatenate([x_neg, x_pos])
y = np.concatenate([-np.ones(n_samples//2), np.ones(n_samples//2)])

print(f"Generated {len(X_1d)} samples:")
print(f"Negative class: {len(x_neg)} samples, range: [{x_neg.min():.2f}, {x_neg.max():.2f}]")
print(f"Positive class: {len(x_pos)} samples, range: [{x_pos.min():.2f}, {x_pos.max():.2f}]")

# Define the two feature mappings
def phi_1(x):
    """Mapping 1: φ₁(x) = [x, x²]ᵀ"""
    return np.column_stack([x, x**2])

def phi_2(x):
    """Mapping 2: φ₂(x) = [2x, 2x²]ᵀ"""
    return np.column_stack([2*x, 2*x**2])

print("\nFeature Mappings:")
print("$\\phi_1(x) = [x, x^2]^T$")
print("$\\phi_2(x) = [2x, 2x^2]^T = 2\\phi_1(x)$")

# Apply feature mappings
X_phi1 = phi_1(X_1d)
X_phi2 = phi_2(X_1d)

print(f"\nFeature space dimensions:")
print(f"$\\phi_1(x)$: {X_phi1.shape}")
print(f"$\\phi_2(x)$: {X_phi2.shape}")

# Train SVMs with both feature mappings
print("\nTraining SVMs...")

# SVM with φ₁(x)
svm_phi1 = SVC(kernel='linear', C=1000)  # High C for hard margin
svm_phi1.fit(X_phi1, y)

# SVM with φ₂(x)
svm_phi2 = SVC(kernel='linear', C=1000)  # High C for hard margin
svm_phi2.fit(X_phi2, y)

# Get support vectors and weights
support_vectors_phi1 = svm_phi1.support_vectors_
support_vectors_phi2 = svm_phi2.support_vectors_
w_phi1 = svm_phi1.coef_[0]
w_phi2 = svm_phi2.coef_[0]
b_phi1 = svm_phi1.intercept_[0]
b_phi2 = svm_phi2.intercept_[0]

print(f"\nSVM Results:")
print(f"$\\phi_1(x)$ - Number of support vectors: {len(support_vectors_phi1)}")
print(f"$\\phi_2(x)$ - Number of support vectors: {len(support_vectors_phi2)}")

print(f"\nWeight vectors:")
print(f"$\\mathbf{{w}}_1$ ($\\phi_1$): {w_phi1}")
print(f"$\\mathbf{{w}}_2$ ($\\phi_2$): {w_phi2}")

print(f"\nBias terms:")
print(f"$b_1$ ($\\phi_1$): {b_phi1:.6f}")
print(f"$b_2$ ($\\phi_2$): {b_phi2:.6f}")

# Calculate geometric margins
def geometric_margin(w, b, X, y):
    """Calculate geometric margin for given weight vector and data"""
    margins = []
    for i, (x, label) in enumerate(zip(X, y)):
        # Distance from point to decision boundary
        distance = abs(np.dot(w, x) + b) / np.linalg.norm(w)
        margins.append(distance)
    return np.array(margins)

margins_phi1 = geometric_margin(w_phi1, b_phi1, X_phi1, y)
margins_phi2 = geometric_margin(w_phi2, b_phi2, X_phi2, y)

print(f"\nGeometric Margins:")
print(f"$\\phi_1(x)$ - Min margin: {margins_phi1.min():.6f}")
print(f"$\\phi_1(x)$ - Max margin: {margins_phi1.max():.6f}")
print(f"$\\phi_2(x)$ - Min margin: {margins_phi2.min():.6f}")
print(f"$\\phi_2(x)$ - Max margin: {margins_phi2.max():.6f}")

# Theoretical analysis
print(f"\nTheoretical Analysis:")
print(f"$\\phi_2(x) = 2\\phi_1(x)$, so feature vectors are scaled by factor 2")

# Check if w_phi2 ≈ 0.5 * w_phi1 (theoretical expectation)
w_ratio = w_phi2 / w_phi1
print(f"Ratio $\\mathbf{{w}}_2/\\mathbf{{w}}_1$: {w_ratio}")

# Expected margin ratio
expected_margin_ratio = 2.0  # φ₂ scales features by 2, so margin should be 2x larger
actual_margin_ratio = margins_phi2.min() / margins_phi1.min()
print(f"Expected margin ratio ($\\phi_2/\\phi_1$): {expected_margin_ratio}")
print(f"Actual margin ratio ($\\phi_2/\\phi_1$): {actual_margin_ratio:.6f}")

# Visualization 1: Original 1D data
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(x_neg, np.zeros_like(x_neg), c='red', alpha=0.6, label='Class -1')
plt.scatter(x_pos, np.zeros_like(x_pos), c='blue', alpha=0.6, label='Class +1')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Original 1D Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 2: Feature space φ₁(x)
plt.subplot(2, 3, 2)
plt.scatter(X_phi1[y == -1, 0], X_phi1[y == -1, 1], c='red', alpha=0.6, label='Class -1')
plt.scatter(X_phi1[y == 1, 0], X_phi1[y == 1, 1], c='blue', alpha=0.6, label='Class +1')

# Plot decision boundary for φ₁
x1_min, x1_max = X_phi1[:, 0].min() - 0.5, X_phi1[:, 0].max() + 0.5
x1_line = np.linspace(x1_min, x1_max, 100)
x2_line = (-w_phi1[0] * x1_line - b_phi1) / w_phi1[1]
plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary')

# Plot margin lines for φ₁
margin_distance = 1 / np.linalg.norm(w_phi1)
x2_margin_plus = x2_line + margin_distance * w_phi1[1] / np.linalg.norm(w_phi1)
x2_margin_minus = x2_line - margin_distance * w_phi1[1] / np.linalg.norm(w_phi1)
plt.plot(x1_line, x2_margin_plus, 'g--', alpha=0.7, label='Margin')
plt.plot(x1_line, x2_margin_minus, 'g--', alpha=0.7)

plt.xlabel('$x$')
plt.ylabel('$x^2$')
plt.title('Feature Space $\\phi_1(x) = [x, x^2]^T$')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 3: Feature space φ₂(x)
plt.subplot(2, 3, 3)
plt.scatter(X_phi2[y == -1, 0], X_phi2[y == -1, 1], c='red', alpha=0.6, label='Class -1')
plt.scatter(X_phi2[y == 1, 0], X_phi2[y == 1, 1], c='blue', alpha=0.6, label='Class +1')

# Plot decision boundary for φ₂
x1_min, x1_max = X_phi2[:, 0].min() - 1, X_phi2[:, 0].max() + 1
x1_line = np.linspace(x1_min, x1_max, 100)
x2_line = (-w_phi2[0] * x1_line - b_phi2) / w_phi2[1]
plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary')

# Plot margin lines for φ₂
margin_distance = 1 / np.linalg.norm(w_phi2)
x2_margin_plus = x2_line + margin_distance * w_phi2[1] / np.linalg.norm(w_phi2)
x2_margin_minus = x2_line - margin_distance * w_phi2[1] / np.linalg.norm(w_phi2)
plt.plot(x1_line, x2_margin_plus, 'g--', alpha=0.7, label='Margin')
plt.plot(x1_line, x2_margin_minus, 'g--', alpha=0.7)

plt.xlabel('$2x$')
plt.ylabel('$2x^2$')
plt.title('Feature Space $\\phi_2(x) = [2x, 2x^2]^T$')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 4: Comparison of margins
plt.subplot(2, 3, 4)
plt.hist(margins_phi1, bins=20, alpha=0.7, label='$\\phi_1(x)$', color='blue')
plt.hist(margins_phi2, bins=20, alpha=0.7, label='$\\phi_2(x)$', color='red')
plt.xlabel('Geometric Margin')
plt.ylabel('Frequency')
plt.title('Distribution of Geometric Margins')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 5: Weight vector comparison
plt.subplot(2, 3, 5)
components = ['w₁', 'w₂']
plt.bar([1, 2], [np.linalg.norm(w_phi1), np.linalg.norm(w_phi2)], 
        color=['blue', 'red'], alpha=0.7)
plt.xticks([1, 2], ['$\\phi_1(x)$', '$\\phi_2(x)$'])
plt.ylabel('$\\|\\mathbf{w}\\|$')
plt.title('Weight Vector Magnitudes')
plt.grid(True, alpha=0.3)

# Add text annotations
plt.text(1, np.linalg.norm(w_phi1) + 0.01, f'{np.linalg.norm(w_phi1):.3f}', 
         ha='center', va='bottom')
plt.text(2, np.linalg.norm(w_phi2) + 0.01, f'{np.linalg.norm(w_phi2):.3f}', 
         ha='center', va='bottom')

# Visualization 6: Margin comparison
plt.subplot(2, 3, 6)
plt.bar([1, 2], [margins_phi1.min(), margins_phi2.min()], 
        color=['blue', 'red'], alpha=0.7)
plt.xticks([1, 2], ['$\\phi_1(x)$', '$\\phi_2(x)$'])
plt.ylabel('Minimum Geometric Margin')
plt.title('Minimum Geometric Margins')
plt.grid(True, alpha=0.3)

# Add text annotations
plt.text(1, margins_phi1.min() + 0.001, f'{margins_phi1.min():.4f}', 
         ha='center', va='bottom')
plt.text(2, margins_phi2.min() + 0.001, f'{margins_phi2.min():.4f}', 
         ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_scaling_comparison.png'), dpi=300, bbox_inches='tight')

# Detailed analysis visualization
plt.figure(figsize=(12, 8))

# Plot decision boundaries in original space
x_plot = np.linspace(-3, 3, 1000)

# For φ₁(x): decision boundary is w₁₁*x + w₁₂*x² + b₁ = 0
# Solving for x²: x² = (-w₁₁*x - b₁)/w₁₂
x2_phi1 = (-w_phi1[0] * x_plot - b_phi1) / w_phi1[1]

# For φ₂(x): decision boundary is w₂₁*2x + w₂₂*2x² + b₂ = 0
# Solving for x²: x² = (-w₂₁*2x - b₂)/(2*w₂₂)
x2_phi2 = (-w_phi2[0] * 2 * x_plot - b_phi2) / (2 * w_phi2[1])

plt.subplot(2, 2, 1)
plt.scatter(x_neg, np.zeros_like(x_neg), c='red', alpha=0.6, label='Class -1')
plt.scatter(x_pos, np.zeros_like(x_pos), c='blue', alpha=0.6, label='Class +1')

# Plot decision boundaries in original space
valid_mask1 = x2_phi1 >= 0
valid_mask2 = x2_phi2 >= 0

plt.plot(x_plot[valid_mask1], x2_phi1[valid_mask1], 'g-', linewidth=2, label='$\\phi_1(x)$ boundary')
plt.plot(x_plot[valid_mask2], x2_phi2[valid_mask2], 'r-', linewidth=2, label='$\\phi_2(x)$ boundary')

plt.xlabel('$x$')
plt.ylabel('$x^2$')
plt.title('Decision Boundaries in Original Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot margin comparison
plt.subplot(2, 2, 2)
plt.plot(range(len(margins_phi1)), margins_phi1, 'b-', alpha=0.7, label='$\\phi_1(x)$ margins')
plt.plot(range(len(margins_phi2)), margins_phi2, 'r-', alpha=0.7, label='$\\phi_2(x)$ margins')
plt.xlabel('Sample Index')
plt.ylabel('Geometric Margin')
plt.title('Geometric Margins Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot weight vector components
plt.subplot(2, 2, 3)
x_pos = np.arange(2)
width = 0.35

plt.bar(x_pos - width/2, [w_phi1[0], w_phi1[1]], width, label='$\\phi_1(x)$', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, [w_phi2[0], w_phi2[1]], width, label='$\\phi_2(x)$', color='red', alpha=0.7)

plt.xlabel('Weight Component')
plt.ylabel('Weight Value')
plt.title('Weight Vector Components')
plt.xticks(x_pos, ['$w_1$', '$w_2$'])
plt.legend()
plt.grid(True, alpha=0.3)

# Plot theoretical vs actual ratios
plt.subplot(2, 2, 4)
ratios = ['Weight Magnitude', 'Min Margin', 'Max Margin']
theoretical = [0.5, 2.0, 2.0]  # Expected ratios
actual = [np.linalg.norm(w_phi2)/np.linalg.norm(w_phi1), 
          margins_phi2.min()/margins_phi1.min(),
          margins_phi2.max()/margins_phi1.max()]

x_pos = np.arange(len(ratios))
width = 0.35

plt.bar(x_pos - width/2, theoretical, width, label='Theoretical', color='green', alpha=0.7)
plt.bar(x_pos + width/2, actual, width, label='Actual', color='orange', alpha=0.7)

plt.xlabel('Ratio Type')
plt.ylabel('Ratio Value')
plt.title('Theoretical vs Actual Ratios')
plt.xticks(x_pos, ratios)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')

# Mathematical derivation visualization
plt.figure(figsize=(10, 6))

# Create a simple illustration of the scaling effect
x = np.linspace(-2, 2, 100)
y1 = x**2  # φ₁(x) = [x, x²]
y2 = 2*x**2  # φ₂(x) = [2x, 2x²]

plt.subplot(1, 2, 1)
plt.plot(x, y1, 'b-', linewidth=2, label='$\\phi_1(x) = [x, x^2]$')
plt.plot(x, y2, 'r-', linewidth=2, label='$\\phi_2(x) = [2x, 2x^2]$')
plt.xlabel('$x$')
plt.ylabel('$x^2$')
plt.title('Feature Mapping Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Illustrate the scaling effect on a unit circle
theta = np.linspace(0, 2*np.pi, 100)
r = 1
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)

x_scaled = 2 * x_circle
y_scaled = 2 * y_circle

plt.subplot(1, 2, 2)
plt.plot(x_circle, y_circle, 'b-', linewidth=2, label='Original ($\\phi_1$)')
plt.plot(x_scaled, y_scaled, 'r-', linewidth=2, label='Scaled ($\\phi_2$)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Scaling Effect on Unit Circle')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mathematical_derivation.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")

# Summary of findings
print(f"\n" + "="*60)
print("SUMMARY OF FINDINGS")
print("="*60)

print(f"1. Feature Scaling Effect:")
print(f"   - $\\phi_2(x) = 2\\phi_1(x)$ scales all features by factor 2")
print(f"   - This scaling affects the optimal weight vector and margin")

print(f"\n2. Weight Vector Analysis:")
print(f"   - ||w₁|| = {np.linalg.norm(w_phi1):.6f}")
print(f"   - ||w₂|| = {np.linalg.norm(w_phi2):.6f}")
print(f"   - Ratio ||w₂||/||w₁|| = {np.linalg.norm(w_phi2)/np.linalg.norm(w_phi1):.6f}")
print(f"   - Expected ratio: 0.5 (w₂ should be approximately 0.5*w₁)")

print(f"\n3. Geometric Margin Analysis:")
print(f"   - $\\phi_1(x)$ minimum margin: {margins_phi1.min():.6f}")
print(f"   - $\\phi_2(x)$ minimum margin: {margins_phi2.min():.6f}")
print(f"   - Ratio $\\phi_2/\\phi_1$: {margins_phi2.min()/margins_phi1.min():.6f}")
print(f"   - Expected ratio: 2.0 ($\\phi_2$ should have 2x larger margin)")

print(f"\n4. Answer to Question 25:")
print(f"   The geometric margin using $\\phi_2(x)$ is GREATER THAN the margin from $\\phi_1(x)$")
print(f"   This is because scaling features by factor 2 increases the margin by factor 2")
print(f"   while the weight vector magnitude decreases by factor 0.5")
print(f"   Result: margin = $1/\\|\\mathbf{{w}}\\|$ increases by factor 2")
