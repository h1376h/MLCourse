import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_36")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX with proper configuration for subscripts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Configure LaTeX to handle subscripts properly
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 36: KERNEL TRICK CALCULATIONS AND SVM PREDICTION")
print("=" * 80)

# Define the phi mapping function
def phi_mapping(d):
    """
    Apply the phi mapping to a 2D point d = (d1, d2)
    Returns a 6-dimensional vector
    """
    d1, d2 = d
    return np.array([
        d1**2,                    # phi_0
        d2**2,                    # phi_1
        np.sqrt(2) * d1 * d2,     # phi_2
        np.sqrt(2) * d1,          # phi_3
        np.sqrt(2) * d2,          # phi_4
        1                         # phi_5
    ])

# Define the kernel function
def kernel_function(d, q):
    """
    Polynomial kernel K(d,q) = (d·q + 1)^2
    """
    dot_product = np.dot(d, q)
    return (dot_product + 1)**2

# Given vectors
v1 = np.array([0.9, 1.0])
v2 = np.array([1.0, 0.9])

print(f"Given vectors:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print()

# Task 1: Calculate phi(v1) and phi(v2)
print("TASK 1: Calculate 6-dimensional vectors φ(v1) and φ(v2)")
print("-" * 60)

phi_v1 = phi_mapping(v1)
phi_v2 = phi_mapping(v2)

print(f"φ(v1) = φ({v1}) = {phi_v1}")
print(f"φ(v2) = φ({v2}) = {phi_v2}")
print()

# Task 2: Calculate dot product φ(v1) · φ(v2)
print("TASK 2: Calculate φ(v1) · φ(v2)")
print("-" * 60)

dot_product_phi = np.dot(phi_v1, phi_v2)
print(f"φ(v1) · φ(v2) = {dot_product_phi}")
print()

# Task 3: Calculate K(v1, v2) using kernel function
print("TASK 3: Calculate K(v1, v2) using kernel function")
print("-" * 60)

kernel_value = kernel_function(v1, v2)
print(f"K(v1, v2) = (v1 · v2 + 1)²")
print(f"v1 · v2 = {np.dot(v1, v2)}")
print(f"K(v1, v2) = ({np.dot(v1, v2)} + 1)² = {kernel_value}")
print()

# Verify that they match
print("VERIFICATION:")
print(f"φ(v1) · φ(v2) = {dot_product_phi}")
print(f"K(v1, v2) = {kernel_value}")
print(f"Match: {np.isclose(dot_product_phi, kernel_value)}")
print()

# Task 4: SVM prediction
print("TASK 4: SVM Prediction")
print("-" * 60)

# Support vectors and their classes
v0 = np.array([0, 1])  # class = -1
v1_sv = np.array([0.9, 1])  # class = +1

# Trained parameters
w0 = 0.11  # bias term
alpha0 = 0.83  # dual weight for v0
alpha1 = 0.99  # dual weight for v1

print(f"Support vectors:")
print(f"v0 = {v0} (class = -1)")
print(f"v1 = {v1_sv} (class = +1)")
print(f"Trained parameters:")
print(f"  w0 (bias) = {w0}")
print(f"  α0 = {alpha0}")
print(f"  α1 = {alpha1}")
print()

# SVM decision function: f(x) = Σ(αi * yi * K(xi, x)) + b
def svm_decision_function(x, support_vectors, classes, alphas, bias):
    """
    Calculate SVM decision function value
    f(x) = Σ(αi * yi * K(xi, x)) + b
    """
    result = bias
    for i, (sv, y, alpha) in enumerate(zip(support_vectors, classes, alphas)):
        kernel_val = kernel_function(sv, x)
        result += alpha * y * kernel_val
        print(f"  α{i} * y{i} * K(v{i}, v2) = {alpha} * {y} * {kernel_val} = {alpha * y * kernel_val}")
    return result

print("SVM Decision Function Calculation:")
print(f"f(v2) = Σ(αi * yi * K(vi, v2)) + w0")
print(f"f(v2) = α0 * y0 * K(v0, v2) + α1 * y1 * K(v1, v2) + w0")

support_vectors = [v0, v1_sv]
classes = [-1, 1]
alphas = [alpha0, alpha1]

decision_value = svm_decision_function(v2, support_vectors, classes, alphas, w0)

print(f"f(v2) = {decision_value}")
print()

# Prediction
predicted_class = np.sign(decision_value)
print(f"Prediction:")
print(f"Decision value = {decision_value}")
print(f"Predicted class = sign({decision_value}) = {predicted_class}")
print()

# Create visualizations
print("Creating visualizations...")

# Visualization 1: Original 2D space with vectors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original 2D space
ax1.scatter([v1[0], v2[0], v0[0], v1_sv[0]], [v1[1], v2[1], v0[1], v1_sv[1]], 
           c=['blue', 'red', 'green', 'orange'], s=100, alpha=0.7)
ax1.annotate(r'$v_1$ (0.9, 1)', (v1[0], v1[1]), xytext=(5, 5), textcoords='offset points')
ax1.annotate(r'$v_2$ (1, 0.9)', (v2[0], v2[1]), xytext=(5, 5), textcoords='offset points')
ax1.annotate(r'$v_0$ (0, 1)', (v0[0], v0[1]), xytext=(5, 5), textcoords='offset points')
ax1.annotate(r'$v_1^{sv}$ (0.9, 1)', (v1_sv[0], v1_sv[1]), xytext=(5, 5), textcoords='offset points')
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_title('Original 2D Feature Space')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.2, 1.2)

# Plot 2: 6D feature space projection (first 3 dimensions)
ax2.scatter([phi_v1[0], phi_v2[0], phi_mapping(v0)[0], phi_mapping(v1_sv)[0]], 
           [phi_v1[1], phi_v2[1], phi_mapping(v0)[1], phi_mapping(v1_sv)[1]], 
           c=['blue', 'red', 'green', 'orange'], s=100, alpha=0.7)
ax2.annotate(r'$\phi(v_1)$', (phi_v1[0], phi_v1[1]), xytext=(5, 5), textcoords='offset points')
ax2.annotate(r'$\phi(v_2)$', (phi_v2[0], phi_v2[1]), xytext=(5, 5), textcoords='offset points')
ax2.annotate(r'$\phi(v_0)$', (phi_mapping(v0)[0], phi_mapping(v0)[1]), xytext=(5, 5), textcoords='offset points')
ax2.annotate(r'$\phi(v_1^{sv})$', (phi_mapping(v1_sv)[0], phi_mapping(v1_sv)[1]), xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel(r'$\phi_0$ ($d_1^2$)')
ax2.set_ylabel(r'$\phi_1$ ($d_2^2$)')
ax2.set_title('6D Feature Space Projection (First 2 Dimensions)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Kernel values heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate kernel values for all pairs
vectors = [v0, v1_sv, v1, v2]
vector_names = [r'$v_0$', r'$v_1^{sv}$', r'$v_1$', r'$v_2$']
n = len(vectors)

kernel_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        kernel_matrix[i, j] = kernel_function(vectors[i], vectors[j])

# Create heatmap
im = ax.imshow(kernel_matrix, cmap='viridis', aspect='auto')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(vector_names)
ax.set_yticklabels(vector_names)
ax.set_xlabel(r'$v_i$')
ax.set_ylabel(r'$v_j$')
ax.set_title(r'Kernel Matrix $K(v_i, v_j)$')

# Add text annotations
for i in range(n):
    for j in range(n):
        text = ax.text(j, i, f'{kernel_matrix[i, j]:.3f}',
                      ha="center", va="center", color="white", fontweight='bold')

plt.colorbar(im, ax=ax, label='Kernel Value')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix_heatmap.png'), dpi=300, bbox_inches='tight')

# Visualization 3: 3D feature space visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract first 3 dimensions of phi mapping
phi_v1_3d = phi_v1[:3]
phi_v2_3d = phi_v2[:3]
phi_v0_3d = phi_mapping(v0)[:3]
phi_v1_sv_3d = phi_mapping(v1_sv)[:3]

ax.scatter([phi_v1_3d[0]], [phi_v1_3d[1]], [phi_v1_3d[2]], 
          c='blue', s=100, alpha=0.7, label=r'$\phi(v_1)$')
ax.scatter([phi_v2_3d[0]], [phi_v2_3d[1]], [phi_v2_3d[2]], 
          c='red', s=100, alpha=0.7, label=r'$\phi(v_2)$')
ax.scatter([phi_v0_3d[0]], [phi_v0_3d[1]], [phi_v0_3d[2]], 
          c='green', s=100, alpha=0.7, label=r'$\phi(v_0)$')
ax.scatter([phi_v1_sv_3d[0]], [phi_v1_sv_3d[1]], [phi_v1_sv_3d[2]], 
          c='orange', s=100, alpha=0.7, label=r'$\phi(v_1^{sv})$')

ax.set_xlabel(r'$\phi_0$ ($d_1^2$)')
ax.set_ylabel(r'$\phi_1$ ($d_2^2$)')
ax.set_zlabel(r'$\phi_2$ ($\sqrt{2} d_1 d_2$)')
ax.set_title('3D Feature Space Visualization')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3d_feature_space.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Decision boundary visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Create a grid of points
x1_range = np.linspace(-0.5, 1.5, 100)
x2_range = np.linspace(-0.5, 1.5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate decision values for each point
decision_values = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        point = np.array([X1[i, j], X2[i, j]])
        decision_values[i, j] = svm_decision_function(point, support_vectors, classes, alphas, w0)

# Plot decision boundary
contour = ax.contour(X1, X2, decision_values, levels=[0], colors='black', linewidths=2)
ax.contourf(X1, X2, decision_values, levels=[-100, 0, 100], colors=['lightblue', 'lightpink'], alpha=0.3)

# Plot support vectors and test point
ax.scatter([v0[0], v1_sv[0], v2[0]], [v0[1], v1_sv[1], v2[1]], 
          c=['green', 'orange', 'red'], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.annotate(r'$v_0$ (class=-1)', (v0[0], v0[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate(r'$v_1^{sv}$ (class=+1)', (v1_sv[0], v1_sv[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate(r'$v_2$ (predicted=' + str(predicted_class) + ')', (v2[0], v2[1]), xytext=(5, 5), textcoords='offset points')

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('SVM Decision Boundary with Kernel')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_decision_boundary.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Detailed calculation breakdown
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create a text box with detailed calculations
calculation_text = f"""
DETAILED CALCULATIONS FOR QUESTION 36

TASK 1: $\\phi(v_1)$ and $\\phi(v_2)$ Calculation
$\\phi(v_1)$ = $\\phi({v1})$ = {phi_v1}
$\\phi(v_2)$ = $\\phi({v2})$ = {phi_v2}

TASK 2: Dot Product
$\\phi(v_1) \\cdot \\phi(v_2)$ = {dot_product_phi}

TASK 3: Kernel Function
$K(v_1, v_2)$ = $(v_1 \\cdot v_2 + 1)^2$ = $({np.dot(v1, v2)} + 1)^2$ = {kernel_value}

VERIFICATION: {dot_product_phi} = {kernel_value} $\\checkmark$

TASK 4: SVM Prediction
Support Vectors:
  $v_0$ = {v0} (class = -1)
  $v_1$ = {v1_sv} (class = +1)

Parameters:
  $w_0$ = {w0}
  $\\alpha_0$ = {alpha0}
  $\\alpha_1$ = {alpha1}

Decision Function:
$f(v_2)$ = $\\alpha_0 \\times y_0 \\times K(v_0, v_2)$ + $\\alpha_1 \\times y_1 \\times K(v_1, v_2)$ + $w_0$
$f(v_2)$ = {alpha0} × (-1) × {kernel_function(v0, v2)} + {alpha1} × 1 × {kernel_function(v1_sv, v2)} + {w0}
$f(v_2)$ = {decision_value}

Prediction: sign({decision_value}) = {predicted_class}
"""

ax.text(0.05, 0.95, calculation_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_calculations.png'), dpi=300, bbox_inches='tight')

print(f"All visualizations saved to: {save_dir}")
print("\n" + "=" * 80)
print("SOLUTION SUMMARY")
print("=" * 80)
print(f"1. $\\phi(v_1)$ = {phi_v1}")
print(f"   $\\phi(v_2)$ = {phi_v2}")
print(f"2. $\\phi(v_1) \\cdot \\phi(v_2)$ = {dot_product_phi}")
print(f"3. $K(v_1, v_2)$ = {kernel_value}")
print(f"   Verification: {np.isclose(dot_product_phi, kernel_value)}")
print(f"4. SVM prediction for $v_2$: {predicted_class} (decision value: {decision_value})")
print("=" * 80)
