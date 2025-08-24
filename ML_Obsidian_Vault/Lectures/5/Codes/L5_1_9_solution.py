import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 9: KKT Conditions Analysis for SVM")
print("=" * 80)

# Task 1: Write out all KKT conditions
print("\n1. KKT CONDITIONS FOR SVM OPTIMALITY")
print("-" * 50)

print("Given Lagrangian:")
print("L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w^T xᵢ + b) - 1]")
print("\nThe KKT conditions for optimality are:")
print("\n1. STATIONARITY CONDITIONS:")
print("   ∇_w L = 0  =>  w - Σᵢ αᵢ yᵢ xᵢ = 0")
print("   ∂L/∂b = 0  =>  Σᵢ αᵢ yᵢ = 0")
print("\n2. PRIMAL FEASIBILITY:")
print("   yᵢ(w^T xᵢ + b) ≥ 1  for all i")
print("\n3. DUAL FEASIBILITY:")
print("   αᵢ ≥ 0  for all i")
print("\n4. COMPLEMENTARY SLACKNESS:")
print("   αᵢ[yᵢ(w^T xᵢ + b) - 1] = 0  for all i")

# Visualization of KKT conditions
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Stationarity condition visualization
ax1.text(0.5, 0.8, 'STATIONARITY CONDITIONS', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax1.transAxes)

stationarity_text = (
    r'$\nabla_{\mathbf{w}} L = \mathbf{0}$' + '\n\n' +
    r'$\mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0}$' + '\n\n' +
    r'$\Rightarrow \mathbf{w}^* = \sum_{i=1}^n \alpha_i^* y_i \mathbf{x}_i$' + '\n\n' +
    r'$\frac{\partial L}{\partial b} = 0$' + '\n\n' +
    r'$\sum_{i=1}^n \alpha_i y_i = 0$'
)

ax1.text(0.5, 0.4, stationarity_text, ha='center', va='center', 
         fontsize=12, transform=ax1.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Subplot 2: Primal feasibility
ax2.text(0.5, 0.8, 'PRIMAL FEASIBILITY', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax2.transAxes)

primal_text = (
    r'$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$' + '\n\n' +
    'for all training points $i = 1, 2, \ldots, n$' + '\n\n' +
    'This ensures all points are correctly' + '\n' +
    'classified with sufficient margin'
)

ax2.text(0.5, 0.4, primal_text, ha='center', va='center', 
         fontsize=12, transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Subplot 3: Dual feasibility
ax3.text(0.5, 0.8, 'DUAL FEASIBILITY', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax3.transAxes)

dual_text = (
    r'$\alpha_i \geq 0$' + '\n\n' +
    'for all Lagrange multipliers' + '\n' +
    '$i = 1, 2, \ldots, n$' + '\n\n' +
    'Non-negativity constraint on' + '\n' +
    'dual variables'
)

ax3.text(0.5, 0.4, dual_text, ha='center', va='center', 
         fontsize=12, transform=ax3.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Subplot 4: Complementary slackness
ax4.text(0.5, 0.8, 'COMPLEMENTARY SLACKNESS', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax4.transAxes)

comp_slack_text = (
    r'$\alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$' + '\n\n' +
    'For each training point:' + '\n' +
    r'Either $\alpha_i = 0$ OR' + '\n' +
    r'$y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$' + '\n\n' +
    'Key to identifying support vectors'
)

ax4.text(0.5, 0.4, comp_slack_text, ha='center', va='center', 
         fontsize=12, transform=ax4.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.suptitle('KKT Conditions for SVM Optimization', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kkt_conditions_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 2: Show that w* = Σ αᵢ* yᵢ xᵢ
print("\n2. DERIVATION OF OPTIMAL WEIGHT VECTOR")
print("-" * 50)

print("From the stationarity condition ∇_w L = 0:")
print("∇_w L = ∇_w [(1/2)||w||² - Σᵢ αᵢ[yᵢ(w^T xᵢ + b) - 1]]")
print("      = w - Σᵢ αᵢ yᵢ xᵢ = 0")
print("\nTherefore: w* = Σᵢ αᵢ* yᵢ xᵢ")
print("\nThis shows that the optimal weight vector is a linear combination")
print("of the training points, weighted by αᵢ* yᵢ.")

# Visualization of weight vector decomposition
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a simple 2D example
np.random.seed(42)
# Support vectors
sv_pos = np.array([[1, 0.5], [0.8, 0.8]])
sv_neg = np.array([[-1, -0.5], [-0.8, -0.8]])
# Non-support vectors
nsv_pos = np.array([[2, 1], [1.5, 1.2]])
nsv_neg = np.array([[-2, -1], [-1.5, -1.2]])

# Lagrange multipliers (only non-zero for support vectors)
alpha_sv = [0.5, 0.3, 0.4, 0.4]  # For support vectors
alpha_nsv = [0, 0, 0, 0]  # For non-support vectors

# Plot all points
ax.scatter(sv_pos[:, 0], sv_pos[:, 1], c='red', s=150, marker='o', 
           edgecolor='black', linewidth=3, label='Support Vectors (+)')
ax.scatter(sv_neg[:, 0], sv_neg[:, 1], c='blue', s=150, marker='s', 
           edgecolor='black', linewidth=3, label='Support Vectors (-)')
ax.scatter(nsv_pos[:, 0], nsv_pos[:, 1], c='red', s=100, marker='o', 
           alpha=0.5, label='Non-Support Vectors (+)')
ax.scatter(nsv_neg[:, 0], nsv_neg[:, 1], c='blue', s=100, marker='s', 
           alpha=0.5, label='Non-Support Vectors (-)')

# Calculate and plot weight vector components
w_components = []
labels = []
colors = ['red', 'red', 'blue', 'blue']
all_sv = np.vstack([sv_pos, sv_neg])
y_values = [1, 1, -1, -1]

origin = np.array([0, 0])
for i, (sv, alpha, y, color) in enumerate(zip(all_sv, alpha_sv, y_values, colors)):
    component = alpha * y * sv
    w_components.append(component)
    
    # Draw component vector
    ax.arrow(origin[0], origin[1], component[0], component[1], 
             head_width=0.1, head_length=0.1, fc=color, ec=color, 
             alpha=0.7, linewidth=2)
    
    # Label component
    mid_point = origin + 0.5 * component
    ax.annotate(f'$\\alpha_{{{i+1}}} y_{{{i+1}}} \\mathbf{{x}}_{{{i+1}}}$', 
                xy=mid_point, xytext=(mid_point[0]+0.2, mid_point[1]+0.2),
                fontsize=10, color=color, fontweight='bold')

# Calculate and draw final weight vector
w_final = sum(w_components)
ax.arrow(origin[0], origin[1], w_final[0], w_final[1], 
         head_width=0.15, head_length=0.15, fc='purple', ec='purple', 
         linewidth=4, alpha=0.9)

ax.annotate(f'$\\mathbf{{w}}^* = \\sum_i \\alpha_i^* y_i \\mathbf{{x}}_i$', 
            xy=w_final, xytext=(w_final[0]+0.3, w_final[1]+0.3),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2),
            fontsize=14, color='purple', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.8))

# Draw decision boundary (perpendicular to w)
if w_final[1] != 0:
    x_range = np.linspace(-3, 3, 100)
    # Decision boundary: w^T x = 0 (simplified for visualization)
    y_boundary = -(w_final[0] / w_final[1]) * x_range
    ax.plot(x_range, y_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Weight Vector as Linear Combination of Support Vectors')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_vector_decomposition.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 3 & 4: Support vector identification through complementary slackness
print("\n3. & 4. SUPPORT VECTOR IDENTIFICATION")
print("-" * 50)

print("From complementary slackness: αᵢ[yᵢ(w^T xᵢ + b) - 1] = 0")
print("\nThis means for each training point, either:")
print("1. αᵢ = 0  (non-support vector)")
print("2. yᵢ(w^T xᵢ + b) - 1 = 0  (support vector)")
print("\nFor SUPPORT VECTORS:")
print("- αᵢ* > 0 (active constraint)")
print("- yᵢ(w^T xᵢ + b) = 1 (point lies on margin boundary)")
print("\nFor NON-SUPPORT VECTORS:")
print("- αᵢ* = 0 (inactive constraint)")
print("- yᵢ(w^T xᵢ + b) > 1 (point lies beyond margin boundary)")

# Visualization of support vector identification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Geometric interpretation
x1_range = np.linspace(-3, 3, 100)
x2_boundary = -x1_range  # Decision boundary: x1 + x2 = 0
x2_pos_margin = -x1_range + np.sqrt(2)  # Positive margin
x2_neg_margin = -x1_range - np.sqrt(2)  # Negative margin

ax1.plot(x1_range, x2_boundary, 'k-', linewidth=3, label='Decision Boundary')
ax1.plot(x1_range, x2_pos_margin, 'r--', linewidth=2, label='Positive Margin')
ax1.plot(x1_range, x2_neg_margin, 'b--', linewidth=2, label='Negative Margin')

# Support vectors (on margin)
sv_pos_demo = np.array([[1, -1+np.sqrt(2)]])
sv_neg_demo = np.array([[-1, 1-np.sqrt(2)]])
ax1.scatter(sv_pos_demo[:, 0], sv_pos_demo[:, 1], c='red', s=200, marker='o', 
           edgecolor='black', linewidth=4, label='Support Vectors', zorder=5)
ax1.scatter(sv_neg_demo[:, 0], sv_neg_demo[:, 1], c='blue', s=200, marker='s', 
           edgecolor='black', linewidth=4, zorder=5)

# Non-support vectors (beyond margin)
nsv_pos_demo = np.array([[2, 0], [1.5, 0.5]])
nsv_neg_demo = np.array([[-2, 0], [-1.5, -0.5]])
ax1.scatter(nsv_pos_demo[:, 0], nsv_pos_demo[:, 1], c='red', s=100, marker='o', 
           alpha=0.6, label='Non-Support Vectors')
ax1.scatter(nsv_neg_demo[:, 0], nsv_neg_demo[:, 1], c='blue', s=100, marker='s', 
           alpha=0.6)

# Add annotations
ax1.annotate('$\\alpha_i > 0$\n$y_i(\\mathbf{w}^T\\mathbf{x}_i + b) = 1$', 
             xy=sv_pos_demo[0], xytext=(sv_pos_demo[0,0]+0.5, sv_pos_demo[0,1]+0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.8))

ax1.annotate('$\\alpha_i = 0$\n$y_i(\\mathbf{w}^T\\mathbf{x}_i + b) > 1$', 
             xy=nsv_pos_demo[0], xytext=(nsv_pos_demo[0,0]+0.3, nsv_pos_demo[0,1]+0.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=1),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Support Vector Identification')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)

# Right plot: Alpha values and constraints
points = ['SV1', 'SV2', 'NSV1', 'NSV2', 'NSV3', 'NSV4']
alpha_values = [0.5, 0.3, 0, 0, 0, 0]
constraint_values = [1.0, 1.0, 1.8, 2.1, 1.5, 1.9]  # y_i(w^T x_i + b)

colors = ['red' if alpha > 0 else 'gray' for alpha in alpha_values]
bars = ax2.bar(points, alpha_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add constraint value annotations
for i, (bar, constraint) in enumerate(zip(bars, constraint_values)):
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'$y_i(\\mathbf{{w}}^T\\mathbf{{x}}_i + b) = {constraint}$',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8))
    else:
        ax2.annotate(f'$y_i(\\mathbf{{w}}^T\\mathbf{{x}}_i + b) = {constraint}$',
                     xy=(bar.get_x() + bar.get_width()/2, 0.05),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))

ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylabel('$\\alpha_i$ values')
ax2.set_title('Lagrange Multipliers and Constraint Values')
ax2.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Support Vectors ($\\alpha_i > 0$)'),
                   Patch(facecolor='gray', alpha=0.7, label='Non-Support Vectors ($\\alpha_i = 0$)')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_identification.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 5: Derive condition for computing bias term b*
print("\n5. COMPUTING THE BIAS TERM b*")
print("-" * 50)

print("For any support vector xₛ (where αₛ > 0):")
print("From complementary slackness: yₛ(w^T xₛ + b) = 1")
print("\nSolving for b:")
print("b* = yₛ - w^T xₛ")
print("   = yₛ - (Σᵢ αᵢ* yᵢ xᵢ)^T xₛ")
print("   = yₛ - Σᵢ αᵢ* yᵢ (xᵢ^T xₛ)")
print("\nFor numerical stability, average over all support vectors:")
print("b* = (1/|S|) Σₛ∈S [yₛ - Σᵢ αᵢ* yᵢ (xᵢ^T xₛ)]")
print("where S is the set of support vector indices.")

# Visualization of bias computation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Geometric interpretation of bias
x1_range = np.linspace(-2, 2, 100)
# Example: w = [1, 1], different bias values
w_example = np.array([1, 1])
bias_values = [-0.5, 0, 0.5]
colors_bias = ['blue', 'black', 'red']
labels_bias = ['$b = -0.5$', '$b = 0$', '$b = 0.5$']

for bias, color, label in zip(bias_values, colors_bias, labels_bias):
    x2_line = (-w_example[0] * x1_range - bias) / w_example[1]
    ax1.plot(x1_range, x2_line, color=color, linewidth=2, label=label)

# Show how bias shifts the hyperplane
ax1.arrow(0, 0, 0, 0.5, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
ax1.annotate('Increasing $b$\nshifts hyperplane', xy=(0, 0.25), xytext=(0.5, 0.8),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Effect of Bias Term on Hyperplane Position')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)

# Right plot: Bias computation from support vectors
# Example support vectors and their contributions to bias
sv_labels = ['SV1', 'SV2', 'SV3']
y_sv = [1, 1, -1]  # Labels of support vectors
w_dot_x_sv = [0.8, 1.2, -0.9]  # w^T x_s for each support vector
b_contributions = [y - w_dot for y, w_dot in zip(y_sv, w_dot_x_sv)]

ax2.bar(sv_labels, b_contributions, color=['red', 'red', 'blue'], alpha=0.7,
        edgecolor='black', linewidth=2)

# Add value annotations
for i, (label, contrib, y_val, w_dot) in enumerate(zip(sv_labels, b_contributions, y_sv, w_dot_x_sv)):
    ax2.annotate(f'$y_s = {y_val}$\n$\\mathbf{{w}}^T\\mathbf{{x}}_s = {w_dot}$\n$b = {contrib:.1f}$',
                 xy=(i, contrib), xytext=(0, 10 if contrib > 0 else -40),
                 textcoords='offset points', ha='center',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Show average
b_average = np.mean(b_contributions)
ax2.axhline(y=b_average, color='purple', linewidth=3, linestyle='--',
            label=f'Average: $b^* = {b_average:.2f}$')

ax2.set_ylabel('Bias Contribution')
ax2.set_title('Computing Bias from Support Vectors')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bias_computation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Simple visualization: KKT conditions in action
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a simple SVM example showing KKT conditions
np.random.seed(42)
# Points with different alpha values
points_pos = np.array([[1.0, 0.5], [1.5, 1.0], [2.0, 1.5]])  # SV, SV, NSV
points_neg = np.array([[-1.0, -0.5], [-1.5, -1.0], [-2.0, -1.5]])  # SV, SV, NSV
alpha_values = [0.8, 0.6, 0.0, 0.7, 0.5, 0.0]  # Corresponding alpha values

# Decision boundary
x_range = np.linspace(-3, 3, 100)
y_boundary = -0.7 * x_range
y_pos_margin = -0.7 * x_range + 1.0
y_neg_margin = -0.7 * x_range - 1.0

# Plot decision boundary and margins
ax.plot(x_range, y_boundary, 'k-', linewidth=3, label='Decision Boundary')
ax.plot(x_range, y_pos_margin, 'r--', linewidth=2, alpha=0.7)
ax.plot(x_range, y_neg_margin, 'b--', linewidth=2, alpha=0.7)

# Plot points with different sizes based on alpha values
all_points = np.vstack([points_pos, points_neg])
colors = ['red', 'red', 'red', 'blue', 'blue', 'blue']
markers = ['o', 'o', 'o', 's', 's', 's']

for i, (point, alpha, color, marker) in enumerate(zip(all_points, alpha_values, colors, markers)):
    size = 50 + alpha * 150  # Size proportional to alpha
    edge_width = 3 if alpha > 0 else 1
    edge_color = 'black' if alpha > 0 else 'gray'
    alpha_vis = 1.0 if alpha > 0 else 0.5

    ax.scatter(point[0], point[1], c=color, s=size, marker=marker,
               edgecolor=edge_color, linewidth=edge_width, alpha=alpha_vis)

# Add legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12,
           markeredgecolor='black', markeredgewidth=3, label='Support Vectors (+)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=12,
           markeredgecolor='black', markeredgewidth=3, label='Support Vectors (-)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8,
           markeredgecolor='gray', markeredgewidth=1, alpha=0.5, label='Non-Support Vectors')
]

ax.legend(handles=legend_elements, loc='upper right')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('KKT Conditions: Support Vector Identification')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kkt_simple.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
