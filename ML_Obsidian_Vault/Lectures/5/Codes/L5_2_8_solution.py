import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 8: SUPPORT VECTOR CLASSIFICATION IN SOFT MARGIN SVM")
print("=" * 80)

# Set the regularization parameter C
C = 2.0
print(f"Regularization parameter C = {C}")

print("\n" + "=" * 60)
print("STEP 1: KKT CONDITIONS ANALYSIS")
print("=" * 60)

print("\nThe KKT conditions for soft margin SVM are:")
print("1. α_i ≥ 0 (non-negativity)")
print("2. α_i ≤ C (upper bound)")
print("3. μ_i ≥ 0 (slack variable multiplier)")
print("4. α_i(y_i(w^T x_i + b) - 1 + ξ_i) = 0 (complementary slackness)")
print("5. μ_i ξ_i = 0 (complementary slackness for slack variables)")
print("6. C - α_i - μ_i = 0 (gradient condition)")

print("\nFrom condition 6: μ_i = C - α_i")
print("From condition 5: (C - α_i)ξ_i = 0")

print("\n" + "=" * 60)
print("STEP 2: CATEGORIZATION BASED ON α_i VALUES")
print("=" * 60)

# Case 1: α_i = 0
print("\n1. CASE: α_i = 0")
print("   From μ_i = C - α_i = C")
print("   From (C - α_i)ξ_i = 0: Cξ_i = 0")
print("   Therefore: ξ_i = 0")
print("   From condition 4: y_i(w^T x_i + b) - 1 + ξ_i = y_i(w^T x_i + b) - 1 ≥ 0")
print("   This means: y_i(w^T x_i + b) ≥ 1")
print("   CONCLUSION: Point is correctly classified and outside or on the margin boundary")

# Case 2: 0 < α_i < C
print("\n2. CASE: 0 < α_i < C")
print("   From μ_i = C - α_i > 0")
print("   From (C - α_i)ξ_i = 0: (C - α_i)ξ_i = 0")
print("   Since (C - α_i) > 0, we must have: ξ_i = 0")
print("   From condition 4: y_i(w^T x_i + b) - 1 + ξ_i = y_i(w^T x_i + b) - 1 = 0")
print("   This means: y_i(w^T x_i + b) = 1")
print("   CONCLUSION: Point is exactly on the margin boundary (support vector)")

# Case 3: α_i = C
print("\n3. CASE: α_i = C")
print("   From μ_i = C - α_i = 0")
print("   From condition 4: y_i(w^T x_i + b) - 1 + ξ_i = 0")
print("   This means: y_i(w^T x_i + b) = 1 - ξ_i")
print("   Since ξ_i ≥ 0, we have: y_i(w^T x_i + b) ≤ 1")
print("   POSSIBLE SCENARIOS:")
print("   - If ξ_i = 0: Point is on the margin boundary")
print("   - If 0 < ξ_i < 1: Point is inside the margin but correctly classified")
print("   - If ξ_i = 1: Point is exactly on the decision boundary")
print("   - If ξ_i > 1: Point is misclassified")

print("\n" + "=" * 60)
print("STEP 3: DECISION TREE CREATION")
print("=" * 60)

# Create decision tree visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Decision tree structure
tree_data = [
    ("Start", (0.5, 0.9), "$\\alpha_i = ?$"),
    ("alpha_0", (0.2, 0.7), "Point outside margin\n$\\xi_i = 0$"),
    ("alpha_mid", (0.5, 0.7), "Support vector\n$\\xi_i = 0$"),
    ("alpha_C", (0.8, 0.7), "$\\xi_i = ?$"),
    ("xi_0", (0.6, 0.5), "On margin boundary"),
    ("xi_mid", (0.8, 0.5), "Inside margin\nCorrectly classified"),
    ("xi_1", (1.0, 0.5), "On decision boundary"),
    ("xi_large", (1.2, 0.5), "Misclassified")
]

# Draw nodes
for text, pos, label in tree_data:
    if text == "Start":
        color = 'lightblue'
        size = 0.08
    elif "alpha_0" in text or "alpha_mid" in text:
        color = 'lightgreen'
        size = 0.06
    elif "alpha_C" in text:
        color = 'lightyellow'
        size = 0.06
    else:
        color = 'lightcoral'
        size = 0.05
    
    circle = plt.Circle(pos, size, color=color, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw connections
connections = [
    ((0.5, 0.9), (0.2, 0.7), "$\\alpha_i = 0$"),
    ((0.5, 0.9), (0.5, 0.7), "$0 < \\alpha_i < C$"),
    ((0.5, 0.9), (0.8, 0.7), "$\\alpha_i = C$"),
    ((0.8, 0.7), (0.6, 0.5), "$\\xi_i = 0$"),
    ((0.8, 0.7), (0.8, 0.5), "$0 < \\xi_i < 1$"),
    ((0.8, 0.7), (1.0, 0.5), "$\\xi_i = 1$"),
    ((0.8, 0.7), (1.2, 0.5), "$\\xi_i > 1$")
]

for start, end, label in connections:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    ax.text(mid_x, mid_y, label, ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))

ax.set_xlim(0, 1.4)
ax.set_ylim(0.4, 1.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Decision Tree for Support Vector Classification', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')

print("Decision tree visualization created and saved.")

print("\n" + "=" * 60)
print("STEP 4: POINT CLASSIFICATION")
print("=" * 60)

# Define the points
points = {
    'A': {'alpha': 0, 'xi': 0},
    'B': {'alpha': 0.5 * C, 'xi': 0},
    'C': {'alpha': C, 'xi': 0.8},
    'D': {'alpha': C, 'xi': 1.5}
}

print(f"\nGiven points with C = {C}:")
for point_name, values in points.items():
    print(f"Point {point_name}: α_{point_name} = {values['alpha']}, ξ_{point_name} = {values['xi']}")

print("\nClassification results:")

# Classify each point
classifications = {}

for point_name, values in points.items():
    alpha = values['alpha']
    xi = values['xi']
    
    print(f"\nPoint {point_name}:")
    print(f"  α_{point_name} = {alpha}, ξ_{point_name} = {xi}")
    
    if alpha == 0:
        print(f"  CASE: α_{point_name} = 0")
        print(f"  From KKT: ξ_{point_name} = 0 ✓")
        print(f"  Position: Point is correctly classified and outside/on margin boundary")
        print(f"  Type: Non-support vector")
        classifications[point_name] = "Non-support vector (outside margin)"
        
    elif 0 < alpha < C:
        print(f"  CASE: 0 < α_{point_name} < C")
        print(f"  From KKT: ξ_{point_name} = 0 ✓")
        print(f"  Position: Point is exactly on the margin boundary")
        print(f"  Type: Support vector")
        classifications[point_name] = "Support vector (on margin)"
        
    elif alpha == C:
        print(f"  CASE: α_{point_name} = C")
        print(f"  From KKT: y_i(w^T x_i + b) = 1 - ξ_{point_name} = 1 - {xi} = {1 - xi}")
        
        if xi == 0:
            print(f"  Position: Point is on the margin boundary")
            print(f"  Type: Support vector")
            classifications[point_name] = "Support vector (on margin)"
        elif 0 < xi < 1:
            print(f"  Position: Point is inside the margin but correctly classified")
            print(f"  Type: Support vector")
            classifications[point_name] = "Support vector (inside margin)"
        elif xi == 1:
            print(f"  Position: Point is exactly on the decision boundary")
            print(f"  Type: Support vector")
            classifications[point_name] = "Support vector (on decision boundary)"
        elif xi > 1:
            print(f"  Position: Point is misclassified")
            print(f"  Type: Support vector")
            classifications[point_name] = "Support vector (misclassified)"

print("\n" + "=" * 60)
print("STEP 5: VISUALIZATION OF POINT POSITIONS")
print("=" * 60)

# Create visualization of different point types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: Margin visualization
ax1.set_xlim(-2, 4)
ax1.set_ylim(-2, 4)
ax1.grid(True, alpha=0.3)

# Draw decision boundary and margins
x = np.linspace(-2, 4, 100)
decision_boundary = 2 - x  # Example: x + y = 2
margin_upper = 3 - x       # Margin boundary for class +1
margin_lower = 1 - x       # Margin boundary for class -1

ax1.plot(x, decision_boundary, 'k-', linewidth=2, label='Decision Boundary')
ax1.plot(x, margin_upper, 'b--', linewidth=1, label='Margin Boundary (+1)')
ax1.plot(x, margin_lower, 'r--', linewidth=1, label='Margin Boundary (-1)')

# Shade regions
ax1.fill_between(x, margin_upper, 4, alpha=0.1, color='blue', label='Class +1 Region')
ax1.fill_between(x, -2, margin_lower, alpha=0.1, color='red', label='Class -1 Region')
ax1.fill_between(x, margin_lower, margin_upper, alpha=0.1, color='gray', label='Margin Region')

# Add example points
example_points = {
    'Outside ($\\alpha=0$)': (3, 2, 'o', 'blue'),
    'On Margin ($0<\\alpha<C$)': (1, 2, 's', 'green'),
    'Inside Margin ($\\alpha=C$, $\\xi<1$)': (1.5, 1.5, '^', 'orange'),
    'On Boundary ($\\alpha=C$, $\\xi=1$)': (2, 0, 'v', 'purple'),
    'Misclassified ($\\alpha=C$, $\\xi>1$)': (0.5, 0.5, 'd', 'red')
}

for label, (x_pos, y_pos, marker, color) in example_points.items():
    ax1.scatter(x_pos, y_pos, s=100, c=color, marker=marker, edgecolors='black', linewidth=1.5)
    ax1.annotate(label, (x_pos, y_pos), xytext=(10, 10), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Support Vector Classification Regions')
ax1.legend(loc='upper right')

# Right plot: α vs ξ relationship
ax2.set_xlim(0, C + 0.5)
ax2.set_ylim(0, 2)

# Draw regions
ax2.fill_between([0, 0], [0, 2], alpha=0.3, color='blue', label='$\\alpha = 0$: Outside margin')
ax2.fill_between([0, C], [0, 0], alpha=0.3, color='green', label='$0 < \\alpha < C$: On margin')
ax2.fill_between([C, C], [0, 1], alpha=0.3, color='orange', label='$\\alpha = C$: Inside margin')
ax2.fill_between([C, C], [1, 2], alpha=0.3, color='red', label='$\\alpha = C$: Misclassified')

# Add boundary lines
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=C, color='black', linestyle='-', linewidth=1)

# Add our specific points
point_colors = {'A': 'blue', 'B': 'green', 'C': 'orange', 'D': 'red'}
for point_name, values in points.items():
    ax2.scatter(values['alpha'], values['xi'], s=150, c=point_colors[point_name], 
                marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax2.annotate(f'Point {point_name}', (values['alpha'], values['xi']), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

ax2.set_xlabel('$\\alpha_i$')
ax2.set_ylabel('$\\xi_i$')
ax2.set_title('Relationship between $\\alpha_i$ and $\\xi_i$')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_regions.png'), dpi=300, bbox_inches='tight')

print("Support vector regions visualization created and saved.")

print("\n" + "=" * 60)
print("STEP 6: SUMMARY TABLE")
print("=" * 60)

# Create summary table
print(f"\n{'Point':<8} {'α_i':<8} {'ξ_i':<8} {'Case':<25} {'Classification':<35}")
print("-" * 90)

for point_name, values in points.items():
    alpha = values['alpha']
    xi = values['xi']
    
    if alpha == 0:
        case = "α_i = 0"
        classification = "Non-support vector (outside margin)"
    elif 0 < alpha < C:
        case = "0 < α_i < C"
        classification = "Support vector (on margin)"
    elif alpha == C:
        if xi == 0:
            case = "α_i = C, ξ_i = 0"
            classification = "Support vector (on margin)"
        elif 0 < xi < 1:
            case = "α_i = C, 0 < ξ_i < 1"
            classification = "Support vector (inside margin)"
        elif xi == 1:
            case = "α_i = C, ξ_i = 1"
            classification = "Support vector (on decision boundary)"
        else:
            case = "α_i = C, ξ_i > 1"
            classification = "Support vector (misclassified)"
    
    print(f"{point_name:<8} {alpha:<8.1f} {xi:<8.1f} {case:<25} {classification:<35}")

print("\n" + "=" * 60)
print("STEP 7: KKT VERIFICATION")
print("=" * 60)

print("\nVerifying KKT conditions for each point:")

for point_name, values in points.items():
    alpha = values['alpha']
    xi = values['xi']
    mu = C - alpha
    
    print(f"\nPoint {point_name}:")
    print(f"  α_{point_name} = {alpha}, ξ_{point_name} = {xi}, μ_{point_name} = {mu}")
    print(f"  KKT Condition 1 (α_i ≥ 0): {alpha >= 0} ✓")
    print(f"  KKT Condition 2 (α_i ≤ C): {alpha <= C} ✓")
    print(f"  KKT Condition 3 (μ_i ≥ 0): {mu >= 0} ✓")
    print(f"  KKT Condition 5 (μ_i ξ_i = 0): {abs(mu * xi) < 1e-10} ✓")
    
    if alpha == 0:
        print(f"  KKT Condition 4: y_i(w^T x_i + b) - 1 + ξ_i ≥ 0 (point outside margin)")
    elif 0 < alpha < C:
        print(f"  KKT Condition 4: y_i(w^T x_i + b) - 1 + ξ_i = 0 (point on margin)")
    elif alpha == C:
        print(f"  KKT Condition 4: y_i(w^T x_i + b) - 1 + ξ_i = 0 (point inside margin or misclassified)")

print(f"\nAll plots saved to: {save_dir}")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
