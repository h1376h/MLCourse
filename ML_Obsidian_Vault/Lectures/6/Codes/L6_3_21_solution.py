import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 21: Decision Tree Interpretability and Decision Boundaries")
print("We need to compare how different decision tree algorithms create decision boundaries")
print("and understand their geometric interpretation in feature space.")
print()
print("Tasks:")
print("1. Compare tree interpretability across algorithms")
print("2. Understand geometric interpretation of decision boundaries")
print("3. Analyze how different algorithms create different boundary shapes")
print()

# Step 2: Understanding Decision Tree Algorithms
print_step_header(2, "Understanding Decision Tree Algorithms")

print("Key Decision Tree Algorithms:")
print("1. ID3 (Iterative Dichotomiser 3)")
print("   - Uses information gain for feature selection")
print("   - Creates axis-parallel splits only")
print("   - Handles categorical features")
print()
print("2. C4.5 (Successor to ID3)")
print("   - Extends ID3 with continuous feature handling")
print("   - Still creates axis-parallel splits")
print("   - More sophisticated than ID3 but similar geometric properties")
print()
print("3. CART (Classification and Regression Trees)")
print("   - Uses Gini impurity or MSE for splitting")
print("   - Creates binary splits (two children per node)")
print("   - Also creates axis-parallel splits")
print()

# Step 3: Geometric Interpretation of Decision Boundaries
print_step_header(3, "Geometric Interpretation of Decision Boundaries")

print("Decision boundaries in feature space:")
print("- Each split creates a hyperplane perpendicular to one feature axis")
print("- Multiple splits create rectangular regions (in 2D) or hyperrectangular regions (in higher dimensions)")
print("- All standard decision tree algorithms create axis-parallel boundaries")
print("- This is a fundamental limitation of decision trees")
print()

# Step 4: Creating Visual Examples
print_step_header(4, "Creating Visual Examples of Decision Boundaries")

# Create sample data for demonstration
np.random.seed(42)

# Generate synthetic data with clear patterns
n_samples = 1000
X = np.random.randn(n_samples, 2)
y = np.zeros(n_samples)

# Create a complex decision boundary using multiple axis-parallel splits
for i in range(n_samples):
    x1, x2 = X[i]
    if x1 < -0.5:
        if x2 < 0.2:
            y[i] = 0
        else:
            y[i] = 1
    elif x1 < 0.5:
        if x2 < -0.3:
            y[i] = 1
        else:
            y[i] = 0
    else:
        if x2 < 0.1:
            y[i] = 0
        else:
            y[i] = 1

# Step 5: Visualizing Raw Data with Decision Boundaries
print_step_header(5, "Visualizing Raw Data with Decision Boundaries")

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle(r'Raw Data with True Decision Boundary', fontsize=16, fontweight='bold')

# Plot the data points
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6, s=20)
ax.set_xlabel(r'Feature 1 ($x_1$)', fontsize=14)
ax.set_ylabel(r'Feature 2 ($x_2$)', fontsize=14)
ax.grid(True, alpha=0.3)

# Add decision boundary lines (axis-parallel)
ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, label=r'Split 1: $x_1 < -0.5$')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label=r'Split 2: $x_1 < 0.5$')
ax.axhline(y=0.2, color='blue', linestyle='--', linewidth=2, label=r'Split 3: $x_2 < 0.2$')
ax.axhline(y=-0.3, color='blue', linestyle='--', linewidth=2, label=r'Split 4: $x_2 < -0.3$')
ax.axhline(y=0.1, color='blue', linestyle='--', linewidth=2, label=r'Split 5: $x_2 < 0.1$')

ax.legend(fontsize=12)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'raw_data_with_boundaries.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 6: Visualizing ID3-like Decision Regions
print_step_header(6, "Visualizing ID3-like Decision Regions")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r'Feature 1 ($x_1$)', fontsize=14)
ax.set_ylabel(r'Feature 2 ($x_2$)', fontsize=14)
ax.set_title(r'ID3-like Decision Regions', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Define the rectangular regions created by ID3-like splits
regions = [
    # Region 1: x1 < -0.5, x2 < 0.2
    Rectangle((-3, -3), 2.5, 3.2, facecolor='lightblue', alpha=0.7, edgecolor='black'),
    # Region 2: x1 < -0.5, x2 >= 0.2
    Rectangle((-3, 0.2), 2.5, 2.8, facecolor='lightcoral', alpha=0.7, edgecolor='black'),
    # Region 3: -0.5 <= x1 < 0.5, x2 < -0.3
    Rectangle((-0.5, -3), 1.0, 2.7, facecolor='lightcoral', alpha=0.7, edgecolor='black'),
    # Region 4: -0.5 <= x1 < 0.5, x2 >= -0.3
    Rectangle((-0.5, -0.3), 1.0, 3.3, facecolor='lightblue', alpha=0.7, edgecolor='black'),
    # Region 5: x1 >= 0.5, x2 < 0.1
    Rectangle((0.5, -3), 2.5, 3.1, facecolor='lightblue', alpha=0.7, edgecolor='black'),
    # Region 6: x1 >= 0.5, x2 >= 0.1
    Rectangle((0.5, 0.1), 2.5, 2.9, facecolor='lightcoral', alpha=0.7, edgecolor='black')
]

for region in regions:
    ax.add_patch(region)

# Add region labels with LaTeX formatting
ax.text(-1.75, -1.5, r'Class 0: $x_1 < -0.5, x_2 < 0.2$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(-1.75, 1.5, r'Class 1: $x_1 < -0.5, x_2 \geq 0.2$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(0, -1.5, r'Class 1: $-0.5 \leq x_1 < 0.5, x_2 < -0.3$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(0, 1.5, r'Class 0: $-0.5 \leq x_1 < 0.5, x_2 \geq -0.3$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.75, -1.5, r'Class 0: $x_1 \geq 0.5, x_2 < 0.1$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.75, 1.5, r'Class 1: $x_1 \geq 0.5, x_2 \geq 0.1$', ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'id3_decision_regions.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 7: Visualizing CART-like Binary Splits
print_step_header(7, "Visualizing CART-like Binary Splits")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r'Feature 1 ($x_1$)', fontsize=14)
ax.set_ylabel(r'Feature 2 ($x_2$)', fontsize=14)
ax.set_title(r'CART-like Binary Splits', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Show how CART creates binary splits
ax.axvline(x=-0.5, color='red', linestyle='-', linewidth=3, label=r'Split 1: $x_1 < -0.5$')
ax.axvline(x=0.5, color='red', linestyle='-', linewidth=3, label=r'Split 2: $x_1 < 0.5$')
ax.axhline(y=0.2, color='blue', linestyle='-', linewidth=3, label=r'Split 3: $x_2 < 0.2$')
ax.axhline(y=-0.3, color='blue', linestyle='-', linewidth=3, label=r'Split 4: $x_2 < -0.3$')
ax.axhline(y=0.1, color='blue', linestyle='-', linewidth=3, label=r'Split 5: $x_2 < 0.1$')

# Add some sample points to show classification
sample_points = np.array([[-1, 0], [0, -1], [1, 0]])
sample_colors = ['lightblue', 'lightcoral', 'lightblue']
sample_labels = ['Class 0', 'Class 1', 'Class 0']

for i, (point, color, label) in enumerate(zip(sample_points, sample_colors, sample_labels)):
    ax.scatter(point[0], point[1], c=color, s=200, edgecolor='black', linewidth=2)
    ax.annotate(label, (point[0], point[1]), xytext=(10, 10), textcoords='offset points',
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

ax.legend(fontsize=12)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_binary_splits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 8: Visualizing Non-Axis-Parallel Comparison
print_step_header(8, "Visualizing Non-Axis-Parallel Comparison")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r'Feature 1 ($x_1$)', fontsize=14)
ax.set_ylabel(r'Feature 2 ($x_2$)', fontsize=14)
ax.set_title(r'Non-Axis-Parallel vs Axis-Parallel Boundaries', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Show what a non-axis-parallel boundary would look like
x_line = np.linspace(-3, 3, 100)
y_line = 0.5 * x_line + 0.3  # Diagonal line
ax.plot(x_line, y_line, 'g-', linewidth=3, label=r'Non-axis-parallel boundary: $x_2 = 0.5x_1 + 0.3$')

# Show axis-parallel approximation
ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label=r'Axis-parallel approximation')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=0.2, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=-0.3, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=0.1, color='blue', linestyle='--', linewidth=2, alpha=0.7)

# Add explanation text
ax.text(0, -2, r'Standard decision trees cannot create diagonal boundaries', 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.8))

ax.legend(fontsize=12)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'non_axis_parallel_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 9: Algorithm Comparison Table
print_step_header(9, "Algorithm Comparison Table")

print("Decision Boundary Characteristics by Algorithm:")
print("-" * 80)
print(f"{'Algorithm':<15} {'Split Type':<15} {'Boundary Shape':<20} {'Feature Handling':<20}")
print("-" * 80)
print(f"{'ID3':<15} {'Multi-way':<15} {'Axis-parallel':<20} {'Categorical only':<20}")
print(f"{'C4.5':<15} {'Multi-way':<15} {'Axis-parallel':<20} {'Categorical + Continuous':<20}")
print(f"{'CART':<15} {'Binary':<15} {'Axis-parallel':<20} {'Categorical + Continuous':<20}")
print("-" * 80)
print()

# Step 10: Key Insights
print_step_header(10, "Key Insights")

print("1. All standard decision tree algorithms create axis-parallel decision boundaries")
print("   - This is a fundamental geometric limitation")
print("   - Boundaries are always perpendicular to feature axes")
print("   - Results in rectangular decision regions")
print()
print("2. The difference between algorithms is in:")
print("   - How they select features (information gain vs Gini vs MSE)")
print("   - How they handle continuous features")
print("   - Split strategy (binary vs multi-way)")
print("   - Pruning and stopping criteria")
print()
print("3. Geometric implications:")
print("   - Trees work well when data has axis-parallel separability")
print("   - Trees struggle with diagonal or curved decision boundaries")
print("   - This limitation can be overcome with ensemble methods (Random Forest)")
print()

# Step 11: Answer to the Question
print_step_header(11, "Answer to the Question")

print("Question: Which statement correctly describes decision boundaries?")
print()
print("Correct Answer: A) ID3 creates axis-parallel rectangular regions in feature space")
print()
print("Explanation:")
print("- ID3, like all standard decision tree algorithms, creates axis-parallel splits")
print("- Each split is perpendicular to one feature axis")
print("- Multiple splits create rectangular regions in 2D (hyperrectangular in higher dimensions)")
print("- This is a fundamental property, not a limitation of ID3 specifically")
print()
print("Why other options are incorrect:")
print("- B) C4.5 cannot create diagonal boundaries - it still uses axis-parallel splits")
print("- C) CART's binary splits don't necessarily create more complex boundaries")
print("- D) Different algorithms may create different trees for the same dataset")
print()

print(f"\nVisualizations saved to: {save_dir}")
print("The plots show how decision trees create axis-parallel boundaries and rectangular decision regions.")
print("Each plot is saved as a separate file for better clarity and use.")
