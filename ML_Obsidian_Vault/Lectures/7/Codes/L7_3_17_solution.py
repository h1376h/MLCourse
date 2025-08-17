import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 17: RANDOM FOREST DECISION BOUNDARIES VISUALIZATION")
print("=" * 80)

# Given decision rules for each tree
print("\nGIVEN DECISION RULES:")
print("-" * 50)
print("Tree 1: X ≤ 3 → Class A, X > 3 → Class B")
print("Tree 2: Y ≤ 2 → Class A, Y > 2 → Class B")
print("Tree 3: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B")
print("Tree 4: X + Y ≤ 6 → Class A, X + Y > 6 → Class B")
print()

# Grid parameters
x_min, x_max = 0, 8
y_min, y_max = 0, 8
grid_size = 100

# Create coordinate grid
x = np.linspace(x_min, x_max, grid_size)
y = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x, y)

print("=" * 80)
print("SOLUTION")
print("=" * 80)

# Task 1: Individual Tree Decision Boundaries
print("\nTASK 1: INDIVIDUAL TREE DECISION BOUNDARIES")
print("-" * 50)

# Tree 1: X ≤ 3 → Class A, X > 3 → Class B
print("\nTree 1: X ≤ 3 → Class A, X > 3 → Class B")
print("This tree creates a vertical decision boundary at X = 3")
print("Left side (X ≤ 3): Class A (Blue)")
print("Right side (X > 3): Class B (Red)")

def tree1_decision(X, Y):
    """Tree 1 decision: X ≤ 3 → Class A, X > 3 → Class B"""
    return np.where(X <= 3, 0, 1)  # 0 for Class A, 1 for Class B

tree1_result = tree1_decision(X, Y)

# Plot Tree 1
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, tree1_result, levels=[-0.5, 0.5, 1.5], 
             colors=['blue', 'red'], alpha=0.7)
ax.contour(X, Y, tree1_result, levels=[0.5], colors='black', linewidths=2)
ax.axvline(x=3, color='black', linestyle='--', linewidth=3, label='X = 3')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_title('Tree 1 Decision Boundary: $X \\leq 3$ → Class A, $X > 3$ → Class B')
plt.colorbar(contour, ax=ax, ticks=[0.25, 0.75], label='Class')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(os.path.join(save_dir, 'tree1_decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Tree 2: Y ≤ 2 → Class A, Y > 2 → Class B
print("\nTree 2: Y ≤ 2 → Class A, Y > 2 → Class B")
print("This tree creates a horizontal decision boundary at Y = 2")
print("Bottom side (Y ≤ 2): Class A (Blue)")
print("Top side (Y > 2): Class B (Red)")

def tree2_decision(X, Y):
    """Tree 2 decision: Y ≤ 2 → Class A, Y > 2 → Class B"""
    return np.where(Y <= 2, 0, 1)  # 0 for Class A, 1 for Class B

tree2_result = tree2_decision(X, Y)

# Plot Tree 2
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, tree2_result, levels=[-0.5, 0.5, 1.5], 
             colors=['blue', 'red'], alpha=0.7)
ax.contour(X, Y, tree2_result, levels=[0.5], colors='black', linewidths=2)
ax.axhline(y=2, color='black', linestyle='--', linewidth=3, label='Y = 2')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_title('Tree 2 Decision Boundary: $Y \\leq 2$ → Class A, $Y > 2$ → Class B')
plt.colorbar(contour, ax=ax, ticks=[0.25, 0.75], label='Class')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(os.path.join(save_dir, 'tree2_decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Tree 3: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B
print("\nTree 3: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B")
print("This tree creates a rectangular decision boundary")
print("Inside rectangle (X ≤ 5 AND Y ≤ 4): Class A (Blue)")
print("Outside rectangle: Class B (Red)")

def tree3_decision(X, Y):
    """Tree 3 decision: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B"""
    return np.where((X <= 5) & (Y <= 4), 0, 1)  # 0 for Class A, 1 for Class B

tree3_result = tree3_decision(X, Y)

# Plot Tree 3
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, tree3_result, levels=[-0.5, 0.5, 1.5], 
             colors=['blue', 'red'], alpha=0.7)
ax.contour(X, Y, tree3_result, levels=[0.5], colors='black', linewidths=2)
ax.axvline(x=5, color='black', linestyle='--', linewidth=3, label='X = 5')
ax.axhline(y=4, color='black', linestyle='--', linewidth=3, label='Y = 4')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_title('Tree 3 Decision Boundary: $X \\leq 5$ AND $Y \\leq 4$ → Class A, otherwise Class B')
plt.colorbar(contour, ax=ax, ticks=[0.25, 0.75], label='Class')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(os.path.join(save_dir, 'tree3_decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Tree 4: X + Y ≤ 6 → Class A, X + Y > 6 → Class B
print("\nTree 4: X + Y ≤ 6 → Class A, X + Y > 6 → Class B")
print("This tree creates a diagonal decision boundary")
print("Below diagonal (X + Y ≤ 6): Class A (Blue)")
print("Above diagonal (X + Y > 6): Class B (Red)")

def tree4_decision(X, Y):
    """Tree 4 decision: X + Y ≤ 6 → Class A, X + Y > 6 → Class B"""
    return np.where(X + Y <= 6, 0, 1)  # 0 for Class A, 1 for Class B

tree4_result = tree4_decision(X, Y)

# Plot Tree 4
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, tree4_result, levels=[-0.5, 0.5, 1.5], 
             colors=['blue', 'red'], alpha=0.7)
ax.contour(X, Y, tree4_result, levels=[0.5], colors='black', linewidths=2)
# Plot the diagonal line X + Y = 6
diagonal_x = np.linspace(0, 6, 100)
diagonal_y = 6 - diagonal_x
ax.plot(diagonal_x, diagonal_y, 'black', linestyle='--', linewidth=3, label='X + Y = 6')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_title('Tree 4 Decision Boundary: $X + Y \\leq 6$ → Class A, $X + Y > 6$ → Class B')
plt.colorbar(contour, ax=ax, ticks=[0.25, 0.75], label='Class')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(os.path.join(save_dir, 'tree4_decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 2: All Trees Combined Visualization
print("\nTASK 2: ALL TREES COMBINED VISUALIZATION")
print("-" * 50)
print("Showing all decision boundaries on the same plot for comparison")

# Create combined plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all decision boundaries
ax.contour(X, Y, tree1_result, levels=[0.5], colors='red', linewidths=3, label='Tree 1: X = 3')
ax.contour(X, Y, tree2_result, levels=[0.5], colors='blue', linewidths=3, label='Tree 2: Y = 2')
ax.contour(X, Y, tree3_result, levels=[0.5], colors='green', linewidths=3, label='Tree 3: X=5, Y=4')
ax.contour(X, Y, tree4_result, levels=[0.5], colors='purple', linewidths=3, label='Tree 4: X+Y=6')

# Add the specific point (4, 3) for Task 3
ax.scatter(4, 3, color='orange', s=200, marker='*', edgecolor='black', linewidth=2, 
           label='Point (4, 3)', zorder=10)

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_title('All Random Forest Tree Decision Boundaries')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_trees_combined.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 3: Ensemble Prediction for Point (4, 3)
print("\nTASK 3: ENSEMBLE PREDICTION FOR POINT (4, 3)")
print("-" * 50)

point = (4, 3)
print(f"Analyzing point: {point}")

print("\nDETAILED STEP-BY-STEP CALCULATIONS:")
print("=" * 50)

# Tree 1: X ≤ 3 → Class A, X > 3 → Class B
print(f"\nTree 1 Decision Rule: X ≤ 3 → Class A, X > 3 → Class B")
print(f"Given point: X = {point[0]}, Y = {point[1]}")
print(f"Step 1: Check condition X ≤ 3")
print(f"         {point[0]} ≤ 3? {point[0] <= 3}")
print(f"Step 2: Since {point[0]} > 3, the condition is FALSE")
print(f"Step 3: Therefore, Tree 1 predicts: Class B")
tree1_pred = tree1_decision(point[0], point[1])
print(f"Verification: tree1_decision({point[0]}, {point[1]}) = {tree1_pred} → Class {'A' if tree1_pred == 0 else 'B'}")

# Tree 2: Y ≤ 2 → Class A, Y > 2 → Class B
print(f"\nTree 2 Decision Rule: Y ≤ 2 → Class A, Y > 2 → Class B")
print(f"Given point: X = {point[0]}, Y = {point[1]}")
print(f"Step 1: Check condition Y ≤ 2")
print(f"         {point[1]} ≤ 2? {point[1] <= 2}")
print(f"Step 2: Since {point[1]} > 2, the condition is FALSE")
print(f"Step 3: Therefore, Tree 2 predicts: Class B")
tree2_pred = tree2_decision(point[0], point[1])
print(f"Verification: tree2_decision({point[0]}, {point[1]}) = {tree2_pred} → Class {'A' if tree2_pred == 0 else 'B'}")

# Tree 3: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B
print(f"\nTree 3 Decision Rule: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B")
print(f"Given point: X = {point[0]}, Y = {point[1]}")
print(f"Step 1: Check first condition X ≤ 5")
print(f"         {point[0]} ≤ 5? {point[0] <= 5}")
print(f"Step 2: Check second condition Y ≤ 4")
print(f"         {point[1]} ≤ 4? {point[1] <= 4}")
print(f"Step 3: Apply AND logic: {point[0] <= 5} AND {point[1] <= 4} = {(point[0] <= 5) and (point[1] <= 4)}")
print(f"Step 4: Since both conditions are TRUE, Tree 3 predicts: Class A")
tree3_pred = tree3_decision(point[0], point[1])
print(f"Verification: tree3_decision({point[0]}, {point[1]}) = {tree3_pred} → Class {'A' if tree3_pred == 0 else 'B'}")

# Tree 4: X + Y ≤ 6 → Class A, X + Y > 6 → Class B
print(f"\nTree 4 Decision Rule: X + Y ≤ 6 → Class A, X + Y > 6 → Class B")
print(f"Given point: X = {point[0]}, Y = {point[1]}")
print(f"Step 1: Calculate X + Y")
print(f"         X + Y = {point[0]} + {point[1]} = {point[0] + point[1]}")
print(f"Step 2: Check condition X + Y ≤ 6")
print(f"         {point[0] + point[1]} ≤ 6? {point[0] + point[1] <= 6}")
print(f"Step 3: Since {point[0] + point[1]} > 6, the condition is FALSE")
print(f"Step 4: Therefore, Tree 4 predicts: Class B")
tree4_pred = tree4_decision(point[0], point[1])
print(f"Verification: tree4_decision({point[0]}, {point[1]}) = {tree4_pred} → Class {'A' if tree4_pred == 0 else 'B'}")

print(f"\nENSEMBLE VOTING CALCULATION:")
print("=" * 50)
print(f"Step 1: Collect all tree predictions:")
print(f"         Tree 1: Class {'A' if tree1_pred == 0 else 'B'}")
print(f"         Tree 2: Class {'A' if tree2_pred == 0 else 'B'}")
print(f"         Tree 3: Class {'A' if tree3_pred == 0 else 'B'}")
print(f"         Tree 4: Class {'A' if tree4_pred == 0 else 'B'}")

predictions = [tree1_pred, tree2_pred, tree3_pred, tree4_pred]
class_a_votes = sum(1 for p in predictions if p == 0)
class_b_votes = sum(1 for p in predictions if p == 1)

print(f"\nStep 2: Count votes for each class:")
print(f"         Class A votes: {class_a_votes}")
print(f"         Class B votes: {class_b_votes}")

print(f"\nStep 3: Apply majority voting rule:")
print(f"         If Class A votes > Class B votes → Final prediction: Class A")
print(f"         If Class B votes > Class A votes → Final prediction: Class B")
print(f"         If votes are equal → Random choice (tie)")

if class_a_votes > class_b_votes:
    ensemble_prediction = "Class A"
    winning_votes = class_a_votes
elif class_b_votes > class_a_votes:
    ensemble_prediction = "Class B"
    winning_votes = class_b_votes
else:
    ensemble_prediction = "Tie (random choice)"
    winning_votes = "N/A"

print(f"\nStep 4: Determine winner:")
print(f"         {class_a_votes} vs {class_b_votes} votes")
print(f"         Winner: {ensemble_prediction}")
if winning_votes != "N/A":
    print(f"         Winning percentage: {winning_votes}/4 = {winning_votes/4*100:.1f}%")

print(f"\nFINAL RESULT:")
print(f"Ensemble prediction for point ({point[0]}, {point[1]}): {ensemble_prediction}")
if winning_votes != "N/A":
    print(f"Confidence: {winning_votes}/4 votes ({winning_votes/4*100:.1f}%)")

# Task 4: Most Interesting Geometric Pattern
print("\nTASK 4: MOST INTERESTING GEOMETRIC PATTERN")
print("-" * 50)

print("\nDETAILED GEOMETRIC ANALYSIS:")
print("=" * 50)

print("\nTree 1: X ≤ 3 → Class A, X > 3 → Class B")
print("Geometric characteristics:")
print("         - Boundary: Vertical line at X = 3")
print("         - Shape: Infinite half-planes (left and right)")
print("         - Complexity: Simple univariate split")
print("         - Direction: Parallel to Y-axis")

print("\nTree 2: Y ≤ 2 → Class A, Y > 2 → Class B")
print("Geometric characteristics:")
print("         - Boundary: Horizontal line at Y = 2")
print("         - Shape: Infinite half-planes (bottom and top)")
print("         - Complexity: Simple univariate split")
print("         - Direction: Parallel to X-axis")

print("\nTree 3: X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B")
print("Geometric characteristics:")
print("         - Boundary: Rectangle with corners at (0,0), (5,0), (5,4), (0,4)")
print("         - Shape: Bounded rectangular region")
print("         - Complexity: Multivariate split with AND condition")
print("         - Direction: Creates enclosed area with finite boundaries")

print("\nTree 4: X + Y ≤ 6 → Class A, X + Y > 6 → Class B")
print("Geometric characteristics:")
print("         - Boundary: Diagonal line X + Y = 6")
print("         - Shape: Infinite half-planes (below and above diagonal)")
print("         - Complexity: Linear combination of features")
print("         - Direction: 45-degree angle (slope = -1)")

print("\nCOMPARATIVE ANALYSIS:")
print("=" * 50)
print("Complexity ranking (from simple to complex):")
print("1. Trees 1 & 2: Simple linear boundaries (univariate)")
print("2. Tree 4: Diagonal boundary (linear combination)")
print("3. Tree 3: Rectangular boundary (multivariate with AND)")

print("\nTree 3 creates the most interesting geometric pattern because:")
print("1. BOUNDED REGION: Unlike infinite half-planes, creates finite rectangular area")
print("2. MULTIVARIATE SPLIT: Uses both X and Y coordinates simultaneously")
print("3. LOGICAL COMPLEXITY: AND condition creates intersection of constraints")
print("4. PRACTICAL RELEVANCE: Represents real-world scenarios with multiple conditions")
print("5. GEOMETRIC UNIQUENESS: Only tree that creates enclosed classification region")

print("\nMATHEMATICAL REPRESENTATION:")
print("Tree 3 boundary: {(X,Y) | X ≤ 5 AND Y ≤ 4}")
print("This creates a closed set in 2D space, unlike the open half-planes of other trees.")

# Task 5: Calculate Area Percentage Where Ensemble Differs
print("\nTASK 5: AREA PERCENTAGE WHERE ENSEMBLE DIFFERS FROM INDIVIDUAL TREES")
print("-" * 50)

print("\nDETAILED CALCULATION STEPS:")
print("=" * 50)

def ensemble_decision(X, Y):
    """Ensemble decision using majority voting"""
    tree1_pred = tree1_decision(X, Y)
    tree2_pred = tree2_decision(X, Y)
    tree3_pred = tree3_decision(X, Y)
    tree4_pred = tree4_decision(X, Y)
    
    # Count votes for each class at each point
    votes = tree1_pred + tree2_pred + tree3_pred + tree4_pred
    # Majority vote: if votes >= 2, Class B (1), otherwise Class A (0)
    return np.where(votes >= 2, 1, 0)

print("Step 1: Define ensemble decision function")
print("         For each point (X, Y), collect predictions from all 4 trees")
print("         Apply majority voting: if ≥2 trees predict Class B → Class B, else Class A")

ensemble_result = ensemble_decision(X, Y)
print("Step 2: Generate ensemble decision for entire grid")
print(f"         Grid dimensions: {X.shape[0]} × {X.shape[1]} = {X.shape[0] * X.shape[1]} points")

print("\nStep 3: Calculate differences between ensemble and individual trees")
print("         For each tree, compute: |ensemble_prediction - tree_prediction|")
print("         Result: 0 if same prediction, 1 if different prediction")

# Calculate differences between ensemble and individual trees
diff_tree1 = np.abs(ensemble_result - tree1_result)
diff_tree2 = np.abs(ensemble_result - tree2_result)
diff_tree3 = np.abs(ensemble_result - tree3_result)
diff_tree4 = np.abs(ensemble_result - tree4_result)

print(f"         Tree 1 differences: {np.sum(diff_tree1)} points")
print(f"         Tree 2 differences: {np.sum(diff_tree2)} points")
print(f"         Tree 3 differences: {np.sum(diff_tree3)} points")
print(f"         Tree 4 differences: {np.sum(diff_tree4)} points")

print("\nStep 4: Find total area where ensemble differs from ANY individual tree")
print("         Use logical OR operation: total_diff = diff_tree1 OR diff_tree2 OR diff_tree3 OR diff_tree4")
print("         This identifies points where ensemble differs from at least one tree")

# Total area where ensemble differs from any individual tree
total_diff = np.logical_or.reduce([diff_tree1, diff_tree2, diff_tree3, diff_tree4])

print("\nStep 5: Calculate percentage")
print(f"         Grid size: {grid_size} × {grid_size} = {grid_size * grid_size} total points")
print(f"         Points with differences: {np.sum(total_diff)}")
print(f"         Percentage calculation: ({np.sum(total_diff)} / {grid_size * grid_size}) × 100")

# Calculate percentages
total_points = grid_size * grid_size
diff_points = np.sum(total_diff)
diff_percentage = (diff_points / total_points) * 100

print(f"\nFINAL RESULT:")
print(f"Grid size: {grid_size} × {grid_size} = {total_points} total points")
print(f"Points where ensemble differs from any individual tree: {diff_points}")
print(f"Percentage of grid area: {diff_percentage:.2f}%")

print(f"\nINTERPRETATION:")
print(f"This means that in {diff_percentage:.1f}% of the feature space, the ensemble")
print(f"makes a different prediction than at least one of the individual trees.")
print(f"This demonstrates the ensemble's ability to create more nuanced decision boundaries.")

# Visualize the differences
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Subplot 1: Ensemble decision boundary
contour1 = axes[0, 0].contourf(X, Y, ensemble_result, levels=[-0.5, 0.5, 1.5], 
             colors=['blue', 'red'], alpha=0.7)
axes[0, 0].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2)
axes[0, 0].set_xlabel('$X$')
axes[0, 0].set_ylabel('$Y$')
axes[0, 0].set_title('Ensemble Decision Boundary\n(Majority Vote)')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Differences with Tree 1
contour2 = axes[0, 1].contourf(X, Y, diff_tree1, levels=[-0.5, 0.5, 1.5], 
             colors=['white', 'yellow'], alpha=0.7)
axes[0, 1].contour(X, Y, tree1_result, levels=[0.5], colors='red', linewidths=2, label='Tree 1')
axes[0, 1].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2, label='Ensemble')
axes[0, 1].set_xlabel('$X$')
axes[0, 1].set_ylabel('$Y$')
axes[0, 1].set_title('Differences: Ensemble vs Tree 1\n(Yellow = Different)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Subplot 3: Differences with Tree 2
contour3 = axes[0, 2].contourf(X, Y, diff_tree2, levels=[-0.5, 0.5, 1.5], 
             colors=['white', 'yellow'], alpha=0.7)
axes[0, 2].contour(X, Y, tree2_result, levels=[0.5], colors='blue', linewidths=2, label='Tree 2')
axes[0, 2].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2, label='Ensemble')
axes[0, 2].set_xlabel('$X$')
axes[0, 2].set_ylabel('$Y$')
axes[0, 2].set_title('Differences: Ensemble vs Tree 2\n(Yellow = Different)')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# Subplot 4: Differences with Tree 3
contour4 = axes[1, 0].contourf(X, Y, diff_tree3, levels=[-0.5, 0.5, 1.5], 
             colors=['white', 'yellow'], alpha=0.7)
axes[1, 0].contour(X, Y, tree3_result, levels=[0.5], colors='green', linewidths=2, label='Tree 3')
axes[1, 0].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2, label='Ensemble')
axes[1, 0].set_xlabel('$X$')
axes[1, 0].set_yticks([])
axes[1, 0].set_title('Differences: Ensemble vs Tree 3\n(Yellow = Different)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Subplot 5: Differences with Tree 4
contour5 = axes[1, 1].contourf(X, Y, diff_tree4, levels=[-0.5, 0.5, 1.5], 
             colors=['white', 'yellow'], alpha=0.7)
axes[1, 1].contour(X, Y, tree4_result, levels=[0.5], colors='purple', linewidths=2, label='Tree 4')
axes[1, 1].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2, label='Ensemble')
axes[1, 1].set_xlabel('$X$')
axes[1, 1].set_yticks([])
axes[1, 1].set_title('Differences: Ensemble vs Tree 4\n(Yellow = Different)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# Subplot 6: Total differences
contour6 = axes[1, 2].contourf(X, Y, total_diff, levels=[-0.5, 0.5, 1.5], 
             colors=['white', 'yellow'], alpha=0.7)
axes[1, 2].contour(X, Y, ensemble_result, levels=[0.5], colors='black', linewidths=2, label='Ensemble')
axes[1, 2].set_xlabel('$X$')
axes[1, 2].set_yticks([])
axes[1, 2].set_title(f'Total Differences\n(Yellow = {diff_percentage:.1f}% of area)')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ensemble_differences_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Individual tree decision boundaries have been visualized")
print(f"2. All boundaries are shown on a combined plot")
print(f"3. Point (4, 3) ensemble prediction: {ensemble_prediction}")
print(f"4. Most interesting pattern: Tree 3 (rectangular boundary)")
print(f"5. Area where ensemble differs from individual trees: {diff_percentage:.2f}%")
print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
