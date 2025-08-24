import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("SVM Slack Variable Analysis")
print("=" * 50)

# Given data points and their slack variables
points = np.array([
    [2, 1],   # x1
    [1, 2],   # x2
    [0, 0],   # x3
    [1, 0]    # x4
])

labels = np.array([1, 1, -1, -1])  # y1, y2, y3, y4
slack_vars = np.array([0, 0.3, 0, 1.2])  # ξ1, ξ2, ξ3, ξ4
point_names = ['x1', 'x2', 'x3', 'x4']

# Given hyperplane: x1 + x2 - 1.5 = 0
# This can be written as w^T x + b = 0 where w = [1, 1] and b = -1.5
w = np.array([1, 1])
b = -1.5

print("\nGiven Information:")
print(f"Hyperplane equation: x1 + x2 - 1.5 = 0")
print(f"Weight vector w = {w}")
print(f"Bias term b = {b}")
print("\nTraining points and slack variables:")
for i in range(len(points)):
    print(f"Point {point_names[i]}: {points[i]}, label y{i+1} = {labels[i]:+2d}, slack ξ{i+1} = {slack_vars[i]}")

# Step 1: Geometric interpretation of slack variables
print("\n" + "="*50)
print("STEP 1: GEOMETRIC INTERPRETATION OF SLACK VARIABLES")
print("="*50)

print("\nSlack variable interpretation:")
print("ξ = 0     : Point is correctly classified and outside or on the margin")
print("0 < ξ < 1 : Point is correctly classified but inside the margin")  
print("ξ = 1     : Point lies exactly on the decision boundary")
print("ξ > 1     : Point is misclassified")

for i in range(len(slack_vars)):
    xi = slack_vars[i]
    if xi == 0:
        interpretation = "Correctly classified, outside or on the margin"
    elif 0 < xi < 1:
        interpretation = "Correctly classified, but inside the margin"
    elif xi == 1:
        interpretation = "On the decision boundary"
    else:  # xi > 1
        interpretation = "Misclassified"
    
    print(f"Point {point_names[i]}: ξ{i+1} = {xi} → {interpretation}")

# Step 2: Calculate distances and verify classifications
print("\n" + "="*50)
print("STEP 2: VERIFY CLASSIFICATIONS AND DISTANCES")
print("="*50)

# Calculate the margin width
margin_width = 2 / np.linalg.norm(w)
print(f"\nMargin width = 2/||w|| = 2/{np.linalg.norm(w):.3f} = {margin_width:.3f}")

# For each point, calculate:
# 1. Distance to hyperplane
# 2. Functional margin
# 3. Geometric margin
# 4. Classification result

print("\nDetailed analysis for each point:")
for i in range(len(points)):
    x = points[i]
    y = labels[i]
    xi = slack_vars[i]
    
    print(f"\n--- Point {point_names[i]}: {x}, y{i+1} = {y} ---")
    
    # Calculate signed distance to hyperplane
    signed_distance = (np.dot(w, x) + b) / np.linalg.norm(w)
    print(f"Signed distance to hyperplane: ({w[0]}*{x[0]} + {w[1]}*{x[1]} + {b}) / {np.linalg.norm(w):.3f} = {signed_distance:.3f}")
    
    # Functional margin
    functional_margin = y * (np.dot(w, x) + b)
    print(f"Functional margin: y * (w^T x + b) = {y} * {np.dot(w, x) + b:.3f} = {functional_margin:.3f}")
    
    # Geometric margin
    geometric_margin = functional_margin / np.linalg.norm(w)
    print(f"Geometric margin: {functional_margin:.3f} / {np.linalg.norm(w):.3f} = {geometric_margin:.3f}")
    
    # Classification
    predicted = 1 if np.dot(w, x) + b > 0 else -1
    is_correct = predicted == y
    print(f"Predicted class: {predicted:+2d}, True class: {y:+2d}, Correct: {is_correct}")
    
    # Margin analysis
    if geometric_margin >= 1:
        margin_status = "Outside margin (correctly classified)"
    elif geometric_margin > 0:
        margin_status = "Inside margin (correctly classified)"
    elif geometric_margin == 0:
        margin_status = "On decision boundary"
    else:
        margin_status = "Wrong side of boundary (misclassified)"
    
    print(f"Margin status: {margin_status}")
    
    # Verify slack variable
    expected_slack = max(0, 1 - geometric_margin)
    print(f"Expected slack variable: max(0, 1 - {geometric_margin:.3f}) = {expected_slack:.3f}")
    print(f"Given slack variable: ξ{i+1} = {xi}")
    
    if abs(expected_slack - xi) < 1e-10:
        print("✓ Slack variable matches calculation")
    else:
        print("✗ Slack variable does not match calculation")

# Step 3: Create comprehensive visualization
print("\n" + "="*50)
print("STEP 3: CREATING VISUALIZATIONS")
print("="*50)

# Create the main plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Complete SVM visualization
x1_range = np.linspace(-1, 3, 100)
x2_hyperplane = (-w[0] * x1_range - b) / w[1]
x2_margin_pos = (-w[0] * x1_range - b + 1) / w[1]  # Positive margin
x2_margin_neg = (-w[0] * x1_range - b - 1) / w[1]  # Negative margin

# Plot hyperplane and margins
ax1.plot(x1_range, x2_hyperplane, 'k-', linewidth=2, label='Decision Boundary')
ax1.plot(x1_range, x2_margin_pos, 'k--', linewidth=1, alpha=0.7, label='Margin Boundaries')
ax1.plot(x1_range, x2_margin_neg, 'k--', linewidth=1, alpha=0.7)

# Fill margin area
ax1.fill_between(x1_range, x2_margin_pos, x2_margin_neg, alpha=0.2, color='gray', label='Margin')

# Plot points with different styles based on their properties
colors = ['red', 'blue']  # red for positive class, blue for negative class
markers = ['o', 's', '^', 'D']  # Different markers for each point

for i in range(len(points)):
    x = points[i]
    y = labels[i]
    xi = slack_vars[i]
    
    # Color based on class
    color = colors[0] if y == 1 else colors[1]
    
    # Marker size based on slack variable
    size = 100 + xi * 50  # Larger for higher slack
    
    # Edge style based on classification
    if xi == 0:
        edge_style = 'solid'
        edge_width = 2
    elif 0 < xi < 1:
        edge_style = 'dashed'
        edge_width = 2
    else:  # xi >= 1 (misclassified)
        edge_style = 'dotted'
        edge_width = 3
    
    ax1.scatter(x[0], x[1], c=color, s=size, marker=markers[i], 
               edgecolors='black', linewidth=edge_width, linestyle=edge_style,
               label=f'{point_names[i]} ($\\xi={xi}$)', alpha=0.8)
    
    # Add point labels
    ax1.annotate(f'{point_names[i]}\n({x[0]}, {x[1]})\n$\\xi={xi}$',
                (x[0], x[1]), xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('SVM with Slack Variables')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)

# Plot 2: Slack variable values as bar chart
ax2.bar(range(len(slack_vars)), slack_vars, 
        color=['lightcoral' if xi > 1 else 'lightblue' if xi > 0 else 'lightgreen' for xi in slack_vars],
        edgecolor='black', linewidth=1)

ax2.set_xlabel('Data Points')
ax2.set_ylabel('Slack Variable Value ($\\xi$)')
ax2.set_title('Slack Variable Values')
ax2.set_xticks(range(len(slack_vars)))
ax2.set_xticklabels([f'{point_names[i]}\n{points[i]}' for i in range(len(points))])
ax2.grid(True, alpha=0.3, axis='y')

# Add horizontal lines for interpretation
ax2.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='$\\xi = 0$ (on/outside margin)')
ax2.axhline(y=1, color='orange', linestyle='-', alpha=0.7, label='$\\xi = 1$ (on boundary)')

# Add value labels on bars
for i, v in enumerate(slack_vars):
    ax2.text(i, v + 0.05, f'{v}', ha='center', va='bottom', fontweight='bold')

ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_slack_variables_analysis.png'), dpi=300, bbox_inches='tight')

# Step 4: Detailed geometric analysis plot
fig2, ax3 = plt.subplots(1, 1, figsize=(12, 10))

# Plot the same elements as before
ax3.plot(x1_range, x2_hyperplane, 'k-', linewidth=3, label='Decision Boundary: $x_1 + x_2 - 1.5 = 0$')
ax3.plot(x1_range, x2_margin_pos, 'k--', linewidth=2, alpha=0.8, label='Positive Margin: $x_1 + x_2 - 0.5 = 0$')
ax3.plot(x1_range, x2_margin_neg, 'k--', linewidth=2, alpha=0.8, label='Negative Margin: $x_1 + x_2 - 2.5 = 0$')

# Fill regions with different colors
ax3.fill_between(x1_range, x2_margin_pos, 3, alpha=0.2, color='red', label='Positive Class Region')
ax3.fill_between(x1_range, -1, x2_margin_neg, alpha=0.2, color='blue', label='Negative Class Region')
ax3.fill_between(x1_range, x2_margin_pos, x2_margin_neg, alpha=0.3, color='yellow', label='Margin (no points should be here)')

# Plot points with distance vectors
for i in range(len(points)):
    x = points[i]
    y = labels[i]
    xi = slack_vars[i]
    
    color = 'red' if y == 1 else 'blue'
    
    # Plot the point
    ax3.scatter(x[0], x[1], c=color, s=200, marker='o', 
               edgecolors='black', linewidth=2, zorder=5)
    
    # Calculate the closest point on the hyperplane
    # The perpendicular from point to line w^T x + b = 0
    # Closest point = x - (w^T x + b) / ||w||^2 * w
    distance_to_hyperplane = (np.dot(w, x) + b) / np.linalg.norm(w)**2
    closest_point = x - distance_to_hyperplane * w
    
    # Draw distance line
    ax3.plot([x[0], closest_point[0]], [x[1], closest_point[1]], 
            'g--', linewidth=2, alpha=0.7)
    
    # Add detailed annotations
    distance = abs(np.dot(w, x) + b) / np.linalg.norm(w)
    geometric_margin = y * (np.dot(w, x) + b) / np.linalg.norm(w)
    
    annotation = f'{point_names[i]}: ({x[0]}, {x[1]})\n'
    annotation += f'Class: {y:+d}\n'
    annotation += f'Distance: {distance:.3f}\n'
    annotation += f'Geo. Margin: {geometric_margin:.3f}\n'
    annotation += f'Slack $\\xi$: {xi}'
    
    ax3.annotate(annotation, (x[0], x[1]), xytext=(15, 15), 
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.set_title('Detailed Geometric Analysis of SVM with Slack Variables')
ax3.grid(True, alpha=0.3)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.set_xlim(-0.5, 2.5)
ax3.set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_geometric_analysis.png'), dpi=300, bbox_inches='tight')

# Step 5: Calculate total penalty
print("\n" + "="*50)
print("STEP 5: CALCULATE TOTAL PENALTY")
print("="*50)

total_penalty = np.sum(slack_vars)
print(f"\nTotal penalty: Σξᵢ = {' + '.join([str(xi) for xi in slack_vars])} = {total_penalty}")

print(f"\nThis penalty is added to the objective function:")
print(f"Minimize: (1/2)||w||² + C * Σξᵢ")
print(f"Where C is the regularization parameter that controls the trade-off")
print(f"between margin maximization and training error minimization.")

# Summary table
print("\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)

print(f"{'Point':<6} {'Coordinates':<12} {'Label':<6} {'Slack':<6} {'Status':<30}")
print("-" * 70)

for i in range(len(points)):
    x = points[i]
    y = labels[i]
    xi = slack_vars[i]
    
    if xi == 0:
        status = "Correct, outside/on margin"
    elif 0 < xi < 1:
        status = "Correct, inside margin"
    elif xi == 1:
        status = "On decision boundary"
    else:
        status = "Misclassified"
    
    print(f"{point_names[i]:<6} {str(tuple(x)):<12} {y:+2d}    {xi:<6} {status:<30}")

print(f"\nTotal penalty: {total_penalty}")
print(f"Number of support vectors: {np.sum(slack_vars > 0)} (points with ξ > 0)")
print(f"Number of misclassified points: {np.sum(slack_vars > 1)}")

print(f"\nPlots saved to: {save_dir}")

# Create a summary visualization
fig3, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Classification accuracy
correct = [xi <= 1 for xi in slack_vars]
accuracy_labels = ['Correct', 'Misclassified']
accuracy_counts = [sum(correct), len(correct) - sum(correct)]
colors_acc = ['lightgreen', 'lightcoral']

ax4.pie(accuracy_counts, labels=accuracy_labels, colors=colors_acc, autopct='%1.1f%%')
ax4.set_title('Classification Accuracy')

# Subplot 2: Margin analysis
margin_categories = []
margin_counts = []
margin_colors = []

outside_margin = sum([1 for xi in slack_vars if xi == 0])
inside_margin = sum([1 for xi in slack_vars if 0 < xi < 1])
on_boundary = sum([1 for xi in slack_vars if xi == 1])
misclassified = sum([1 for xi in slack_vars if xi > 1])

if outside_margin > 0:
    margin_categories.append('Outside/On Margin')
    margin_counts.append(outside_margin)
    margin_colors.append('lightgreen')

if inside_margin > 0:
    margin_categories.append('Inside Margin')
    margin_counts.append(inside_margin)
    margin_colors.append('lightyellow')

if on_boundary > 0:
    margin_categories.append('On Boundary')
    margin_counts.append(on_boundary)
    margin_colors.append('orange')

if misclassified > 0:
    margin_categories.append('Misclassified')
    margin_counts.append(misclassified)
    margin_colors.append('lightcoral')

ax5.pie(margin_counts, labels=margin_categories, colors=margin_colors, autopct='%1.1f%%')
ax5.set_title('Margin Analysis')

# Subplot 3: Slack variable distribution
ax6.hist(slack_vars, bins=np.arange(0, max(slack_vars) + 0.5, 0.3), 
         color='skyblue', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Slack Variable Value')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Slack Variables')
ax6.grid(True, alpha=0.3)

# Subplot 4: Point classification details
point_info = []
for i in range(len(points)):
    geometric_margin = labels[i] * (np.dot(w, points[i]) + b) / np.linalg.norm(w)
    point_info.append([point_names[i], geometric_margin, slack_vars[i]])

# Sort by geometric margin
point_info.sort(key=lambda x: x[1])

x_pos = range(len(point_info))
geometric_margins = [info[1] for info in point_info]
slack_values = [info[2] for info in point_info]
point_labels = [info[0] for info in point_info]

ax7.bar(x_pos, geometric_margins, alpha=0.7, color='lightblue', label='Geometric Margin')
ax7.bar(x_pos, [-sv for sv in slack_values], alpha=0.7, color='lightcoral', label='Negative Slack')
ax7.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Margin Boundary')
ax7.axhline(y=0, color='black', linestyle='-', alpha=0.7, label='Decision Boundary')
ax7.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Opposite Margin')

ax7.set_xlabel('Data Points')
ax7.set_ylabel('Value')
ax7.set_title('Geometric Margins vs Slack Variables')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(point_labels)
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')

plt.close()

# Step 6: Create a simple, clean visualization without text and formulas
print("\n" + "="*50)
print("STEP 6: CREATING SIMPLE VISUALIZATION")
print("="*50)

fig4, ax8 = plt.subplots(1, 1, figsize=(10, 8))

# Plot decision boundary
ax8.plot(x1_range, x2_hyperplane, 'k-', linewidth=3)

# Plot margin boundaries
ax8.plot(x1_range, x2_margin_pos, 'k--', linewidth=2, alpha=0.7)
ax8.plot(x1_range, x2_margin_neg, 'k--', linewidth=2, alpha=0.7)

# Fill regions with subtle colors
ax8.fill_between(x1_range, x2_margin_pos, 3, alpha=0.1, color='red')
ax8.fill_between(x1_range, -1, x2_margin_neg, alpha=0.1, color='blue')
ax8.fill_between(x1_range, x2_margin_pos, x2_margin_neg, alpha=0.15, color='yellow')

# Plot points with different styles based on slack variables
for i in range(len(points)):
    x = points[i]
    y = labels[i]
    xi = slack_vars[i]
    
    # Color based on class
    color = 'red' if y == 1 else 'blue'
    
    # Marker size based on slack variable
    size = 100 + xi * 50
    
    # Edge style based on classification
    if xi == 0:
        edge_style = 'solid'
        edge_width = 2
    elif 0 < xi < 1:
        edge_style = 'dashed'
        edge_width = 2
    else:  # xi >= 1
        edge_style = 'dotted'
        edge_width = 3
    
    ax8.scatter(x[0], x[1], c=color, s=size, marker='o', 
               edgecolors='black', linewidth=edge_width, linestyle=edge_style, alpha=0.8)

# Clean styling
ax8.set_xlabel('$x_1$')
ax8.set_ylabel('$x_2$')
ax8.grid(True, alpha=0.3)
ax8.set_xlim(-0.5, 2.5)
ax8.set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_simple_visualization.png'), dpi=300, bbox_inches='tight')

plt.close()

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("All visualizations have been generated and saved.")
print("The code has provided step-by-step verification of all slack variables")
print("and detailed geometric interpretation of the SVM behavior.")
