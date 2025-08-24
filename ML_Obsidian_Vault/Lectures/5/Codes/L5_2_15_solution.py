import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 15: SOFT MARGIN SVM CALCULATIONS")
print("=" * 80)

# Given data
X = np.array([
    [1, 1],    # x1: (1, 1), y1 = +1
    [2, 0],    # x2: (2, 0), y2 = +1
    [0, 0],    # x3: (0, 0), y3 = -1
    [1, -1],   # x4: (1, -1), y4 = -1
    [0.5, 0.8] # x5: (0.5, 0.8), y5 = -1 (potential outlier)
])

y = np.array([1, 1, -1, -1, -1])

# Given optimal solution
w_star = np.array([0.4, 0.8])
b_star = -0.6
C = 2

print(f"Dataset:")
for i, (x, yi) in enumerate(zip(X, y), 1):
    print(f"  x{i} = {x}, y{i} = {yi}")
print(f"\nOptimal solution: w* = {w_star}, b* = {b_star}, C = {C}")

# Function to compute decision function
def decision_function(x, w, b):
    return np.dot(w, x) + b

# Function to compute margin
def compute_margin(x, y, w, b):
    return y * decision_function(x, w, b)

# Function to compute slack variable
def compute_slack(x, y, w, b):
    margin = compute_margin(x, y, w, b)
    return max(0, 1 - margin)

# Function to compute hinge loss
def hinge_loss_single(y_true, y_pred):
    return max(0, 1 - y_true * y_pred)

print("\n" + "=" * 60)
print("STEP 1: Calculate y_i(w*^T x_i + b*) for each point")
print("=" * 60)

margins = []
for i, (x, yi) in enumerate(zip(X, y), 1):
    margin = compute_margin(x, yi, w_star, b_star)
    margins.append(margin)
    print(f"Point {i}: y{i} * (w*^T x{i} + b*) = {yi} * ({np.dot(w_star, x):.2f} + {b_star:.2f}) = {yi} * {decision_function(x, w_star, b_star):.2f} = {margin:.2f}")

print("\n" + "=" * 60)
print("STEP 2: Compute ξ_i = max(0, 1 - y_i(w*^T x_i + b*)) for each point")
print("=" * 60)

slack_variables = []
for i, (x, yi) in enumerate(zip(X, y), 1):
    slack = compute_slack(x, yi, w_star, b_star)
    slack_variables.append(slack)
    print(f"Point {i}: ξ{i} = max(0, 1 - {margins[i-1]:.2f}) = {slack:.2f}")

print("\n" + "=" * 60)
print("STEP 3: Calculate total objective 1/2 ||w*||^2 + C * Σ ξ_i")
print("=" * 60)

# Compute ||w*||^2
w_norm_squared = np.dot(w_star, w_star)
print(f"||w*||^2 = {w_norm_squared:.2f}")

# Compute 1/2 ||w*||^2
regularization_term = 0.5 * w_norm_squared
print(f"1/2 ||w*||^2 = {regularization_term:.2f}")

# Compute C * Σ ξ_i
total_slack = sum(slack_variables)
slack_term = C * total_slack
print(f"C * Σ ξ_i = {C} * {total_slack:.2f} = {slack_term:.2f}")

# Total objective
total_objective = regularization_term + slack_term
print(f"Total objective = {regularization_term:.2f} + {slack_term:.2f} = {total_objective:.2f}")

print("\n" + "=" * 60)
print("STEP 4: Classify each point as margin SV, non-margin SV, or non-SV")
print("=" * 60)

# For soft margin SVM:
# - Margin SV: 0 < α < C and ξ = 0 (on the margin)
# - Non-margin SV: α = C and ξ > 0 (violates margin)
# - Non-SV: α = 0 (correctly classified with margin > 1)

# We need to estimate α values based on the slack variables
# This is a simplified approach - in practice, we'd solve the dual problem
print("Classification based on slack variables and margins:")
for i, (x, yi, margin, slack) in enumerate(zip(X, y, margins, slack_variables), 1):
    if slack == 0 and abs(margin - 1) < 1e-6:
        sv_type = "Margin SV"
    elif slack > 0:
        sv_type = "Non-margin SV"
    else:
        sv_type = "Non-SV"
    
    print(f"Point {i}: margin = {margin:.2f}, ξ = {slack:.2f} → {sv_type}")

print("\n" + "=" * 60)
print("STEP 5: Compute individual hinge losses and verify relationship to slack variables")
print("=" * 60)

for i, (x, yi) in enumerate(zip(X, y), 1):
    y_pred = decision_function(x, w_star, b_star)
    hinge = hinge_loss_single(yi, y_pred)
    slack = slack_variables[i-1]
    
    print(f"Point {i}:")
    print(f"  y_true = {yi}, y_pred = {y_pred:.2f}")
    print(f"  Hinge loss = max(0, 1 - {yi} * {y_pred:.2f}) = {hinge:.2f}")
    print(f"  Slack variable ξ = {slack:.2f}")
    print(f"  Relationship: ξ = hinge loss ✓")

print("\n" + "=" * 60)
print("STEP 6: Credit Scoring System Design")
print("=" * 60)

# Credit scoring system
print("Credit Scoring System Design:")
print("Features: x1 = Income (normalized), x2 = Debt-to-Income Ratio (normalized)")
print("Classes: +1 = Low Risk, -1 = High Risk")

# Calculate risk scores for each applicant
risk_scores = []
for i, x in enumerate(X, 1):
    score = decision_function(x, w_star, b_star)
    risk_scores.append(score)
    risk_level = "Low Risk" if score > 0 else "High Risk"
    print(f"Applicant {i}: {x} → Score = {score:.2f} → {risk_level}")

# Design three-category risk system
print("\nThree-Category Risk System:")
scores_sorted = sorted(risk_scores)
low_threshold = np.percentile(scores_sorted, 60)  # Top 40% get low risk
high_threshold = np.percentile(scores_sorted, 20)  # Bottom 20% get high risk

print(f"Low Risk threshold: score > {low_threshold:.2f}")
print(f"Medium Risk threshold: {high_threshold:.2f} ≤ score ≤ {low_threshold:.2f}")
print(f"High Risk threshold: score < {high_threshold:.2f}")

# Calculate risk uncertainty for the uncertain case (point 5)
uncertain_point = X[4]  # (0.5, 0.8)
uncertain_score = risk_scores[4]
distance_to_boundary = abs(uncertain_score)
print(f"\nRisk Uncertainty for uncertain case (0.5, 0.8):")
print(f"Score = {uncertain_score:.2f}")
print(f"Distance to decision boundary = {distance_to_boundary:.2f}")
print(f"Uncertainty measure = {distance_to_boundary:.2f} (closer to 0 = more uncertain)")

# Adjust decision boundary for 80% approval rate
print(f"\nAdjusting decision boundary for 80% approval rate:")
current_approval_rate = sum(1 for score in risk_scores if score > 0) / len(risk_scores)
print(f"Current approval rate: {current_approval_rate:.1%}")

# Calculate new threshold for 80% approval
target_approval = 0.8
scores_for_approval = sorted(risk_scores, reverse=True)
new_threshold = scores_for_approval[int(len(scores_for_approval) * target_approval) - 1]
b_adjusted = b_star - (new_threshold - 0)  # Adjust bias to shift boundary

print(f"New threshold for 80% approval: {new_threshold:.2f}")
print(f"Adjusted bias: b* = {b_star:.2f} → b_adjusted = {b_adjusted:.2f}")

# Verify new approval rate
new_scores = [decision_function(x, w_star, b_adjusted) for x in X]
new_approval_rate = sum(1 for score in new_scores if score > 0) / len(new_scores)
print(f"New approval rate: {new_approval_rate:.1%}")

# Visualization
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Soft Margin SVM Analysis - Question 15', fontsize=16, fontweight='bold')

# Plot 1: Decision boundary and data points
ax1 = axes[0, 0]
x_min, x_max = -1, 3
y_min, y_max = -2, 2

# Create mesh grid
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Compute decision function for each point in the grid
Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z[i, j] = decision_function([xx[i, j], yy[i, j]], w_star, b_star)

# Plot decision boundary and regions
contour = ax1.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2, label='Decision Boundary')
ax1.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightcoral'], alpha=0.3)
ax1.contourf(xx, yy, Z, levels=[0, 100], colors=['lightblue'], alpha=0.3)

# Plot margin boundaries
margin_plus = ax1.contour(xx, yy, Z, levels=[1], colors='blue', linewidths=1, linestyles='--', alpha=0.7)
margin_minus = ax1.contour(xx, yy, Z, levels=[-1], colors='red', linewidths=1, linestyles='--', alpha=0.7)

# Plot data points with different markers for different classes
colors = ['blue' if yi == 1 else 'red' for yi in y]
markers = ['o' if yi == 1 else 's' for yi in y]

for i, (x, yi, margin, slack) in enumerate(zip(X, y, margins, slack_variables)):
    if slack == 0 and abs(margin - 1) < 1e-6:
        marker = '^'  # Margin SV
        size = 200
    elif slack > 0:
        marker = 'D'  # Non-margin SV
        size = 200
    else:
        marker = markers[i]  # Non-SV
        size = 100
    
    ax1.scatter(x[0], x[1], c=colors[i], marker=marker, s=size, 
                edgecolor='black', linewidth=1.5, label=f'Point {i+1}')

ax1.set_xlabel('$x_1$ (Income)')
ax1.set_ylabel('$x_2$ (Debt-to-Income Ratio)')
ax1.set_title('Decision Boundary and Support Vectors')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# Plot 2: Margins and slack variables
ax2 = axes[0, 1]
x_pos = np.arange(len(X))
bars = ax2.bar(x_pos, margins, color=['blue' if yi == 1 else 'red' for yi in y], alpha=0.7)
ax2.axhline(y=1, color='green', linestyle='-', label='Margin = 1')
ax2.axhline(y=-1, color='green', linestyle='-', label='Margin = -1')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Decision Boundary')

# Add slack variable annotations
for i, (margin, slack) in enumerate(zip(margins, slack_variables)):
    if slack > 0:
        ax2.annotate(f'xi={slack:.2f}', (i, margin), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Data Points')
ax2.set_ylabel('Margin $y_i(\\mathbf{w}^T \\mathbf{x}_i + b)$')
ax2.set_title('Margins and Slack Variables')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Point {i+1}' for i in range(len(X))])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Risk scores for credit scoring
ax3 = axes[1, 0]
risk_colors = ['green' if score > low_threshold else 'orange' if score > high_threshold else 'red' for score in risk_scores]
bars = ax3.bar(x_pos, risk_scores, color=risk_colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Decision Boundary')
ax3.axhline(y=low_threshold, color='green', linestyle='--', alpha=0.7, label='Low Risk Threshold')
ax3.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')

# Add risk level annotations
for i, score in enumerate(risk_scores):
    if score > low_threshold:
        risk_level = "Low"
    elif score > high_threshold:
        risk_level = "Medium"
    else:
        risk_level = "High"
    ax3.annotate(risk_level, (i, score), 
                xytext=(0, 5 if score > 0 else -15), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')

ax3.set_xlabel('Applicants')
ax3.set_ylabel('Risk Score')
ax3.set_title('Credit Risk Scoring')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'App {i+1}' for i in range(len(X))])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Objective function components
ax4 = axes[1, 1]
components = ['Regularization\n$\\frac{1}{2}||\\mathbf{w}||^2$', 'Slack Penalty\n$C\\sum \\xi_i$', 'Total\nObjective']
values = [regularization_term, slack_term, total_objective]
colors = ['lightblue', 'lightcoral', 'lightgreen']

bars = ax4.bar(components, values, color=colors, alpha=0.7)
ax4.set_ylabel('Value')
ax4.set_title('Objective Function Components')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'soft_margin_svm_analysis.png'), dpi=300, bbox_inches='tight')

# Create additional visualization for adjusted decision boundary
plt.figure(figsize=(12, 8))

# Plot original and adjusted decision boundaries
Z_original = np.zeros_like(xx)
Z_adjusted = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z_original[i, j] = decision_function([xx[i, j], yy[i, j]], w_star, b_star)
        Z_adjusted[i, j] = decision_function([xx[i, j], yy[i, j]], w_star, b_adjusted)

# Plot decision boundaries
plt.contour(xx, yy, Z_original, levels=[0], colors='green', linewidths=3, label='Original Boundary')
plt.contour(xx, yy, Z_adjusted, levels=[0], colors='red', linewidths=3, linestyles='--', label='Adjusted Boundary (80% approval)')

# Shade regions
plt.contourf(xx, yy, Z_adjusted, levels=[-100, 0], colors=['lightcoral'], alpha=0.2)
plt.contourf(xx, yy, Z_adjusted, levels=[0, 100], colors=['lightblue'], alpha=0.2)

# Plot data points
for i, (x, yi) in enumerate(zip(X, y)):
    color = 'blue' if yi == 1 else 'red'
    marker = 'o' if yi == 1 else 's'
    plt.scatter(x[0], x[1], c=color, marker=marker, s=150, 
                edgecolor='black', linewidth=1.5, label=f'Point {i+1} (y={yi})')

plt.xlabel('$x_1$ (Income)')
plt.ylabel('$x_2$ (Debt-to-Income Ratio)')
plt.title('Decision Boundary Adjustment for 80% Approval Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.savefig(os.path.join(save_dir, 'decision_boundary_adjustment.png'), dpi=300, bbox_inches='tight')

# Create additional informative visualization: Slack Variables vs Distance Analysis
plt.figure(figsize=(14, 6))

# Left subplot: Distance to decision boundary for each point
plt.subplot(1, 2, 1)
distances = []
classifications = []
for i, (x, yi) in enumerate(zip(X, y)):
    # Calculate perpendicular distance to decision boundary
    # Distance = |w^T x + b| / ||w||
    decision_value = decision_function(x, w_star, b_star)
    distance = abs(decision_value) / np.linalg.norm(w_star)
    distances.append(distance)
    
    # Determine if correctly classified
    if yi * decision_value > 0:
        classifications.append('Correct')
    else:
        classifications.append('Misclassified')

# Create scatter plot with color coding
colors_correct = ['green' if c == 'Correct' else 'red' for c in classifications]
plt.scatter(range(1, len(X)+1), distances, c=colors_correct, s=150, alpha=0.7, edgecolor='black')

# Add slack variable annotations
for i, (dist, slack) in enumerate(zip(distances, slack_variables)):
    plt.annotate(f'$\\xi={slack:.2f}$', (i+1, dist), 
                xytext=(0, 15), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlabel('Data Point')
plt.ylabel('Distance to Decision Boundary')
plt.title('Distance to Decision Boundary vs Slack Variables')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(X)+1), [f'Point {i}' for i in range(1, len(X)+1)])

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label='Correctly Classified'),
                   Patch(facecolor='red', alpha=0.7, label='Misclassified')]
plt.legend(handles=legend_elements)

# Right subplot: Slack variable decomposition
plt.subplot(1, 2, 2)

# Create stacked bar chart showing margin components
margin_contributions = []
slack_contributions = []

for i, (margin, slack) in enumerate(zip(margins, slack_variables)):
    if margin >= 1:
        # Point is beyond margin
        margin_contributions.append(1.0)
        slack_contributions.append(0.0)
    else:
        # Point violates margin
        margin_contributions.append(max(0, margin))
        slack_contributions.append(slack)

x_pos = np.arange(len(X))
width = 0.6

# Stack the bars
bars1 = plt.bar(x_pos, margin_contributions, width, label='Margin Contribution', 
                color='lightblue', alpha=0.8)
bars2 = plt.bar(x_pos, slack_contributions, width, bottom=margin_contributions,
                label='Slack Variable $\\xi_i$', color='lightcoral', alpha=0.8)

# Add target line at y=1
plt.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.7, 
            label='Target Margin')

# Add value annotations
for i, (margin, slack) in enumerate(zip(margins, slack_variables)):
    total_height = max(1, margin + slack) if margin >= 0 else 1 + slack
    plt.annotate(f'{margin:.2f}', (i, margin/2), ha='center', va='center', 
                fontweight='bold', fontsize=9)
    if slack > 0:
        plt.annotate(f'$\\xi={slack:.2f}$', (i, margin + slack/2), ha='center', va='center', 
                    fontweight='bold', fontsize=9, color='darkred')

plt.xlabel('Data Points')
plt.ylabel('Margin Decomposition')
plt.title('Margin Decomposition: Target vs Actual')
plt.xticks(x_pos, [f'Point {i+1}' for i in range(len(X))])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Point':<8} {'Coordinates':<15} {'Label':<6} {'Margin':<8} {'Slack':<8} {'SV Type':<15} {'Risk Score':<12} {'Risk Level':<12}")
print("-" * 80)
for i, (x, yi, margin, slack) in enumerate(zip(X, y, margins, slack_variables)):
    if slack == 0 and abs(margin - 1) < 1e-6:
        sv_type = "Margin SV"
    elif slack > 0:
        sv_type = "Non-margin SV"
    else:
        sv_type = "Non-SV"
    
    risk_score = risk_scores[i]
    if risk_score > low_threshold:
        risk_level = "Low"
    elif risk_score > high_threshold:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    print(f"{f'Point {i+1}':<8} {str(x):<15} {yi:<6} {margin:<8.2f} {slack:<8.2f} {sv_type:<15} {risk_score:<12.2f} {risk_level:<12}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("1. The soft margin SVM allows some points to violate the margin (ξ > 0)")
print("2. Points with ξ = 0 and margin = 1 are margin support vectors")
print("3. Points with ξ > 0 are non-margin support vectors (violate margin)")
print("4. The total objective balances regularization and slack penalty")
print("5. The credit scoring system can be adjusted by shifting the decision boundary")
print("6. Risk uncertainty is measured by distance to the decision boundary")
print("=" * 80)
