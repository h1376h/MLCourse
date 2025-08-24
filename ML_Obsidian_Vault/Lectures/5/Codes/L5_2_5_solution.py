import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 5: HINGE LOSS ANALYSIS")
print("=" * 80)

# 1. Calculate hinge loss for given predictions
print("\n1. CALCULATING HINGE LOSS FOR GIVEN PREDICTIONS")
print("-" * 50)

def hinge_loss(y, f_x):
    """Calculate hinge loss: L_h(y, f(x)) = max(0, 1 - y * f(x))"""
    return max(0, 1 - y * f_x)

# Given predictions
predictions = [
    (1, 2.5, "y = +1, f(x) = 2.5"),
    (1, 0.8, "y = +1, f(x) = 0.8"),
    (1, -0.3, "y = +1, f(x) = -0.3"),
    (-1, -1.7, "y = -1, f(x) = -1.7"),
    (-1, 0.4, "y = -1, f(x) = 0.4")
]

print("Hinge Loss Calculations:")
print("L_h(y, f(x)) = max(0, 1 - y * f(x))")
print()

for y, f_x, description in predictions:
    y_fx = y * f_x
    loss = hinge_loss(y, f_x)
    print(f"{description}:")
    print(f"  y * f(x) = {y} * {f_x} = {y_fx}")
    print(f"  L_h = max(0, 1 - {y_fx}) = max(0, {1 - y_fx}) = {loss}")
    print()

# 2. Sketch hinge loss as function of y * f(x)
print("\n2. VISUALIZING HINGE LOSS FUNCTION")
print("-" * 50)

# Create data for plotting
y_fx_range = np.linspace(-3, 3, 1000)
hinge_loss_values = np.maximum(0, 1 - y_fx_range)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot hinge loss
plt.plot(y_fx_range, hinge_loss_values, 'b-', linewidth=3, label='Hinge Loss')

# Highlight specific points from our calculations
for y, f_x, description in predictions:
    y_fx = y * f_x
    loss = hinge_loss(y, f_x)
    plt.scatter(y_fx, loss, s=100, color='red', zorder=5)
    plt.annotate(f'({y_fx:.1f}, {loss:.1f})', 
                (y_fx, loss), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add reference lines
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Margin boundary')

# Add regions
plt.fill_between(y_fx_range, 0, hinge_loss_values, alpha=0.2, color='blue', label='Loss region')
plt.fill_between(y_fx_range, 0, 0, where=(y_fx_range >= 1), alpha=0.2, color='green', label='Zero loss region')

plt.xlabel('$y \\cdot f(x)$', fontsize=14)
plt.ylabel('$L_h(y, f(x))$', fontsize=14)
plt.title('Hinge Loss Function: $L_h(y, f(x)) = \\max(0, 1 - y \\cdot f(x))$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-3, 3)
plt.ylim(-0.5, 4.5)

# Add text annotations
plt.text(0.05, 0.95, 'Perfect classification\n(no loss)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.8))

plt.text(0.05, 0.7, 'Classification with margin\n(small loss)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", alpha=0.8))

plt.text(0.05, 0.45, 'Misclassification\n(high loss)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="red", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hinge_loss_function.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Show that ξ_i = L_h(y_i, f(x_i)) in soft margin formulation
print("\n3. SOFT MARGIN FORMULATION AND SLACK VARIABLES")
print("-" * 50)

print("In the soft margin SVM formulation:")
print("minimize: (1/2)||w||² + C * Σᵢ ξᵢ")
print("subject to: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ")
print("            ξᵢ ≥ 0 for all i")
print()
print("The slack variable ξᵢ represents the amount by which the constraint is violated.")
print("At optimality, ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b)) = L_h(yᵢ, f(xᵢ))")
print()

# Demonstrate with our examples
print("Demonstration with our examples:")
for y, f_x, description in predictions:
    slack = hinge_loss(y, f_x)
    print(f"{description}:")
    print(f"  ξ = max(0, 1 - {y} * {f_x}) = {slack}")
    print(f"  This equals the hinge loss: L_h({y}, {f_x}) = {slack}")
    print()

# 4. Compare derivative properties of hinge loss vs squared loss
print("\n4. DERIVATIVE PROPERTIES: HINGE LOSS vs SQUARED LOSS")
print("-" * 50)

def squared_loss(y, f_x):
    """Calculate squared loss: L_s(y, f(x)) = (1 - y * f(x))²"""
    return (1 - y * f_x) ** 2

def hinge_loss_derivative(y, f_x):
    """Derivative of hinge loss with respect to f(x)"""
    if 1 - y * f_x > 0:
        return -y
    else:
        return 0

def squared_loss_derivative(y, f_x):
    """Derivative of squared loss with respect to f(x)"""
    return -2 * y * (1 - y * f_x)

# Create comparison plot
y_fx_range = np.linspace(-2, 2, 1000)
hinge_derivatives = [hinge_loss_derivative(1, x) for x in y_fx_range]
squared_derivatives = [squared_loss_derivative(1, x) for x in y_fx_range]

plt.figure(figsize=(15, 6))

# Plot 1: Loss functions
plt.subplot(1, 2, 1)
hinge_loss_values = np.maximum(0, 1 - y_fx_range)
squared_loss_values = (1 - y_fx_range) ** 2

plt.plot(y_fx_range, hinge_loss_values, 'b-', linewidth=3, label='Hinge Loss')
plt.plot(y_fx_range, squared_loss_values, 'r-', linewidth=3, label='Squared Loss')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='green', linestyle='--', alpha=0.7)

plt.xlabel('$y \\cdot f(x)$', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Functions Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)
plt.ylim(-0.5, 4)

# Plot 2: Derivatives
plt.subplot(1, 2, 2)
plt.plot(y_fx_range, hinge_derivatives, 'b-', linewidth=3, label='Hinge Loss Derivative')
plt.plot(y_fx_range, squared_derivatives, 'r-', linewidth=3, label='Squared Loss Derivative')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='green', linestyle='--', alpha=0.7)

plt.xlabel('$y \\cdot f(x)$', fontsize=12)
plt.ylabel('Derivative', fontsize=12)
plt.title('Loss Function Derivatives', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)
plt.ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_functions_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Key differences in derivative properties:")
print("1. Hinge Loss Derivative:")
print("   - Constant gradient (-y) when y·f(x) < 1")
print("   - Zero gradient when y·f(x) ≥ 1")
print("   - Non-differentiable at y·f(x) = 1")
print()
print("2. Squared Loss Derivative:")
print("   - Linear gradient: -2y(1 - y·f(x))")
print("   - Always differentiable")
print("   - Gradient approaches zero as y·f(x) approaches 1")
print()

# 5. Prove that hinge loss upper bounds the 0-1 loss
print("\n5. HINGE LOSS UPPER BOUNDS 0-1 LOSS")
print("-" * 50)

def zero_one_loss(y, f_x):
    """Calculate 0-1 loss: L_01(y, f(x)) = 1 if y * f(x) ≤ 0, 0 otherwise"""
    return 1 if y * f_x <= 0 else 0

# Create comparison plot
y_fx_range = np.linspace(-2, 2, 1000)
hinge_loss_values = np.maximum(0, 1 - y_fx_range)
zero_one_loss_values = np.where(y_fx_range <= 0, 1, 0)

plt.figure(figsize=(12, 8))

plt.plot(y_fx_range, hinge_loss_values, 'b-', linewidth=3, label='Hinge Loss')
plt.plot(y_fx_range, zero_one_loss_values, 'r-', linewidth=3, label='0-1 Loss')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='green', linestyle='--', alpha=0.7)

plt.xlabel('$y \\cdot f(x)$', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Hinge Loss Upper Bounds 0-1 Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)
plt.ylim(-0.2, 3)

# Add proof annotation
plt.text(0.05, 0.95, 'Proof: L_h(y, f(x)) >= L_01(y, f(x))', transform=plt.gca().transAxes, 
         fontsize=14, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))

plt.text(0.05, 0.85, 'Case 1: y*f(x) <= 0', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')
plt.text(0.05, 0.8, 'L_01 = 1, L_h >= 1 (OK)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')

plt.text(0.05, 0.7, 'Case 2: 0 < y*f(x) < 1', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')
plt.text(0.05, 0.65, 'L_01 = 0, L_h > 0 (OK)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')

plt.text(0.05, 0.55, 'Case 3: y*f(x) >= 1', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')
plt.text(0.05, 0.5, 'L_01 = 0, L_h = 0 (OK)', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hinge_loss_upper_bound.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Proof that hinge loss upper bounds 0-1 loss:")
print("We need to show: L_h(y, f(x)) ≥ L_01(y, f(x)) for all y and f(x)")
print()
print("Case 1: y·f(x) ≤ 0 (misclassification)")
print("  L_01(y, f(x)) = 1")
print("  L_h(y, f(x)) = max(0, 1 - y·f(x)) ≥ 1 (since y·f(x) ≤ 0)")
print("  Therefore, L_h ≥ L_01 ✓")
print()
print("Case 2: 0 < y·f(x) < 1 (correct but within margin)")
print("  L_01(y, f(x)) = 0")
print("  L_h(y, f(x)) = 1 - y·f(x) > 0")
print("  Therefore, L_h > L_01 ✓")
print()
print("Case 3: y·f(x) ≥ 1 (correct classification with margin)")
print("  L_01(y, f(x)) = 0")
print("  L_h(y, f(x)) = max(0, 1 - y·f(x)) = 0")
print("  Therefore, L_h = L_01 ✓")
print()
print("In all cases, L_h(y, f(x)) ≥ L_01(y, f(x)), so hinge loss upper bounds 0-1 loss.")

# 6. Quality control system design
print("\n6. QUALITY CONTROL SYSTEM DESIGN")
print("-" * 50)

# Given data
accept_scores = [2.5, 0.8, -0.3]  # Good products
reject_scores = [-1.7, 0.4]       # Bad products

print("Given data:")
print(f"Accept products (good): scores = {accept_scores}")
print(f"Reject products (bad): scores = {reject_scores}")
print()

# Calculate hinge losses for each product
print("Hinge Loss Analysis:")
print("For accept products (y = +1):")
for score in accept_scores:
    loss = hinge_loss(1, score)
    print(f"  Score {score}: L_h = max(0, 1 - {score}) = {loss}")

print("\nFor reject products (y = -1):")
for score in reject_scores:
    loss = hinge_loss(-1, score)
    print(f"  Score {score}: L_h = max(0, 1 - (-1)*{score}) = {loss}")

# Design confidence scoring system (0-10 scale)
def score_to_confidence(f_x):
    """Convert f(x) to confidence score (0-10)"""
    # Map f(x) to confidence: f(x) = -3 → 0, f(x) = 3 → 10
    confidence = (f_x + 3) / 6 * 10
    return np.clip(confidence, 0, 10)

print("\nConfidence Scoring System (0-10 scale):")
print("Formula: confidence = (f(x) + 3) / 6 * 10")
print("This maps f(x) from [-3, 3] to [0, 10]")

for score in accept_scores + reject_scores:
    conf = score_to_confidence(score)
    print(f"  Score {score}: confidence = {conf:.1f}/10")

# Design tolerance zone
print("\nTolerance Zone Design:")
print("Products with confidence scores in [3.5, 6.5] need additional inspection")
print("This corresponds to f(x) values in [-1.5, 0.5]")

# Cost analysis
print("\nCost Analysis:")
print("Cost of rejecting good product = 3 × Cost of accepting bad product")
print("Let C = cost of accepting bad product")
print("Then cost of rejecting good product = 3C")

# Calculate expected costs for different thresholds
def calculate_costs(threshold):
    """Calculate expected costs for a given threshold"""
    false_positives = sum(1 for score in reject_scores if score > threshold)
    false_negatives = sum(1 for score in accept_scores if score <= threshold)
    
    cost_fp = false_positives * 1  # Cost of accepting bad product
    cost_fn = false_negatives * 3  # Cost of rejecting good product (3x higher)
    
    return cost_fp + cost_fn, false_positives, false_negatives

# Test different thresholds
thresholds = np.linspace(-2, 1, 31)
costs = []
fp_counts = []
fn_counts = []

for threshold in thresholds:
    total_cost, fp, fn = calculate_costs(threshold)
    costs.append(total_cost)
    fp_counts.append(fp)
    fn_counts.append(fn)

# Find optimal threshold
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

print(f"\nOptimal threshold analysis:")
print(f"Optimal threshold: f(x) = {optimal_threshold:.2f}")
print(f"Optimal confidence threshold: {score_to_confidence(optimal_threshold):.1f}/10")
print(f"Expected cost: {optimal_cost:.1f}C")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Score distribution
plt.subplot(2, 2, 1)
plt.hist(accept_scores, bins=10, alpha=0.7, label='Accept (Good)', color='green')
plt.hist(reject_scores, bins=10, alpha=0.7, label='Reject (Bad)', color='red')
plt.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2, label=f'Optimal threshold: {optimal_threshold:.2f}')
plt.xlabel('Score f(x)')
plt.ylabel('Frequency')
plt.title('Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cost analysis
plt.subplot(2, 2, 2)
plt.plot(thresholds, costs, 'b-', linewidth=2)
plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Expected Cost (in units of C)')
plt.title('Cost vs Threshold')
plt.grid(True, alpha=0.3)

# Plot 3: Error rates
plt.subplot(2, 2, 3)
plt.plot(thresholds, fp_counts, 'r-', linewidth=2, label='False Positives')
plt.plot(thresholds, fn_counts, 'g-', linewidth=2, label='False Negatives')
plt.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.title('Error Counts vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Confidence scoring
plt.subplot(2, 2, 4)
all_scores = accept_scores + reject_scores
confidences = [score_to_confidence(score) for score in all_scores]
colors = ['green'] * len(accept_scores) + ['red'] * len(reject_scores)

plt.scatter(all_scores, confidences, c=colors, s=100, alpha=0.7)
plt.axhline(y=score_to_confidence(optimal_threshold), color='blue', linestyle='--', linewidth=2, 
           label=f'Optimal confidence: {score_to_confidence(optimal_threshold):.1f}')
plt.axhspan(3.5, 6.5, alpha=0.2, color='yellow', label='Tolerance zone')
plt.xlabel('Score f(x)')
plt.ylabel('Confidence (0-10)')
plt.title('Confidence Scoring System')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'quality_control_system.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nQuality Control System Summary:")
print(f"1. Confidence scoring: (f(x) + 3) / 6 * 10")
print(f"2. Optimal threshold: {optimal_threshold:.2f} (confidence: {score_to_confidence(optimal_threshold):.1f}/10)")
print(f"3. Tolerance zone: confidence scores [3.5, 6.5] for additional inspection")
print(f"4. Expected cost: {optimal_cost:.1f}C")
print(f"5. False positives: {fp_counts[optimal_idx]}, False negatives: {fn_counts[optimal_idx]}")

# 7. Simple margin visualization
print("\n7. SIMPLE MARGIN VISUALIZATION")
print("-" * 50)

# Create a simple 2D visualization showing margin concept
np.random.seed(42)
n_samples = 100

# Generate two classes with some overlap
class1_x = np.random.normal(2, 1, n_samples//2)
class1_y = np.random.normal(2, 1, n_samples//2)
class2_x = np.random.normal(-2, 1, n_samples//2)
class2_y = np.random.normal(-2, 1, n_samples//2)

# Decision boundary (simple linear separator)
x_boundary = np.linspace(-4, 4, 100)
y_boundary = -x_boundary  # Simple diagonal line

# Margin boundaries
margin = 1.5
y_margin_upper = -x_boundary + margin
y_margin_lower = -x_boundary - margin

plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(class1_x, class1_y, c='blue', s=50, alpha=0.7, label='Class +1')
plt.scatter(class2_x, class2_y, c='red', s=50, alpha=0.7, label='Class -1')

# Plot decision boundary and margins
plt.plot(x_boundary, y_boundary, 'k-', linewidth=3, label='Decision Boundary')
plt.plot(x_boundary, y_margin_upper, 'g--', linewidth=2, alpha=0.7, label='Margin')
plt.plot(x_boundary, y_margin_lower, 'g--', linewidth=2, alpha=0.7)

# Shade margin region
plt.fill_between(x_boundary, y_margin_lower, y_margin_upper, alpha=0.2, color='green')

# Add some support vectors (points near the margin)
support_vectors_x = [-1, 1, 0, -0.5, 0.5]
support_vectors_y = [1, -1, 0, -0.5, 0.5]
plt.scatter(support_vectors_x, support_vectors_y, c='yellow', s=200, 
           edgecolors='black', linewidth=2, marker='o', label='Support Vectors')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('SVM Margin Concept', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_margin_concept.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Simple margin visualization created!")
print("This shows the basic concept of SVM margin maximization.")

print(f"\nPlots saved to: {save_dir}")
print("\n" + "=" * 80)
print("SOLUTION COMPLETE")
print("=" * 80)
