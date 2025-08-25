import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_37")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Configure LaTeX to handle mathematical notation
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 37: MEDICAL DIAGNOSIS WITH RBF KERNEL SVM")
print("=" * 80)

# Given parameters
support_vectors = np.array([
    [2.5, 1.8],  # SV1: x^(1), y^(1) = +1, α1 = 0.8
    [1.2, 3.1],  # SV2: x^(2), y^(2) = +1, α2 = 0.6
    [4.1, 0.9],  # SV3: x^(3), y^(3) = -1, α3 = 0.4
    [0.8, 2.5]   # SV4: x^(4), y^(4) = -1, α4 = 0.7
])

labels = np.array([1, 1, -1, -1])  # y values
alphas = np.array([0.8, 0.6, 0.4, 0.7])  # α values
sigma = 1.5  # RBF kernel parameter

# New patient to classify
new_patient = np.array([2.0, 2.2])

print(f"Support Vectors:")
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    print(f"  SV{i+1}: x^({i+1}) = {sv}, y^({i+1}) = {label}, α{i+1} = {alpha}")

print(f"\nRBF Kernel Parameter: σ = {sigma}")
print(f"New Patient: x = {new_patient}")

# RBF kernel function
def rbf_kernel(x1, x2, sigma):
    """Compute RBF kernel between two points"""
    squared_distance = np.sum((x1 - x2)**2)
    return np.exp(-squared_distance / (2 * sigma**2))

# Function to compute kernel matrix
def compute_kernel_matrix(points, sigma):
    """Compute kernel matrix for given points"""
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(points[i], points[j], sigma)
    return K

print("\n" + "=" * 50)
print("STEP 1: CALCULATE THE BIAS TERM w₀")
print("=" * 50)

# Using support vector x^(s) = (2.5, 1.8) with y^(s) = +1
sv_index = 0  # First support vector
sv_s = support_vectors[sv_index]
y_s = labels[sv_index]

print(f"Using support vector x^({sv_index+1}) = {sv_s} with y^({sv_index+1}) = {y_s}")

# Calculate kernel values between sv_s and all support vectors
kernel_values_s = np.array([rbf_kernel(sv_s, sv, sigma) for sv in support_vectors])

print(f"\nKernel values k(x^({sv_index+1}), x^(n)) for all support vectors:")
for i, k_val in enumerate(kernel_values_s):
    print(f"  k(x^({sv_index+1}), x^({i+1})) = exp(-||{sv_s} - {support_vectors[i]}||²/(2×{sigma}²))")
    print(f"    = exp(-{np.sum((sv_s - support_vectors[i])**2):.4f}/(2×{sigma**2:.2f}))")
    print(f"    = exp(-{np.sum((sv_s - support_vectors[i])**2):.4f}/{2*sigma**2:.2f})")
    print(f"    = exp(-{np.sum((sv_s - support_vectors[i])**2)/(2*sigma**2):.4f})")
    print(f"    = {k_val:.6f}")

# Calculate bias term
bias_sum = 0
print(f"\nCalculating bias term:")
print(f"w₀ = y^({sv_index+1}) - Σ(αₙ > 0) αₙ y^(n) k(x^(n), x^({sv_index+1}))")
print(f"w₀ = {y_s} - ", end="")

for i, (alpha, label, k_val) in enumerate(zip(alphas, labels, kernel_values_s)):
    if alpha > 0:
        term = alpha * label * k_val
        bias_sum += term
        print(f"({alpha} × {label} × {k_val:.6f})", end="")
        if i < len(alphas) - 1 and any(alphas[i+1:] > 0):
            print(" - ", end="")

w0 = y_s - bias_sum
print(f"\nw₀ = {y_s} - {bias_sum:.6f}")
print(f"w₀ = {w0:.6f}")

print("\n" + "=" * 50)
print("STEP 2: CLASSIFY THE NEW PATIENT")
print("=" * 50)

# Calculate kernel values between new patient and all support vectors
kernel_values_new = np.array([rbf_kernel(new_patient, sv, sigma) for sv in support_vectors])

print(f"Kernel values k(x^(n), x_new) for new patient x = {new_patient}:")
for i, k_val in enumerate(kernel_values_new):
    print(f"  k(x^({i+1}), x_new) = exp(-||{support_vectors[i]} - {new_patient}||²/(2×{sigma}²))")
    print(f"    = exp(-{np.sum((support_vectors[i] - new_patient)**2):.4f}/(2×{sigma**2:.2f}))")
    print(f"    = {k_val:.6f}")

# Calculate decision function
decision_sum = 0
print(f"\nCalculating decision function:")
print(f"f(x) = w₀ + Σ(αₙ > 0) αₙ y^(n) k(x^(n), x)")
print(f"f(x) = {w0:.6f} + ", end="")

for i, (alpha, label, k_val) in enumerate(zip(alphas, labels, kernel_values_new)):
    if alpha > 0:
        term = alpha * label * k_val
        decision_sum += term
        print(f"({alpha} × {label} × {k_val:.6f})", end="")
        if i < len(alphas) - 1 and any(alphas[i+1:] > 0):
            print(" + ", end="")

decision_value = w0 + decision_sum
print(f"\nf(x) = {w0:.6f} + {decision_sum:.6f}")
print(f"f(x) = {decision_value:.6f}")

# Classification
prediction = np.sign(decision_value)
print(f"\nPrediction: ŷ = sign({decision_value:.6f}) = {prediction}")
print(f"Patient is classified as: {'DISEASE PRESENT (+1)' if prediction > 0 else 'DISEASE ABSENT (-1)'}")

print("\n" + "=" * 50)
print("STEP 3: CALCULATE CONFIDENCE SCORE")
print("=" * 50)

print(f"Confidence score = {decision_value:.6f}")
print(f"Absolute confidence = |{decision_value:.6f}| = {abs(decision_value):.6f}")

# Normalize confidence to [0, 1] range (approximate)
max_possible_confidence = sum(alphas) + abs(w0)  # Rough estimate
normalized_confidence = abs(decision_value) / max_possible_confidence
print(f"Normalized confidence ≈ {normalized_confidence:.4f}")

print("\n" + "=" * 50)
print("STEP 4: MOST INFLUENTIAL SUPPORT VECTOR")
print("=" * 50)

# Calculate contribution of each support vector
contributions = []
print("Contribution of each support vector to the decision:")
print("Contribution = αₙ × y^(n) × k(x^(n), x)")

for i, (alpha, label, k_val) in enumerate(zip(alphas, labels, kernel_values_new)):
    if alpha > 0:
        contribution = alpha * label * k_val
        contributions.append((i, contribution))
        print(f"  SV{i+1}: {alpha} × {label} × {k_val:.6f} = {contribution:.6f}")

# Find most influential
if contributions:
    most_influential_idx, max_contribution = max(contributions, key=lambda x: abs(x[1]))
    print(f"\nMost influential support vector: SV{most_influential_idx+1}")
    print(f"Contribution: {max_contribution:.6f}")
    print(f"Support vector: x^({most_influential_idx+1}) = {support_vectors[most_influential_idx]}")

print("\n" + "=" * 50)
print("STEP 5: DECISION THRESHOLD ANALYSIS")
print("=" * 50)

# Given parameters
prevalence = 0.15  # 15% disease prevalence
false_positive_cost = 500  # $500 for unnecessary tests
false_negative_cost = 50000  # $50,000 for delayed treatment

print(f"Disease prevalence: {prevalence:.1%}")
print(f"False positive cost: ${false_positive_cost:,}")
print(f"False negative cost: ${false_negative_cost:,}")

# Calculate optimal threshold
# P(y=1|x) = 1 / (1 + exp(-f(x)))
# Optimal threshold: log(C_FP * P(y=0) / (C_FN * P(y=1)))
p_y1 = prevalence
p_y0 = 1 - prevalence

optimal_threshold = np.log((false_positive_cost * p_y0) / (false_negative_cost * p_y1))
print(f"\nOptimal threshold calculation:")
print(f"Threshold = log((C_FP × P(y=0)) / (C_FN × P(y=1)))")
print(f"Threshold = log(({false_positive_cost} × {p_y0:.3f}) / ({false_negative_cost} × {p_y1:.3f}))")
print(f"Threshold = log({false_positive_cost * p_y0:.1f} / {false_negative_cost * p_y1:.1f})")
print(f"Threshold = log({(false_positive_cost * p_y0) / (false_negative_cost * p_y1):.3f})")
print(f"Threshold = {optimal_threshold:.6f}")

# Current decision (threshold = 0)
current_decision = decision_value > 0
optimal_decision = decision_value > optimal_threshold

print(f"\nCurrent decision (threshold = 0): {'DISEASE PRESENT' if current_decision else 'DISEASE ABSENT'}")
print(f"Optimal decision (threshold = {optimal_threshold:.6f}): {'DISEASE PRESENT' if optimal_decision else 'DISEASE ABSENT'}")

if current_decision != optimal_decision:
    print(f"\nRECOMMENDATION: Adjust decision threshold!")
    print(f"Current threshold leads to suboptimal decisions given the cost structure.")
else:
    print(f"\nRECOMMENDATION: Current threshold is appropriate.")
    print(f"Decision remains the same with optimal threshold.")

# Calculate expected costs
def expected_cost(threshold, decision_value, prevalence, fp_cost, fn_cost):
    p_y1 = prevalence
    p_y0 = 1 - prevalence
    
    if decision_value > threshold:  # Predict positive
        # Cost = P(y=0) * C_FP (false positive)
        return p_y0 * fp_cost
    else:  # Predict negative
        # Cost = P(y=1) * C_FN (false negative)
        return p_y1 * fn_cost

current_cost = expected_cost(0, decision_value, prevalence, false_positive_cost, false_negative_cost)
optimal_cost = expected_cost(optimal_threshold, decision_value, prevalence, false_positive_cost, false_negative_cost)

print(f"\nExpected costs:")
print(f"Current threshold (0): ${current_cost:.2f}")
print(f"Optimal threshold ({optimal_threshold:.6f}): ${optimal_cost:.2f}")
print(f"Cost difference: ${abs(current_cost - optimal_cost):.2f}")

# Create visualizations
print("\n" + "=" * 50)
print("CREATING VISUALIZATIONS")
print("=" * 50)

# 1. Support vectors and decision boundary visualization
plt.figure(figsize=(12, 10))

# Create grid for decision boundary
x_min, x_max = 0, 5
y_min, y_max = 0, 4
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Calculate decision values for grid points
grid_decisions = []
for point in grid_points:
    k_vals = [rbf_kernel(point, sv, sigma) for sv in support_vectors]
    decision_sum = sum(alpha * label * k_val for alpha, label, k_val in zip(alphas, labels, k_vals) if alpha > 0)
    grid_decisions.append(w0 + decision_sum)

grid_decisions = np.array(grid_decisions).reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, grid_decisions, levels=[0], colors='red', linewidths=2)
plt.plot([], [], 'r-', linewidth=2, label='Decision Boundary')
plt.contourf(xx, yy, grid_decisions, levels=[-10, 0, 10], colors=['lightblue', 'lightpink'], alpha=0.3)

# Plot support vectors
colors = ['green' if label == 1 else 'red' for label in labels]
markers = ['o' if label == 1 else 's' for label in labels]

for i, (sv, color, marker, alpha) in enumerate(zip(support_vectors, colors, markers, alphas)):
    plt.scatter(sv[0], sv[1], c=color, marker=marker, s=200, alpha=0.7, 
                edgecolors='black', linewidth=2, label=f'SV{i+1} (a={alpha})')

# Plot new patient
plt.scatter(new_patient[0], new_patient[1], c='purple', marker='*', s=300, 
            edgecolors='black', linewidth=2, label=f'New Patient (f(x)={decision_value:.3f})')

# Add RBF kernel influence circles
for i, sv in enumerate(support_vectors):
    circle = Circle(sv, sigma, fill=False, linestyle='--', alpha=0.5, color='gray')
    plt.gca().add_patch(circle)

plt.xlabel(r'Blood Marker $X_1$')
plt.ylabel(r'Blood Marker $X_2$')
plt.title(r'SVM with RBF Kernel - Medical Diagnosis')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'serif'})
plt.grid(True, alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_rbf_decision_boundary.png'), dpi=300, bbox_inches='tight')

# 2. Kernel values heatmap
plt.figure(figsize=(10, 8))

# Create kernel matrix
kernel_matrix = compute_kernel_matrix(support_vectors, sigma)

# Add new patient to the matrix
extended_points = np.vstack([support_vectors, new_patient])
extended_kernel_matrix = compute_kernel_matrix(extended_points, sigma)

# Create labels
labels_kernel = [f'SV{i+1}' for i in range(len(support_vectors))] + ['New Patient']

# Plot heatmap
sns.heatmap(extended_kernel_matrix, annot=True, fmt='.4f', cmap='viridis',
            xticklabels=labels_kernel, yticklabels=labels_kernel)
plt.title(r'RBF Kernel Matrix ($\sigma = 1.5$)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_kernel_matrix.png'), dpi=300, bbox_inches='tight')

# 3. Support vector contributions
plt.figure(figsize=(10, 6))

sv_labels = [f'SV{i+1}' for i in range(len(support_vectors))]
contributions_abs = [abs(contrib[1]) for contrib in contributions]
contributions_labels = [f'SV{contrib[0]+1}' for contrib in contributions]

plt.bar(contributions_labels, contributions_abs, color=['blue', 'green', 'orange', 'red'][:len(contributions)])
plt.xlabel('Support Vector')
plt.ylabel(r'Absolute Contribution $|\alpha_n \times y^{(n)} \times k(\mathbf{x}^{(n)}, \mathbf{x})|$')
plt.title(r'Support Vector Contributions to Decision')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(contributions_abs):
    plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_contributions.png'), dpi=300, bbox_inches='tight')

# 4. Decision threshold analysis
plt.figure(figsize=(12, 8))

# Create range of thresholds
threshold_range = np.linspace(-2, 2, 100)
costs = [expected_cost(t, decision_value, prevalence, false_positive_cost, false_negative_cost) 
         for t in threshold_range]

plt.plot(threshold_range, costs, 'b-', linewidth=2, label='Expected Cost')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Current Threshold (0)')
plt.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
            label=f'Optimal Threshold ({optimal_threshold:.3f})')
plt.axvline(x=decision_value, color='purple', linestyle=':', alpha=0.7, 
            label=f'Patient Decision Value ({decision_value:.3f})')

plt.xlabel(r'Decision Threshold')
plt.ylabel(r'Expected Cost (\$)')
plt.title(r'Expected Cost vs Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations (without LaTeX to avoid dollar sign issues)
plt.annotate(f'Current Cost: ${current_cost:.2f}', 
             xy=(0, current_cost), xytext=(0.5, current_cost + 1000),
             arrowprops=dict(arrowstyle='->', color='red'), color='red',
             usetex=False)

plt.annotate(f'Optimal Cost: ${optimal_cost:.2f}', 
             xy=(optimal_threshold, optimal_cost), xytext=(optimal_threshold + 0.5, optimal_cost + 1000),
             arrowprops=dict(arrowstyle='->', color='green'), color='green',
             usetex=False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_threshold_analysis.png'), dpi=300, bbox_inches='tight')

print(f"All visualizations saved to: {save_dir}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Bias term w₀ = {w0:.6f}")
print(f"2. New patient classification: {'DISEASE PRESENT' if prediction > 0 else 'DISEASE ABSENT'}")
print(f"3. Confidence score: {decision_value:.6f}")
print(f"4. Most influential support vector: SV{most_influential_idx+1}")
print(f"5. Recommendation: {'ADJUST THRESHOLD' if current_decision != optimal_decision else 'KEEP CURRENT THRESHOLD'}")
print(f"   Expected cost savings: ${abs(current_cost - optimal_cost):.2f}")
