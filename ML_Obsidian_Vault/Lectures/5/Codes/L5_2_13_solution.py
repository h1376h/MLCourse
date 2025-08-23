import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Add Unicode support for LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("Question 13: Geometric Interpretation of Soft Margin SVM")
print("=" * 60)

# Task 1: Draw margin boundaries for hyperplane x1 + 2x2 - 3 = 0
print("\nTask 1: Margin Boundaries for Hyperplane x1 + 2x2 - 3 = 0")
print("-" * 50)

# Define the hyperplane: w1*x1 + w2*x2 + b = 0
w = np.array([1, 2])  # weight vector
b = -3               # bias term

# Calculate margin width (1/||w||)
margin_width = 1 / np.linalg.norm(w)
print(f"Weight vector w = {w}")
print(f"Bias term b = {b}")
print(f"Margin width = 1/||w|| = 1/{np.linalg.norm(w):.3f} = {margin_width:.3f}")

# Create margin boundaries
def margin_boundary(x1, w, b, margin_offset):
    """Calculate x2 for margin boundary with given offset"""
    return (-w[0]*x1 - b + margin_offset) / w[1]

# Plot margin boundaries
plt.figure(figsize=(12, 10))

# Define x1 range
x1_range = np.linspace(-2, 8, 100)

# Plot decision boundary (hyperplane)
x2_decision = margin_boundary(x1_range, w, b, 0)
plt.plot(x1_range, x2_decision, 'k-', linewidth=3, label='Decision Boundary')

# Plot positive margin boundary
x2_pos_margin = margin_boundary(x1_range, w, b, 1)
plt.plot(x1_range, x2_pos_margin, 'g--', linewidth=2, label='Positive Margin')

# Plot negative margin boundary
x2_neg_margin = margin_boundary(x1_range, w, b, -1)
plt.plot(x1_range, x2_neg_margin, 'r--', linewidth=2, label='Negative Margin')

# Shade the margin region
plt.fill_between(x1_range, x2_neg_margin, x2_pos_margin, alpha=0.2, color='yellow', label='Margin Region')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Soft Margin SVM: Decision Boundary and Margins')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.xlim(-2, 8)
plt.ylim(-2, 6)

# Add equation annotations
plt.annotate(f'Decision: {w[0]}x₁ + {w[1]}x₂ + {b} = 0', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
plt.annotate(f'Margin width = {margin_width:.3f}', 
             xy=(0.05, 0.85), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1))

plt.savefig(os.path.join(save_dir, 'margin_boundaries.png'), dpi=300, bbox_inches='tight')
plt.show()

# Task 2: Points with different slack variable values
print("\nTask 2: Points with Different Slack Variable Values")
print("-" * 50)

# Define points with different slack values
points = [
    {'x': [2, 1], 'y': 1, 'xi': 0.0, 'label': 'xi = 0 (Correctly classified)'},
    {'x': [1.5, 0.8], 'y': 1, 'xi': 0.5, 'label': 'xi = 0.5 (Margin violation)'},
    {'x': [1, 0.5], 'y': 1, 'xi': 1.0, 'label': 'xi = 1.0 (On decision boundary)'},
    {'x': [0.5, 0.2], 'y': 1, 'xi': 1.5, 'label': 'xi = 1.5 (Misclassified)'}
]

plt.figure(figsize=(12, 10))

# Plot decision boundary and margins again
plt.plot(x1_range, x2_decision, 'k-', linewidth=3, label='Decision Boundary')
plt.plot(x1_range, x2_pos_margin, 'g--', linewidth=2, label='Positive Margin')
plt.plot(x1_range, x2_neg_margin, 'r--', linewidth=2, label='Negative Margin')
plt.fill_between(x1_range, x2_neg_margin, x2_pos_margin, alpha=0.2, color='yellow', label='Margin Region')

# Plot points with different slack values
colors = ['green', 'orange', 'red', 'purple']
markers = ['o', 's', '^', 'D']

for i, point in enumerate(points):
    x, y, xi, label = point['x'], point['y'], point['xi'], point['label']
    
    # Calculate actual slack value
    activation = w[0]*x[0] + w[1]*x[1] + b
    actual_xi = max(0, 1 - y * activation)
    
    print(f"Point {i+1}: {x}, y={y}, xi={xi}")
    print(f"  Activation = {activation:.3f}")
    print(f"  Actual xi = {actual_xi:.3f}")
    print(f"  Distance to margin = {abs(activation - y):.3f}")
    
    # Plot point
    plt.scatter(x[0], x[1], s=200, color=colors[i], marker=markers[i], 
                edgecolor='black', linewidth=2, label=f'{label}')
    
    # Add annotation
    plt.annotate(f'xi={xi}', (x[0], x[1]), xytext=(10, 10), 
                 textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Points with Different Slack Variable Values')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('equal')
plt.xlim(-2, 8)
plt.ylim(-2, 6)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_variables.png'), dpi=300, bbox_inches='tight')
plt.show()

# Task 3: Show how margin changes as C varies
print("\nTask 3: Margin Changes with Different C Values")
print("-" * 50)

# Create synthetic dataset
np.random.seed(42)
X, y = make_blobs(n_samples=50, centers=2, cluster_std=1.5, random_state=42)
y = 2*y - 1  # Convert to {-1, 1}

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different C values
C_values = [0.1, 1, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, C in enumerate(C_values):
    # Train SVM with different C
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_scaled, y)
    
    # Get decision boundary
    w_svm = svm.coef_[0]
    b_svm = svm.intercept_[0]
    
    # Calculate margin width
    margin_width_svm = 2 / np.linalg.norm(w_svm)
    
    print(f"C = {C}:")
    print(f"  Weight vector: {w_svm}")
    print(f"  Bias: {b_svm:.3f}")
    print(f"  Margin width: {margin_width_svm:.3f}")
    print(f"  Number of support vectors: {len(svm.support_vectors_)}")
    
    # Plot
    ax = axes[i]
    
    # Create mesh for decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'green'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Plot data points
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, 
               edgecolors='black', s=50, alpha=0.8)
    
    # Highlight support vectors
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
               s=200, facecolors='none', edgecolors='black', linewidth=2)
    
    ax.set_title(f'C = {C}\nMargin width = {margin_width_svm:.3f}\nSVs = {len(svm.support_vectors_)}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_vs_C.png'), dpi=300, bbox_inches='tight')
plt.show()

# Task 4: Effect of adding an outlier
print("\nTask 4: Effect of Adding an Outlier")
print("-" * 50)

# Create clean dataset
np.random.seed(42)
X_clean, y_clean = make_blobs(n_samples=40, centers=2, cluster_std=1.0, random_state=42)
y_clean = 2*y_clean - 1

# Add outlier
X_with_outlier = np.vstack([X_clean, [[2, -2]]])
y_with_outlier = np.append(y_clean, 1)  # Outlier belongs to positive class

# Scale data
scaler = StandardScaler()
X_clean_scaled = scaler.fit_transform(X_clean)
X_outlier_scaled = scaler.transform(X_with_outlier)

# Train SVMs
svm_clean = SVC(kernel='linear', C=1, random_state=42)
svm_clean.fit(X_clean_scaled, y_clean)

svm_outlier = SVC(kernel='linear', C=1, random_state=42)
svm_outlier.fit(X_outlier_scaled, y_with_outlier)

# Compare results
print("Without outlier:")
print(f"  Weight vector: {svm_clean.coef_[0]}")
print(f"  Margin width: {2/np.linalg.norm(svm_clean.coef_[0]):.3f}")
print(f"  Support vectors: {len(svm_clean.support_vectors_)}")

print("\nWith outlier:")
print(f"  Weight vector: {svm_outlier.coef_[0]}")
print(f"  Margin width: {2/np.linalg.norm(svm_outlier.coef_[0]):.3f}")
print(f"  Support vectors: {len(svm_outlier.support_vectors_)}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot without outlier
x_min, x_max = X_clean_scaled[:, 0].min() - 0.5, X_clean_scaled[:, 0].max() + 0.5
y_min, y_max = X_clean_scaled[:, 1].min() - 0.5, X_clean_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z_clean = svm_clean.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_clean = Z_clean.reshape(xx.shape)

ax1.contour(xx, yy, Z_clean, levels=[-1, 0, 1], colors=['red', 'black', 'green'], 
            linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
ax1.scatter(X_clean_scaled[:, 0], X_clean_scaled[:, 1], c=y_clean, cmap=plt.cm.RdYlBu, 
            edgecolors='black', s=50, alpha=0.8)
ax1.scatter(svm_clean.support_vectors_[:, 0], svm_clean.support_vectors_[:, 1], 
            s=200, facecolors='none', edgecolors='black', linewidth=2)
ax1.set_title('Without Outlier')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.grid(True, alpha=0.3)

# Plot with outlier
Z_outlier = svm_outlier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_outlier = Z_outlier.reshape(xx.shape)

ax2.contour(xx, yy, Z_outlier, levels=[-1, 0, 1], colors=['red', 'black', 'green'], 
            linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
ax2.scatter(X_outlier_scaled[:, 0], X_outlier_scaled[:, 1], c=y_with_outlier, cmap=plt.cm.RdYlBu, 
            edgecolors='black', s=50, alpha=0.8)
ax2.scatter(svm_outlier.support_vectors_[:, 0], svm_outlier.support_vectors_[:, 1], 
            s=200, facecolors='none', edgecolors='black', linewidth=2)
# Highlight outlier
ax2.scatter(X_outlier_scaled[-1, 0], X_outlier_scaled[-1, 1], 
            s=300, facecolors='red', edgecolors='black', linewidth=3, marker='*', label='Outlier')
ax2.set_title('With Outlier')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'outlier_effect.png'), dpi=300, bbox_inches='tight')
plt.show()

# Task 5: Compare margins with C=1 vs C=100
print("\nTask 5: Compare Margins with C=1 vs C=100")
print("-" * 50)

# Use the same dataset
C_comparison = [1, 100]
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for i, C in enumerate(C_comparison):
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_scaled, y)
    
    w_svm = svm.coef_[0]
    b_svm = svm.intercept_[0]
    margin_width_svm = 2 / np.linalg.norm(w_svm)
    
    print(f"C = {C}:")
    print(f"  Weight vector: {w_svm}")
    print(f"  Margin width: {margin_width_svm:.3f}")
    print(f"  Support vectors: {len(svm.support_vectors_)}")
    
    # Calculate total slack
    decision_values = svm.decision_function(X_scaled)
    slack_values = np.maximum(0, 1 - y * decision_values)
    total_slack = np.sum(slack_values)
    print(f"  Total slack: {total_slack:.3f}")
    
    # Plot
    ax = axes[i]
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'green'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, 
               edgecolors='black', s=50, alpha=0.8)
    
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
               s=200, facecolors='none', edgecolors='black', linewidth=2)
    
    ax.set_title(f'C = {C}\nMargin = {margin_width_svm:.3f}\nSlack = {total_slack:.3f}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'C_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Summary table
print("\nSummary Table:")
print("-" * 50)
print(f"{'C':<8} {'Margin Width':<15} {'Support Vectors':<18} {'Total Slack':<15}")
print("-" * 50)
for C in [0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_scaled, y)
    margin_width = 2 / np.linalg.norm(svm.coef_[0])
    decision_values = svm.decision_function(X_scaled)
    slack_values = np.maximum(0, 1 - y * decision_values)
    total_slack = np.sum(slack_values)
    print(f"{C:<8} {margin_width:<15.3f} {len(svm.support_vectors_):<18} {total_slack:<15.3f}")

print(f"\nAll plots saved to: {save_dir}")
