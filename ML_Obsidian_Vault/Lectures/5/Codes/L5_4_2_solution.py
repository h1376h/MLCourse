import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 2: ONE-VS-REST (OvR) MATHEMATICAL ANALYSIS")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 3-class dataset for demonstration
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a 3-class problem (K=3)
K = 3
classes = [0, 1, 2]

print(f"\n1. OPTIMIZATION PROBLEM FOR THE k-TH BINARY CLASSIFIER")
print("-" * 60)

# For each class k, we create a binary classifier: class k vs all others
for k in classes:
    print(f"\nBinary Classifier {k}: Class {k} vs Rest")
    print(f"Training data transformation:")
    print(f"  - Positive class (y = +1): samples from class {k}")
    print(f"  - Negative class (y = -1): samples from all other classes")
    
    # Create binary labels for class k vs rest
    y_binary = np.where(y == k, 1, -1)
    
    # Count samples in each class
    n_positive = np.sum(y_binary == 1)
    n_negative = np.sum(y_binary == -1)
    
    print(f"  - Positive samples: {n_positive}")
    print(f"  - Negative samples: {n_negative}")
    print(f"  - Class imbalance ratio: {n_negative/n_positive:.2f}:1")
    
    print(f"\nOptimization problem for classifier {k}:")
    print(f"minimize: (1/2)||w_{k}||² + C ∑ᵢ ξᵢ")
    print(f"subject to:")
    print(f"  yᵢ(w_{k}ᵀxᵢ + b_{k}) ≥ 1 - ξᵢ, ∀i")
    print(f"  ξᵢ ≥ 0, ∀i")
    print(f"where:")
    print(f"  - w_{k} is the weight vector for class {k}")
    print(f"  - b_{k} is the bias term for class {k}")
    print(f"  - C is the regularization parameter")
    print(f"  - ξᵢ are slack variables for soft margin")

print(f"\n2. OVR DECISION RULE: ŷ = argmax_k f_k(x)")
print("-" * 60)

# Train OvR classifiers
classifiers = []
for k in classes:
    y_binary = np.where(y == k, 1, -1)
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X, y_binary)
    classifiers.append(clf)

# Create test points to demonstrate decision making
test_points = np.array([
    [0, 0],      # Center point
    [1, 1],      # Top right
    [-1, -1],    # Bottom left
    [0.5, -0.5], # Mixed region
])

print(f"Decision rule demonstration:")
print(f"For each test point x, we compute f_k(x) = w_kᵀx + b_k for all k")
print(f"Then predict: ŷ = argmax_k f_k(x)")

for i, test_point in enumerate(test_points):
    print(f"\nTest Point {i+1}: x = {test_point}")
    
    # Get decision values from all classifiers
    decision_values = []
    for k, clf in enumerate(classifiers):
        # Get decision function value (distance from margin)
        decision_val = clf.decision_function([test_point])[0]
        decision_values.append(decision_val)
        print(f"  f_{k}(x) = {decision_val:.3f}")
    
    # Apply OvR decision rule
    predicted_class = np.argmax(decision_values)
    print(f"  Predicted class: {predicted_class} (argmax)")
    print(f"  Decision values: {decision_values}")

print(f"\n3. PROBLEMS WITH MULTIPLE POSITIVE VALUES")
print("-" * 60)

# Demonstrate ambiguous case
ambiguous_point = np.array([0.2, 0.1])
print(f"Ambiguous case demonstration:")
print(f"Test point: x = {ambiguous_point}")

decision_values_amb = []
for k, clf in enumerate(classifiers):
    decision_val = clf.decision_function([ambiguous_point])[0]
    decision_values_amb.append(decision_val)
    print(f"  f_{k}(x) = {decision_val:.3f}")

# Find positive values
positive_indices = [i for i, val in enumerate(decision_values_amb) if val > 0]
print(f"\nPositive classifier outputs: {positive_indices}")
print(f"Problems:")
print(f"  1. Multiple classes claim the sample (ambiguous prediction)")
print(f"  2. No clear winner - need tie-breaking mechanism")
print(f"  3. Confidence is low due to competing positive signals")

print(f"\n4. PROBLEMS WITH ALL NEGATIVE VALUES")
print("-" * 60)

# Find a point where all classifiers output negative values
# This typically happens in regions far from all class centers
far_point = np.array([3, 3])
print(f"All negative case demonstration:")
print(f"Test point: x = {far_point}")

decision_values_neg = []
for k, clf in enumerate(classifiers):
    decision_val = clf.decision_function([far_point])[0]
    decision_values_neg.append(decision_val)
    print(f"  f_{k}(x) = {decision_val:.3f}")

print(f"\nAll classifier outputs are negative!")
print(f"Problems:")
print(f"  1. No class claims the sample (rejection case)")
print(f"  2. Sample may be outlier or noise")
print(f"  3. Need rejection threshold or confidence measure")
print(f"  4. May indicate need for better feature representation")

print(f"\n5. CONFIDENCE MEASURE BASED ON MARGIN")
print("-" * 60)

def calculate_confidence(decision_values):
    """Calculate confidence based on margin between top two scores"""
    sorted_indices = np.argsort(decision_values)[::-1]  # Descending order
    top_score = decision_values[sorted_indices[0]]
    second_score = decision_values[sorted_indices[1]]
    
    margin = top_score - second_score
    confidence = margin / (abs(top_score) + abs(second_score) + 1e-8)  # Avoid division by zero
    
    return margin, confidence, sorted_indices[0]

# Test confidence calculation on different scenarios
test_scenarios = [
    ("Clear winner", [2.0, -1.0, -0.5]),
    ("Close competition", [1.0, 0.9, -0.5]),
    ("Ambiguous", [0.8, 0.7, -0.3]),
    ("All negative", [-0.5, -1.0, -1.5])
]

for scenario_name, decision_vals in test_scenarios:
    margin, confidence, predicted_class = calculate_confidence(decision_vals)
    print(f"\n{scenario_name}:")
    print(f"  Decision values: {decision_vals}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Margin: {margin:.3f}")
    print(f"  Confidence: {confidence:.3f}")

# Visualization 1: OvR Decision Boundaries
plt.figure(figsize=(15, 10))

# Plot training data
colors = ['red', 'blue', 'green']
for k in classes:
    mask = y == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], label=f'Class {k}', alpha=0.7, s=50)

# Plot decision boundaries for each classifier
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

for k, clf in enumerate(classifiers):
    # Get decision function for the mesh
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (where decision function = 0)
    plt.contour(xx, yy, Z, levels=[0], colors=[colors[k]], linewidths=2, 
                linestyles=['--'], label=f'Class {k} vs Rest')

# Plot test points
test_colors = ['purple', 'orange', 'brown', 'pink']
for i, test_point in enumerate(test_points):
    plt.scatter(test_point[0], test_point[1], c=test_colors[i], s=200, 
                marker='*', edgecolor='black', linewidth=2, 
                label=f'Test Point {i+1}')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('One-vs-Rest Decision Boundaries')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'ovr_decision_boundaries.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Decision Function Values
plt.figure(figsize=(15, 10))

# Create a finer mesh for smooth visualization
xx_fine, yy_fine = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Plot decision function values for each classifier
for k, clf in enumerate(classifiers):
    plt.subplot(2, 2, k+1)
    
    # Get decision function values
    Z = clf.decision_function(np.c_[xx_fine.ravel(), yy_fine.ravel()])
    Z = Z.reshape(xx_fine.shape)
    
    # Plot decision function as heatmap
    contour = plt.contourf(xx_fine, yy_fine, Z, levels=20, cmap='RdBu_r', alpha=0.7)
    plt.colorbar(contour, label=f'$f_{k}(x)$')
    
    # Plot decision boundary
    plt.contour(xx_fine, yy_fine, Z, levels=[0], colors='black', linewidths=2)
    
    # Plot training data
    for class_idx in classes:
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx], 
                   label=f'Class {class_idx}', alpha=0.7, s=30)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'Decision Function $f_{k}(x)$: Class {k} vs Rest')
    plt.legend()

# Plot final OvR decision regions
plt.subplot(2, 2, 4)
Z_final = np.zeros_like(xx_fine)
for k, clf in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx_fine.ravel(), yy_fine.ravel()])
    Z = Z.reshape(xx_fine.shape)
    Z_final = np.maximum(Z_final, Z * (Z > 0))  # Only positive values contribute

# Create discrete decision regions
Z_discrete = np.argmax([clf.decision_function(np.c_[xx_fine.ravel(), yy_fine.ravel()]).reshape(xx_fine.shape) 
                       for clf in classifiers], axis=0)

plt.contourf(xx_fine, yy_fine, Z_discrete, levels=len(classes)-1, 
            colors=colors, alpha=0.3)
plt.contour(xx_fine, yy_fine, Z_discrete, levels=len(classes)-1, 
           colors='black', linewidths=1)

# Plot training data
for k in classes:
    mask = y == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], 
               label=f'Class {k}', alpha=0.7, s=30)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Final OvR Decision Regions')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_decision_functions.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Confidence Analysis
plt.figure(figsize=(15, 5))

# Generate points across the feature space
xx_conf, yy_conf = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
confidence_map = np.zeros_like(xx_conf)
margin_map = np.zeros_like(xx_conf)

for i in range(xx_conf.shape[0]):
    for j in range(xx_conf.shape[1]):
        point = np.array([xx_conf[i, j], yy_conf[i, j]])
        decision_vals = [clf.decision_function([point])[0] for clf in classifiers]
        margin, confidence, _ = calculate_confidence(decision_vals)
        confidence_map[i, j] = confidence
        margin_map[i, j] = margin

# Plot confidence map
plt.subplot(1, 3, 1)
contour = plt.contourf(xx_conf, yy_conf, confidence_map, levels=20, cmap='viridis')
plt.colorbar(contour, label='Confidence')
plt.contour(xx_conf, yy_conf, confidence_map, levels=[0.1, 0.3, 0.5], colors='white', linewidths=1)
plt.title('Prediction Confidence')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot margin map
plt.subplot(1, 3, 2)
contour = plt.contourf(xx_conf, yy_conf, margin_map, levels=20, cmap='plasma')
plt.colorbar(contour, label='Margin')
plt.contour(xx_conf, yy_conf, margin_map, levels=[0.5, 1.0, 1.5], colors='white', linewidths=1)
plt.title('Decision Margin')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot rejection regions (low confidence)
plt.subplot(1, 3, 3)
rejection_mask = confidence_map < 0.2
plt.contourf(xx_conf, yy_conf, rejection_mask, levels=[0, 0.5, 1], 
            colors=['white', 'red'], alpha=0.3)
plt.title('Rejection Regions (Confidence < 0.2)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_confidence_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Decision Value Distributions
plt.figure(figsize=(12, 8))

# Generate test points across the space
test_grid = np.random.uniform(x_min, x_max, (1000, 2))

# Collect decision values for all test points
all_decision_values = []
for point in test_grid:
    decision_vals = [clf.decision_function([point])[0] for clf in classifiers]
    all_decision_values.append(decision_vals)

all_decision_values = np.array(all_decision_values)

# Plot histograms of decision values
plt.subplot(2, 2, 1)
for k in range(K):
    plt.hist(all_decision_values[:, k], bins=30, alpha=0.7, 
             label=f'Class {k} vs Rest', color=colors[k])
plt.xlabel('Decision Value')
plt.ylabel('Frequency')
plt.title('Distribution of Decision Values')
plt.legend()

# Plot correlation between decision values
plt.subplot(2, 2, 2)
correlation_matrix = np.corrcoef(all_decision_values.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            xticklabels=[f'f_{k}' for k in range(K)],
            yticklabels=[f'f_{k}' for k in range(K)])
plt.title('Correlation Between Decision Functions')

# Plot margin distribution
plt.subplot(2, 2, 3)
margins = []
for decision_vals in all_decision_values:
    sorted_vals = np.sort(decision_vals)[::-1]
    margin = sorted_vals[0] - sorted_vals[1]
    margins.append(margin)

plt.hist(margins, bins=30, alpha=0.7, color='purple')
plt.xlabel('Margin (Top Score - Second Score)')
plt.ylabel('Frequency')
plt.title('Distribution of Decision Margins')

# Plot confidence distribution
plt.subplot(2, 2, 4)
confidences = []
for decision_vals in all_decision_values:
    sorted_vals = np.sort(decision_vals)[::-1]
    confidence = (sorted_vals[0] - sorted_vals[1]) / (abs(sorted_vals[0]) + abs(sorted_vals[1]) + 1e-8)
    confidences.append(confidence)

plt.hist(confidences, bins=30, alpha=0.7, color='orange')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidence')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_decision_distributions.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "=" * 80)
print("VISUALIZATIONS GENERATED")
print("=" * 80)
print(f"1. ovr_decision_boundaries.png - Shows decision boundaries for each binary classifier")
print(f"2. ovr_decision_functions.png - Shows decision function values and final regions")
print(f"3. ovr_confidence_analysis.png - Shows confidence and margin analysis")
print(f"4. ovr_decision_distributions.png - Shows statistical distributions")
print(f"\nAll figures saved to: {save_dir}")

# Summary of key findings
print(f"\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print(f"1. Optimization Problem: Each binary classifier solves a separate SVM problem")
print(f"2. Decision Rule: argmax_k f_k(x) correctly identifies the winning class")
print(f"3. Multiple Positive Values: Create ambiguity and require tie-breaking")
print(f"4. All Negative Values: Indicate rejection cases or outliers")
print(f"5. Confidence Measure: Margin-based confidence helps quantify prediction reliability")
