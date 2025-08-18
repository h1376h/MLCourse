import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("ADABOOST FOUNDATIONS - COMPREHENSIVE SOLUTION")
print("=" * 80)

# ============================================================================
# PART 1: Understanding Weak Learners
# ============================================================================
print("\n" + "="*60)
print("PART 1: UNDERSTANDING WEAK LEARNERS")
print("="*60)

# Generate synthetic data for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          class_sep=0.8, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset size: {len(X_train)} training samples, {len(X_test)} test samples")
print(f"Class distribution: {np.bincount(y_train)}")

# Create weak learners with different depths (complexities)
weak_learners = []
depths = [1, 2, 3, 5]
accuracies = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    weak_learners.append(clf)
    accuracies.append(acc)
    print(f"Weak Learner (depth={depth}): Accuracy = {acc:.3f}")

# Visualize weak learners
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Weak Learners with Different Complexities', fontsize=16)

for i, (clf, depth, acc) in enumerate(zip(weak_learners, depths, accuracies)):
    ax = axes[i//2, i%2]
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
               edgecolors='black', s=50, alpha=0.8)
    
    ax.set_title(f'Depth {depth} (Accuracy: {acc:.3f})')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weak_learners_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 2: Weak Learner Performance Threshold
# ============================================================================
print("\n" + "="*60)
print("PART 2: WEAK LEARNER PERFORMANCE THRESHOLD")
print("="*60)

# Demonstrate the 50% accuracy threshold
print("AdaBoost requires weak learners to have accuracy > 50% (error < 50%)")
print("This is because:")

# Calculate theoretical bounds
errors = [1 - acc for acc in accuracies]
theoretical_bounds = []

for i, (err, acc) in enumerate(zip(errors, accuracies)):
    if err < 0.5:  # Valid weak learner
        alpha = 0.5 * np.log((1 - err) / err)
        bound = 2 * np.sqrt(err * (1 - err))
        theoretical_bounds.append(bound)
        print(f"  Weak Learner {i+1} (depth={depths[i]}):")
        print(f"    Error: {err:.3f} < 0.5 ✓ (Valid)")
        print(f"    Weight α: {alpha:.3f}")
        print(f"    Theoretical bound: {bound:.3f}")
    else:
        print(f"  Weak Learner {i+1} (depth={depths[i]}):")
        print(f"    Error: {err:.3f} >= 0.5 ✗ (Invalid for AdaBoost)")

# Visualize the 50% threshold
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(depths))
bars = plt.bar(x_pos, accuracies, color=['green' if acc > 0.5 else 'red' for acc in accuracies], alpha=0.7)

plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% Threshold')
plt.xlabel('Weak Learner Depth')
plt.ylabel('Accuracy')
plt.title('Weak Learner Performance vs. 50% Threshold')
plt.xticks(x_pos, [f'Depth {d}' for d in depths])
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weak_learner_threshold.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 3: Why AdaBoost Focuses on Misclassified Samples
# ============================================================================
print("\n" + "="*60)
print("PART 3: WHY ADABOOST FOCUSES ON MISCLASSIFIED SAMPLES")
print("="*60)

print("AdaBoost focuses on misclassified samples because:")
print("1. They represent the 'hard' cases that need more attention")
print("2. Focusing on them helps reduce overall training error")
print("3. It creates a sequence of complementary weak learners")

# Simulate AdaBoost weight updates
n_samples = len(X_train)
initial_weights = np.ones(n_samples) / n_samples
print(f"\nInitial sample weights: {initial_weights[:5]}... (all equal)")

# First iteration with weak learner 1 (depth=1)
clf1 = weak_learners[0]
y_pred1 = clf1.predict(X_train)
errors1 = (y_pred1 != y_train).astype(int)
weighted_error1 = np.sum(initial_weights * errors1)

print(f"\nFirst iteration (Weak Learner 1):")
print(f"  Weighted error: {weighted_error1:.3f}")
print(f"  Sample errors: {errors1[:10]}...")

# Calculate alpha for first weak learner
alpha1 = 0.5 * np.log((1 - weighted_error1) / weighted_error1)
print(f"  Weight α₁: {alpha1:.3f}")

# Update weights
new_weights = initial_weights * np.exp(alpha1 * errors1)
new_weights = new_weights / np.sum(new_weights)  # Normalize

print(f"\nUpdated weights after first iteration:")
print(f"  Correctly classified samples: {new_weights[y_pred1 == y_train][:5]}...")
print(f"  Misclassified samples: {new_weights[y_pred1 != y_train][:5]}...")

# Visualize weight distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(initial_weights, bins=20, alpha=0.7, color='blue', label='Initial Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Initial Weight Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(new_weights, bins=20, alpha=0.7, color='red', label='Updated Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Weight Distribution After First Iteration')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_distribution_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: How AdaBoost Combines Weak Learners
# ============================================================================
print("\n" + "="*60)
print("PART 4: HOW ADABOOST COMBINES WEAK LEARNERS")
print("="*60)

print("AdaBoost combines weak learners using weighted voting:")
print("Final prediction = sign(Σ αₘ × hₘ(x))")

# Demonstrate ensemble prediction
sample_idx = 0
sample_x = X_train[sample_idx:sample_idx+1]
sample_y = y_train[sample_idx]

print(f"\nSample prediction demonstration:")
print(f"Sample: x = {sample_x[0]}, True label: {sample_y}")

# Get predictions from all weak learners
predictions = []
for i, clf in enumerate(weak_learners):
    pred = clf.predict(sample_x)[0]
    predictions.append(pred)
    print(f"  Weak Learner {i+1} (depth={depths[i]}): prediction = {pred}")

# Calculate ensemble prediction
valid_learners = [(i, clf, acc) for i, (clf, acc) in enumerate(zip(weak_learners, accuracies)) if acc > 0.5]
ensemble_score = 0

print(f"\nEnsemble prediction calculation:")
for i, clf, acc in valid_learners:
    pred = clf.predict(sample_x)[0]
    alpha = 0.5 * np.log((1 - (1-acc)) / (1-acc))
    contribution = alpha * pred
    ensemble_score += contribution
    print(f"  α_{i+1} × h_{i+1}(x) = {alpha:.3f} × {pred} = {contribution:.3f}")

print(f"  Final score: {ensemble_score:.3f}")
print(f"  Ensemble prediction: {np.sign(ensemble_score)}")

# Visualize ensemble decision boundary
plt.figure(figsize=(10, 8))

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Calculate ensemble predictions
ensemble_predictions = np.zeros(xx.shape)
for i, clf, acc in valid_learners:
    alpha = 0.5 * np.log((1 - (1-acc)) / (1-acc))
    ensemble_predictions += alpha * clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Final ensemble prediction
Z = np.sign(ensemble_predictions)
Z = Z.reshape(xx.shape)

# Plot ensemble decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
           edgecolors='black', s=50, alpha=0.8)

plt.title('AdaBoost Ensemble Decision Boundary')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ensemble_decision_boundary.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: AdaBoost vs Bagging
# ============================================================================
print("\n" + "="*60)
print("PART 5: ADABOOST VS BAGGING")
print("="*60)

print("Key differences between AdaBoost and Bagging:")
print("\n1. Sample Selection:")
print("   - AdaBoost: Adaptively updates sample weights, focuses on hard cases")
print("   - Bagging: Random sampling with replacement, equal importance")

print("\n2. Weak Learner Training:")
print("   - AdaBoost: Sequential training, each learner depends on previous")
print("   - Bagging: Parallel training, learners are independent")

print("\n3. Combination Method:")
print("   - AdaBoost: Weighted voting based on learner performance")
print("   - Bagging: Simple majority voting (equal weights)")

print("\n4. Focus:")
print("   - AdaBoost: Reduces bias by focusing on misclassified samples")
print("   - Bagging: Reduces variance by averaging independent predictions")

# Demonstrate the difference with a simple example
print(f"\nDemonstration with our weak learners:")
print(f"Bagging approach: Simple majority vote")
bagging_predictions = []
for clf in weak_learners:
    pred = clf.predict(sample_x)[0]
    bagging_predictions.append(pred)
    print(f"  Prediction: {pred}")

bagging_final = np.sign(np.mean(bagging_predictions))
print(f"  Bagging final prediction: {bagging_final}")

print(f"\nAdaBoost approach: Weighted vote")
print(f"  AdaBoost final prediction: {np.sign(ensemble_score)}")

# Visualize the difference
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bagging visualization
ax1.set_title('Bagging: Equal Weight Voting')
ax1.bar(range(len(weak_learners)), [1/len(weak_learners)]*len(weak_learners), 
         color='skyblue', alpha=0.7)
ax1.set_xlabel('Weak Learner')
ax1.set_ylabel('Weight')
ax1.set_xticks(range(len(weak_learners)))
ax1.set_xticklabels([f'WL{i+1}' for i in range(len(weak_learners))])
ax1.grid(True, alpha=0.3)

# AdaBoost visualization
valid_weights = []
valid_labels = []
for i, (clf, acc) in enumerate(zip(weak_learners, accuracies)):
    if acc > 0.5:
        alpha = 0.5 * np.log((1 - (1-acc)) / (1-acc))
        valid_weights.append(alpha)
        valid_labels.append(f'WL{i+1}')

ax2.set_title('AdaBoost: Weighted Voting')
ax2.bar(range(len(valid_weights)), valid_weights, color='lightcoral', alpha=0.7)
ax2.set_xlabel('Weak Learner')
ax2.set_ylabel('Weight ($\\alpha$)')
ax2.set_xticks(range(len(valid_weights)))
ax2.set_xticklabels(valid_labels)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_vs_bagging.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: 45% Accuracy Weak Learner Suitability
# ============================================================================
print("\n" + "="*60)
print("PART 6: 45% ACCURACY WEAK LEARNER SUITABILITY")
print("="*60)

accuracy_45 = 0.45
error_45 = 1 - accuracy_45

print(f"Question: Is a weak learner with 45% accuracy suitable for AdaBoost?")
print(f"Answer: NO - This weak learner is NOT suitable for AdaBoost")
print(f"\nReasoning:")
print(f"1. Accuracy: {accuracy_45:.1%}")
print(f"2. Error rate: {error_45:.1%}")
print(f"3. AdaBoost requirement: Error < 50% (Accuracy > 50%)")
print(f"4. Since {error_45:.1%} >= 50%, this weak learner fails the requirement")

# Calculate what would happen if we tried to use it
try:
    alpha_45 = 0.5 * np.log((1 - error_45) / error_45)
    print(f"\nIf we tried to use it anyway:")
    print(f"  α = 0.5 × ln((1-{error_45:.3f})/{error_45:.3f}) = {alpha_45:.3f}")
    print(f"  This would be negative, meaning the weak learner gets negative weight")
    print(f"  Negative weight means the weak learner's predictions are inverted")
    print(f"  This breaks AdaBoost's theoretical guarantees")
except:
    print(f"\nMathematical issue: Cannot compute α for error >= 50%")

# Visualize the problem
plt.figure(figsize=(10, 6))
accuracies_with_45 = accuracies + [0.45]
depths_with_45 = depths + [4]

colors = ['green' if acc > 0.5 else 'red' for acc in accuracies_with_45]
bars = plt.bar(range(len(accuracies_with_45)), accuracies_with_45, 
               color=colors, alpha=0.7)

plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% Threshold')
plt.xlabel('Weak Learner')
plt.ylabel('Accuracy')
plt.title('Weak Learner Suitability for AdaBoost')
plt.xticks(range(len(accuracies_with_45)), 
           [f'Depth {d}' for d in depths_with_45])
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies_with_45)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weak_learner_suitability.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# ADDITIONAL VISUALIZATIONS FOR VISUAL EXPLANATIONS
# ============================================================================
print("\n" + "="*60)
print("ADDITIONAL VISUALIZATIONS FOR VISUAL EXPLANATIONS")
print("="*60)

# 1. AdaBoost Algorithm Flow Visualization
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)

# Simulate AdaBoost iterations
iterations = 3
sample_weights_history = []
alphas_history = []
errors_history = []

# Initialize weights
current_weights = np.ones(n_samples) / n_samples
sample_weights_history.append(current_weights.copy())

for iteration in range(iterations):
    # Train weak learner (simplified - using existing ones)
    clf = weak_learners[iteration % len(weak_learners)]
    y_pred = clf.predict(X_train)
    
    # Calculate weighted error
    errors = (y_pred != y_train).astype(int)
    weighted_error = np.sum(current_weights * errors)
    errors_history.append(weighted_error)
    
    # Calculate alpha
    if weighted_error > 0:
        alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
        alphas_history.append(alpha)
        
        # Update weights
        current_weights = current_weights * np.exp(alpha * errors)
        current_weights = current_weights / np.sum(current_weights)
        sample_weights_history.append(current_weights.copy())
    else:
        # If error is 0, we have a perfect classifier
        alphas_history.append(10.0)  # Large positive weight
        sample_weights_history.append(current_weights.copy())

# Plot weight evolution
for i in range(min(5, n_samples)):  # Plot first 5 samples
    weights_i = [w[i] for w in sample_weights_history]
    plt.plot(range(len(weights_i)), weights_i, marker='o', label=f'Sample {i+1}')

plt.xlabel('Iteration')
plt.ylabel('Sample Weight')
plt.title('Sample Weight Evolution During AdaBoost Training')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Alpha values vs Error rates
plt.subplot(2, 2, 2)
error_range = np.linspace(0.01, 0.49, 100)
alpha_values = 0.5 * np.log((1 - error_range) / error_range)

plt.plot(error_range, alpha_values, 'b-', linewidth=2)
plt.axvline(x=0.5, color='red', linestyle='--', label='50% Error Threshold')
plt.xlabel('Error Rate ($\\epsilon$)')
plt.ylabel('Weight ($\\alpha$)')
plt.title('Relationship Between Error Rate and Weight')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Theoretical bound visualization
plt.subplot(2, 2, 3)
bound_values = 2 * np.sqrt(error_range * (1 - error_range))
plt.plot(error_range, bound_values, 'g-', linewidth=2)
plt.axvline(x=0.5, color='red', linestyle='--', label='50% Error Threshold')
plt.xlabel('Error Rate ($\\epsilon$)')
plt.ylabel('Theoretical Bound')
plt.title('Theoretical Error Bound vs Error Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Ensemble performance improvement
plt.subplot(2, 2, 4)
cumulative_errors = []
for i in range(len(errors_history)):
    cumulative_bound = np.prod([2 * np.sqrt(errors_history[j] * (1 - errors_history[j])) 
                               for j in range(i+1)])
    cumulative_errors.append(cumulative_bound)

plt.plot(range(1, len(cumulative_errors) + 1), cumulative_errors, 'ro-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cumulative Error Bound')
plt.title('AdaBoost Error Bound Convergence')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_algorithm_flow.png'), dpi=300, bbox_inches='tight')

# 5. Sample weight heatmap
plt.figure(figsize=(12, 8))
weight_matrix = np.array(sample_weights_history).T
plt.imshow(weight_matrix, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='Sample Weight')
plt.xlabel('Iteration')
plt.ylabel('Sample Index')
plt.title('Sample Weight Heatmap During AdaBoost Training')
plt.xticks(range(len(sample_weights_history)), [f'Iter {i}' for i in range(len(sample_weights_history))])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sample_weight_heatmap.png'), dpi=300, bbox_inches='tight')

# 6. Decision boundary evolution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Decision Boundary Evolution During AdaBoost Training', fontsize=16)

for iteration in range(min(4, len(sample_weights_history) - 1)):
    ax = axes[iteration//2, iteration%2]
    
    # Use current weights for this iteration
    current_weights = sample_weights_history[iteration]
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Calculate ensemble predictions up to this iteration
    ensemble_predictions = np.zeros(xx.shape)
    for i in range(iteration + 1):
        clf = weak_learners[i % len(weak_learners)]
        alpha = alphas_history[i] if i < len(alphas_history) else 1.0
        ensemble_predictions += alpha * clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    Z = np.sign(ensemble_predictions)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot samples with size proportional to weight
    sample_sizes = current_weights * 1000  # Scale weights for visualization
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
               edgecolors='black', s=sample_sizes, alpha=0.8)
    
    ax.set_title(f'Iteration {iteration + 1} ($\\alpha$ = {alphas_history[iteration]:.3f})')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundary_evolution.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND KEY INSIGHTS")
print("="*80)

print("\n1. Weak Learner Definition:")
print("   - A classifier that performs slightly better than random guessing")
print("   - Must have error rate < 50% (accuracy > 50%)")
print("   - Typically simple models like decision stumps or shallow trees")

print("\n2. Why Focus on Misclassified Samples:")
print("   - They represent the 'hard' cases that need attention")
print("   - Focusing on them helps reduce overall training error")
print("   - Creates a sequence of complementary weak learners")

print("\n3. Combination Method:")
print("   - Weighted voting: Final prediction = sign(Σ αₘ × hₘ(x))")
print("   - Weights αₘ = 0.5 × ln((1-εₘ)/εₘ)")
print("   - Better weak learners get higher weights")

print("\n4. AdaBoost vs Bagging:")
print("   - AdaBoost: Sequential, adaptive, weighted voting")
print("   - Bagging: Parallel, random, equal voting")
print("   - AdaBoost focuses on reducing bias, Bagging on reducing variance")

print("\n5. Weak Learner Suitability:")
print("   - Must have accuracy > 50%")
print("   - 45% accuracy is NOT suitable")
print("   - Below 50% breaks theoretical guarantees")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
