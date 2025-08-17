import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("Question 26: AdaBoost Weight Detective")
print("=" * 60)

# Given dataset
X = np.array([1, 2, 3, 4, 5, 6])  # Features
y = np.array([1, 1, -1, -1, 1, -1])  # True labels
n_samples = len(X)

print(f"Dataset:")
print(f"Features (x): {X}")
print(f"Labels (y): {y}")
print(f"Number of samples: {n_samples}")

# Weak learners
def h1(x):
    """h1(x): +1 if x ≤ 3.5, -1 otherwise"""
    return np.where(x <= 3.5, 1, -1)

def h2(x):
    """h2(x): +1 if x ≤ 2.5, -1 otherwise"""
    return np.where(x <= 2.5, 1, -1)

def h3(x):
    """h3(x): +1 if x ≤ 4.5, -1 otherwise"""
    return np.where(x <= 4.5, 1, -1)

print(f"\nWeak Learners:")
print(f"h1(x): +1 if x ≤ 3.5, -1 otherwise")
print(f"h2(x): +1 if x ≤ 2.5, -1 otherwise")
print(f"h3(x): +1 if x ≤ 4.5, -1 otherwise")

# Step 1: Initial weights
print("\n" + "="*60)
print("STEP 1: Initial Weights")
print("="*60)

# All samples start with equal weights
w_initial = np.ones(n_samples) / n_samples
print(f"Initial weights (all equal): {w_initial}")
print(f"Sum of weights: {np.sum(w_initial):.6f}")

# Step 2: Iteration 1 with h1
print("\n" + "="*60)
print("STEP 2: Iteration 1 with h1")
print("="*60)

# Get predictions from h1
h1_predictions = h1(X)
print(f"h1 predictions: {h1_predictions}")

# Calculate errors (1 if misclassified, 0 if correct)
h1_errors = (h1_predictions != y).astype(int)
print(f"h1 errors (1=misclassified, 0=correct): {h1_errors}")

# Calculate weighted error step by step
print(f"\nCalculating weighted error ε₁ = Σ(w_i × error_i):")
h1_weighted_error = 0
for i in range(n_samples):
    term = w_initial[i] * h1_errors[i]
    h1_weighted_error += term
    print(f"  w_{i+1} × error_{i+1} = {w_initial[i]:.6f} × {h1_errors[i]} = {term:.6f}")
print(f"ε₁ = {h1_weighted_error:.6f}")

# Calculate alpha1 step by step
print(f"\nCalculating α₁ = 0.5 × ln((1-ε₁)/ε₁):")
print(f"  1 - ε₁ = 1 - {h1_weighted_error:.6f} = {1 - h1_weighted_error:.6f}")
print(f"  (1-ε₁)/ε₁ = {1 - h1_weighted_error:.6f} / {h1_weighted_error:.6f} = {(1 - h1_weighted_error) / h1_weighted_error:.6f}")
h1_alpha = 0.5 * np.log((1 - h1_weighted_error) / h1_weighted_error)
print(f"  ln({(1 - h1_weighted_error) / h1_weighted_error:.6f}) = {np.log((1 - h1_weighted_error) / h1_weighted_error):.6f}")
print(f"  α₁ = 0.5 × {np.log((1 - h1_weighted_error) / h1_weighted_error):.6f} = {h1_alpha:.6f}")

# Update weights after h1 step by step
print(f"\nUpdating weights: w_i^new = w_i^old × exp(α₁ × y_i × h₁(x_i))")
w_after_h1 = np.zeros(n_samples)
for i in range(n_samples):
    exponent = h1_alpha * y[i] * h1_predictions[i]
    w_after_h1[i] = w_initial[i] * np.exp(exponent)
    print(f"  w_{i+1}^new = {w_initial[i]:.6f} × exp({h1_alpha:.6f} × {y[i]} × {h1_predictions[i]})")
    print(f"        = {w_initial[i]:.6f} × exp({exponent:.6f}) = {w_initial[i]:.6f} × {np.exp(exponent):.6f} = {w_after_h1[i]:.6f}")

print(f"\nNew weights before normalization: {w_after_h1}")

# Normalize weights step by step
print(f"\nNormalizing weights:")
sum_weights = np.sum(w_after_h1)
print(f"Sum of new weights = {sum_weights:.6f}")
w_after_h1_normalized = w_after_h1 / sum_weights
for i in range(n_samples):
    print(f"  w_{i+1}^normalized = {w_after_h1[i]:.6f} / {sum_weights:.6f} = {w_after_h1_normalized[i]:.6f}")

print(f"Normalized weights after h1: {w_after_h1_normalized}")
print(f"Sum of normalized weights: {np.sum(w_after_h1_normalized):.6f}")

# Visualization for Iteration 1
print(f"\n" + "="*60)
print("ITERATION 1 VISUALIZATIONS")
print("="*60)

# Create a comprehensive visualization for iteration 1
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Iteration 1: AdaBoost Analysis with h₁', fontsize=16, fontweight='bold')

# Subplot 1: Sample positions and h1 decision boundary
x_range = np.linspace(0, 7, 100)
ax1.axvline(x=3.5, color='red', linestyle='--', linewidth=3, label='h₁: x ≤ 3.5')
ax1.scatter(X[y == 1], [0.5] * len(X[y == 1]), s=200, color='green', marker='o', 
           label='Positive samples (y=+1)', alpha=0.7)
ax1.scatter(X[y == -1], [0.5] * len(X[y == -1]), s=200, color='red', marker='s', 
           label='Negative samples (y=-1)', alpha=0.7)

# Add sample labels with predictions
for i, (x, label, pred) in enumerate(zip(X, y, h1_predictions)):
    color = 'green' if pred == label else 'red'
    ax1.annotate(f'S{i+1}\n(x={x}, y={label})\nh₁={pred}', 
                 (x, 0.5), xytext=(0, 30), textcoords='offset points',
                 ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="black", alpha=0.8))

ax1.set_xlabel('Feature Value (x)')
ax1.set_ylabel('Decision Space')
ax1.set_title('h₁ Decision Boundary and Sample Classifications')
ax1.set_xlim(0, 7)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Subplot 2: Weight evolution from initial to after h1
x_pos = np.arange(n_samples)
width = 0.35

ax2.bar(x_pos - width/2, w_initial, width, label='Initial Weights', alpha=0.8, color='skyblue')
ax2.bar(x_pos + width/2, w_after_h1_normalized, width, label='Weights after h₁', alpha=0.8, color='lightgreen')

ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Weight')
ax2.set_title('Weight Evolution: Initial → After h₁')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'S{i+1}' for i in range(n_samples)])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add weight values on bars
for i in range(n_samples):
    ax2.text(i - width/2, w_initial[i] + 0.01, f'{w_initial[i]:.3f}', 
             ha='center', va='bottom', fontsize=8)
    ax2.text(i + width/2, w_after_h1_normalized[i] + 0.01, f'{w_after_h1_normalized[i]:.3f}', 
             ha='center', va='bottom', fontsize=8)

# Subplot 3: Error analysis for h1
correct_samples = (h1_errors == 0).sum()
incorrect_samples = (h1_errors == 1).sum()

colors = ['lightgreen', 'lightcoral']
labels = ['Correct', 'Incorrect']
sizes = [correct_samples, incorrect_samples]

ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
ax3.set_title(f'h₁ Performance: {correct_samples}/{n_samples} Correct ({correct_samples/n_samples*100:.1f}%)')

# Subplot 4: Weight update factors for each sample
weight_factors = np.exp(h1_alpha * y * h1_predictions)
ax4.bar(range(n_samples), weight_factors, color=['lightgreen' if f > 1 else 'lightcoral' for f in weight_factors], alpha=0.7)
ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='No change (factor=1)')
ax4.set_xlabel('Sample Number')
ax4.set_ylabel('Weight Update Factor')
ax4.set_title('Weight Update Factors: exp(α₁ × y × h₁(x))')
ax4.set_xticks(range(n_samples))
ax4.set_xticklabels([f'S{i+1}' for i in range(n_samples)])
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add factor values on bars
for i, factor in enumerate(weight_factors):
    ax4.text(i, factor + 0.05, f'{factor:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'iteration1_analysis.png'), dpi=300, bbox_inches='tight')
print(f"Iteration 1 visualization saved to: {save_dir}/iteration1_analysis.png")

# Step 3: Iteration 2 with h2
print("\n" + "="*60)
print("STEP 3: Iteration 2 with h2")
print("="*60)

# Get predictions from h2
h2_predictions = h2(X)
print(f"h2 predictions: {h2_predictions}")

# Calculate errors
h2_errors = (h2_predictions != y).astype(int)
print(f"h2 errors (1=misclassified, 0=correct): {h2_errors}")

# Calculate weighted error using weights from h1 step by step
print(f"\nCalculating weighted error ε₂ = Σ(w_i × error_i):")
h2_weighted_error = 0
for i in range(n_samples):
    term = w_after_h1_normalized[i] * h2_errors[i]
    h2_weighted_error += term
    print(f"  w_{i+1} × error_{i+1} = {w_after_h1_normalized[i]:.6f} × {h2_errors[i]} = {term:.6f}")
print(f"ε₂ = {h2_weighted_error:.6f}")

# Calculate alpha2 step by step
print(f"\nCalculating α₂ = 0.5 × ln((1-ε₂)/ε₂):")
print(f"  1 - ε₂ = 1 - {h2_weighted_error:.6f} = {1 - h2_weighted_error:.6f}")
print(f"  (1-ε₂)/ε₂ = {1 - h2_weighted_error:.6f} / {h2_weighted_error:.6f} = {(1 - h2_weighted_error) / h2_weighted_error:.6f}")
h2_alpha = 0.5 * np.log((1 - h2_weighted_error) / h2_weighted_error)
print(f"  ln({(1 - h2_weighted_error) / h2_weighted_error:.6f}) = {np.log((1 - h2_weighted_error) / h2_weighted_error):.6f}")
print(f"  α₂ = 0.5 × {np.log((1 - h2_weighted_error) / h2_weighted_error):.6f} = {h2_alpha:.6f}")

# Update weights after h2 step by step
print(f"\nUpdating weights: w_i^new = w_i^old × exp(α₂ × y_i × h₂(x_i))")
w_after_h2 = np.zeros(n_samples)
for i in range(n_samples):
    exponent = h2_alpha * y[i] * h2_predictions[i]
    w_after_h2[i] = w_after_h1_normalized[i] * np.exp(exponent)
    print(f"  w_{i+1}^new = {w_after_h1_normalized[i]:.6f} × exp({h2_alpha:.6f} × {y[i]} × {h2_predictions[i]})")
    print(f"        = {w_after_h1_normalized[i]:.6f} × exp({exponent:.6f}) = {w_after_h1_normalized[i]:.6f} × {np.exp(exponent):.6f} = {w_after_h2[i]:.6f}")

print(f"\nNew weights before normalization: {w_after_h2}")

# Normalize weights step by step
print(f"\nNormalizing weights:")
sum_weights = np.sum(w_after_h2)
print(f"Sum of new weights = {sum_weights:.6f}")
w_after_h2_normalized = w_after_h2 / sum_weights
for i in range(n_samples):
    print(f"  w_{i+1}^normalized = {w_after_h2[i]:.6f} / {sum_weights:.6f} = {w_after_h2[i]:.6f}")

print(f"Normalized weights after h2: {w_after_h2_normalized}")
print(f"Sum of normalized weights: {np.sum(w_after_h2_normalized):.6f}")

# Step 4: Analysis of weight evolution
print("\n" + "="*60)
print("STEP 4: Weight Evolution Analysis")
print("="*60)

# Create weight evolution table
weight_evolution = pd.DataFrame({
    'Sample': [f'Sample {i+1}' for i in range(n_samples)],
    'Feature (x)': X,
    'True Label (y)': y,
    'Initial Weight': w_initial,
    'Weight after h1': w_after_h1_normalized,
    'Weight after h2': w_after_h2_normalized
})

print("Weight Evolution Table:")
print(weight_evolution.to_string(index=False, float_format='%.6f'))

# Find samples with highest weights after 2 iterations
max_weight_idx = np.argmax(w_after_h2_normalized)
print(f"\nSample with highest weight after 2 iterations: Sample {max_weight_idx + 1} (x={X[max_weight_idx]}, y={y[max_weight_idx]})")
print(f"Weight: {w_after_h2_normalized[max_weight_idx]:.6f}")

# Analyze why this sample has high weight
print(f"\nAnalysis of why Sample {max_weight_idx + 1} has high weight:")
print(f"- Feature value: x = {X[max_weight_idx]}")
print(f"- True label: y = {y[max_weight_idx]}")

# Check h1 and h2 predictions for this sample
h1_pred = h1_predictions[max_weight_idx]
h2_pred = h2_predictions[max_weight_idx]
print(f"- h1 prediction: {h1_pred}")
print(f"- h2 prediction: {h2_pred}")

if h1_pred != y[max_weight_idx]:
    print(f"- Sample was misclassified by h1 (contributed to weight increase)")
if h2_pred != y[max_weight_idx]:
    print(f"- Sample was misclassified by h2 (contributed to weight increase)")

# Step 5: Final ensemble prediction
print("\n" + "="*60)
print("STEP 5: Final Ensemble Prediction")
print("="*60)

# Given predictions from h1 and h2
h1_final_pred = np.array([1, 1, -1, -1, 1, -1])
h2_final_pred = np.array([1, 1, -1, -1, 1, -1])

print(f"h1 final predictions: {h1_final_pred}")
print(f"h2 final predictions: {h2_final_pred}")

# Calculate ensemble prediction: sign(α₁×h₁ + α₂×h₂)
print(f"\nCalculating ensemble prediction: sign(α₁×h₁ + α₂×h₂)")
print(f"where α₁ = {h1_alpha:.6f} and α₂ = {h2_alpha:.6f}")

# Calculate the weighted sum for each sample step by step
weighted_sum = np.zeros(n_samples)
ensemble_pred = np.zeros(n_samples)

print(f"\nFor each sample, calculate: α₁×h₁(x_i) + α₂×h₂(x_i)")
for i in range(n_samples):
    term1 = h1_alpha * h1_final_pred[i]
    term2 = h2_alpha * h2_final_pred[i]
    weighted_sum[i] = term1 + term2
    ensemble_pred[i] = np.sign(weighted_sum[i])
    
    print(f"  Sample {i+1}: α₁×h₁({X[i]}) + α₂×h₂({X[i]}) = {h1_alpha:.6f}×{h1_final_pred[i]} + {h2_alpha:.6f}×{h2_final_pred[i]}")
    print(f"        = {term1:.6f} + {term2:.6f} = {weighted_sum[i]:.6f}")
    print(f"        sign({weighted_sum[i]:.6f}) = {ensemble_pred[i]:.0f}")

print(f"\nWeighted sums: {weighted_sum}")
print(f"Final ensemble predictions: {ensemble_pred}")

# Create detailed prediction table
prediction_table = pd.DataFrame({
    'Sample': [f'Sample {i+1}' for i in range(n_samples)],
    'Feature (x)': X,
    'True Label (y)': y,
    'h1 Prediction': h1_final_pred,
    'h2 Prediction': h2_final_pred,
    'Weighted Sum': weighted_sum,
    'Ensemble Prediction': ensemble_pred,
    'Correct?': ensemble_pred == y
})

print(f"\nDetailed Prediction Table:")
print(prediction_table.to_string(index=False, float_format='%.6f'))

# Calculate accuracy
ensemble_accuracy = np.mean(ensemble_pred == y)
print(f"\nEnsemble accuracy: {ensemble_accuracy:.2%}")

# Visualization 1: Weight evolution
plt.figure(figsize=(12, 8))

# Plot weight evolution
x_pos = np.arange(n_samples)
width = 0.25

plt.bar(x_pos - width, w_initial, width, label='Initial Weights', alpha=0.8, color='skyblue')
plt.bar(x_pos, w_after_h1_normalized, width, label='After h1', alpha=0.8, color='lightgreen')
plt.bar(x_pos + width, w_after_h2_normalized, width, label='After h2', alpha=0.8, color='lightcoral')

plt.xlabel('Sample Number')
plt.ylabel('Weight')
plt.title('AdaBoost Sample Weight Evolution')
plt.xticks(x_pos, [f'S{i+1}' for i in range(n_samples)])
plt.legend()
plt.grid(True, alpha=0.3)

# Add weight values on bars
for i in range(n_samples):
    plt.text(i - width, w_initial[i] + 0.01, f'{w_initial[i]:.3f}', 
             ha='center', va='bottom', fontsize=8)
    plt.text(i, w_after_h1_normalized[i] + 0.01, f'{w_after_h1_normalized[i]:.3f}', 
             ha='center', va='bottom', fontsize=8)
    plt.text(i + width, w_after_h2_normalized[i] + 0.01, f'{w_after_h2_normalized[i]:.3f}', 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_weight_evolution.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Decision boundaries and sample positions
plt.figure(figsize=(14, 10))

# Create feature space
x_range = np.linspace(0, 7, 100)

# Plot decision boundaries
plt.axvline(x=3.5, color='red', linestyle='--', linewidth=2, label='h1: x <= 3.5')
plt.axvline(x=2.5, color='blue', linestyle='--', linewidth=2, label='h2: x <= 2.5')
plt.axvline(x=4.5, color='green', linestyle='--', linewidth=2, label='h3: x <= 4.5')

# Plot samples with different colors for labels
positive_samples = X[y == 1]
negative_samples = X[y == -1]

plt.scatter(positive_samples, [0.5] * len(positive_samples), 
           s=200, color='green', marker='o', label='Positive samples (y=+1)', alpha=0.7)
plt.scatter(negative_samples, [0.5] * len(negative_samples), 
           s=200, color='red', marker='s', label='Negative samples (y=-1)', alpha=0.7)

# Add sample labels
for i, (x, label) in enumerate(zip(X, y)):
    plt.annotate(f'S{i+1}\n(x={x}, y={label})', 
                 (x, 0.5), xytext=(0, 20), textcoords='offset points',
                 ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add weight information
weight_text = "Sample Weights After 2 Iterations:\n"
for i in range(n_samples):
    weight_text += f"S{i+1}: {w_after_h2_normalized[i]:.3f}\n"

plt.text(0.02, 0.98, weight_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

plt.xlabel('Feature Value (x)')
plt.ylabel('Decision Space')
plt.title('AdaBoost Decision Boundaries and Sample Positions')
plt.xlim(0, 7)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_decision_boundaries.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Error analysis
plt.figure(figsize=(12, 8))

# Create subplots for error analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Error patterns for each weak learner
error_data = [h1_errors, h2_errors]
learner_names = ['h1 (x <= 3.5)', 'h2 (x <= 2.5)']
colors = ['red', 'blue']

for i, (errors, name, color) in enumerate(zip(error_data, learner_names, colors)):
    correct = (errors == 0).sum()
    incorrect = (errors == 1).sum()
    
    ax1.bar([f'{name}\nCorrect', f'{name}\nIncorrect'], 
             [correct, incorrect], color=color, alpha=0.7, label=name)
    
    # Add count labels
    ax1.text(i*2, correct + 0.1, str(correct), ha='center', va='bottom', fontweight='bold')
    ax1.text(i*2 + 1, incorrect + 0.1, str(incorrect), ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Number of Samples')
ax1.set_title('Error Analysis: Correct vs Incorrect Classifications')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Weighted error comparison
weighted_errors = [h1_weighted_error, h2_weighted_error]
x_pos = np.arange(len(learner_names))

bars = ax2.bar(x_pos, weighted_errors, color=['red', 'blue'], alpha=0.7)
ax2.set_xlabel('Weak Learner')
ax2.set_ylabel('Weighted Error (ε)')
ax2.set_title('Weighted Error Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(learner_names)
ax2.grid(True, alpha=0.3)

# Add error values on bars
for bar, error in zip(bars, weighted_errors):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{error:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_error_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Alpha values and their impact
plt.figure(figsize=(10, 6))

alphas = [h1_alpha, h2_alpha]
learner_names = ['h1', 'h2']

bars = plt.bar(learner_names, alphas, color=['red', 'blue'], alpha=0.7)
plt.xlabel('Weak Learner')
plt.ylabel('Alpha Value (α)')
plt.title('Alpha Values for Weak Learners')
plt.grid(True, alpha=0.3)

# Add alpha values on bars
for bar, alpha in zip(bars, alphas):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{alpha:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaboost_alpha_values.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1. Initial weights: All samples start with equal weight {1/n_samples:.6f}")
print(f"2. After h1: Sample weights updated based on h1's performance")
print(f"3. After h2: Sample weights further updated based on h2's performance")
print(f"4. Highest weight sample: Sample {max_weight_idx + 1} with weight {w_after_h2_normalized[max_weight_idx]:.6f}")
print(f"5. Final ensemble accuracy: {ensemble_accuracy:.2%}")
print(f"6. Alpha values: α₁ = {h1_alpha:.6f}, α₂ = {h2_alpha:.6f}")
