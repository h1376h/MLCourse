import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 20: ALGORITHM COMPARISON - PERCEPTRON vs MAX MARGIN SVM")
print("=" * 80)

# Given classifier parameters
w_P = np.array([2, 1])    # Perceptron weight vector
b_P = -3                  # Perceptron bias
w_M = np.array([1, 0.5])  # Max Margin SVM weight vector
b_M = -1.5                # Max Margin SVM bias

print("Given classifiers:")
print(f"Perceptron: w_P = {w_P}, b_P = {b_P}")
print(f"Max Margin: w_M = {w_M}, b_M = {b_M}")

# Test point
test_point = np.array([4, 2])
print(f"\nTest point: x = {test_point}")

print("\n" + "="*80)
print("STEP 1: CALCULATE AND COMPARE MARGIN WIDTHS")
print("="*80)

# Calculate margin widths
# Margin width = 2 / ||w||
margin_P = 2 / np.linalg.norm(w_P)
margin_M = 2 / np.linalg.norm(w_M)

print("Margin width formula: 2 / ||w||")
print(f"\nPerceptron:")
print(f"  ||w_P|| = ||{w_P}|| = √({w_P[0]}² + {w_P[1]}²) = √{w_P[0]**2 + w_P[1]**2} = {np.linalg.norm(w_P):.4f}")
print(f"  Margin width = 2 / {np.linalg.norm(w_P):.4f} = {margin_P:.4f}")

print(f"\nMax Margin SVM:")
print(f"  ||w_M|| = ||{w_M}|| = √({w_M[0]}² + {w_M[1]}²) = √{w_M[0]**2 + w_M[1]**2} = {np.linalg.norm(w_M):.4f}")
print(f"  Margin width = 2 / {np.linalg.norm(w_M):.4f} = {margin_M:.4f}")

print(f"\nComparison:")
print(f"  Max Margin SVM has {margin_M/margin_P:.2f}x wider margin than Perceptron")
print(f"  Margin difference: {margin_M - margin_P:.4f}")

print("\n" + "="*80)
print("STEP 2: COMPUTE DECISION VALUES FOR TEST POINT")
print("="*80)

# Compute decision values f(x) = w^T x + b
f_P = np.dot(w_P, test_point) + b_P
f_M = np.dot(w_M, test_point) + b_M

print(f"Decision function: f(x) = w^T x + b")
print(f"\nPerceptron:")
print(f"  f_P(x) = w_P^T x + b_P")
print(f"  f_P({test_point}) = {w_P} · {test_point} + {b_P}")
print(f"  f_P({test_point}) = {w_P[0]}×{test_point[0]} + {w_P[1]}×{test_point[1]} + {b_P}")
print(f"  f_P({test_point}) = {w_P[0]*test_point[0]} + {w_P[1]*test_point[1]} + {b_P}")
print(f"  f_P({test_point}) = {f_P}")

print(f"\nMax Margin SVM:")
print(f"  f_M(x) = w_M^T x + b_M")
print(f"  f_M({test_point}) = {w_M} · {test_point} + {b_M}")
print(f"  f_M({test_point}) = {w_M[0]}×{test_point[0]} + {w_M[1]}×{test_point[1]} + {b_M}")
print(f"  f_M({test_point}) = {w_M[0]*test_point[0]} + {w_M[1]*test_point[1]} + {b_M}")
print(f"  f_M({test_point}) = {f_M}")

# Determine classifications
class_P = 1 if f_P > 0 else -1
class_M = 1 if f_M > 0 else -1

print(f"\nClassifications:")
print(f"  Perceptron: sign({f_P}) = {class_P}")
print(f"  Max Margin: sign({f_M}) = {class_M}")

print("\n" + "="*80)
print("STEP 3: NORMALIZE WEIGHT VECTORS AND RECALCULATE")
print("="*80)

# Normalize weight vectors
w_P_norm = w_P / np.linalg.norm(w_P)
w_M_norm = w_M / np.linalg.norm(w_M)

# Normalize bias terms accordingly
b_P_norm = b_P / np.linalg.norm(w_P)
b_M_norm = b_M / np.linalg.norm(w_M)

print("Normalized weight vectors (unit length):")
print(f"Step-by-step normalization:")
print(f"  ||w_P|| = √({w_P[0]}² + {w_P[1]}²) = √({w_P[0]**2} + {w_P[1]**2}) = √{w_P[0]**2 + w_P[1]**2} = {np.linalg.norm(w_P):.4f}")
print(f"  w_P_normalized = ({w_P[0]}, {w_P[1]}) / {np.linalg.norm(w_P):.4f}")
print(f"                 = ({w_P[0]}/{np.linalg.norm(w_P):.4f}, {w_P[1]}/{np.linalg.norm(w_P):.4f})")
print(f"                 = ({w_P_norm[0]:.6f}, {w_P_norm[1]:.6f})")

print(f"\n  ||w_M|| = √({w_M[0]}² + {w_M[1]}²) = √({w_M[0]**2} + {w_M[1]**2}) = √{w_M[0]**2 + w_M[1]**2} = {np.linalg.norm(w_M):.4f}")
print(f"  w_M_normalized = ({w_M[0]}, {w_M[1]}) / {np.linalg.norm(w_M):.4f}")
print(f"                 = ({w_M[0]}/{np.linalg.norm(w_M):.4f}, {w_M[1]}/{np.linalg.norm(w_M):.4f})")
print(f"                 = ({w_M_norm[0]:.6f}, {w_M_norm[1]:.6f})")

print(f"\nNormalized bias terms:")
print(f"  b_P_normalized = {b_P} / {np.linalg.norm(w_P):.4f} = {b_P_norm:.4f}")
print(f"  b_M_normalized = {b_M} / {np.linalg.norm(w_M):.4f} = {b_M_norm:.4f}")

# Recalculate decision values with normalized vectors
f_P_norm = np.dot(w_P_norm, test_point) + b_P_norm
f_M_norm = np.dot(w_M_norm, test_point) + b_M_norm

print(f"\nNormalized decision values (step-by-step):")
print(f"  f_P_norm({test_point}) = w_P_norm · x + b_P_norm")
print(f"                        = ({w_P_norm[0]:.6f}, {w_P_norm[1]:.6f}) · ({test_point[0]}, {test_point[1]}) + {b_P_norm:.4f}")
print(f"                        = {w_P_norm[0]:.6f}×{test_point[0]} + {w_P_norm[1]:.6f}×{test_point[1]} + {b_P_norm:.4f}")
print(f"                        = {w_P_norm[0]*test_point[0]:.6f} + {w_P_norm[1]*test_point[1]:.6f} + {b_P_norm:.4f}")
print(f"                        = {f_P_norm:.4f}")

print(f"\n  f_M_norm({test_point}) = w_M_norm · x + b_M_norm")
print(f"                        = ({w_M_norm[0]:.6f}, {w_M_norm[1]:.6f}) · ({test_point[0]}, {test_point[1]}) + {b_M_norm:.4f}")
print(f"                        = {w_M_norm[0]:.6f}×{test_point[0]} + {w_M_norm[1]:.6f}×{test_point[1]} + {b_M_norm:.4f}")
print(f"                        = {w_M_norm[0]*test_point[0]:.6f} + {w_M_norm[1]*test_point[1]:.6f} + {b_M_norm:.4f}")
print(f"                        = {f_M_norm:.4f}")

print(f"\nNote: Normalized decision values represent signed distances to hyperplanes")
print(f"Key insight: Both normalized values are identical! This means both")
print(f"classifiers represent the same geometric hyperplane, just with different scaling.")

print("\n" + "="*80)
print("STEP 4: CONFIDENCE COMPARISON")
print("="*80)

print("Confidence analysis:")
print(f"\nOriginal decision values:")
print(f"  |f_P(x)| = |{f_P}| = {abs(f_P):.4f}")
print(f"  |f_M(x)| = |{f_M}| = {abs(f_M):.4f}")

print(f"\nNormalized decision values (signed distances):")
print(f"  |f_P_norm(x)| = |{f_P_norm:.4f}| = {abs(f_P_norm):.4f}")
print(f"  |f_M_norm(x)| = |{f_M_norm:.4f}| = {abs(f_M_norm):.4f}")

# Determine which is more confident
if abs(f_P_norm) > abs(f_M_norm):
    more_confident = "Perceptron"
    confidence_ratio = abs(f_P_norm) / abs(f_M_norm)
else:
    more_confident = "Max Margin SVM"
    confidence_ratio = abs(f_M_norm) / abs(f_P_norm)

print(f"\nConfidence comparison:")
print(f"  {more_confident} is more confident by a factor of {confidence_ratio:.2f}")
print(f"  The test point is {abs(f_P_norm):.4f} units from Perceptron boundary")
print(f"  The test point is {abs(f_M_norm):.4f} units from Max Margin boundary")

print("\n" + "="*80)
print("STEP 5: GENERALIZATION PERFORMANCE ANALYSIS")
print("="*80)

print("Margin Theory and Generalization:")
print(f"\n1. Margin Width Comparison:")
print(f"   - Perceptron margin: {margin_P:.4f}")
print(f"   - Max Margin SVM: {margin_M:.4f}")
print(f"   - SVM has {margin_M/margin_P:.2f}x larger margin")

print(f"\n2. Generalization Bound (VC Theory):")
print(f"   - Generalization error prop. to 1/margin^2")
print(f"   - Perceptron bound prop. to 1/{margin_P:.4f}^2 = {1/margin_P**2:.2f}")
print(f"   - SVM bound prop. to 1/{margin_M:.4f}^2 = {1/margin_M**2:.2f}")
print(f"   - SVM bound is {(1/margin_P**2)/(1/margin_M**2):.2f}x better")

print(f"\n3. Robustness to Noise:")
print(f"   - Larger margin → more robust to small perturbations")
print(f"   - SVM can tolerate noise up to {margin_M/2:.4f} units")
print(f"   - Perceptron can tolerate noise up to {margin_P/2:.4f} units")

print(f"\n4. Expected Performance:")
print(f"   - Max Margin SVM should generalize better")
print(f"   - Perceptron may overfit to training data")
print(f"   - SVM optimizes for worst-case scenario")

print("\n" + "="*80)
print("STEP 6: VISUALIZATION")
print("="*80)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Common plotting parameters
x_range = np.linspace(-1, 6, 100)
colors = ['blue', 'red']

# Plot 1: Decision boundaries comparison
ax1.set_title('Decision Boundaries Comparison', fontsize=14)

# Plot Perceptron boundary
if abs(w_P[1]) > 1e-10:
    y_P = -(w_P[0] * x_range + b_P) / w_P[1]
    ax1.plot(x_range, y_P, 'b-', linewidth=2, label='Perceptron')

# Plot Max Margin boundary
if abs(w_M[1]) > 1e-10:
    y_M = -(w_M[0] * x_range + b_M) / w_M[1]
    ax1.plot(x_range, y_M, 'r-', linewidth=2, label='Max Margin SVM')

# Plot test point
ax1.scatter(test_point[0], test_point[1], s=200, c='green', marker='*', 
           edgecolors='black', linewidth=2, label=f'Test Point {test_point}')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 6)
ax1.set_ylim(-1, 4)

# Plot 2: Margin visualization
ax2.set_title('Margin Width Comparison', fontsize=14)

# Plot boundaries with margins
if abs(w_P[1]) > 1e-10:
    y_P = -(w_P[0] * x_range + b_P) / w_P[1]
    # Margin boundaries for Perceptron
    margin_offset_P = margin_P / 2
    normal_P = w_P / np.linalg.norm(w_P)
    y_P_upper = -(w_P[0] * x_range + (b_P - np.linalg.norm(w_P))) / w_P[1]
    y_P_lower = -(w_P[0] * x_range + (b_P + np.linalg.norm(w_P))) / w_P[1]
    
    ax2.plot(x_range, y_P, 'b-', linewidth=2, label='Perceptron')
    ax2.plot(x_range, y_P_upper, 'b--', linewidth=1, alpha=0.7)
    ax2.plot(x_range, y_P_lower, 'b--', linewidth=1, alpha=0.7)
    ax2.fill_between(x_range, y_P_upper, y_P_lower, alpha=0.2, color='blue')

if abs(w_M[1]) > 1e-10:
    y_M = -(w_M[0] * x_range + b_M) / w_M[1]
    # Margin boundaries for SVM
    y_M_upper = -(w_M[0] * x_range + (b_M - np.linalg.norm(w_M))) / w_M[1]
    y_M_lower = -(w_M[0] * x_range + (b_M + np.linalg.norm(w_M))) / w_M[1]
    
    ax2.plot(x_range, y_M, 'r-', linewidth=2, label='Max Margin SVM')
    ax2.plot(x_range, y_M_upper, 'r--', linewidth=1, alpha=0.7)
    ax2.plot(x_range, y_M_lower, 'r--', linewidth=1, alpha=0.7)
    ax2.fill_between(x_range, y_M_upper, y_M_lower, alpha=0.2, color='red')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 6)
ax2.set_ylim(-1, 4)

# Add margin width annotations
ax2.text(1, 3, f'Perceptron\nMargin: {margin_P:.3f}', 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))
ax2.text(4, 1, f'SVM\nMargin: {margin_M:.3f}', 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.8))

# Plot 3: Decision values comparison
ax3.set_title('Decision Values for Test Point', fontsize=14)

methods = ['Perceptron', 'Max Margin']
original_values = [f_P, f_M]
normalized_values = [f_P_norm, f_M_norm]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, original_values, width, label='Original f(x)', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, normalized_values, width, label='Normalized f(x)', alpha=0.8)

ax3.set_xlabel('Classifier')
ax3.set_ylabel('Decision Value')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax3.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1,
             f'{height1:.2f}', ha='center', va='bottom')
    ax3.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1,
             f'{height2:.3f}', ha='center', va='bottom')

# Plot 4: Generalization comparison
ax4.set_title('Generalization Performance Comparison', fontsize=14)

# Create a conceptual plot showing generalization bounds
margins = [margin_P, margin_M]
gen_bounds = [1/m**2 for m in margins]
gen_bounds_normalized = [b/max(gen_bounds) for b in gen_bounds]

bars = ax4.bar(methods, gen_bounds_normalized, color=['blue', 'red'], alpha=0.7)
ax4.set_ylabel('Relative Generalization Bound')
ax4.set_title(r'Lower is Better ($\propto 1/margin^2$)')

# Add value labels
for bar, bound in zip(bars, gen_bounds):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{bound:.1f}', ha='center', va='bottom')

ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure instead of showing it

print(f"Visualization saved to: {save_dir}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Margin widths: Perceptron = {margin_P:.4f}, SVM = {margin_M:.4f}")
print(f"✓ Decision values: f_P = {f_P}, f_M = {f_M}")
print(f"✓ Normalized distances: {abs(f_P_norm):.4f} (P), {abs(f_M_norm):.4f} (SVM)")
print(f"✓ More confident: {more_confident}")
print(f"✓ Better generalization expected: Max Margin SVM (larger margin)")
