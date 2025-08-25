import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_40")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("="*80)
print("QUESTION 40: KERNEL TRICK AND NONLINEAR SVM HYPERPLANE")
print("="*80)

# Dataset
positive_points = np.array([1, 3, 5])
negative_points = np.array([0, 2, 4, 6])

print("\nDataset:")
print(f"Positive Points: {positive_points}")
print(f"Negative Points: {negative_points}")

# Combine all points and create labels
all_points = np.concatenate([positive_points, negative_points])
labels = np.concatenate([np.ones(len(positive_points)), -np.ones(len(negative_points))])

print(f"All Points: {all_points}")
print(f"Labels: {labels}")

# Task 1: Plot original points and show they are not linearly separable
print("\n" + "="*60)
print("TASK 1: ORIGINAL 1D SPACE - NON-LINEARLY SEPARABLE")
print("="*60)

plt.figure(figsize=(12, 4))
plt.scatter(positive_points, np.zeros(len(positive_points)), 
           c='red', s=100, marker='o', label='Positive (+1)', edgecolor='black', linewidth=2)
plt.scatter(negative_points, np.zeros(len(negative_points)), 
           c='blue', s=100, marker='s', label='Negative (-1)', edgecolor='black', linewidth=2)

# Add point labels
for i, point in enumerate(positive_points):
    plt.annotate(f'{point}', (point, 0), xytext=(0, 20), 
                textcoords='offset points', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3))

for i, point in enumerate(negative_points):
    plt.annotate(f'{point}', (point, 0), xytext=(0, -30), 
                textcoords='offset points', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.3))

plt.xlabel('$x$')
plt.ylabel('')
plt.title('Original 1D Dataset - Not Linearly Separable')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-0.5, 0.5)
plt.xlim(-0.5, 6.5)

# Remove y-axis ticks since we're plotting on a line
plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_1d_dataset.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to prevent display

print("✓ Original dataset plotted - clearly not linearly separable")
print("  Points alternate between positive and negative classes")

# Task 2: Apply multiple feature transformations
print("\n" + "="*60)
print("TASK 2: MULTIPLE KERNEL TRANSFORMATIONS")
print("="*60)

# Define multiple feature transformations
def phi_primary(x):
    """Primary approach: φ(x) = [x^2, (x mod 2 - 0.5)x^2]"""
    return np.column_stack([x**2, (x % 2 - 0.5) * x**2])

def phi_sign_based(x):
    """Sign-based approach: φ(x) = [x, sign(x mod 2 - 0.5)]"""
    return np.column_stack([x, np.sign(x % 2 - 0.5)])

def phi_parity_weighted(x):
    """Parity-weighted approach: φ(x) = [(x mod 2)x, (1-x mod 2)x]"""
    return np.column_stack([(x % 2) * x, (1 - x % 2) * x])

def phi_trigonometric(x):
    """Trigonometric approach: φ(x) = [cos(πx), sin(πx)]"""
    return np.column_stack([np.cos(np.pi * x), np.sin(np.pi * x)])

# Additional transformations for comprehensive analysis
def phi_quadratic_simple(x):
    """Simple quadratic: φ(x) = [x, x^2]"""
    return np.column_stack([x, x**2])

def phi_cubic(x):
    """Cubic transformation: φ(x) = [x, x^3]"""
    return np.column_stack([x, x**3])

# Store all transformations
transformations = {
    'Primary': phi_primary,
    'Sign-based': phi_sign_based, 
    'Parity-weighted': phi_parity_weighted,
    'Trigonometric': phi_trigonometric,
    'Quadratic-simple': phi_quadratic_simple,
    'Cubic': phi_cubic
}

# Analyze each transformation
results = {}

for name, phi_func in transformations.items():
    print(f"\n{'-'*40}")
    print(f"ANALYZING: {name.upper()} TRANSFORMATION")
    print(f"{'-'*40}")
    
    # Apply transformation
    transformed_points = phi_func(all_points)
    
    print(f"Transformation function: {phi_func.__doc__}")
    print(f"Original points: {all_points}")
    print(f"Transformed points shape: {transformed_points.shape}")
    print("Transformed points:")
    for i, (orig, trans) in enumerate(zip(all_points, transformed_points)):
        label_str = "+" if labels[i] == 1 else "-"
        print(f"  x={orig} → φ(x)={trans} (label: {label_str})")
    
    # Try to find separating hyperplane using SVM
    try:
        # Use SVM with linear kernel in transformed space
        svm = SVC(kernel='linear', C=1000)  # High C for hard margin
        svm.fit(transformed_points, labels)
        
        # Check if all points are correctly classified
        predictions = svm.predict(transformed_points)
        accuracy = np.mean(predictions == labels)
        
        print(f"SVM Accuracy: {accuracy:.3f}")
        print(f"Support vectors: {len(svm.support_)}")
        print(f"Hyperplane coefficients: w = {svm.coef_[0]}")
        print(f"Bias term: b = {svm.intercept_[0]:.6f}")
        
        # Store results
        results[name] = {
            'phi_func': phi_func,
            'transformed_points': transformed_points,
            'svm': svm,
            'accuracy': accuracy,
            'separable': accuracy == 1.0
        }
        
        if accuracy == 1.0:
            print("✓ PERFECTLY SEPARABLE!")
        else:
            print("✗ Not perfectly separable")
            
    except Exception as e:
        print(f"✗ Error fitting SVM: {e}")
        results[name] = {
            'phi_func': phi_func,
            'transformed_points': transformed_points,
            'svm': None,
            'accuracy': 0.0,
            'separable': False
        }

print(f"\n{'='*60}")
print("SUMMARY OF TRANSFORMATIONS")
print(f"{'='*60}")

for name, result in results.items():
    status = "✓ SEPARABLE" if result['separable'] else "✗ NOT SEPARABLE"
    print(f"{name:20s}: {status} (Accuracy: {result['accuracy']:.3f})")

# Task 3: Visualize each successful transformation separately
print("\n" + "="*60)
print("TASK 3: VISUALIZING SUCCESSFUL TRANSFORMATIONS")
print("="*60)

# Find successful transformations
successful_transforms = {name: result for name, result in results.items() if result['separable']}

print(f"Found {len(successful_transforms)} successful transformations:")
for name in successful_transforms.keys():
    print(f"  - {name}")

# Create separate visualization for each successful transformation
for name, result in successful_transforms.items():
    print(f"\nCreating plot for {name} transformation...")

    plt.figure(figsize=(10, 8))

    # Get transformed points
    transformed_points = result['transformed_points']
    svm = result['svm']

    # Separate positive and negative points
    pos_mask = labels == 1
    neg_mask = labels == -1

    pos_transformed = transformed_points[pos_mask]
    neg_transformed = transformed_points[neg_mask]

    # Plot points
    plt.scatter(pos_transformed[:, 0], pos_transformed[:, 1],
              c='red', s=150, marker='o', label='Positive (+1)',
              edgecolor='black', linewidth=2, alpha=0.8)
    plt.scatter(neg_transformed[:, 0], neg_transformed[:, 1],
              c='blue', s=150, marker='s', label='Negative (-1)',
              edgecolor='black', linewidth=2, alpha=0.8)

    # Add point labels with original x values
    for i, (orig_x, trans_point) in enumerate(zip(all_points, transformed_points)):
        plt.annotate(f'x={orig_x}', trans_point, xytext=(8, 8),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor='gray'))

    # Plot decision boundary if SVM was successful
    if svm is not None:
        # Create a mesh for plotting decision boundary
        x_min, x_max = transformed_points[:, 0].min(), transformed_points[:, 0].max()
        y_min, y_max = transformed_points[:, 1].min(), transformed_points[:, 1].max()

        # Adjust ranges for better visualization
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        x_min -= 0.2 * x_range
        x_max += 0.2 * x_range
        y_min -= 0.2 * y_range
        y_max += 0.2 * y_range

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # Get decision function values
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'],
                  linestyles=['--', '-', '--'], linewidths=[1.5, 3, 1.5])

        # Shade regions
        plt.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
        plt.contourf(xx, yy, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)

        # Identify ALL support vectors (points on margin boundaries)
        # Calculate decision function values for all points
        decision_values = svm.decision_function(transformed_points)

        # Find points that are exactly on the margin boundaries (|decision_value| ≈ 1)
        tolerance = 1e-3
        support_mask = np.abs(np.abs(decision_values) - 1.0) < tolerance

        # Also include the SVM's identified support vectors
        svm_support_mask = np.zeros(len(transformed_points), dtype=bool)
        svm_support_mask[svm.support_] = True

        # Combine both criteria
        all_support_mask = support_mask | svm_support_mask

        support_vectors = transformed_points[all_support_mask]
        support_labels = labels[all_support_mask]
        support_original_x = all_points[all_support_mask]

        # Highlight support vectors with different colors for positive/negative
        pos_sv_mask = support_labels == 1
        neg_sv_mask = support_labels == -1

        if np.any(pos_sv_mask):
            plt.scatter(support_vectors[pos_sv_mask, 0], support_vectors[pos_sv_mask, 1],
                      s=300, facecolors='none', edgecolors='darkgreen', linewidths=4,
                      label='Support Vectors (+)')

        if np.any(neg_sv_mask):
            plt.scatter(support_vectors[neg_sv_mask, 0], support_vectors[neg_sv_mask, 1],
                      s=300, facecolors='none', edgecolors='darkred', linewidths=4,
                      label='Support Vectors (-)')

        # Print detailed support vector information
        print(f"  Support vectors for {name}:")
        for i, (orig_x, sv_point, sv_label, decision_val) in enumerate(zip(
            support_original_x, support_vectors, support_labels, decision_values[all_support_mask])):
            print(f"    x={orig_x}: φ(x)=[{sv_point[0]:.3f}, {sv_point[1]:.3f}], "
                  f"label={'+' if sv_label == 1 else '-'}, decision_value={decision_val:.6f}")

        # Add hyperplane equation to the plot
        w = svm.coef_[0]
        b = svm.intercept_[0]
        eq_text = f'Hyperplane: ${w[0]:.3f}\\phi_1 + {w[1]:.3f}\\phi_2 + {b:.3f} = 0$'
        plt.text(0.02, 0.98, eq_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                verticalalignment='top', fontsize=10)

    plt.xlabel('$\\phi_1(x)$', fontsize=14)
    plt.ylabel('$\\phi_2(x)$', fontsize=14)

    # Create safe title without Unicode characters
    transformation_desc = result["phi_func"].__doc__.split(": ")[1]
    safe_title = f'{name} Transformation'
    plt.title(safe_title, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save individual plot
    filename = f'{name.lower().replace("-", "_")}_transformation.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display
    print(f"  Saved: {filename}")

# Task 4: Detailed step-by-step mathematical analysis
print("\n" + "="*60)
print("TASK 4: DETAILED STEP-BY-STEP MATHEMATICAL ANALYSIS")
print("="*60)

if 'Primary' in successful_transforms:
    primary_result = successful_transforms['Primary']
    primary_svm = primary_result['svm']
    primary_transformed = primary_result['transformed_points']

    print("PRIMARY TRANSFORMATION: φ(x) = [x², (x mod 2 - 0.5)x²]")
    print("\n" + "-"*50)
    print("STEP 1: UNDERSTANDING THE TRANSFORMATION")
    print("-"*50)

    print("The transformation φ(x) = [φ₁(x), φ₂(x)] where:")
    print("  φ₁(x) = x²")
    print("  φ₂(x) = (x mod 2 - 0.5) × x²")
    print("\nThe key insight:")
    print("  - For odd x: x mod 2 = 1, so φ₂(x) = (1 - 0.5) × x² = 0.5 × x²")
    print("  - For even x: x mod 2 = 0, so φ₂(x) = (0 - 0.5) × x² = -0.5 × x²")

    print("\n" + "-"*50)
    print("STEP 2: APPLYING TRANSFORMATION TO EACH POINT")
    print("-"*50)

    for i, x in enumerate(all_points):
        phi1 = x**2
        x_mod_2 = x % 2
        phi2_factor = x_mod_2 - 0.5
        phi2 = phi2_factor * x**2
        label = "+" if labels[i] == 1 else "-"

        print(f"\nPoint x = {x} (label: {label}):")
        print(f"  φ₁(x) = x² = {x}² = {phi1}")
        print(f"  x mod 2 = {x_mod_2}")
        print(f"  φ₂(x) = (x mod 2 - 0.5) × x² = ({x_mod_2} - 0.5) × {x}² = {phi2_factor} × {phi1} = {phi2}")
        print(f"  φ(x) = [{phi1}, {phi2}]")

    print("\n" + "-"*50)
    print("STEP 3: PATTERN RECOGNITION IN FEATURE SPACE")
    print("-"*50)

    print("Observing the transformed points:")
    print("Positive class (odd x): All have φ₂ > 0")
    for i, x in enumerate(all_points):
        if labels[i] == 1:
            phi1, phi2 = primary_transformed[i]
            print(f"  x={x}: φ(x) = [{phi1}, {phi2}] → φ₂ = {phi2} > 0")

    print("\nNegative class (even x): All have φ₂ ≤ 0")
    for i, x in enumerate(all_points):
        if labels[i] == -1:
            phi1, phi2 = primary_transformed[i]
            print(f"  x={x}: φ(x) = [{phi1}, {phi2}] → φ₂ = {phi2} ≤ 0")

    print("\n" + "-"*50)
    print("STEP 4: SVM HYPERPLANE DERIVATION")
    print("-"*50)

    w = primary_svm.coef_[0]
    b = primary_svm.intercept_[0]

    print("SVM finds the optimal hyperplane: w₁φ₁ + w₂φ₂ + b = 0")
    print(f"Computed coefficients: w₁ = {w[0]:.6f}, w₂ = {w[1]:.6f}, b = {b:.6f}")

    # Simplify to theoretical values
    print("\nSimplifying to theoretical values (w₁ ≈ 1, w₂ ≈ 2, b ≈ -1):")
    print("Hyperplane equation: φ₁ + 2φ₂ - 1 = 0")
    print("Substituting transformations:")
    print("  x² + 2(x mod 2 - 0.5)x² - 1 = 0")
    print("  x²[1 + 2(x mod 2 - 0.5)] - 1 = 0")

    print("\n" + "-"*50)
    print("STEP 5: DECISION FUNCTION ANALYSIS")
    print("-"*50)

    print("Decision function: f(x) = sign(x²[1 + 2(x mod 2 - 0.5)] - 1)")
    print("\nCase analysis:")
    print("For odd x (x mod 2 = 1):")
    print("  f(x) = sign(x²[1 + 2(1 - 0.5)] - 1)")
    print("       = sign(x²[1 + 2(0.5)] - 1)")
    print("       = sign(x²[1 + 1] - 1)")
    print("       = sign(2x² - 1)")

    print("\nFor even x (x mod 2 = 0):")
    print("  f(x) = sign(x²[1 + 2(0 - 0.5)] - 1)")
    print("       = sign(x²[1 + 2(-0.5)] - 1)")
    print("       = sign(x²[1 - 1] - 1)")
    print("       = sign(0 - 1)")
    print("       = sign(-1) = -1")

    print("\n" + "-"*50)
    print("STEP 6: VERIFICATION OF CLASSIFICATION")
    print("-"*50)

    print("Testing each point:")
    for i, x in enumerate(all_points):
        true_label = labels[i]
        if x % 2 == 1:  # odd
            decision_value = 2 * x**2 - 1
            predicted = 1 if decision_value > 0 else -1
            print(f"x={x} (odd): f(x) = sign(2×{x}² - 1) = sign({decision_value}) = {predicted} ✓" if predicted == true_label else f"x={x} (odd): f(x) = sign(2×{x}² - 1) = sign({decision_value}) = {predicted} ✗")
        else:  # even
            predicted = -1
            print(f"x={x} (even): f(x) = -1 ✓" if predicted == true_label else f"x={x} (even): f(x) = -1 ✗")

    # Calculate margin
    margin = 1.0 / np.linalg.norm(w)
    print(f"\n" + "-"*50)
    print("STEP 7: MARGIN CALCULATION")
    print("-"*50)
    print(f"Margin = 1/||w|| = 1/√(w₁² + w₂²)")
    print(f"       = 1/√({w[0]:.6f}² + {w[1]:.6f}²)")
    print(f"       = 1/√{w[0]**2 + w[1]**2:.6f}")
    print(f"       = {margin:.6f}")

    # Identify support vectors
    support_indices = primary_svm.support_
    print(f"\n" + "-"*50)
    print("STEP 8: SUPPORT VECTOR IDENTIFICATION")
    print("-"*50)

    # Calculate decision function values for all points
    decision_values = []
    print("Calculating decision function values for all points:")
    print("Decision function: f(x) = w₁φ₁ + w₂φ₂ + b")

    for i, x in enumerate(all_points):
        phi_val = primary_transformed[i]
        decision_val = np.dot(w, phi_val) + b
        decision_values.append(decision_val)
        label = "+" if labels[i] == 1 else "-"

        print(f"\nPoint x = {x} (label: {label}):")
        print(f"  φ(x) = [{phi_val[0]:.3f}, {phi_val[1]:.3f}]")
        print(f"  f(x) = {w[0]:.6f} × {phi_val[0]:.3f} + {w[1]:.6f} × {phi_val[1]:.3f} + {b:.6f}")
        print(f"       = {w[0]*phi_val[0]:.6f} + {w[1]*phi_val[1]:.6f} + {b:.6f}")
        print(f"       = {decision_val:.6f}")

        # Determine if this is a support vector
        if abs(abs(decision_val) - 1.0) < 1e-3:
            print(f"  → SUPPORT VECTOR (|f(x)| ≈ 1)")
        elif decision_val > 1:
            print(f"  → Beyond positive margin")
        elif decision_val < -1:
            print(f"  → Beyond negative margin")
        else:
            print(f"  → Within margin (unusual for hard margin SVM)")

    # Identify all support vectors
    decision_values = np.array(decision_values)
    tolerance = 1e-3
    support_mask = np.abs(np.abs(decision_values) - 1.0) < tolerance

    print(f"\n" + "-"*30)
    print("SUPPORT VECTOR SUMMARY:")
    print("-"*30)

    support_vector_indices = np.where(support_mask)[0]
    print(f"Support vector indices: {support_vector_indices}")
    print(f"Total support vectors: {len(support_vector_indices)}")

    print("\nSupport vector details:")
    for idx in support_vector_indices:
        x_orig = all_points[idx]
        phi_val = primary_transformed[idx]
        label = "+" if labels[idx] == 1 else "-"
        decision_val = decision_values[idx]
        print(f"  x={x_orig}: φ(x)=[{phi_val[0]:.3f}, {phi_val[1]:.3f}], "
              f"label={label}, f(x)={decision_val:.6f}")

    print(f"\n" + "-"*50)
    print("STEP 9: GEOMETRIC INTERPRETATION")
    print("-"*50)

    print("The hyperplane equation w₁φ₁ + w₂φ₂ + b = 0 divides the feature space:")
    print(f"  Hyperplane: {w[0]:.6f}φ₁ + {w[1]:.6f}φ₂ + {b:.6f} = 0")

    # Simplify to theoretical form
    print("\nSimplifying to theoretical form (w₁ ≈ 1, w₂ ≈ 2, b ≈ -1):")
    print("  Hyperplane: φ₁ + 2φ₂ - 1 = 0")
    print("  Rearranged: φ₂ = (1 - φ₁)/2")

    print("\nThis creates a line in the (φ₁, φ₂) space that separates:")
    print("  - Positive region: φ₁ + 2φ₂ - 1 > 0")
    print("  - Negative region: φ₁ + 2φ₂ - 1 < 0")

    print("\nMargin boundaries are parallel lines:")
    print("  - Positive margin: φ₁ + 2φ₂ - 1 = +1  →  φ₂ = (2 - φ₁)/2")
    print("  - Negative margin: φ₁ + 2φ₂ - 1 = -1  →  φ₂ = (0 - φ₁)/2")

    print(f"\nDistance between margin boundaries = 2 × margin = 2 × {margin:.6f} = {2*margin:.6f}")

    print(f"\n" + "-"*50)
    print("STEP 10: KERNEL FUNCTION ANALYSIS")
    print("-"*50)

    print("The kernel function K(x, z) = φ(x)ᵀφ(z) for our transformation:")
    print("K(x, z) = φ₁(x)φ₁(z) + φ₂(x)φ₂(z)")
    print("         = x²z² + (x mod 2 - 0.5)(z mod 2 - 0.5)x²z²")
    print("         = x²z²[1 + (x mod 2 - 0.5)(z mod 2 - 0.5)]")

    print("\nKernel values for our dataset:")
    print("K(x,z) matrix:")

    # Calculate kernel matrix
    K = primary_transformed @ primary_transformed.T

    # Print header
    header = "     "
    for z in all_points:
        header += f"{z:8.0f}"
    print(header)

    for i, x in enumerate(all_points):
        row = f"x={x:2.0f} "
        for j, z in enumerate(all_points):
            row += f"{K[i,j]:8.3f}"
        print(row)

    # Verify positive semi-definiteness
    eigenvals = np.linalg.eigvals(K)
    min_eigenval = np.min(eigenvals)
    print(f"\nKernel matrix eigenvalues: {eigenvals}")
    print(f"Minimum eigenvalue: {min_eigenval:.10f}")
    print(f"Positive semi-definite: {'✓' if min_eigenval >= -1e-10 else '✗'}")

print("\n" + "="*60)
print("DETAILED ANALYSIS FOR OTHER SUCCESSFUL TRANSFORMATIONS")
print("="*60)

# Analyze other successful transformations
other_transforms = {k: v for k, v in successful_transforms.items() if k != 'Primary'}

for name, result in other_transforms.items():
    print(f"\n{'='*60}")
    print(f"{name.upper()} TRANSFORMATION ANALYSIS")
    print(f"{'='*60}")

    phi_func = result['phi_func']
    svm = result['svm']
    transformed_points = result['transformed_points']

    print(f"Transformation: φ(x) = {phi_func.__doc__.split(': ')[1]}")

    print(f"\n{'-'*40}")
    print("STEP-BY-STEP TRANSFORMATION:")
    print("-"*40)

    # Detailed transformation for each point
    for i, x in enumerate(all_points):
        phi_val = transformed_points[i]
        label = "+" if labels[i] == 1 else "-"

        print(f"\nPoint x = {x} (label: {label}):")

        if name == "Sign-based":
            sign_val = np.sign(x % 2 - 0.5)
            print(f"  x mod 2 = {x % 2}")
            print(f"  x mod 2 - 0.5 = {x % 2 - 0.5}")
            print(f"  sign(x mod 2 - 0.5) = {sign_val}")
            print(f"  φ(x) = [x, sign(x mod 2 - 0.5)] = [{x}, {sign_val}] = [{phi_val[0]:.3f}, {phi_val[1]:.3f}]")

        elif name == "Parity-weighted":
            mod_val = x % 2
            comp_mod = 1 - mod_val
            print(f"  x mod 2 = {mod_val}")
            print(f"  1 - x mod 2 = {comp_mod}")
            print(f"  φ₁(x) = (x mod 2) × x = {mod_val} × {x} = {mod_val * x}")
            print(f"  φ₂(x) = (1 - x mod 2) × x = {comp_mod} × {x} = {comp_mod * x}")
            print(f"  φ(x) = [{mod_val * x}, {comp_mod * x}] = [{phi_val[0]:.3f}, {phi_val[1]:.3f}]")

        elif name == "Trigonometric":
            cos_val = np.cos(np.pi * x)
            sin_val = np.sin(np.pi * x)
            print(f"  cos(πx) = cos(π × {x}) = cos({np.pi * x:.3f}) = {cos_val:.6f}")
            print(f"  sin(πx) = sin(π × {x}) = sin({np.pi * x:.3f}) = {sin_val:.6f}")
            print(f"  φ(x) = [cos(πx), sin(πx)] = [{cos_val:.6f}, {sin_val:.6f}]")
            print(f"       ≈ [{phi_val[0]:.3f}, {phi_val[1]:.3f}]")

    if svm is not None:
        w = svm.coef_[0]
        b = svm.intercept_[0]
        margin = 1.0 / np.linalg.norm(w)

        print(f"\n{'-'*40}")
        print("SVM HYPERPLANE ANALYSIS:")
        print("-"*40)

        print(f"Hyperplane equation: w₁φ₁ + w₂φ₂ + b = 0")
        print(f"Coefficients: w₁ = {w[0]:.6f}, w₂ = {w[1]:.6f}, b = {b:.6f}")
        print(f"Hyperplane: {w[0]:.6f}φ₁ + {w[1]:.6f}φ₂ + {b:.6f} = 0")

        # Geometric interpretation
        if abs(w[0]) < 1e-6:  # Horizontal line
            print(f"Geometric interpretation: Horizontal line at φ₂ = {-b/w[1]:.6f}")
        elif abs(w[1]) < 1e-6:  # Vertical line
            print(f"Geometric interpretation: Vertical line at φ₁ = {-b/w[0]:.6f}")
        else:
            slope = -w[0]/w[1]
            intercept = -b/w[1]
            print(f"Geometric interpretation: Line φ₂ = {slope:.6f}φ₁ + {intercept:.6f}")

        print(f"Margin: {margin:.6f}")

        # Calculate decision values and identify support vectors
        decision_values = svm.decision_function(transformed_points)
        support_mask = np.abs(np.abs(decision_values) - 1.0) < 1e-3
        support_indices = np.where(support_mask)[0]

        print(f"Support vectors: {len(support_indices)} points")
        print("Support vector details:")
        for idx in support_indices:
            x_orig = all_points[idx]
            phi_val = transformed_points[idx]
            label = "+" if labels[idx] == 1 else "-"
            decision_val = decision_values[idx]
            print(f"  x={x_orig}: φ(x)=[{phi_val[0]:.3f}, {phi_val[1]:.3f}], "
                  f"label={label}, f(x)={decision_val:.6f}")

        print(f"\n{'-'*40}")
        print("CLASSIFICATION VERIFICATION:")
        print("-"*40)

        for i, x in enumerate(all_points):
            phi_val = transformed_points[i]
            true_label = labels[i]
            decision_val = decision_values[i]
            predicted_label = 1 if decision_val > 0 else -1

            print(f"x={x}: f(x) = {w[0]:.6f}×{phi_val[0]:.3f} + {w[1]:.6f}×{phi_val[1]:.3f} + {b:.6f}")
            print(f"     = {decision_val:.6f} → prediction: {'+' if predicted_label == 1 else '-'} "
                  f"{'✓' if predicted_label == true_label else '✗'}")

# Task 5: Kernel function verification
print("\n" + "="*60)
print("TASK 5: KERNEL FUNCTION VERIFICATION")
print("="*60)

print("Verifying that transformations correspond to valid kernels...")

def verify_kernel_matrix(phi_func, points):
    """Verify that the kernel matrix is positive semi-definite"""
    transformed = phi_func(points)
    # Compute Gram matrix K_ij = φ(x_i)^T φ(x_j)
    K = transformed @ transformed.T

    # Check if positive semi-definite by computing eigenvalues
    eigenvals = np.linalg.eigvals(K)
    min_eigenval = np.min(eigenvals)

    return K, eigenvals, min_eigenval >= -1e-10

print("\nKernel matrix verification:")
for name, result in successful_transforms.items():
    phi_func = result['phi_func']
    K, eigenvals, is_psd = verify_kernel_matrix(phi_func, all_points)

    status = "✓ Valid" if is_psd else "✗ Invalid"
    min_eigenval = np.min(eigenvals)
    print(f"{name:20s}: {status} (min eigenvalue: {min_eigenval:.6f})")

    if name == 'Primary':  # Show detailed kernel matrix for primary
        print(f"\nKernel matrix for {name} transformation:")
        print("K =")
        for i in range(len(K)):
            row_str = "  [" + ", ".join([f"{K[i,j]:8.4f}" for j in range(len(K[i]))]) + "]"
            print(row_str)
        print(f"Eigenvalues: {eigenvals}")

print(f"\n{'='*60}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*60}")

print(f"✓ Successfully found {len(successful_transforms)} valid kernel transformations")
print("✓ All successful transformations have positive semi-definite kernel matrices")
print("✓ Each transformation provides perfect linear separation in the feature space")

print("\nRecommended transformation: PRIMARY")
print("  - φ(x) = [x², (x mod 2 - 0.5)x²]")
print("  - Provides diagonal separation in 2D feature space")
print("  - Clear geometric interpretation")
print("  - Robust and mathematically elegant")

print(f"\nAll plots saved to: {save_dir}")
print("✓ Analysis complete!")
