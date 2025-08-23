import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 1: SEPARATING HYPERPLANE ANALYSIS")
print("=" * 80)

# Given dataset
class_plus1 = np.array([[2, 3], [3, 4], [4, 2]])  # Class +1 points
class_minus1 = np.array([[0, 1], [1, 0], [0, 0]])  # Class -1 points

print("Dataset:")
print(f"Class +1: {class_plus1}")
print(f"Class -1: {class_minus1}")

# Given hyperplane parameters
w1, w2, b = 1, 1, -2
w = np.array([w1, w2])

print(f"\nGiven hyperplane: {w1}x₁ + {w2}x₂ + {b} = 0")
print(f"Weight vector w = [{w1}, {w2}]")
print(f"Bias term b = {b}")

# Step 1: Draw points and sketch separating hyperplane
print("\n" + "="*50)
print("STEP 1: VISUALIZATION OF DATASET AND HYPERPLANE")
print("="*50)

def plot_dataset_and_hyperplane():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot data points
    ax.scatter(class_plus1[:, 0], class_plus1[:, 1], c='red', s=200, marker='o', 
               label='Class +1', edgecolors='black', linewidth=2)
    ax.scatter(class_minus1[:, 0], class_minus1[:, 1], c='blue', s=200, marker='s', 
               label='Class -1', edgecolors='black', linewidth=2)
    
    # Plot hyperplane
    x_range = np.linspace(-1, 5, 100)
    y_hyperplane = (-w1 * x_range - b) / w2
    ax.plot(x_range, y_hyperplane, 'g-', linewidth=3, label='Separating Hyperplane')
    
    # Plot margin boundaries
    y_margin_plus = (-w1 * x_range - b + 1) / w2
    y_margin_minus = (-w1 * x_range - b - 1) / w2
    ax.plot(x_range, y_margin_plus, 'g--', linewidth=2, alpha=0.7, label='Margin +1')
    ax.plot(x_range, y_margin_minus, 'g--', linewidth=2, alpha=0.7, label='Margin -1')
    
    # Shade regions
    ax.fill_between(x_range, y_hyperplane, 6, alpha=0.2, color='red', label='Class +1 Region')
    ax.fill_between(x_range, y_hyperplane, -1, alpha=0.2, color='blue', label='Class -1 Region')
    
    # Add point labels
    for i, point in enumerate(class_plus1):
        ax.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1))
    
    for i, point in enumerate(class_minus1):
        ax.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1))
    
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Dataset with Separating Hyperplane', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    
    # Add hyperplane equation
    eq_text = f'Hyperplane: ${w1}x_1 + {w2}x_2 + {b} = 0$'
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_and_hyperplane.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_dataset_and_hyperplane()

# Step 2: Verify hyperplane separates the classes
print("\n" + "="*50)
print("STEP 2: VERIFYING CLASS SEPARATION")
print("="*50)

def verify_separation():
    print("Verifying that the hyperplane separates the two classes:")
    print(f"Hyperplane: {w1}x₁ + {w2}x₂ + {b} = 0")
    print()
    
    # Check Class +1 points
    print("Class +1 points:")
    for i, point in enumerate(class_plus1):
        activation = w1 * point[0] + w2 * point[1] + b
        print(f"  Point {i+1}: ({point[0]}, {point[1]})")
        print(f"    Activation = {w1}×{point[0]} + {w2}×{point[1]} + {b} = {activation}")
        print(f"    Should be > 0: {activation > 0}")
        print()
    
    # Check Class -1 points
    print("Class -1 points:")
    for i, point in enumerate(class_minus1):
        activation = w1 * point[0] + w2 * point[1] + b
        print(f"  Point {i+1}: ({point[0]}, {point[1]})")
        print(f"    Activation = {w1}×{point[0]} + {w2}×{point[1]} + {b} = {activation}")
        print(f"    Should be < 0: {activation < 0}")
        print()
    
    # Overall verification
    all_plus1_positive = all(w1 * p[0] + w2 * p[1] + b > 0 for p in class_plus1)
    all_minus1_negative = all(w1 * p[0] + w2 * p[1] + b < 0 for p in class_minus1)
    
    print(f"All Class +1 points have positive activation: {all_plus1_positive}")
    print(f"All Class -1 points have negative activation: {all_minus1_negative}")
    print(f"Hyperplane successfully separates classes: {all_plus1_positive and all_minus1_negative}")

verify_separation()

# Step 3: Calculate functional margin for each point
print("\n" + "="*50)
print("STEP 3: FUNCTIONAL MARGIN CALCULATIONS")
print("="*50)

def calculate_functional_margins():
    print("Functional margin = y_i × (w^T x_i + b)")
    print()
    
    all_points = np.vstack([class_plus1, class_minus1])
    all_labels = np.array([1, 1, 1, -1, -1, -1])  # +1 for first 3, -1 for last 3
    
    print("Functional margins for all points:")
    for i, (point, label) in enumerate(zip(all_points, all_labels)):
        activation = w1 * point[0] + w2 * point[1] + b
        functional_margin = label * activation
        print(f"  Point {i+1}: ({point[0]}, {point[1]}) with label y = {label}")
        print(f"    Activation = {w1}×{point[0]} + {w2}×{point[1]} + {b} = {activation}")
        print(f"    Functional margin = {label} × {activation} = {functional_margin}")
        print()
    
    # Find minimum functional margin
    functional_margins = [all_labels[i] * (w1 * all_points[i, 0] + w2 * all_points[i, 1] + b) 
                         for i in range(len(all_points))]
    min_margin = min(functional_margins)
    print(f"Minimum functional margin: {min_margin}")
    print(f"Points with minimum margin: {[i+1 for i, m in enumerate(functional_margins) if m == min_margin]}")

calculate_functional_margins()

# Step 4: Calculate geometric margin for point (2, 3)
print("\n" + "="*50)
print("STEP 4: GEOMETRIC MARGIN CALCULATION")
print("="*50)

def calculate_geometric_margin():
    point = np.array([2, 3])
    label = 1  # Class +1
    
    print(f"Calculating geometric margin for point ({point[0]}, {point[1]}) with label y = {label}")
    
    # Method 1: Using functional margin divided by ||w||
    activation = w1 * point[0] + w2 * point[1] + b
    functional_margin = label * activation
    w_norm = np.linalg.norm(w)
    geometric_margin = functional_margin / w_norm
    
    print(f"Method 1: Geometric margin = Functional margin / ||w||")
    print(f"  Functional margin = {label} × {activation} = {functional_margin}")
    print(f"  ||w|| = √({w1}² + {w2}²) = √{w1**2 + w2**2} = {w_norm}")
    print(f"  Geometric margin = {functional_margin} / {w_norm} = {geometric_margin}")
    print()
    
    # Method 2: Direct distance calculation
    distance = abs(activation) / w_norm
    print(f"Method 2: Distance from point to hyperplane")
    print(f"  Distance = |w^T x + b| / ||w|| = |{activation}| / {w_norm} = {distance}")
    print(f"  Geometric margin = {label} × {distance} = {geometric_margin}")
    
    return geometric_margin

geometric_margin_point = calculate_geometric_margin()

# Step 5: City planning problem
print("\n" + "="*50)
print("STEP 5: CITY PLANNING PROBLEM - DETAILED SVM DERIVATION")
print("="*50)

def solve_city_planning():
    print("City Planning Problem:")
    print("Zone A (houses): (2, 3), (3, 4), (4, 2)")
    print("Zone B (houses): (0, 1), (1, 0), (0, 0)")
    print("New house: (2.5, 2.5)")
    print()

    # Prepare data
    X = np.vstack([class_plus1, class_minus1])
    y = np.array([1, 1, 1, -1, -1, -1])

    print("DETAILED SVM MATHEMATICAL DERIVATION:")
    print("="*60)

    print("\nStep 1: SVM Primal Problem")
    print("Minimize: (1/2)||w||²")
    print("Subject to: y_i(w^T x_i + b) ≥ 1 for all i")
    print()

    print("Step 2: Data Matrix Setup")
    print("Training points and labels:")
    for i, (point, label) in enumerate(zip(X, y)):
        print(f"  x_{i+1} = [{point[0]}, {point[1]}], y_{i+1} = {label:+d}")
    print()

    # Use SVM to find optimal boundary
    svm = SVC(kernel='linear', C=1e6)  # Very large C for hard margin
    svm.fit(X, y)

    # Get optimal hyperplane parameters
    w_opt = svm.coef_[0]
    b_opt = svm.intercept_[0]

    print("Step 3: SVM Solution")
    print(f"Optimal weight vector: w* = [{w_opt[0]:.6f}, {w_opt[1]:.6f}]")
    print(f"Optimal bias: b* = {b_opt:.6f}")
    print(f"||w*|| = {np.linalg.norm(w_opt):.6f}")
    print()

    print("Step 4: Support Vector Analysis")
    print("Support vectors are points with α_i > 0 (on the margin boundary)")

    # Calculate functional margins to identify support vectors
    functional_margins = []
    for i, (point, label) in enumerate(zip(X, y)):
        activation = w_opt[0] * point[0] + w_opt[1] * point[1] + b_opt
        functional_margin = label * activation
        functional_margins.append(functional_margin)

        print(f"  Point {i+1}: ({point[0]}, {point[1]})")
        print(f"    Activation = {w_opt[0]:.6f}×{point[0]} + {w_opt[1]:.6f}×{point[1]} + {b_opt:.6f} = {activation:.6f}")
        print(f"    Functional margin = {label} × {activation:.6f} = {functional_margin:.6f}")

        # Check if it's a support vector (functional margin ≈ 1)
        if abs(functional_margin - 1.0) < 1e-6:
            print(f"    → SUPPORT VECTOR (functional margin = 1)")
        print()

    # Find support vectors
    support_vector_indices = [i for i, fm in enumerate(functional_margins) if abs(fm - 1.0) < 1e-6]
    print(f"Support vector indices: {[i+1 for i in support_vector_indices]}")
    print(f"Support vectors: {[f'({X[i][0]}, {X[i][1]})' for i in support_vector_indices]}")
    print()

    print("Step 5: Lagrange Multipliers (Dual Solution)")
    print("From KKT conditions: w* = Σ α_i y_i x_i")
    print("For support vectors, we can solve for α_i:")

    if len(support_vector_indices) == 3:  # Expected for this problem
        print("With 3 support vectors, we solve the system:")
        print("  Constraint 1: α₁y₁ + α₄y₄ + α₅y₅ = 0  (dual constraint)")
        print("  Constraint 2: w* = α₁y₁x₁ + α₄y₄x₄ + α₅y₅x₅")
        print()

        # Get support vector data
        sv_points = [X[i] for i in support_vector_indices]
        sv_labels = [y[i] for i in support_vector_indices]

        print("Support vector data:")
        for i, (idx, point, label) in enumerate(zip(support_vector_indices, sv_points, sv_labels)):
            print(f"  α_{idx+1}: point ({point[0]}, {point[1]}), label {label:+d}")
        print()

        # Set up the system of equations more carefully
        # We have support vectors at indices 0, 3, 4 (corresponding to points 1, 4, 5)
        # Let's use the correct indexing

        print("Setting up the system:")
        print("Support vectors and their contributions:")
        for i, (idx, point, label) in enumerate(zip(support_vector_indices, sv_points, sv_labels)):
            print(f"  α_{idx+1} * {label:+d} * [{point[0]}, {point[1]}] = α_{idx+1} * [{label*point[0]}, {label*point[1]}]")

        # w* = α₁(+1)[2,3] + α₄(-1)[0,1] + α₅(-1)[1,0]
        # w* = α₁[2,3] + α₄[0,-1] + α₅[-1,0]
        # [0.5, 0.5] = [2α₁ + 0α₄ - 1α₅, 3α₁ - 1α₄ + 0α₅]

        print("  w₁* = 2α₁ + 0α₄ - 1α₅ = 0.5")
        print("  w₂* = 3α₁ - 1α₄ + 0α₅ = 0.5")
        print("  α₁(+1) + α₄(-1) + α₅(-1) = 0  (dual constraint)")
        print("  α₁ - α₄ - α₅ = 0")
        print()

        # Solve the 3x3 system
        A = np.array([
            [2, 0, -1],   # 2α₁ - α₅ = 0.5
            [3, -1, 0],   # 3α₁ - α₄ = 0.5
            [1, -1, -1]   # α₁ - α₄ - α₅ = 0
        ])
        b = np.array([0.5, 0.5, 0])

        print("Matrix form: A * [α₁, α₄, α₅]ᵀ = b")
        print("A =")
        print(A)
        print("b =", b)

        try:
            alphas = np.linalg.solve(A, b)
            alpha1, alpha4, alpha5 = alphas

            print("Solving the linear system:")
            print(f"  α₁ = {alpha1:.6f}")
            print(f"  α₄ = {alpha4:.6f}")
            print(f"  α₅ = {alpha5:.6f}")
            print()

            # Verify the solution
            print("Verification:")
            w_check = alpha1 * sv_labels[0] * sv_points[0] + alpha4 * sv_labels[1] * sv_points[1] + alpha5 * sv_labels[2] * sv_points[2]
            print(f"  Reconstructed w* = {w_check}")
            print(f"  Original w* = {w_opt}")
            print(f"  Difference = {np.linalg.norm(w_check - w_opt):.10f}")

            dual_constraint = alpha1 * sv_labels[0] + alpha4 * sv_labels[1] + alpha5 * sv_labels[2]
            print(f"  Dual constraint check: α₁y₁ + α₄y₄ + α₅y₅ = {dual_constraint:.10f} (should be 0)")
            print()

            # Check which points are actually support vectors (α > 0)
            print("Support vector analysis:")
            alpha_values = [alpha1, alpha4, alpha5]
            for i, (idx, alpha_val) in enumerate(zip(support_vector_indices, alpha_values)):
                point = X[idx]
                if abs(alpha_val) > 1e-10:
                    print(f"  Point ({point[0]}, {point[1]}): α_{idx+1} = {alpha_val:.6f} > 0 → TRUE support vector")
                else:
                    print(f"  Point ({point[0]}, {point[1]}): α_{idx+1} = {alpha_val:.6f} ≈ 0 → NOT a support vector")

        except np.linalg.LinAlgError:
            print("  System is singular, using approximate values")
            alpha1 = alpha4 = alpha5 = 1.0 / (3 * np.linalg.norm(w_opt))
            print(f"  Approximate α values: {alpha1:.6f}")
    print()

    print("Step 6: Optimal Hyperplane Equation")
    print(f"  {w_opt[0]:.6f}x₁ + {w_opt[1]:.6f}x₂ + {b_opt:.6f} = 0")

    # Simplify if possible
    if abs(w_opt[0] - w_opt[1]) < 1e-6:  # If w1 ≈ w2
        scale_factor = 1.0 / w_opt[0]
        w1_simple = w_opt[0] * scale_factor
        w2_simple = w_opt[1] * scale_factor
        b_simple = b_opt * scale_factor
        print(f"  Simplified: {w1_simple:.1f}x₁ + {w2_simple:.1f}x₂ + {b_simple:.1f} = 0")
        print(f"  Or: x₁ + x₂ = {-b_simple:.1f}")
    print()

    print("Step 7: Margin Calculation")
    margin_width = 2.0 / np.linalg.norm(w_opt)
    print(f"Margin width = 2/||w*|| = 2/{np.linalg.norm(w_opt):.6f} = {margin_width:.6f}")
    print(f"Half-margin = {margin_width/2:.6f}")
    print()

    print("Step 8: Distance Calculations")
    print("Distance from each house to the optimal road:")
    distances = []
    for i, point in enumerate(X):
        activation = w_opt[0] * point[0] + w_opt[1] * point[1] + b_opt
        distance = abs(activation) / np.linalg.norm(w_opt)
        distances.append(distance)
        zone = "A" if y[i] == 1 else "B"
        print(f"  House {i+1} ({point[0]}, {point[1]}) in Zone {zone}: {distance:.6f} units")

    min_distance = min(distances)
    print(f"\nMinimum distance from any house to the road: {min_distance:.6f} units")
    print(f"This confirms the margin calculation: {min_distance:.6f} ≈ {margin_width/2:.6f}")
    print()

    print("Step 9: New House Classification")
    new_house = np.array([2.5, 2.5])
    new_activation = w_opt[0] * new_house[0] + w_opt[1] * new_house[1] + b_opt
    new_distance = abs(new_activation) / np.linalg.norm(w_opt)
    new_zone = "A" if new_activation > 0 else "B"

    print(f"New house at ({new_house[0]}, {new_house[1]}):")
    print(f"  Activation = {w_opt[0]:.6f}×{new_house[0]} + {w_opt[1]:.6f}×{new_house[1]} + {b_opt:.6f}")
    print(f"             = {new_activation:.6f}")
    print(f"  Distance to road = |{new_activation:.6f}|/{np.linalg.norm(w_opt):.6f} = {new_distance:.6f} units")
    print(f"  Since activation {'>' if new_activation > 0 else '<'} 0, assign to Zone {new_zone}")

    if abs(new_distance - min_distance) < 1e-6:
        print(f"  Special note: This house lies exactly on the margin boundary!")

    return w_opt, b_opt, new_zone

w_opt, b_opt, new_zone = solve_city_planning()

# Step 6: Visualization of optimal solution
print("\n" + "="*50)
print("STEP 6: OPTIMAL SOLUTION VISUALIZATION")
print("="*50)

def plot_optimal_solution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create dataset for plotting
    X_plot = np.vstack([class_plus1, class_minus1])
    y_plot = np.array([1, 1, 1, -1, -1, -1])
    
    # Plot 1: Original hyperplane vs optimal hyperplane
    ax1.scatter(class_plus1[:, 0], class_plus1[:, 1], c='red', s=200, marker='o', 
                label='Zone A (Class +1)', edgecolors='black', linewidth=2)
    ax1.scatter(class_minus1[:, 0], class_minus1[:, 1], c='blue', s=200, marker='s', 
                label='Zone B (Class -1)', edgecolors='black', linewidth=2)
    
    # Original hyperplane
    x_range = np.linspace(-1, 5, 100)
    y_original = (-w1 * x_range - b) / w2
    ax1.plot(x_range, y_original, 'g-', linewidth=3, label='Given Hyperplane')
    
    # Optimal hyperplane
    y_optimal = (-w_opt[0] * x_range - b_opt) / w_opt[1]
    ax1.plot(x_range, y_optimal, 'r-', linewidth=3, label='Optimal Hyperplane (SVM)')
    
    # Optimal margin boundaries
    y_margin_plus_opt = (-w_opt[0] * x_range - b_opt + 1) / w_opt[1]
    y_margin_minus_opt = (-w_opt[0] * x_range - b_opt - 1) / w_opt[1]
    ax1.plot(x_range, y_margin_plus_opt, 'r--', linewidth=2, alpha=0.7, label='Optimal Margin +1')
    ax1.plot(x_range, y_margin_minus_opt, 'r--', linewidth=2, alpha=0.7, label='Optimal Margin -1')
    
    ax1.set_xlabel('$x_1$', fontsize=14)
    ax1.set_ylabel('$x_2$', fontsize=14)
    ax1.set_title('Comparison: Given vs Optimal Hyperplane', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.legend(fontsize=12)
    ax1.set_aspect('equal')
    
    # Plot 2: City planning with new house
    ax2.scatter(class_plus1[:, 0], class_plus1[:, 1], c='red', s=200, marker='o', 
                label='Zone A Houses', edgecolors='black', linewidth=2)
    ax2.scatter(class_minus1[:, 0], class_minus1[:, 1], c='blue', s=200, marker='s', 
                label='Zone B Houses', edgecolors='black', linewidth=2)
    
    # New house
    new_house = np.array([2.5, 2.5])
    new_color = 'red' if new_zone == 'A' else 'blue'
    ax2.scatter(new_house[0], new_house[1], c=new_color, s=300, marker='*', 
                label=f'New House (Zone {new_zone})', edgecolors='black', linewidth=2)
    
    # Optimal road boundary
    ax2.plot(x_range, y_optimal, 'k-', linewidth=4, label='Optimal Road Boundary')
    
    # Shade zones
    ax2.fill_between(x_range, y_optimal, 6, alpha=0.2, color='red', label='Zone A')
    ax2.fill_between(x_range, y_optimal, -1, alpha=0.2, color='blue', label='Zone B')
    
    # Add distance annotations
    for i, point in enumerate(X_plot):
        activation = w_opt[0] * point[0] + w_opt[1] * point[1] + b_opt
        distance = abs(activation) / np.linalg.norm(w_opt)
        ax2.annotate(f'{distance:.2f}', (point[0], point[1]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1))
    
    # Distance for new house
    new_activation = w_opt[0] * new_house[0] + w_opt[1] * new_house[1] + b_opt
    new_distance = abs(new_activation) / np.linalg.norm(w_opt)
    ax2.annotate(f'{new_distance:.2f}', (new_house[0], new_house[1]), 
                xytext=(10, 10), textcoords='offset points', fontsize=12, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=2))
    
    ax2.set_xlabel('$x_1$', fontsize=14)
    ax2.set_ylabel('$x_2$', fontsize=14)
    ax2.set_title('City Planning: Optimal Road Boundary', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.legend(fontsize=12)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimal_solution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_optimal_solution()

# Step 7: Summary and analysis
print("\n" + "="*50)
print("STEP 7: SUMMARY AND ANALYSIS")
print("="*50)

def print_summary():
    print("SUMMARY OF RESULTS:")
    print()
    
    print("1. DATASET VISUALIZATION:")
    print("   - Class +1 points: (2, 3), (3, 4), (4, 2)")
    print("   - Class -1 points: (0, 1), (1, 0), (0, 0)")
    print("   - Dataset is linearly separable")
    print()
    
    print("2. GIVEN HYPERPLANE VERIFICATION:")
    print(f"   - Hyperplane: {w1}x₁ + {w2}x₂ + {b} = 0")
    print("   - Successfully separates all points")
    print("   - All Class +1 points have positive activation")
    print("   - All Class -1 points have negative activation")
    print()
    
    print("3. FUNCTIONAL MARGINS:")
    print("   - Functional margin = y_i × (w^T x_i + b)")
    print("   - All functional margins are positive (correct classification)")
    print("   - Minimum functional margin indicates closest points to boundary")
    print()
    
    print("4. GEOMETRIC MARGIN:")
    print(f"   - For point (2, 3): {geometric_margin_point:.4f}")
    print("   - Geometric margin = Functional margin / ||w||")
    print("   - Represents actual distance from point to hyperplane")
    print()
    
    print("5. OPTIMAL SOLUTION (SVM):")
    print(f"   - Optimal hyperplane: {w_opt[0]:.4f}x₁ + {w_opt[1]:.4f}x₂ + {b_opt:.4f} = 0")
    print("   - Maximizes the minimum distance from any point to the boundary")
    print("   - Provides better generalization than the given hyperplane")
    print()
    
    print("6. CITY PLANNING SOLUTION:")
    print(f"   - Optimal road boundary maximizes minimum distance from any house")
    print(f"   - New house at (2.5, 2.5) should be assigned to Zone {new_zone}")
    print("   - This solution ensures maximum safety margin for all residents")

print_summary()

print(f"\nAll plots saved to: {save_dir}")
print("=" * 80)
