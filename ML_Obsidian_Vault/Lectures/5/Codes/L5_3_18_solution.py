import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import comb
import os
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting for professional mathematical expressions
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

print("=" * 80)
print("QUESTION 18: POLYNOMIAL KERNEL ANALYSIS")
print("=" * 80)

# ============================================================================
# TASK 1: Explicit feature mapping for 2D input with d=2, c=1
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Explicit Feature Mapping for 2D input, d=2, c=1")
print("="*60)

def polynomial_kernel_2d_d2_c1(x, z):
    """
    Compute polynomial kernel K(x,z) = (x^T z + 1)^2 for 2D vectors
    """
    return (np.dot(x, z) + 1)**2

def explicit_feature_mapping_2d_d2_c1(x):
    """
    Explicit feature mapping for 2D input with d=2, c=1
    Returns the feature vector φ(x) such that K(x,z) = φ(x)^T φ(z)
    """
    x1, x2 = x
    # For (x^T z + 1)^2, the explicit mapping is:
    # φ(x) = [1, √2*x1, √2*x2, x1^2, x2^2, √2*x1*x2]
    return np.array([
        1,                    # constant term
        np.sqrt(2) * x1,     # √2 * x1
        np.sqrt(2) * x2,     # √2 * x2
        x1**2,               # x1^2
        x2**2,               # x2^2
        np.sqrt(2) * x1 * x2 # √2 * x1 * x2
    ])

def detailed_derivation_d2_c1():
    """
    Detailed step-by-step derivation of explicit feature mapping for d=2, c=1
    """
    print("DETAILED DERIVATION: Explicit Feature Mapping for d=2, c=1")
    print("-" * 60)
    
    print("Step 1: Start with the polynomial kernel")
    print("K(x,z) = (x^T z + 1)^2")
    print()
    
    print("Step 2: Expand the kernel using binomial theorem")
    print("(x^T z + 1)^2 = (x^T z)^2 + 2(x^T z) + 1")
    print()
    
    print("Step 3: For 2D vectors x = [x1, x2]^T and z = [z1, z2]^T:")
    print("x^T z = x1*z1 + x2*z2")
    print()
    
    print("Step 4: Substitute and expand (x^T z)^2:")
    print("(x^T z)^2 = (x1*z1 + x2*z2)^2")
    print("         = (x1*z1)^2 + 2(x1*z1)(x2*z2) + (x2*z2)^2")
    print("         = x1^2*z1^2 + 2*x1*x2*z1*z2 + x2^2*z2^2")
    print()
    
    print("Step 5: Expand 2(x^T z):")
    print("2(x^T z) = 2(x1*z1 + x2*z2)")
    print("         = 2*x1*z1 + 2*x2*z2")
    print()
    
    print("Step 6: Combine all terms:")
    print("K(x,z) = x1^2*z1^2 + 2*x1*x2*z1*z2 + x2^2*z2^2 + 2*x1*z1 + 2*x2*z2 + 1")
    print()
    
    print("Step 7: Rewrite as inner product of feature vectors:")
    print("K(x,z) = [1, √2*x1, √2*x2, x1^2, x2^2, √2*x1*x2]^T")
    print("         · [1, √2*z1, √2*z2, z1^2, z2^2, √2*z1*z2]")
    print()
    
    print("Step 8: Verify by expanding the inner product:")
    print("φ(x)^T φ(z) = 1*1 + (√2*x1)*(√2*z1) + (√2*x2)*(√2*z2) + x1^2*z1^2 + x2^2*z2^2 + (√2*x1*x2)*(√2*z1*z2)")
    print("            = 1 + 2*x1*z1 + 2*x2*z2 + x1^2*z1^2 + x2^2*z2^2 + 2*x1*x2*z1*z2")
    print("            = x1^2*z1^2 + 2*x1*x2*z1*z2 + x2^2*z2^2 + 2*x1*z1 + 2*x2*z2 + 1")
    print("            = K(x,z) ✓")
    print()
    
    print("Therefore, the explicit feature mapping is:")
    print("φ(x) = [1, √2*x1, √2*x2, x1^2, x2^2, √2*x1*x2]^T")
    print()

# Test with example vectors
x = np.array([1, 2])
z = np.array([3, 4])

print(f"Input vectors: x = {x}, z = {z}")

# Show detailed derivation
detailed_derivation_d2_c1()

# Method 1: Using kernel trick
print("COMPUTATION USING KERNEL TRICK:")
print("-" * 40)
print("Step 1: Compute x^T z")
dot_product = np.dot(x, z)
print(f"x^T z = {x[0]}*{z[0]} + {x[1]}*{z[1]} = {dot_product}")
print()

print("Step 2: Add constant c = 1")
print(f"x^T z + c = {dot_product} + 1 = {dot_product + 1}")
print()

print("Step 3: Raise to power d = 2")
kernel_value = polynomial_kernel_2d_d2_c1(x, z)
print(f"K(x,z) = ({dot_product + 1})^2 = {kernel_value}")
print()

# Method 2: Using explicit feature mapping
print("COMPUTATION USING EXPLICIT FEATURE MAPPING:")
print("-" * 50)
print("Step 1: Compute φ(x)")
phi_x = explicit_feature_mapping_2d_d2_c1(x)
print(f"φ(x) = [1, √2*{x[0]}, √2*{x[1]}, {x[0]}^2, {x[1]}^2, √2*{x[0]}*{x[1]}]")
print(f"φ(x) = [1, {np.sqrt(2)*x[0]:.6f}, {np.sqrt(2)*x[1]:.6f}, {x[0]**2}, {x[1]**2}, {np.sqrt(2)*x[0]*x[1]:.6f}]")
print(f"φ(x) = {phi_x}")
print()

print("Step 2: Compute φ(z)")
phi_z = explicit_feature_mapping_2d_d2_c1(z)
print(f"φ(z) = [1, √2*{z[0]}, √2*{z[1]}, {z[0]}^2, {z[1]}^2, √2*{z[0]}*{z[1]}]")
print(f"φ(z) = [1, {np.sqrt(2)*z[0]:.6f}, {np.sqrt(2)*z[1]:.6f}, {z[0]**2}, {z[1]**2}, {np.sqrt(2)*z[0]*z[1]:.6f}]")
print(f"φ(z) = {phi_z}")
print()

print("Step 3: Compute φ(x)^T φ(z)")
explicit_value = np.dot(phi_x, phi_z)
print("φ(x)^T φ(z) = Σ(φ(x)_i * φ(z)_i)")
print(f"           = {phi_x[0]}*{phi_z[0]} + {phi_x[1]:.6f}*{phi_z[1]:.6f} + {phi_x[2]:.6f}*{phi_z[2]:.6f} + {phi_x[3]}*{phi_z[3]} + {phi_x[4]}*{phi_z[4]} + {phi_x[5]:.6f}*{phi_z[5]:.6f}")
print(f"           = {phi_x[0]*phi_z[0]} + {phi_x[1]*phi_z[1]:.6f} + {phi_x[2]*phi_z[2]:.6f} + {phi_x[3]*phi_z[3]} + {phi_x[4]*phi_z[4]} + {phi_x[5]*phi_z[5]:.6f}")
print(f"           = {explicit_value}")
print()

print("VERIFICATION:")
print(f"Kernel trick result: {kernel_value}")
print(f"Explicit mapping result: {explicit_value}")
print(f"Match: {np.isclose(kernel_value, explicit_value)}")
print()

# ============================================================================
# TASK 2: Feature space dimension for n-dimensional input with degree d
# ============================================================================
print("\n" + "="*60)
print("TASK 2: Feature Space Dimension Calculation")
print("="*60)

def feature_space_dimension(n, d):
    """
    Calculate the dimension of feature space for polynomial kernel (x^T z + c)^d
    with n-dimensional input and degree d
    """
    # The dimension is C(n+d, d) = (n+d)! / (n! * d!)
    return int(comb(n + d, d))

def detailed_dimension_derivation():
    """
    Detailed explanation of why the feature space dimension is C(n+d, d)
    """
    print("DETAILED DERIVATION: Feature Space Dimension")
    print("-" * 50)
    
    print("Step 1: Understand what we're counting")
    print("For polynomial kernel (x^T z + c)^d, we need all monomials of degree ≤ d")
    print("in n variables x1, x2, ..., xn")
    print()
    
    print("Step 2: Use stars and bars method")
    print("We need to count the number of ways to distribute d 'units' among n+1 'bins'")
    print("(n bins for variables + 1 bin for the constant term)")
    print()
    
    print("Step 3: Apply combination formula")
    print("This is equivalent to choosing d positions from n+d total positions")
    print("Therefore, the number of monomials is C(n+d, d)")
    print()
    
    print("Step 4: Mathematical formula")
    print("C(n+d, d) = (n+d)! / (n! * d!)")
    print()
    
    print("Step 5: Examples for verification:")
    print("For n=2, d=2: C(4,2) = 4!/(2!*2!) = 24/4 = 6")
    print("For n=2, d=3: C(5,3) = 5!/(2!*3!) = 120/12 = 10")
    print("For n=3, d=2: C(5,2) = 5!/(3!*2!) = 120/12 = 10")
    print()

detailed_dimension_derivation()

# Test with various dimensions and degrees
test_cases = [
    (2, 2), (2, 3), (2, 4),  # 2D input
    (3, 2), (3, 3), (3, 4),  # 3D input
    (5, 2), (5, 3), (5, 5),  # 5D input
    (10, 2), (10, 3), (10, 5)  # 10D input
]

print("Feature space dimensions for polynomial kernel (x^T z + c)^d:")
print("n (input dim) | d (degree) | Feature space dimension | Formula")
print("-" * 65)
for n, d in test_cases:
    dim = feature_space_dimension(n, d)
    formula = f"C({n}+{d},{d}) = C({n+d},{d})"
    print(f"{n:^13} | {d:^9} | {dim:^21} | {formula}")

# Create visualization of dimension growth
n_values = np.arange(1, 11)
d_values = [2, 3, 4, 5]

plt.figure(figsize=(12, 8))
for d in d_values:
    dimensions = [feature_space_dimension(n, d) for n in n_values]
    plt.plot(n_values, dimensions, marker='o', linewidth=2, markersize=6, 
             label=f'd = {d}')

plt.xlabel(r'Input Dimension ($n$)')
plt.ylabel(r'Feature Space Dimension')
plt.title(r'Feature Space Dimension Growth for Polynomial Kernels')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Use log scale due to exponential growth
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_dimension_growth.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 3: Compute K(x,z) for specific vectors with d=3, c=0
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Kernel Computation for d=3, c=0")
print("="*60)

def polynomial_kernel_d3_c0(x, z):
    """
    Compute polynomial kernel K(x,z) = (x^T z)^3 for any dimension
    """
    return (np.dot(x, z))**3

def explicit_feature_mapping_d3_c0(x):
    """
    Explicit feature mapping for polynomial kernel (x^T z)^3
    For 2D input, this gives all monomials of degree 3
    """
    x1, x2 = x
    # All monomials of degree 3: x1^3, x1^2*x2, x1*x2^2, x2^3
    # With appropriate coefficients to match the kernel
    return np.array([
        x1**3,
        np.sqrt(3) * x1**2 * x2,
        np.sqrt(3) * x1 * x2**2,
        x2**3
    ])

def detailed_derivation_d3_c0():
    """
    Detailed step-by-step derivation for d=3, c=0
    """
    print("DETAILED DERIVATION: Explicit Feature Mapping for d=3, c=0")
    print("-" * 60)
    
    print("Step 1: Start with the polynomial kernel")
    print("K(x,z) = (x^T z)^3")
    print()
    
    print("Step 2: For 2D vectors x = [x1, x2]^T and z = [z1, z2]^T:")
    print("x^T z = x1*z1 + x2*z2")
    print()
    
    print("Step 3: Expand (x^T z)^3 using binomial theorem")
    print("(x^T z)^3 = (x1*z1 + x2*z2)^3")
    print("         = (x1*z1)^3 + 3(x1*z1)^2(x2*z2) + 3(x1*z1)(x2*z2)^2 + (x2*z2)^3")
    print("         = x1^3*z1^3 + 3*x1^2*x2*z1^2*z2 + 3*x1*x2^2*z1*z2^2 + x2^3*z2^3")
    print()
    
    print("Step 4: Rewrite as inner product of feature vectors:")
    print("K(x,z) = [x1^3, √3*x1^2*x2, √3*x1*x2^2, x2^3]^T")
    print("         · [z1^3, √3*z1^2*z2, √3*z1*z2^2, z2^3]")
    print()
    
    print("Step 5: Verify by expanding the inner product:")
    print("φ(x)^T φ(z) = x1^3*z1^3 + (√3*x1^2*x2)*(√3*z1^2*z2) + (√3*x1*x2^2)*(√3*z1*z2^2) + x2^3*z2^3")
    print("            = x1^3*z1^3 + 3*x1^2*x2*z1^2*z2 + 3*x1*x2^2*z1*z2^2 + x2^3*z2^3")
    print("            = K(x,z) ✓")
    print()
    
    print("Therefore, the explicit feature mapping is:")
    print("φ(x) = [x1^3, √3*x1^2*x2, √3*x1*x2^2, x2^3]^T")
    print()

# Given vectors
x = np.array([2, 1])
z = np.array([1, 3])

print(f"Input vectors: x = {x}, z = {z}")
print(f"Parameters: d = 3, c = 0")

# Show detailed derivation
detailed_derivation_d3_c0()

# Method 1: Using kernel trick
print("COMPUTATION USING KERNEL TRICK:")
print("-" * 40)
print("Step 1: Compute x^T z")
dot_product = np.dot(x, z)
print(f"x^T z = {x[0]}*{z[0]} + {x[1]}*{z[1]} = {dot_product}")
print()

print("Step 2: Raise to power d = 3")
kernel_value = polynomial_kernel_d3_c0(x, z)
print(f"K(x,z) = ({dot_product})^3 = {kernel_value}")
print()

# Method 2: Using explicit feature mapping
print("COMPUTATION USING EXPLICIT FEATURE MAPPING:")
print("-" * 50)
print("Step 1: Compute φ(x)")
phi_x = explicit_feature_mapping_d3_c0(x)
print(f"φ(x) = [{x[0]}^3, √3*{x[0]}^2*{x[1]}, √3*{x[0]}*{x[1]}^2, {x[1]}^3]")
print(f"φ(x) = [{x[0]**3}, {np.sqrt(3)*x[0]**2*x[1]:.6f}, {np.sqrt(3)*x[0]*x[1]**2:6f}, {x[1]**3}]")
print(f"φ(x) = {phi_x}")
print()

print("Step 2: Compute φ(z)")
phi_z = explicit_feature_mapping_d3_c0(z)
print(f"φ(z) = [{z[0]}^3, √3*{z[0]}^2*{z[1]}, √3*{z[0]}*{z[1]}^2, {z[1]}^3]")
print(f"φ(z) = [{z[0]**3}, {np.sqrt(3)*z[0]**2*z[1]:.6f}, {np.sqrt(3)*z[0]*z[1]**2:6f}, {z[1]**3}]")
print(f"φ(z) = {phi_z}")
print()

print("Step 3: Compute φ(x)^T φ(z)")
explicit_value = np.dot(phi_x, phi_z)
print("φ(x)^T φ(z) = Σ(φ(x)_i * φ(z)_i)")
print(f"           = {phi_x[0]}*{phi_z[0]} + {phi_x[1]:.6f}*{phi_z[1]:.6f} + {phi_x[2]:.6f}*{phi_z[2]:.6f} + {phi_x[3]}*{phi_z[3]}")
print(f"           = {phi_x[0]*phi_z[0]} + {phi_x[1]*phi_z[1]:.6f} + {phi_x[2]*phi_z[2]:.6f} + {phi_x[3]*phi_z[3]}")
print(f"           = {explicit_value}")
print()

print("VERIFICATION:")
print(f"Kernel trick result: {kernel_value}")
print(f"Explicit mapping result: {explicit_value}")
print(f"Match: {np.isclose(kernel_value, explicit_value)}")
print()

# ============================================================================
# TASK 4: Show how c affects relative importance of interaction terms
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Effect of c on Interaction Terms")
print("="*60)

def analyze_c_effect(x, z, d=2, c_values=[0, 1, 2, 5]):
    """
    Analyze how different values of c affect the kernel expansion
    """
    print(f"Analyzing kernel K(x,z) = (x^T z + c)^{d} for x = {x}, z = {z}")
    print(f"x^T z = {np.dot(x, z)}")
    print()
    
    print("DETAILED ANALYSIS OF TERM CONTRIBUTIONS:")
    print("-" * 50)
    
    dot_product = np.dot(x, z)
    
    for c in c_values:
        print(f"\nFor c = {c}:")
        print(f"K(x,z) = ({dot_product} + {c})^{d} = {dot_product + c}^{d} = {(dot_product + c)**d}")
        
        if d == 2:
            # Expand (x^T z + c)^2
            quadratic_term = dot_product**2
            linear_term = 2 * dot_product * c
            constant_term = c**2
            
            print(f"Expansion: ({dot_product} + {c})^2 = {dot_product}^2 + 2*{dot_product}*{c} + {c}^2")
            print(f"         = {quadratic_term} + {linear_term} + {constant_term}")
            print(f"         = {quadratic_term + linear_term + constant_term}")
            
            print(f"Term breakdown:")
            print(f"  - Quadratic term (x^T z)^2: {quadratic_term}")
            print(f"  - Linear term 2(x^T z)c: {linear_term}")
            print(f"  - Constant term c^2: {constant_term}")
            
            if c > 0:
                interaction_ratio = (quadratic_term + linear_term) / constant_term
                print(f"  - Interaction/Constant ratio: {interaction_ratio:.2f}")
    
    results = []
    for c in c_values:
        kernel_value = (np.dot(x, z) + c)**d
        results.append((c, kernel_value))
    
    return results

# Test with our vectors
x = np.array([2, 1])
z = np.array([1, 3])
c_values = [0, 1, 2, 5]
results = analyze_c_effect(x, z, d=2, c_values=c_values)

# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: Kernel values vs c
plt.subplot(2, 2, 1)
c_range = np.linspace(0, 10, 100)
kernel_values = [(np.dot(x, z) + c)**2 for c in c_range]
plt.plot(c_range, kernel_values, 'b-', linewidth=2)
plt.xlabel(r'$c$')
plt.ylabel(r'$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^2$')
plt.title(r'Kernel Value vs $c$')
plt.grid(True, alpha=0.3)

# Plot 2: Relative contribution of different terms
plt.subplot(2, 2, 2)
dot_product = np.dot(x, z)
c_range = np.linspace(0, 10, 100)
constant_term = c_range**2
linear_term = 2 * dot_product * c_range
quadratic_term = dot_product**2 * np.ones_like(c_range)

plt.plot(c_range, constant_term, 'r-', label=r'$c^2$ term', linewidth=2)
plt.plot(c_range, linear_term, 'g-', label=r'$2(\mathbf{x}^T\mathbf{z})c$ term', linewidth=2)
plt.plot(c_range, quadratic_term, 'b-', label=r'$(\mathbf{x}^T\mathbf{z})^2$ term', linewidth=2)
plt.xlabel(r'$c$')
plt.ylabel(r'Term contribution')
plt.title(r'Contribution of Different Terms')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Ratio of interaction to constant terms
plt.subplot(2, 2, 3)
interaction_ratio = (2 * dot_product * c_range + dot_product**2) / (c_range**2 + 1e-10)
plt.plot(c_range, interaction_ratio, 'purple', linewidth=2)
plt.xlabel(r'$c$')
plt.ylabel(r'Interaction/Constant ratio')
plt.title(r'Relative Importance of Interaction Terms')
plt.grid(True, alpha=0.3)

# Plot 4: Effect on decision boundary (simplified)
plt.subplot(2, 2, 4)
# Create a simple 2D visualization showing how c affects the "effective" dot product
x1_range = np.linspace(-3, 3, 100)
x2_range = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# For different c values, show the "effective" decision boundary
for i, c in enumerate([0, 1, 2]):
    # This is a simplified visualization - in practice, the decision boundary
    # would depend on the SVM formulation
    Z = (X1 * z[0] + X2 * z[1] + c)**2
    plt.contour(X1, X2, Z, levels=[kernel_value/2], colors=['red', 'green', 'blue'][i], 
                linestyles=['-', '--', ':'], alpha=0.7)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Effect of $c$ on Decision Boundary')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'c_effect_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: Compare complexity of kernel vs explicit computation for d=5
# ============================================================================
print("\n" + "="*60)
print("TASK 5: Computational Complexity Comparison for d=5")
print("="*60)

def polynomial_kernel_d5(x, z):
    """
    Compute polynomial kernel K(x,z) = (x^T z)^5 using kernel trick
    """
    return (np.dot(x, z))**5

def explicit_feature_mapping_d5_2d(x):
    """
    Explicit feature mapping for 2D input with d=5
    Returns all monomials of degree 5
    """
    x1, x2 = x
    # All monomials of degree 5: x1^5, x1^4*x2, x1^3*x2^2, x1^2*x2^3, x1*x2^4, x2^5
    # With appropriate coefficients
    return np.array([
        x1**5,
        np.sqrt(5) * x1**4 * x2,
        np.sqrt(10) * x1**3 * x2**2,
        np.sqrt(10) * x1**2 * x2**3,
        np.sqrt(5) * x1 * x2**4,
        x2**5
    ])

def detailed_complexity_analysis():
    """
    Detailed analysis of computational complexity
    """
    print("DETAILED COMPLEXITY ANALYSIS:")
    print("-" * 40)
    
    print("Step 1: Kernel Trick Complexity")
    print("For kernel K(x,z) = (x^T z)^d:")
    print("  - Compute x^T z: O(n) operations")
    print("  - Raise to power d: O(1) operations")
    print("  - Total: O(n) operations")
    print()
    
    print("Step 2: Explicit Feature Mapping Complexity")
    print("For explicit mapping φ(x):")
    print("  - Feature vector dimension: C(n+d, d)")
    print("  - Computing each feature: O(1) per feature")
    print("  - Total: O(C(n+d, d)) operations")
    print()
    
    print("Step 3: Memory Requirements")
    print("Kernel Trick:")
    print("  - Store weight vector: O(n) memory")
    print("  - No need to store feature vectors")
    print()
    print("Explicit Mapping:")
    print("  - Store feature vector: O(C(n+d, d)) memory")
    print("  - Store weight vector: O(C(n+d, d)) memory")
    print()
    
    print("Step 4: Scalability Analysis")
    print("For d=5:")
    print("  - n=2: C(7,5) = 21 features")
    print("  - n=3: C(8,5) = 56 features")
    print("  - n=5: C(10,5) = 252 features")
    print("  - n=10: C(15,5) = 3003 features")
    print()

detailed_complexity_analysis()

def time_complexity_analysis():
    """
    Compare time complexity of kernel trick vs explicit computation
    """
    print("PRACTICAL COMPLEXITY ANALYSIS for d=5:")
    print()
    
    # Test with different input dimensions
    dimensions = [2, 3, 5, 10]
    n_samples = 1000
    
    print("Input Dim | Kernel Trick | Explicit Mapping | Speedup | Feature Dim")
    print("-" * 65)
    
    for n in dimensions:
        # Generate random data
        X = np.random.randn(n_samples, n)
        
        # Time kernel trick
        start_time = time.time()
        for i in range(n_samples):
            for j in range(n_samples):
                _ = (np.dot(X[i], X[j]))**5
        kernel_time = time.time() - start_time
        
        # Time explicit mapping (for 2D case only)
        if n == 2:
            start_time = time.time()
            for i in range(n_samples):
                for j in range(n_samples):
                    phi_i = explicit_feature_mapping_d5_2d(X[i])
                    phi_j = explicit_feature_mapping_d5_2d(X[j])
                    _ = np.dot(phi_i, phi_j)
            explicit_time = time.time() - start_time
            speedup = explicit_time / kernel_time
            feature_dim = feature_space_dimension(n, 5)
            print(f"{n:^9} | {kernel_time:^11.3f} | {explicit_time:^15.3f} | {speedup:^7.1f}x | {feature_dim:^10}")
        else:
            feature_dim = feature_space_dimension(n, 5)
            print(f"{n:^9} | {kernel_time:^11.3f} | {'N/A':^15} | {'N/A':^7} | {feature_dim:^10}")

time_complexity_analysis()

# Create complexity visualization
plt.figure(figsize=(12, 8))

# Plot 1: Theoretical complexity
n_range = np.arange(2, 21)
kernel_complexity = n_range  # O(n) for dot product
explicit_complexity = [comb(n + 5, 5) for n in n_range]  # O(C(n+5,5)) for explicit mapping

plt.subplot(2, 2, 1)
plt.plot(n_range, kernel_complexity, 'b-', linewidth=2, label=r'Kernel Trick $O(n)$')
plt.plot(n_range, explicit_complexity, 'r-', linewidth=2, label=r'Explicit Mapping $O(C(n+5,5))$')
plt.xlabel(r'Input Dimension ($n$)')
plt.ylabel(r'Computational Complexity')
plt.title(r'Theoretical Complexity Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Memory requirements
plt.subplot(2, 2, 2)
kernel_memory = n_range  # O(n) for storing weight vector
explicit_memory = [comb(n + 5, 5) for n in n_range]  # O(C(n+5,5)) for feature vector

plt.plot(n_range, kernel_memory, 'b-', linewidth=2, label=r'Kernel Trick')
plt.plot(n_range, explicit_memory, 'r-', linewidth=2, label=r'Explicit Mapping')
plt.xlabel(r'Input Dimension ($n$)')
plt.ylabel(r'Memory Requirements')
plt.title(r'Memory Requirements Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Feature space dimension growth
plt.subplot(2, 2, 3)
d_values = [2, 3, 4, 5]
for d in d_values:
    dimensions = [comb(n + d, d) for n in n_range]
    plt.plot(n_range, dimensions, linewidth=2, label=f'd={d}')

plt.xlabel(r'Input Dimension ($n$)')
plt.ylabel(r'Feature Space Dimension')
plt.title(r'Feature Space Dimension Growth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 4: Practical speedup estimation
plt.subplot(2, 2, 4)
# Estimate speedup based on complexity ratio
speedup_estimate = [comb(n + 5, 5) / n for n in n_range[1:]]  # Avoid division by zero
plt.plot(n_range[1:], speedup_estimate, 'g-', linewidth=2)
plt.xlabel(r'Input Dimension ($n$)')
plt.ylabel(r'Estimated Speedup')
plt.title(r'Kernel Trick Speedup Estimation')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# Additional Analysis: Visualizing the feature space
# ============================================================================
print("\n" + "="*60)
print("ADDITIONAL ANALYSIS: Feature Space Visualization")
print("="*60)

def visualize_feature_space_2d_d2():
    """
    Visualize how the polynomial kernel transforms 2D data into higher dimensions
    """
    # Create a simple dataset
    np.random.seed(42)
    n_points = 100
    X = np.random.randn(n_points, 2)
    
    # Create labels based on a non-linear decision boundary
    # y = 1 if x1^2 + x2^2 > 2, else -1
    y = np.where(X[:, 0]**2 + X[:, 1]**2 > 2, 1, -1)
    
    # Compute kernel matrix
    K = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            K[i, j] = (np.dot(X[i], X[j]) + 1)**2
    
    # Visualize original space and kernel-induced space
    plt.figure(figsize=(15, 5))
    
    # Original space
    plt.subplot(1, 3, 1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1', alpha=0.6)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Original 2D Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kernel matrix heatmap
    plt.subplot(1, 3, 2)
    im = plt.imshow(K, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title(r'Kernel Matrix $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T\mathbf{z} + 1)^2$')
    plt.xlabel(r'Sample $j$')
    plt.ylabel(r'Sample $i$')
    
    # Feature space projection (using PCA on kernel matrix)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import KernelCenterer
    
    # Center the kernel matrix
    K_centered = KernelCenterer().fit_transform(K)
    
    # Apply PCA to visualize the feature space
    pca = PCA(n_components=2)
    K_pca = pca.fit_transform(K_centered)
    
    plt.subplot(1, 3, 3)
    plt.scatter(K_pca[y == 1, 0], K_pca[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    plt.scatter(K_pca[y == -1, 0], K_pca[y == -1, 1], c='blue', label='Class -1', alpha=0.6)
    plt.xlabel(r'First Principal Component')
    plt.ylabel(r'Second Principal Component')
    plt.title(r'Feature Space Projection (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_space_visualization.png'), dpi=300, bbox_inches='tight')

visualize_feature_space_2d_d2()

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
