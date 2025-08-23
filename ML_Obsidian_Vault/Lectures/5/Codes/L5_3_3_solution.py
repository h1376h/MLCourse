import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations_with_replacement

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid compilation issues
plt.rcParams['font.size'] = 12

print("=" * 60)
print("QUESTION 3: POLYNOMIAL KERNEL CALCULATIONS")
print("=" * 60)

# Given vectors
x = np.array([2, -1, 3])
z = np.array([1, 2, -1])

print(f"Given vectors:")
print(f"x = {x}")
print(f"z = {z}")

# Task 1: Calculate K(x,z) for c=1, d=2
print("\n" + "="*50)
print("TASK 1: POLYNOMIAL KERNEL WITH c=1, d=2")
print("="*50)

def polynomial_kernel(x, z, c, d):
    """
    Calculate polynomial kernel K(x,z) = (x^T z + c)^d
    """
    dot_product = np.dot(x, z)
    kernel_value = (dot_product + c) ** d
    return dot_product, kernel_value

dot_xz, K_1_2 = polynomial_kernel(x, z, c=1, d=2)

print(f"\nK(x,z) = (x^T z + c)^d with c=1, d=2")
print(f"Step 1: Calculate x^T z")
print(f"x^T z = {x} · {z}")
print(f"     = {x[0]}×{z[0]} + {x[1]}×{z[1]} + {x[2]}×{z[2]}")
print(f"     = {x[0]*z[0]} + {x[1]*z[1]} + {x[2]*z[2]}")
print(f"     = {dot_xz}")

print(f"\nStep 2: Apply kernel formula")
print(f"K(x,z) = (x^T z + c)^d")
print(f"       = ({dot_xz} + {1})^{2}")
print(f"       = {dot_xz + 1}^{2}")
print(f"       = {K_1_2}")

# Task 2: Calculate K(x,z) for c=0, d=3
print("\n" + "="*50)
print("TASK 2: POLYNOMIAL KERNEL WITH c=0, d=3")
print("="*50)

dot_xz, K_0_3 = polynomial_kernel(x, z, c=0, d=3)

print(f"\nK(x,z) = (x^T z + c)^d with c=0, d=3")
print(f"Step 1: x^T z = {dot_xz} (same as before)")
print(f"\nStep 2: Apply kernel formula")
print(f"K(x,z) = (x^T z + c)^d")
print(f"       = ({dot_xz} + {0})^{3}")
print(f"       = {dot_xz}^{3}")
print(f"       = {K_0_3}")

# Task 3: Find explicit feature mapping for c=0, d=2 in 3D
print("\n" + "="*50)
print("TASK 3: EXPLICIT FEATURE MAPPING FOR c=0, d=2")
print("="*50)

def explicit_polynomial_mapping_3d_degree2(vector):
    """
    Explicit feature mapping for polynomial kernel (x^T z)^2 in 3D
    φ(x) = [x1^2, x2^2, x3^2, √2*x1*x2, √2*x1*x3, √2*x2*x3]
    """
    x1, x2, x3 = vector[0], vector[1], vector[2]
    
    # Monomials of degree 2
    phi = np.array([
        x1**2,           # x1^2
        x2**2,           # x2^2  
        x3**2,           # x3^2
        np.sqrt(2)*x1*x2,  # √2*x1*x2
        np.sqrt(2)*x1*x3,  # √2*x1*x3
        np.sqrt(2)*x2*x3   # √2*x2*x3
    ])
    
    return phi

phi_x = explicit_polynomial_mapping_3d_degree2(x)
phi_z = explicit_polynomial_mapping_3d_degree2(z)

print(f"\nFor polynomial kernel (x^T z)^2 in 3D, the explicit mapping is:")
print(f"φ(x) = [x1^2, x2^2, x3^2, √2*x1*x2, √2*x1*x3, √2*x2*x3]")

print(f"\nFor x = {x}:")
print(f"φ(x) = [{x[0]}^2, {x[1]}^2, {x[2]}^2, √2×{x[0]}×{x[1]}, √2×{x[0]}×{x[2]}, √2×{x[1]}×{x[2]}]")
print(f"     = [{x[0]**2}, {x[1]**2}, {x[2]**2}, {np.sqrt(2)*x[0]*x[1]:.3f}, {np.sqrt(2)*x[0]*x[2]:.3f}, {np.sqrt(2)*x[1]*x[2]:.3f}]")
print(f"     = {phi_x}")

print(f"\nFor z = {z}:")
print(f"φ(z) = [{z[0]}^2, {z[1]}^2, {z[2]}^2, √2×{z[0]}×{z[1]}, √2×{z[0]}×{z[2]}, √2×{z[1]}×{z[2]}]")
print(f"     = [{z[0]**2}, {z[1]**2}, {z[2]**2}, {np.sqrt(2)*z[0]*z[1]:.3f}, {np.sqrt(2)*z[0]*z[2]:.3f}, {np.sqrt(2)*z[1]*z[2]:.3f}]")
print(f"     = {phi_z}")

# Task 4: Verify that K(x,z) = φ(x)^T φ(z)
print("\n" + "="*50)
print("TASK 4: VERIFICATION OF KERNEL EQUIVALENCE")
print("="*50)

# Calculate using explicit mapping
explicit_kernel = np.dot(phi_x, phi_z)

# Calculate using kernel trick for c=0, d=2
_, kernel_trick = polynomial_kernel(x, z, c=0, d=2)

print(f"\nVerification that K(x,z) = φ(x)^T φ(z):")
print(f"\nMethod 1 - Explicit mapping:")
print(f"φ(x)^T φ(z) = {phi_x} · {phi_z}")

# Show detailed calculation
terms = []
for i in range(len(phi_x)):
    term = phi_x[i] * phi_z[i]
    terms.append(term)
    print(f"  Term {i+1}: {phi_x[i]:.3f} × {phi_z[i]:.3f} = {term:.3f}")

print(f"Sum = {' + '.join([f'{term:.3f}' for term in terms])}")
print(f"    = {explicit_kernel:.3f}")

print(f"\nMethod 2 - Kernel trick:")
print(f"K(x,z) = (x^T z)^2 = ({dot_xz})^2 = {kernel_trick}")

print(f"\nVerification: {explicit_kernel:.6f} ≈ {kernel_trick:.6f}")
print(f"Difference: {abs(explicit_kernel - kernel_trick):.10f}")
print(f"Methods agree: {np.isclose(explicit_kernel, kernel_trick)}")

# Task 5: Effect of parameter c
print("\n" + "="*50)
print("TASK 5: EFFECT OF PARAMETER c")
print("="*50)

def analyze_c_parameter():
    """
    Analyze how parameter c affects the relative importance of different order terms
    """
    print(f"\nAnalyzing K(x,z) = (x^T z + c)^d for different values of c:")
    print(f"Using x = {x}, z = {z}, x^T z = {dot_xz}")
    
    c_values = [0, 0.5, 1, 2, 5]
    d = 2
    
    print(f"\nFor degree d = {d}:")
    print("c\t(x^T z + c)\tK(x,z)\t\tExpansion")
    print("-" * 70)
    
    for c in c_values:
        kernel_val = (dot_xz + c) ** d
        
        # Expand (x^T z + c)^2 = (x^T z)^2 + 2c(x^T z) + c^2
        term1 = dot_xz ** 2
        term2 = 2 * c * dot_xz  
        term3 = c ** 2
        
        print(f"{c}\t{dot_xz + c}\t\t{kernel_val}\t\t{term1} + {term2} + {term3}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Kernel values vs c
    c_range = np.linspace(0, 5, 100)
    kernel_values = [(dot_xz + c) ** 2 for c in c_range]
    
    ax1.plot(c_range, kernel_values, 'b-', linewidth=2, label=r'$K(x,z) = (x^T z + c)^2$')
    ax1.scatter(c_values, [(dot_xz + c) ** 2 for c in c_values], 
                color='red', s=100, zorder=5, label='Sample points')
    ax1.set_xlabel('Parameter c')
    ax1.set_ylabel('Kernel Value K(x,z)')
    ax1.set_title('Polynomial Kernel vs Parameter c')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Contribution of different terms
    c_range = np.linspace(0, 5, 100)
    term1_contrib = [dot_xz ** 2] * len(c_range)  # Constant
    term2_contrib = [2 * c * dot_xz for c in c_range]  # Linear in c
    term3_contrib = [c ** 2 for c in c_range]  # Quadratic in c
    
    ax2.plot(c_range, term1_contrib, 'r-', linewidth=2, label=rf'$(x^T z)^2 = {dot_xz**2}$')
    ax2.plot(c_range, term2_contrib, 'g-', linewidth=2, label=rf'$2c(x^T z) = {2*dot_xz}c$')
    ax2.plot(c_range, term3_contrib, 'b-', linewidth=2, label=r'$c^2$')
    ax2.set_xlabel('Parameter c')
    ax2.set_ylabel('Term Contribution')
    ax2.set_title('Contribution of Different Order Terms')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_c_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nKey insights about parameter c:")
    print(f"1. c=0: Only pure polynomial terms (x^T z)^d")
    print(f"2. c>0: Adds lower-order terms, giving more weight to constant and linear terms")
    print(f"3. Large c: Dominates the kernel, reducing importance of input similarity")
    print(f"4. c acts as a 'bias' term that shifts the kernel values upward")

analyze_c_parameter()

# Additional visualization: 3D feature space
print("\n" + "="*50)
print("VISUALIZATION: 3D FEATURE SPACE MAPPING")
print("="*50)

def visualize_feature_mapping():
    """
    Visualize the feature mapping in 3D space
    """
    # Create a grid of points in 2D (for visualization, we'll use first 2 dimensions)
    x1_range = np.linspace(-2, 3, 20)
    x2_range = np.linspace(-2, 3, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Fix x3 = 0 for visualization
    x3_fixed = 0
    
    # Calculate feature mapping for each point
    phi_grid = np.zeros((X1.shape[0], X1.shape[1], 6))
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            point = np.array([X1[i,j], X2[i,j], x3_fixed])
            phi_grid[i,j,:] = explicit_polynomial_mapping_3d_degree2(point)
    
    # Plot original space and first few feature dimensions
    fig = plt.figure(figsize=(15, 10))
    
    # Original 2D space
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.contour(X1, X2, X1**2 + X2**2, levels=10, alpha=0.6)
    ax1.scatter(x[0], x[1], color='red', s=100, label='x', zorder=5)
    ax1.scatter(z[0], z[1], color='blue', s=100, label='z', zorder=5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Original Space (x3=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature dimensions
    feature_names = ['x1^2', 'x2^2', 'x3^2', '√2*x1*x2', '√2*x1*x3', '√2*x2*x3']
    
    for idx in range(5):  # Show first 5 feature dimensions
        ax = fig.add_subplot(2, 3, idx + 2)
        
        if idx < 4:  # Only plot non-zero features for x3=0
            feature_values = phi_grid[:,:,idx]
            contour = ax.contour(X1, X2, feature_values, levels=10, alpha=0.6)
            ax.clabel(contour, inline=True, fontsize=8)
            
            # Mark our specific points
            ax.scatter(x[0], x[1], color='red', s=100, label=f'φ(x)[{idx}]={phi_x[idx]:.2f}', zorder=5)
            ax.scatter(z[0], z[1], color='blue', s=100, label=f'φ(z)[{idx}]={phi_z[idx]:.2f}', zorder=5)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Feature: {feature_names[idx]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_mapping_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

visualize_feature_mapping()

print(f"\nPlots saved to: {save_dir}")
print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
