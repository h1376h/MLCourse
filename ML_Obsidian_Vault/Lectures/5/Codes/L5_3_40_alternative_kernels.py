import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
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
print("EXPLORING ALTERNATIVE KERNEL TRANSFORMATIONS")
print("="*80)

# Dataset
positive_points = np.array([1, 3, 5])
negative_points = np.array([0, 2, 4, 6])
all_points = np.concatenate([positive_points, negative_points])
labels = np.concatenate([np.ones(len(positive_points)), -np.ones(len(negative_points))])

print(f"Dataset: {all_points}")
print(f"Labels: {labels}")

# Define novel kernel transformations
def phi_exponential_parity(x):
    """Exponential parity: φ(x) = [e^(-x²/4), (-1)^x * e^(-x²/4)]"""
    exp_term = np.exp(-x**2 / 4)
    parity_term = (-1)**x * exp_term
    return np.column_stack([exp_term, parity_term])

def phi_logarithmic_mod(x):
    """Logarithmic modular: φ(x) = [log(x+1), (x mod 2) * log(x+1)]"""
    log_term = np.log(x + 1)
    mod_log_term = (x % 2) * log_term
    return np.column_stack([log_term, mod_log_term])

def phi_polynomial_alternating(x):
    """Polynomial alternating: φ(x) = [x³, (-1)^x * x]"""
    cubic_term = x**3
    alternating_term = (-1)**x * x
    return np.column_stack([cubic_term, alternating_term])

def phi_hyperbolic_parity(x):
    """Hyperbolic parity: φ(x) = [tanh(x), (2*(x mod 2) - 1) * cosh(x/2)]"""
    tanh_term = np.tanh(x)
    parity_factor = 2 * (x % 2) - 1  # Maps 0->-1, 1->+1
    cosh_term = parity_factor * np.cosh(x / 2)
    return np.column_stack([tanh_term, cosh_term])

def phi_fourier_series(x):
    """Fourier series: φ(x) = [sin(πx/2), cos(πx/2)]"""
    sin_term = np.sin(np.pi * x / 2)
    cos_term = np.cos(np.pi * x / 2)
    return np.column_stack([sin_term, cos_term])

def phi_rational_parity(x):
    """Rational parity: φ(x) = [x/(x+1), (x mod 2 - 0.5) * x/(x+1)]"""
    rational_term = x / (x + 1)
    parity_rational = (x % 2 - 0.5) * rational_term
    return np.column_stack([rational_term, parity_rational])

def phi_power_alternating(x):
    """Power alternating: φ(x) = [x^(1.5), (-1)^(x mod 2) * x^(0.5)]"""
    power_term = x**(1.5)
    alternating_sqrt = (-1)**(x % 2) * np.sqrt(x + 1e-10)  # Add small epsilon for x=0
    return np.column_stack([power_term, alternating_sqrt])

def phi_sigmoid_parity(x):
    """Sigmoid parity: φ(x) = [1/(1+e^(-x)), (2*(x mod 2) - 1) * x]"""
    sigmoid_term = 1 / (1 + np.exp(-x))
    parity_linear = (2 * (x % 2) - 1) * x
    return np.column_stack([sigmoid_term, parity_linear])

def phi_sin_kernel(x):
    """Sine kernel: φ(x) = sin((2x-1)π/2) - 1D transformation"""
    return np.sin((2*x - 1) * np.pi / 2).reshape(-1, 1)

# Store all new transformations
new_transformations = {
    'Exponential-Parity': phi_exponential_parity,
    'Logarithmic-Modular': phi_logarithmic_mod,
    'Polynomial-Alternating': phi_polynomial_alternating,
    'Hyperbolic-Parity': phi_hyperbolic_parity,
    'Fourier-Series': phi_fourier_series,
    'Rational-Parity': phi_rational_parity,
    'Power-Alternating': phi_power_alternating,
    'Sigmoid-Parity': phi_sigmoid_parity,
    'Sin-Kernel': phi_sin_kernel
}

# Test each transformation
successful_new_kernels = {}

for name, phi_func in new_transformations.items():
    print(f"\n{'='*60}")
    print(f"TESTING: {name.upper()} TRANSFORMATION")
    print(f"{'='*60}")
    
    try:
        # Apply transformation
        transformed_points = phi_func(all_points)
        
        print(f"Transformation: {phi_func.__doc__.split(': ')[1]}")
        print(f"Transformed points:")
        for i, (orig, trans) in enumerate(zip(all_points, transformed_points)):
            label_str = "+" if labels[i] == 1 else "-"
            print(f"  x={orig} → φ(x)=[{trans[0]:.6f}, {trans[1]:.6f}] (label: {label_str})")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(transformed_points)) or np.any(np.isinf(transformed_points)):
            print("✗ Contains NaN or infinite values - skipping")
            continue
        
        # Try SVM with linear kernel
        svm = SVC(kernel='linear', C=1000)
        svm.fit(transformed_points, labels)
        
        predictions = svm.predict(transformed_points)
        accuracy = np.mean(predictions == labels)
        
        print(f"SVM Accuracy: {accuracy:.3f}")
        
        if accuracy == 1.0:
            print("✓ PERFECTLY SEPARABLE!")
            
            # Verify kernel validity
            K = transformed_points @ transformed_points.T
            eigenvals = np.linalg.eigvals(K)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval >= -1e-10:
                print(f"✓ Valid kernel (min eigenvalue: {min_eigenval:.10f})")
                
                # Store successful transformation
                successful_new_kernels[name] = {
                    'phi_func': phi_func,
                    'transformed_points': transformed_points,
                    'svm': svm,
                    'accuracy': accuracy,
                    'eigenvals': eigenvals
                }
                
                # Calculate detailed metrics
                w = svm.coef_[0]
                b = svm.intercept_[0]
                margin = 1.0 / np.linalg.norm(w)
                
                print(f"Hyperplane: {w[0]:.6f}φ₁ + {w[1]:.6f}φ₂ + {b:.6f} = 0")
                print(f"Margin: {margin:.6f}")
                
                # Identify support vectors
                decision_values = svm.decision_function(transformed_points)
                support_mask = np.abs(np.abs(decision_values) - 1.0) < 1e-3
                support_count = np.sum(support_mask)
                print(f"Support vectors: {support_count}/{len(all_points)} points")
                
            else:
                print(f"✗ Invalid kernel (negative eigenvalue: {min_eigenval:.10f})")
        else:
            print(f"✗ Not perfectly separable (accuracy: {accuracy:.3f})")
            
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n{'='*60}")
print("SUMMARY OF NEW SUCCESSFUL KERNELS")
print(f"{'='*60}")

if successful_new_kernels:
    for name, result in successful_new_kernels.items():
        w = result['svm'].coef_[0]
        margin = 1.0 / np.linalg.norm(w)
        decision_values = result['svm'].decision_function(result['transformed_points'])
        support_count = np.sum(np.abs(np.abs(decision_values) - 1.0) < 1e-3)
        
        print(f"\n{name}:")
        print(f"  ✓ Perfect separation (100% accuracy)")
        print(f"  ✓ Valid Mercer kernel")
        print(f"  ✓ Margin: {margin:.6f}")
        print(f"  ✓ Support vectors: {support_count}/{len(all_points)}")
        
    print(f"\nFound {len(successful_new_kernels)} new working kernel transformations!")
else:
    print("No new successful kernels found.")

# Create visualizations for successful new kernels
if successful_new_kernels:
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS FOR NEW KERNELS")
    print(f"{'='*60}")
    
    for name, result in successful_new_kernels.items():
        print(f"Creating plot for {name}...")
        
        plt.figure(figsize=(10, 8))
        
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
        
        # Add point labels
        for i, (orig_x, trans_point) in enumerate(zip(all_points, transformed_points)):
            plt.annotate(f'x={orig_x}', trans_point, xytext=(8, 8), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor='gray'))
        
        # Plot decision boundary
        x_min, x_max = transformed_points[:, 0].min(), transformed_points[:, 0].max()
        y_min, y_max = transformed_points[:, 1].min(), transformed_points[:, 1].max()
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        x_min -= 0.2 * x_range
        x_max += 0.2 * x_range
        y_min -= 0.2 * y_range
        y_max += 0.2 * y_range
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'], 
                  linestyles=['--', '-', '--'], linewidths=[1.5, 3, 1.5])
        
        # Shade regions
        plt.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
        plt.contourf(xx, yy, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)
        
        # Highlight support vectors
        decision_values = svm.decision_function(transformed_points)
        support_mask = np.abs(np.abs(decision_values) - 1.0) < 1e-3
        support_vectors = transformed_points[support_mask]
        support_labels = labels[support_mask]
        
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
        
        plt.xlabel('$\\phi_1(x)$', fontsize=14)
        plt.ylabel('$\\phi_2(x)$', fontsize=14)
        plt.title(f'{name} Transformation', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        filename = f'{name.lower().replace("-", "_")}_transformation.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

print(f"\nAnalysis complete! Check {save_dir} for new visualizations.")
