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

# Disable LaTeX style plotting to avoid Unicode issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("="*80)
print("EXPLORING 1D KERNEL TRANSFORMATIONS")
print("="*80)

# Dataset
positive_points = np.array([1, 3, 5])
negative_points = np.array([0, 2, 4, 6])
all_points = np.concatenate([positive_points, negative_points])
labels = np.concatenate([np.ones(len(positive_points)), -np.ones(len(negative_points))])

print(f"Dataset: {all_points}")
print(f"Labels: {labels}")

# Define various 1D kernel transformations
def phi_sin_kernel(x):
    """1D Sine kernel: φ(x) = sin((2x-1)π/2)"""
    return np.sin((2*x - 1) * np.pi / 2)

def phi_cos_kernel(x):
    """1D Cosine kernel: φ(x) = cos(πx)"""
    return np.cos(np.pi * x)

def phi_tanh_kernel(x):
    """1D Tanh kernel: φ(x) = tanh(x - 3)"""
    return np.tanh(x - 3)

def phi_sigmoid_kernel(x):
    """1D Sigmoid kernel: φ(x) = 1/(1 + e^(-(x-3)))"""
    return 1 / (1 + np.exp(-(x - 3)))

def phi_polynomial_kernel(x):
    """1D Polynomial kernel: φ(x) = (x-3)³"""
    return (x - 3)**3

def phi_exponential_kernel(x):
    """1D Exponential kernel: φ(x) = e^(x-3) - e^(-(x-3))"""
    return np.exp(x - 3) - np.exp(-(x - 3))

def phi_logistic_kernel(x):
    """1D Logistic kernel: φ(x) = 2/(1 + e^(-2(x-3))) - 1"""
    return 2 / (1 + np.exp(-2 * (x - 3))) - 1

def phi_parity_kernel(x):
    """1D Parity kernel: φ(x) = 2(x mod 2) - 1"""
    return 2 * (x % 2) - 1

def phi_alternating_kernel(x):
    """1D Alternating kernel: φ(x) = (-1)^x"""
    return (-1)**x

def phi_step_kernel(x):
    """1D Step kernel: φ(x) = sign(x - 2.5)"""
    return np.sign(x - 2.5)

def phi_quadratic_kernel(x):
    """1D Quadratic kernel: φ(x) = (x-3)² - 2.25"""
    return (x - 3)**2 - 2.25

def phi_cubic_kernel(x):
    """1D Cubic kernel: φ(x) = (x-3)³"""
    return (x - 3)**3

def phi_hyperbolic_kernel(x):
    """1D Hyperbolic kernel: φ(x) = sinh(x-3)"""
    return np.sinh(x - 3)

def phi_arcsin_kernel(x):
    """1D Arcsin kernel: φ(x) = arcsin(sin(πx/2))"""
    return np.arcsin(np.sin(np.pi * x / 2))

def phi_floor_kernel(x):
    """1D Floor kernel: φ(x) = floor(x/2) - 1"""
    return np.floor(x / 2) - 1

# Store all 1D transformations
one_d_transformations = {
    'Sin-Kernel': phi_sin_kernel,
    'Cos-Kernel': phi_cos_kernel,
    'Tanh-Kernel': phi_tanh_kernel,
    'Sigmoid-Kernel': phi_sigmoid_kernel,
    'Polynomial-Kernel': phi_polynomial_kernel,
    'Exponential-Kernel': phi_exponential_kernel,
    'Logistic-Kernel': phi_logistic_kernel,
    'Parity-Kernel': phi_parity_kernel,
    'Alternating-Kernel': phi_alternating_kernel,
    'Step-Kernel': phi_step_kernel,
    'Quadratic-Kernel': phi_quadratic_kernel,
    'Cubic-Kernel': phi_cubic_kernel,
    'Hyperbolic-Kernel': phi_hyperbolic_kernel,
    'Arcsin-Kernel': phi_arcsin_kernel,
    'Floor-Kernel': phi_floor_kernel
}

# Test each transformation
successful_1d_kernels = {}

for name, phi_func in one_d_transformations.items():
    print(f"\n{'='*60}")
    print(f"TESTING: {name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Apply transformation
        transformed_points = phi_func(all_points)
        
        print(f"Transformation: {phi_func.__doc__.split(': ')[1]}")
        print(f"Transformed points:")
        for i, (orig, trans) in enumerate(zip(all_points, transformed_points)):
            label_str = "+" if labels[i] == 1 else "-"
            print(f"  x={orig} → φ(x)={trans:.6f} (label: {label_str})")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(transformed_points)) or np.any(np.isinf(transformed_points)):
            print("✗ Contains NaN or infinite values - skipping")
            continue
        
        # Check if it creates separation in 1D
        positive_transformed = transformed_points[labels == 1]
        negative_transformed = transformed_points[labels == -1]
        
        pos_min = np.min(positive_transformed)
        pos_max = np.max(positive_transformed)
        neg_min = np.min(negative_transformed)
        neg_max = np.max(negative_transformed)
        
        print(f"\nPositive range: [{pos_min:.6f}, {pos_max:.6f}]")
        print(f"Negative range: [{neg_min:.6f}, {neg_max:.6f}]")
        
        # Check for overlap
        if pos_max < neg_min or neg_max < pos_min:
            print("✓ Classes are separable in 1D!")
            
            # Try SVM with linear kernel in 1D
            svm = SVC(kernel='linear', C=1000)
            transformed_1d = transformed_points.reshape(-1, 1)
            svm.fit(transformed_1d, labels)
            
            predictions = svm.predict(transformed_1d)
            accuracy = np.mean(predictions == labels)
            print(f"SVM Accuracy in 1D: {accuracy:.3f}")
            
            if accuracy == 1.0:
                print("✓ PERFECTLY SEPARABLE in 1D!")
                
                # Calculate detailed metrics
                w = svm.coef_[0][0]
                b = svm.intercept_[0]
                threshold = -b/w
                margin = 1.0 / abs(w)
                
                print(f"1D Hyperplane: {w:.6f}φ + {b:.6f} = 0")
                print(f"Threshold: φ = {threshold:.6f}")
                print(f"Margin: {margin:.6f}")
                
                # Identify support vectors
                decision_values = svm.decision_function(transformed_1d)
                support_mask = np.abs(np.abs(decision_values) - 1.0) < 1e-3
                support_count = np.sum(support_mask)
                print(f"Support vectors: {support_count}/{len(all_points)} points")
                
                # Verify kernel validity
                K = transformed_points.reshape(-1, 1) @ transformed_points.reshape(1, -1)
                eigenvals = np.linalg.eigvals(K)
                min_eigenval = np.min(eigenvals)
                
                print(f"Kernel matrix eigenvalues: {eigenvals}")
                print(f"Minimum eigenvalue: {min_eigenval:.10f}")
                print(f"Valid Mercer kernel: {'✓' if min_eigenval >= -1e-10 else '✗'}")
                
                if min_eigenval >= -1e-10:
                    print("✓ VALID KERNEL!")
                    
                    # Store successful transformation
                    successful_1d_kernels[name] = {
                        'phi_func': phi_func,
                        'transformed_points': transformed_points,
                        'svm': svm,
                        'accuracy': accuracy,
                        'margin': margin,
                        'support_count': support_count,
                        'threshold': threshold,
                        'eigenvals': eigenvals
                    }
                else:
                    print("✗ Invalid kernel (negative eigenvalues)")
            else:
                print(f"✗ Not perfectly separable (accuracy: {accuracy:.3f})")
        else:
            print("✗ Classes overlap - not separable in 1D")
            
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n{'='*60}")
print("SUMMARY OF SUCCESSFUL 1D KERNELS")
print(f"{'='*60}")

if successful_1d_kernels:
    print(f"Found {len(successful_1d_kernels)} valid 1D kernel transformations!")
    
    for name, result in successful_1d_kernels.items():
        print(f"\n{name}:")
        print(f"  ✓ Perfect separation (100% accuracy)")
        print(f"  ✓ Valid Mercer kernel")
        print(f"  ✓ Margin: {result['margin']:.6f}")
        print(f"  ✓ Support vectors: {result['support_count']}/{len(all_points)}")
        print(f"  ✓ Threshold: {result['threshold']:.6f}")
        
    # Create visualizations for successful kernels
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS FOR 1D KERNELS")
    print(f"{'='*60}")
    
    for name, result in successful_1d_kernels.items():
        print(f"Creating plot for {name}...")
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1D transformation
        plt.subplot(1, 2, 1)
        positive_transformed = result['transformed_points'][labels == 1]
        negative_transformed = result['transformed_points'][labels == -1]
        
        plt.scatter(positive_transformed, np.zeros(len(positive_transformed)), 
                   c='red', s=150, marker='o', label='Positive (+1)', 
                   edgecolor='black', linewidth=2, alpha=0.8)
        plt.scatter(negative_transformed, np.zeros(len(negative_transformed)), 
                   c='blue', s=150, marker='s', label='Negative (-1)', 
                   edgecolor='black', linewidth=2, alpha=0.8)
        
        # Add point labels
        for i, (orig_x, trans_point) in enumerate(zip(all_points, result['transformed_points'])):
            plt.annotate(f'x={orig_x}', (trans_point, 0), xytext=(0, 20), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor='gray'))
        
        # Plot decision boundary
        threshold = result['threshold']
        margin = result['margin']
        plt.axvline(x=threshold, color='black', linestyle='-', linewidth=3, label='Decision Boundary')
        plt.axvline(x=threshold + margin, color='gray', linestyle='--', linewidth=1.5, label='Margin')
        plt.axvline(x=threshold - margin, color='gray', linestyle='--', linewidth=1.5)
        
        plt.xlabel('φ(x)', fontsize=14)
        plt.ylabel('')
        plt.title(f'{name} - 1D Transformation', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.5, 0.5)
        plt.yticks([])
        
        # Plot original vs transformed
        plt.subplot(1, 2, 2)
        plt.scatter(all_points[labels == 1], result['transformed_points'][labels == 1], 
                   c='red', s=100, marker='o', label='Positive (+1)', alpha=0.8)
        plt.scatter(all_points[labels == -1], result['transformed_points'][labels == -1], 
                   c='blue', s=100, marker='s', label='Negative (-1)', alpha=0.8)
        
        # Plot the transformation function
        x_range = np.linspace(-0.5, 6.5, 1000)
        y_range = result['phi_func'](x_range)
        plt.plot(x_range, y_range, 'k-', linewidth=2, alpha=0.7, label=f'{name}')
        
        plt.xlabel('x (Original Space)', fontsize=14)
        plt.ylabel('φ(x) (Transformed Space)', fontsize=14)
        plt.title('Original vs Transformed Space', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        filename = f'{name.lower().replace("-", "_")}_1d_transformation.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

else:
    print("No successful 1D kernels found.")

print(f"\nAnalysis complete! Check {save_dir} for visualizations.")
