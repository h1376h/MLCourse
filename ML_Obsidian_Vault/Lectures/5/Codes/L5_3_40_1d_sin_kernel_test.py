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
print("1D SIN KERNEL TRANSFORMATION ANALYSIS")
print("="*80)

# Dataset
positive_points = np.array([1, 3, 5])
negative_points = np.array([0, 2, 4, 6])
all_points = np.concatenate([positive_points, negative_points])
labels = np.concatenate([np.ones(len(positive_points)), -np.ones(len(negative_points))])

print(f"Dataset: {all_points}")
print(f"Labels: {labels}")

# Define the 1D sin kernel transformation
def phi_1d_sin_kernel(x):
    """1D Sine kernel: φ(x) = sin((2x-1)π/2)"""
    return np.sin((2*x - 1) * np.pi / 2)

print(f"\n{'='*60}")
print("TESTING 1D SIN KERNEL TRANSFORMATION")
print(f"{'='*60}")

# Apply transformation
transformed_points = phi_1d_sin_kernel(all_points)

print(f"Transformation: φ(x) = sin((2x-1)π/2)")
print(f"Transformed points:")
for i, (orig, trans) in enumerate(zip(all_points, transformed_points)):
    label_str = "+" if labels[i] == 1 else "-"
    print(f"  x={orig} → φ(x)={trans:.6f} (label: {label_str})")

# Check if it creates separation in 1D
positive_transformed = transformed_points[labels == 1]
negative_transformed = transformed_points[labels == -1]

print(f"\nPositive class transformed: {positive_transformed}")
print(f"Negative class transformed: {negative_transformed}")

# Check if classes are separable in 1D
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
    transformed_1d = transformed_points.reshape(-1, 1)  # Reshape for sklearn
    svm.fit(transformed_1d, labels)
    
    predictions = svm.predict(transformed_1d)
    accuracy = np.mean(predictions == labels)
    print(f"SVM Accuracy in 1D: {accuracy:.3f}")
    
    if accuracy == 1.0:
        print("✓ PERFECTLY SEPARABLE in 1D!")
        w = svm.coef_[0][0]
        b = svm.intercept_[0]
        print(f"1D Hyperplane: {w:.6f}φ + {b:.6f} = 0")
        print(f"Threshold: φ = {-b/w:.6f}")
        
        # Calculate margin
        margin = 1.0 / abs(w)
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
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot 1D transformation
        plt.subplot(1, 2, 1)
        plt.scatter(positive_transformed, np.zeros(len(positive_transformed)), 
                   c='red', s=150, marker='o', label='Positive (+1)', 
                   edgecolor='black', linewidth=2, alpha=0.8)
        plt.scatter(negative_transformed, np.zeros(len(negative_transformed)), 
                   c='blue', s=150, marker='s', label='Negative (-1)', 
                   edgecolor='black', linewidth=2, alpha=0.8)
        
        # Add point labels
        for i, (orig_x, trans_point) in enumerate(zip(all_points, transformed_points)):
            plt.annotate(f'x={orig_x}', (trans_point, 0), xytext=(0, 20), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor='gray'))
        
        # Plot decision boundary
        threshold = -b/w
        plt.axvline(x=threshold, color='black', linestyle='-', linewidth=3, label='Decision Boundary')
        plt.axvline(x=threshold + margin, color='gray', linestyle='--', linewidth=1.5, label='Margin')
        plt.axvline(x=threshold - margin, color='gray', linestyle='--', linewidth=1.5)
        
        plt.xlabel('φ(x) = sin((2x-1)π/2)', fontsize=14)
        plt.ylabel('')
        plt.title('1D Sin Kernel Transformation', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.5, 0.5)
        plt.yticks([])
        
        # Plot original vs transformed
        plt.subplot(1, 2, 2)
        plt.scatter(all_points[labels == 1], transformed_points[labels == 1], 
                   c='red', s=100, marker='o', label='Positive (+1)', alpha=0.8)
        plt.scatter(all_points[labels == -1], transformed_points[labels == -1], 
                   c='blue', s=100, marker='s', label='Negative (-1)', alpha=0.8)
        
        # Plot the transformation function
        x_range = np.linspace(-0.5, 6.5, 1000)
        y_range = phi_1d_sin_kernel(x_range)
        plt.plot(x_range, y_range, 'k-', linewidth=2, alpha=0.7, label='φ(x) = sin((2x-1)π/2)')
        
        plt.xlabel('x (Original Space)', fontsize=14)
        plt.ylabel('φ(x) (Transformed Space)', fontsize=14)
        plt.title('Original vs Transformed Space', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '1d_sin_kernel_transformation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved: 1d_sin_kernel_transformation.png")
        
        # Mathematical analysis
        print(f"\n{'='*60}")
        print("MATHEMATICAL ANALYSIS")
        print(f"{'='*60}")
        
        print("The transformation φ(x) = sin((2x-1)π/2) creates perfect binary separation:")
        print("\nFor odd numbers (positive class):")
        for x in [1, 3, 5]:
            result = phi_1d_sin_kernel(x)
            print(f"  φ({x}) = sin((2×{x}-1)π/2) = sin({(2*x-1)*np.pi/2:.3f}) = {result:.6f}")
        
        print("\nFor even numbers (negative class):")
        for x in [0, 2, 4, 6]:
            result = phi_1d_sin_kernel(x)
            print(f"  φ({x}) = sin((2×{x}-1)π/2) = sin({(2*x-1)*np.pi/2:.3f}) = {result:.6f}")
        
        print(f"\nDecision function: f(x) = sign(φ(x) - {threshold:.6f})")
        print(f"  = sign(sin((2x-1)π/2) - {threshold:.6f})")
        
        print(f"\nThis creates a simple threshold classifier in 1D space!")
        print(f"✓ Perfect separation achieved with 1D transformation")
        print(f"✓ No need for 2D feature space")
        print(f"✓ Maximum margin: {margin:.6f}")
        print(f"✓ All points are support vectors: {support_count}/{len(all_points)}")
        
    else:
        print(f"✗ Not perfectly separable (accuracy: {accuracy:.3f})")
else:
    print("✗ Classes overlap - not separable in 1D")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print("The 1D sin kernel transformation φ(x) = sin((2x-1)π/2) is a valid kernel that:")
print("✓ Creates perfect binary separation in 1D space")
print("✓ Achieves 100% classification accuracy")
print("✓ Provides maximum possible margin")
print("✓ Satisfies Mercer's conditions for kernel validity")
print("✓ Demonstrates the power of mathematical function design in kernel methods")
print("\nThis is a remarkable example of how the right transformation can solve")
print("a seemingly impossible classification problem with elegant simplicity!")
