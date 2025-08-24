import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.preprocessing import StandardScaler
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 23: Kernel Impact on Support Vectors")
print("=" * 50)

# Part 1: Linear vs Non-linear Separable Data
print("\nPart 1: Linear vs Non-linear Separable Data")
print("-" * 40)

# Create two different datasets
np.random.seed(42)

# Dataset 1: Linearly separable
X1, y1 = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, 
                           random_state=42, class_sep=2.0)

# Dataset 2: Non-linearly separable (circles)
X2, y2 = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# Standardize the data
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Function to train SVM and get support vectors
def train_svm_and_get_support_vectors(X, y, kernel='linear', C=1.0):
    """Train SVM and return support vectors and their indices"""
    svm = SVC(kernel=kernel, C=C, random_state=42)
    svm.fit(X, y)
    
    # Get support vector indices
    support_indices = svm.support_
    support_vectors = svm.support_vectors_
    
    return svm, support_vectors, support_indices

# Function to plot decision boundary and support vectors
def plot_svm_results(X, y, svm, support_vectors, support_indices, title, filename):
    """Plot SVM results with decision boundary and support vectors"""
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions for mesh grid
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='black', linewidths=1)
    
    # Plot all data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=50)
    
    # Highlight support vectors
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               c='red', s=200, marker='o', edgecolors='black', linewidth=2, 
               label=f'Support Vectors ({len(support_vectors)} points)')
    
    # Add support vector indices as annotations
    for i, (x, y_coord) in enumerate(support_vectors):
        plt.annotate(f'SV{i+1}', (x, y_coord), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

# Test different kernels on linearly separable data
print("\nTesting kernels on linearly separable data:")
kernels = ['linear', 'poly', 'rbf']
kernel_names = ['Linear', 'Polynomial (degree=2)', 'RBF']

for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    print(f"\n{kernel.upper()} Kernel:")
    svm, support_vectors, support_indices = train_svm_and_get_support_vectors(X1_scaled, y1, kernel=kernel)
    print(f"  Number of support vectors: {len(support_vectors)}")
    print(f"  Support vector indices: {support_indices}")
    
    plot_svm_results(X1_scaled, y1, svm, support_vectors, support_indices,
                    f'SVM with {name} Kernel - Linearly Separable Data',
                    f'linear_data_{kernel}_kernel.png')

# Test different kernels on non-linearly separable data
print("\n\nTesting kernels on non-linearly separable data:")
for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    print(f"\n{kernel.upper()} Kernel:")
    svm, support_vectors, support_indices = train_svm_and_get_support_vectors(X2_scaled, y2, kernel=kernel)
    print(f"  Number of support vectors: {len(support_vectors)}")
    print(f"  Support vector indices: {support_indices}")
    
    plot_svm_results(X2_scaled, y2, svm, support_vectors, support_indices,
                    f'SVM with {name} Kernel - Non-linearly Separable Data',
                    f'nonlinear_data_{kernel}_kernel.png')

# Part 2: Detailed Analysis of Support Vector Changes
print("\n\nPart 2: Detailed Analysis of Support Vector Changes")
print("-" * 50)

# Create a more complex dataset for detailed analysis
np.random.seed(123)
X_complex, y_complex = make_classification(n_samples=150, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=2,
                                         random_state=123, class_sep=1.5)

scaler_complex = StandardScaler()
X_complex_scaled = scaler_complex.fit_transform(X_complex)

# Test with different polynomial degrees
poly_degrees = [1, 2, 3, 4]
print(f"\nTesting polynomial kernels with different degrees:")
print(f"Dataset: {X_complex_scaled.shape[0]} samples, {X_complex_scaled.shape[1]} features")

support_vector_counts = []
support_vector_sets = []

for degree in poly_degrees:
    kernel_name = 'linear' if degree == 1 else f'poly'
    svm, support_vectors, support_indices = train_svm_and_get_support_vectors(
        X_complex_scaled, y_complex, kernel=kernel_name, C=1.0)
    
    support_vector_counts.append(len(support_vectors))
    support_vector_sets.append(set(support_indices))
    
    print(f"\nPolynomial degree {degree}:")
    print(f"  Number of support vectors: {len(support_vectors)}")
    print(f"  Support vector indices: {sorted(support_indices)}")
    
    plot_svm_results(X_complex_scaled, y_complex, svm, support_vectors, support_indices,
                    f'SVM with Polynomial Kernel (degree={degree})',
                    f'polynomial_degree_{degree}.png')

# Analyze overlap between support vector sets
print("\n\nSupport Vector Overlap Analysis:")
print("-" * 30)

for i in range(len(support_vector_sets)):
    for j in range(i+1, len(support_vector_sets)):
        degree1 = poly_degrees[i]
        degree2 = poly_degrees[j]
        overlap = len(support_vector_sets[i] & support_vector_sets[j])
        union = len(support_vector_sets[i] | support_vector_sets[j])
        overlap_percentage = (overlap / union) * 100 if union > 0 else 0
        
        print(f"Degrees {degree1} vs {degree2}:")
        print(f"  Overlap: {overlap} support vectors")
        print(f"  Overlap percentage: {overlap_percentage:.1f}%")

# Part 3: Feature Space Visualization
print("\n\nPart 3: Feature Space Visualization")
print("-" * 35)

# Create a simple 2D dataset
np.random.seed(456)
X_simple, y_simple = make_classification(n_samples=50, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=1,
                                       random_state=456, class_sep=1.8)

scaler_simple = StandardScaler()
X_simple_scaled = scaler_simple.fit_transform(X_simple)

# Function to visualize feature space transformation
def visualize_feature_space(X, y, kernel, title, filename):
    """Visualize how different kernels transform the feature space"""
    plt.figure(figsize=(12, 5))
    
    # Original space
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Original Feature Space')
    plt.grid(True, alpha=0.3)
    
    # Transformed space (for polynomial kernel)
    plt.subplot(1, 2, 2)
    if kernel == 'poly':
        # For polynomial kernel, show the transformed features
        X_transformed = np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2, X[:, 1]**2, X[:, 0]*X[:, 1]])
        plt.scatter(X_transformed[:, 0], X_transformed[:, 2], c=y, cmap='RdYlBu', s=50)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_1^2$')
        plt.title('Polynomial Feature Space ($x_1$ vs $x_1^2$)')
    elif kernel == 'rbf':
        # For RBF kernel, show distance-based transformation
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        plt.scatter(X[:, 0], distances, c=y, cmap='RdYlBu', s=50)
        plt.xlabel('$x_1$')
        plt.ylabel('Distance from center')
        plt.title('RBF Feature Space (Distance-based)')
    else:
        # Linear kernel - no transformation
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Linear Feature Space (No transformation)')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

# Visualize feature space transformations
for kernel in ['linear', 'poly', 'rbf']:
    kernel_name = 'Linear' if kernel == 'linear' else 'Polynomial' if kernel == 'poly' else 'RBF'
    visualize_feature_space(X_simple_scaled, y_simple, kernel,
                           f'{kernel_name} Kernel Feature Space',
                           f'feature_space_{kernel}.png')

# Part 4: Margin Analysis
print("\n\nPart 4: Margin Analysis")
print("-" * 20)

# Function to calculate and visualize margins
def analyze_margins(X, y, kernel, title, filename):
    """Analyze how different kernels affect the margin"""
    svm, support_vectors, support_indices = train_svm_and_get_support_vectors(X, y, kernel=kernel)
    
    # Calculate decision function values for support vectors
    decision_values = svm.decision_function(support_vectors)
    
    plt.figure(figsize=(10, 6))
    
    # Plot decision function values
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(decision_values)), decision_values, c='red', s=100, marker='o')
    plt.axhline(y=1, color='green', linestyle='--', label='Upper margin')
    plt.axhline(y=-1, color='green', linestyle='--', label='Lower margin')
    plt.axhline(y=0, color='black', linestyle='-', label='Decision boundary')
    plt.xlabel('Support Vector Index')
    plt.ylabel('Decision Function Value')
    plt.title(f'Decision Function Values - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot margin distribution
    plt.subplot(1, 2, 2)
    plt.hist(decision_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=1, color='green', linestyle='--', label='Upper margin')
    plt.axvline(x=-1, color='green', linestyle='--', label='Lower margin')
    plt.axvline(x=0, color='black', linestyle='-', label='Decision boundary')
    plt.xlabel('Decision Function Value')
    plt.ylabel('Frequency')
    plt.title(f'Margin Distribution - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{kernel.upper()} Kernel Margin Analysis:")
    print(f"  Number of support vectors: {len(support_vectors)}")
    print(f"  Decision function range: [{decision_values.min():.3f}, {decision_values.max():.3f}]")
    print(f"  Margin width: {decision_values.max() - decision_values.min():.3f}")

# Analyze margins for different kernels
for kernel in ['linear', 'poly', 'rbf']:
    kernel_name = 'Linear' if kernel == 'linear' else 'Polynomial' if kernel == 'poly' else 'RBF'
    analyze_margins(X_complex_scaled, y_complex, kernel,
                   kernel_name,
                   f'margin_analysis_{kernel}.png')

# Part 5: Summary Statistics
print("\n\nPart 5: Summary Statistics")
print("-" * 25)

# Create summary table
print("\nSupport Vector Counts by Kernel:")
print("Dataset\t\tLinear\tPolynomial\tRBF")
print("-" * 50)

datasets = [
    ("Linearly Separable", X1_scaled, y1),
    ("Non-linearly Separable", X2_scaled, y2),
    ("Complex", X_complex_scaled, y_complex)
]

for name, X, y in datasets:
    counts = []
    for kernel in ['linear', 'poly', 'rbf']:
        svm, support_vectors, _ = train_svm_and_get_support_vectors(X, y, kernel=kernel)
        counts.append(len(support_vectors))
    
    print(f"{name:<20}\t{counts[0]}\t{counts[1]}\t\t{counts[2]}")

print(f"\nPlots saved to: {save_dir}")
print("\nConclusion:")
print("The analysis demonstrates that changing kernels significantly affects support vectors.")
print("This proves that the statement 'support vectors remain the same when moving from linear to polynomial kernels' is FALSE.")
